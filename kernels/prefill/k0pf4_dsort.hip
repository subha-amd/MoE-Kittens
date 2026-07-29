// ================================================================================================
// k0pf4_dsort.hip — k0pf3_dsort with a hierarchical multi-CTA expert histogram.
//
// Identical contract, identical downstream, identical protocol.  The only change is the count
// phase.  k0pf3_dsort gives the whole per-expert count to CTA0, which at prefill means T_loc *
// TOPK LDS atomics on a single CU — on the hottest rank that is over 150k pairs, and it runs long
// after the other CTAs have finished zeroing part and gone idle.  Because that rank sets the
// region's pace, the serial count is region time.
//
// Here every CTA histograms a disjoint grid-stride slice into LDS, publishes its own [E] counts to
// local global scratch (hcnt, expert-major), and after one extra grid barrier CTA0 reduces the
// [E][gridDim] table into exactly the s_cnt the padded scan already consumed.  Integer addition is
// associative, so the counts are bit-identical to the single-CTA version rather than merely
// equivalent.
//
// Everything else is behaviour-preserved from k0pf3_dsort.hip:
//   * P0 doorbell wait (bounded, fail-closed, bit 262144), acquire fence, T_loc range guard.
//   * The scan, cursors and sei, padding, atomic-cursor scatter and pull_src fill are the same
//     stages, expressed through the hkp headers.
//   * zero part[0, T_loc) and the row-major to group-major scale transpose on CTAs 2.., with the
//     same 14 x 1 KB warp chunking and the same live-prefix-only extent.
//   * epoch = sort_count[0] + 1; the dest_counter reset and the epoch advance stay at the tail,
//     which is their sole consumer.
//   * Sort order remains nondeterministic (atomic cursors), so acceptance stays tolerance-based.
//
// hcnt is local scratch, fully overwritten every epoch and never accumulated, so it adds no
// protocol state and the monotonic-epoch graph-replay argument is unchanged.
//
// PHASES:
//   P0  doorbell wait, then T_loc.
//   P1a all CTAs: LDS histogram over a disjoint slice into hcnt[e * gridDim + bid].
//   P1b grid barrier 1, publishing hcnt.
//   P1c CTA0 reduces, runs the padded scan, and writes nvi, cursors and sei;  CTA1 turns pull_cnt
//       into pull_ptr;  CTAs 2.. zero part[0, T_loc) and transpose the scales.  The plan work now
//       hides under the part zero, so the count is off every rank's critical path.
//   P2  grid barrier 2, publishing scratch, nvi, sei, pull_ptr, part and sc_dst.
//   P3  padding sentinel, atomic-cursor scatter, pull_src compaction.
//   P4  tail: dest_counter[0] = 0, sort_count[0] = epoch.
//
// pperr bits are unchanged from k0pf3_dsort: 262144 doorbell spin timeout, 524288 T_loc out of
// range, 1048576 scatter/CSR row overflow, 2097152 internal grid-barrier spin timeout, 4194304
// pad row overflow (PADMAX), 131072 shape guard.
// Scratch layout (LOCAL int scratch[256]): erb[0..64] at 0, cnt at 80, cursors at 160.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#include "hkp_sort.hpp"
#include "hkp_sync.hpp"
#include "hkp_quant.hpp"

#define K0P4_MAXE 64
#define K0P4_NG 56
#define K0P4_MAXCTA 256          // hcnt capacity is K0P4_MAXE * K0P4_MAXCTA ints
#define K0P4_H 7168              // part row length in bf16 elements (14 KiB)
#define K0P4_SC_ERB 0
#define K0P4_SC_CNT 80
#define K0P4_SC_CUR 160

using k0p4_sort = hkp::destination_counting_sort<K0P4_MAXE, 32, K0P4_SC_ERB, K0P4_SC_CNT,
                                                 K0P4_SC_CUR>;

extern "C" __global__ void __launch_bounds__(256, 8) k0pf4_dsort(
    const int* __restrict__ recv_eid,              // SYMMETRIC [T_loc_max, TOPK] (post-doorbell)
    const float* __restrict__ recv_wgt,            // SYMMETRIC [T_loc_max, TOPK]
    const float* __restrict__ sc_stage,            // SYMMETRIC [T_loc_max, 56] row-major
    unsigned int* __restrict__ dest_counter,       // SYMMETRIC [1]: T_loc producer; reset at tail
    const unsigned int* __restrict__ recv_doorbell,// SYMMETRIC [world]: monotonic flags (waited)
    const int* __restrict__ pull_stage,            // LOCAL [T*TOPK, 2]
    const int* __restrict__ pull_cnt,              // LOCAL [T]
    int* __restrict__ sti,                         // LOCAL [PADMAX]
    float* __restrict__ swt,                       // LOCAL [PADMAX]
    int* __restrict__ sei,                         // LOCAL [PADMAX/32]
    int* __restrict__ nvi,                         // LOCAL [2]
    int* __restrict__ pull_ptr,                    // LOCAL [T+1]
    int* __restrict__ pull_src,                    // LOCAL [T*world, 2]
    float* __restrict__ sc_dst,                    // LOCAL group-major flat (MROWS, 56) alloc
    unsigned short* __restrict__ part,             // LOCAL [MROWS, 7168] bf16 — zero rows < T_loc
    int* __restrict__ scratch,                     // LOCAL [256]: erb/cnt/cursor carve
    unsigned int* __restrict__ gbar,               // LOCAL [1]: cumulative grid barrier counter
    unsigned int* __restrict__ sort_count,         // LOCAL [1]: epoch counter (bumped at tail)
    int* __restrict__ pperr, long long spin_limit,
    int T, int TOPK, int E, int cur, int world, int T_loc_max, int PADMAX,
    int* __restrict__ hcnt) {                      // LOCAL [K0P4_MAXE * K0P4_MAXCTA] per-CTA counts
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nct = (int)gridDim.x;

  if (TOPK != 8 || E > K0P4_MAXE || world > 8 || T_loc_max > 0x00FFFFFF || nct < 3 ||
      nct > K0P4_MAXCTA) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 131072);
    return;
  }

  const unsigned int epoch = hkp::epoch_from_cell(sort_count);

  // ================= P0: doorbell wait (also the cross-epoch part guard) ==================
  if (tid < world) {
    hkp::bounded_wait(reinterpret_cast<volatile const unsigned int*>(recv_doorbell) + tid,
                      epoch, spin_limit, hkp::fail_closed{pperr, 262144});
  }
  __syncthreads();
  __threadfence_system();   // acquire: recv payloads, dest_counter, and the part-reuse proof

  const int T_loc = (int)dest_counter[0];
  if (T_loc < 0 || T_loc > T_loc_max) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 524288);
    return;   // grid-uniform: every CTA reads the same dest_counter, so no CTA reaches a barrier
  }

  const int lo = cur * E;
  const int pairs = T_loc * TOPK;
  const hkp::local_grid_epoch bar{gbar, pperr, spin_limit};
  const hkp::fail_closed bar_err{pperr, 2097152};

  // ================= P1a: hierarchical count — every CTA, disjoint slice ==================
  __shared__ int s_cnt[K0P4_MAXE];
  k0p4_sort::count_partial_publish(recv_eid, pairs, lo, E, s_cnt, hcnt, nct, bid, tid);

  // ================= P1b: grid barrier #1 (publishes hcnt) ================================
  hkp::grid_barrier(bar, tid, bar_err);

  // ================= P1c: task-split plan work (no barrier) ===============================
  if (bid == 0) {
    k0p4_sort::count_reduce(hcnt, E, nct, s_cnt, tid);
    k0p4_sort::scan(s_cnt, E, T_loc, scratch, nvi, tid);
    k0p4_sort::cursors_sei(scratch, E, PADMAX, sei, pperr, 4194304, tid);
  } else if (bid == 1) {
    // ---- owner combine CSR row pointers: 256-wide block scans over pull_cnt ----
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    hkp::csr_scan_block256(pull_cnt, pull_ptr, T, tid, s_tmp, &s_carry);
  } else {
    // ---- zero part[0, T_loc) + transpose scales (needs T_loc only) ----
    const int lane = tid & 63;
    const int w = (bid - 2) * (blockDim.x >> 6) + (tid >> 6);
    const int nw = (nct - 2) * (blockDim.x >> 6);
    for (int t = w; t < T_loc; t += nw) {
      hkp::zero_part_scale_transpose<14>(part + (std::size_t)t * K0P4_H, sc_dst,
                                         sc_stage + (std::size_t)t * K0P4_NG, t, T_loc,
                                         K0P4_NG, lane);
    }
  }

  // ================= P2: grid barrier #2 (publishes scratch + nvi + pull_ptr) =============
  hkp::grid_barrier(bar, tid, bar_err);

  // ================= P3: pad sentinel + atomic-cursor scatter + CSR fill ==================
  k0p4_sort::pad(scratch, E, T_loc, TOPK, PADMAX, sti, swt, pperr, 4194304);
  k0p4_sort::scatter(recv_eid, recv_wgt, pairs, lo, E, TOPK, PADMAX, scratch, sti, swt,
                     pperr, 1048576);
  hkp::pull_src_fill(pull_stage, pull_ptr, T, TOPK, T * world, pull_src, pperr, 1048576);

  // ================= P4: tail — reset allocator, advance epoch ============================
  __syncthreads();
  if (bid == 0 && tid == 0) {
    dest_counter[0] = 0u;     // next epoch's allocations restart at row 0
    hkp::advance_epoch_cell(sort_count, epoch);   // plain store; next kernel reads it
  }
}
