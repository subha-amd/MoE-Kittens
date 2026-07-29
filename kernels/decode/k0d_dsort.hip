// ================================================================================================
// k0d_dsort.hip — destination-side sort and plan construction for the push-dispatched decode path.
//
// After k0d_push the receive rows have arrived in nondeterministic atomic order, and the only
// routing knowledge on this rank is the pushed recv_eid / recv_wgt arrays.  This kernel is the
// local sorter for that state, shaped for decode (T_loc <= 512 rows, pairs = T_loc * 8 <= 4096,
// E <= 64 local experts).  It produces exactly the tensors the GEMM and the combine already
// consume, so neither needs modification:
//   sti[row]  = (arrival_row & 0x00FFFFFF) | (slot << 24)   arrival_row indexes a_dst; the GEMM's
//               A-descriptor bounds token >= nvi[1] to zero, so the padding sentinel is load-safe
//   swt[row]  = route weight (0.0f on padding rows)
//   sei[b]    = local expert id of sorted 32-row block b
//   nvi[0]    = BM32-padded row total, nvi[1] = T_loc
//   pull_ptr / pull_src        owner-side combine CSR, compacted from k0d_push's pull_stage
//   sc_dst[g * T_loc + t]      group-major scales, transposed locally from row-major sc_stage
//   part[0, T_loc)             zeroed, as the GEMM's phase-2 atomic epilogue requires
//
// SORT ORDER: within an expert's segment, rows are placed by atomic cursor rather than in a
// deterministic route order.  This is safe because the GEMM's rows are independent (phase 1
// computes each row's tile alone, and phase 2's per-row epilogue atomics already race in the
// production kernel), and acceptance is tolerance-based rather than bitwise.  Atomic cursors
// remove the per-chunk count/base and ordered-compaction machinery a global plan needs.
//
// THE DOORBELL IS ALSO THE CROSS-EPOCH GUARD for part:
//   recv_doorbell[src] >= epoch N+1 proves src's push(N+1) completed, hence src's combine(N)
//   returned in stream order on src.  Once all doorbells are up, every rank's combine(N) is done,
//   so no remote combine can still be pulling this rank's part when it is zeroed below.
//
// PHASES (grid (16, 256); two internal monotonic grid barriers via a cumulative counter):
//   P0  every CTA: threads tid < world spin on recv_doorbell[tid] >= epoch (bounded,
//       fail-closed); acquire fence; T_loc = dest_counter[0].
//   P2  task split, no barrier.  CTA0 counts local-expert pairs with LDS atomics and runs the
//       serial padded scan into erb/cnt/cursor scratch, nvi and sei.  CTA1 builds the combine CSR
//       (one 64-wide wave scan over pull_cnt, then per-token slot compaction).  CTAs 2..15 zero
//       part[0, T_loc) in 14 x 1 KB warp chunks per row and transpose sc_stage into sc_dst.
//   P3  grid barrier, publishing CTA0's scratch and nvi.
//   P4  all CTAs: write the padding sentinel rows over the disjoint ranges [erb+cnt, erb_next),
//       which cannot race the scatter, then scatter live pairs through per-expert atomic cursors.
//   P5  CTA0 thread 0 resets dest_counter and advances sort_count to this epoch.
//
// pperr bits: 1024 doorbell spin timeout, 2048 T_loc out of range, 4096 scatter row overflow,
// 8192 internal grid-barrier spin timeout, 256 shape guard, 2 PADMAX overflow.
// Scratch layout (LOCAL int scratch[256]): erb[0..64] at 0, cnt at 80, cursors at 160.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#define K0DS_MAXE 64
#define K0DS_NG 56
#define K0DS_SC_ERB 0
#define K0DS_SC_CNT 80
#define K0DS_SC_CUR 160

extern "C" __global__ void __launch_bounds__(256, 8) k0d_dsort(
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
    int T, int TOPK, int E, int cur, int world, int T_loc_max, int PADMAX) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  if (TOPK != 8 || E > K0DS_MAXE || world > 8 || T > 64 || T_loc_max > 0x00FFFFFF) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 256);
    return;
  }

  const unsigned int epoch = sort_count[0] + 1u;

  // ================= P0: doorbell wait (also the cross-epoch part guard) ==================
  if (tid < world) {
    volatile const unsigned int* vf = recv_doorbell;
    long long spins = 0;
    while (vf[tid] < epoch) {
      if (++spins > spin_limit) { atomicOr(pperr, 1024); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence_system();   // acquire: recv payloads, dest_counter, and the part-reuse proof

  const int T_loc = (int)dest_counter[0];
  if (T_loc < 0 || T_loc > T_loc_max) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 2048);
    return;
  }

  // ================= P2: task-split plan work (no barrier) ================================
  if (bid == 0) {
    // ---- per-expert counts over the received pairs (LDS atomics) ----
    __shared__ int s_cnt[K0DS_MAXE];
    for (int i = tid; i < E; i += 256) s_cnt[i] = 0;
    __syncthreads();
    const int lo = cur * E, hi = lo + E;
    const int pairs = T_loc * TOPK;
    for (int j = tid; j < pairs; j += 256) {
      const int e = recv_eid[j];   // flat [row][slot] == j
      if (e >= lo && e < hi) atomicAdd(&s_cnt[e - lo], 1);
    }
    __syncthreads();
    // ---- padded BM32 scan (serial, E <= 64) ----
    if (tid == 0) {
      int acc = 0;
      for (int e = 0; e < E; ++e) {
        scratch[K0DS_SC_ERB + e] = acc;
        scratch[K0DS_SC_CNT + e] = s_cnt[e];
        acc += ((s_cnt[e] + 31) / 32) * 32;
      }
      scratch[K0DS_SC_ERB + E] = acc;
      nvi[0] = acc;
      nvi[1] = T_loc;
    }
    __syncthreads();
    // ---- scatter cursors + sorted-expert-ids (needs the scan) ----
    {
      const int bcap = PADMAX / 32;
      for (int e = tid; e < E; e += 256) {
        scratch[K0DS_SC_CUR + e] = scratch[K0DS_SC_ERB + e];
        const int b0 = scratch[K0DS_SC_ERB + e] / 32;
        const int b1 = scratch[K0DS_SC_ERB + e + 1] / 32;
        for (int b = b0; b < b1; ++b) {
          if (b >= bcap) { atomicOr(pperr, 2); break; }
          sei[b] = e;
        }
      }
    }
  } else if (bid == 1) {
    // ---- owner combine CSR: 64-wide wave prefix over pull_cnt, then slot compaction ----
    if (tid < 64) {
      const int tau = tid;
      const int f = (tau < T) ? pull_cnt[tau] : 0;
      int incl = f;
#pragma unroll
      for (int off = 1; off < 64; off <<= 1) {
        const int add = __shfl_up(incl, off, 64);
        if ((tid & 63) >= off) incl += add;
      }
      if (tau < T) pull_ptr[tau] = incl - f;
      if (tau == T - 1) pull_ptr[T] = incl;
      if (tau < T) {
        int k = incl - f;
        for (int s = 0; s < TOPK; ++s) {
          const int pe = pull_stage[((size_t)tau * TOPK + s) * 2 + 0];
          if (pe < 0) continue;
          const int row = pull_stage[((size_t)tau * TOPK + s) * 2 + 1];
          if (k >= T * world) { atomicOr(pperr, 8); break; }
          pull_src[(size_t)k * 2 + 0] = pe;
          pull_src[(size_t)k * 2 + 1] = row;
          ++k;
        }
      }
    }
  } else {
    // ---- zero part[0, T_loc) + transpose scales (needs T_loc only) ----
    const int lane = tid & 63;
    const int w = (bid - 2) * (blockDim.x >> 6) + (tid >> 6);
    const int nw = (gridDim.x - 2) * (blockDim.x >> 6);
    for (int t = w; t < T_loc; t += nw) {
      unsigned char* prow = reinterpret_cast<unsigned char*>(part) + (size_t)t * 14336;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int c = 0; c < 14; ++c)
        *reinterpret_cast<uint4*>(prow + (c << 10) + (lane << 4)) = z;
      if (lane < K0DS_NG)
        sc_dst[(size_t)lane * (size_t)T_loc + t] = sc_stage[(size_t)t * K0DS_NG + lane];
    }
  }

  // ================= P3: grid barrier (publishes scratch + nvi) ===========================
  __syncthreads();
  if (tid == 0) {
    __threadfence();   // device release (scratch/nvi/sei/part/sc_dst writes are this GPU's)
    const unsigned int old =
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    volatile unsigned int* vb = gbar;
    long long spins = 0;
    while (*vb < target) {
      if (++spins > spin_limit) { atomicOr(pperr, 8192); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence();   // device acquire

  // ================= P4: pad sentinel + atomic-cursor scatter =============================
  {
    const int lo = cur * E, hi = lo + E;
    // pad rows live in [erb[e] + cnt[e], erb[e+1]) — disjoint from every scatter position.
    for (int idx = bid * 256 + tid; idx < E * 32; idx += gridDim.x * 256) {
      const int e = idx >> 5, k = idx & 31;
      const int row = scratch[K0DS_SC_ERB + e] + scratch[K0DS_SC_CNT + e] + k;
      if (row < scratch[K0DS_SC_ERB + e + 1]) {
        if (row >= PADMAX) { atomicOr(pperr, 2); continue; }
        sti[row] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
        swt[row] = 0.0f;
      }
    }
    const int pairs = T_loc * TOPK;
    for (int j = bid * 256 + tid; j < pairs; j += gridDim.x * 256) {
      const int e = recv_eid[j];
      if (e < lo || e >= hi) continue;
      const int pos = atomicAdd(&scratch[K0DS_SC_CUR + (e - lo)], 1);
      if (pos >= PADMAX) { atomicOr(pperr, 4096); continue; }
      const int row = j / TOPK;
      const int slot = j - row * TOPK;
      sti[pos] = (row & 0x00FFFFFF) | (slot << 24);
      swt[pos] = recv_wgt[j];
    }
  }

  // ================= P5: tail — reset allocator, advance epoch ============================
  __syncthreads();
  if (bid == 0 && tid == 0) {
    dest_counter[0] = 0u;     // next epoch's allocations restart at row 0
    sort_count[0] = epoch;    // plain store; next kernel on this stream reads it
  }
}
