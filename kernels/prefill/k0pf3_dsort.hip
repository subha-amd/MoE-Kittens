// ================================================================================================
// k0pf3_dsort.hip — destination-side sort and plan construction for the push-dispatched prefill
// region.
//
// After k0pf3_qpush the receive rows have arrived in nondeterministic atomic order and the only
// routing knowledge on this rank is the pushed recv_eid / recv_wgt arrays.  This kernel is the
// local sorter for that state at prefill scale (T_loc roughly 12k-41k rows, pairs = T_loc * 8,
// E = 32 local experts, T = 4096 owner tokens).  It produces exactly the tensors the GEMM and the
// combine already consume, so the GEMM phases, the barrier and the combine body run unmodified:
//   sti[row]  = (arrival_row & 0x00FFFFFF) | (slot << 24)   arrival_row indexes a_dst; the GEMM's
//               A-descriptor bounds token >= nvi[1] to zero, so the padding sentinel is load-safe
//   swt[row]  = route weight (0.0f on padding rows)
//   sei[b]    = local expert id of sorted 32-row block b
//   nvi[0]    = BM32-padded row total, nvi[1] = T_loc
//   pull_ptr / pull_src        owner-side combine CSR, compacted from qpush's pull_stage
//   sc_dst[g * T_loc + t]      group-major scales, transposed locally from row-major sc_stage
//   part[0, T_loc)             zeroed over the EXACT live prefix, which the GEMM's accumulating
//                              phase-2 epilogue requires.  Clearing the live prefix instead of
//                              the full capacity is most of the point at prefill.
//
// SORT ORDER: within an expert's segment, rows are placed by atomic cursor — nondeterministic and
// deliberate.  The GEMM's rows are independent and phase 2's per-row epilogue atomics already
// race, so acceptance is tolerance-based rather than bitwise.  The pull_src compaction is in slot
// order rather than ascending-rank order, which reorders the FP32 combine sum without changing
// its value set.
//
// THE DOORBELL IS ALSO THE CROSS-EPOCH GUARD for part:
//   recv_doorbell[src] >= epoch N+1 proves src's push(N+1) completed, hence src's combine(N)
//   returned in stream order on src.  Once all doorbells are up, every rank's combine(N) is done,
//   so no remote combine can still be pulling this rank's part when the zero phase rewrites it.
//
// PHASES (grid-stride, one internal monotonic grid barrier via a cumulative counter):
//   P0  every CTA: threads tid < world spin on recv_doorbell[tid] >= epoch (bounded,
//       fail-closed); acquire fence; T_loc = dest_counter[0].
//   P1  task split, no barrier.  CTA0 builds the local-expert histogram in LDS and runs the
//       serial padded scan into erb/cnt/cursor scratch, nvi and sei.  CTA1 scans pull_cnt into
//       pull_ptr (256-wide LDS block scan, 16 chunks with a serial carry).  CTAs 2.. zero
//       part[0, T_loc) in 14 x 1 KB warp chunks per row and transpose sc_stage into sc_dst.  The
//       count and the CSR scan hide under the part zero, which is pure store traffic.
//   P2  grid barrier, publishing scratch, nvi and pull_ptr.
//   P3  all CTAs: write the padding sentinel rows over the disjoint ranges [erb+cnt, erb_next),
//       which cannot race the scatter; scatter live pairs through per-expert atomic cursors;
//       compact pull_src from pull_stage via pull_ptr.
//   P4  CTA0 thread 0 resets dest_counter and advances sort_count to this epoch.
//
// pperr bits: 262144 doorbell spin timeout, 524288 T_loc out of range, 1048576 scatter row
// overflow, 2097152 internal grid-barrier spin timeout, 4194304 pad row overflow (PADMAX),
// 131072 shape guard.
// Scratch layout (LOCAL int scratch[256]): erb[0..64] at 0, cnt at 80, cursors at 160.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#define K0P3_MAXE 64
#define K0P3_NG 56
#define K0P3_SC_ERB 0
#define K0P3_SC_CNT 80
#define K0P3_SC_CUR 160

// 256-wide inclusive block scan (LDS), e26 pattern.
__device__ __forceinline__ int k0p3_block_incl_scan(int v, int* tmp) {
  const int tid = threadIdx.x;
  tmp[tid] = v;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    const int add = (tid >= off) ? tmp[tid - off] : 0;
    __syncthreads();
    tmp[tid] += add;
    __syncthreads();
  }
  const int r = tmp[tid];
  __syncthreads();
  return r;
}

extern "C" __global__ void __launch_bounds__(256, 8) k0pf3_dsort(
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

  if (TOPK != 8 || E > K0P3_MAXE || world > 8 || T_loc_max > 0x00FFFFFF || gridDim.x < 3) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 131072);
    return;
  }

  const unsigned int epoch = sort_count[0] + 1u;

  // ================= P0: doorbell wait (also the cross-epoch part guard) ==================
  if (tid < world) {
    volatile const unsigned int* vf = recv_doorbell;
    long long spins = 0;
    while (vf[tid] < epoch) {
      if (++spins > spin_limit) { atomicOr(pperr, 262144); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence_system();   // acquire: recv payloads, dest_counter, and the part-reuse proof

  const int T_loc = (int)dest_counter[0];
  if (T_loc < 0 || T_loc > T_loc_max) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 524288);
    return;
  }

  // ================= P1: task-split plan work (no barrier) ================================
  if (bid == 0) {
    // ---- per-expert counts over the received pairs (LDS histogram) ----
    __shared__ int s_cnt[K0P3_MAXE];
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
        scratch[K0P3_SC_ERB + e] = acc;
        scratch[K0P3_SC_CNT + e] = s_cnt[e];
        acc += ((s_cnt[e] + 31) / 32) * 32;
      }
      scratch[K0P3_SC_ERB + E] = acc;
      nvi[0] = acc;
      nvi[1] = T_loc;
    }
    __syncthreads();
    // ---- scatter cursors + sorted-expert-ids (needs the scan) ----
    {
      const int bcap = PADMAX / 32;
      for (int e = tid; e < E; e += 256) {
        scratch[K0P3_SC_CUR + e] = scratch[K0P3_SC_ERB + e];
        const int b0 = scratch[K0P3_SC_ERB + e] / 32;
        const int b1 = scratch[K0P3_SC_ERB + e + 1] / 32;
        for (int b = b0; b < b1; ++b) {
          if (b >= bcap) { atomicOr(pperr, 4194304); break; }
          sei[b] = e;
        }
      }
    }
  } else if (bid == 1) {
    // ---- owner combine CSR row pointers: 256-wide block scans over pull_cnt ----
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    if (tid == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < T; base += 256) {
      const int tau = base + tid;
      const int f = (tau < T) ? pull_cnt[tau] : 0;
      const int incl = k0p3_block_incl_scan(f, s_tmp);
      if (tau < T) pull_ptr[tau] = s_carry + (incl - f);
      __syncthreads();
      if (tid == 255) s_carry += incl;
      __syncthreads();
    }
    if (tid == 0) pull_ptr[T] = s_carry;
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
      if (lane < K0P3_NG)
        sc_dst[(size_t)lane * (size_t)T_loc + t] = sc_stage[(size_t)t * K0P3_NG + lane];
    }
  }

  // ================= P2: grid barrier (publishes scratch + nvi + pull_ptr) ================
  __syncthreads();
  if (tid == 0) {
    __threadfence();   // device release (scratch/nvi/sei/pull_ptr/part/sc_dst are this GPU's)
    const unsigned int old =
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    volatile unsigned int* vb = gbar;
    long long spins = 0;
    while (*vb < target) {
      if (++spins > spin_limit) { atomicOr(pperr, 2097152); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence();   // device acquire

  // ================= P3: pad sentinel + atomic-cursor scatter + CSR fill ==================
  {
    const int lo = cur * E, hi = lo + E;
    // pad rows live in [erb[e] + cnt[e], erb[e+1]) — disjoint from every scatter position.
    for (int idx = bid * 256 + tid; idx < E * 32; idx += gridDim.x * 256) {
      const int e = idx >> 5, k = idx & 31;
      const int row = scratch[K0P3_SC_ERB + e] + scratch[K0P3_SC_CNT + e] + k;
      if (row < scratch[K0P3_SC_ERB + e + 1]) {
        if (row >= PADMAX) { atomicOr(pperr, 4194304); continue; }
        sti[row] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
        swt[row] = 0.0f;
      }
    }
    const int pairs = T_loc * TOPK;
    for (int j = bid * 256 + tid; j < pairs; j += gridDim.x * 256) {
      const int e = recv_eid[j];
      if (e < lo || e >= hi) continue;
      const int pos = atomicAdd(&scratch[K0P3_SC_CUR + (e - lo)], 1);
      if (pos >= PADMAX) { atomicOr(pperr, 1048576); continue; }
      const int row = j / TOPK;
      const int slot = j - row * TOPK;
      sti[pos] = (row & 0x00FFFFFF) | (slot << 24);
      swt[pos] = recv_wgt[j];
    }
    // owner combine CSR fill: per token, compact its primary (dest,row) pairs in slot order
    for (int tau = bid * 256 + tid; tau < T; tau += gridDim.x * 256) {
      int k = pull_ptr[tau];
      for (int s = 0; s < TOPK; ++s) {
        const int pe = pull_stage[((size_t)tau * TOPK + s) * 2 + 0];
        if (pe < 0) continue;
        const int row = pull_stage[((size_t)tau * TOPK + s) * 2 + 1];
        if (k >= T * world) { atomicOr(pperr, 1048576); break; }
        pull_src[(size_t)k * 2 + 0] = pe;
        pull_src[(size_t)k * 2 + 1] = row;
        ++k;
      }
    }
  }

  // ================= P4: tail — reset allocator, advance epoch ============================
  __syncthreads();
  if (bid == 0 && tid == 0) {
    dest_counter[0] = 0u;     // next epoch's allocations restart at row 0
    sort_count[0] = epoch;    // plain store; next kernel on this stream reads it
  }
}
