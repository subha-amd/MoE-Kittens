// ================================================================================================
// k0d_push.hip — plan-free push dispatch for decode, with a folded doorbell rendezvous.
//
// A pull-based dispatch has to pay for three things a push producer does not:
//   1. a route all-gather plus a full plan build, needed only because a pull consumer must know
//      every rank's routes to decide what to pull;
//   2. a barrier before the pull, needed only because a pull consumer must wait for REMOTE
//      quantization before reading remote rows;
//   3. the pull itself.
// A push producer reads only its own routes (already local) and its own quantized rows (local,
// in stream order after the quant kernel, so it never waits on a remote quant), allocates receive
// rows on the destination with a remote-atomic slot allocator, and posts payload stores
// fire-and-forget through ShmemPtrP2p.  The sorted GEMM metadata the plan used to build is
// rebuilt destination-side from the received expert ids by k0d_dsort.
//
// WHAT (one warp per source token tau):
//   lane s < TOPK holds slot s:  eid = my_ids[tau*TOPK+s], dest = eid / E (E = experts per rank).
//   Routes are DEDUPLICATED by destination: a (token, dest) pair is pushed once no matter how
//   many of the token's slots land there, and the lowest such slot is PRIMARY.  Primary lanes
//   allocate one receive row each with a remote atomicAdd on the destination's dest_counter (an
//   agent-scope local atomic when the destination is this rank); the returned value IS the
//   arrival row.  Arrival order is nondeterministic.
//   The warp then loads its 7168 B FP8 row once (7 x 16 B per lane) and, for each destination,
//   posts 7 row stores plus a 224 B row-major scale store, a 32 B expert-id store and a 32 B
//   route-weight store into that rank's symmetric receive buffers.
//   Owner-side combine staging (pull_stage / pull_cnt) is written locally as it goes: the owner
//   already knows where each of its tokens landed, so pull_src needs no all-gather either.
//
// RENDEZVOUS (folded, reset-free, monotonic):
//   every CTA does __syncthreads, then thread 0 issues __threadfence_system to release all of its
//   payload stores and bumps an arrival counter.  The last arriving CTA bumps recv_doorbell[cur]
//   on every rank.  recv_doorbell[src] on rank R therefore counts R's completed pushes from src,
//   forever.  k0d_dsort waits for recv_doorbell[src] >= epoch against a local epoch counter.  No
//   flag is ever reset; dest_counter is reset by k0d_dsort's tail, its only reader.
//
// CROSS-EPOCH SAFETY:
//   * push(N+1) cannot write rank R's receive buffers while R still reads them for epoch N:
//     push(N+1) starts only after this rank's combine(N) returned in stream order, and the
//     combine head fires only after every rank's phase-2 GEMM for epoch N completed, which is
//     after R's sort and GEMM.
//   * the reverse hazard (a remote combine(N) still pulling this rank's part while the next epoch
//     reuses it) is closed by k0d_dsort's doorbell wait, not here.
//
// GRAPH SAFETY: posted peer stores and remote atomics through ShmemPtrP2p only.  No host reads;
// every count is device-side.
//
// pperr bits owned here: 128 receive-row overflow (T_loc_max), 256 shape guard.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#define K0PD_MAXE 64
#define K0PD_K 7168
#define K0PD_NG 56

extern "C" __global__ void __launch_bounds__(256, 8) k0d_push(
    const int* __restrict__ my_ids,              // SYMMETRIC [T*TOPK] route expert ids (LOCAL read)
    const float* __restrict__ my_wgt,            // SYMMETRIC [T*TOPK] route weights  (LOCAL read)
    const unsigned char* __restrict__ a_src,     // SYMMETRIC [T, 7168] fp8 rows      (LOCAL read)
    const float* __restrict__ sc_src,            // SYMMETRIC [T, 56] row-major scales(LOCAL read)
    unsigned char* __restrict__ a_dst,           // SYMMETRIC [T_loc_max, 7168] fp8 recv rows
    float* __restrict__ sc_stage,                // SYMMETRIC [T_loc_max, 56] row-major recv scales
    int* __restrict__ recv_eid,                  // SYMMETRIC [T_loc_max, TOPK] recv expert ids
    float* __restrict__ recv_wgt,                // SYMMETRIC [T_loc_max, TOPK] recv route weights
    unsigned int* __restrict__ dest_counter,     // SYMMETRIC [1]: arrival-row allocator (dsort resets)
    unsigned int* __restrict__ recv_doorbell,    // SYMMETRIC [world]: monotonic push-done flags
    int* __restrict__ pull_stage,                // LOCAL [T*TOPK, 2]: (dest,row) or (-1,-1)
    int* __restrict__ pull_cnt,                  // LOCAL [T]: per-token fanout
    unsigned int* __restrict__ grid_count,       // LOCAL [1]: cumulative arrival counter
    int* __restrict__ pperr,
    int T, int TOPK, int E, int cur, int world, int T_loc_max) {
  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nw = gridDim.x * (blockDim.x >> 6);

  if (TOPK != 8 || E > K0PD_MAXE || world > 8 || T_loc_max > 0x00FFFFFF) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(pperr, 256);
    return;
  }

  for (int tau = gw; tau < T; tau += nw) {
    // ---- slot s on lane s; lanes >= TOPK are inert (dest = -1, never primary) ----
    const int s = lane;
    const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
    const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
    const int dest = (eid >= 0) ? eid / E : -1;

    // Dedup by destination.  The tau loop trip count is warp-uniform, so the match executes in
    // convergent context (the k0d_fused rule: the collective must not sit in a divergent guard).
    const unsigned long long match = __match_any_sync(__activemask(), dest);
    const bool primary = (s < TOPK) && ((match & ((1ULL << s) - 1ULL)) == 0ULL);
    const unsigned long long pmask = __ballot(primary);
    const int fanout = __popcll(pmask);

    // ---- receive-row allocation: one remote atomicAdd per (token, dest), lane-parallel ----
    unsigned int row = 0u;
    if (primary) {
      if (dest == cur) {
        row = __hip_atomic_fetch_add(dest_counter, 1u, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_AGENT);
      } else {
        unsigned int* peer = reinterpret_cast<unsigned int*>(
            mori::shmem::ShmemPtrP2p((uint64_t)dest_counter, dest));
        row = __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      if (row >= (unsigned int)T_loc_max) atomicOr(pperr, 128);
    }

    // ---- owner-side combine staging (LOCAL): the owner knows where its tokens landed ----
    if (s < TOPK) {
      pull_stage[((size_t)tau * TOPK + s) * 2 + 0] = primary ? dest : -1;
      pull_stage[((size_t)tau * TOPK + s) * 2 + 1] = primary ? (int)row : -1;
    }
    if (lane == 0) pull_cnt[tau] = fanout;

    if (fanout == 0) continue;   // defensive; a live token always has destinations

    // ---- payload registers: row loaded ONCE (7 x 16 B/lane); scales one 224 B request ----
    const unsigned char* srow = a_src + (size_t)tau * K0PD_K;
    uint4 v[7];
#pragma unroll
    for (int c = 0; c < 7; ++c)
      v[c] = *reinterpret_cast<const uint4*>(srow + (c << 10) + (lane << 4));
    uint4 sv;
    if (lane < 14)
      sv = *reinterpret_cast<const uint4*>(
          reinterpret_cast<const unsigned char*>(sc_src) + (size_t)tau * (K0PD_NG * 4) +
          (lane << 4));

    // metadata vectors packed from the per-slot lane registers (uniform shfls)
    const uint4 ep0 = make_uint4((unsigned int)__shfl(eid, 0), (unsigned int)__shfl(eid, 1),
                                 (unsigned int)__shfl(eid, 2), (unsigned int)__shfl(eid, 3));
    const uint4 ep1 = make_uint4((unsigned int)__shfl(eid, 4), (unsigned int)__shfl(eid, 5),
                                 (unsigned int)__shfl(eid, 6), (unsigned int)__shfl(eid, 7));
    const float4 wp0 = make_float4(__shfl(wgt, 0), __shfl(wgt, 1), __shfl(wgt, 2), __shfl(wgt, 3));
    const float4 wp1 = make_float4(__shfl(wgt, 4), __shfl(wgt, 5), __shfl(wgt, 6), __shfl(wgt, 7));

    // ---- fanout stores, ascending slot order; every store POSTED (fire-and-forget) ----
    unsigned long long pm = pmask;
#pragma unroll 1
    for (int j = 0; j < fanout; ++j) {
      const int src_lane = __ffsll((long long)pm) - 1;
      pm &= pm - 1;
      const int pe = __shfl(dest, src_lane);
      const int drow_id = __shfl((int)row, src_lane);
      if (drow_id < 0 || drow_id >= T_loc_max) continue;   // overflow already flagged
      unsigned char* drow = reinterpret_cast<unsigned char*>(mori::shmem::ShmemPtrP2p(
          (uint64_t)(a_dst + (size_t)drow_id * K0PD_K), pe));
#pragma unroll
      for (int c = 0; c < 7; ++c)
        *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = v[c];
      if (lane < 14) {
        unsigned char* dsc = reinterpret_cast<unsigned char*>(mori::shmem::ShmemPtrP2p(
            (uint64_t)(reinterpret_cast<unsigned char*>(sc_stage) +
                       (size_t)drow_id * (K0PD_NG * 4)),
            pe));
        *reinterpret_cast<uint4*>(dsc + (lane << 4)) = sv;
      }
      if (lane < 2) {
        int* de = reinterpret_cast<int*>(mori::shmem::ShmemPtrP2p(
            (uint64_t)(recv_eid + (size_t)drow_id * TOPK), pe));
        *reinterpret_cast<uint4*>(de + (lane << 2)) = (lane == 0) ? ep0 : ep1;
      } else if (lane < 4) {
        float* dw = reinterpret_cast<float*>(mori::shmem::ShmemPtrP2p(
            (uint64_t)(recv_wgt + (size_t)drow_id * TOPK), pe));
        *reinterpret_cast<float4*>(dw + ((lane - 2) << 2)) = (lane == 2) ? wp0 : wp1;
      }
    }
  }

  // ---- folded doorbell: publish ALL this rank's pushes before any consumer reads them ----
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();   // release this CTA's payload stores system-wide BEFORE arrival
    const unsigned int old =
        __hip_atomic_fetch_add(grid_count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if ((old % gridDim.x) == (gridDim.x - 1)) {
      for (int pe = 0; pe < world; ++pe) {
        if (pe == cur) {
          __hip_atomic_fetch_add(&recv_doorbell[cur], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        } else {
          unsigned int* peer = reinterpret_cast<unsigned int*>(
              mori::shmem::ShmemPtrP2p((uint64_t)(recv_doorbell + cur), pe));
          __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
}
