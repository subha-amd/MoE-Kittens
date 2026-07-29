// Plan-free decode push. One warp handles a token and deduplicates routes by destination; the
// lowest matching slot allocates a receive row. Arrival order is nondeterministic.
//
// Each CTA fences its posted payload stores before incrementing grid_count. The last CTA signals
// recv_doorbell[cur] on every rank. Doorbells are monotonic and never reset; k0d_dsort resets the
// receive-row allocator after consuming it.
//
// Stream order prevents push(N+1) from overwriting receive buffers used by epoch N. The
// k0d_dsort doorbell wait prevents part reuse while a remote combine still reads it.
// pperr: 128 receive-row overflow, 256 shape guard.

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
    // Lanes below TOPK own slots; other lanes use dest = -1.
    const int s = lane;
    const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
    const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
    const int dest = (eid >= 0) ? eid / E : -1;

    // __match_any_sync must execute in warp-uniform control flow.
    const unsigned long long match = __match_any_sync(__activemask(), dest);
    const bool primary = (s < TOPK) && ((match & ((1ULL << s) - 1ULL)) == 0ULL);
    const unsigned long long pmask = __ballot(primary);
    const int fanout = __popcll(pmask);

    // One receive-row allocation per distinct destination.
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

    // Record destination rows for owner-side combine.
    if (s < TOPK) {
      pull_stage[((size_t)tau * TOPK + s) * 2 + 0] = primary ? dest : -1;
      pull_stage[((size_t)tau * TOPK + s) * 2 + 1] = primary ? (int)row : -1;
    }
    if (lane == 0) pull_cnt[tau] = fanout;

    if (fanout == 0) continue;

    // Load the row once before fanout.
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

    const uint4 ep0 = make_uint4((unsigned int)__shfl(eid, 0), (unsigned int)__shfl(eid, 1),
                                 (unsigned int)__shfl(eid, 2), (unsigned int)__shfl(eid, 3));
    const uint4 ep1 = make_uint4((unsigned int)__shfl(eid, 4), (unsigned int)__shfl(eid, 5),
                                 (unsigned int)__shfl(eid, 6), (unsigned int)__shfl(eid, 7));
    const float4 wp0 = make_float4(__shfl(wgt, 0), __shfl(wgt, 1), __shfl(wgt, 2), __shfl(wgt, 3));
    const float4 wp1 = make_float4(__shfl(wgt, 4), __shfl(wgt, 5), __shfl(wgt, 6), __shfl(wgt, 7));

    // Post fanout stores in ascending slot order.
    unsigned long long pm = pmask;
#pragma unroll 1
    for (int j = 0; j < fanout; ++j) {
      const int src_lane = __ffsll((long long)pm) - 1;
      pm &= pm - 1;
      const int pe = __shfl(dest, src_lane);
      const int drow_id = __shfl((int)row, src_lane);
      if (drow_id < 0 || drow_id >= T_loc_max) continue;
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

  // Publish this rank's pushes before consumers read them.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();   // Release this CTA's payload stores before arrival.
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
