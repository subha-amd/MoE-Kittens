// Prefill destination-pull gather. One warp reads a 7168-byte FP8 row as seven 16-byte chunks per
// lane and transposes its 56 scales to group-major order. Local and peer rows share the same
// ShmemPtrP2p path. Invalid rows are zero-filled to avoid stale replay data.

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstddef>
#include <cstdint>

extern "C" __global__ void __launch_bounds__(256, 4) k0pf_gather_rows_fp8(
    unsigned char* __restrict__ a_dst,        // [T_loc_max, K] fp8, LOCAL destination
    float* __restrict__ sc_dst,               // group-major scales, LOCAL: sc_dst[g*T_loc + t]
    const unsigned char* __restrict__ a_src,  // [T_src, K] fp8, SYMMETRIC source (all ranks)
    const float* __restrict__ sc_src,         // [T_src, NG] fp32 row-major, SYMMETRIC source
    const int* __restrict__ gath,             // [T_loc_max, 2] (src_rank, src_token)
    const int* __restrict__ nvi,              // [2]; nvi[1] = T_loc, read ON DEVICE
    float* __restrict__ sc_stage,             // ABI compat; unused
    int T_src, int T_loc_max, int K, int cur) {
  (void)sc_stage;
  (void)T_loc_max;
  (void)cur;   // ShmemPtrP2p returns the local pointer when pe == cur.

  const int T_loc = nvi[1];
  const int NG = K >> 7;
  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nw = gridDim.x * (blockDim.x >> 6);

  for (int t = gw; t < T_loc; t += nw) {
    const int src_rank = gath[(size_t)t * 2 + 0];
    const int src_tok = gath[(size_t)t * 2 + 1];
    // Validate src_rank before ShmemPtrP2p indexes the peer table.
    const bool valid =
        (src_rank >= 0) && (src_rank < 8) && (src_tok >= 0) && (src_tok < T_src);

    unsigned char* __restrict__ drow = a_dst + (size_t)t * (size_t)K;

    if (!valid) {
      for (int i = lane; i < K; i += 64) drow[i] = 0;
      if (lane < NG) sc_dst[(size_t)lane * (size_t)T_loc + t] = 0.0f;
      continue;
    }

    const unsigned char* srow = reinterpret_cast<const unsigned char*>(
        mori::shmem::ShmemPtrP2p((uint64_t)(a_src + (size_t)src_tok * (size_t)K), src_rank));
    const float* ssc = reinterpret_cast<const float*>(
        mori::shmem::ShmemPtrP2p((uint64_t)(sc_src + (size_t)src_tok * (size_t)NG), src_rank));

    // Issue all row and scale loads before stores.
    if (K == 7168) {
      uint4 v[7];
      float sv = 0.0f;
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        v[c] = *reinterpret_cast<const uint4*>(srow + (c << 10) + (lane << 4));
      }
      if (lane < NG) sv = ssc[lane];
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = v[c];
      }
      if (lane < NG) sc_dst[(size_t)lane * (size_t)T_loc + t] = sv;
    } else {
      // Generic 16-byte chunk path with a scalar tail.
      const int nvec = K >> 10;
      for (int c = 0; c < nvec; ++c) {
        const uint4 vv =
            *reinterpret_cast<const uint4*>(srow + (c << 10) + (lane << 4));
        *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = vv;
      }
      for (int i = (nvec << 10) + lane; i < K; i += 64) drow[i] = srow[i];
      if (lane < NG) sc_dst[(size_t)lane * (size_t)T_loc + t] = ssc[lane];
    }
  }
}
