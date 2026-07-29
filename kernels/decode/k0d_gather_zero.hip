// ================================================================================================
// k0d_gather_zero.hip — pull-side gather with the partial-buffer clear folded in.
//
// A standalone part.zero_() costs a launch plus a few microseconds even when the buffer is small,
// and at prefill it clears the full capacity rather than only the live rows.  The gather already
// visits every live row t < T_loc, so the [0, T_loc) clear rides along in the same warp visit.
// Zeroed rows are exactly what the GEMM's accumulating phase-2 epilogue requires; rows >= T_loc
// are never read.
//
// Semantics are the plain gather (same direct-P2P pull, same group-major scale scatter, same
// invalid-row zero fill) plus part[t][:] = 0 in BF16 for every t < T_loc.  The part write is a
// local 16 B/lane store stream of 14 x 1 KB warp chunks issued after the row pull, so it is
// nearly free next to the fabric pull.
//
// Argument order is the plain gather's with `part` appended.  Any grid is correct (grid-stride);
// decode uses (128, 256) and prefill (2048, 256).
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstddef>
#include <cstdint>

extern "C" __global__ void __launch_bounds__(256, 8) k0d_gather_zero(
    unsigned char* __restrict__ a_dst,        // [T_loc_max, K] fp8, LOCAL destination
    float* __restrict__ sc_dst,               // group-major scales, LOCAL: sc_dst[g*T_loc + t]
    const unsigned char* __restrict__ a_src,  // [T_src, K] fp8, SYMMETRIC source (all ranks)
    const float* __restrict__ sc_src,         // [T_src, NG] fp32 row-major, SYMMETRIC source
    const int* __restrict__ gath,             // [T_loc_max, 2] (src_rank, src_token)
    const int* __restrict__ nvi,              // [2]; nvi[1] = T_loc, read ON DEVICE
    float* __restrict__ sc_stage,             // ABI compat; unused
    int T_src, int T_loc_max, int K, int cur,
    unsigned short* __restrict__ part,        // [T_loc_max, 7168] bf16 — zero rows < T_loc
    // ---- optional arm-2 per-source readiness (pass quant_flags == nullptr for barrier arm) ----
    const unsigned int* __restrict__ quant_flags,   // SYMMETRIC [world]: quant_done per source
    const unsigned int* __restrict__ quant_count,   // LOCAL [1]: current quant epoch (stable)
    int* __restrict__ pperr2, long long spin_limit, int world) {
  (void)sc_stage;
  (void)T_loc_max;
  (void)cur;

  const int T_loc = nvi[1];
  const int NG = K >> 7;
  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nw = gridDim.x * (blockDim.x >> 6);
  // Arm 2 epoch: quant_count[0] is bumped by the quant tail before this kernel launches
  // (stream order), so it stably reads the current replay's epoch N.
  const unsigned int epoch = (quant_flags != nullptr) ? quant_count[0] : 0u;

  for (int t = gw; t < T_loc; t += nw) {
    // ---- the fold: zero this row's partial buffer (7168 bf16 = 14 warp chunks) ----
    {
      unsigned char* prow = reinterpret_cast<unsigned char*>(part) + (size_t)t * 14336;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int c = 0; c < 14; ++c)
        *reinterpret_cast<uint4*>(prow + (c << 10) + (lane << 4)) = z;
    }

    const int src_rank = gath[(size_t)t * 2 + 0];
    const int src_tok = gath[(size_t)t * 2 + 1];
    const bool valid =
        (src_rank >= 0) && (src_rank < 8) && (src_tok >= 0) && (src_tok < T_src);

    // Arm 2 per-source readiness: wait only for THIS row's source rank's quant to land.
    // Rows whose source is already done proceed immediately (stragglers don't gate them).
    if (quant_flags != nullptr && valid) {
      if (lane == 0) {
        volatile const unsigned int* vf = quant_flags;
        long long spins = 0;
        while (vf[src_rank] < epoch) {
          if (++spins > spin_limit) { atomicOr(pperr2, 4); break; }
          __builtin_amdgcn_s_sleep(4);
        }
      }
      __builtin_amdgcn_wave_barrier();                        // hold the warp until lane 0 confirms
      __threadfence_system();                                // acquire
    }

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
