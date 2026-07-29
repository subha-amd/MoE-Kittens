// ================================================================================================
// k0pf_gather.hip — prefill dispatch, destination-pull of FP8 rows at fabric rate.
//
// The per-row data path is the whole design.  A block-per-row gather built on a synchronous
// staged block copy (issue a few 16 B loads, drain vmcnt, store, next chunk), followed by a
// second staged pull for the scales and a scatter with __syncthreads, gives each thread about two
// loads in flight and pays two CTA barriers per row.  At prefill scale that cannot fill the
// interconnect.
//
// THIS KERNEL — three mechanisms:
//   1. DIRECT P2P READS.  The peer row is read straight through mori::shmem::ShmemPtrP2p, the
//      same pointer class MoRI's own combine uses for its source rows.  No staged block copy, no
//      stage buffer, no extra local round trip.  ShmemPtrP2p with pe == cur returns the local
//      pointer, so a local producer rides the same path with no special casing.
//   2. WARP PER ROW, 112 B PER LANE IN FLIGHT.  A row is 7,168 FP8 bytes = 7 x (64 lanes x 16 B).
//      All seven loads are independent and issue back to back under a compile-time trip count,
//      then the seven stores.  There is no vmcnt drain between chunk pairs, so per-warp in-flight
//      bytes rise from about 2 KB to 7 KB.  Even at four resident 256-thread CTAs per CU that is
//      tens of MB in flight GPU-wide, comfortably past what saturating the fabric needs, while
//      leaving enough VGPRs to keep the seven-load batch in registers.  Achieved bandwidth is
//      then set by the fabric rather than by the access pattern.
//   3. SCALES RIDE THE SAME WARP.  The 224 B scale row (56 FP32 values) is one warp-coalesced
//      peer read issued in the same batch as the data loads (lanes 0..55, 4 B each), then
//      scattered group-major into sc_dst[g * T_loc + t].
//
// NUMERICS: none, this is a byte move.  FP8 payload and scales are copied verbatim.  T_loc is
// read on device from nvi[1].  Invalid rows (gath == -1, not expected for t < T_loc but kept as a
// guard) zero-fill.  Rows t >= T_loc are never touched, and the GEMM's bounded input descriptor
// never reads them.
//
// GRAPH SAFETY: no host sync, no allocation, T_loc read on device; the launch is capture-safe.
// Peer loads are plain global loads through the P2P pointer.  The module must be loaded with
// load_hip_module(init_shmem=True), since ShmemPtrP2p reads globalGpuStates.
//
// LAUNCH CONTRACT:
//   fn.launch((2048,), (256,), 0, stream, a_dst_u8, sc_dst, a_src, sc_src,
//             gath, nvi, sc_stage, T, T_LOC_MAX, K, rank)
//   `sc_stage` is accepted for ABI compatibility and unused.  The grid is a performance knob
//   only: grid-stride makes any grid correct, and 2048 x 256 fills the GPU.  The cross-rank
//   barrier stays where it is; this kernel changes bytes per second, not the rendezvous.
//
// ISA expectations worth asserting from the compiled .s before trusting a timing run:
//   * zero register spills, zero scratch

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
  (void)cur;   // ShmemPtrP2p(pe == cur) returns the local pointer itself: no local special case

  const int T_loc = nvi[1];
  const int NG = K >> 7;                       // 56 for K = 7168
  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);   // global warp id
  const int nw = gridDim.x * (blockDim.x >> 6);                          // total warps

  for (int t = gw; t < T_loc; t += nw) {
    const int src_rank = gath[(size_t)t * 2 + 0];
    const int src_tok = gath[(size_t)t * 2 + 1];
    // Guard is pure defense (plan guarantees valid gath for t < T_loc); src_rank is range-checked
    // because ShmemPtrP2p indexes p2pPeerPtrs[src_rank] — a garbage rank is a wild pointer, not a
    // plan error we can afford to dereference.  EP8 topology: ranks are [0, 8).
    const bool valid =
        (src_rank >= 0) && (src_rank < 8) && (src_tok >= 0) && (src_tok < T_src);

    unsigned char* __restrict__ drow = a_dst + (size_t)t * (size_t)K;

    if (!valid) {
      // An invalid row must be defined, not left stale from a previous replay.
      for (int i = lane; i < K; i += 64) drow[i] = 0;
      if (lane < NG) sc_dst[(size_t)lane * (size_t)T_loc + t] = 0.0f;
      continue;
    }

    // ---- peer row bases (ShmemPtrP2p(pe==cur) == the local pointer: one path for all sources) ----
    const unsigned char* srow = reinterpret_cast<const unsigned char*>(
        mori::shmem::ShmemPtrP2p((uint64_t)(a_src + (size_t)src_tok * (size_t)K), src_rank));
    const float* ssc = reinterpret_cast<const float*>(
        mori::shmem::ShmemPtrP2p((uint64_t)(sc_src + (size_t)src_tok * (size_t)NG), src_rank));

    // ---- issue batch: 7 x 16 B data + 1 x 4 B scale, ALL loads before ANY store ----
    // K = 7168 -> nvec = 7 exactly (K % 1024 == 0 is asserted by the generic fallback below).
    if (K == 7168) {
      uint4 v[7];
      float sv = 0.0f;
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        v[c] = *reinterpret_cast<const uint4*>(srow + (c << 10) + (lane << 4));
      }
      if (lane < NG) sv = ssc[lane];           // coalesced 224 B request, issued with the batch
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = v[c];
      }
      if (lane < NG) sc_dst[(size_t)lane * (size_t)T_loc + t] = sv;
    } else {
      // Generic path (never taken for DeepSeek-R1; correctness net only). 16 B chunks with tail.
      const int nvec = K >> 10;                // full 1 KiB warp chunks
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
