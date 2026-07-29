// ================================================================================================
// k0pf_combine.hip — prefill combine: owner-pull over producers at fabric rate, no stage copy.
//
// The pull-side data path is the whole design.  A block-per-token combine that loops producers
// serially pays, per producer, a synchronous block copy of the 14,336 B peer row through
// registers into a global stage buffer, two __syncthreads, and an FP32 LDS accumulate over a
// 28 KB accumulator — roughly four sequential fabric round trips per token, three times the
// necessary local traffic, and CTA barriers between every producer.
//
// THIS KERNEL:
//   1. DIRECT P2P READS of the symmetric `part` rows through ShmemPtrP2p.  MoRI's own combine
//      uses the same pointer class for its source rows, but additionally stages every received
//      row locally before its barrier; we read the producer's published row in place, so that
//      copy does not exist on our side at all.
//   2. WARP PER TOKEN, FOUR PRODUCERS IN FLIGHT.  Lanes 0..fanout-1 translate the up-to-8
//      producer row pointers once, then the row is consumed in 14 x 1 KiB warp chunks.  Per
//      chunk the producers are handled as two ordered groups of four: four 16 B/lane loads issue
//      back to back, then their FP32 accumulation in producer order, then the next four.  That
//      keeps roughly 4 KB per warp in flight without spilling the loaded vectors to scratch.
//   3. ZERO LDS, ZERO __syncthreads, ZERO stage buffer.  The FP32 accumulator lives in 8
//      registers per lane per chunk and the output is one 16 B BF16 RNE store per chunk.
//      Tokens with empty fanout write zeros.
//
// NUMERICS: same FP32 accumulator, same producer order (ascending pull_src index), same BF16 to
// FP32 unpack and RNE pack as the simple combine.  Only the load schedule changed, never the
// arithmetic order.  The rows this kernel sums are already weighted by the GEMM's epilogue, so
// this is a pure sum with no top-k weights.  The cross-rank barrier stays where it is; this
// kernel changes bytes per second, not the rendezvous.
//
// GRAPH SAFETY: device-read pull_ptr / pull_src only, no host sync, capture-safe.  Load the
// module with load_hip_module(init_shmem=True), since ShmemPtrP2p reads globalGpuStates.
//
// LAUNCH CONTRACT:
//   fn.launch((1024,), (256,), 0, stream, out, part, pull_ptr, pull_src, stage, T, H, rank)
//   `stage` is accepted for ABI compatibility and unused.  Grid-stride, so any grid is correct;
//   1024 x 256 gives 4 warps per block and covers 4096 tokens.
//
// ISA expectations worth asserting from the compiled .s before trusting a timing run:
//   * zero spills, zero scratch, no s_barrier, no ds_read/ds_write (no LDS at all)
//   * producer loads batched: at least 4 vector loads before the first dependent add or store
//   * ShmemPtrP2p present
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstddef>
#include <cstdint>

// bf16 (stored as uint16) <-> fp32:
//   unpack : bf16 IS the top 16 bits of the fp32 -> __uint_as_float(bits << 16)   (exact)
//   pack   : round-to-nearest-EVEN  (u + 0x7FFF + lsb) >> 16                       (matches v_cvt_pk_bf16_f32)
__device__ __forceinline__ float k0pf_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ unsigned short k0pf_f32_to_bf16_rne(float f) {
  const unsigned int u = __float_as_uint(f);
  const unsigned int lsb = (u >> 16) & 1u;
  return (unsigned short)((u + 0x7FFFu + lsb) >> 16);
}

extern "C" __global__ void __launch_bounds__(256, 8) k0pf_combine(
    unsigned short* __restrict__ out,         // [T, H] bf16, LOCAL — written exactly once per token
    const unsigned short* __restrict__ part,  // [MROWS, H] bf16, SYMMETRIC producer partials
    const int* __restrict__ pull_ptr,         // [T+1]
    const int* __restrict__ pull_src,         // [T*world, 2] (producer_rank, producer_row)
    unsigned short* __restrict__ stage,       // ABI compat; unused
    int T, int H, int cur) {
  (void)stage;

  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nw = gridDim.x * (blockDim.x >> 6);

  for (int tok = gw; tok < T; tok += nw) {
    const int lo = pull_ptr[tok];
    const int hi = pull_ptr[tok + 1];
    const int fanout = hi - lo;
    unsigned short* __restrict__ orow = out + (size_t)tok * (size_t)H;

    if (fanout <= 0) {
      // A token with no producer must be defined, not left stale from a previous replay.
      if (H == 7168) {
        unsigned char* __restrict__ ob = reinterpret_cast<unsigned char*>(orow);
#pragma unroll
        for (int c = 0; c < 14; ++c) {
          *reinterpret_cast<uint4*>(ob + (c << 10) + (lane << 4)) = make_uint4(0u, 0u, 0u, 0u);
        }
      } else {
        for (int h = lane; h < H; h += 64) orow[h] = 0;
      }
      continue;
    }

    // ---- producer row pointers, translated once.  Lane j < fanout owns producer j. ----
    // Every lane that can be shuffled (0..7) owns a valid translated pointer. Inactive producer
    // lanes point at local row 0. This lets each four-load batch issue unconditionally: guarding
    // each load separately makes clang reconverge EXEC and drain vmcnt between producers.
    unsigned long long pbase = 0ull;
    if (lane < 8) {
      int p = cur;
      int row = 0;
      if (lane < fanout) {
        p = pull_src[(size_t)(lo + lane) * 2 + 0];
        row = pull_src[(size_t)(lo + lane) * 2 + 1];
      }
      pbase = (unsigned long long)mori::shmem::ShmemPtrP2p(
          (uint64_t)(part + (size_t)row * (size_t)H), p);
    }

    if (H == 7168) {
      // 14 x 1 KiB warp chunks; all producer loads of a chunk in flight before the ordered FMA.
#pragma unroll 1
      for (int c = 0; c < 14; ++c) {
        const size_t off = (size_t)(c << 10) + (size_t)(lane << 4);
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int jb = 0; jb < 8; jb += 4) {
          uint4 v[4];
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j = jb + jj;
            const unsigned long long pb = __shfl(pbase, j);
            v[jj] = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const unsigned char*>(pb) + off);
          }
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j = jb + jj;
            if (j < fanout) {
              const unsigned short* u16 = reinterpret_cast<const unsigned short*>(&v[jj]);
#pragma unroll
              for (int e = 0; e < 8; ++e) acc[e] += k0pf_bf16_to_f32(u16[e]);
            }
          }
        }
        union {
          uint4 pack;
          unsigned short u16[8];
        } pk;
#pragma unroll
        for (int e = 0; e < 8; ++e) pk.u16[e] = k0pf_f32_to_bf16_rne(acc[e]);
        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(orow) + off) = pk.pack;
      }
    } else {
      // Generic path (never taken for DeepSeek-R1; correctness net only).
      for (int h = lane * 8; h < H; h += 64 * 8) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < fanout; ++j) {
          const unsigned long long pb = __shfl(pbase, j);
          const unsigned short* prow =
              reinterpret_cast<const unsigned short*>(pb) + h;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] += k0pf_bf16_to_f32(prow[e]);
        }
        unsigned short tmp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) tmp[e] = k0pf_f32_to_bf16_rne(acc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (h + e < H) orow[h + e] = tmp[e];
        }
      }
    }
  }
}
