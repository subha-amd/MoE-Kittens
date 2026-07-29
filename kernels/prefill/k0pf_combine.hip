// Prefill owner-pull combine with direct P2P reads. One warp handles a token, translating up to
// eight producer rows once and accumulating them in pull_src order. Producer loads are issued in
// groups of four. Empty fanout writes zero; stage is retained only for ABI compatibility.

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstddef>
#include <cstdint>

// BF16 unpack is exact; packing uses round-to-nearest-even.
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

    // Translate producer row pointers once.
    // Inactive producer lanes use local row 0 so four-load batches remain unconditional.
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
      // Issue producer loads before ordered accumulation.
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
      // Generic-width path.
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
