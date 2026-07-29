// ================================================================================================
// k0d_combine.hip — owner-pull combine with the all-rank rendezvous folded into the kernel head,
// plus an optional completion signal for the next epoch's ingress.  No standalone barrier kernel.
//
// Placement rationale: the release fence sits at the COMBINE head, after the phase-2 GEMM has
// fully completed in stream order, so its epilogue writes are already in memory and the fence is
// nearly free.  Fencing inside the GEMM tail instead has to drain every outstanding epilogue
// atomic per CTA and measures slower at decode.  This mirrors where MoRI places its own
// cross-device barrier: inside the combine kernel, not as a separate launch.
//
// RENDEZVOUS HEAD (all blocks): a cumulative grid counter; the last block to arrive bumps
// arr_flags[cur] on every peer under a system release; every block then spins on its local copies
// of the world's flags (bounded, fail-closed), takes an acquire fence, and proceeds to the pulls.
// Epoch N = old / gridDim.x + 1, derived by each block from its own pre-increment value, so no
// counter is ever reset and graph replay stays safe.
// A combine block can only launch after this rank's phase 2 completed (stream order), so all
// combine blocks arriving on all ranks implies every rank's phase 2 is done.  Phase 2 itself is
// unmodified.
//
// TAIL SIGNAL (combine_done): a second counter after the pulls; the last block bumps
// don_flags[cur] on every peer.  An ingress kernel that wants to overlap epochs waits on
// don_flags.  The pulls are reads, so the fence before the poke only orders the counter after
// them.
//
// COMBINE BODY: warp per token; up to 8 producer pointers translated once; all producers' 16 B
// loads in flight per 1 KB chunk; FP32 accumulation in producer order in registers; one BF16 RNE
// store per chunk; direct P2P reads through ShmemPtrP2p with no stage buffer, no LDS and no
// __syncthreads in the body.  Tokens with empty fanout write zeros.
//
// Launch: with arr_flags == nullptr the head and tail compile away and any grid is accepted.
// With the fused rendezvous enabled the grid must be the fixed (16, 256) decode shape so the
// cumulative counters keep one stable blocks-per-epoch denominator across every graph replay.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstddef>
#include <cstdint>

__device__ __forceinline__ float k0dc_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ unsigned short k0dc_f32_to_bf16_rne(float f) {
  const unsigned int u = __float_as_uint(f);
  const unsigned int lsb = (u >> 16) & 1u;
  return (unsigned short)((u + 0x7FFFu + lsb) >> 16);
}

extern "C" __global__ void __launch_bounds__(256, 8) k0d_combine(
    unsigned short* __restrict__ out,         // [T, H] bf16, LOCAL
    const unsigned short* __restrict__ part,  // [MROWS, H] bf16, SYMMETRIC producer partials
    const int* __restrict__ pull_ptr,         // [T+1]
    const int* __restrict__ pull_src,         // [T*world, 2]
    unsigned short* __restrict__ stage,       // ABI compat; unused
    int T, int H, int cur,
    // ---- fused rendezvous state (pass arr_flags == nullptr to disable it) ----
    unsigned int* __restrict__ arr_flags,     // SYMMETRIC [world]: arrival flags
    unsigned int* __restrict__ arr_count,     // LOCAL [1]: cumulative arrival counter
    unsigned int* __restrict__ don_flags,     // SYMMETRIC [world]: combine_done flags
    unsigned int* __restrict__ don_count,     // LOCAL [1]: cumulative done counter
    int* __restrict__ pperr, long long spin_limit, int world) {
  (void)stage;
  (void)cur;
  const int tid = threadIdx.x;
  if (arr_flags != nullptr && gridDim.x != 16) {
    if (blockIdx.x == 0 && tid == 0) atomicOr(pperr, 16);
    return;
  }

  // ================= fused all-rank rendezvous ==================================================
  if (arr_flags != nullptr) {
    __shared__ unsigned int s_N;
    if (tid == 0) {
      const unsigned int old =
          __hip_atomic_fetch_add(arr_count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      s_N = old / gridDim.x + 1u;                          // this replay's epoch
      if ((old % gridDim.x) == (gridDim.x - 1u)) {         // last combine block to arrive
        __threadfence_system();                            // release the phase-2 writes
        for (int pe = 0; pe < world; ++pe) {
          if (pe == cur) {
            __hip_atomic_fetch_add(&arr_flags[cur], 1u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
          } else {
            unsigned int* peer = reinterpret_cast<unsigned int*>(
                mori::shmem::ShmemPtrP2p((uint64_t)(arr_flags + cur), pe));
            __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }
      }
    }
    __syncthreads();                                       // s_N visible; arrival pokes issued
    if (tid < world) {
      const unsigned int N = s_N;
      volatile const unsigned int* vf = arr_flags;
      long long spins = 0;
      while (vf[tid] < N) {
        if (++spins > spin_limit) { atomicOr(pperr, 8); break; }
        __builtin_amdgcn_s_sleep(4);
      }
    }
    __syncthreads();
    __threadfence_system();                                // acquire
  }

  // ================= combine body ===============================================================
  const int lane = threadIdx.x & 63;
  const int gw = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int nw = gridDim.x * (blockDim.x >> 6);

  for (int tok = gw; tok < T; tok += nw) {
    const int lo = pull_ptr[tok];
    const int hi = pull_ptr[tok + 1];
    const int fanout = hi - lo;
    unsigned short* __restrict__ orow = out + (size_t)tok * (size_t)H;

    if (fanout <= 0) {
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

    unsigned long long pbase = 0ull;
    if (lane < fanout) {
      const int p = pull_src[(size_t)(lo + lane) * 2 + 0];
      const int row = pull_src[(size_t)(lo + lane) * 2 + 1];
      pbase = (unsigned long long)mori::shmem::ShmemPtrP2p(
          (uint64_t)(part + (size_t)row * (size_t)H), p);
    }

    if (H == 7168) {
#pragma unroll 1
      for (int c = 0; c < 14; ++c) {
        const size_t off = (size_t)(c << 10) + (size_t)(lane << 4);
        uint4 v[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (j < fanout) {
            const unsigned long long pb = __shfl(pbase, j);
            v[j] = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const unsigned char*>(pb) + off);
          }
        }
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (j < fanout) {
            const unsigned short* u16 = reinterpret_cast<const unsigned short*>(&v[j]);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[e] += k0dc_bf16_to_f32(u16[e]);
          }
        }
        union {
          uint4 pack;
          unsigned short u16[8];
        } pk;
#pragma unroll
        for (int e = 0; e < 8; ++e) pk.u16[e] = k0dc_f32_to_bf16_rne(acc[e]);
        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(orow) + off) = pk.pack;
      }
    } else {
      for (int h = lane * 8; h < H; h += 64 * 8) {
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int j = 0; j < fanout; ++j) {
          const unsigned long long pb = __shfl(pbase, j);
          const unsigned short* prow = reinterpret_cast<const unsigned short*>(pb) + h;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[e] += k0dc_bf16_to_f32(prow[e]);
        }
        unsigned short tmp[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) tmp[e] = k0dc_f32_to_bf16_rne(acc[e]);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          if (h + e < H) orow[h + e] = tmp[e];
        }
      }
    }
  }

  // ================= tail signal: combine_done (feeds next epoch's ingress head-wait) ==========
  if (arr_flags != nullptr && don_flags != nullptr) {
    __syncthreads();                                       // all pulls precede the signal
    if (tid == 0) {
      __threadfence_system();                              // order reads before the counter
      const unsigned int old =
          __hip_atomic_fetch_add(don_count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if ((old % gridDim.x) == (gridDim.x - 1u)) {
        for (int pe = 0; pe < world; ++pe) {
          if (pe == cur) {
            __hip_atomic_fetch_add(&don_flags[cur], 1u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
          } else {
            unsigned int* peer = reinterpret_cast<unsigned int*>(
                mori::shmem::ShmemPtrP2p((uint64_t)(don_flags + cur), pe));
            __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          }
        }
      }
    }
  }
}
