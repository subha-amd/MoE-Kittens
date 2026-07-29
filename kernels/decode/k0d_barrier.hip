// Cross-rank stream barrier using monotonic per-source flags.
// Thread 0 releases prior writes and signals every rank. Threads 0..world-1 wait on one local flag
// each, then a system fence acquires peer writes. flags and count are zeroed once and never reset.
// Each barrier instance requires its own pair, and ranks must invoke instances in the same order.

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <cstdint>

extern "C" __global__ void __launch_bounds__(64, 1) k0d_barrier(
    unsigned int* __restrict__ flags,      // SYMMETRIC [world]
    unsigned int* __restrict__ count,      // LOCAL [1]
    int* __restrict__ pperr,
    long long spin_limit,
    int cur, int world) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    const unsigned int epoch = count[0] + 1;
    count[0] = epoch;
    __threadfence_system();                                  // release
    for (int pe = 0; pe < world; ++pe) {
      if (pe == cur) {
        __hip_atomic_fetch_add(&flags[cur], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      } else {
        unsigned int* peer = reinterpret_cast<unsigned int*>(
            mori::shmem::ShmemPtrP2p((uint64_t)(flags + cur), pe));
        __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
    }
  }
  __syncthreads();
  if (tid < world) {
    const unsigned int epoch = count[0];
    volatile const unsigned int* vf = flags;
    long long spins = 0;
    while (vf[tid] < epoch) {
      if (++spins > spin_limit) { atomicOr(pperr, 1); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence_system();                                    // acquire
}
