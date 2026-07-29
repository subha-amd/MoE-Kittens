// Monotonic peer doorbells and local grid barriers.
// Flags and counters are zeroed once. Relaxed signals are ordered by a preceding system fence.
// Waits use bounded volatile polling and acquire with a CTA barrier plus system fence.
// Epoch derivation and release-fence placement are policy choices because callers publish
// different data. Cumulative epochs require a fixed gridDim.x across replays.

#ifndef HKP_SYNC_HPP
#define HKP_SYNC_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

#include "hkp_topology.hpp"

namespace hkp {

// Release-fence placement at publication (see header comment).
enum class release_site {
  fence_per_cta_before_arrival,  // Each CTA publishes its own stores.
  fence_on_last_before_poke,     // Payload was completed by stream order.
};

// Poll timeouts set an error bit and continue to the downstream gate.
struct fail_closed {
  int* pperr;
  int bit;
  __device__ __forceinline__ void on_timeout() const { atomicOr(pperr, bit); }
};

// Dedicated local epoch cell; wait for one past the completed epoch.
__device__ __forceinline__ unsigned int epoch_from_cell(const unsigned int* epoch_cell) {
  return epoch_cell[0] + 1u;
}

// Cumulative CTA arrival counter; blocks must remain fixed across replays.
__device__ __forceinline__ unsigned int epoch_from_cumulative(unsigned int old,
                                                              unsigned int blocks) {
  return old / blocks + 1u;
}

// Lagging target for the previous epoch's completion signal.
__device__ __forceinline__ unsigned int epoch_current(const unsigned int* epoch_cell) {
  return epoch_cell[0];
}

// Sole-writer advance after the phase barrier excludes earlier readers.
__device__ __forceinline__ void advance_epoch_cell(unsigned int* epoch_cell,
                                                   unsigned int epoch) {
  epoch_cell[0] = epoch;
}

// Returns false after invoking the timeout policy; callers continue.
template <typename ErrorPolicy>
__device__ __forceinline__ bool bounded_wait(volatile const unsigned int* flag,
                                             unsigned int target, long long spin_limit,
                                             const ErrorPolicy& on_timeout) {
  long long spins = 0;
  while (*flag < target) {
    if (++spins > spin_limit) {
      on_timeout.on_timeout();
      return false;
    }
    __builtin_amdgcn_s_sleep(4);
  }
  return true;
}

// flags[src] counts publications from src. arrive_count is a local cumulative CTA counter.
struct peer_epoch {
  unsigned int* flags;
  unsigned int* arrive_count;
  unsigned int* epoch_cell;  // nullptr for cumulative-derived arms
  int* pperr;
  long long spin_limit;
};

// Call from one thread to signal every rank.
__device__ __forceinline__ void serial_fanout_signal(const peer_topology& topo,
                                                     unsigned int* flags) {
  const int cur = topo.rank;
  for (int pe = 0; pe < topo.world; ++pe) {
    if (pe == cur) {
      __hip_atomic_fetch_add(&flags[cur], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else {
      unsigned int* peer = topo.translate(pe, flags + cur);
      __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
}

// Lane d signals rank d. All lanes of the publishing warp must call this.
template <int World>
__device__ __forceinline__ void lane_fanout_signal(const peer_topology& topo,
                                                   unsigned int* flags, int lane) {
  static_assert(World <= 64, "one destination rank per lane");
  if (lane >= World) return;
  const int cur = topo.rank;
  if (lane == cur) {
    __hip_atomic_fetch_add(&flags[cur], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    unsigned int* peer = topo.translate(lane, flags + cur);
    __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Call after __syncthreads() from tid 0. Returns true for the CTA that signals peers.
template <release_site RS>
__device__ __forceinline__ bool publish_after_writes(const peer_topology& topo,
                                                     const peer_epoch& ch) {
  if (RS == release_site::fence_per_cta_before_arrival) __threadfence_system();
  const unsigned int old = __hip_atomic_fetch_add(ch.arrive_count, 1u, __ATOMIC_ACQ_REL,
                                                  __HIP_MEMORY_SCOPE_AGENT);
  if ((old % gridDim.x) != (gridDim.x - 1u)) return false;
  if (RS == release_site::fence_on_last_before_poke) __threadfence_system();
  serial_fanout_signal(topo, ch.flags);
  return true;
}

// First-warp variant. Every signaling lane fences because one lane cannot order another's poke.
template <release_site RS, int World>
__device__ __forceinline__ bool publish_after_writes_warp(const peer_topology& topo,
                                                          const peer_epoch& ch, int lane) {
  unsigned int old = 0u;
  if (lane == 0) {
    if (RS == release_site::fence_per_cta_before_arrival) __threadfence_system();
    old = __hip_atomic_fetch_add(ch.arrive_count, 1u, __ATOMIC_ACQ_REL,
                                 __HIP_MEMORY_SCOPE_AGENT);
  }
  old = __shfl(old, 0);
  if ((old % gridDim.x) != (gridDim.x - 1u)) return false;
  if (RS == release_site::fence_on_last_before_poke) __threadfence_system();
  lane_fanout_signal<World>(topo, ch.flags, lane);
  return true;
}

// Full-CTA wait followed by an acquire fence.
template <typename ErrorPolicy>
__device__ __forceinline__ void wait_before_reads(const peer_topology& topo,
                                                  const peer_epoch& ch,
                                                  unsigned int epoch, int tid,
                                                  const ErrorPolicy& on_timeout) {
  if (tid < topo.world) {
    bounded_wait(reinterpret_cast<volatile const unsigned int*>(ch.flags) + tid, epoch,
                 ch.spin_limit, on_timeout);
  }
  __syncthreads();
  __threadfence_system();
}

// Device-scope monotonic grid barrier. The cumulative target separates barrier phases.
struct local_grid_epoch {
  unsigned int* counter;  // LOCAL [1], cumulative
  int* pperr;
  long long spin_limit;
};

template <typename ErrorPolicy>
__device__ __forceinline__ void grid_barrier(const local_grid_epoch& bar, int tid,
                                             const ErrorPolicy& on_timeout) {
  __syncthreads();
  if (tid == 0) {
    __threadfence();  // Device release.
    const unsigned int old = __hip_atomic_fetch_add(bar.counter, 1u, __ATOMIC_ACQ_REL,
                                                    __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    bounded_wait(reinterpret_cast<volatile const unsigned int*>(bar.counter), target,
                 bar.spin_limit, on_timeout);
  }
  __syncthreads();
  __threadfence();  // Device acquire.
}

}  // namespace hkp

#endif  // HKP_SYNC_HPP
