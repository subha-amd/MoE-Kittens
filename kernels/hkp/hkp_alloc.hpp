// Remote receive-row reservation and wave destination deduplication.
// Local reservations use agent-scope relaxed atomics; peer reservations use system scope.
// Arrival order is nondeterministic and row indices fit the packed 24-bit field. Only the consumer
// resets the allocator after excluding next-epoch producers. __match_any_sync requires convergent,
// warp-uniform control flow.

#ifndef HKP_ALLOC_HPP
#define HKP_ALLOC_HPP

#include <hip/hip_runtime.h>

#include "hkp_topology.hpp"

namespace hkp {

constexpr unsigned int kRowFieldMask = 0x00FFFFFFu;

// The caller must reject an overflowing row after setting its error bit.
template <int Bit>
struct overflow_flag {
  int* pperr;
  __device__ __forceinline__ void on_overflow() const { atomicOr(pperr, Bit); }
};

struct peer_slot_allocator {
  unsigned int* counter;   // SYMMETRIC [1] on the owning rank.
  unsigned int capacity;   // Must be <= kRowFieldMask.
};

template <typename OverflowPolicy>
__device__ __forceinline__ unsigned int reserve_local_slot(
    const peer_slot_allocator& alloc, const OverflowPolicy& on_overflow) {
  const unsigned int row = __hip_atomic_fetch_add(alloc.counter, 1u, __ATOMIC_RELAXED,
                                                  __HIP_MEMORY_SCOPE_AGENT);
  if (row >= alloc.capacity) on_overflow.on_overflow();
  return row;
}

template <typename OverflowPolicy>
__device__ __forceinline__ unsigned int reserve_peer_slot(
    const peer_topology& topo, int dst, const peer_slot_allocator& alloc,
    const OverflowPolicy& on_overflow) {
  unsigned int* peer = topo.translate(dst, alloc.counter);
  const unsigned int row =
      __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  if (row >= alloc.capacity) on_overflow.on_overflow();
  return row;
}

// Reset only after every next-epoch producer is ordered behind this kernel.
__device__ __forceinline__ void reset_slots(const peer_slot_allocator& alloc) {
  alloc.counter[0] = 0u;
}

// Deduplicate destinations; the lowest matching slot is primary.
struct wave_dedup {
  unsigned long long primary_mask;
  int fanout;
  bool primary;
};

// Lanes below topk own slots; inert lanes pass a negative destination.
__device__ __forceinline__ wave_dedup wave_dedup_by_destination(int dest, int lane,
                                                                int topk) {
  const unsigned long long match = __match_any_sync(__activemask(), dest);
  const bool primary = (lane < topk) && ((match & ((1ULL << lane) - 1ULL)) == 0ULL);
  const unsigned long long pmask = __ballot(primary);
  return wave_dedup{pmask, static_cast<int>(__popcll(pmask)), primary};
}

// Extract primary lanes in ascending slot order.
__device__ __forceinline__ int next_primary_lane(unsigned long long& pmask) {
  const int src_lane = __ffsll((long long)pmask) - 1;
  pmask &= pmask - 1;
  return src_lane;
}

}  // namespace hkp

#endif  // HKP_ALLOC_HPP
