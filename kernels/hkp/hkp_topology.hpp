// Peer address translation for symmetric heaps:
//
//     remote_ptr = bases[dst] + (local_ptr - local_base)     (dst != rank)
//     remote_ptr = local_ptr                                 (dst == rank)
//
// MoRI uses p2pPeerPtrs with heapBaseAddr; IRIS uses heap_bases with heap_bases[rank]. The self
// shortcut is required because MoRI's self entry is not usable. Build this POD from the initialized
// module and pass it by value; separately loaded HIP modules have independent device globals.

#ifndef HKP_TOPOLOGY_HPP
#define HKP_TOPOLOGY_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

constexpr int kMaxRanks = 8;

struct peer_topology {
  std::uint64_t bases[kMaxRanks];  // Per-rank heap bases mapped on this device.
  std::uint64_t local_base;        // This rank's heap base.
  int rank;
  int world;

  __device__ __forceinline__ std::uint64_t translate(int dst,
                                                     std::uint64_t local_ptr) const {
    if (dst == rank) return local_ptr;
    return bases[dst] + (local_ptr - local_base);
  }

  template <typename T>
  __device__ __forceinline__ T* translate(int dst, T* local_ptr) const {
    return reinterpret_cast<T*>(
        translate(dst, reinterpret_cast<std::uint64_t>(local_ptr)));
  }

  template <typename T>
  __device__ __forceinline__ const T* translate(int dst, const T* local_ptr) const {
    return reinterpret_cast<const T*>(
        translate(dst, reinterpret_cast<std::uint64_t>(local_ptr)));
  }
};

// MoRI-style peer table.
__host__ __device__ __forceinline__ peer_topology peer_topology_from_peer_table(
    const std::uint64_t* p2p_peer_ptrs, std::uint64_t heap_base, int rank, int world) {
  peer_topology t;
  for (int r = 0; r < kMaxRanks; ++r) t.bases[r] = (r < world) ? p2p_peer_ptrs[r] : 0u;
  t.local_base = heap_base;
  t.rank = rank;
  t.world = world;
  return t;
}

// IRIS-style heap_bases table.
__host__ __device__ __forceinline__ peer_topology peer_topology_from_heap_bases(
    const std::uint64_t* heap_bases, int rank, int world) {
  peer_topology t;
  for (int r = 0; r < kMaxRanks; ++r) t.bases[r] = (r < world) ? heap_bases[r] : 0u;
  t.local_base = heap_bases[rank];
  t.rank = rank;
  t.world = world;
  return t;
}

// Typed view of one symmetric object.
template <typename T>
struct peer_span {
  T* ptr;
  std::size_t count;

  __device__ __forceinline__ T* on(const peer_topology& topo, int dst) const {
    return topo.translate(dst, ptr);
  }
  __device__ __forceinline__ T& at(const peer_topology& topo, int dst,
                                   std::size_t i) const {
    return on(topo, dst)[i];
  }
};

template <typename T>
__host__ __device__ __forceinline__ peer_span<T> make_peer_span(T* ptr,
                                                                std::size_t count) {
  return peer_span<T>{ptr, count};
}

}  // namespace hkp

#endif  // HKP_TOPOLOGY_HPP
