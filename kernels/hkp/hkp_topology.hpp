// ================================================================================================
// hkp_topology.hpp — peer address translation for the symmetric-heap MoE transport.
//
// ONE translation rule unifies the two source models in the tree:
//
//     remote_ptr = bases[dst] + (local_ptr - local_base)     (dst != rank)
//     remote_ptr = local_ptr                                 (dst == rank)
//
//   (a) MoRI device-global model (baseline/mori/include/mori/shmem/shmem_device_api.hpp,
//       ShmemPtrP2p): bases[] is SymmMemObj::p2pPeerPtrs[]
//       and local_base is GpuStates::heapBaseAddr.  Symmetric allocation guarantees every
//       rank's object sits at the same heap offset, so the offset is rank-invariant.
//   (b) IRIS heap_bases model: bases[] is the device heap_bases[] table and
//       local_base = heap_bases[rank].
//
// The dst == rank shortcut is NOT redundant under model (a): p2pPeerPtrs[rank] is not a
// usable self mapping there, so self must never go through the table.
//
// HOST SETUP (minimal by design): build the POD ONCE at setup from the runtime's tables and
// pass it BY VALUE as a kernel argument.  Do not re-derive it from
// mori::shmem::GetGlobalGpuStatesPtr() inside a kernel: every separately-loaded HIP module
// carries its OWN device globalGpuStates (heapBaseAddr / p2pPeerPtrs), and ShmemPtrP2p reads
// the CALLING module's state (an uninitialized
// second module returns 0 and faults at addr 0).  A by-value POD captured from the primary
// module is the only module-independent form, and it is graph-capture safe (no host reads).
// ================================================================================================

#ifndef HKP_TOPOLOGY_HPP
#define HKP_TOPOLOGY_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

constexpr int kMaxRanks = 8;  // the protocols here cap world at 8

struct peer_topology {
  std::uint64_t bases[kMaxRanks];  // per-rank heap base as mapped for THIS device
  std::uint64_t local_base;        // translation origin (this rank's heap base)
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

// Model (a): MoRI-style per-peer pointer table + this rank's heap base.
__host__ __device__ __forceinline__ peer_topology peer_topology_from_peer_table(
    const std::uint64_t* p2p_peer_ptrs, std::uint64_t heap_base, int rank, int world) {
  peer_topology t;
  for (int r = 0; r < kMaxRanks; ++r) t.bases[r] = (r < world) ? p2p_peer_ptrs[r] : 0u;
  t.local_base = heap_base;
  t.rank = rank;
  t.world = world;
  return t;
}

// Model (b): IRIS-style heap_bases table (heap_bases[r] = rank r's base, self included).
__host__ __device__ __forceinline__ peer_topology peer_topology_from_heap_bases(
    const std::uint64_t* heap_bases, int rank, int world) {
  peer_topology t;
  for (int r = 0; r < kMaxRanks; ++r) t.bases[r] = (r < world) ? heap_bases[r] : 0u;
  t.local_base = heap_bases[rank];
  t.rank = rank;
  t.world = world;
  return t;
}

// Typed view of one symmetric object.  `ptr` is the LOCAL address of the object on this
// rank; on()/at() hand out the per-rank translated address.
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
