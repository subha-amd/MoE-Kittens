// ================================================================================================
// hkp_alloc.hpp — remote receive-row reservation and the wave destination dedup that feeds
// it (see k0d_mega.hip, k0pf3_qpush.hip,
// k0d_push.hip, k0pf2_push — one atomicAdd per deduplicated (token, destination)).
//
// Semantics preserved:
//   * self reservation: relaxed fetch_add at AGENT scope on the LOCAL counter.
//   * peer reservation: relaxed fetch_add at SYSTEM scope through the translated counter.
//     The returned value IS the arrival row on the destination rank; arrival order is
//     nondeterministic, exactly like production's dispTokOffset.
//   * capacity fits the 24-bit row field of the packed sti word (k0d_mega.hip); the
//     overflow bit is a template parameter because it differs per arm (128 vs 65536).
//   * The counter is reset ONLY by the sole consumer at its phase tail, after its barrier
//     proved next-epoch producers are stream-ordered behind it (k0pf3_dsort.hip,
//     263-265).  Everything else in the protocol is monotonic; this is the one reset cell.
//
// DEDUP CONTEXT RULE (k0pf3_qpush.hip): __match_any_sync must execute in convergent,
// warp-uniform context — the calling token loop's trip count must be warp-uniform so the
// collective never sits in a divergent guard.
// ================================================================================================

#ifndef HKP_ALLOC_HPP
#define HKP_ALLOC_HPP

#include <hip/hip_runtime.h>

#include "hkp_topology.hpp"

namespace hkp {

constexpr unsigned int kRowFieldMask = 0x00FFFFFFu;  // 24-bit row capacity (sti packing)

// Overflow action; Bit is the arm's pperr bit (k0d_mega.hip uses 128,
// k0pf3_qpush.hip uses 65536).  The caller must still reject the row.
template <int Bit>
struct overflow_flag {
  int* pperr;
  __device__ __forceinline__ void on_overflow() const { atomicOr(pperr, Bit); }
};

struct peer_slot_allocator {
  unsigned int* counter;   // SYMMETRIC [1]: arrival-row allocator on the OWNING rank
  unsigned int capacity;   // T_loc_max; must be <= kRowFieldMask
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

// Sole-consumer tail reset (k0pf3_dsort.hip).  Plain store; safe only at the point
// where every producer of the next epoch is stream-ordered behind this kernel.
__device__ __forceinline__ void reset_slots(const peer_slot_allocator& alloc) {
  alloc.counter[0] = 0u;
}

// ---------------------------------------------------------------------------
// Wave dedup of destinations (3 sites): a (token, dest) pair is pushed ONCE no matter how
// many of the token's slots land there; the lowest such slot is PRIMARY.
// ---------------------------------------------------------------------------
struct wave_dedup {
  unsigned long long primary_mask;  // ballot of primary lanes
  int fanout;                       // popcount(primary_mask)
  bool primary;                     // is THIS lane the primary for its destination
};

// lane s < TOPK holds slot s with destination dest (dest < 0 for inert lanes).  See the
// CONTEXT RULE in the header comment: call only in warp-uniform control flow.
__device__ __forceinline__ wave_dedup wave_dedup_by_destination(int dest, int lane,
                                                                int topk) {
  const unsigned long long match = __match_any_sync(__activemask(), dest);
  const bool primary = (lane < topk) && ((match & ((1ULL << lane) - 1ULL)) == 0ULL);
  const unsigned long long pmask = __ballot(primary);
  return wave_dedup{pmask, static_cast<int>(__popcll(pmask)), primary};
}

// Fanout iteration order is ascending slot order (k0pf3_qpush.hip): extract the
// lowest primary lane, clear it, repeat.  per-iteration values (dest, row) come from
// __shfl against the returned lane.
__device__ __forceinline__ int next_primary_lane(unsigned long long& pmask) {
  const int src_lane = __ffsll((long long)pmask) - 1;
  pmask &= pmask - 1;
  return src_lane;
}

}  // namespace hkp

#endif  // HKP_ALLOC_HPP
