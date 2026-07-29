// ================================================================================================
// hkp_sync.hpp — monotonic-epoch doorbells and the local grid barrier.
//
// Protocol invariants (do not weaken when refactoring):
//   * Flags and counters are zeroed ONCE at setup and NEVER reset (k0d_mega.hip).
//     Epochs are derived from cumulative state, so graph replay advances the protocol with
//     no host involvement.
//   * Publication pokes are RELAXED: self = __HIP_MEMORY_SCOPE_AGENT, peer =
//     __HIP_MEMORY_SCOPE_SYSTEM through a translated pointer.  Ordering comes from a
//     __threadfence_system() executed BEFORE the pokes, never from the pokes themselves.
//   * The rendezvous arrival RMW is __ATOMIC_ACQ_REL at AGENT scope.
//   * The consumer wait is a VOLATILE poll with __builtin_amdgcn_s_sleep(4) backoff and a
//     spin_limit; on timeout it flags an error bit and proceeds (fail-closed: the gate
//     downstream reads pperr, the error word).  The wait is sealed with
//     __syncthreads() + __threadfence_system() (acquire).
//
// The three doorbell implementations in the tree diverge on TWO axes, which is why the
// primitives below are policy-parameterized:
//   * epoch derivation: sort_count[0] + 1 (k0d_mega.hip, k0pf3_dsort.hip) vs
//     old / gridDim.x + 1 from the cumulative arrival counter (k0d_combine.hip — requires
//     a FIXED gridDim.x across every capture and replay, k0d_combine.hip) vs the
//     lagging cell value quant_count[0] (k0d_fused.hip, waits on epoch N-1).
//   * release-fence placement: every CTA fences before its arrival because each CTA
//     publishes its OWN payload stores (k0d_mega.hip, k0pf3_qpush.hip) vs only the
//     LAST CTA fences before the poke because the published data landed by stream order
//     (k0d_combine.hip; the combine tail at k0d_combine.hip fences to order READS —
//     the pulls — before the done poke, not stores) vs the single-CTA arm which fences and
//     pokes directly with a plain-store epoch bump (k0d_fused.hip).
// Error bits differ per arm, so the bit is a RUNTIME argument, not a template parameter.
// ================================================================================================

#ifndef HKP_SYNC_HPP
#define HKP_SYNC_HPP

#include <hip/hip_runtime.h>

#include <cstdint>

#include "hkp_topology.hpp"

namespace hkp {

// Release-fence placement at publication (see header comment).
enum class release_site {
  fence_per_cta_before_arrival,  // producers publishing their own stores (default, safest)
  fence_on_last_before_poke,     // barriers whose payload landed by stream order
};

// Fail-closed timeout action shared by every poll: flag the caller's error bit and
// proceed; the downstream correctness gate reads pperr.
struct fail_closed {
  int* pperr;
  int bit;
  __device__ __forceinline__ void on_timeout() const { atomicOr(pperr, bit); }
};

// ---------------------------------------------------------------------------
// Epoch derivation — three styles.  All are monotonic; none resets.
// ---------------------------------------------------------------------------

// k0d_mega.hip / k0pf3_dsort.hip — dedicated LOCAL epoch cell, bumped by the
// consumer at the phase tail; the wait target is one past the last completed epoch.
__device__ __forceinline__ unsigned int epoch_from_cell(const unsigned int* epoch_cell) {
  return epoch_cell[0] + 1u;
}

// k0d_combine.hip — derived from the cumulative CTA arrival counter.  gridDim.x is part
// of the protocol: it must be identical across every capture and replay.
__device__ __forceinline__ unsigned int epoch_from_cumulative(unsigned int old,
                                                              unsigned int blocks) {
  return old / blocks + 1u;
}

// k0d_fused.hip — lagging target: wait for the PREVIOUS epoch's completion signal
// while this epoch's cell has not yet been bumped.
__device__ __forceinline__ unsigned int epoch_current(const unsigned int* epoch_cell) {
  return epoch_cell[0];
}

// Sole-writer plain-store advance (k0pf3_dsort.hip, k0d_fused.hip).  The next
// kernel on this stream reads the cell; the writer's phase barrier already proved no
// earlier-epoch consumer is still reading.
__device__ __forceinline__ void advance_epoch_cell(unsigned int* epoch_cell,
                                                   unsigned int epoch) {
  epoch_cell[0] = epoch;
}

// ---------------------------------------------------------------------------
// Bounded volatile poll — the one wait body shared by every polling loop here.
// Returns false on timeout (after running the policy); callers proceed regardless.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Doorbell channel state.  flags is SYMMETRIC [world]: flags[src] on rank R counts R's
// completed epoch publications FROM src, forever.  arrive_count is the LOCAL cumulative
// CTA rendezvous counter; epoch_cell is the LOCAL dedicated epoch cell where used.
// ---------------------------------------------------------------------------
struct peer_epoch {
  unsigned int* flags;
  unsigned int* arrive_count;
  unsigned int* epoch_cell;  // nullptr for cumulative-derived arms
  int* pperr;
  long long spin_limit;
};

// Serial fanout — the exact publication loop in every current kernel (one thread pokes
// recv_doorbell[cur] on every rank).  Call from a single thread.
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

// Lane-parallel fanout, roughly 9x faster than poking peers serially: lane d pokes
// rank d.  OPT-IN fast path; the serial form above remains the behavior-preserving
// default.  Must be called by all lanes of the publishing warp.
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

// Rendezvous publication.  Call AFTER __syncthreads(), from tid == 0.  Returns true when
// this CTA was the last arriver (i.e. it issued the peer pokes).  Semantics per RS match
// the kernels here bit-for-bit: fence-per-CTA publishes each CTA's own payload stores
// before arrival; fence-on-last publishes data that landed by stream order.
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

// Warp-collective variant for the lane-parallel fanout.  Call AFTER __syncthreads(), from
// every lane of the first warp.  In the fence-on-last case EVERY lane fences: each fanout
// lane's relaxed poke must be ordered after the release, and one lane's fence cannot order
// another lane's poke.
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

// Consumer wait — FULL-CTA collective: lanes tid < world poll one flag each, then every
// thread seals with __syncthreads + __threadfence_system (the acquire that orders all
// subsequent reads of the published payloads, k0pf3_dsort.hip).
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

// ---------------------------------------------------------------------------
// local_grid_epoch — the AGENT-scope monotonic grid barrier (k0d_mega.hip,
// k0pf3_dsort.hip).  Distinct from a doorbell: everything it publishes is on THIS
// GPU, so both fences are device-scope __threadfence() and the counter is agent-scope.
// The target (old / grid + 1) * grid makes the barrier phase-indexed by the cumulative
// count — stragglers from an earlier phase cannot satisfy a later one, with no reset.
// ---------------------------------------------------------------------------
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
    __threadfence();  // device release
    const unsigned int old = __hip_atomic_fetch_add(bar.counter, 1u, __ATOMIC_ACQ_REL,
                                                    __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    bounded_wait(reinterpret_cast<volatile const unsigned int*>(bar.counter), target,
                 bar.spin_limit, on_timeout);
  }
  __syncthreads();
  __threadfence();  // device acquire
}

}  // namespace hkp

#endif  // HKP_SYNC_HPP
