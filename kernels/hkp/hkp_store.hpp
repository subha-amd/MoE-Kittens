// ================================================================================================
// hkp_store.hpp — vectorized posted peer stores and peer pulls (see 
// k0d_mega.hip, k0pf3_qpush.hip, k0d_push.hip, k0pf2_push — FP8 row =
// 7 x 16 B/lane, scale row = 14 lanes x 16 B, ids/weights = 2 x uint4/float4; combine pull
// loads: k0d_combine.hip, k0pf_combine.hip).
//
// Every store today is POSTED (fire-and-forget) with NO cache hints — that is the class
// the fastest class measured on this fabric, and the class
// that is graph-replay safe (k0pf3_qpush.hip).  cache_policy therefore DEFAULTS to
// posted_default = today's plain stores, so a refactor onto these wrappers is
// behavior-preserving.  streaming / non_temporal exist only behind -DHKP_ENABLE_CACHE_HINTS
// and are EXPERIMENTAL: unmeasured on this fabric, and any hint that weakens the posted
// store's visibility ordering relative to the doorbell's release fence invalidates the
// rendezvous proof — re-run the poisoned two-sided gate before adopting one.
// ================================================================================================

#ifndef HKP_STORE_HPP
#define HKP_STORE_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

enum class cache_policy { posted_default, streaming, non_temporal };

namespace detail {

template <cache_policy CP>
__device__ __forceinline__ void store_16(void* addr, const uint4& v) {
#ifdef HKP_ENABLE_CACHE_HINTS
  if (CP != cache_policy::posted_default) {
    const unsigned int w[4] = {v.x, v.y, v.z, v.w};
    unsigned int* a = reinterpret_cast<unsigned int*>(addr);
#pragma unroll
    for (int i = 0; i < 4; ++i) __builtin_nontemporal_store(w[i], a + i);
    return;
  }
#endif
  *reinterpret_cast<uint4*>(addr) = v;
}

template <cache_policy CP>
__device__ __forceinline__ uint4 load_16(const void* addr) {
#ifdef HKP_ENABLE_CACHE_HINTS
  if (CP != cache_policy::posted_default) {
    const unsigned int* a = reinterpret_cast<const unsigned int*>(addr);
    uint4 v;
    v.x = __builtin_nontemporal_load(a + 0);
    v.y = __builtin_nontemporal_load(a + 1);
    v.z = __builtin_nontemporal_load(a + 2);
    v.w = __builtin_nontemporal_load(a + 3);
    return v;
  }
#endif
  return *reinterpret_cast<const uint4*>(addr);
}

}  // namespace detail

// Full-warp posted row: chunk c of lane l lands at dst[c * 1024 + l * 16] — the
// (c << 10) + (lane << 4) geometry.  BytesPerLane is fixed at 16: all 11 sites are uint4.
template <int BytesPerLane, int Chunks,
          cache_policy CP = cache_policy::posted_default>
__device__ __forceinline__ void post_peer_row(char* dst, const uint4* v, int lane) {
  static_assert(BytesPerLane == 16, "posts are 16 B per lane per chunk");
#pragma unroll
  for (int c = 0; c < Chunks; ++c) {
    detail::store_16<CP>(dst + ((std::size_t)c << 10) + ((std::size_t)lane << 4), v[c]);
  }
}

// Guarded single-chunk span: lanes [0, ActiveLanes) each post one 16 B chunk contiguously.
// Covers the 14-lane scale row (k0d_mega.hip) and the 2-lane ids/weights posts
// (k0d_mega.hip).
template <int BytesPerLane, cache_policy CP = cache_policy::posted_default>
__device__ __forceinline__ void post_peer_span(char* dst, const uint4& v, int lane,
                                               int active_lanes) {
  static_assert(BytesPerLane == 16, "posts are 16 B per lane");
  if (lane < active_lanes) {
    detail::store_16<CP>(dst + ((std::size_t)lane << 4), v);
  }
}

// Full-warp pull: chunk c of lane l is read from src[c * 1024 + l * 16] — the combine's
// per-chunk peer load geometry (k0pf_combine.hip).
template <int BytesPerLane, int Chunks,
          cache_policy CP = cache_policy::posted_default>
__device__ __forceinline__ void pull_peer_row(uint4* v, const char* src, int lane) {
  static_assert(BytesPerLane == 16, "pulls are 16 B per lane per chunk");
#pragma unroll
  for (int c = 0; c < Chunks; ++c) {
    v[c] = detail::load_16<CP>(src + ((std::size_t)c << 10) + ((std::size_t)lane << 4));
  }
}

}  // namespace hkp

#endif  // HKP_STORE_HPP
