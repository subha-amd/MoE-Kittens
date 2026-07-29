#pragma once

#include <cstdint>

// Keep this header usable by an ordinary host C++ compiler.  HIP compilation
// adds the device annotation without requiring a HIP runtime include here.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define PRODUCTION_FUSED_MOE_HOST_DEVICE __host__ __device__
#else
#define PRODUCTION_FUSED_MOE_HOST_DEVICE
#endif

namespace production_fused_moe::aiter_gfx950 {

using offset_t = std::uint64_t;

inline constexpr offset_t kNInner = 16;
inline constexpr offset_t kKInner = 16;
inline constexpr offset_t kKBlock = 32;
inline constexpr offset_t kKHalvesPerBlock = kKBlock / kKInner;
inline constexpr offset_t kScaleBlockN = 128;
inline constexpr offset_t kScaleBlockK = 128;
inline constexpr offset_t kFp32Bytes = 4;

static_assert(kKHalvesPerBlock == 2);
static_assert(sizeof(float) == kFp32Bytes);

// AITER's gfx950 FP8 weight shuffle starts with logical [E, N, K], views it as
//
//   [E, N/16, 16, K/32, 2, 16]
//
// and stores permute(0, 1, 3, 4, 2, 5) contiguously.  Therefore the physical
// dimension order is [E, N/16, K/32, 2, 16(N), 16(K)].  The function below
// returns a BYTE offset because every production FP8 element occupies one
// byte.  It is deliberately unchecked for use in a GEMM inner loop; callers
// must provide N % 16 == 0, K % 32 == 0, and an in-bounds (e, n, k).
PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr offset_t
fp8_weight_byte_offset(offset_t e,
                       offset_t n,
                       offset_t k,
                       offset_t logical_n,
                       offset_t logical_k) {
  const offset_t n_outer = n / kNInner;
  const offset_t n_inner = n % kNInner;
  const offset_t k_outer = k / kKBlock;
  const offset_t k_remainder = k % kKBlock;
  const offset_t k_half = k_remainder / kKInner;
  const offset_t k_inner = k_remainder % kKInner;

  const offset_t expert_base = e * logical_n * logical_k;
  const offset_t within_expert =
      (((n_outer * (logical_k / kKBlock) + k_outer) *
             kKHalvesPerBlock +
         k_half) *
            kNInner +
        n_inner) *
           kKInner +
       k_inner;
  return expert_base + within_expert;
}

// Production FP32 block scales are not passed through AITER's weight shuffle.
// They remain ordinary contiguous storage with logical shape
// [E, N/128, K/128].  A checkpoint may expose the first two dimensions
// flattened as [E*N/128, K/128]; its linear address is identical.
// Preconditions: N % 128 == 0, K % 128 == 0, and (e, n, k) is in bounds.
PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr offset_t
fp32_blockscale_element_offset(offset_t e,
                               offset_t n,
                               offset_t k,
                               offset_t logical_n,
                               offset_t logical_k) {
  const offset_t n_blocks = logical_n / kScaleBlockN;
  const offset_t k_blocks = logical_k / kScaleBlockK;
  return (e * n_blocks + n / kScaleBlockN) * k_blocks +
         k / kScaleBlockK;
}

PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr offset_t
fp32_blockscale_byte_offset(offset_t e,
                            offset_t n,
                            offset_t k,
                            offset_t logical_n,
                            offset_t logical_k) {
  return kFp32Bytes *
         fp32_blockscale_element_offset(e, n, k, logical_n, logical_k);
}

PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr bool
is_supported_weight_shape(offset_t experts,
                          offset_t logical_n,
                          offset_t logical_k) {
  return experts != 0 && logical_n != 0 && logical_k != 0 &&
         logical_n % kNInner == 0 && logical_k % kKBlock == 0;
}

PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr bool
is_supported_blockscale_shape(offset_t experts,
                              offset_t logical_n,
                              offset_t logical_k) {
  return is_supported_weight_shape(experts, logical_n, logical_k) &&
         logical_n % kScaleBlockN == 0 &&
         logical_k % kScaleBlockK == 0;
}

PRODUCTION_FUSED_MOE_HOST_DEVICE inline constexpr bool
is_valid_coordinate(offset_t e,
                    offset_t n,
                    offset_t k,
                    offset_t experts,
                    offset_t logical_n,
                    offset_t logical_k) {
  return e < experts && n < logical_n && k < logical_k;
}

}  // namespace production_fused_moe::aiter_gfx950

#undef PRODUCTION_FUSED_MOE_HOST_DEVICE
