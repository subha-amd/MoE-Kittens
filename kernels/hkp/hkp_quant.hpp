// ================================================================================================
// hkp_quant.hpp — bf16 -> FP8 e4m3 group quantization and the scale-layout utilities
// (see ; quant math byte-identical to k0pf3_qpush.hip, which is
// byte-identical to k0pf_quant and to n1g's A2 register quant at n2_phase1.cpp).
//
// Numerics that must NOT change:
//   * groups are exactly 128 elements: in the warp tiling, chunk c covers elements
//     [1024c, 1024c+1024) and lanes 8g..8g+7 cover group 8c+g — the amax reduction is an
//     8-lane __shfl_xor tree over masks 1/2/4 and never leaves the segment.
//   * scale = max(amax, 1e-6) / 448.  THE EPSILON IS NOT OPTIONAL: empty experts give
//     amax = 0 and 448/0 = NaN (n2_phase1.cpp).
//   * quantize: clamp x/scale to +-448, __HIP_SATFINITE, __HIP_E4M3.
//   * combine-side bf16: unpack = top 16 bits of fp32 (exact); pack = RNE
//     (u + 0x7FFF + lsb) >> 16, matching v_cvt_pk_bf16_f32.
// ================================================================================================

#ifndef HKP_QUANT_HPP
#define HKP_QUANT_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

__device__ __forceinline__ float bf16_unpack(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

__device__ __forceinline__ unsigned short bf16_pack_rne(float f) {
  const unsigned int u = __float_as_uint(f);
  const unsigned int lsb = (u >> 16) & 1u;
  return (unsigned short)((u + 0x7FFFu + lsb) >> 16);
}

template <int GroupSize = 128>
struct fp8_e4m3_group_quantizer {
  static_assert(GroupSize == 128, "the tiling is 128-element groups (8 lanes x 16)");
  static constexpr float kFP8Max = 448.0f;

  static __device__ __forceinline__ float dequant_scale(float amax) {
    return fmaxf(amax, 1.0e-6f) / kFP8Max;
  }

  static __device__ __forceinline__ unsigned char quantize_scalar(float x, float scale) {
    const float q = fminf(fmaxf(x / scale, -kFP8Max), kFP8Max);
    return __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
  }

  // One 1024-element chunk: h16 points at THIS lane's 16 contiguous bf16 elements
  // ([1024c + 16*lane, +16) of the row).  Returns the lane's 16 FP8 bytes; scale_out is
  // the group scale, kept by the caller only on owner lanes (lane & 7) == 0 — the same
  // distribution k0pf3_qpush.hip uses for sc_arr.
  static __device__ __forceinline__ uint4 quantize_chunk16(const unsigned short* h16,
                                                           int lane, float& scale_out) {
    float f[16];
    float a = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      f[e] = bf16_unpack(h16[e]);
      a = fmaxf(a, fabsf(f[e]));
    }
#pragma unroll
    for (int o = 1; o <= 4; o <<= 1) a = fmaxf(a, __shfl_xor(a, o, 64));
    const float scale = dequant_scale(a);
    scale_out = scale;
    union {
      unsigned char b[16];
      uint4 v;
    } qo;
#pragma unroll
    for (int e = 0; e < 16; ++e) qo.b[e] = quantize_scalar(f[e], scale);
    return qo.v;
  }
};

// Prefill scale-store idiom (k0pf3_qpush.hip): owner lanes (lane & 7) == 0 write
// the 56-group row directly as scalars, dst[c*8 + lane>>3] = sc_arr[c] — no cross-lane
// redistribution.  The decode idiom (14 lanes x 16 B vectorized copy) is
// post_peer_span<16>(..., 14) in hkp_store.hpp.
template <int Chunks = 7>
__device__ __forceinline__ void scatter_group_scales(float* dst, const float* sc_arr,
                                                     int lane) {
  if ((lane & 7) != 0) return;
#pragma unroll
  for (int c = 0; c < Chunks; ++c) dst[c * 8 + (lane >> 3)] = sc_arr[c];
}

// ---------------------------------------------------------------------------
// Scale layouts.  group_major is the production live-prefix layout
// input_scale[k128 * T + token] — never the [capacity, 56] capacity stride
// (n2_phase1.cpp).  row_major is the symmetric staging layout sc_stage[t * NG + g]
// written by the push arms and transposed dest-side.
// ---------------------------------------------------------------------------
enum class scale_layout { row_major, group_major, expert_nk };

__device__ __forceinline__ std::size_t scale_index(scale_layout L, int g, int t, int NG,
                                                   int T) {
  return (L == scale_layout::group_major) ? ((std::size_t)g * T + t)
                                          : ((std::size_t)t * NG + g);
}

template <scale_layout Src, scale_layout Dst>
__device__ __forceinline__ void transform_scale_element(const float* src, float* dst,
                                                        int g, int t, int NG, int T) {
  static_assert((Src == scale_layout::row_major && Dst == scale_layout::group_major) ||
                    (Src == scale_layout::group_major && Dst == scale_layout::row_major),
                "expert_nk is addressed via expert_nk_offset, not transformed");
  dst[scale_index(Dst, g, t, NG, T)] = src[scale_index(Src, g, t, NG, T)];
}

// [expert, n128, k128] weight-scale addressing: fc1_scale [32,32,56] and fc2_scale
// [32,56,16] (n2_phase1.cpp, n2_phase2.cpp).
__device__ __forceinline__ std::size_t expert_nk_offset(int e, int ng, int k, int n_groups,
                                                        int k_groups) {
  return ((std::size_t)e * n_groups + ng) * k_groups + k;
}

// ---------------------------------------------------------------------------
// Fused zero-part + scale-transpose (3 sites: k0d_mega.hip, k0pf3_dsort.hip:
// 189-203, k0d_dsort.hip): one warp zeros Chunks1K x 1 KiB bf16 chunks of part row t while
// lanes < ng transpose that row's scales into group-major sc_dst.  The zero prefix is
// EXACTLY [0, T_loc) — the phase-2 atomic epilogue requires it and the full-capacity host
// memset is ~1.9x the store traffic (k0pf3_dsort.hip).  Per-row body; callers keep
// their grid-stride row loop.
// ---------------------------------------------------------------------------
template <int Chunks1K = 14>
__device__ __forceinline__ void zero_part_scale_transpose(unsigned short* part_row,
                                                          float* sc_dst,
                                                          const float* sc_stage_row, int t,
                                                          int T_loc, int ng, int lane) {
  const uint4 z = make_uint4(0u, 0u, 0u, 0u);
  unsigned char* prow = reinterpret_cast<unsigned char*>(part_row);
#pragma unroll
  for (int c = 0; c < Chunks1K; ++c) {
    *reinterpret_cast<uint4*>(prow + (c << 10) + (lane << 4)) = z;
  }
  if (lane < ng) sc_dst[(std::size_t)lane * (std::size_t)T_loc + t] = sc_stage_row[lane];
}

}  // namespace hkp

#endif  // HKP_QUANT_HPP
