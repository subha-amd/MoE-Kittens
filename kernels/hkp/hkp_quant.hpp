// BF16-to-FP8 E4M3 group quantization and scale-layout utilities.
// Groups contain 128 elements and use an 8-lane amax reduction. Scale is
// max(amax,1e-6)/448; the epsilon prevents NaN for empty experts. Conversion clamps to +/-448.
// BF16 packing uses round-to-nearest-even.

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

  // Quantize this lane's 16 elements; lanes with lane%8==0 retain the group scale.
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

// Scale-owner lanes write directly without cross-lane redistribution.
template <int Chunks = 7>
__device__ __forceinline__ void scatter_group_scales(float* dst, const float* sc_arr,
                                                     int lane) {
  if ((lane & 7) != 0) return;
#pragma unroll
  for (int c = 0; c < Chunks; ++c) dst[c * 8 + (lane >> 3)] = sc_arr[c];
}

// group_major uses input_scale[k128*T+token]; row_major uses sc_stage[t*NG+g].
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

// [expert, n128, k128] weight-scale addressing.
__device__ __forceinline__ std::size_t expert_nk_offset(int e, int ng, int k, int n_groups,
                                                        int k_groups) {
  return ((std::size_t)e * n_groups + ng) * k_groups + k;
}

// Clear one live partial row while transposing its scales to group-major order.
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
