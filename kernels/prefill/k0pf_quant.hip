// Prefill BF16-to-FP8 E4M3 quantization with row-major FP32 scales.
// The fast path uses one 896-thread block per 7168-element token. Each 16-thread group reduces
// one 128-element group with scale=max(amax,1e-6)/448. Other widths use k0pf_quant_generic.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

__device__ __forceinline__ float k0pfq_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Fast path for K=7168.
extern "C" __global__ void __launch_bounds__(896, 2) k0pf_quant_to_shmem(
    const unsigned short* __restrict__ hidden,  // [T, H] bf16 (H == K == 7168)
    unsigned char* __restrict__ a_src,          // [T, K] fp8 e4m3, SYMMETRIC staging
    float* __restrict__ sc_src,                 // [T, 56] fp32 row-major, SYMMETRIC
    int T, int K) {
  // Other configurations must use k0pf_quant_generic.
  if (K != 7168 || blockDim.x != 896) return;
  const int tid = threadIdx.x;
  const int g = tid >> 4;
  const int l16 = tid & 15;
  const size_t base = (size_t)tid * 8;

  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const uint4 raw = *reinterpret_cast<const uint4*>(hidden + (size_t)t * 7168 + base);
    const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw);
    float f[8];
    float a = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      f[e] = k0pfq_bf16_to_f32(h16[e]);
      a = fmaxf(a, fabsf(f[e]));
    }
    // Reduce amax over 128 consecutive elements.
#pragma unroll
    for (int o = 1; o <= 8; o <<= 1) a = fmaxf(a, __shfl_xor(a, o, 64));
    const float scale = fmaxf(a, 1.0e-6f) / 448.0f;
    if (l16 == 0) sc_src[(size_t)t * 56 + g] = scale;

    unsigned char out[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const float q = fminf(fmaxf(f[e] / scale, -448.0f), 448.0f);
      out[e] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
    }
    *reinterpret_cast<uint2*>(a_src + (size_t)t * 7168 + base) =
        *reinterpret_cast<const uint2*>(out);
  }
}

// Generic path: one element per thread.
extern "C" __global__ void k0pf_quant_generic(
    const unsigned short* __restrict__ hidden,
    unsigned char* __restrict__ a_src,
    float* __restrict__ sc_src,
    int T, int K) {
  const int NG = K / 128;
  const int b = blockIdx.x;
  const int t = b / NG;
  const int g = b % NG;
  if (t >= T) return;
  const int tid = threadIdx.x;
  const size_t k = (size_t)g * 128 + tid;

  const float f = k0pfq_bf16_to_f32(hidden[(size_t)t * K + k]);
  float a = fabsf(f);
  __shared__ float wmax[2];
  for (int o = 32; o >= 1; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));
  if ((tid & 63) == 0) wmax[tid >> 6] = a;
  __syncthreads();
  const float amax = fmaxf(wmax[0], wmax[1]);
  const float scale = fmaxf(amax, 1.0e-6f) / 448.0f;

  if (tid == 0) sc_src[(size_t)t * NG + g] = scale;
  const float q = fminf(fmaxf(f / scale, -448.0f), 448.0f);
  a_src[(size_t)t * K + k] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
}
