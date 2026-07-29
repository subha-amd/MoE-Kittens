// ================================================================================================
// k0pf_quant.hip — prefill origin quantization: BF16 hidden into FP8 E4M3 plus an FP32 group
// scale, one pass, straight into the symmetric staging buffers the dispatch reads from.
//
// The variable here is work granularity.  A one-element-per-thread version launches T x 56 blocks
// of 128 threads — hundreds of thousands of tiny blocks at prefill, each paying a 64-lane shuffle
// reduce, an LDS combine and two __syncthreads for 128 B of payload.  This kernel uses one block
// per token (896 threads); each thread owns 8 contiguous elements as a single 16 B vector load,
// the per-(token, K128) amax is a 16-lane shuffle reduce with no LDS and no __syncthreads, and
// the FP8 row leaves as 8 B stores.  That is T real blocks instead of T * 56 hollow ones.
//
// NUMERICS: identical to the element-per-thread version.
//   scale = max(amax, 1e-6) / 448 ;  q = cvt_fp8(clamp(x / scale, +-448), SATFINITE, E4M3)
// amax is taken over exactly the same 128-element (token, K128) groups: thread t covers elements
// [8t, 8t+8), and the 16 threads of group g = t/16 cover [128g, 128g+128).  Identical groups,
// identical max tree (max is order-insensitive), identical scale.  sc_src is written row-major as
// sc_src[t * NG + g]; the dispatch's group-major scatter consumes it unchanged.
//
// LAUNCH CONTRACT:
//   fast path:  fn.launch((T,), (896,), 0, stream, hidden, a_src, sc_src, T, K)
//     K must be 7168.  Grid-stride over tokens, so any grid >= 1 is correct; T blocks is the
//     recommended launch.
//   generic fallback (any K % 128 == 0): k0pf_quant_generic,
//     fn.launch((T * NG,), (128,), 0, stream, hidden, a_src, sc_src, T, K)
//
// ISA expectations for the fast path: zero spills, no s_barrier, no LDS, one 16 B load and one
// 8 B store per thread in the token body.
// ================================================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

__device__ __forceinline__ float k0pfq_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

// Fast path: block per token, 896 threads x 8 elements, K == 7168 (NG == 56).
extern "C" __global__ void __launch_bounds__(896, 2) k0pf_quant_to_shmem(
    const unsigned short* __restrict__ hidden,  // [T, H] bf16 (H == K == 7168)
    unsigned char* __restrict__ a_src,          // [T, K] fp8 e4m3, SYMMETRIC staging
    float* __restrict__ sc_src,                 // [T, 56] fp32 row-major, SYMMETRIC
    int T, int K) {
  // Hard contract: exactly 896 threads (56 groups x 16 threads) and K == 7168.  Any other
  // configuration must call k0pf_quant_generic — returning here leaves a_src stale and the
  // harness gate FAILS LOUD, which is the point of the guard.
  if (K != 7168 || blockDim.x != 896) return;
  const int tid = threadIdx.x;                  // 0..895
  const int g = tid >> 4;                       // K128 group 0..55
  const int l16 = tid & 15;                     // lane within the 16-thread group
  const size_t base = (size_t)tid * 8;          // element offset within the token row

  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    // one 16 B vector load = this thread's 8 bf16 elements
    const uint4 raw = *reinterpret_cast<const uint4*>(hidden + (size_t)t * 7168 + base);
    const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw);
    float f[8];
    float a = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      f[e] = k0pfq_bf16_to_f32(h16[e]);
      a = fmaxf(a, fabsf(f[e]));
    }
    // amax over the 16-lane group (128 consecutive elements) — max is order-insensitive,
    // identical result to a two-warp LDS combine.
#pragma unroll
    for (int o = 1; o <= 8; o <<= 1) a = fmaxf(a, __shfl_xor(a, o, 64));
    const float scale = fmaxf(a, 1.0e-6f) / 448.0f;   // epsilon + e4m3 max
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

// Generic fallback: one element per thread, grid (T*NG,), block 128.
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
