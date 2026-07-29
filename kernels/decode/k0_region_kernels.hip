// ================================================================================================
// k0_region_kernels.hip — the region's cross-rank P2P sites hosted on MoRI's graph-safe device
// shmem primitives (ShmemGetMemBlock / ShmemPtrP2p).
//
// Raw peer loads through a translated device view are eager-correct but return zeros under HIP
// graph capture, so every cross-rank read here goes through the MoRI primitives instead.  The
// file is compiled with mori.jit.compile_genco, the same JIT path MoRI's own dispatch and combine
// kernels use.
//
// Kernels in this file:
//   k0_region_quant_to_shmem      BF16 -> FP8 E4M3 origin quantization straight into the
//                                 symmetric staging buffers.  Used by the decode graph.
//   k0_region_ag_route            all-gather of the top-k expert ids and route weights.
//   k0_region_gather_rows_fp8     pull-side dispatch: one FP8 row per distinct token.
//   k0_region_combine             owner-pull combine, FP32 accumulate in producer order.
//   k0_region_combine_push_store  push-store combine variant.
//
// There is no barrier kernel here.  The collective barrier is issued from the host as
// mori::shmem::shmem_barrier_on_stream(stream), which is stream-ordered and graph-capturable.
// Purely local work (plan construction, partial zeroing, the GEMM) lives in the pybind modules
// under kernels/plan and kernels/gemm; only cross-rank reads needed re-hosting.
// ================================================================================================
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#define K0_H_MAX 7168   // H == K == 7168 for DeepSeek-R1 decode; static LDS accumulator bound

// bf16 (stored as uint16) <-> fp32:
//   unpack : bf16 IS the top 16 bits of the fp32 -> __uint_as_float(bits << 16)   (exact)
//   pack   : round-to-nearest-even  (u + 0x7FFF + lsb) >> 16   (matches v_cvt_pk_bf16_f32)
__device__ __forceinline__ float k0_bf16_to_f32(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}
__device__ __forceinline__ unsigned short k0_f32_to_bf16_rne(float f) {
    const unsigned int u   = __float_as_uint(f);
    const unsigned int lsb = (u >> 16) & 1u;
    return (unsigned short)((u + 0x7FFFu + lsb) >> 16);   // 0x7FFF = the RNE bias
}

// ================================================================================================
// k0_region_quant_to_shmem — origin quantization, one pass, no intermediate copies.
//
// Writes both the FP8 row and the FP32 group scale DIRECTLY into the symmetric source buffers the
// dispatch reads from.  The alternative is to quantize into library-owned scratch and then memcpy
// the two results into the staging buffers, which costs an extra kernel and two copies.
//
// The math is the same dynamic per-(row, K128) amax quantization the expert GEMM uses on its own
// intermediate activation:
//   scale = max(amax, 1e-6) / 448 ;  q = cvt_fp8(clamp(x / scale, +-448), SATFINITE, E4M3)
//
// input_scale is written ROW-MAJOR here as sc_src[t * NG + g].  The dispatch scatters it
// GROUP-MAJOR into sc_dst[g * T_loc + t]; the GEMM's descriptor expects the group-major form.
//
// Grid: (T * NG,) blocks of 128 threads.  Block b owns (token t = b / NG, group g = b % NG) and
// thread tid owns element hidden[t, g * 128 + tid].  amax is reduced over the 128-lane group with
// two 64-lane warp shuffles plus a 2-entry LDS combine.  Fully parallel, deterministic, no atomics.
// ================================================================================================
extern "C" __global__ void k0_region_quant_to_shmem(
    const unsigned short* hidden,   // [T, H] bf16 (H == K)
    unsigned char* a_src,           // [T, K] fp8 e4m3, SYMMETRIC staging  (== gather source)
    float* sc_src,                  // [T, NG] fp32 row-major, SYMMETRIC   (== gather scale source)
    int T, int K) {
    const int NG  = K / 128;
    const int b   = blockIdx.x;
    const int t   = b / NG;
    const int g   = b % NG;
    if (t >= T) return;
    const int tid = threadIdx.x;                 // 0..127, one element per thread
    const size_t k = (size_t)g * 128 + tid;

    const float f = k0_bf16_to_f32(hidden[(size_t)t * K + k]);
    float a = fabsf(f);

    // amax over the 128-element group: 64-lane warp reduce, then combine the 2 warps via LDS.
    __shared__ float wmax[2];
    for (int o = 32; o >= 1; o >>= 1) a = fmaxf(a, __shfl_xor(a, o));
    if ((tid & 63) == 0) wmax[tid >> 6] = a;
    __syncthreads();
    const float amax  = fmaxf(wmax[0], wmax[1]);
    const float scale = fmaxf(amax, 1.0e-6f) / 448.0f;   // epsilon + e4m3 max

    if (tid == 0) sc_src[(size_t)t * NG + g] = scale;
    const float q = fminf(fmaxf(f / scale, -448.0f), 448.0f);
    a_src[(size_t)t * K + k] =
        __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
}

// ================================================================================================
// ag_route — all-gather top-k ids AND weights.  One block per source rank r.
//   r == cur : local copy of my_ids/my_wgt into all_*[r*n ..]
//   r != cur : ShmemGetMemBlock pulls peer r's ENTIRE my_ids/my_wgt slice (symmetric source addr).
// ================================================================================================
extern "C" __global__ void k0_region_ag_route(
    int* all_ids, float* all_wgt, const int* my_ids, const float* my_wgt,
    int world, int n, int cur) {
    const int r = blockIdx.x;
    if (r >= world) return;
    if (r == cur) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            all_ids[(size_t)r * n + i] = my_ids[i];
            all_wgt[(size_t)r * n + i] = my_wgt[i];
        }
    } else {
        mori::shmem::ShmemGetMemBlock((void*)(all_ids + (size_t)r * n),
                                      (const void*)my_ids, (size_t)n * sizeof(int), r);
        mori::shmem::ShmemGetMemBlock((void*)(all_wgt + (size_t)r * n),
                                      (const void*)my_wgt, (size_t)n * sizeof(float), r);
    }
}

// ================================================================================================
// gather_rows_fp8 — pull each distinct token's fp8 row once, write input_scale group-major.
//   a_dst  : LOCAL staging  == production `input`      [T_loc_max, K] fp8 (uint8)
//   sc_dst : LOCAL, GROUP-MAJOR == `input_scale`       sc_dst[grp * T_loc + t]
//   a_src / sc_src : SYMMETRIC per-rank source buffers (ShmemGetMemBlock reads peer's rows)
//   sc_stage : LOCAL global scratch [T_loc_max, NG] — the peer scale row lands here, then scattered.
// ================================================================================================
extern "C" __global__ void k0_region_gather_rows_fp8(
    unsigned char* a_dst, float* sc_dst,
    const unsigned char* a_src, const float* sc_src,
    const int* gath, const int* nvi, float* sc_stage,
    int T_src, int T_loc_max, int K, int cur) {
    const int t     = blockIdx.x;
    const int T_loc = nvi[1];
    if (t >= T_loc || t >= T_loc_max) return;

    const int NG       = K / 128;
    const int src_rank = gath[t * 2 + 0];
    const int src_tok  = gath[t * 2 + 1];
    const bool valid   = (src_rank >= 0 && src_tok >= 0 && src_tok < T_src);

    // ---- fp8 row (K bytes) ----
    if (!valid) {
        for (int i = threadIdx.x; i < K; i += blockDim.x) a_dst[(size_t)t * K + i] = 0;
    } else if (src_rank == cur) {
        for (int i = threadIdx.x; i < K; i += blockDim.x)
            a_dst[(size_t)t * K + i] = a_src[(size_t)src_tok * K + i];
    } else {
        mori::shmem::ShmemGetMemBlock((void*)(a_dst + (size_t)t * K),
                                      (const void*)(a_src + (size_t)src_tok * K),
                                      (size_t)K, src_rank);
    }

    // ---- scales -> GROUP-MAJOR write ----
    float* stg = sc_stage + (size_t)t * NG;
    if (valid) {
        if (src_rank == cur) {
            for (int g = threadIdx.x; g < NG; g += blockDim.x) stg[g] = sc_src[(size_t)src_tok * NG + g];
        } else {
            mori::shmem::ShmemGetMemBlock((void*)stg,
                                          (const void*)(sc_src + (size_t)src_tok * NG),
                                          (size_t)NG * sizeof(float), src_rank);
        }
        __syncthreads();
        for (int g = threadIdx.x; g < NG; g += blockDim.x) sc_dst[(size_t)g * T_loc + t] = stg[g];
    } else {
        for (int g = threadIdx.x; g < NG; g += blockDim.x) sc_dst[(size_t)g * T_loc + t] = 0.f;
    }
}

// ================================================================================================
// combine — owner pulls its <=8 producer partials, accumulates in fp32, one local store.
//   part  : symmetric-heap partial base  [T_loc_max, H] bf16
//   out   : LOCAL moe_out  [T, H] bf16 — written exactly once, locally
//   stage : LOCAL global scratch [T, H] bf16 — the current producer's peer row lands here per block
//   fp32 accumulate in LDS acc[H], in producer order lo..hi.
// ================================================================================================
extern "C" __global__ void k0_region_combine(
    unsigned short* out, const unsigned short* part,
    const int* pull_ptr, const int* pull_src, unsigned short* stage,
    int T, int H, int cur) {
    const int tok = blockIdx.x;
    if (tok >= T) return;

    __shared__ float acc[K0_H_MAX];
    const int lo = pull_ptr[tok];
    const int hi = pull_ptr[tok + 1];
    unsigned short* orow = out + (size_t)tok * H;

    // A token with NO producer must still be DEFINED: write zero (never leave stale across replays).
    if (hi <= lo) {
        for (int h = threadIdx.x; h < H; h += blockDim.x) orow[h] = 0;
        return;
    }

    for (int h = threadIdx.x; h < H; h += blockDim.x) acc[h] = 0.f;
    __syncthreads();

    unsigned short* srow = stage + (size_t)tok * H;
    for (int s = lo; s < hi; ++s) {
        const int p   = pull_src[s * 2 + 0];               // producer rank (block-uniform)
        const int row = pull_src[s * 2 + 1];               // its local row  (block-uniform)
        const unsigned short* peer = part + (size_t)row * H;
        const unsigned short* use;
        if (p == cur) {
            use = peer;                                    // local producer: read part directly
        } else {
            mori::shmem::ShmemGetMemBlock((void*)srow, (const void*)peer, (size_t)H * 2, p);
            __syncthreads();
            use = srow;
        }
        for (int h = threadIdx.x; h < H; h += blockDim.x)  // THE ACCUMULATE, fp32, producer order
            acc[h] += k0_bf16_to_f32(use[h]);
        __syncthreads();                                   // srow reusable next iteration
    }

    for (int h = threadIdx.x; h < H; h += blockDim.x)      // ONE local bf16 store, RNE pack
        orow[h] = k0_f32_to_bf16_rne(acc[h]);
}

// ================================================================================================
// Negative control used by the harness: an intentionally wrong combine that stores each
// producer's row (last writer wins) instead of accumulating.  It must FAIL the correctness gate;
// if it passes, the gate is not actually testing the combine.
// ================================================================================================
extern "C" __global__ void k0_region_combine_push_store(
    unsigned short* out, const unsigned short* part,
    const int* pull_ptr, const int* pull_src, unsigned short* stage,
    int T, int H, int cur) {
    const int tok = blockIdx.x;
    if (tok >= T) return;
    const int lo = pull_ptr[tok];
    const int hi = pull_ptr[tok + 1];
    unsigned short* orow = out + (size_t)tok * H;
    unsigned short* srow = stage + (size_t)tok * H;
    for (int s = lo; s < hi; ++s) {
        const int p   = pull_src[s * 2 + 0];
        const int row = pull_src[s * 2 + 1];
        const unsigned short* peer = part + (size_t)row * H;
        if (p == cur) {
            for (int h = threadIdx.x; h < H; h += blockDim.x) orow[h] = peer[h];  // STORE, not accumulate
        } else {
            mori::shmem::ShmemGetMemBlock((void*)srow, (const void*)peer, (size_t)H * 2, p);
            __syncthreads();
            for (int h = threadIdx.x; h < H; h += blockDim.x) orow[h] = srow[h];   // store, not accumulate
            __syncthreads();
        }
    }
}
