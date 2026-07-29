// Single-kernel FP8 expert GEMM for gfx950.
// The persistent work count comes from device-resident num_valid_ids[0]. A bounded descriptor
// maps padding-row loads to zero. Each CTA gathers a 32x128-byte activation tile once in memory
// order and restores MFMA layout through XOR-swizzled LDS.
//
// The K loop double-buffers activation and weight loads. Its barrier is scoped to local memory so
// it does not drain VMEM. The swizzle uses physical_chunk=chunk^(row&7) to reduce bank conflicts.

#include "kittens.cuh"
#include "n1g_fused_moe.hpp"
#include "aiter_gfx950_fp8_layout.hpp"
#include "pyutils/pyutils.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

using namespace kittens;

namespace production_fused_moe::n1g {

namespace aiter = production_fused_moe::aiter_gfx950;

// ---- geometry --------------------------------------------------------------
constexpr int kExperts = 32;
constexpr int kHidden = 7168;    // K of W13, N of W2
constexpr int kW13N = 4096;      // rows [0,2048) gate, [2048,4096) up
constexpr int kInter = 2048;     // intermediate dim
constexpr int kBlockM = 32;      // Opus sorted block
constexpr int kDispatchRows = 4096;

// ---- decomposition ---------------------------------------------------------
constexpr int kChunks = 8;                        // intermediate chunks
constexpr int kChunkCols = kInter / kChunks;      // 256
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * 64;             // 256
constexpr int kCTAs = 256;                        // one per CU, persistent
constexpr int kWaveCols = kChunkCols / kWaves;    // 64 intermediate cols/wave
constexpr int kColTiles = kWaveCols / 16;         // 4 x 16-col MFMA tiles

constexpr int kKGroups = kHidden / 128;           // 56 K128 groups in W13
constexpr int kNGroups13 = kW13N / 128;           // 32
constexpr int kNGroups2 = kHidden / 128;          // 56
constexpr int kKGroups2 = kInter / 128;           // 16
constexpr int kN16Tiles = kHidden / 16;           // 448 output tiles for W2
constexpr int kN2Unroll = 4;
constexpr int kN2Iters = kN16Tiles / (kN2Unroll * kWaves);  // 28

// A2 LDS row stride: 256 + 16.
constexpr int kA2Stride = 272;

// ---- the CTA-shared A tile -------------------------------------------------
// One K128 group of the 32 sorted rows: 32 rows x 8 chunks x 16 B = 4,096 B.
// Double-buffered size is 8,192 B.
constexpr int kAChunks = 8;        // 128 B per row / 16 B per thread
constexpr int kAChunkBytes = 16;

static_assert(kKGroups % 2 == 0, "the K loop is unrolled by two");
static_assert(kN2Iters % 2 == 0, "the N loop is unrolled by two");
static_assert(kN16Tiles == kN2Iters * kN2Unroll * kWaves, "W2 N coverage");
static_assert(kChunkCols % (16 * kWaves) == 0, "W13 chunk / wave split");
static_assert(kThreads == kBlockM * kAChunks,
              "the A tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A tile row");

using tensor_bf16 = gl<bf16, -1, -1, -1, -1>;
using tensor_fp32 = gl<float, -1, -1, -1, -1>;
using tensor_i32 = gl<int, -1, -1, -1, -1>;

// The MFMA operand tile (16 x 128 FP8) and the MFMA accumulator (16 x 16 FP32).
using rfp8 = rt_fp8e4m3<16, 128>;
using racc = rt_fl<16, 16, col_l, rt_16x16_s>;

// The fragment map this whole file depends on.
static_assert(rfp8::height == 1 && rfp8::width == 1);
static_assert(rfp8::packed_per_thread == 8);              // 32 bytes / lane
static_assert(rfp8::base_tile_stride == 16);
static_assert(rfp8::base_tile_num_strides == 2);
static_assert(rfp8::base_tile_elements_per_stride_group == 64);
static_assert(racc::height == 1 && racc::width == 1);
static_assert(racc::packed_per_thread == 2);              // 4 floats / lane

// ---------------------------------------------------------------------------
// Weight loader: two 16-byte vector loads per lane,
// no LDS, no swizzle, no barrier, register destination.  The AITER preshuffled
// tile's second stride group sits one 1,024 B wave window later.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_bfrag(rfp8& dst, const std::uint8_t* p) {
  float4* d = reinterpret_cast<float4*>(&dst.tiles[0][0].data[0]);
  d[0] = *reinterpret_cast<const float4*>(p);
  d[1] = *reinterpret_cast<const float4*>(p + 1024);
}

__device__ __forceinline__ float* accf(racc& a) {
  return reinterpret_cast<float*>(&a.tiles[0][0].data[0]);
}
__device__ __forceinline__ const float* accf(const racc& a) {
  return reinterpret_cast<const float*>(&a.tiles[0][0].data[0]);
}

// The backend stages MFMA operands in AGPRs while keeping destinations in VGPRs. Packed float2
// accumulation selects paired FP32 operations without changing arithmetic order.
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int agpr_frag_t;
typedef __attribute__((__vector_size__(2 * sizeof(float)))) float f32x2;

// Optional explicit AGPR staging. The default backend flag selects VGPR destinations without
// introducing VMEM drains.
#ifndef N1G_STAGE_AGPR
#define N1G_STAGE_AGPR 0
#endif
__device__ __forceinline__ void stage_agpr(rfp8& f) {
#if N1G_STAGE_AGPR
  agpr_frag_t* d = reinterpret_cast<agpr_frag_t*>(&f.tiles[0][0].data[0]);
  asm("" : "+a"(*d));
#else
  (void)f;
#endif
}

// The racc payload, viewed as the two float2 pairs it already is.
__device__ __forceinline__ f32x2* accv(racc& a) {
  return reinterpret_cast<f32x2*>(&a.tiles[0][0].data[0]);
}
__device__ __forceinline__ const f32x2* accv(const racc& a) {
  return reinterpret_cast<const f32x2*>(&a.tiles[0][0].data[0]);
}

// ---------------------------------------------------------------------------
// Local-memory-only CTA barrier for the activation tile.
//
// `__syncthreads()` fences global memory at workgroup scope and emits
// `s_waitcnt vmcnt(0)`, which would retire all 17 in-flight weight loads once
// per K step.  Restricting the fence to LDS makes the memory legalizer emit
// `s_waitcnt lgkmcnt(0)` and nothing else.  The build gate verifies from the ISA
// that no `vmcnt(0)` reaches either inner loop.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void lds_cta_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Fused kernel.
__global__ __launch_bounds__(kThreads, 1) void n1g_fused_moe_kernel(
    const std::uint8_t* __restrict__ A_bytes,   // input  [4096, 7168] FP8
    const float* __restrict__ A_scale,          // input_scale, group-major live
    const std::uint8_t* __restrict__ W13,       // gate   [32, 4096, 7168] AITER
    const float* __restrict__ S13,              // fc1_scale [32, 32, 56]
    const std::uint8_t* __restrict__ W2,        // down   [32, 7168, 2048] AITER
    const float* __restrict__ S2,               // fc2_scale [32, 56, 16]
    const int* __restrict__ sorted_ids,         // packed: token|slot<<24
    const float* __restrict__ sorted_w,
    const int* __restrict__ sorted_eid,
    const int* __restrict__ nvi,                // [padded_rows, T]
    __hip_bfloat16* __restrict__ OUT) {         // out [4096, 7168] BF16
  __shared__ int tok_lds[kBlockM];
  __shared__ float w_lds[kBlockM];
  __shared__ float ascale_lds[kKGroups][kBlockM];  // [k128][sorted row]
  __shared__ float b1s_lds[2][2][kKGroups];        // [gate/up][n128 half][k128]
  __shared__ float b2s_lds[kNGroups2][2];          // [n128][kb]
  __shared__ float amax_lds[2][kBlockM];           // [kb][sorted row]
  __shared__ __align__(16) std::uint8_t a2_lds[kBlockM][kA2Stride];

  // The wave-local BF16 output transpose.
  __shared__ __align__(16) std::uint32_t xp_lds[kWaves][kBlockM][32];

  // CTA-shared, double-buffered activation tile: [buffer][row][16-byte chunk].
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kBlockM][kAChunks][kAChunkBytes];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;   // wave 0..3
  const int q = lane >> 4;   // 0..3
  const int r = lane & 15;   // 0..15
  const int T = nvi[1];

  // Device-derived work count; the wave-uniform load is hoisted above the persistent loop.
  const int num_tasks = (((nvi[0] + (kBlockM - 1)) >> 5) << 3);

  // Bound the descriptor to T rows so padding loads return zero.
  const std::uint32_t a_num_records =
      static_cast<std::uint32_t>(T) * static_cast<std::uint32_t>(kHidden);
  const i32x4 a_rsrc = make_srsrc(A_bytes, a_num_records, 0);

  // Eight consecutive lanes cover one row's 128-byte K group.
  const int a_row = tid >> 3;              // 0..31
  const int a_chunk = tid & 7;             // 0..7
  const int a_cp = a_chunk ^ (a_row & 7);  // the XOR swizzle (see header)

  // Each wave owns intermediate columns [64*wv, +64) of this chunk, so it sits
  // entirely inside one 128-wide block-scale group and A2 quant group.
  const int j_w = wv >> 1;

  for (int task = blockIdx.x; task < num_tasks; task += kCTAs) {
    const int b = task >> 3;   // sorted block
    const int g = task & 7;    // intermediate chunk
    const int e = sorted_eid[b];

    // ---- per-task LDS setup --------------------------------------------------
    if (tid < kBlockM) {
      // Packed sorted_token_ids: token = low 24 bits.  Stock tests token < T and
      // Nothing else; the slot bits are never read.
      tok_lds[tid] = sorted_ids[b * kBlockM + tid] & 0x00FFFFFF;
      w_lds[tid] = sorted_w[b * kBlockM + tid];
    }
    if (tid < 2 * kBlockM) {
      amax_lds[tid >> 5][tid & 31] = 0.0f;
    }
    if (tid < 2 * 2 * kKGroups) {
      const int k = tid % kKGroups;
      const int j = (tid / kKGroups) & 1;
      const int gu = tid / (2 * kKGroups);
      const int ng = (gu == 0) ? (2 * g + j) : (kInter / 128 + 2 * g + j);
      b1s_lds[gu][j][k] = S13[(e * kNGroups13 + ng) * kKGroups + k];
    }
    if (tid < kNGroups2 * 2) {
      const int kb = tid & 1;
      const int ng = tid >> 1;
      b2s_lds[ng][kb] = S2[(e * kNGroups2 + ng) * kKGroups2 + (2 * g + kb)];
    }
    __syncthreads();

    // input_scale uses group-major live-prefix stride: input_scale[k128*T+token].
    for (int idx = tid; idx < kKGroups * kBlockM; idx += kThreads) {
      const int k = idx >> 5;
      const int i = idx & 31;
      const int token = tok_lds[i];
      ascale_lds[k][i] = (token < T)
                             ? A_scale[static_cast<std::size_t>(k) * T + token]
                             : 0.0f;
    }
    __syncthreads();

    // Byte offset of this thread's 16 B slice of the A tile at K128 group 0.
    // The descriptor maps padding-row offsets to zero, including wrapped sentinels.
    const std::uint32_t a_voff0 =
        static_cast<std::uint32_t>(tok_lds[a_row]) *
            static_cast<std::uint32_t>(kHidden) +
        static_cast<std::uint32_t>(a_chunk * kAChunkBytes);

    // Paired gate/up GEMM over matching column tiles.
    racc acc[2][2][kColTiles];
#pragma unroll
    for (int gu = 0; gu < 2; ++gu) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int c = 0; c < kColTiles; ++c) {
          zero(acc[gu][m][c]);
        }
      }
    }

    const int n_gate = kChunkCols * g + kWaveCols * wv;
    const int n_up = kInter + n_gate;

    rfp8 af[2];                // [rowtile]  A(k), read from LDS just in time
    __uint128_t aTmp[2];       // [k&1]      A(k) in flight / awaiting its ds_write
    rfp8 bf[2][2][kColTiles];  // [buf][gate/up][col]

    // Coalesced activation load through the bounded descriptor.
    auto load_a = [&](int k, int buf) __attribute__((always_inline)) {
      aTmp[buf] = llvm_amdgcn_raw_buffer_load_b128(
          a_rsrc, a_voff0 + static_cast<std::uint32_t>(k * 128), 0u, 0u);
    };
    // ---- wA: one ds_write_b128 per thread -> the CTA-shared tile.
    auto store_a = [&](int buf, int lb) __attribute__((always_inline)) {
      *reinterpret_cast<__uint128_t*>(&a_lds[lb][a_row][a_cp][0]) = aTmp[buf];
    };
    // ---- rA: the MFMA fragment read.  Lane 16q+r takes sorted rows {r, 16+r}
    //      and, within each, chunks q and 4|q -- i.e. exactly the two 16 B runs
    //      that rt_fp8e4m3<16,128> expects at byte 16q and byte 64+16q.
    auto read_a = [&](int lb) __attribute__((always_inline)) {
      const int xr = r & 7;
      const int cp0 = q ^ xr;
      const int cp1 = cp0 ^ 4;  // (4|q) ^ xr == (q ^ xr) ^ 4, since q < 4
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = 16 * m + r;  // (16m + r) & 7 == r & 7 == xr
        float4* d = reinterpret_cast<float4*>(&af[m].tiles[0][0].data[0]);
        d[0] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp0][0]);
        d[1] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp1][0]);
      }
    };
    // ---- gB: the weight loader.
    auto load_b = [&](int k, int buf) __attribute__((always_inline)) {
      const int kbyte = k * 128;
#pragma unroll
      for (int c = 0; c < kColTiles; ++c) {
        const std::size_t tg = aiter::fp8_weight_byte_offset(
            e, n_gate + 16 * c, kbyte, kW13N, kHidden);
        const std::size_t tu = aiter::fp8_weight_byte_offset(
            e, n_up + 16 * c, kbyte, kW13N, kHidden);
        load_bfrag(bf[buf][0][c], W13 + tg + 16 * lane);
        load_bfrag(bf[buf][1][c], W13 + tu + 16 * lane);
        stage_agpr(bf[buf][0][c]);   // N1g: MFMA srcA lives in the AGPR file
        stage_agpr(bf[buf][1][c]);
      }
    };

    // MFMA block with a fresh FP32 partial per K128 group.
    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
      const float4 as0 =
          *reinterpret_cast<const float4*>(&ascale_lds[k][4 * q]);
      const float4 as1 =
          *reinterpret_cast<const float4*>(&ascale_lds[k][16 + 4 * q]);
      const float bsg = b1s_lds[0][j_w][k];
      const float bsu = b1s_lds[1][j_w][k];
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const float4 as = (m == 0) ? as0 : as1;
#pragma unroll
        for (int gu = 0; gu < 2; ++gu) {
          const float bs = (gu == 0) ? bsg : bsu;
          const f32x2 bsv = {bs, bs};
          const f32x2 s0 = f32x2{as.x, as.y} * bsv;
          const f32x2 s1 = f32x2{as.z, as.w} * bsv;
#pragma unroll
          for (int c = 0; c < kColTiles; ++c) {
            racc p;
            zero(p);
            mma_ABt(p, af[m], bf[buf][gu][c], p);
            // cf[t] += pf[t] * (bs * asv[t]) for t in [0,4), spelled packed:
            const f32x2* pf = accv(p);
            f32x2* cf = accv(acc[gu][m][c]);
            cf[0] += pf[0] * s0;
            cf[1] += pf[1] * s1;
          }
        }
      }
    };

    // ---- the pipeline.  A(j) lives in aTmp[j&1] and lands in a_lds[j&1].
    load_a(0, 0);   // gA(0)
    load_a(1, 1);   // gA(1)
    load_b(0, 0);   // gB(0) -- 16 VMEM, issued AFTER both A loads
    store_a(0, 0);  // wA(0): partial vmcnt wait only (the 16 gB(0) stay in flight)
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroups; k += 2) {
      // Clamp prefetches so both halves keep the same control flow.
      const int ka2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);
      const int ka3 = (k + 3 < kKGroups) ? (k + 3) : (kKGroups - 1);
      const int kb1 = (k + 1 < kKGroups) ? (k + 1) : (kKGroups - 1);
      const int kb2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);

      // ---- half 0: step k.   LDS buf 0, B buf 0. -------------------------
      read_a(0);       //  4 ds_read_b128  <- a_lds[0] = A(k)
      load_a(ka2, 0);  //  1 VMEM          -> aTmp[0]   (A(k) is already stored)
      load_b(kb1, 1);  // 16 VMEM          -> bf[1]
      mfma_k(k, 0);    // 16 MFMA          (17 loads stay in flight across it)
      store_a(1, 1);   //  1 ds_write_b128 : aTmp[1] = A(k+1) -> a_lds[1]
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);   //  4 DS read
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);  // 17 VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);  // 16 MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   //  1 DS write
      lds_cta_barrier();

      // ---- half 1: step k+1.  LDS buf 1, B buf 1. ------------------------
      read_a(1);       //  4 ds_read_b128  <- a_lds[1] = A(k+1)
      load_a(ka3, 1);  //  1 VMEM          -> aTmp[1]
      load_b(kb2, 0);  // 16 VMEM          -> bf[0]
      mfma_k(kb1, 1);  // 16 MFMA
      store_a(0, 0);   //  1 ds_write_b128 : aTmp[0] = A(k+2) -> a_lds[0]
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();
    }

    // =======================================================================
    // SiLU and per-row K128 FP8 quantization into LDS.

    // =======================================================================
    bool live[2][4];
    float a2v[2][kColTiles][4];
    float lmax[2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        live[m][t] = tok_lds[16 * m + 4 * q + t] < T;
        lmax[m][t] = 0.0f;
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int c = 0; c < kColTiles; ++c) {
        const float* gf = accf(acc[0][m][c]);
        const float* uf = accf(acc[1][m][c]);
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const float gv = gf[t];
          const float hv = (gv / (1.0f + __expf(-gv))) * uf[t];
          const float h = live[m][t] ? hv : 0.0f;
          a2v[m][c][t] = h;
          lmax[m][t] = fmaxf(lmax[m][t], fabsf(h));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int mask = 1; mask < 16; mask <<= 1) {
          lmax[m][t] = fmaxf(lmax[m][t], __shfl_xor(lmax[m][t], mask, 64));
        }
      }
    }
    if (r == 0) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          atomicMax(reinterpret_cast<int*>(&amax_lds[j_w][16 * m + 4 * q + t]),
                    __float_as_int(lmax[m][t]));
        }
      }
    }
    __syncthreads();

    // The epsilon prevents NaN for empty experts.
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      float dq[4];
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        dq[t] = fmaxf(amax_lds[j_w][16 * m + 4 * q + t], 1.0e-6f) / 448.0f;
      }
#pragma unroll
      for (int c = 0; c < kColTiles; ++c) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          const float qv =
              fminf(fmaxf(a2v[m][c][t] / dq[t], -448.0f), 448.0f);
          a2_lds[16 * m + 4 * q + t][kWaveCols * wv + 16 * c + r] =
              __hip_cvt_float_to_fp8(qv, __HIP_SATFINITE, __HIP_E4M3);
        }
      }
    }
    __syncthreads();

    // =======================================================================
    // Split-K W2 and BF16 output transpose.
    // 2 cache lines per atomic.
    // =======================================================================
    rfp8 a2f[2][2];  // [rowtile][kb]
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const std::uint8_t* p = &a2_lds[16 * m + r][128 * kb + 16 * q];
        float4* d = reinterpret_cast<float4*>(&a2f[m][kb].tiles[0][0].data[0]);
        d[0] = *reinterpret_cast<const float4*>(p);
        d[1] = *reinterpret_cast<const float4*>(p + 64);
      }
    }
    // Dequantize before the atomic.
    float dq2[2][2];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        dq2[m][kb] = fmaxf(amax_lds[kb][16 * m + r], 1.0e-6f) / 448.0f;
      }
    }
    const int tok2[2] = {tok_lds[r], tok_lds[16 + r]};
    const float sw2[2] = {w_lds[r], w_lds[16 + r]};
    const bool lv2[2] = {tok2[0] < T, tok2[1] < T};

    const int rowh = lane >> 5;   // 0 or 1: which of the atomic's two rows
    const int dcol = lane & 31;   // which dword (= 2 adjacent output columns)
    int xtok[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      xtok[i] = tok_lds[2 * i + rowh];
    }

    rfp8 wf[2][kN2Unroll][2];  // [buf][n tile][kb]
    racc c2[kN2Unroll][2];     // [n tile][rowtile]

    auto load_n = [&](int it, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int j = 0; j < kN2Unroll; ++j) {
        const int nt = (kN2Unroll * kWaves) * it + kN2Unroll * wv + j;
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const std::size_t tb = aiter::fp8_weight_byte_offset(
              e, 16 * nt, kChunkCols * g + 128 * kb, kHidden, kInter);
          load_bfrag(wf[buf][j][kb], W2 + tb + 16 * lane);
          stage_agpr(wf[buf][j][kb]);  // N1g: MFMA srcA lives in the AGPR file
        }
      }
    };

    auto compute_n = [&](int it, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int j = 0; j < kN2Unroll; ++j) {
        const int nt = (kN2Unroll * kWaves) * it + kN2Unroll * wv + j;
        const int ng = nt >> 3;  // n / 128
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          zero(c2[j][m]);
        }
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          const float fs = b2s_lds[ng][kb];
#pragma unroll
          for (int m = 0; m < 2; ++m) {
            racc p;
            zero(p);
            mma_ABt(p, wf[buf][j][kb], a2f[m][kb], p);
            const float s = fs * dq2[m][kb];
            // cf[t] += pf[t] * s, spelled packed: 2 v_pk_fma_f32, no AGPR read.
            const f32x2 sv = {s, s};
            const f32x2* pf = accv(p);
            f32x2* cf = accv(c2[j][m]);
            cf[0] += pf[0] * sv;
            cf[1] += pf[1] * sv;
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);

      // Write MFMA layout; lane 16q+r holds rows {r,16+r}.
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = 16 * m + r;
        const float sw = lv2[m] ? sw2[m] : 0.0f;
        const int xorm = 2 * (row & 15);
#pragma unroll
        for (int j = 0; j < kN2Unroll; ++j) {
          const float* cf = accf(c2[j][m]);
          __hip_bfloat162 v0, v1;
          v0.x = __float2bfloat16(cf[0] * sw);
          v0.y = __float2bfloat16(cf[1] * sw);
          v1.x = __float2bfloat16(cf[2] * sw);
          v1.y = __float2bfloat16(cf[3] * sw);
          const int phys = (8 * j + 2 * q) ^ xorm;
          uint2 packed;
          packed.x = *reinterpret_cast<const std::uint32_t*>(&v0);
          packed.y = *reinterpret_cast<const std::uint32_t*>(&v1);
          *reinterpret_cast<uint2*>(&xp_lds[wv][row][phys]) = packed;
        }
      }

      // The transpose is wave-local; no CTA barrier is needed.
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
      __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0); vmcnt/expcnt free
      __builtin_amdgcn_wave_barrier();

      // Read atomic layout. token<T prevents address formation for padding rows.
      const int col_base = (kN2Unroll * kWaves * 16) * it + 64 * wv;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int row = 2 * i + rowh;
        const std::uint32_t d = xp_lds[wv][row][dcol ^ (2 * (row & 15))];
        if (xtok[i] < T) {
          __hip_bfloat162* p = reinterpret_cast<__hip_bfloat162*>(
              OUT + static_cast<std::size_t>(xtok[i]) * kHidden + col_base +
              2 * dcol);
          unsafeAtomicAdd(p, *reinterpret_cast<const __hip_bfloat162*>(&d));
        }
      }
      // WAR: the next iteration's ds_write must not overtake these ds_reads.
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
    };

    load_n(0, 0);
#pragma unroll 1
    for (int it = 0; it < kN2Iters; it += 2) {
      const int in0 = (it + 1 < kN2Iters) ? (it + 1) : (kN2Iters - 1);
      const int in1 = (it + 2 < kN2Iters) ? (it + 2) : (kN2Iters - 1);
      load_n(in0, 1);
      compute_n(it, 0);
      load_n(in1, 0);
      compute_n(it + 1, 1);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        __builtin_amdgcn_sched_group_barrier(0x020, 16, 0);  // 16 VMEM reads
        __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);  // 16 MFMA
      }
    }
    // The next task refills every LDS buffer this one just read.
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host side.  Exactly the production tensors: no private weight repack, no
// derived host tensor, and no host-frozen scalar.  `num_tasks` is gone from the
// struct, the kernarg and the launch path.  No host work is left outside the
// timer, so this is directly comparable to the production kernel.
// ---------------------------------------------------------------------------
struct n1g_globals {
  tensor_bf16 input_bytes;        // BF16 view of production `input`
  tensor_fp32 input_scale;        // production `input_scale`
  tensor_bf16 w13_bytes;          // BF16 view of production `gate` (AITER)
  tensor_fp32 w13_scales;         // production `fc1_scale`
  tensor_bf16 w2_bytes;           // BF16 view of production `down` (AITER)
  tensor_fp32 w2_scales;          // production `fc2_scale`
  tensor_i32 sorted_token_ids;
  tensor_fp32 sorted_weights;
  tensor_i32 sorted_expert_ids;
  tensor_i32 num_valid_ids;
  tensor_bf16 out_bf16;           // production `out`, pre-zeroed in [0,T)
  std::uintptr_t stream_ptr;
};

[[noreturn]] void invalid(const char* message) {
  throw std::invalid_argument(std::string("n1g_fused_moe: ") + message);
}

void require(bool condition, const char* message) {
  if (!condition) {
    invalid(message);
  }
}

void validate_contract(const n1g_globals& g) {
  require(aiter::is_supported_blockscale_shape(kExperts, kW13N, kHidden),
          "W13 AITER shape unsupported");
  require(aiter::is_supported_blockscale_shape(kExperts, kHidden, kInter),
          "W2 AITER shape unsupported");
  require(g.input_bytes.batch() == 1 && g.input_bytes.depth() == 1 &&
              g.input_bytes.rows() == kDispatchRows &&
              g.input_bytes.cols() == kHidden / 2,
          "input must have BF16-view shape [4096,3584]");
  require(g.input_scale.batch() == 1 && g.input_scale.depth() == 1 &&
              g.input_scale.rows() == kDispatchRows &&
              g.input_scale.cols() == kKGroups,
          "input_scale must have FP32 shape [4096,56]");
  require(g.w13_bytes.batch() == 1 && g.w13_bytes.depth() == 1 &&
              g.w13_bytes.rows() == kExperts * kW13N &&
              g.w13_bytes.cols() == kHidden / 2,
          "gate must have BF16-view shape [32*4096,3584]");
  require(g.w13_scales.batch() == 1 && g.w13_scales.depth() == kExperts &&
              g.w13_scales.rows() == kNGroups13 &&
              g.w13_scales.cols() == kKGroups,
          "fc1_scale must have FP32 shape [32,32,56]");
  require(g.w2_bytes.batch() == 1 && g.w2_bytes.depth() == 1 &&
              g.w2_bytes.rows() == kExperts * kHidden &&
              g.w2_bytes.cols() == kInter / 2,
          "down must have BF16-view shape [32*7168,1024]");
  require(g.w2_scales.batch() == 1 && g.w2_scales.depth() == kExperts &&
              g.w2_scales.rows() == kNGroups2 &&
              g.w2_scales.cols() == kKGroups2,
          "fc2_scale must have FP32 shape [32,56,16]");
  require(g.num_valid_ids.cols() == 2, "num_valid_ids must hold two values");
  require(g.out_bf16.batch() == 1 && g.out_bf16.depth() == 1 &&
              g.out_bf16.rows() == kDispatchRows &&
              g.out_bf16.cols() == kHidden,
          "out must have BF16 shape [4096,7168]");
  require(static_cast<std::size_t>(g.sorted_token_ids.cols()) / kBlockM <=
              static_cast<std::size_t>(g.sorted_expert_ids.cols()),
          "sorted_token_ids capacity implies more blocks than sorted_expert_ids "
          "can name");
  require(g.sorted_weights.cols() == g.sorted_token_ids.cols(),
          "sorted_weights and sorted_token_ids must have equal capacity");
}

void dispatch_n1g_fused_moe(n1g_globals g) {
  validate_contract(g);
  hipStream_t stream = reinterpret_cast<hipStream_t>(g.stream_ptr);
  n1g_fused_moe_kernel<<<kCTAs, kThreads, 0, stream>>>(
      reinterpret_cast<const std::uint8_t*>(g.input_bytes.raw_ptr),
      g.input_scale.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w13_bytes.raw_ptr),
      g.w13_scales.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w2_bytes.raw_ptr),
      g.w2_scales.raw_ptr, g.sorted_token_ids.raw_ptr,
      g.sorted_weights.raw_ptr, g.sorted_expert_ids.raw_ptr,
      g.num_valid_ids.raw_ptr,
      reinterpret_cast<__hip_bfloat16*>(g.out_bf16.raw_ptr));
}

}  // namespace production_fused_moe::n1g

void bind_n1g_fused_moe(pybind11::module_& module) {
  using production_fused_moe::n1g::dispatch_n1g_fused_moe;
  using production_fused_moe::n1g::n1g_globals;
  py::bind_function<dispatch_n1g_fused_moe>(
      module, "n1g_fused_moe",
      &n1g_globals::input_bytes,
      &n1g_globals::input_scale,
      &n1g_globals::w13_bytes,
      &n1g_globals::w13_scales,
      &n1g_globals::w2_bytes,
      &n1g_globals::w2_scales,
      &n1g_globals::sorted_token_ids,
      &n1g_globals::sorted_weights,
      &n1g_globals::sorted_expert_ids,
      &n1g_globals::num_valid_ids,
      &n1g_globals::out_bf16,
      &n1g_globals::stream_ptr);
}
