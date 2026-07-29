// ===========================================================================
// n2_phase1.cpp — expert GEMM, phase 1 of 2: the W13 projection, with the
// intermediate activation handed to phase 2 through a small global buffer.
//
// The pair (n2_phase1, n2_phase2) exists to change one thing about the W2
// reduction: a split-K W2 writes each output element eight times through BF16
// atomic partials, an 8x write amplification.  Phase 2 instead does split-N
// full-K, so every output element is written exactly once.  Phase 1 is what
// makes that possible: it materializes the quantized intermediate activation
// rather than passing it through LDS.
//
// The GEMM itself is unchanged from the single-kernel version: task = (sorted
// block b, intermediate chunk g), a CTA-shared A tile, the 56-step paired
// gate/up K-loop with 17 loads in flight, register SiLU, a dynamic
// per-(row, K128) amax, and FP8 E4M3 quantization.  The math, the schedule and
// every scale are identical.  Only the destination changes:
//
//   A2q[row_global, 256g : +256]  FP8 E4M3, row-major, 2048 B rows
//   DQ2[row_global, 2g + kb]      FP32 dequant scale = max(amax, 1e-6) / 448
//
// Both are fully written for every row < nvi[0] on every call — padding rows
// carry exact zeros from the live-select — so phase 2 never reads a stale byte.
// No zeroing, no epoch state, and safe under a route swap by construction.
//
// This kernel writes no output rows: the entire split-K GEMM-2, its LDS staging
// and every output atomic are gone from phase 1.
// ===========================================================================

#include "kittens.cuh"
#include "n2_fused_moe.hpp"
#include "aiter_gfx950_fp8_layout.hpp"
#include "pyutils/pyutils.cuh"

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

using namespace kittens;

namespace production_fused_moe::n2 {

namespace aiter = production_fused_moe::aiter_gfx950;

// ---- production geometry (CLAUDE.md I/O contract) --------------------------
constexpr int kExperts = 32;
constexpr int kHidden = 7168;    // K of W13, N of W2
constexpr int kW13N = 4096;      // rows [0,2048) gate, [2048,4096) up
constexpr int kInter = 2048;     // intermediate dim
constexpr int kBlockM = 32;      // Opus sorted block

// ---- the decomposition (stock's) ------------------------------------------
constexpr int kChunks = 8;                        // intermediate chunks
constexpr int kChunkCols = kInter / kChunks;      // 256
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * 64;             // 256
constexpr int kCTAs = 256;                        // one per CU, persistent
constexpr int kWaveCols = kChunkCols / kWaves;    // 64 intermediate cols/wave
constexpr int kColTiles = kWaveCols / 16;         // 4 x 16-col MFMA tiles

constexpr int kKGroups = kHidden / 128;           // 56 K128 groups in W13
constexpr int kNGroups13 = kW13N / 128;           // 32
constexpr int kKGroups2 = kInter / 128;           // 16 (A2 K128 groups)

// A2 LDS row stride: 256 + 16.
constexpr int kA2Stride = 272;

// ---- the CTA-shared A tile -------------------------------------------------
constexpr int kAChunks = 8;        // 128 B per row / 16 B per thread
constexpr int kAChunkBytes = 16;

static_assert(kKGroups % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkCols % (16 * kWaves) == 0, "W13 chunk / wave split");
static_assert(kThreads == kBlockM * kAChunks,
              "the A tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A tile row");

using tensor_bf16 = gl<bf16, -1, -1, -1, -1>;
using tensor_fp32 = gl<float, -1, -1, -1, -1>;
using tensor_i32 = gl<int, -1, -1, -1, -1>;

using rfp8 = rt_fp8e4m3<16, 128>;
using racc = rt_fl<16, 16, col_l, rt_16x16_s>;

static_assert(rfp8::height == 1 && rfp8::width == 1);
static_assert(rfp8::packed_per_thread == 8);              // 32 bytes / lane
static_assert(rfp8::base_tile_stride == 16);
static_assert(rfp8::base_tile_num_strides == 2);
static_assert(rfp8::base_tile_elements_per_stride_group == 64);
static_assert(racc::height == 1 && racc::width == 1);
static_assert(racc::packed_per_thread == 2);              // 4 floats / lane

// ---------------------------------------------------------------------------
// The WEIGHT loader.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_bfrag(rfp8& dst, const std::uint8_t* p) {
  float4* d = reinterpret_cast<float4*>(&dst.tiles[0][0].data[0]);
  d[0] = *reinterpret_cast<const float4*>(p);
  d[1] = *reinterpret_cast<const float4*>(p + 1024);
}

__device__ __forceinline__ float* accf(racc& a) {
  return reinterpret_cast<float*>(&a.tiles[0][0].data[0]);
}

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int agpr_frag_t;
typedef __attribute__((__vector_size__(2 * sizeof(float)))) float f32x2;

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

__device__ __forceinline__ f32x2* accv(racc& a) {
  return reinterpret_cast<f32x2*>(&a.tiles[0][0].data[0]);
}

// The CTA barrier for the LDS A tile.  LOCAL ADDRESS SPACE ONLY — no vmcnt(0)
// may reach the K-loop.
__device__ __forceinline__ void lds_cta_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ===========================================================================
// PHASE 1: W13 -> register SiLU -> dynamic amax -> FP8 quant -> GLOBAL A2 handoff
// ===========================================================================
__global__ __launch_bounds__(kThreads, 1) void n2_phase1_kernel(
    const std::uint8_t* __restrict__ A_bytes,   // input  [R, 7168] FP8
    const float* __restrict__ A_scale,          // input_scale, group-major live
    const std::uint8_t* __restrict__ W13,       // gate   [32, 4096, 7168] AITER
    const float* __restrict__ S13,              // fc1_scale [32, 32, 56]
    const int* __restrict__ sorted_ids,         // packed: token|slot<<24
    const int* __restrict__ sorted_eid,
    const int* __restrict__ nvi,                // [padded_rows, T]
    std::uint8_t* __restrict__ A2q,             // OUT [rowcap, 2048] FP8
    float* __restrict__ DQ2) {                  // OUT [rowcap, 16] FP32
  __shared__ int tok_lds[kBlockM];
  __shared__ float ascale_lds[kKGroups][kBlockM];  // [k128][sorted row]
  __shared__ float b1s_lds[2][2][kKGroups];        // [gate/up][n128 half][k128]
  __shared__ float amax_lds[2][kBlockM];           // [kb][sorted row]
  __shared__ __align__(16) std::uint8_t a2_lds[kBlockM][kA2Stride];

  // The CTA-shared, double-buffered A tile.
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kBlockM][kAChunks][kAChunkBytes];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;   // wave 0..3
  const int q = lane >> 4;   // 0..3
  const int r = lane & 15;   // 0..15
  const int T = nvi[1];      // live token count -- read ON DEVICE

  // The work count, read ON DEVICE.
  const int num_tasks = (((nvi[0] + (kBlockM - 1)) >> 5) << 3);

  // THE BOUNDED INPUT DESCRIPTOR.
  const std::uint32_t a_num_records =
      static_cast<std::uint32_t>(T) * static_cast<std::uint32_t>(kHidden);
  const i32x4 a_rsrc = make_srsrc(A_bytes, a_num_records, 0);

  const int a_row = tid >> 3;              // 0..31
  const int a_chunk = tid & 7;             // 0..7
  const int a_cp = a_chunk ^ (a_row & 7);  // the XOR swizzle

  const int j_w = wv >> 1;

  for (int task = blockIdx.x; task < num_tasks; task += kCTAs) {
    const int b = task >> 3;   // sorted block
    const int g = task & 7;    // intermediate chunk
    const int e = sorted_eid[b];

    // ---- per-task LDS setup (sorted_weights is not needed here) -------------
    if (tid < kBlockM) {
      tok_lds[tid] = sorted_ids[b * kBlockM + tid] & 0x00FFFFFF;
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
    __syncthreads();

    // input_scale live prefix is GROUP-MAJOR: input_scale[k128 * T + token].
    for (int idx = tid; idx < kKGroups * kBlockM; idx += kThreads) {
      const int k = idx >> 5;
      const int i = idx & 31;
      const int token = tok_lds[i];
      ascale_lds[k][i] = (token < T)
                             ? A_scale[static_cast<std::size_t>(k) * T + token]
                             : 0.0f;
    }
    __syncthreads();

    const std::uint32_t a_voff0 =
        static_cast<std::uint32_t>(tok_lds[a_row]) *
            static_cast<std::uint32_t>(kHidden) +
        static_cast<std::uint32_t>(a_chunk * kAChunkBytes);

    // =======================================================================
    // GEMM-1 : PAIRED gate/up.
    // =======================================================================
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

    rfp8 af[2];
    __uint128_t aTmp[2];
    rfp8 bf[2][2][kColTiles];

    auto load_a = [&](int k, int buf) __attribute__((always_inline)) {
      aTmp[buf] = llvm_amdgcn_raw_buffer_load_b128(
          a_rsrc, a_voff0 + static_cast<std::uint32_t>(k * 128), 0u, 0u);
    };
    auto store_a = [&](int buf, int lb) __attribute__((always_inline)) {
      *reinterpret_cast<__uint128_t*>(&a_lds[lb][a_row][a_cp][0]) = aTmp[buf];
    };
    auto read_a = [&](int lb) __attribute__((always_inline)) {
      const int xr = r & 7;
      const int cp0 = q ^ xr;
      const int cp1 = cp0 ^ 4;
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        const int row = 16 * m + r;
        float4* d = reinterpret_cast<float4*>(&af[m].tiles[0][0].data[0]);
        d[0] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp0][0]);
        d[1] = *reinterpret_cast<const float4*>(&a_lds[lb][row][cp1][0]);
      }
    };
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
        stage_agpr(bf[buf][0][c]);
        stage_agpr(bf[buf][1][c]);
      }
    };

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
            const f32x2* pf = accv(p);
            f32x2* cf = accv(acc[gu][m][c]);
            cf[0] += pf[0] * s0;
            cf[1] += pf[1] * s1;
          }
        }
      }
    };

    // ---- the pipeline.
    load_a(0, 0);
    load_a(1, 1);
    load_b(0, 0);
    store_a(0, 0);
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroups; k += 2) {
      const int ka2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);
      const int ka3 = (k + 3 < kKGroups) ? (k + 3) : (kKGroups - 1);
      const int kb1 = (k + 1 < kKGroups) ? (k + 1) : (kKGroups - 1);
      const int kb2 = (k + 2 < kKGroups) ? (k + 2) : (kKGroups - 1);

      // ---- half 0: step k.   LDS buf 0, B buf 0. -------------------------
      read_a(0);
      load_a(ka2, 0);
      load_b(kb1, 1);
      mfma_k(k, 0);
      store_a(1, 1);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();

      // ---- half 1: step k+1.  LDS buf 1, B buf 1. ------------------------
      read_a(1);
      load_a(ka3, 1);
      load_b(kb2, 0);
      mfma_k(kb1, 1);
      store_a(0, 0);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 17, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 16, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();
    }

    // =======================================================================
    // REGISTER SiLU + DYNAMIC per-(row, K128) amax + FP8 quant.
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

    // THE EPSILON IS NOT OPTIONAL (rank 2 has empty experts; 448/0 = NaN).
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
    // THE ONE VARIABLE (phase-1 half): A2 -> GLOBAL, coalesced, in place of the
    // split-K GEMM-2.  a2_lds[row][col] -> A2q[(b*32+row)*2048 + 256g + col].
    // 256 threads x 32 B = the 8 KiB tile; 16 B vectors, fully coalesced.
    // =======================================================================
    {
      const int cr = tid >> 3;   // row 0..31
      const int cc = tid & 7;    // 32-B chunk 0..7
      const uint4 s0 =
          *reinterpret_cast<const uint4*>(&a2_lds[cr][32 * cc]);
      const uint4 s1 =
          *reinterpret_cast<const uint4*>(&a2_lds[cr][32 * cc + 16]);
      const std::size_t dst =
          (static_cast<std::size_t>(b) * kBlockM + cr) * kInter +
          kChunkCols * g + 32 * cc;
      *reinterpret_cast<uint4*>(A2q + dst) = s0;
      *reinterpret_cast<uint4*>(A2q + dst + 16) = s1;
    }
    // DQ2[row, 2g+kb] = max(amax,1e-6)/448 — one lane per row writes (q==0).
    if (q == 0) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int kb = 0; kb < 2; ++kb) {
          DQ2[(static_cast<std::size_t>(b) * kBlockM + 16 * m + r) * kKGroups2 +
              2 * g + kb] =
              fmaxf(amax_lds[kb][16 * m + r], 1.0e-6f) / 448.0f;
        }
      }
    }
    // The next task refills every LDS buffer this one just read.
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host side.  Production tensors + the two scratch handoff buffers (allocated
// once at setup by the caller; no host work inside the measured region).
// input capacity is a RUNTIME property (device-driven walk) — only column
// counts are contract-checked, so the same module serves decode and prefill.
// ---------------------------------------------------------------------------
struct n2_phase1_globals {
  tensor_bf16 input_bytes;        // BF16 view of production `input`
  tensor_fp32 input_scale;        // production `input_scale`
  tensor_bf16 w13_bytes;          // BF16 view of production `gate` (AITER)
  tensor_fp32 w13_scales;         // production `fc1_scale`
  tensor_i32 sorted_token_ids;
  tensor_i32 sorted_expert_ids;
  tensor_i32 num_valid_ids;
  tensor_bf16 a2q_bytes;          // scratch [rowcap, 1024] BF16-view (=2048 B rows)
  tensor_fp32 dq2;                // scratch [rowcap, 16] FP32
  std::uintptr_t stream_ptr;
};

[[noreturn]] static void invalid(const char* message) {
  throw std::invalid_argument(std::string("n2_phase1: ") + message);
}

static void require(bool condition, const char* message) {
  if (!condition) {
    invalid(message);
  }
}

void validate_contract(const n2_phase1_globals& g) {
  require(aiter::is_supported_blockscale_shape(kExperts, kW13N, kHidden),
          "W13 AITER shape unsupported");
  require(g.input_bytes.batch() == 1 && g.input_bytes.depth() == 1 &&
              g.input_bytes.cols() == kHidden / 2,
          "input must have BF16-view cols 3584");
  require(g.input_scale.batch() == 1 && g.input_scale.depth() == 1 &&
              g.input_scale.cols() == kKGroups,
          "input_scale must have FP32 cols 56");
  require(g.w13_bytes.batch() == 1 && g.w13_bytes.depth() == 1 &&
              g.w13_bytes.rows() == kExperts * kW13N &&
              g.w13_bytes.cols() == kHidden / 2,
          "gate must have BF16-view shape [32*4096,3584]");
  require(g.w13_scales.batch() == 1 && g.w13_scales.depth() == kExperts &&
              g.w13_scales.rows() == kNGroups13 &&
              g.w13_scales.cols() == kKGroups,
          "fc1_scale must have FP32 shape [32,32,56]");
  require(g.num_valid_ids.cols() == 2, "num_valid_ids must hold two values");
  require(g.a2q_bytes.cols() == kInter / 2,
          "a2q must have BF16-view cols 1024 (=2048 B rows)");
  require(g.dq2.cols() == kKGroups2, "dq2 must have FP32 cols 16");
  require(g.sorted_token_ids.cols() / kBlockM <= g.sorted_expert_ids.cols(),
          "sorted_token_ids capacity implies more blocks than sorted_expert_ids "
          "can name");
  require(static_cast<std::size_t>(g.a2q_bytes.rows()) >=
              static_cast<std::size_t>(g.sorted_token_ids.cols()),
          "a2q must cover the sorted row capacity");
}

void dispatch_n2_phase1(n2_phase1_globals g) {
  validate_contract(g);
  hipStream_t stream = reinterpret_cast<hipStream_t>(g.stream_ptr);
  n2_phase1_kernel<<<kCTAs, kThreads, 0, stream>>>(
      reinterpret_cast<const std::uint8_t*>(g.input_bytes.raw_ptr),
      g.input_scale.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w13_bytes.raw_ptr),
      g.w13_scales.raw_ptr, g.sorted_token_ids.raw_ptr,
      g.sorted_expert_ids.raw_ptr, g.num_valid_ids.raw_ptr,
      reinterpret_cast<std::uint8_t*>(g.a2q_bytes.raw_ptr),
      g.dq2.raw_ptr);
}

}  // namespace production_fused_moe::n2

void bind_n2_phase1(pybind11::module_& module) {
  using production_fused_moe::n2::dispatch_n2_phase1;
  using production_fused_moe::n2::n2_phase1_globals;
  py::bind_function<dispatch_n2_phase1>(
      module, "n2_phase1",
      &n2_phase1_globals::input_bytes,
      &n2_phase1_globals::input_scale,
      &n2_phase1_globals::w13_bytes,
      &n2_phase1_globals::w13_scales,
      &n2_phase1_globals::sorted_token_ids,
      &n2_phase1_globals::sorted_expert_ids,
      &n2_phase1_globals::num_valid_ids,
      &n2_phase1_globals::a2q_bytes,
      &n2_phase1_globals::dq2,
      &n2_phase1_globals::stream_ptr);
}
