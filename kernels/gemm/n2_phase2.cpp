// Phase 2 of the two-phase expert GEMM: split-N full-K W2.
// Each task owns a 32x448 output tile and reduces all 16 K128 groups. A2 uses a bounded,
// double-buffered LDS tile; W2 streams from global memory. The epilogue multiplies route weights,
// packs BF16, transposes wave-locally, and masks unbounded output atomics with token<T.
// Mapping n-chunk modulo 8 keeps an expert's W2 slices on a stable XCD.

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

constexpr int kExperts = 32;
constexpr int kHidden = 7168;    // N of W2 (output dim)
constexpr int kInter = 2048;     // K of W2 (intermediate dim)
constexpr int kBlockM = 32;

constexpr int kNChunks = 16;                   // output n-chunks per block
constexpr int kChunkN = kHidden / kNChunks;    // 448 output cols per task
constexpr int kWaves = 4;
constexpr int kThreads = kWaves * 64;          // 256
constexpr int kCTAs = 256;
constexpr int kWaveTiles = kChunkN / 16 / kWaves;  // 7 N16 tiles per wave
constexpr int kWaveCols = 16 * kWaveTiles;         // 112 cols per wave

constexpr int kKGroups2 = kInter / 128;        // 16 K128 groups in W2
constexpr int kNGroups2 = kHidden / 128;       // 56 fc2_scale n-groups
constexpr int kTaskTiles = kChunkN / 16;       // 28 N16 tiles per task

constexpr int kAChunks = 8;                    // 128 B per row / 16 B per thread
constexpr int kAChunkBytes = 16;

static_assert(kKGroups2 % 2 == 0, "the K loop is unrolled by two");
static_assert(kChunkN % (16 * kWaves) == 0, "N chunk / wave split");
static_assert(kNChunks % 8 == 0, "nc -> XCD (nc mod 8) must be stable");
static_assert(kThreads == kBlockM * kAChunks,
              "the A2 tile fill must be exactly one 16 B load per thread");
static_assert(kAChunks * kAChunkBytes == 128, "one K128 group per A2 tile row");

using tensor_bf16 = gl<bf16, -1, -1, -1, -1>;
using tensor_fp32 = gl<float, -1, -1, -1, -1>;
using tensor_i32 = gl<int, -1, -1, -1, -1>;

using rfp8 = rt_fp8e4m3<16, 128>;
using racc = rt_fl<16, 16, col_l, rt_16x16_s>;

static_assert(rfp8::packed_per_thread == 8);   // 32 bytes / lane
static_assert(racc::packed_per_thread == 2);   // 4 floats / lane

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

__device__ __forceinline__ void lds_cta_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Write JMAX N16 tiles through a wave-local transpose.
template <int JMAX>
__device__ __forceinline__ void epilogue_write(
    std::uint32_t (&xp)[kBlockM][32], racc (&acc)[JMAX][2],
    const float (&sw2)[2], const bool (&lv2)[2], const int (&xtok)[16], int T,
    int q, int r, int rowh, int dcol, std::size_t col_base,
    __hip_bfloat16* __restrict__ OUT) {
  // Write MFMA layout; lane 16q+r holds rows {r,16+r}.
#pragma unroll
  for (int m = 0; m < 2; ++m) {
    const int row = 16 * m + r;
    const float sw = lv2[m] ? sw2[m] : 0.0f;
    const int xorm = 2 * (row & 15);
#pragma unroll
    for (int j = 0; j < JMAX; ++j) {
      const float* cf = accf(acc[j][m]);
      __hip_bfloat162 v0, v1;
      v0.x = __float2bfloat16(cf[0] * sw);
      v0.y = __float2bfloat16(cf[1] * sw);
      v1.x = __float2bfloat16(cf[2] * sw);
      v1.y = __float2bfloat16(cf[3] * sw);
      const int phys = (8 * j + 2 * q) ^ xorm;
      uint2 packed;
      packed.x = *reinterpret_cast<const std::uint32_t*>(&v0);
      packed.y = *reinterpret_cast<const std::uint32_t*>(&v1);
      *reinterpret_cast<uint2*>(&xp[row][phys]) = packed;
    }
  }
  // The transpose is wave-local; no CTA barrier is needed.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
  __builtin_amdgcn_s_waitcnt(0xc07f);  // lgkmcnt(0); vmcnt/expcnt free
  __builtin_amdgcn_wave_barrier();

  // Read atomic layout. token<T prevents address formation for padding rows.
  if (dcol < 8 * JMAX) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int row = 2 * i + rowh;
      const std::uint32_t d = xp[row][dcol ^ (2 * (row & 15))];
      if (xtok[i] < T) {
        __hip_bfloat162* p = reinterpret_cast<__hip_bfloat162*>(
            OUT + static_cast<std::size_t>(xtok[i]) * kHidden + col_base +
            2 * dcol);
        unsafeAtomicAdd(p, *reinterpret_cast<const __hip_bfloat162*>(&d));
      }
    }
  }
  // WAR: the next pass's ds_write must not overtake these ds_reads.
  __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
}

// Split-N full-K W2.
__global__ __launch_bounds__(kThreads, 1) void n2_phase2_kernel(
    const std::uint8_t* __restrict__ A2q,       // [rowcap, 2048] FP8 (phase 1)
    const float* __restrict__ DQ2,              // [rowcap, 16] FP32 (phase 1)
    const std::uint8_t* __restrict__ W2,        // down [32, 7168, 2048] AITER
    const float* __restrict__ S2,               // fc2_scale [32, 56, 16]
    const int* __restrict__ sorted_ids,         // packed: token|slot<<24
    const float* __restrict__ sorted_w,
    const int* __restrict__ sorted_eid,
    const int* __restrict__ nvi,                // [padded_rows, T]
    __hip_bfloat16* __restrict__ OUT) {         // out [R, 7168] BF16, pre-zeroed
  __shared__ int tok_lds[kBlockM];
  __shared__ float w_lds[kBlockM];
  __shared__ float b2s_lds[kTaskTiles][kKGroups2];   // [28 tiles][kb]
  __shared__ float dq2_lds[kKGroups2][kBlockM];      // [kb][sorted row]
  __shared__ __align__(16)
      std::uint8_t a_lds[2][kBlockM][kAChunks][kAChunkBytes];
  __shared__ __align__(16) std::uint32_t xp_lds[kWaves][kBlockM][32];

  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 63;
  const int wv = tid >> 6;
  const int q = lane >> 4;
  const int r = lane & 15;
  const int T = nvi[1];

  // task = (block b, n-chunk nc), read ON DEVICE.  16 tasks per block.
  const int num_tasks = (((nvi[0] + (kBlockM - 1)) >> 5) << 4);

  // Bounded A2 descriptor: rows < nvi[0] are the only ones phase 2 can name,
  // and phase 1 wrote every one of them this epoch.
  const std::uint32_t a2_num_records =
      static_cast<std::uint32_t>(nvi[0]) * static_cast<std::uint32_t>(kInter);
  const i32x4 a2_rsrc = make_srsrc(A2q, a2_num_records, 0);

  const int a_row = tid >> 3;
  const int a_chunk = tid & 7;
  const int a_cp = a_chunk ^ (a_row & 7);

  const int rowh = lane >> 5;   // 0 or 1: which of the atomic's two rows
  const int dcol = lane & 31;   // which dword (= 2 adjacent output columns)

  for (int task = blockIdx.x; task < num_tasks; task += kCTAs) {
    const int b = task >> 4;     // sorted block
    const int nc = task & 15;    // output n-chunk
    const int e = sorted_eid[b];

    // ---- per-task LDS setup -------------------------------------------------
    if (tid < kBlockM) {
      tok_lds[tid] = sorted_ids[b * kBlockM + tid] & 0x00FFFFFF;
      w_lds[tid] = sorted_w[b * kBlockM + tid];
    }
    // b2s_lds[t][kb] = fc2_scale[e, ng(tile t), kb], tile t in [0,28):
    // ng = (448*nc + 16*t) / 128.
    for (int idx = tid; idx < kTaskTiles * kKGroups2; idx += kThreads) {
      const int t = idx >> 4;
      const int k = idx & 15;
      const int ng = (kChunkN * nc + 16 * t) >> 7;
      b2s_lds[t][k] = S2[(e * kNGroups2 + ng) * kKGroups2 + k];
    }
    // dq2_lds[kb][row] = DQ2[(b*32+row), kb]
    for (int idx = tid; idx < kKGroups2 * kBlockM; idx += kThreads) {
      const int k = idx >> 5;
      const int i = idx & 31;
      dq2_lds[k][i] = DQ2[(static_cast<std::size_t>(b) * kBlockM + i) *
                              kKGroups2 +
                          k];
    }
    __syncthreads();

    // A2 tile fill coordinates: thread t owns sorted row t>>3 and
    // bytes [16*(t&7), +16) of that row's 128 B K128 slice.
    const std::uint32_t a2_voff0 =
        (static_cast<std::uint32_t>(b) * kBlockM + a_row) * kInter +
        static_cast<std::uint32_t>(a_chunk * kAChunkBytes);

    racc acc[kWaveTiles][2];   // [n tile][rowtile] — the full-K accumulator
#pragma unroll
    for (int t = 0; t < kWaveTiles; ++t) {
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        zero(acc[t][m]);
      }
    }

    rfp8 af[2];             // [rowtile] A2(k), read from LDS just in time
    __uint128_t aTmp[2];    // [k&1] A2(k) in flight
    rfp8 bf[2][kWaveTiles]; // [buf][n tile] W2 fragments

    auto load_a2 = [&](int k, int buf) __attribute__((always_inline)) {
      aTmp[buf] = llvm_amdgcn_raw_buffer_load_b128(
          a2_rsrc, a2_voff0 + static_cast<std::uint32_t>(k * 128), 0u, 0u);
    };
    auto store_a2 = [&](int buf, int lb) __attribute__((always_inline)) {
      *reinterpret_cast<__uint128_t*>(&a_lds[lb][a_row][a_cp][0]) = aTmp[buf];
    };
    auto read_a2 = [&](int lb) __attribute__((always_inline)) {
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
    auto load_w2 = [&](int k, int buf) __attribute__((always_inline)) {
#pragma unroll
      for (int t = 0; t < kWaveTiles; ++t) {
        const std::size_t tb = aiter::fp8_weight_byte_offset(
            e, kChunkN * nc + kWaveCols * wv + 16 * t, 128 * k, kHidden,
            kInter);
        load_bfrag(bf[buf][t], W2 + tb + 16 * lane);
        stage_agpr(bf[buf][t]);
      }
    };
    auto mfma_k = [&](int k, int buf) __attribute__((always_inline)) {
      const float sdq0 = dq2_lds[k][r];
      const float sdq1 = dq2_lds[k][16 + r];
#pragma unroll
      for (int t = 0; t < kWaveTiles; ++t) {
        const float fs = b2s_lds[kWaveTiles * wv + t][k];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          racc p;
          zero(p);
          mma_ABt(p, bf[buf][t], af[m], p);
          const float s = fs * (m == 0 ? sdq0 : sdq1);
          const f32x2 sv = {s, s};
          const f32x2* pf = accv(p);
          f32x2* cf = accv(acc[t][m]);
          cf[0] += pf[0] * sv;
          cf[1] += pf[1] * sv;
        }
      }
    };

    // K loop over the full intermediate dimension.
    load_a2(0, 0);
    load_a2(1, 1);
    load_w2(0, 0);
    store_a2(0, 0);
    lds_cta_barrier();

#pragma unroll 1
    for (int k = 0; k < kKGroups2; k += 2) {
      const int ka2 = (k + 2 < kKGroups2) ? (k + 2) : (kKGroups2 - 1);
      const int ka3 = (k + 3 < kKGroups2) ? (k + 3) : (kKGroups2 - 1);
      const int kb1 = (k + 1 < kKGroups2) ? (k + 1) : (kKGroups2 - 1);
      const int kb2 = (k + 2 < kKGroups2) ? (k + 2) : (kKGroups2 - 1);

      // ---- half 0: step k.   LDS buf 0, B buf 0. -------------------------
      read_a2(0);
      load_a2(ka2, 0);
      load_w2(kb1, 1);
      mfma_k(k, 0);
      store_a2(1, 1);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);   //  4 DS read
      __builtin_amdgcn_sched_group_barrier(0x020, 15, 0);  // 15 VMEM read
      __builtin_amdgcn_sched_group_barrier(0x008, 14, 0);  // 14 MFMA
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   //  1 DS write
      lds_cta_barrier();

      // ---- half 1: step k+1.  LDS buf 1, B buf 1. ------------------------
      read_a2(1);
      load_a2(ka3, 1);
      load_w2(kb2, 0);
      mfma_k(kb1, 1);
      store_a2(0, 0);
      __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
      __builtin_amdgcn_sched_group_barrier(0x020, 15, 0);
      __builtin_amdgcn_sched_group_barrier(0x008, 14, 0);
      __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
      lds_cta_barrier();
    }

    // Write the 32x448 tile in two passes.
    const int tok2[2] = {tok_lds[r], tok_lds[16 + r]};
    const float sw2[2] = {w_lds[r], w_lds[16 + r]};
    const bool lv2[2] = {tok2[0] < T, tok2[1] < T};
    int xtok[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      xtok[i] = tok_lds[2 * i + rowh];
    }
    epilogue_write<4>(xp_lds[wv], *reinterpret_cast<racc(*)[4][2]>(&acc[0]),
                      sw2, lv2, xtok, T, q, r, rowh, dcol,
                      static_cast<std::size_t>(kChunkN) * nc + kWaveCols * wv,
                      OUT);
    epilogue_write<3>(xp_lds[wv], *reinterpret_cast<racc(*)[3][2]>(&acc[4]),
                      sw2, lv2, xtok, T, q, r, rowh, dcol,
                      static_cast<std::size_t>(kChunkN) * nc + kWaveCols * wv +
                          64,
                      OUT);
    // The next task refills every LDS buffer this one just read.
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------
struct n2_phase2_globals {
  tensor_bf16 a2q_bytes;          // scratch [rowcap, 1024] BF16-view (=2048 B rows)
  tensor_fp32 dq2;                // scratch [rowcap, 16] FP32
  tensor_bf16 w2_bytes;           // BF16 view of production `down` (AITER)
  tensor_fp32 w2_scales;          // production `fc2_scale`
  tensor_i32 sorted_token_ids;
  tensor_fp32 sorted_weights;
  tensor_i32 sorted_expert_ids;
  tensor_i32 num_valid_ids;
  tensor_bf16 out_bf16;           // production `out`, pre-zeroed in [0,T)
  std::uintptr_t stream_ptr;
};

[[noreturn]] static void invalid(const char* message) {
  throw std::invalid_argument(std::string("n2_phase2: ") + message);
}

static void require(bool condition, const char* message) {
  if (!condition) {
    invalid(message);
  }
}

void validate_contract(const n2_phase2_globals& g) {
  require(aiter::is_supported_blockscale_shape(kExperts, kHidden, kInter),
          "W2 AITER shape unsupported");
  require(g.a2q_bytes.cols() == kInter / 2,
          "a2q must have BF16-view cols 1024 (=2048 B rows)");
  require(g.dq2.cols() == kKGroups2, "dq2 must have FP32 cols 16");
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
              g.out_bf16.cols() == kHidden,
          "out must have BF16 cols 7168");
  require(g.sorted_token_ids.cols() / kBlockM <= g.sorted_expert_ids.cols(),
          "sorted_token_ids capacity implies more blocks than sorted_expert_ids "
          "can name");
  require(g.sorted_weights.cols() == g.sorted_token_ids.cols(),
          "sorted_weights and sorted_token_ids must have equal capacity");
  require(static_cast<std::size_t>(g.a2q_bytes.rows()) >=
              static_cast<std::size_t>(g.sorted_token_ids.cols()),
          "a2q must cover the sorted row capacity");
}

void dispatch_n2_phase2(n2_phase2_globals g) {
  validate_contract(g);
  hipStream_t stream = reinterpret_cast<hipStream_t>(g.stream_ptr);
  n2_phase2_kernel<<<kCTAs, kThreads, 0, stream>>>(
      reinterpret_cast<const std::uint8_t*>(g.a2q_bytes.raw_ptr),
      g.dq2.raw_ptr,
      reinterpret_cast<const std::uint8_t*>(g.w2_bytes.raw_ptr),
      g.w2_scales.raw_ptr, g.sorted_token_ids.raw_ptr,
      g.sorted_weights.raw_ptr, g.sorted_expert_ids.raw_ptr,
      g.num_valid_ids.raw_ptr,
      reinterpret_cast<__hip_bfloat16*>(g.out_bf16.raw_ptr));
}

}  // namespace production_fused_moe::n2

void bind_n2_phase2(pybind11::module_& module) {
  using production_fused_moe::n2::dispatch_n2_phase2;
  using production_fused_moe::n2::n2_phase2_globals;
  py::bind_function<dispatch_n2_phase2>(
      module, "n2_phase2",
      &n2_phase2_globals::a2q_bytes,
      &n2_phase2_globals::dq2,
      &n2_phase2_globals::w2_bytes,
      &n2_phase2_globals::w2_scales,
      &n2_phase2_globals::sorted_token_ids,
      &n2_phase2_globals::sorted_weights,
      &n2_phase2_globals::sorted_expert_ids,
      &n2_phase2_globals::num_valid_ids,
      &n2_phase2_globals::out_bf16,
      &n2_phase2_globals::stream_ptr);
}
