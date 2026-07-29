// Prefill-scale deterministic plan construction.
// Token offsets use a three-level scan over 2048-entry chunks. The pull plan uses a fanout scan
// followed by a fill. Padding is bounded by device-computed nvi[0]. tof_part stores per-rank chunk
// partials and supports at most 16 chunks per rank.
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <hip/hip_fp8.h>
#include <cstdint>

using namespace kittens;
using fp8_t = __hip_fp8_storage_t;

#define E12_QGROUP   128
#define E12_BLOCK_M  32
#define E12_MAXE     64
#define E12_CHUNK    256
#define K0PF_TOF_ITEMS 2048          // per-L1-block chunk of the own[p] scan
#define K0PF_TOF_MAXCHUNK 16         // ceil(32768 / 2048) — world*K0PF_TOF_MAXCHUNK <= tof_part cap

struct plan_globals_k0pf {
    // --- in ---
    gl<int,   -1, -1, -1, -1> all_ids;    // [world*T*TOPK, 1] GLOBAL expert ids
    gl<float, -1, -1, -1, -1> all_wgt;    // [world*T*TOPK, 1] GLOBAL route weights
    // --- scratch ---
    gl<int,   -1, -1, -1, -1> own;        // [world, N]
    gl<int,   -1, -1, -1, -1> tof;        // [world, N]
    gl<int,   -1, -1, -1, -1> tloc;       // [world, 1]
    gl<int,   -1, -1, -1, -1> tof_part;   // [world*16, 1]  L1 chunk partials (NEW vs e12)
    gl<int,   -1, -1, -1, -1> cnt;        // [E, 1]
    gl<int,   -1, -1, -1, -1> erb;        // [E+1, 1]
    gl<int,   -1, -1, -1, -1> ccnt;       // [NCHUNK, E]
    // --- output ---
    gl<int,   -1, -1, -1, -1> gath;
    gl<int,   -1, -1, -1, -1> sorted_token_ids;
    gl<float, -1, -1, -1, -1> sorted_weights;
    gl<int,   -1, -1, -1, -1> sorted_expert_ids;
    gl<int,   -1, -1, -1, -1> num_valid_ids;
    // --- combine pull plan ---
    gl<int,   -1, -1, -1, -1> pull_ptr;
    gl<int,   -1, -1, -1, -1> pull_src;
    // --- capacity errors ---
    gl<int,   -1, -1, -1, -1> err;
    int world, T, TOPK, E, cur_rank, T_loc_max, PADMAX;
    uint64_t stream_handle;  // explicit capture stream for build_plan_prod_stream
    hipStream_t stream;
    dim3 grid()  { return dim3(1); }
    dim3 block() { return dim3(256); }
};

// --- reset ------------------------------------------------------------------------
__global__ void e12_reset_kernel(plan_globals_k0pf g) {
    const int stride = gridDim.x * blockDim.x;
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < g.E; i += stride) g.cnt[{0, 0, i, 0}] = 0;
    const int NCHUNK = (g.world * g.T * g.TOPK + E12_CHUNK - 1) / E12_CHUNK;
    for (int i = tid; i < NCHUNK * g.E; i += stride) g.ccnt[{0, 0, i, 0}] = 0;
    for (int i = tid; i < g.world; i += stride)      g.tloc[{0, 0, i, 0}] = 0;
    if (tid == 0) g.err[{0, 0, 0, 0}] = 0;
}

// --- own[p][gtok] -----------------------------------------------------------------
__global__ void e12_own_kernel(plan_globals_k0pf g) {
    const int N   = g.world * g.T;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g.world * N) return;
    const int p    = idx / N;
    const int gtok = idx % N;
    const int lo = p * g.E, hi = lo + g.E;

    int o = 0;
    const int* ids = &g.all_ids[{0, 0, 0, 0}];
    for (int s = 0; s < g.TOPK; ++s) {
        const int e = ids[gtok * g.TOPK + s];
        if (e >= lo && e < hi) { o = 1; break; }
    }
    g.own[{0, 0, idx, 0}] = o;
}

// --- two-level tof scan --------------------------------------------------------------------------
// Inclusive block scan helper (Hillis-Steele, deterministic).
__device__ __forceinline__ int e26_block_incl_scan(int v, int* tmp) {
    const int tid = threadIdx.x;
    tmp[tid] = v;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {
        const int add = (tid >= off) ? tmp[tid - off] : 0;
        __syncthreads();
        tmp[tid] += add;
        __syncthreads();
    }
    const int r = tmp[tid];
    __syncthreads();
    return r;
}

// L1: exclusive prefix within each 2048-entry rank chunk.
__global__ void k0pf_tof_l1_kernel(plan_globals_k0pf g) {
    const int N   = g.world * g.T;
    const int ntc = (N + K0PF_TOF_ITEMS - 1) / K0PF_TOF_ITEMS;   // chunks per rank (<= 16)
    if (ntc > K0PF_TOF_MAXCHUNK) {                               // tof_part capacity guard
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(&g.err[{0, 0, 0, 0}], 16);
        return;
    }
    const int p   = blockIdx.x / ntc;
    const int c   = blockIdx.x % ntc;
    if (p >= g.world) return;

    const int base0 = c * K0PF_TOF_ITEMS;
    const int cend  = (base0 + K0PF_TOF_ITEMS < N) ? (base0 + K0PF_TOF_ITEMS) : N;
    const int* own = &g.own[{0, 0, 0, 0}] + (size_t)p * N;
    int*       tof = &g.tof[{0, 0, 0, 0}] + (size_t)p * N;
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    if (threadIdx.x == 0) s_carry = 0;
    __syncthreads();
    for (int base = base0; base < cend; base += blockDim.x) {
        const int i    = base + threadIdx.x;
        const int v    = (i < N) ? own[i] : 0;
        const int incl = e26_block_incl_scan(v, s_tmp);
        if (i < N) tof[i] = s_carry + (incl - v);
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) s_carry += incl;
        __syncthreads();
    }
    if (threadIdx.x == blockDim.x - 1) g.tof_part[{0, 0, p * ntc + c, 0}] = s_carry;
}

// L2: one block per rank — exclusive scan of that rank's ntc chunk partials (one tile), and tloc.
__global__ void k0pf_tof_l2_kernel(plan_globals_k0pf g) {
    const int N   = g.world * g.T;
    const int ntc = (N + K0PF_TOF_ITEMS - 1) / K0PF_TOF_ITEMS;
    if (ntc > K0PF_TOF_MAXCHUNK) return;                   // L1 already raised err bit 16
    const int p   = blockIdx.x;
    if (p >= g.world) return;
    int* part = &g.tof_part[{0, 0, p * ntc, 0}];
    __shared__ int s_tmp[256];
    const int i = threadIdx.x;
    const int v = (i < ntc) ? part[i] : 0;
    const int incl = e26_block_incl_scan(v, s_tmp);
    if (i < ntc) part[i] = incl - v;                       // exclusive chunk base, in place
    if (i == ntc - 1) g.tloc[{0, 0, p, 0}] = incl;         // rank total = last inclusive
}

// L3: one block per (rank, chunk) — add the chunk base to every entry of the chunk.
__global__ void k0pf_tof_l3_kernel(plan_globals_k0pf g) {
    const int N   = g.world * g.T;
    const int ntc = (N + K0PF_TOF_ITEMS - 1) / K0PF_TOF_ITEMS;
    if (ntc > K0PF_TOF_MAXCHUNK) return;                   // L1 already raised err bit 16
    const int p   = blockIdx.x / ntc;
    const int c   = blockIdx.x % ntc;
    if (p >= g.world || c == 0) return;                    // chunk 0 already has base 0
    const int base_add = g.tof_part[{0, 0, p * ntc + c, 0}];
    const int i0 = c * K0PF_TOF_ITEMS;
    const int i1 = (i0 + K0PF_TOF_ITEMS < N) ? (i0 + K0PF_TOF_ITEMS) : N;
    int* tof = &g.tof[{0, 0, 0, 0}] + (size_t)p * N;
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x)
        tof[i] += base_add;
}

// --- gath[t] = (src_rank, src_token) ----------------------------------------------
__global__ void e12_gath_kernel(plan_globals_k0pf g) {
    const int N   = g.world * g.T;
    const int gtok = blockIdx.x * blockDim.x + threadIdx.x;
    const int p   = g.cur_rank;

    if (gtok == 0) g.num_valid_ids[{0, 0, 1, 0}] = g.tloc[{0, 0, p, 0}];
    if (gtok >= N) return;

    const int* own = &g.own[{0, 0, 0, 0}] + (size_t)p * N;
    const int* tof = &g.tof[{0, 0, 0, 0}] + (size_t)p * N;
    if (!own[gtok]) return;

    const int t = tof[gtok];
    if (t >= g.T_loc_max) { atomicOr(&g.err[{0, 0, 0, 0}], 1); return; }
    g.gath[{0, 0, t, 0}] = gtok / g.T;
    g.gath[{0, 0, t, 1}] = gtok % g.T;
}

// --- per-expert counts -------------------------------------------------------------
__global__ void e12_count_kernel(plan_globals_k0pf g) {
    const int n  = g.world * g.T * g.TOPK;
    const int lo = g.cur_rank * g.E, hi = lo + g.E;
    const int* ids = &g.all_ids[{0, 0, 0, 0}];
    int* cnt  = &g.cnt[{0, 0, 0, 0}];
    int* ccnt = &g.ccnt[{0, 0, 0, 0}];
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const int e = ids[j];
        if (e < lo || e >= hi) continue;
        const int le = e - lo;
        atomicAdd(cnt + le, 1);
        atomicAdd(ccnt + (j / E12_CHUNK) * g.E + le, 1);
    }
}

// --- 32-padded expert prefix and per-chunk bases -----------------------------------------------
__global__ void e12_scan_kernel(plan_globals_k0pf g) {
    __shared__ int s_erb[E12_MAXE + 1];
    const int E      = g.E;
    const int NCHUNK = (g.world * g.T * g.TOPK + E12_CHUNK - 1) / E12_CHUNK;
    if (E > E12_MAXE) { if (threadIdx.x == 0) atomicOr(&g.err[{0, 0, 0, 0}], 4); return; }

    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < E; ++e) {
            s_erb[e] = acc;
            g.erb[{0, 0, e, 0}] = acc;
            const int c = g.cnt[{0, 0, e, 0}];
            acc += ((c + E12_BLOCK_M - 1) / E12_BLOCK_M) * E12_BLOCK_M;
        }
        s_erb[E] = acc;
        g.erb[{0, 0, E, 0}] = acc;
        g.num_valid_ids[{0, 0, 0, 0}] = acc;
    }
    __syncthreads();

    // Bounds-check before writing sorted_expert_ids.
    const int bcap = g.PADMAX / E12_BLOCK_M;
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int b0 = s_erb[e] / E12_BLOCK_M;
        const int b1 = s_erb[e + 1] / E12_BLOCK_M;
        for (int b = b0; b < b1; ++b) {
            if (b >= bcap) { atomicOr(&g.err[{0, 0, 0, 0}], 2); break; }
            g.sorted_expert_ids[{0, 0, b, 0}] = e;
        }
    }

    // Convert per-chunk counts to deterministic expert-relative prefixes.
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        int acc = s_erb[e];
        for (int c = 0; c < NCHUNK; ++c) {
            const int v = g.ccnt[{0, 0, c * E + e, 0}];
            g.ccnt[{0, 0, c * E + e, 0}] = acc;
            acc += v;
        }
    }
}

// --- padding sentinel -------------------------------------------------------------------------
// Initialize only rows below device-computed nvi[0].
__global__ void e12_pad_kernel(plan_globals_k0pf g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.PADMAX) return;
    const int padded = g.num_valid_ids[{0, 0, 0, 0}];
    if (i >= padded) return;
    const int T_loc = g.num_valid_ids[{0, 0, 1, 0}];
    g.sorted_token_ids[{0, 0, i, 0}] = (T_loc & 0x00FFFFFF) | (g.TOPK << 24);
    g.sorted_weights [{0, 0, i, 0}] = 0.f;
}

// --- ordered compaction -----------------------------------------------------------------------
__global__ void e12_sorted_kernel(plan_globals_k0pf g) {
    __shared__ int s_cur[E12_MAXE];
    const int c  = blockIdx.x;
    const int n  = g.world * g.T * g.TOPK;
    const int E  = g.E;
    const int lo = g.cur_rank * g.E, hi = lo + g.E;
    const int N  = g.world * g.T;

    const int j0 = c * E12_CHUNK;
    if (j0 >= n) return;

    for (int e = threadIdx.x; e < E; e += blockDim.x)
        s_cur[e] = g.ccnt[{0, 0, c * E + e, 0}];
    __syncthreads();

    __shared__ int s_le[E12_CHUNK];
    const int*   ids = &g.all_ids[{0, 0, 0, 0}];
    const int j = j0 + threadIdx.x;
    int le = -1;
    if (j < n) {
        const int e = ids[j];
        if (e >= lo && e < hi) le = e - lo;
    }
    s_le[threadIdx.x] = le;
    __syncthreads();

    if (le >= 0) {
        int rank = 0;
        for (int u = 0; u < threadIdx.x; ++u) rank += (s_le[u] == le);
        const int row = s_cur[le] + rank;
        if (row >= g.PADMAX) { atomicOr(&g.err[{0, 0, 0, 0}], 2); }
        else {
            const float* wgs = &g.all_wgt[{0, 0, 0, 0}];
            const int*   tof = &g.tof[{0, 0, 0, 0}] + (size_t)g.cur_rank * N;
            const int gtok = j / g.TOPK;
            const int slot = j % g.TOPK;
            const int t    = tof[gtok];
            g.sorted_token_ids[{0, 0, row, 0}] = (t & 0x00FFFFFF) | (slot << 24);
            g.sorted_weights  [{0, 0, row, 0}] = wgs[j];
        }
    }
}

// --- combine pull plan -------------------------------------------------------------------------
// Scan per-token fanout into pull_ptr.
__global__ void k0pf_pscan_kernel(plan_globals_k0pf g) {
    const int N = g.world * g.T;
    const int W = g.world;
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    if (threadIdx.x == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < g.T; base += blockDim.x) {
        const int tau = base + threadIdx.x;
        int f = 0;
        if (tau < g.T) {
            const int gtok = g.cur_rank * g.T + tau;
            for (int p = 0; p < W; ++p)                          // runtime bound: own[] is [world, N]
                f += g.own[{0, 0, p * N + gtok, 0}];
        }
        const int incl = e26_block_incl_scan(f, s_tmp);
        if (tau < g.T) g.pull_ptr[{0, 0, tau, 0}] = s_carry + (incl - f);
        if (tau == g.T - 1)                                      // last token: write pull_ptr[T]
            g.pull_ptr[{0, 0, g.T, 0}] = s_carry + incl;
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) s_carry += incl;
        __syncthreads();
    }
}

// Emit producer rank and row in rank order.
__global__ void k0pf_pfill_kernel(plan_globals_k0pf g) {
    const int tau = blockIdx.x * blockDim.x + threadIdx.x;
    if (tau >= g.T) return;
    const int N = g.world * g.T;
    const int W = g.world;
    const int gtok = g.cur_rank * g.T + tau;
    int k = g.pull_ptr[{0, 0, tau, 0}];
    for (int p = 0; p < W; ++p) {
        if (!g.own[{0, 0, p * N + gtok, 0}]) continue;
        if (k >= g.T * W) { atomicOr(&g.err[{0, 0, 0, 0}], 8); break; }
        g.pull_src[{0, 0, k, 0}] = p;
        g.pull_src[{0, 0, k, 1}] = g.tof[{0, 0, p * N + gtok, 0}];
        ++k;
    }
}

void dispatch_build_plan_prod(plan_globals_k0pf g) {
    if (g.stream_handle)
        g.stream = reinterpret_cast<hipStream_t>(g.stream_handle);
    const int n = g.world * g.T * g.TOPK;
    const int N = g.world * g.T;
    if (n <= 0 || g.E <= 0 || g.PADMAX <= 0) return;
    const int NCHUNK = (n + E12_CHUNK - 1) / E12_CHUNK;
    const int ntc = (N + K0PF_TOF_ITEMS - 1) / K0PF_TOF_ITEMS;
    auto nb = [](int x) { return (x + 255) / 256; };

    e12_reset_kernel <<<nb(NCHUNK * g.E + g.E + g.world), 256, 0, g.stream>>>(g);
    e12_own_kernel   <<<nb(g.world * N),                   256, 0, g.stream>>>(g);
    k0pf_tof_l1_kernel<<<g.world * ntc,                    256, 0, g.stream>>>(g);
    k0pf_tof_l2_kernel<<<g.world,                          256, 0, g.stream>>>(g);
    k0pf_tof_l3_kernel<<<g.world * ntc,                    256, 0, g.stream>>>(g);
    e12_gath_kernel  <<<nb(N),                             256, 0, g.stream>>>(g);
    e12_count_kernel <<<nb(n),                             256, 0, g.stream>>>(g);
    e12_scan_kernel  <<<1,                                 256, 0, g.stream>>>(g);
    e12_pad_kernel   <<<nb(g.PADMAX),                      256, 0, g.stream>>>(g);   // device-bounded
    e12_sorted_kernel<<<NCHUNK,                            256, 0, g.stream>>>(g);
    k0pf_pscan_kernel<<<1,                                 256, 0, g.stream>>>(g);
    k0pf_pfill_kernel<<<nb(g.T),                           256, 0, g.stream>>>(g);
}

struct zero_globals_k0pf {
    gl<bf16, -1, -1, -1, -1> buf;              // [T_loc_max, H]
    gl<int,  -1, -1, -1, -1> num_valid_ids;    // [2,1]; [1] = T_loc
    int T_loc_max, H;
    hipStream_t stream;
    dim3 grid()  { return dim3(T_loc_max > 0 ? T_loc_max : 1); }
    dim3 block() { return dim3(256); }
};

// Already T_loc-bounded on device (grid is capacity, work exits past T_loc).
__global__ void zero_partial_kernel(zero_globals_k0pf g) {
    const int t     = blockIdx.x;
    const int T_loc = g.num_valid_ids[{0, 0, 1, 0}];
    if (t >= T_loc || t >= g.T_loc_max) return;
    bf16* row = reinterpret_cast<bf16*>(&g.buf[{0, 0, 0, 0}]) + (size_t)t * g.H;
    const int Hv = g.H >> 3;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int hv = threadIdx.x; hv < Hv; hv += blockDim.x)
        *reinterpret_cast<uint4*>(row + (hv << 3)) = z;
}

void dispatch_zero_partial(zero_globals_k0pf g) {
    if (g.T_loc_max <= 0) return;
    zero_partial_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
PYBIND11_MODULE(k0pf_plan, m) {
    m.doc() = "Prefill plan: same plan math, with the pull scan and the token-offset scan "
              "rebuilt for T=4096 scale. Adds tof_part scratch after tloc.";
    py::bind_function<dispatch_build_plan_prod>(m, "build_plan_prod",
        &plan_globals_k0pf::all_ids, &plan_globals_k0pf::all_wgt,
        &plan_globals_k0pf::own, &plan_globals_k0pf::tof, &plan_globals_k0pf::tloc,
        &plan_globals_k0pf::tof_part,
        &plan_globals_k0pf::cnt, &plan_globals_k0pf::erb, &plan_globals_k0pf::ccnt,
        &plan_globals_k0pf::gath,
        &plan_globals_k0pf::sorted_token_ids, &plan_globals_k0pf::sorted_weights,
        &plan_globals_k0pf::sorted_expert_ids, &plan_globals_k0pf::num_valid_ids,
        &plan_globals_k0pf::pull_ptr, &plan_globals_k0pf::pull_src, &plan_globals_k0pf::err,
        &plan_globals_k0pf::world, &plan_globals_k0pf::T, &plan_globals_k0pf::TOPK,
        &plan_globals_k0pf::E, &plan_globals_k0pf::cur_rank,
        &plan_globals_k0pf::T_loc_max, &plan_globals_k0pf::PADMAX);
    py::bind_function<dispatch_build_plan_prod>(m, "build_plan_prod_stream",
        &plan_globals_k0pf::all_ids, &plan_globals_k0pf::all_wgt,
        &plan_globals_k0pf::own, &plan_globals_k0pf::tof, &plan_globals_k0pf::tloc,
        &plan_globals_k0pf::tof_part,
        &plan_globals_k0pf::cnt, &plan_globals_k0pf::erb, &plan_globals_k0pf::ccnt,
        &plan_globals_k0pf::gath,
        &plan_globals_k0pf::sorted_token_ids, &plan_globals_k0pf::sorted_weights,
        &plan_globals_k0pf::sorted_expert_ids, &plan_globals_k0pf::num_valid_ids,
        &plan_globals_k0pf::pull_ptr, &plan_globals_k0pf::pull_src, &plan_globals_k0pf::err,
        &plan_globals_k0pf::world, &plan_globals_k0pf::T, &plan_globals_k0pf::TOPK,
        &plan_globals_k0pf::E, &plan_globals_k0pf::cur_rank,
        &plan_globals_k0pf::T_loc_max, &plan_globals_k0pf::PADMAX,
        &plan_globals_k0pf::stream_handle);
    py::bind_function<dispatch_zero_partial>(m, "zero_partial",
        &zero_globals_k0pf::buf, &zero_globals_k0pf::num_valid_ids,
        &zero_globals_k0pf::T_loc_max, &zero_globals_k0pf::H);
}
