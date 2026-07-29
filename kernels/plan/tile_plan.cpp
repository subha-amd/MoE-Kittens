// Deterministic local plan construction for the pull-based path.
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <hip/hip_fp8.h>

using namespace kittens;
using fp8_t = __hip_fp8_storage_t;

#define E12_QGROUP   128
#define E12_BLOCK_M  32
#define E12_MAXE     64
#define E12_CHUNK    256

struct plan_globals_e12 {
    // --- in ---
    gl<int,   -1, -1, -1, -1> all_ids;    // [world*T*TOPK, 1] GLOBAL expert ids
    gl<float, -1, -1, -1, -1> all_wgt;    // [world*T*TOPK, 1] GLOBAL route weights
    // --- scratch ---
    gl<int,   -1, -1, -1, -1> own;        // [world, N]      own[p][gtok] = 1 if rank p holds an expert of gtok
    gl<int,   -1, -1, -1, -1> tof;        // [world, N]      exclusive prefix of own[p] -> p's LOCAL token index
    gl<int,   -1, -1, -1, -1> tloc;       // [world, 1]      T_loc per rank (= popcount of own[p])
    gl<int,   -1, -1, -1, -1> cnt;        // [E, 1]          rows per local expert
    gl<int,   -1, -1, -1, -1> erb;        // [E+1, 1]        32-padded exclusive prefix; erb[E] = padded_rows
    gl<int,   -1, -1, -1, -1> ccnt;       // [NCHUNK, E]     per-(chunk,expert) counts (the ordered compaction)
    // --- output ---
    gl<int,   -1, -1, -1, -1> gath;       // [T_loc_max, 2]  (src_rank, src_token) — the DEDUP'd gather map
    gl<int,   -1, -1, -1, -1> sorted_token_ids;   // [PADMAX, 1]  token | slot<<24
    gl<float, -1, -1, -1, -1> sorted_weights;     // [PADMAX, 1]
    gl<int,   -1, -1, -1, -1> sorted_expert_ids;  // [PADMAX/32, 1]
    gl<int,   -1, -1, -1, -1> num_valid_ids;      // [2, 1]  {padded_rows, T_loc}
    // --- combine pull plan ---
    gl<int,   -1, -1, -1, -1> pull_ptr;   // [T+1, 1]
    gl<int,   -1, -1, -1, -1> pull_src;   // [T*world, 2]  (producer_rank, producer_row)
    // Capacity errors:
    //   bit0 = T_loc  overflowed T_loc_max      bit1 = a sorted row overflowed PADMAX
    //   bit2 = E > E12_MAXE                     bit3 = fanout overflowed the pull_src capacity
    gl<int,   -1, -1, -1, -1> err;        // [1, 1]
    int world, T, TOPK, E, cur_rank, T_loc_max, PADMAX;
    hipStream_t stream;
    dim3 grid()  { return dim3(1); }
    dim3 block() { return dim3(256); }
};

// --- reset ---------------------------------------------------------------------------------------
__global__ void e12_reset_kernel(plan_globals_e12 g) {
    const int stride = gridDim.x * blockDim.x;
    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < g.E; i += stride) g.cnt[{0, 0, i, 0}] = 0;
    const int NCHUNK = (g.world * g.T * g.TOPK + E12_CHUNK - 1) / E12_CHUNK;
    for (int i = tid; i < NCHUNK * g.E; i += stride) g.ccnt[{0, 0, i, 0}] = 0;
    for (int i = tid; i < g.world; i += stride)      g.tloc[{0, 0, i, 0}] = 0;
    if (tid == 0) g.err[{0, 0, 0, 0}] = 0;
}

// --- own[p][gtok] : does rank p hold at least one of gtok's top-8 experts? -----------------------
// Pull each token once per destination rank.
__global__ void e12_own_kernel(plan_globals_e12 g) {
    const int N   = g.world * g.T;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // over world * N
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

// --- tof[p][*] : exclusive prefix of own[p]; tloc[p] = total.  One BLOCK per rank p. ------------
// Inclusive Hillis-Steele block scan.
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

// Per-rank exclusive prefix using 256-thread tiles and a running carry.
__global__ void e12_tof_kernel(plan_globals_e12 g) {
    const int p = blockIdx.x;                       // one block per rank
    if (p >= g.world) return;
    const int N = g.world * g.T;
    const int* own = &g.own[{0, 0, 0, 0}] + (size_t)p * N;
    int*       tof = &g.tof[{0, 0, 0, 0}] + (size_t)p * N;
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    if (threadIdx.x == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < N; base += blockDim.x) {
        const int i    = base + threadIdx.x;
        const int v    = (i < N) ? own[i] : 0;
        const int incl = e26_block_incl_scan(v, s_tmp);        // inclusive within this tile
        if (i < N) tof[i] = s_carry + (incl - v);              // exclusive = inclusive - self
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) s_carry += incl;    // running base += tile total
        __syncthreads();
    }
    if (threadIdx.x == 0) g.tloc[{0, 0, p, 0}] = s_carry;
}

// --- gath[t] for this rank; num_valid_ids[1] = T_loc -------------------------------------------
__global__ void e12_gath_kernel(plan_globals_e12 g) {
    const int N   = g.world * g.T;
    const int gtok = blockIdx.x * blockDim.x + threadIdx.x;
    const int p   = g.cur_rank;

    if (gtok == 0) g.num_valid_ids[{0, 0, 1, 0}] = g.tloc[{0, 0, p, 0}];   // T_loc, on device
    if (gtok >= N) return;

    const int* own = &g.own[{0, 0, 0, 0}] + (size_t)p * N;
    const int* tof = &g.tof[{0, 0, 0, 0}] + (size_t)p * N;
    if (!own[gtok]) return;

    const int t = tof[gtok];
    if (t >= g.T_loc_max) { atomicOr(&g.err[{0, 0, 0, 0}], 1); return; }
    g.gath[{0, 0, t, 0}] = gtok / g.T;               // src_rank
    g.gath[{0, 0, t, 1}] = gtok % g.T;               // src_token
}

// --- per-expert row counts, and per-(chunk,expert) counts for the ordered compaction ------------
__global__ void e12_count_kernel(plan_globals_e12 g) {
    const int n  = g.world * g.T * g.TOPK;
    const int lo = g.cur_rank * g.E, hi = lo + g.E;
    const int* ids = &g.all_ids[{0, 0, 0, 0}];
    int* cnt  = &g.cnt[{0, 0, 0, 0}];
    int* ccnt = &g.ccnt[{0, 0, 0, 0}];
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const int e = ids[j];
        if (e < lo || e >= hi) continue;
        const int le = e - lo;
        atomicAdd(cnt + le, 1);                                    // order-independent: safe
        atomicAdd(ccnt + (j / E12_CHUNK) * g.E + le, 1);           // order-independent: safe
    }
}

// --- 32-padded expert prefix and per-chunk bases -----------------------------------------------
// erb[E] = padded_rows = num_valid_ids[0].
// Also writes sorted_expert_ids (one entry per 32-row block) and turns ccnt into an exclusive prefix
// over CHUNKS within each expert, based at erb[e].  One block; E = 32 and NCHUNK ~ 102.
__global__ void e12_scan_kernel(plan_globals_e12 g) {
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
            acc += ((c + E12_BLOCK_M - 1) / E12_BLOCK_M) * E12_BLOCK_M;   // pad to 32, not 256
        }
        s_erb[E] = acc;
        g.erb[{0, 0, E, 0}] = acc;
        g.num_valid_ids[{0, 0, 0, 0}] = acc;                              // padded_rows, on device
    }
    __syncthreads();

    // sorted_expert_ids: one LOCAL expert id per 32-row block, expert-major, contiguous.
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        const int b0 = s_erb[e] / E12_BLOCK_M;
        const int b1 = s_erb[e + 1] / E12_BLOCK_M;
        for (int b = b0; b < b1; ++b) g.sorted_expert_ids[{0, 0, b, 0}] = e;
    }

    // ccnt[chunk][e] -> exclusive prefix over chunks, based at erb[e].  Serial over NCHUNK per
    // expert (deterministic); one thread per expert.
    for (int e = threadIdx.x; e < E; e += blockDim.x) {
        int acc = s_erb[e];
        for (int c = 0; c < NCHUNK; ++c) {
            const int v = g.ccnt[{0, 0, c * E + e, 0}];
            g.ccnt[{0, 0, c * E + e, 0}] = acc;
            acc += v;
        }
    }
}

// --- initialize padding rows ------------------------------------------------------------------
// The bounded GEMM descriptor maps token=T_loc padding loads to zero. Run after gath writes T_loc.
__global__ void e12_pad_kernel(plan_globals_e12 g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= g.PADMAX) return;
    const int T_loc = g.num_valid_ids[{0, 0, 1, 0}];
    g.sorted_token_ids[{0, 0, i, 0}] = (T_loc & 0x00FFFFFF) | (g.TOPK << 24);
    g.sorted_weights [{0, 0, i, 0}] = 0.f;
}

// --- ordered compaction -----------------------------------------------------------------------
// Preserve increasing pair order and pack slot in the high eight bits.
__global__ void e12_sorted_kernel(plan_globals_e12 g) {
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

    // Rank each pair by earlier same-expert pairs in this chunk.
    __shared__ int s_le[E12_CHUNK];                       // per-thread local expert, -1 if not ours
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
        int rank = 0;                                     // earlier same-expert pairs, index order
        for (int u = 0; u < threadIdx.x; ++u) rank += (s_le[u] == le);
        const int row = s_cur[le] + rank;
        if (row >= g.PADMAX) { atomicOr(&g.err[{0, 0, 0, 0}], 2); }
        else {
            const float* wgs = &g.all_wgt[{0, 0, 0, 0}];
            const int*   tof = &g.tof[{0, 0, 0, 0}] + (size_t)g.cur_rank * N;
            const int gtok = j / g.TOPK;
            const int slot = j % g.TOPK;
            const int t    = tof[gtok];                   // this rank's LOCAL token index (dedup'd)
            g.sorted_token_ids[{0, 0, row, 0}] = (t & 0x00FFFFFF) | (slot << 24);
            g.sorted_weights  [{0, 0, row, 0}] = wgs[j];
        }
    }
}

// --- combine pull plan -------------------------------------------------------------------------
// List each producer rank and its deterministic local row for every owned token.
__global__ void e12_pull_kernel(plan_globals_e12 g) {
    const int tau = blockIdx.x * blockDim.x + threadIdx.x;
    if (tau > g.T) return;                                 // one extra thread writes pull_ptr[T]
    const int N = g.world * g.T;
    const int W = g.world;

    // pull_ptr is an exclusive prefix of fanout; fanout is cheap enough to recount per token.
    int base = 0;
    for (int u = 0; u < tau; ++u) {
        const int gu = g.cur_rank * g.T + u;
        for (int p = 0; p < W; ++p) base += g.own[{0, 0, p * N + gu, 0}];
    }
    g.pull_ptr[{0, 0, tau, 0}] = base;
    if (tau == g.T) return;

    const int gtok = g.cur_rank * g.T + tau;
    int k = base;
    for (int p = 0; p < W; ++p) {
        if (!g.own[{0, 0, p * N + gtok, 0}]) continue;
        if (k >= g.T * W) { atomicOr(&g.err[{0, 0, 0, 0}], 8); break; }
        g.pull_src[{0, 0, k, 0}] = p;                               // producer rank
        g.pull_src[{0, 0, k, 1}] = g.tof[{0, 0, p * N + gtok, 0}];  // that producer's local row
        ++k;
    }
}

void dispatch_build_plan_prod(plan_globals_e12 g) {
    const int n = g.world * g.T * g.TOPK;
    const int N = g.world * g.T;
    if (n <= 0 || g.E <= 0 || g.PADMAX <= 0) return;      // a 0-block grid is hipErrorInvalidConfiguration
    const int NCHUNK = (n + E12_CHUNK - 1) / E12_CHUNK;
    auto nb = [](int x) { return (x + 255) / 256; };

    e12_reset_kernel <<<nb(NCHUNK * g.E + g.E + g.world), 256, 0, g.stream>>>(g);
    e12_own_kernel   <<<nb(g.world * N),                   256, 0, g.stream>>>(g);
    e12_tof_kernel   <<<g.world,                           256, 0, g.stream>>>(g);
    e12_gath_kernel  <<<nb(N),                             256, 0, g.stream>>>(g);
    e12_count_kernel <<<nb(n),                             256, 0, g.stream>>>(g);
    e12_scan_kernel  <<<1,                                 256, 0, g.stream>>>(g);
    e12_pad_kernel   <<<nb(g.PADMAX),                      256, 0, g.stream>>>(g);   // after gath
    e12_sorted_kernel<<<NCHUNK,                            256, 0, g.stream>>>(g);   // after pad
    e12_pull_kernel  <<<nb(g.T + 1),                       256, 0, g.stream>>>(g);
}

struct zero_globals_e12 {
    gl<bf16, -1, -1, -1, -1> buf;              // [T_loc_max, H]
    gl<int,  -1, -1, -1, -1> num_valid_ids;    // [2,1]; [1] = T_loc
    int T_loc_max, H;
    hipStream_t stream;
    dim3 grid()  { return dim3(T_loc_max > 0 ? T_loc_max : 1); }
    dim3 block() { return dim3(256); }
};

__global__ void zero_partial_kernel(zero_globals_e12 g) {
    const int t     = blockIdx.x;
    const int T_loc = g.num_valid_ids[{0, 0, 1, 0}];
    if (t >= T_loc || t >= g.T_loc_max) return;
    bf16* row = reinterpret_cast<bf16*>(&g.buf[{0, 0, 0, 0}]) + (size_t)t * g.H;
    const int Hv = g.H >> 3;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int hv = threadIdx.x; hv < Hv; hv += blockDim.x)
        *reinterpret_cast<uint4*>(row + (hv << 3)) = z;
}

void dispatch_zero_partial(zero_globals_e12 g) {
    if (g.T_loc_max <= 0) return;
    zero_partial_kernel<<<g.grid(), g.block(), 0, g.stream>>>(g);
}

// ================================================================================================
PYBIND11_MODULE(tile_plan, m) {
    m.doc() = "Tile-local plan: build_plan_prod + zero_partial. HipKittens and pybind only.";
    py::bind_function<dispatch_build_plan_prod>(m, "build_plan_prod",
        &plan_globals_e12::all_ids, &plan_globals_e12::all_wgt,
        &plan_globals_e12::own, &plan_globals_e12::tof, &plan_globals_e12::tloc,
        &plan_globals_e12::cnt, &plan_globals_e12::erb, &plan_globals_e12::ccnt,
        &plan_globals_e12::gath,
        &plan_globals_e12::sorted_token_ids, &plan_globals_e12::sorted_weights,
        &plan_globals_e12::sorted_expert_ids, &plan_globals_e12::num_valid_ids,
        &plan_globals_e12::pull_ptr, &plan_globals_e12::pull_src, &plan_globals_e12::err,
        &plan_globals_e12::world, &plan_globals_e12::T, &plan_globals_e12::TOPK,
        &plan_globals_e12::E, &plan_globals_e12::cur_rank,
        &plan_globals_e12::T_loc_max, &plan_globals_e12::PADMAX);
    py::bind_function<dispatch_zero_partial>(m, "zero_partial",
        &zero_globals_e12::buf, &zero_globals_e12::num_valid_ids,
        &zero_globals_e12::T_loc_max, &zero_globals_e12::H);
}
