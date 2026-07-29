// ================================================================================================
// hkp_sort.hpp — destination counting sort (see pipelines: k0d_mega.hip,
// k0pf3_dsort.hip, k0d_dsort.hip) with the count / scan / cursor+sei / pad /
// scatter stages as SEPARATELY callable device functions, so kernels keep their
// interleavings (CTA0 counts while CTA1 scans the combine CSR and CTAs 2..15 zero part).
//
// Contracts preserved:
//   * scratch carve: int[256] with erb[0..E] at 0, cnt at 80, cursors at 160
//     (k0d_mega.hip, k0pf3_dsort.hip).
//   * sti packing: (row & 0x00FFFFFF) | (slot << 24); pad sentinel = (T_loc | TOPK << 24),
//     swt = 0.0f (k0d_mega.hip).
//   * Scatter order within an expert segment is ATOMIC-CURSOR — nondeterministic,
//     deliberate, tolerance-gated (k0pf3_dsort.hip).  Do not "stabilize" it.
//   * count/scan run on ONE CTA with the full block; pad/scatter run grid-stride.
//   * The hierarchical count (count_partial_publish + count_reduce) is an ADDITIVE
//     alternative to `count` only: it produces the identical s_cnt and leaves every
//     downstream stage untouched (k0pf4_dsort.hip).
// ================================================================================================

#ifndef HKP_SORT_HPP
#define HKP_SORT_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

template <int MaxExperts, int BlockM = 32, int OffErb = 0, int OffCnt = 80, int OffCur = 160>
struct destination_counting_sort {
  static_assert((BlockM & (BlockM - 1)) == 0, "BlockM is a power of two in every pipeline");

  // ---- stage 1: per-expert counts over received pairs (LDS histogram). ----------------
  // One CTA, full block.  s_cnt is caller-owned LDS [MaxExperts] so the scan stage reads
  // the same array without a global round trip.
  static __device__ __forceinline__ void count(const int* recv_eid, int pairs, int lo,
                                               int E, int* s_cnt, int tid) {
    for (int i = tid; i < E; i += blockDim.x) s_cnt[i] = 0;
    __syncthreads();
    for (int j = tid; j < pairs; j += blockDim.x) {
      const int e = recv_eid[j];
      if (e >= lo && e < lo + E) atomicAdd(&s_cnt[e - lo], 1);
    }
    __syncthreads();
  }

  // ---- stage 1h: HIERARCHICAL count — the same s_cnt, produced by the whole grid. -----
  // stage 1 above runs on ONE CTA and is O(pairs) LDS atomics on a single CU: at prefill
  // (pairs = T_loc * 8 ~ 158k on the hottest rank) it measures 298 us while the other 62
  // CTAs idle (K0PF3_PREFILL_BOTTLENECK_REPORT_20260728.md section 5 / section 8 item 3).
  // The two calls below split it: every CTA histograms a disjoint grid-stride slice and
  // publishes its own [E] counts, then ONE CTA reduces the [E][num_ctas] table back into
  // the identical s_cnt that `scan` consumes.
  //
  // The result is BIT-IDENTICAL to `count`, not merely equivalent: the reduction only
  // re-associates integer additions.  The caller owns the cross-CTA release/acquire —
  // a grid barrier (hkp::grid_barrier) MUST separate publish from reduce.
  //
  // hcnt is caller-owned LOCAL global scratch of hcnt_elems(E, num_ctas) ints, laid out
  // expert-major (hcnt[e * num_ctas + cta]) so the reduction reads it coalesced.  Every
  // cell is overwritten (never accumulated) each epoch, so it needs no reset and adds no
  // protocol state — the monotonic-epoch replay proof is untouched.
  static __device__ __forceinline__ std::size_t hcnt_elems(int E, int num_ctas) {
    return (std::size_t)E * (std::size_t)num_ctas;
  }

  // 1h-a: this CTA's slice -> LDS histogram -> published row of the global table.
  static __device__ __forceinline__ void count_partial_publish(const int* recv_eid,
                                                               int pairs, int lo, int E,
                                                               int* s_cnt, int* hcnt,
                                                               int num_ctas, int cta,
                                                               int tid) {
    for (int i = tid; i < E; i += blockDim.x) s_cnt[i] = 0;
    __syncthreads();
    const int stride = num_ctas * (int)blockDim.x;
    for (int j = cta * (int)blockDim.x + tid; j < pairs; j += stride) {
      const int e = recv_eid[j];
      if (e >= lo && e < lo + E) atomicAdd(&s_cnt[e - lo], 1);
    }
    __syncthreads();
    for (int i = tid; i < E; i += blockDim.x) hcnt[i * num_ctas + cta] = s_cnt[i];
  }

  // 1h-b: cross-CTA reduction on ONE CTA -> the s_cnt `scan` expects.  Call AFTER the
  // grid barrier that publishes hcnt.
  static __device__ __forceinline__ void count_reduce(const int* hcnt, int E, int num_ctas,
                                                      int* s_cnt, int tid) {
    for (int i = tid; i < E; i += blockDim.x) s_cnt[i] = 0;
    __syncthreads();
    const int n = E * num_ctas;
    for (int idx = tid; idx < n; idx += blockDim.x) {
      const int v = hcnt[idx];
      if (v) atomicAdd(&s_cnt[idx / num_ctas], v);
    }
    __syncthreads();
  }

  // ---- stage 2: padded BM32 scan (serial on tid 0, E <= 64) + nvi. -------------------
  static __device__ __forceinline__ void scan(const int* s_cnt, int E, int T_loc,
                                              int* scratch, int* nvi, int tid) {
    if (tid == 0) {
      int acc = 0;
      for (int e = 0; e < E; ++e) {
        scratch[OffErb + e] = acc;
        scratch[OffCnt + e] = s_cnt[e];
        acc += ((s_cnt[e] + BlockM - 1) / BlockM) * BlockM;
      }
      scratch[OffErb + E] = acc;
      nvi[0] = acc;
      nvi[1] = T_loc;
    }
    __syncthreads();
  }

  // ---- stage 3: scatter cursors + sorted-expert-ids (needs the scan). -----------------
  static __device__ __forceinline__ void cursors_sei(int* scratch, int E, int PADMAX,
                                                     int* sei, int* pperr, int pad_bit,
                                                     int tid) {
    const int bcap = PADMAX / BlockM;
    for (int e = tid; e < E; e += blockDim.x) {
      scratch[OffCur + e] = scratch[OffErb + e];
      const int b0 = scratch[OffErb + e] / BlockM;
      const int b1 = scratch[OffErb + e + 1] / BlockM;
      for (int b = b0; b < b1; ++b) {
        if (b >= bcap) {
          atomicOr(pperr, pad_bit);
          break;
        }
        sei[b] = e;
      }
    }
  }

  // ---- stage 4: pad sentinel rows, disjoint from every scatter position. --------------
  // Grid-stride, ALL CTAs.
  static __device__ __forceinline__ void pad(const int* scratch, int E, int T_loc, int TOPK,
                                             int PADMAX, int* sti, float* swt, int* pperr,
                                             int pad_bit) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < E * BlockM;
         idx += gridDim.x * blockDim.x) {
      const int e = idx / BlockM, k = idx % BlockM;
      const int row = scratch[OffErb + e] + scratch[OffCnt + e] + k;
      if (row < scratch[OffErb + e + 1]) {
        if (row >= PADMAX) {
          atomicOr(pperr, pad_bit);
          continue;
        }
        sti[row] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
        swt[row] = 0.0f;
      }
    }
  }

  // ---- stage 5: live-pair scatter through per-expert atomic cursors. -------------------
  static __device__ __forceinline__ void scatter(const int* recv_eid,
                                                 const float* recv_wgt, int pairs, int lo,
                                                 int E, int TOPK, int PADMAX, int* scratch,
                                                 int* sti, float* swt, int* pperr,
                                                 int ovf_bit) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < pairs;
         j += gridDim.x * blockDim.x) {
      const int e = recv_eid[j];
      if (e < lo || e >= lo + E) continue;
      const int pos = atomicAdd(&scratch[OffCur + (e - lo)], 1);
      if (pos >= PADMAX) {
        atomicOr(pperr, ovf_bit);
        continue;
      }
      const int row = j / TOPK;
      const int slot = j - row * TOPK;
      sti[pos] = (row & 0x00FFFFFF) | (slot << 24);
      swt[pos] = recv_wgt[j];
    }
  }
};

// ---------------------------------------------------------------------------
// Owner-combine CSR.  Two variants — the choice is forced by T, not taste:
//   * warp64 (k0d_mega.hip): one warp, __shfl_up inclusive scan, requires T <= 64.
//   * block256 (k0pf3_dsort.hip): 256-wide LDS block scan with serial carry over
//     256-token chunks, any T.
// ---------------------------------------------------------------------------

// 256-wide inclusive block scan (k0pf3_dsort.hip); s_tmp is caller LDS [256].
__device__ __forceinline__ int block_incl_scan_256(int v, int* s_tmp) {
  const int tid = threadIdx.x;
  s_tmp[tid] = v;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    const int add = (tid >= off) ? s_tmp[tid - off] : 0;
    __syncthreads();
    s_tmp[tid] += add;
    __syncthreads();
  }
  const int r = s_tmp[tid];
  __syncthreads();
  return r;
}

__device__ __forceinline__ void csr_scan_warp64(const int* pull_cnt, int* pull_ptr, int T,
                                                int tid) {
  if (tid >= 64) return;
  const int tau = tid;
  const int f = (tau < T) ? pull_cnt[tau] : 0;
  int incl = f;
#pragma unroll
  for (int off = 1; off < 64; off <<= 1) {
    const int add = __shfl_up(incl, off, 64);
    if ((tid & 63) >= off) incl += add;
  }
  if (tau < T) pull_ptr[tau] = incl - f;
  if (tau == T - 1) pull_ptr[T] = incl;
}

// s_tmp is caller LDS [256], s_carry is caller LDS [1].
__device__ __forceinline__ void csr_scan_block256(const int* pull_cnt, int* pull_ptr,
                                                  int T, int tid, int* s_tmp,
                                                  int* s_carry) {
  if (tid == 0) *s_carry = 0;
  __syncthreads();
  for (int base = 0; base < T; base += 256) {
    const int tau = base + tid;
    const int f = (tau < T) ? pull_cnt[tau] : 0;
    const int incl = block_incl_scan_256(f, s_tmp);
    if (tau < T) pull_ptr[tau] = *s_carry + (incl - f);
    __syncthreads();
    if (tid == 255) *s_carry += incl;
    __syncthreads();
  }
  if (tid == 0) pull_ptr[T] = *s_carry;
}

// pull_src fill: per token, compact its primary (dest,row) pairs in SLOT order (not the
// frozen plan's ascending-rank order — the fp32 combine sum is reordered, never changed in
// value set, k0pf3_dsort.hip).  Grid-stride form (k0pf3_dsort.hip);
// k0d_mega.hip fuses the same fill into its warp scan with identical content.
__device__ __forceinline__ void pull_src_fill(const int* pull_stage, const int* pull_ptr,
                                              int T, int TOPK, int cap, int* pull_src,
                                              int* pperr, int ovf_bit) {
  for (int tau = blockIdx.x * blockDim.x + threadIdx.x; tau < T;
       tau += gridDim.x * blockDim.x) {
    int k = pull_ptr[tau];
    for (int s = 0; s < TOPK; ++s) {
      const int pe = pull_stage[((std::size_t)tau * TOPK + s) * 2 + 0];
      if (pe < 0) continue;
      const int row = pull_stage[((std::size_t)tau * TOPK + s) * 2 + 1];
      if (k >= cap) {
        atomicOr(pperr, ovf_bit);
        break;
      }
      pull_src[(std::size_t)k * 2 + 0] = pe;
      pull_src[(std::size_t)k * 2 + 1] = row;
      ++k;
    }
  }
}

// ---------------------------------------------------------------------------
// EXPERIMENTAL — per-source segmented counting sort (one slice per producer rank + merge).
// NOT behaviour-preserving versus destination_counting_sort, and not recommended: both
// fused variants built on per-row readiness measured slower.  Work being serial does not
// make it fusable, and fusing it is not the same as overlapping it.  Interface stub only;
// do not implement without a graph-mode A/B against the monolithic sort.
// ---------------------------------------------------------------------------
template <int MaxExperts, int BlockM = 32, int MaxSources = 8>
struct destination_counting_sort_segmented {
  static __device__ __forceinline__ void count_per_source(const int* /*recv_eid*/,
                                                          int /*pairs*/, int /*lo*/,
                                                          int /*E*/, int* /*s_cnt*/,
                                                          int /*tid*/) {
    static_assert(MaxExperts > 0 && MaxSources > 0,
                  "experimental stub — see the regression caveat above");
  }
  static __device__ __forceinline__ void merge_source_scans() {
    static_assert(MaxExperts > 0 && MaxSources > 0,
                  "experimental stub — see the regression caveat above");
  }
};

}  // namespace hkp

#endif  // HKP_SORT_HPP
