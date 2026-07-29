// Destination counting-sort stages.
// scratch[256] stores erb at 0, counts at 80, and cursors at 160. sti packs a 24-bit row and
// 8-bit slot. Expert-local scatter order is nondeterministic. count and scan use one CTA;
// pad and scatter are grid-stride. Hierarchical counting produces the same integer totals.

#ifndef HKP_SORT_HPP
#define HKP_SORT_HPP

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace hkp {

template <int MaxExperts, int BlockM = 32, int OffErb = 0, int OffCnt = 80, int OffCur = 160>
struct destination_counting_sort {
  static_assert((BlockM & (BlockM - 1)) == 0, "BlockM is a power of two in every pipeline");

  // Count received pairs per expert in caller-owned LDS.
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

  // Hierarchical count. hcnt is expert-major and fully overwritten each epoch. A grid barrier
  // must separate count_partial_publish from count_reduce.
  static __device__ __forceinline__ std::size_t hcnt_elems(int E, int num_ctas) {
    return (std::size_t)E * (std::size_t)num_ctas;
  }

  // Publish this CTA's partial histogram.
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

  // Reduce published histograms on one CTA.
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

  // Build padded expert segments and nvi.
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

  // Initialize scatter cursors and sorted expert IDs.
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

  // Initialize padding rows; ranges are disjoint from scatter positions.
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

  // Scatter live pairs through per-expert atomic cursors.
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

// Owner-combine CSR scans: warp64 requires T<=64; block256 supports any T.

// 256-wide inclusive scan using caller-owned LDS.
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

// s_tmp is LDS[256]; s_carry is LDS[1].
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

// Compact primary destination rows in slot order.
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

// Interface stub for per-source segmented counting sort.
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
