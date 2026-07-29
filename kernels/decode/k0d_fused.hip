// Decode route all-gather, plan construction, and origin quantization in one 1024-thread block.
// Ordered compaction ranks each pair by earlier same-expert pairs in its 256-pair chunk.
// Quantization uses 128-element groups and row-major scales.
// Limits: N<=512, NCHUNK<=16, TOPK=8, world<=8, E<=64, and T<=512.

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
// K0D_SORT_SERIAL enables the reference serial compaction.

#define K0D_MAXN 512         // world*T guard (decode B64 at EP8 is exactly 512)
#define K0D_MAXCHUNK 16      // (world*T*TOPK)/256 guard (B64 at EP8 is exactly 16)
#define K0D_TOPK 8
#define K0D_MAXE 64
#define K0D_BLOCK_M 32
#define K0D_CHUNK 256
#define K0D_NG 56            // 7168/128
#define K0D_SCAN 512         // tof scan tile (K0D_MAXN/8)

__device__ __forceinline__ float k0d_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

extern "C" __global__ void __launch_bounds__(1024, 1) k0d_fused(
    int* __restrict__ all_ids, float* __restrict__ all_wgt,   // [world*T*TOPK]
    const int* __restrict__ my_ids, const float* __restrict__ my_wgt,   // SYMMETRIC [T*TOPK]
    int* __restrict__ own, int* __restrict__ tof, int* __restrict__ tloc,
    int* __restrict__ gath, int* __restrict__ sti, float* __restrict__ swt,
    int* __restrict__ sei, int* __restrict__ nvi,
    int* __restrict__ pull_ptr, int* __restrict__ pull_src,
    int* __restrict__ perr,
    const unsigned short* __restrict__ hidden,                // [T, 7168] bf16
    unsigned char* __restrict__ a_src,                        // [T, 7168] fp8 SYMMETRIC
    float* __restrict__ sc_src,                               // [T, 56] fp32 SYMMETRIC
    int world, int T, int TOPK, int E, int cur, int T_loc_max, int PADMAX,
    // ---- optional arm-2 signaling (pass comb_flags == nullptr for the standalone-barrier arm) ----
    unsigned int* __restrict__ comb_flags,      // SYMMETRIC [world]: combine_done flags (waited on)
    unsigned int* __restrict__ quant_flags,     // SYMMETRIC [world]: quant_done flags (poked)
    unsigned int* __restrict__ quant_count,     // LOCAL [1]: monotonic quant epoch (reset-free)
    int* __restrict__ pperr2, long long spin_limit,
    int do_quant) {                             // 0 = route+plan only; caller launches fast quant
  const int tid = threadIdx.x;
  const int N = world * T;
  const int n = N * TOPK;
  const int NCHUNK = (n + K0D_CHUNK - 1) / K0D_CHUNK;
  const int NG = K0D_NG;

  if (N > K0D_MAXN || NCHUNK > K0D_MAXCHUNK || TOPK != K0D_TOPK || world > 8 || E > K0D_MAXE ||
      T > 512) {
    if (tid == 0) atomicOr(perr, 64);
    return;
  }

  __shared__ int s_own[8][K0D_SCAN];
  __shared__ int s_tof[8][K0D_SCAN];
  __shared__ int s_cnt[K0D_MAXE];
  __shared__ int s_ccnt[K0D_MAXCHUNK][K0D_MAXE];
  __shared__ int s_tmp[K0D_SCAN];
  __shared__ int s_wcnt[4][4][K0D_MAXE];     // [group][wave][expert]
  __shared__ int s_cur[4][K0D_MAXE];
  __shared__ int s_erb[K0D_MAXE + 1];

  // s_ccnt has a fixed K0D_MAXE row stride; clear all used rows at that stride.
  for (int i = tid; i < E; i += 1024) s_cnt[i] = 0;
  for (int i = tid; i < NCHUNK * K0D_MAXE; i += 1024) (&s_ccnt[0][0])[i] = 0;
  if (tid == 0) *perr = 0;
  __syncthreads();

  // Route all-gather.
  {
    const int cnt = T * TOPK;
    for (int r = 0; r < world; ++r) {
      if (r == cur) {
        for (int i = tid; i < cnt; i += 1024) {
          all_ids[(size_t)r * cnt + i] = my_ids[i];
          all_wgt[(size_t)r * cnt + i] = my_wgt[i];
        }
      } else {
        mori::shmem::ShmemGetMemBlock((void*)(all_ids + (size_t)r * cnt),
                                      (const void*)my_ids, (size_t)cnt * sizeof(int), r);
        mori::shmem::ShmemGetMemBlock((void*)(all_wgt + (size_t)r * cnt),
                                      (const void*)my_wgt, (size_t)cnt * sizeof(float), r);
      }
      __syncthreads();
    }
  }

  // Build per-rank ownership.
  for (int gtok = tid; gtok < N; gtok += 1024) {
    const int* ids = all_ids + (size_t)gtok * TOPK;
    const uint4 i01 = *reinterpret_cast<const uint4*>(ids);
    const uint4 i45 = *reinterpret_cast<const uint4*>(ids + 4);
    const int e[8] = {(int)i01.x, (int)i01.y, (int)i01.z, (int)i01.w,
                      (int)i45.x, (int)i45.y, (int)i45.z, (int)i45.w};
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const int lo = p * E, hi = lo + E;
      int o = 0;
#pragma unroll
      for (int s = 0; s < 8; ++s) o |= (e[s] >= lo && e[s] < hi) ? 1 : 0;
      if (p < world) s_own[p][gtok] = o;
    }
  }
  __syncthreads();

  // Prefix ownership into local token offsets.
  for (int p = 0; p < world; ++p) {
    const int i = tid;
    const int v = (i < N) ? s_own[p][i] : 0;
    if (i < K0D_SCAN) s_tmp[i] = v;
    __syncthreads();
    for (int off = 1; off < K0D_SCAN; off <<= 1) {
      const int add = (i >= off && i < K0D_SCAN) ? s_tmp[i - off] : 0;
      __syncthreads();
      if (i < K0D_SCAN) s_tmp[i] += add;
      __syncthreads();
    }
    if (i < N) {
      const int incl = s_tmp[i];
      s_tof[p][i] = incl - v;
      tof[(size_t)p * N + i] = incl - v;         // global write-through
      own[(size_t)p * N + i] = v;                // global write-through
    }
    if (i == N - 1) tloc[p] = s_tmp[i];
    __syncthreads();
  }

  // Build the gather map.
  if (tid == 0) nvi[1] = tloc[cur];
  for (int gtok = tid; gtok < N; gtok += 1024) {
    if (!s_own[cur][gtok]) continue;
    const int t = s_tof[cur][gtok];
    if (t >= T_loc_max) { atomicOr(perr, 1); continue; }
    gath[(size_t)t * 2 + 0] = gtok / T;
    gath[(size_t)t * 2 + 1] = gtok % T;
  }
  __syncthreads();

  // Count local-expert pairs.
  {
    const int lo = cur * E, hi = lo + E;
    for (int j = tid; j < n; j += 1024) {
      const int e = all_ids[j];
      if (e < lo || e >= hi) continue;
      const int le = e - lo;
      atomicAdd(&s_cnt[le], 1);
      atomicAdd(&s_ccnt[j / K0D_CHUNK][le], 1);
    }
  }
  __syncthreads();

  // Build padded expert segments.
  if (tid == 0) {
    int acc = 0;
    for (int e = 0; e < E; ++e) {
      s_erb[e] = acc;
      const int c = s_cnt[e];
      acc += ((c + K0D_BLOCK_M - 1) / K0D_BLOCK_M) * K0D_BLOCK_M;
    }
    s_erb[E] = acc;
    nvi[0] = acc;
  }
  __syncthreads();
  {
    const int bcap = PADMAX / K0D_BLOCK_M;
    for (int e = tid; e < E; e += 1024) {
      const int b0 = s_erb[e] / K0D_BLOCK_M;
      const int b1 = s_erb[e + 1] / K0D_BLOCK_M;
      for (int b = b0; b < b1; ++b) {
        if (b >= bcap) { atomicOr(perr, 2); break; }
        sei[b] = e;
      }
    }
    __syncthreads();
    for (int e = tid; e < E; e += 1024) {
      int acc = s_erb[e];
      for (int c = 0; c < NCHUNK; ++c) {
        const int v = s_ccnt[c][e];
        s_ccnt[c][e] = acc;
        acc += v;
      }
    }
  }
  __syncthreads();

  // Initialize padding rows.
  {
    const int padded = nvi[0];
    const int T_loc = nvi[1];
    for (int i = tid; i < padded; i += 1024) {
      sti[i] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
      swt[i] = 0.0f;
    }
  }
  __syncthreads();

  // __match_any_sync is wave-local and must be called by every thread before wave counts merge.
  {
    const int grp = tid >> 8;          // 0..3
    const int gtid = tid & 255;        // pair index within the chunk
    const int lane = gtid & 63;
    const int wv = gtid >> 6;          // wave within the group (0..3)
    const int lo = cur * E, hi = lo + E;
    for (int c0 = 0; c0 < NCHUNK; c0 += 4) {
      const int c = c0 + grp;
      const bool act = (c < NCHUNK);
      int le = -1;
      if (act) {
        for (int e = gtid; e < E; e += 256) s_cur[grp][e] = s_ccnt[c][e];
        for (int e = lane; e < E; e += 64) s_wcnt[grp][wv][e] = 0;
        const int j = c * K0D_CHUNK + gtid;
        if (j < n) {
          const int e = all_ids[j];
          if (e >= lo && e < hi) le = e - lo;
        }
      }
      __syncthreads();
#ifndef K0D_SORT_SERIAL
      const unsigned long long match = __match_any_sync(__activemask(), le);
      const int first = __ffsll((long long)match) - 1;
      if (act && le >= 0 && lane == first) s_wcnt[grp][wv][le] = __popcll(match);
      __syncthreads();
      if (act && le >= 0) {
        const int j = c * K0D_CHUNK + gtid;
        int rank = __popcll(match & ((1ULL << lane) - 1ULL));
#pragma unroll
        for (int w = 0; w < 4; ++w)
          if (w < wv) rank += s_wcnt[grp][w][le];
        const int row = s_cur[grp][le] + rank;
        if (row >= PADMAX) {
          atomicOr(perr, 2);
        } else {
          const int gtok = j / TOPK;
          const int slot = j % TOPK;
          const int t = s_tof[cur][gtok];
          sti[row] = (t & 0x00FFFFFF) | (slot << 24);
          swt[row] = all_wgt[j];
        }
      }
#else
      __syncthreads();
      if (act && le >= 0) {
        const int j = c * K0D_CHUNK + gtid;
        int rank = 0;
        for (int u = 0; u < gtid; ++u) {
          const int ju = c * K0D_CHUNK + u;
          if (ju < n) {
            const int e = all_ids[ju];
            if (e >= lo && e < hi && (e - lo) == le) ++rank;
          }
        }
        const int row = s_cur[grp][le] + rank;
        if (row >= PADMAX) { atomicOr(perr, 2); }
        else {
          const int gtok = j / TOPK;
          const int slot = j % TOPK;
          const int t = s_tof[cur][gtok];
          sti[row] = (t & 0x00FFFFFF) | (slot << 24);
          swt[row] = all_wgt[j];
        }
      }
#endif
      __syncthreads();
    }
  }

  // Build the owner pull plan.
  {
    int carry = 0;
    for (int base = 0; base < T; base += 64) {
      const int tau = base + (tid & 63);
      int f = 0;
      if (tid < 64 && tau < T) {
        const int gtok = cur * T + tau;
#pragma unroll
        for (int p = 0; p < 8; ++p) f += (p < world) ? s_own[p][gtok] : 0;
      }
      int incl = f;
      if (tid < 64) {
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
          const int add = __shfl_up(incl, off, 64);
          if ((tid & 63) >= off) incl += add;
        }
        if (tau < T) pull_ptr[tau] = carry + incl - f;
        if (tau == T - 1) pull_ptr[T] = carry + incl;
        if (tau < T) {
          const int gtok = cur * T + tau;
          int k = carry + incl - f;
          for (int p = 0; p < world; ++p) {
            if (!s_own[p][gtok]) continue;
            if (k >= T * world) { atomicOr(perr, 8); break; }
            pull_src[(size_t)k * 2 + 0] = p;
            pull_src[(size_t)k * 2 + 1] = s_tof[p][gtok];
            ++k;
          }
        }
        if ((tid & 63) == 63) carry += incl;
      }
      __syncthreads();
    }
  }

  // Wait before quantization reuses a_src or part; combine completion implies gather completion.
  if (do_quant && comb_flags != nullptr) {
    if (tid < world) {
      const unsigned int target = quant_count[0];        // Previous epoch.
      volatile const unsigned int* vf = comb_flags;
      long long spins = 0;
      while (vf[tid] < target) {
        if (++spins > spin_limit) { atomicOr(pperr2, 2); break; }
        __builtin_amdgcn_s_sleep(4);
      }
    }
    __syncthreads();
    __threadfence_system();                                // Acquire completion.
  }

  // Quantize activations.
  if (do_quant) {
    const int groups = T * NG;
    for (int gidx = tid; gidx < groups; gidx += 1024) {
      const int t = gidx / NG;
      const int g = gidx % NG;
      const unsigned short* hrow = hidden + (size_t)t * 7168 + (size_t)g * 128;
      // Compute amax over one 128-element group.
      float amax = 0.0f;
#pragma unroll
      for (int v = 0; v < 16; ++v) {
        const uint4 raw = *reinterpret_cast<const uint4*>(hrow + v * 8);
        const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          amax = fmaxf(amax, fabsf(k0d_bf16_to_f32(h16[e])));
      }
      const float scale = fmaxf(amax, 1.0e-6f) / 448.0f;
      sc_src[(size_t)t * NG + g] = scale;
      // Convert and store eight FP8 values per write.
#pragma unroll
      for (int v = 0; v < 16; ++v) {
        const uint4 raw = *reinterpret_cast<const uint4*>(hrow + v * 8);
        const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw);
        unsigned char out[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float ff = k0d_bf16_to_f32(h16[e]);
          const float q = fminf(fmaxf(ff / scale, -448.0f), 448.0f);
          out[e] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
        }
        *reinterpret_cast<uint2*>(a_src + (size_t)t * 7168 + (size_t)g * 128 + v * 8) =
            *reinterpret_cast<const uint2*>(out);
      }
    }
  }

  // Publish quantization completion for gather.
  if (do_quant && comb_flags != nullptr) {
    __syncthreads();                                       // Quant writes precede the release.
    if (tid == 0) {
      __threadfence_system();                              // Release quantized data.
      const unsigned int epoch = quant_count[0] + 1;
      quant_count[0] = epoch;
      for (int pe = 0; pe < world; ++pe) {
        if (pe == cur) {
          __hip_atomic_fetch_add(&quant_flags[cur], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        } else {
          unsigned int* peer = reinterpret_cast<unsigned int*>(
              mori::shmem::ShmemPtrP2p((uint64_t)(quant_flags + cur), pe));
          __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }
}
