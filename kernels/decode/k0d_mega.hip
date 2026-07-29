// Decode push, cross-rank synchronization, and destination sort in one 16-CTA launch.
// Doorbells are monotonic. Waiting for every peer also proves the previous epoch's combines have
// stopped reading part before it is cleared. Expert-local scatter order is nondeterministic.
//
// CTA0 builds expert segments, CTA1 builds the combine CSR, and CTAs 2..15 clear live partial
// rows and transpose scales. A device-scope grid barrier publishes their outputs before scatter.
// pperr bits: 2 padding overflow, 8 CSR overflow, 128 receive overflow, 256 shape guard,
// 1024 doorbell timeout, 2048 T_loc range, 4096 scatter overflow, 8192 grid-barrier timeout.

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

#define K0DM_MAXE 64
#define K0DM_K 7168
#define K0DM_NG 56
#define K0DM_SC_ERB 0
#define K0DM_SC_CNT 80
#define K0DM_SC_CUR 160

extern "C" __global__ void __launch_bounds__(256, 8) k0d_mega(
    // ---- route + quantized source (LOCAL reads) ----
    const int* __restrict__ my_ids,              // SYMMETRIC [T*TOPK]
    const float* __restrict__ my_wgt,            // SYMMETRIC [T*TOPK]
    const unsigned char* __restrict__ a_src,     // SYMMETRIC [T, 7168] fp8
    const float* __restrict__ sc_src,            // SYMMETRIC [T, 56] row-major
    // ---- symmetric receive buffers (REMOTE writes in M1, LOCAL reads in M4+) ----
    unsigned char* __restrict__ a_dst,           // SYMMETRIC [T_loc_max, 7168] fp8
    float* __restrict__ sc_stage,                // SYMMETRIC [T_loc_max, 56] row-major
    int* __restrict__ recv_eid,                  // SYMMETRIC [T_loc_max, TOPK]
    float* __restrict__ recv_wgt,                // SYMMETRIC [T_loc_max, TOPK]
    unsigned int* __restrict__ dest_counter,     // SYMMETRIC [1]
    unsigned int* __restrict__ recv_doorbell,    // SYMMETRIC [world]
    // ---- owner staging + plan outputs (LOCAL) ----
    int* __restrict__ pull_stage,                // LOCAL [T*TOPK, 2]
    int* __restrict__ pull_cnt,                  // LOCAL [T]
    int* __restrict__ sti, float* __restrict__ swt, int* __restrict__ sei, int* __restrict__ nvi,
    int* __restrict__ pull_ptr, int* __restrict__ pull_src,
    float* __restrict__ sc_dst,                  // LOCAL group-major (MROWS, 56) alloc
    unsigned short* __restrict__ part,           // LOCAL [MROWS, 7168] bf16
    // ---- local protocol state (zeroed once, never reset) ----
    int* __restrict__ scratch,                   // LOCAL [256]
    unsigned int* __restrict__ grid_count,       // LOCAL [1]: M2 arrival counter
    unsigned int* __restrict__ gbar,             // LOCAL [1]: M5 barrier counter
    unsigned int* __restrict__ sort_count,       // LOCAL [1]: epoch counter
    int* __restrict__ pperr, long long spin_limit,
    int T, int TOPK, int E, int cur, int world, int T_loc_max, int PADMAX) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int lane = tid & 63;

  if (TOPK != 8 || E > K0DM_MAXE || world > 8 || T > 64 || T_loc_max > 0x00FFFFFF) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 256);
    return;
  }

  const unsigned int epoch = sort_count[0] + 1u;

  // Push sweep: one warp per token.
  {
    const int gw = bid * (blockDim.x >> 6) + (tid >> 6);
    const int nw = gridDim.x * (blockDim.x >> 6);
    for (int tau = gw; tau < T; tau += nw) {
      const int s = lane;
      const int eid = (s < TOPK) ? my_ids[(size_t)tau * TOPK + s] : -1;
      const float wgt = (s < TOPK) ? my_wgt[(size_t)tau * TOPK + s] : 0.0f;
      const int dest = (eid >= 0) ? eid / E : -1;

      const unsigned long long match = __match_any_sync(__activemask(), dest);
      const bool primary = (s < TOPK) && ((match & ((1ULL << s) - 1ULL)) == 0ULL);
      const unsigned long long pmask = __ballot(primary);
      const int fanout = __popcll(pmask);

      unsigned int row = 0u;
      if (primary) {
        if (dest == cur) {
          row = __hip_atomic_fetch_add(dest_counter, 1u, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
        } else {
          unsigned int* peer = reinterpret_cast<unsigned int*>(
              mori::shmem::ShmemPtrP2p((uint64_t)dest_counter, dest));
          row = __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        if (row >= (unsigned int)T_loc_max) atomicOr(pperr, 128);
      }

      if (s < TOPK) {
        pull_stage[((size_t)tau * TOPK + s) * 2 + 0] = primary ? dest : -1;
        pull_stage[((size_t)tau * TOPK + s) * 2 + 1] = primary ? (int)row : -1;
      }
      if (lane == 0) pull_cnt[tau] = fanout;

      if (fanout == 0) continue;

      const unsigned char* srow = a_src + (size_t)tau * K0DM_K;
      uint4 v[7];
#pragma unroll
      for (int c = 0; c < 7; ++c)
        v[c] = *reinterpret_cast<const uint4*>(srow + (c << 10) + (lane << 4));
      uint4 sv;
      if (lane < 14)
        sv = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned char*>(sc_src) + (size_t)tau * (K0DM_NG * 4) +
            (lane << 4));
      const uint4 ep0 = make_uint4((unsigned int)__shfl(eid, 0), (unsigned int)__shfl(eid, 1),
                                   (unsigned int)__shfl(eid, 2), (unsigned int)__shfl(eid, 3));
      const uint4 ep1 = make_uint4((unsigned int)__shfl(eid, 4), (unsigned int)__shfl(eid, 5),
                                   (unsigned int)__shfl(eid, 6), (unsigned int)__shfl(eid, 7));
      const float4 wp0 =
          make_float4(__shfl(wgt, 0), __shfl(wgt, 1), __shfl(wgt, 2), __shfl(wgt, 3));
      const float4 wp1 =
          make_float4(__shfl(wgt, 4), __shfl(wgt, 5), __shfl(wgt, 6), __shfl(wgt, 7));

      unsigned long long pm = pmask;
#pragma unroll 1
      for (int j = 0; j < fanout; ++j) {
        const int src_lane = __ffsll((long long)pm) - 1;
        pm &= pm - 1;
        const int pe = __shfl(dest, src_lane);
        const int drow_id = __shfl((int)row, src_lane);
        if (drow_id < 0 || drow_id >= T_loc_max) continue;
        unsigned char* drow = reinterpret_cast<unsigned char*>(mori::shmem::ShmemPtrP2p(
            (uint64_t)(a_dst + (size_t)drow_id * K0DM_K), pe));
#pragma unroll
        for (int c = 0; c < 7; ++c)
          *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = v[c];
        if (lane < 14) {
          unsigned char* dsc = reinterpret_cast<unsigned char*>(mori::shmem::ShmemPtrP2p(
              (uint64_t)(reinterpret_cast<unsigned char*>(sc_stage) +
                         (size_t)drow_id * (K0DM_NG * 4)),
              pe));
          *reinterpret_cast<uint4*>(dsc + (lane << 4)) = sv;
        }
        if (lane < 2) {
          int* de = reinterpret_cast<int*>(mori::shmem::ShmemPtrP2p(
              (uint64_t)(recv_eid + (size_t)drow_id * TOPK), pe));
          *reinterpret_cast<uint4*>(de + (lane << 2)) = (lane == 0) ? ep0 : ep1;
        } else if (lane < 4) {
          float* dw = reinterpret_cast<float*>(mori::shmem::ShmemPtrP2p(
              (uint64_t)(recv_wgt + (size_t)drow_id * TOPK), pe));
          *reinterpret_cast<float4*>(dw + ((lane - 2) << 2)) = (lane == 2) ? wp0 : wp1;
        }
      }
    }
  }

  // Publish pushes through the doorbell.
  __syncthreads();
  if (tid == 0) {
    __threadfence_system();
    const unsigned int old =
        __hip_atomic_fetch_add(grid_count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if ((old % gridDim.x) == (gridDim.x - 1)) {
      for (int pe = 0; pe < world; ++pe) {
        if (pe == cur) {
          __hip_atomic_fetch_add(&recv_doorbell[cur], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        } else {
          unsigned int* peer = reinterpret_cast<unsigned int*>(
              mori::shmem::ShmemPtrP2p((uint64_t)(recv_doorbell + cur), pe));
          __hip_atomic_fetch_add(peer, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
      }
    }
  }

  // Wait for peer pushes and previous-epoch readers.
  if (tid < world) {
    volatile const unsigned int* vf = recv_doorbell;
    long long spins = 0;
    while (vf[tid] < epoch) {
      if (++spins > spin_limit) { atomicOr(pperr, 1024); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence_system();

  const int T_loc = (int)dest_counter[0];
  if (T_loc < 0 || T_loc > T_loc_max) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 2048);
    return;
  }

  // Build plan metadata and prepare live rows.
  if (bid == 0) {
    __shared__ int s_cnt[K0DM_MAXE];
    for (int i = tid; i < E; i += 256) s_cnt[i] = 0;
    __syncthreads();
    const int lo = cur * E, hi = lo + E;
    const int pairs = T_loc * TOPK;
    for (int j = tid; j < pairs; j += 256) {
      const int e = recv_eid[j];
      if (e >= lo && e < hi) atomicAdd(&s_cnt[e - lo], 1);
    }
    __syncthreads();
    if (tid == 0) {
      int acc = 0;
      for (int e = 0; e < E; ++e) {
        scratch[K0DM_SC_ERB + e] = acc;
        scratch[K0DM_SC_CNT + e] = s_cnt[e];
        acc += ((s_cnt[e] + 31) / 32) * 32;
      }
      scratch[K0DM_SC_ERB + E] = acc;
      nvi[0] = acc;
      nvi[1] = T_loc;
    }
    __syncthreads();
    {
      const int bcap = PADMAX / 32;
      for (int e = tid; e < E; e += 256) {
        scratch[K0DM_SC_CUR + e] = scratch[K0DM_SC_ERB + e];
        const int b0 = scratch[K0DM_SC_ERB + e] / 32;
        const int b1 = scratch[K0DM_SC_ERB + e + 1] / 32;
        for (int b = b0; b < b1; ++b) {
          if (b >= bcap) { atomicOr(pperr, 2); break; }
          sei[b] = e;
        }
      }
    }
  } else if (bid == 1) {
    if (tid < 64) {
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
      if (tau < T) {
        int k = incl - f;
        for (int s = 0; s < TOPK; ++s) {
          const int pe = pull_stage[((size_t)tau * TOPK + s) * 2 + 0];
          if (pe < 0) continue;
          const int row = pull_stage[((size_t)tau * TOPK + s) * 2 + 1];
          if (k >= T * world) { atomicOr(pperr, 8); break; }
          pull_src[(size_t)k * 2 + 0] = pe;
          pull_src[(size_t)k * 2 + 1] = row;
          ++k;
        }
      }
    }
  } else {
    const int w = (bid - 2) * (blockDim.x >> 6) + (tid >> 6);
    const int nw = (gridDim.x - 2) * (blockDim.x >> 6);
    for (int t = w; t < T_loc; t += nw) {
      unsigned char* prow = reinterpret_cast<unsigned char*>(part) + (size_t)t * 14336;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int c = 0; c < 14; ++c)
        *reinterpret_cast<uint4*>(prow + (c << 10) + (lane << 4)) = z;
      if (lane < K0DM_NG)
        sc_dst[(size_t)lane * (size_t)T_loc + t] = sc_stage[(size_t)t * K0DM_NG + lane];
    }
  }

  // Publish local plan work.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const unsigned int old =
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    volatile unsigned int* vb = gbar;
    long long spins = 0;
    while (*vb < target) {
      if (++spins > spin_limit) { atomicOr(pperr, 8192); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence();

  // Pad expert segments and scatter live pairs.
  {
    const int lo = cur * E, hi = lo + E;
    for (int idx = bid * 256 + tid; idx < E * 32; idx += gridDim.x * 256) {
      const int e = idx >> 5, k = idx & 31;
      const int row = scratch[K0DM_SC_ERB + e] + scratch[K0DM_SC_CNT + e] + k;
      if (row < scratch[K0DM_SC_ERB + e + 1]) {
        if (row >= PADMAX) { atomicOr(pperr, 2); continue; }
        sti[row] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
        swt[row] = 0.0f;
      }
    }
    const int pairs = T_loc * TOPK;
    for (int j = bid * 256 + tid; j < pairs; j += gridDim.x * 256) {
      const int e = recv_eid[j];
      if (e < lo || e >= hi) continue;
      const int pos = atomicAdd(&scratch[K0DM_SC_CUR + (e - lo)], 1);
      if (pos >= PADMAX) { atomicOr(pperr, 4096); continue; }
      const int row = j / TOPK;
      const int slot = j - row * TOPK;
      sti[pos] = (row & 0x00FFFFFF) | (slot << 24);
      swt[pos] = recv_wgt[j];
    }
  }

  // Reset the allocator and advance the epoch.
  __syncthreads();
  if (bid == 0 && tid == 0) {
    dest_counter[0] = 0u;
    sort_count[0] = epoch;
  }
}
