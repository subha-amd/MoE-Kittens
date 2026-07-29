// ================================================================================================
// k0pf3_mega.hip — prefill megakernel: k0pf3_qpush and k0pf3_dsort in ONE launch.
//
// Fused-quant plan-free source push, then a grid barrier and doorbell, then the destination sort
// and plan build.  Two internal grid barriers replace the launch boundary; the rendezvous and
// every memory contract match the two-kernel pair, whose headers document the phases in full.
//
// PHASE MAP:
//   M0  the qpush body: warp per token — dedup, slot allocation, fused BF16 -> FP8 quantization,
//       posted payload stores, owner-side pull_stage / pull_cnt.
//   M1  grid barrier 1 (system fences): each CTA's thread 0 issues __threadfence_system then adds
//       its arrival to gbar; the last arriver bumps recv_doorbell[cur] on every rank.  This is
//       the qpush tail with gbar playing the arrival counter's role.
//   M2  the dsort doorbell wait: every CTA's threads tid < world spin on
//       recv_doorbell[tid] >= epoch; acquire fence; T_loc = dest_counter[0]; range check.
//   M3  the dsort task split: CTA0 count, padded scan, nvi/sei/cursors; CTA1 CSR scan into
//       pull_ptr; CTAs 2.. zero part[0, T_loc) and transpose sc_stage into sc_dst.
//   M4  grid barrier 2 (device fences; scratch, nvi and pull_ptr are local to this GPU).
//   M5  padding sentinel, atomic-cursor scatter, pull_src compaction.
//   M6  tail: CTA0 thread 0 resets dest_counter and advances sort_count.
//
// CROSS-EPOCH SAFETY is identical to the two-kernel path: the downstream barrier closes the
// forward hazard and the M2 doorbell wait closes the reverse hazard.
//
// GRID: launch with the qpush grid (many CTAs x 256).  The sort phases use per-CTA task splits,
// so any grid >= 3 is correct; >= 64 is recommended so the part zero saturates store bandwidth.
// One consequence of fusing: the M1/M2 spin phases idle the WHOLE grid during cross-rank push
// skew, whereas the split version idles only the sort kernel.  The idle is never larger.
//
// pperr bits match the two-kernel pair.
// ================================================================================================

#define MORI_SHMEM_NO_STATIC_INIT
#define MORI_SHMEM_ENABLE_WEAK_GLOBAL_GPU_STATES
#include "mori/shmem/shmem.hpp"

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstddef>
#include <cstdint>

#define K0PM_MAXE 64
#define K0PM_K 7168
#define K0PM_NG 56
#define K0PM_SC_ERB 0
#define K0PM_SC_CNT 80
#define K0PM_SC_CUR 160

__device__ __forceinline__ float k0pm_bf16_to_f32(unsigned short b) {
  return __uint_as_float((unsigned int)b << 16);
}

__device__ __forceinline__ int k0pm_block_incl_scan(int v, int* tmp) {
  const int tid = threadIdx.x;
  tmp[tid] = v;
  __syncthreads();
  for (int off = 1; off < 256; off <<= 1) {
    const int add = (tid >= off) ? tmp[tid - off] : 0;
    __syncthreads();
    tmp[tid] += add;
    __syncthreads();
  }
  const int r = tmp[tid];
  __syncthreads();
  return r;
}

extern "C" __global__ void __launch_bounds__(256, 8) k0pf3_mega(
    const int* __restrict__ my_ids,              // LOCAL [T*TOPK]
    const float* __restrict__ my_wgt,            // LOCAL [T*TOPK]
    const unsigned short* __restrict__ hidden,   // LOCAL [T, 7168] bf16
    unsigned char* __restrict__ a_dst,           // SYMMETRIC [T_loc_max, 7168] fp8
    float* __restrict__ sc_stage,                // SYMMETRIC [T_loc_max, 56] row-major
    int* __restrict__ recv_eid,                  // SYMMETRIC [T_loc_max, TOPK]
    float* __restrict__ recv_wgt,                // SYMMETRIC [T_loc_max, TOPK]
    unsigned int* __restrict__ dest_counter,     // SYMMETRIC [1]
    unsigned int* __restrict__ recv_doorbell,    // SYMMETRIC [world]
    int* __restrict__ pull_stage,                // LOCAL [T*TOPK, 2]
    int* __restrict__ pull_cnt,                  // LOCAL [T]
    int* __restrict__ sti,                       // LOCAL [PADMAX]
    float* __restrict__ swt,                     // LOCAL [PADMAX]
    int* __restrict__ sei,                       // LOCAL [PADMAX/32]
    int* __restrict__ nvi,                       // LOCAL [2]
    int* __restrict__ pull_ptr,                  // LOCAL [T+1]
    int* __restrict__ pull_src,                  // LOCAL [T*world, 2]
    float* __restrict__ sc_dst,                  // LOCAL group-major flat
    unsigned short* __restrict__ part,           // LOCAL [MROWS, 7168] bf16
    int* __restrict__ scratch,                   // LOCAL [256]
    unsigned int* __restrict__ gbar,             // LOCAL [1]: cumulative grid barrier counter
    unsigned int* __restrict__ sort_count,       // LOCAL [1]: epoch counter
    int* __restrict__ pperr, long long spin_limit,
    int T, int TOPK, int E, int cur, int world, int T_loc_max, int PADMAX) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int lane = tid & 63;

  if (TOPK != 8 || E > K0PM_MAXE || world > 8 || T_loc_max > 0x00FFFFFF || gridDim.x < 3) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 131072);
    return;
  }

  const unsigned int epoch = sort_count[0] + 1u;

  // ================= M0: fused-quant plan-free push (== k0pf3_qpush body) =================
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
        if (row >= (unsigned int)T_loc_max) atomicOr(pperr, 65536);
      }

      if (s < TOPK) {
        pull_stage[((size_t)tau * TOPK + s) * 2 + 0] = primary ? dest : -1;
        pull_stage[((size_t)tau * TOPK + s) * 2 + 1] = primary ? (int)row : -1;
      }
      if (lane == 0) pull_cnt[tau] = fanout;

      if (fanout == 0) continue;

      const unsigned short* hrow = hidden + (size_t)tau * K0PM_K;
      uint4 v[7];
      float sc_arr[7];
#pragma unroll
      for (int c = 0; c < 7; ++c) {
        const uint4 raw0 = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned char*>(hrow) + (c << 11) + (lane << 5));
        const uint4 raw1 = *reinterpret_cast<const uint4*>(
            reinterpret_cast<const unsigned char*>(hrow) + (c << 11) + (lane << 5) + 16);
        const unsigned short* h16 = reinterpret_cast<const unsigned short*>(&raw0);
        const unsigned short* h16b = reinterpret_cast<const unsigned short*>(&raw1);
        float f[16];
        float a = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          f[e] = k0pm_bf16_to_f32(h16[e]);
          a = fmaxf(a, fabsf(f[e]));
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          f[8 + e] = k0pm_bf16_to_f32(h16b[e]);
          a = fmaxf(a, fabsf(f[8 + e]));
        }
#pragma unroll
        for (int o = 1; o <= 4; o <<= 1) a = fmaxf(a, __shfl_xor(a, o, 64));
        const float scale = fmaxf(a, 1.0e-6f) / 448.0f;
        if ((lane & 7) == 0) sc_arr[c] = scale;

        union { unsigned char b[16]; uint4 v; } qo;
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          const float q = fminf(fmaxf(f[e] / scale, -448.0f), 448.0f);
          qo.b[e] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3);
        }
        v[c] = qo.v;
      }

      const uint4 ep0 = make_uint4((unsigned int)__shfl(eid, 0), (unsigned int)__shfl(eid, 1),
                                   (unsigned int)__shfl(eid, 2), (unsigned int)__shfl(eid, 3));
      const uint4 ep1 = make_uint4((unsigned int)__shfl(eid, 4), (unsigned int)__shfl(eid, 5),
                                   (unsigned int)__shfl(eid, 6), (unsigned int)__shfl(eid, 7));
      const float4 wp0 = make_float4(__shfl(wgt, 0), __shfl(wgt, 1), __shfl(wgt, 2), __shfl(wgt, 3));
      const float4 wp1 = make_float4(__shfl(wgt, 4), __shfl(wgt, 5), __shfl(wgt, 6), __shfl(wgt, 7));

      unsigned long long pm = pmask;
#pragma unroll 1
      for (int j = 0; j < fanout; ++j) {
        const int src_lane = __ffsll((long long)pm) - 1;
        pm &= pm - 1;
        const int pe = __shfl(dest, src_lane);
        const int drow_id = __shfl((int)row, src_lane);
        if (drow_id < 0 || drow_id >= T_loc_max) continue;
        unsigned char* drow = reinterpret_cast<unsigned char*>(mori::shmem::ShmemPtrP2p(
            (uint64_t)(a_dst + (size_t)drow_id * K0PM_K), pe));
#pragma unroll
        for (int c = 0; c < 7; ++c)
          *reinterpret_cast<uint4*>(drow + (c << 10) + (lane << 4)) = v[c];
        // Direct owner-lane stores; see k0pf3_qpush.hip.
        if ((lane & 7) == 0) {
          float* dsc = reinterpret_cast<float*>(mori::shmem::ShmemPtrP2p(
              (uint64_t)(sc_stage + (size_t)drow_id * K0PM_NG), pe));
#pragma unroll
          for (int c = 0; c < 7; ++c) dsc[c * 8 + (lane >> 3)] = sc_arr[c];
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

  // ============== M1: grid barrier #1 (system release; last arriver pokes doorbells) ======
  __syncthreads();
  if (tid == 0) {
    __threadfence_system();   // release this CTA's payload stores system-wide BEFORE arrival
    const unsigned int old =
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
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
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    volatile unsigned int* vb = gbar;
    long long spins = 0;
    while (*vb < target) {
      if (++spins > spin_limit) { atomicOr(pperr, 2097152); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();

  // ================= M2: doorbell wait (== dsort P0, the cross-epoch part guard) ==========
  if (tid < world) {
    volatile const unsigned int* vf = recv_doorbell;
    long long spins = 0;
    while (vf[tid] < epoch) {
      if (++spins > spin_limit) { atomicOr(pperr, 262144); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence_system();   // acquire: recv payloads, dest_counter, and the part-reuse proof

  const int T_loc = (int)dest_counter[0];
  if (T_loc < 0 || T_loc > T_loc_max) {
    if (bid == 0 && tid == 0) atomicOr(pperr, 524288);
    return;
  }

  // ================= M3: task-split plan work (== dsort P1) ===============================
  if (bid == 0) {
    __shared__ int s_cnt[K0PM_MAXE];
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
        scratch[K0PM_SC_ERB + e] = acc;
        scratch[K0PM_SC_CNT + e] = s_cnt[e];
        acc += ((s_cnt[e] + 31) / 32) * 32;
      }
      scratch[K0PM_SC_ERB + E] = acc;
      nvi[0] = acc;
      nvi[1] = T_loc;
    }
    __syncthreads();
    {
      const int bcap = PADMAX / 32;
      for (int e = tid; e < E; e += 256) {
        scratch[K0PM_SC_CUR + e] = scratch[K0PM_SC_ERB + e];
        const int b0 = scratch[K0PM_SC_ERB + e] / 32;
        const int b1 = scratch[K0PM_SC_ERB + e + 1] / 32;
        for (int b = b0; b < b1; ++b) {
          if (b >= bcap) { atomicOr(pperr, 4194304); break; }
          sei[b] = e;
        }
      }
    }
  } else if (bid == 1) {
    __shared__ int s_tmp[256];
    __shared__ int s_carry;
    if (tid == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < T; base += 256) {
      const int tau = base + tid;
      const int f = (tau < T) ? pull_cnt[tau] : 0;
      const int incl = k0pm_block_incl_scan(f, s_tmp);
      if (tau < T) pull_ptr[tau] = s_carry + (incl - f);
      __syncthreads();
      if (tid == 255) s_carry += incl;
      __syncthreads();
    }
    if (tid == 0) pull_ptr[T] = s_carry;
  } else {
    const int w = (bid - 2) * (blockDim.x >> 6) + (tid >> 6);
    const int nw = (gridDim.x - 2) * (blockDim.x >> 6);
    for (int t = w; t < T_loc; t += nw) {
      unsigned char* prow = reinterpret_cast<unsigned char*>(part) + (size_t)t * 14336;
      const uint4 z = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
      for (int c = 0; c < 14; ++c)
        *reinterpret_cast<uint4*>(prow + (c << 10) + (lane << 4)) = z;
      if (lane < K0PM_NG)
        sc_dst[(size_t)lane * (size_t)T_loc + t] = sc_stage[(size_t)t * K0PM_NG + lane];
    }
  }

  // ============== M4: grid barrier #2 (device scope; publishes scratch/nvi/pull_ptr) ======
  __syncthreads();
  if (tid == 0) {
    __threadfence();   // device release
    const unsigned int old =
        __hip_atomic_fetch_add(gbar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    const unsigned int target = (old / gridDim.x + 1u) * gridDim.x;
    volatile unsigned int* vb = gbar;
    long long spins = 0;
    while (*vb < target) {
      if (++spins > spin_limit) { atomicOr(pperr, 2097152); break; }
      __builtin_amdgcn_s_sleep(4);
    }
  }
  __syncthreads();
  __threadfence();   // device acquire

  // ================= M5: pad sentinel + atomic-cursor scatter + CSR fill ==================
  {
    const int lo = cur * E, hi = lo + E;
    for (int idx = bid * 256 + tid; idx < E * 32; idx += gridDim.x * 256) {
      const int e = idx >> 5, k = idx & 31;
      const int row = scratch[K0PM_SC_ERB + e] + scratch[K0PM_SC_CNT + e] + k;
      if (row < scratch[K0PM_SC_ERB + e + 1]) {
        if (row >= PADMAX) { atomicOr(pperr, 4194304); continue; }
        sti[row] = (T_loc & 0x00FFFFFF) | (TOPK << 24);
        swt[row] = 0.0f;
      }
    }
    const int pairs = T_loc * TOPK;
    for (int j = bid * 256 + tid; j < pairs; j += gridDim.x * 256) {
      const int e = recv_eid[j];
      if (e < lo || e >= hi) continue;
      const int pos = atomicAdd(&scratch[K0PM_SC_CUR + (e - lo)], 1);
      if (pos >= PADMAX) { atomicOr(pperr, 1048576); continue; }
      const int row = j / TOPK;
      const int slot = j - row * TOPK;
      sti[pos] = (row & 0x00FFFFFF) | (slot << 24);
      swt[pos] = recv_wgt[j];
    }
    for (int tau = bid * 256 + tid; tau < T; tau += gridDim.x * 256) {
      int k = pull_ptr[tau];
      for (int s = 0; s < TOPK; ++s) {
        const int pe = pull_stage[((size_t)tau * TOPK + s) * 2 + 0];
        if (pe < 0) continue;
        const int row = pull_stage[((size_t)tau * TOPK + s) * 2 + 1];
        if (k >= T * world) { atomicOr(pperr, 1048576); break; }
        pull_src[(size_t)k * 2 + 0] = pe;
        pull_src[(size_t)k * 2 + 1] = row;
        ++k;
      }
    }
  }

  // ================= M6: tail — reset allocator, advance epoch ============================
  __syncthreads();
  if (bid == 0 && tid == 0) {
    dest_counter[0] = 0u;
    sort_count[0] = epoch;
  }
}
