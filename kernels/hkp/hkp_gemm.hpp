// ================================================================================================
// hkp_gemm.hpp — scheduling abstraction over the persistent expert GEMM phases
// (kernels/gemm/n2_phase1.cpp and n2_phase2.cpp).
//
// SCOPE: this header owns ONLY the scheduling idiom — the persistent grid-stride task
// decomposition over a device-derived work count, the unroll-2 K-pipeline prefetch-index
// policy, the A-tile XOR swizzle coordinates, and the LDS phase barrier.  It deliberately
// does NOT own the MFMA machinery: the register tiles (kittens::rt_fp8e4m3 / rt_fl),
// mma_ABt, the bounded raw-buffer descriptors (make_srsrc with num_records derived from
// nvi), and the AGPR staging stay in n2_phase1/2 (HipKittens, see kittens.cuh) — refactors
// adopt these primitives WITHOUT rewriting those files.
//
// Invariants preserved:
//   * the work count is read ON DEVICE from nvi[0] — PADDED rows, never nvi[1]
//     (the readiness/task extent must derive from the same padded extent
//     the GEMM walks).  num_tasks = ceil(nvi[0]/BlockM) << Log2ChunksPerBlock.
//   * the grid is fixed persistent CTAs (256 = one per CU); the task loop is grid-stride.
//   * the K loop is unrolled by two with CLAMPED prefetch indices (a step past the end
//     re-reads the final group rather than forming a new address).
// ================================================================================================

#ifndef HKP_GEMM_HPP
#define HKP_GEMM_HPP

#include <hip/hip_runtime.h>

namespace hkp {

// Task decomposition: task = (sorted BlockM-row block) x (secondary chunk).  Phase 1 uses
// Log2ChunksPerBlock = 3 (8 intermediate chunks, n2_phase1.cpp); phase 2 uses 4
// (16 output n-chunks, n2_phase2.cpp).
template <int Log2ChunksPerBlock, int BlockM = 32>
struct persistent_expert_gemm {
  static_assert(BlockM == 32, "the GEMM phases use BlockM=32");
  static constexpr int kChunksPerBlock = 1 << Log2ChunksPerBlock;

  struct task {
    int block;   // sorted block b
    int chunk;   // secondary chunk (g or nc)
  };

  // Device-derived work count (n2_phase1.cpp, n2_phase2.cpp).
  static __device__ __forceinline__ int num_tasks(const int* nvi) {
    return (((nvi[0] + (BlockM - 1)) >> 5) << Log2ChunksPerBlock);
  }

  static __device__ __forceinline__ task decode(int t) {
    return task{t >> Log2ChunksPerBlock, t & (kChunksPerBlock - 1)};
  }

  // Grid-stride persistent loop bound check:
  //   for (int t = blockIdx.x; t < num_tasks(nvi); t += gridDim.x) { decode(t); ... }
  static __device__ __forceinline__ bool in_range(int t, const int* nvi) {
    return t < num_tasks(nvi);
  }
};

// Unroll-2 K-pipeline prefetch policy (n2_phase1.cpp, n2_phase2.cpp):
// A prefetches two groups ahead, B one group ahead, both clamped to the final group.
struct unroll2_k_policy {
  static __device__ __forceinline__ int next_a(int k, int k_groups) {
    return (k + 2 < k_groups) ? (k + 2) : (k_groups - 1);
  }
  static __device__ __forceinline__ int next_a_second(int k, int k_groups) {
    return (k + 3 < k_groups) ? (k + 3) : (k_groups - 1);
  }
  static __device__ __forceinline__ int next_b(int k, int k_groups) {
    return (k + 1 < k_groups) ? (k + 1) : (k_groups - 1);
  }
  static __device__ __forceinline__ int next_b_second(int k, int k_groups) {
    return (k + 2 < k_groups) ? (k + 2) : (k_groups - 1);
  }
};

// CTA-shared A-tile XOR swizzle coordinates (n2_phase1.cpp fill, 244-254 read;
// n2_phase2.cpp, 280-291).  Fill: thread owns row a_row = tid>>3, chunk
// a_chunk = tid&7, stored at chunk a_chunk ^ (a_row & 7).  Read: lane split q = lane>>4,
// r = lane&15; the two 16 B fragments of a row live at chunks q ^ (r&7) and that ^ 4.
struct a_tile_swizzle {
  static __device__ __forceinline__ int fill_chunk(int a_chunk, int a_row) {
    return a_chunk ^ (a_row & 7);
  }
  static __device__ __forceinline__ int read_chunk0(int q, int r) { return q ^ (r & 7); }
  static __device__ __forceinline__ int read_chunk1(int q, int r) {
    return (q ^ (r & 7)) ^ 4;
  }
};

// LDS phase barrier (n2_phase1.cpp, n2_phase2.cpp): LOCAL address space
// ONLY — the release/acquire pair is scoped "workgroup"/"local" so no vmcnt(0) reaches the
// K-loop pipeline (fact 2).
__device__ __forceinline__ void lds_phase_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace hkp

#endif  // HKP_GEMM_HPP
