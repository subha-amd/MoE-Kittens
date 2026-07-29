// Scheduling helpers for persistent expert GEMMs.
// Work derives from padded rows in nvi[0]. The persistent task loop is grid-stride. Unroll-2
// prefetch indices clamp to the final K group. The LDS barrier is restricted to local memory.

#ifndef HKP_GEMM_HPP
#define HKP_GEMM_HPP

#include <hip/hip_runtime.h>

namespace hkp {

// task = sorted BlockM-row block x secondary chunk.
template <int Log2ChunksPerBlock, int BlockM = 32>
struct persistent_expert_gemm {
  static_assert(BlockM == 32, "the GEMM phases use BlockM=32");
  static constexpr int kChunksPerBlock = 1 << Log2ChunksPerBlock;

  struct task {
    int block;
    int chunk;
  };

  static __device__ __forceinline__ int num_tasks(const int* nvi) {
    return (((nvi[0] + (BlockM - 1)) >> 5) << Log2ChunksPerBlock);
  }

  static __device__ __forceinline__ task decode(int t) {
    return task{t >> Log2ChunksPerBlock, t & (kChunksPerBlock - 1)};
  }

  static __device__ __forceinline__ bool in_range(int t, const int* nvi) {
    return t < num_tasks(nvi);
  }
};

// Unroll-2 prefetch indices, clamped to the final group.
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

// CTA-shared A-tile XOR swizzle coordinates.
struct a_tile_swizzle {
  static __device__ __forceinline__ int fill_chunk(int a_chunk, int a_row) {
    return a_chunk ^ (a_row & 7);
  }
  static __device__ __forceinline__ int read_chunk0(int q, int r) { return q ^ (r & 7); }
  static __device__ __forceinline__ int read_chunk1(int q, int r) {
    return (q ^ (r & 7)) ^ 4;
  }
};

// Local-memory-only CTA barrier; it must not drain VMEM in the K loop.
__device__ __forceinline__ void lds_phase_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

}  // namespace hkp

#endif  // HKP_GEMM_HPP
