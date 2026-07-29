// Replay-safe decode phase marker. Captured HIP events may retain their capture-time timestamp,
// so this kernel records the device's monotonic realtime counter on each replay.

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" __global__ void __launch_bounds__(1, 1) k0d_phase_marker(
    unsigned long long* __restrict__ stamps,
    int index) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    stamps[index] = __builtin_amdgcn_s_memrealtime();
  }
}

// Calibrate the realtime-counter period against a HIP event outside profiled graphs.
extern "C" __global__ void __launch_bounds__(1, 1) k0d_phase_timer_calibration(
    unsigned long long* __restrict__ stamps,
    unsigned long long target_ticks) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long begin = __builtin_amdgcn_s_memrealtime();
    stamps[0] = begin;
    while ((__builtin_amdgcn_s_memrealtime() - begin) < target_ticks) {
      __builtin_amdgcn_s_sleep(1);
    }
    stamps[1] = __builtin_amdgcn_s_memrealtime();
  }
}
