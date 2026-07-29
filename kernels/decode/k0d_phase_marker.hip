// A graph-replay-safe phase marker for decode profiling.
//
// Captured HIP events on the current ROCm stack retain their capture-time timestamp instead of
// refreshing it on every graph replay.  This one-thread kernel writes the device's monotonic
// realtime counter into a fixed buffer, so every replay produces fresh boundary timestamps.
// A back-to-back marker graph is measured alongside profiled graphs to quantify launch overhead.

#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" __global__ void __launch_bounds__(1, 1) k0d_phase_marker(
    unsigned long long* __restrict__ stamps,
    int index) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    stamps[index] = __builtin_amdgcn_s_memrealtime();
  }
}

// Run only outside the measured regional graphs.  It establishes the realtime-counter period
// against an ordinary HIP event instead of assuming a fixed gfx clock or a nominal 100 MHz timer.
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
