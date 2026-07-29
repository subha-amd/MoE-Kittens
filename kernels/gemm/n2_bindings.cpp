// pybind module `k0_n2`: the two-phase expert GEMM.
//   n2_phase1  W13 projection, intermediate activation handed off through global
//   n2_phase2  split-N full-K W2 with a write-once epilogue

#include <pybind11/pybind11.h>

void bind_n2_phase1(pybind11::module_& module);
void bind_n2_phase2(pybind11::module_& module);

PYBIND11_MODULE(k0_n2, module) {
  module.doc() =
      "Two-phase expert GEMM. The W2 reduction is split-N full-K in registers, so "
      "every output element is written once — 8x less atomic traffic than a "
      "split-K epilogue. Phase 1 is the W13 projection with a global handoff of "
      "the quantized intermediate activation; phase 2 is full-K W2 with the "
      "wave-local transpose epilogue and an XCD-stable n-chunk map.";
  bind_n2_phase1(module);
  bind_n2_phase2(module);
}
