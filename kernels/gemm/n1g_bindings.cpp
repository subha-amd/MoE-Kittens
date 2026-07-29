// pybind module `k0_n1g`: the single-kernel fused expert GEMM.
//
// It is built as its own target so the shared library contains exactly one
// kernel, which keeps a disassembly of it unambiguous.

#include <pybind11/pybind11.h>

void bind_n1g_fused_moe(pybind11::module_& module);

PYBIND11_MODULE(k0_n1g, module) {
  module.doc() =
      "Single-kernel fused expert GEMM. The work count is read on device from "
      "num_valid_ids[0] (no by-value kernargs, no host-frozen scalar), and the "
      "activation tile is read in memory order — 8 fully-utilised cache lines per "
      "instruction — once per CTA into an XOR-swizzled LDS tile, through an "
      "`input` buffer descriptor bounded to T*7168.";
  bind_n1g_fused_moe(module);
}
