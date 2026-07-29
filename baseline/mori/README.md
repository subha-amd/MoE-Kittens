# MoRI movement source (EpDispatch / EpCombine)

A readable snapshot of the device-kernel source for the two production cross-GPU movement
kernels in the DeepSeek-R1 MoE region: `ep_dispatch` and `ep_combine`. MoRI JIT-compiles
these at run time from this tree into `~/.mori/jit/*.hsaco`.

This is the readable source behind the movement the custom kernels are compared against.
It is **not** AITER's expert GEMM, which ships only as a precompiled `.co` blob — see
`../aiter/` for AITER's control layer and the recovered disassembly.

## Provenance

Pulled from a serving node at
`/opt/venv/lib/python3.12/site-packages/mori/_jit-sources/`, excluding `3rdparty/`.
The host-side Python wrapper is not part of `_jit-sources`; it is the installed
`mori/ops/dispatch_combine.py` (`EpDispatchCombineOp`).

## The two production kernels

With the default configuration (`kernel_type=IntraNode`, `use_external_inp_buf=True`,
`block_num=80`) exactly two device kernels run, one per collective. There is **no separate
barrier kernel** — the 8-GPU rendezvous is fused in-kernel, which is why it does not show up
as its own launch in a profile.

| leg | entry symbol | body | registered | launched |
|---|---|---|---|---|
| dispatch | `EpDispatchIntraNodeKernel_bf16` | `src/ops/dispatch_combine/intranode.hpp` | `src/ops/kernels/ep_intranode.hip` | `src/ops/dispatch_combine/launch.cpp` |
| combine | `EpCombineIntraNodeKernel_bf16_nop2p` | `src/ops/dispatch_combine/intranode.hpp` | `src/ops/kernels/ep_intranode.hip` | `src/ops/dispatch_combine/launch.cpp` |

The in-kernel cross-device barrier — combine's fused rendezvous — sits near the top of
`intranode.hpp`.

## Supporting primitives

- `include/mori/application/memory/symmetric_memory.hpp` — the symmetric heap and P2P peer
  pointers. `SymmMemObjPtr::GetAs<T*>(pe)` returns peer `pe`'s pointer over xGMI; a "remote
  store" is an ordinary vectorized store through it.
- `include/mori/core/transport/p2p/device_primitives.hpp` — `core::WarpCopy` (128-bit `uint4`
  vectorized push) and `core::WarpAccum` (atomic-free local reduce). The `rdma/` and `sdma/`
  variants are the multi-node transports, not the intra-node path.
- `src/ops/dispatch_combine/common.hpp` — `SendBufSlotOffset(config, pe, slot)`, the
  producer-keyed collision-free staging-slot layout that lets combine's up-to-8 producers
  write disjoint stripes with no atomics.

## Which files are the intra-node path

Only the intra-node path runs at EP8 on a single node. The `internode*`, `low_latency*`,
`*_ll` and `async` families are the RDMA multi-node paths — and they are the only place a
*separate* combine sync barrier is launched, which is a common source of confusion when
reading a profile.
