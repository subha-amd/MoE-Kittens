# The production baseline

Everything in this repository is measured against the same production path:

```
MoRI EpDispatch  ->  AITER fused_moe (FP8 block-scale)  ->  MoRI EpCombine
```

That exact sequence is the `production` arm in
[`harness/region_ab.py`](../harness/region_ab.py), and it runs in the same
process, on the same inputs, with the same timed I/O boundaries as every custom arm.

## MoRI movement kernels

[`baseline/mori/`](../baseline/mori/) is a readable snapshot of MoRI's JIT sources, taken
from the installed node package at
`/opt/venv/lib/python3.12/site-packages/mori/_jit-sources/`. The installed Python wrapper
lives separately at `.../mori/ops/dispatch_combine.py`.

The files worth reading:

| what | where |
|---|---|
| dispatch body | [`src/ops/dispatch_combine/intranode.hpp`](../baseline/mori/src/ops/dispatch_combine/intranode.hpp) |
| combine body | same file, further down |
| kernel registration | [`src/ops/kernels/ep_intranode.hip`](../baseline/mori/src/ops/kernels/ep_intranode.hip) |
| runtime kernel selection and launch | [`src/ops/dispatch_combine/launch.cpp`](../baseline/mori/src/ops/dispatch_combine/launch.cpp) |
| device peer addressing | [`include/mori/shmem/shmem_device_api.hpp`](../baseline/mori/include/mori/shmem/shmem_device_api.hpp) |

The custom kernels use two MoRI design details:

- **The remote-atomic slot allocator.** A dispatch allocates a receive row on the
  destination with a remote atomic add; the returned value *is* the arrival row, and
  arrival order is nondeterministic. Our push kernels do the same thing, which is why our
  destination sort also has to tolerate nondeterministic arrival order.
- **Per-destination deduplication.** A token routed to several experts on the same
  destination GPU is sent once, not once per expert slot. That is a structural fact, and it
  is why receive-buffer capacity is bounded by `world * T` rather than by `world * T * topk`.

The custom path differs in two places:

- MoRI's combine stages every received row into a local buffer before its barrier. Our
  combine reads the producer's published `part` row in place through a peer pointer, so that
  copy is absent from the custom path.
- MoRI's on-stream barrier uses a centralized two-phase coordinator. The custom barrier is
  folded into the kernel that needs it, with one parallel release poke per
  peer and a local flag spin.

Disassembly of MoRI's two hot movement kernels, recovered for gfx950, is in
[`baseline/aiter/disassembly/`](../baseline/aiter/disassembly/) alongside AITER's:
`ep_dispatch.gfx950.s` and `ep_combine.gfx950.s`.

## AITER expert kernel

[`baseline/aiter/`](../baseline/aiter/) holds AITER's control layer:

| what | where |
|---|---|
| Python orchestration and selector | [`aiter/fused_moe.py`](../baseline/aiter/aiter/fused_moe.py) |
| launch wrapper | [`csrc/asm_fmoe.cu`](../baseline/aiter/csrc/asm_fmoe.cu) |
| configuration table | [`asm_fmoe_configs.hpp`](../baseline/aiter/asm_fmoe_configs.hpp) |
| recovered gfx950 disassembly | [`disassembly/aiter_fmoe.gfx950.s`](../baseline/aiter/disassembly/aiter_fmoe.gfx950.s) |

AITER's production `fmoe_fp8_blockscale_g1u1` microkernel is a precompiled `.co` assembly
blob. The checkout contains the Python orchestration, the
selector, the launch wrapper and the configuration tables — but no readable HIP source for
the hot kernel. The disassembly above is what there is to read.

A fixed `.co` blob cannot be modified to fuse movement with compute. An editable kernel
allows the two to be combined and rescheduled.

On a live node, an editable AITER checkout typically sits at `~/aiter-src` inside the
container, installed editable, with the relevant paths being `aiter/fused_moe.py`,
`csrc/py_itfs_cu/asm_fmoe.cu`,
`aiter/jit/build/module_moe_fmoe_asm/blob/asm_fmoe_configs.hpp`, and
`hsa/gfx950/fmoe/silu/*.co`.

## The I/O contract the custom GEMM has to honour

The expert GEMM accepts exactly the production tensors, with no extra tensor derived on the
host and no host-frozen scalar:

```
out, input, gate, down, sorted_token_ids, sorted_weights, sorted_expert_ids,
num_valid_ids, input_scale, fc1_scale, fc2_scale
topk=8, fc_scale_blkn=128, fc_scale_blkk=128, activation=SiLU, block_size_M=32
```

| tensor | shape | dtype | notes |
|---|---|---|---|
| `input` | `[4096, 7168]` | FP8 E4M3 | live count `T = num_valid_ids[1]` |
| `input_scale` | `[4096, 56]` | FP32 | live prefix is **group-major**: `input_scale[k128_group * T + token]`, not the `[4096, 56]` capacity stride |
| `gate` (W13) | `[32, 4096, 7168]` | FP8 E4M3 | rows `[0, 2048)` gate, `[2048, 4096)` up |
| `fc1_scale` | `[32, 32, 56]` | FP32 | `[expert, N/128, K/128]` |
| `down` (W2) | `[32, 7168, 2048]` | FP8 E4M3 | N=7168 out, K=2048 intermediate |
| `fc2_scale` | `[32, 56, 16]` | FP32 | `[expert, N/128, K/128]` |
| `sorted_token_ids` | `[40984]` | INT32 | packed: token in the low 24 bits, top-k slot in the high 8 |
| `sorted_weights` | `[40984]` | FP32 | route weight per sorted row |
| `sorted_expert_ids` | `[1281]` | INT32 | expert per padded 32-row block |
| `num_valid_ids` | `[2]` | INT32 | `[0]` padded rows, `[1]` live token count `T` |
| `out` | `[4096, 7168]` | BF16 | pre-zeroed in `[0, T)` by the upstream sorting kernel |

A row is padding iff `token >= T` or `slot >= topk`; its `sorted_weights` is zero.

The reference is not bitwise reproducible — the production epilogue uses packed-BF16 atomics
that race, and production-versus-production `rel_L2` runs 0.0046–0.0060. Acceptance is
therefore tolerance-based, at a frozen `max_abs <= 0.02`, `rel_L2 <= 0.01`, zero non-finite.

## Licensing

`baseline/mori/` and `baseline/aiter/` are third-party sources included for reference and
reproducibility, under their upstream licenses. They are not our work.
