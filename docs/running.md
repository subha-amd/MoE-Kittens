# Building and running

## What you need

- An 8-GPU MI350X node (`gfx950`, device id `0x75a0`). Identify the part from the host with
  `rocminfo` / `amd-smi` / `rocm-smi`; inside a container the GPU reports a generic name.
- ROCm 7.2+ with `hipcc`, in a container that has the GPUs mapped.
- A [HipKittens](https://github.com/HazyResearch/HipKittens) checkout. The GEMM includes
  `kittens.cuh`, so `HIPKITTENS_ROOT/include/kittens.cuh` must exist.
- MoRI installed (the harness imports `mori` and uses its JIT loader and symmetric heap)
  and AITER installed (the `production` arm calls `aiter.fused_moe`).
- A captured route/weight corpus for the regime you want to run (see below).

## Build

```bash
HIPKITTENS_ROOT=/path/to/HipKittens build/build_all.sh ~/moe-kittens-build
```

This is a pure cross-compile for gfx950 and does not need GPUs. It produces:

| artifact | built how | contents |
|---|---|---|
| `k0_n1g*.so` | CMake | single-kernel expert GEMM |
| `k0_n2*.so` | CMake | two-phase expert GEMM (`n2_phase1`, `n2_phase2`) |
| `tile_plan*.so`, `k0pf_plan*.so`, `k0pf_frozen_plan*.so` | `hipcc` + pybind | local plan construction |
| `hsaco/k0pf_{gather,combine,quant}.hsaco` | `hipcc --genco` | prefill movement kernels loaded as prebuilt code objects |

The `.hip` kernels under `kernels/decode/` and `kernels/prefill/` are not built here.
The harness JIT-compiles them at run time through MoRI's loader, which is how they pick up
MoRI's include paths and their own device shmem globals. Every separately loaded HIP module
carries its own `globalGpuStates`, so each must be loaded with `init_shmem=True` or
`ShmemPtrP2p` returns 0 and you get a fault at address 0.

The build script also prints the register/spill remarks from
`-Rpass-analysis=kernel-resource-usage`. The GEMM requires zero spills and zero scratch.

## Run

```bash
MOE_KITTENS_BUILD=~/moe-kittens-build \
MOE_KITTENS_CORPUS=/path/to/decode/tp1_dp8_ep8/B64/<capture> \
harness/run_ab.sh decode production,k0d_mega 5

MOE_KITTENS_BUILD=~/moe-kittens-build \
MOE_KITTENS_CORPUS=/path/to/prefill/tp1_dp8_ep8/B4096/<capture> \
harness/run_ab.sh prefill production,pf3_pd,pf4h 5
```

`run_ab.sh` launches `torchrun --standalone --nnodes=1 --nproc_per_node=8` inside the
container, once per run, and refuses to start if another distributed job is present or if
any GPU reports non-zero utilization.

### Arms

Select with `K0_ARMS` (a comma-separated list). `run_ab.sh` passes it through.

| arm | regime | what it is |
|---|---|---|
| `production` | both | MoRI dispatch -> AITER `fused_moe` -> MoRI combine |
| `k0d_mega` | decode | primary decode path |
| `k0d_pd` | decode | same, with push and destination sort as two kernels |
| `k0d_s`, `k0d_sq`, `k0d_sq_b2f`, `k0d_f` | decode | earlier pull-based decode compositions |
| `pf4h` | prefill | primary prefill path |
| `pf3_pd` | prefill | same, with the single-CTA destination sort |
| `pf3_mega` | prefill | push and sort fused into one launch |
| `pf_*`, `pf2_*` | prefill | earlier plan/pull-based prefill compositions |
| `frozen_n2`, `frozen_n2r` | both | pull dispatch and combine, custom GEMM |
| `compute_swap*` | both | keep MoRI transport, swap only the GEMM |
| `fp_*` | both | weight-footprint measurements; outputs are not used for numerical comparison |

Decode arms refuse to run at prefill shape and vice versa.

### Key environment variables

| variable | meaning |
|---|---|
| `K0_PYBIND_DIR` | directory holding the built `.so` modules |
| `K0_PF_HSACO_DIR` | directory holding the prebuilt `.hsaco` files |
| `K0_HIP_SRC` | path to `kernels/decode/k0_region_kernels.hip` |
| `K0D_SRC_DIR` | `kernels/decode` |
| `K0PF3_SRC_DIR` | `kernels/prefill` |
| `K0_HKP_DIR` | `kernels/hkp` (installed as flat siblings for the JIT) |
| `K0_MORI_KERNELS_DIR` | MoRI's JIT kernel directory, where sources are staged for compilation |
| `K0_REGION_CORPUS` | the captured corpus to replay |
| `K0_ROUTE_SWAP_CORPUS` | a *different* route, replayed through the captured graph to prove nothing was frozen at capture time |
| `K0_ARMS` | arms to run |
| `K0_T`, `K0_T_LOC_MAX`, `K0_PADMAX`, `K0_MAXTOK` | regime shape and capacities |
| `K0_NTIMED` | timed rotations per arm (default 100) |
| `K0_STRESS`, `K0_XSTRESS` | grouped and interleaved graph-stress replay counts |
| `MORI_GPU_ARCHS=gfx950` | required |
| `HSA_XNACK=1` | required |

Regime shapes, as set by `run_ab.sh`:

| | decode | prefill |
|---|---:|---:|
| `T` | 64 | 4096 |
| `T_LOC_MAX` | 512 | 40960 |
| `PADMAX` | 5088 | 263136 |

`T_LOC_MAX` is the number of rows a rank receives, bounded by `world * T`, because dispatch
deduplicates per destination. This differs from `world * T * topk`; the values happen to be
equal only when `world == topk == 8`; use `world * T`.

## The corpus

A corpus is a directory of per-rank raw tensor dumps:

```
<corpus>/rank-<r>/03-gate.raw.bin
                  04-down.raw.bin
                  10-fc1_scale.raw.bin
                  11-fc2_scale.raw.bin
                  12-hidden_states.raw.bin
                  13-topk_ids.raw.bin
                  14-topk_weights.raw.bin
                  15-moe_out.raw.bin
```

`15-moe_out` is the reference output the correctness gate compares against. The router
outputs (`13`, `14`) are loaded directly into the symmetric route buffers that both the
production arm and every candidate arm consume, so a changed route cannot be hidden behind
an out-of-graph staging copy.

Captures come from a live DeepSeek-R1 serving run at TP1/DP8/EP8. Capture nodes must use
stock R1 rather than an MXFP4 preview checkpoint.

For decode, `harness/synthetic_routes.py` can generate synthetic route families as an
alternative to a captured route (decode shape only: world 8, T 64, topk 8, E 32); set
`K0_SYNTH_ROUTE`.

## Reducing results

Each run writes one JSON per rank into the output directory.

```bash
python3 harness/summarize_k0d.py  <run-dir> [...]   # decode
python3 harness/summarize_pf4h.py <run-dir> [...]   # prefill
```

The primary metric is the aligned `MAX(rank)` region time, reported as a paired ratio
against production computed per process, with the median and range across processes. Both
warm and 1 GiB-scrubbed cold conditions are reported.

## Shared-node rules

`run_ab.sh` enforces the first two shared-node rules:

1. Check for another `mpirun`/`torchrun` before launching. Never start two 8-GPU jobs.
2. All 8 GPUs must be idle before a timed run, or the numbers are noise.
3. Use detached `setsid` plus `timeout` for long runs.
4. Do not `SIGKILL` a job; leaked GPU IPC can wedge the whole node.
5. Never touch another tenant's container, processes or disk. Never `docker system prune`.
6. Large data goes on the data volume, never the root filesystem.
