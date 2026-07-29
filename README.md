# MoE-Kittens

Expert-parallel MoE kernels for DeepSeek-R1 on 8x AMD MI350X (`gfx950`), built with
[HipKittens](https://github.com/HazyResearch/HipKittens).

The custom path replaces MoRI dispatch, AITER fused MoE, and MoRI combine with fused
movement, sorting, and GEMM kernels.

| workload | custom path | warm p50 vs. production |
|---|---|---:|
| decode, B=64 | `k0d_mega` | 0.9781x |
| prefill, B=4096 | `pf4h` | 0.96865x |

Results are same-process, position-balanced HIP graph comparisons. See
[decode results](docs/decode.md) and [prefill results](docs/prefill.md).

## Layout

- `kernels/decode`: decode dispatch, sorting, and combine kernels
- `kernels/prefill`: prefill dispatch, sorting, and combine kernels
- `kernels/gemm`: two-phase and fused expert GEMMs
- `kernels/hkp`: shared device primitives
- `kernels/plan`: plan construction for pull-based paths
- `harness`: graph composition, correctness checks, and benchmarks
- `baseline`: MoRI and AITER reference implementations
- `docs`: implementation notes, results, and setup instructions

## Pipelines

Decode:

```text
BF16 -> FP8 -> k0d_mega -> n2_phase1 -> n2_phase2 -> k0d_combine
```

Prefill:

```text
k0pf3_qpush -> k0pf4_dsort -> n2_phase1 -> n2_phase2 -> k0pf_combine
```

## Build and run

Requirements: ROCm with `hipcc`, HipKittens, MoRI, AITER, and an 8-GPU MI350X node.

```bash
HIPKITTENS_ROOT=/path/to/HipKittens build/build_all.sh ~/moe-kittens-build

MOE_KITTENS_BUILD=~/moe-kittens-build \
MOE_KITTENS_CORPUS=/path/to/decode/corpus \
harness/run_ab.sh decode production,k0d_mega 5
```

See [docs/running.md](docs/running.md) for configuration and input formats.
