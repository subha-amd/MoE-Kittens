# MoE-Kittens

Custom expert-parallel MoE kernels for DeepSeek-R1 on 8x AMD MI350X (`gfx950`, CDNA4),
written on [HipKittens](https://github.com/HazyResearch/HipKittens), with the production
path they are measured against.

The production path is `MoRI EpDispatch -> AITER fused_moe -> MoRI EpCombine`.
This repository replaces all three stages with kernels we own, so that movement,
sorting and compute can be fused and rescheduled rather than treated as three
opaque library calls.

Two paths are complete and measured end to end:

| regime | path | versus production (aligned region, same run) |
|---|---|---|
| decode, B=64 | `k0d_mega` composition | **0.9781x warm p50** — parity, not a confirmed win |
| prefill, B=4096 | `pf4h` composition | **0.96865x warm p50**, p95 ratio also below 1.0 |

Both numbers are same-run, position-balanced, HIP-graph-mode comparisons against
production measured in the same process. Neither is compared against a historical
baseline. Read [docs/decode.md](docs/decode.md) and [docs/prefill.md](docs/prefill.md)
for what those numbers do and do not establish — in particular, the prefill result rests
on a corpus that was never formally accepted, and the decode result is parity rather than
a win.

## Repository map

```
kernels/
  decode/    the decode movement kernels; k0d_mega.hip is the headline one
  prefill/   the prefill movement kernels; k0pf3_qpush.hip + k0pf4_dsort.hip are the headline pair
  gemm/      the expert GEMM: n2_phase1/n2_phase2 (two-phase, write-once W2) and n1g (single kernel)
  plan/      local plan construction for the pull-based paths (pybind modules)
  hkp/       shared device primitives: peer addressing, posted stores, slot allocation,
             monotonic-epoch doorbells, destination sort stages, group quantization
build/       CMake for the GEMM modules, and build_all.sh for everything else
harness/     the A/B harness that composes and times the graphs, plus its reducers
baseline/    the production kernels these are measured against: the MoRI movement source
             and AITER's control layer + recovered gfx950 disassembly
docs/        mechanism write-ups, results, how to build and run, and the meeting decks
```

## The decode path — `k0d_mega`

Five graph nodes against production's eight:

```
BF16 -> FP8 quantization   kernels/decode/k0_region_kernels.hip
k0d_mega                   kernels/decode/k0d_mega.hip
n2_phase1                  kernels/gemm/n2_phase1.cpp
n2_phase2                  kernels/gemm/n2_phase2.cpp
k0d_combine                kernels/decode/k0d_combine.hip
```

`k0d_mega` fuses push dispatch, the cross-rank rendezvous and the destination-side sort
into one 16-CTA launch. `k0d_combine` folds the post-GEMM rendezvous into its own head, so
there is no standalone barrier kernel anywhere in the graph. See [docs/decode.md](docs/decode.md).

## The prefill path — `pf4h`

```
k0pf3_qpush                kernels/prefill/k0pf3_qpush.hip   (origin quant fused into the push)
k0pf4_dsort                kernels/prefill/k0pf4_dsort.hip   (hierarchical destination sort)
n2_phase1 / n2_phase2      kernels/gemm/
k0pf_combine               kernels/prefill/k0pf_combine.hip
```

`k0pf4_dsort` moves the per-expert histogram off a single CTA and onto every CTA, which
takes the critical rank's destination sort from 423 µs to 277 µs at p50. That is a real,
isolated mechanism win — and it is the *third* largest prefill problem. The largest is
cross-rank expert-load imbalance, which is still open. See [docs/prefill.md](docs/prefill.md).

## Building and running

You need ROCm with `hipcc`, a HipKittens checkout, MoRI, AITER, and an 8-GPU MI350X node.

```bash
HIPKITTENS_ROOT=/path/to/HipKittens build/build_all.sh ~/moe-kittens-build

MOE_KITTENS_BUILD=~/moe-kittens-build \
MOE_KITTENS_CORPUS=/path/to/decode/corpus \
harness/run_ab.sh decode production,k0d_mega 5
```

Full instructions, the environment variables, and the corpus format are in
[docs/running.md](docs/running.md).

## What the harness guarantees

Every arm passes a correctness gate against the captured reference on every replay, with a
frozen tolerance that is never widened: `max_abs <= 0.02`, `rel_L2 <= 0.01`, zero non-finite
values. Beyond that, each campaign runs epoch-poison replays, grouped and interleaved graph
stress, fail-closed wait controls, and deliberately wrong "poison" arms whose job is to FAIL
the gate — if a poison arm passes, the gate is not testing what it claims to test.

The production comparison is measured in the same process, on the same inputs, with the
same timed I/O boundaries, rotated position-balanced across replays.

## Meeting decks

- [docs/meetings/meeting-5-final.pdf](docs/meetings/meeting-5-final.pdf)
- [docs/meetings/meeting-4-hipkittens-x-iris.pdf](docs/meetings/meeting-4-hipkittens-x-iris.pdf)
