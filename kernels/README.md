# Kernel index

Headline paths are marked ★. Everything else is a sibling stage or an earlier composition
that the harness still loads, and that is useful for attribution: to say a fusion helped,
you need the unfused version in the same run.

## `decode/` — B=64

| file | role |
|---|---|
| ★ `k0d_mega.hip` | push dispatch + rendezvous + destination sort, fused into one 16-CTA launch |
| ★ `k0d_combine.hip` | owner-pull combine with the all-rank rendezvous folded into its head |
| ★ `k0_region_kernels.hip` | `k0_region_quant_to_shmem` (the BF16→FP8 node of the headline graph), plus the pull-side route all-gather, gather and combine |
| `k0d_push.hip` | the push half of `k0d_mega`, as its own kernel |
| `k0d_dsort.hip` | the destination-sort half of `k0d_mega`, as its own kernel |
| `k0d_fused.hip` | pull-path ingress collapse: route all-gather + plan + quant in one block |
| `k0d_gather_zero.hip` | pull-side gather with the partial-buffer clear folded in |
| `k0d_barrier.hip` | standalone cross-rank stream barrier, for arms that do not fold it |
| `k0d_phase_marker.hip` | replay-safe timestamps for in-graph phase profiling |

## `prefill/` — B=4096

| file | role |
|---|---|
| ★ `k0pf3_qpush.hip` | origin quantization fused into a source push, with the folded doorbell rendezvous |
| ★ `k0pf4_dsort.hip` | destination sort with a hierarchical multi-CTA expert histogram |
| ★ `k0pf_combine.hip` | owner-pull combine, direct P2P reads, no stage copy |
| `k0pf3_dsort.hip` | the single-CTA-count destination sort `k0pf4_dsort` replaces |
| `k0pf3_mega.hip` | `k0pf3_qpush` + `k0pf3_dsort` in one launch |
| `k0pf_gather.hip` | pull-side dispatch at fabric rate |
| `k0pf_quant.hip` | standalone origin quantization, one block per token |

## `gemm/`

| file | role |
|---|---|
| ★ `n2_phase1.cpp` | W13 projection; hands the quantized intermediate activation to phase 2 through global |
| ★ `n2_phase2.cpp` | W2 as split-N full-K, so every output element is written once |
| `n1g_fused_moe.cpp` | the single-kernel fused GEMM the two-phase version forked from |
| `aiter_gfx950_fp8_layout.hpp` | the preshuffled FP8 weight layout both consume |
| `*_bindings.cpp` | pybind modules `k0_n2` and `k0_n1g` |

## `plan/`

Local plan construction for the pull-based paths — `build_plan_prod` and `zero_partial`.
The push paths do not need these: the sorted GEMM metadata is rebuilt destination-side from
the received expert ids instead.

| file | role |
|---|---|
| `tile_plan.cpp` | the decode-scale plan |
| `k0pf_plan.cpp` | the same plan rebuilt for prefill scale |
| `k0pf_frozen_plan.cpp` | the plan with an explicit capture-stream handle |

## `hkp/`

Shared device primitives distilled from the kernels above, so a refactor can adopt them
without rewriting behaviour. Include `hkp.hpp` for all of them.

| header | what it owns |
|---|---|
| `hkp_topology.hpp` | peer address translation (one rule for both the MoRI and IRIS heap models) |
| `hkp_store.hpp` | vectorized posted peer stores and peer pulls |
| `hkp_alloc.hpp` | remote receive-row reservation and wave destination dedup |
| `hkp_sync.hpp` | monotonic-epoch doorbells, bounded fail-closed waits, local grid barrier |
| `hkp_sort.hpp` | destination counting-sort stages and the owner-combine CSR |
| `hkp_quant.hpp` | BF16→FP8 E4M3 group quantization, scale layouts, fused zero + transpose |
| `hkp_gemm.hpp` | persistent expert-GEMM task decomposition and pipeline policy |

Three invariants these encode, which are easy to break and expensive to debug:

- **Flags and counters are zeroed once at setup and never reset.** Epochs come from
  cumulative state, so graph replay advances the protocol with no host involvement.
- **Publication pokes are relaxed.** Ordering comes from a `__threadfence_system()` executed
  *before* the pokes, never from the pokes themselves.
- **Waits are bounded and fail-closed**: a volatile poll with `s_sleep` backoff and a spin
  limit; on timeout it sets an error bit and proceeds, and a downstream gate reads that bit.
  A wait that can hang forever will eventually wedge the node.
