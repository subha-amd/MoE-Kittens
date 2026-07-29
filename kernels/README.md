# Kernel index

## `decode/` — B=64

| file | role |
|---|---|
| `k0d_mega.hip` | push dispatch, cross-rank synchronization, and destination sort in one 16-CTA launch |
| `k0d_combine.hip` | owner-pull combine with cross-rank synchronization at the head |
| `k0_region_kernels.hip` | BF16→FP8 quantization, route all-gather, gather, and combine |
| `k0d_push.hip` | the push half of `k0d_mega`, as its own kernel |
| `k0d_dsort.hip` | the destination-sort half of `k0d_mega`, as its own kernel |
| `k0d_fused.hip` | pull-path ingress collapse: route all-gather + plan + quant in one block |
| `k0d_gather_zero.hip` | pull-side gather with the partial-buffer clear folded in |
| `k0d_barrier.hip` | standalone cross-rank stream barrier |
| `k0d_phase_marker.hip` | replay-safe timestamps for in-graph phase profiling |

## `prefill/` — B=4096

| file | role |
|---|---|
| `k0pf3_qpush.hip` | origin quantization fused into a source push with doorbell synchronization |
| `k0pf4_dsort.hip` | destination sort with a hierarchical multi-CTA expert histogram |
| `k0pf_combine.hip` | owner-pull combine with direct P2P reads |
| `k0pf3_dsort.hip` | the single-CTA-count destination sort `k0pf4_dsort` replaces |
| `k0pf3_mega.hip` | `k0pf3_qpush` + `k0pf3_dsort` in one launch |
| `k0pf_gather.hip` | pull-side dispatch at fabric rate |
| `k0pf_quant.hip` | standalone origin quantization, one block per token |

## `gemm/`

| file | role |
|---|---|
| `n2_phase1.cpp` | W13 projection and quantized intermediate handoff |
| `n2_phase2.cpp` | split-N full-K W2 |
| `n1g_fused_moe.cpp` | the single-kernel fused GEMM the two-phase version forked from |
| `aiter_gfx950_fp8_layout.hpp` | the preshuffled FP8 weight layout both consume |
| `*_bindings.cpp` | pybind modules `k0_n2` and `k0_n1g` |

## `plan/`

Local plan construction for pull-based paths: `build_plan_prod` and `zero_partial`.

| file | role |
|---|---|
| `tile_plan.cpp` | the decode-scale plan |
| `k0pf_plan.cpp` | the same plan rebuilt for prefill scale |
| `k0pf_frozen_plan.cpp` | the plan with an explicit capture-stream handle |

## `hkp/`

Shared device primitives. Include `hkp.hpp` for all headers.

| header | what it owns |
|---|---|
| `hkp_topology.hpp` | peer address translation (one rule for both the MoRI and IRIS heap models) |
| `hkp_store.hpp` | vectorized posted peer stores and peer pulls |
| `hkp_alloc.hpp` | remote receive-row reservation and wave destination dedup |
| `hkp_sync.hpp` | monotonic-epoch doorbells, bounded fail-closed waits, local grid barrier |
| `hkp_sort.hpp` | destination counting-sort stages and the owner-combine CSR |
| `hkp_quant.hpp` | BF16→FP8 E4M3 group quantization, scale layouts, fused zero + transpose |
| `hkp_gemm.hpp` | persistent expert-GEMM task decomposition and pipeline policy |

- Flags and counters are zeroed once at setup and never reset. Epochs come from
  cumulative state, so graph replay advances the protocol with no host involvement.
- Publication pokes are relaxed. Ordering comes from a `__threadfence_system()` before the pokes.
- Waits are bounded and fail-closed: a volatile poll with `s_sleep` backoff and a spin
  limit; on timeout it sets an error bit and proceeds, and a downstream gate reads that bit.
