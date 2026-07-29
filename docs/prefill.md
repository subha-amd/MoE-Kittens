# The prefill path: `pf4h`

Prefill is B=4096 tokens per rank, 8 ranks, top-k 8, 32 local experts, hidden 7168.
A rank receives 12k–21k rows depending on how the routes land.

## Composition

```
k0pf3_qpush     origin quantization fused into a source push, plus the folded barrier
k0pf4_dsort     destination-side sort with a hierarchical multi-CTA expert histogram
n2_phase1       W13
n2_phase2       W2, write-once
k0pf_combine    owner-pull combine at fabric rate, no stage copy
```

Wired in [`harness/region_ab.py`](../harness/region_ab.py) as `pf4h_body`.
`pf4h` differs from its predecessor `pf3_pd` in exactly one place — the destination sort's
count phase. Same ingress, same GEMM, same barrier, same combine.

### `k0pf3_qpush`

[`kernels/prefill/k0pf3_qpush.hip`](../kernels/prefill/k0pf3_qpush.hip). One warp per source
token. It deduplicates the token's routes by destination, allocates one receive row per
distinct destination with a remote atomic, quantizes its BF16 row to FP8 **in registers**,
and posts the payload fire-and-forget through peer pointers. Because the quantization is
local and register-resident, no rank ever waits on a remote quantization.

Relative to pull-based prefill dispatch, this removes the route all-gather, plan build,
standalone quantization kernel, pre-dispatch barrier, pull, and scale transpose. The
destination rebuilds the sorted GEMM metadata.

### `k0pf4_dsort` — the change this arm isolates

[`kernels/prefill/k0pf4_dsort.hip`](../kernels/prefill/k0pf4_dsort.hip).

The predecessor gave the entire per-expert count to CTA 0. At prefill that is `T_loc * 8`
LDS atomics on a single CU — over 150k pairs on the hottest rank — and it ran long after
every other CTA had finished zeroing `part` and gone idle. Because that rank sets the
region's pace, the serial count *was* region time.

Here every CTA histograms a disjoint grid-stride slice into LDS, publishes its own `[E]`
counts to local scratch, and after one extra grid barrier CTA 0 reduces the `[E][gridDim]`
table into exactly the counts the padded scan already consumed. Integer addition is
associative, so the counts are bit-identical to the serial version.
Everything downstream is untouched.

## Results

Five independent processes, arms `production`, `pf3_pd`, `pf4h`, graph mode, aligned
`MAX(rank)` region time, warm and 1 GiB-scrubbed cold.

| arm | warm p50 | warm p95 | cold p50 | cold p95 |
|---|---:|---:|---:|---:|
| production | 8418.27 µs | 8561.21 µs | 8302.45 µs | 8399.52 µs |
| `pf3_pd` | 8285.69 µs | 8368.03 µs | 8248.31 µs | 8321.48 µs |
| **`pf4h`** | **8154.69 µs** | **8225.56 µs** | **8105.25 µs** | **8177.89 µs** |

| paired ratio | warm p50 | warm p95 | cold p50 | cold p95 |
|---|---:|---:|---:|---:|
| `pf4h / production` | 0.96865 | 0.98573 | 0.97472 | 0.99069 |
| `pf4h / pf3_pd` | 0.98291 | 0.99588 | 0.98234 | 0.99677 |

The ranges also separate: the worst `pf4h` warm p50 across the five
processes (8161.39 µs) is below the best production (8408.05 µs) and the best `pf3_pd`
(8272.49 µs). Every `pf4h` p95 ratio is below 1.0 against both comparators in both cache
conditions.

Correctness across all 40 rank artifacts: worst rel-L2 0.00682, worst max-abs 0.0078125,
zero non-finite values, zero timed-replay failures, zero `pperr`.

### The mechanism, isolated

Device-marker phase attribution of the destination sort, per rank:

| rank | `T_loc` | p50 before | p50 after | Δ p50 | p95 before | p95 after | Δ p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12031 | 197.52 | 146.64 | −50.88 | 265.27 | 186.28 | −78.99 |
| 1 | 18311 | 466.96 | 322.38 | −144.59 | 498.83 | 338.71 | −160.12 |
| 2 | 17937 | 344.52 | 208.81 | −135.71 | 354.28 | 213.26 | −141.02 |
| **3** | **19756** | **423.14** | **276.82** | **−146.32** | **450.90** | **308.96** | **−141.94** |
| 4 | 13482 | 252.19 | 190.71 | −61.47 | 283.42 | 221.73 | −61.69 |

Rank 3 is the critical rank. Its 146 µs reduction is what shows up as the ~140–146 µs
region delta against `pf3_pd` — roughly 1.7–1.8% of the complete region.

## Remaining costs

PF4H changes destination sorting; it does not address cross-rank expert-load imbalance.

| problem | evidence | approximate headroom | addressed? |
|---|---|---:|---|
| cross-rank expert-load imbalance | `n2` ranges 2.81–5.71 ms; cool ranks wait up to 3.1 ms at the barrier | perfect-balance model: ~1.24 ms / 15.7% of the region | no |
| destination sorting | 198–462 µs across ranks; critical rank 423 µs | ~140–146 µs measured | **yes** |
| qpush | 702–828 µs, already at ~200–215 GB/s | ~75 µs estimated | no |
| combine | 965–1014 µs, ~300 GB/s, little rank skew | relatively healthy | no |

### The dominant problem is load imbalance

The critical prefill rank receives 19,756 rows; the lightest receives 11,089. Captured
`group_tokens` across the eight ranks are `[21395, 38092, 39517, 41844, 25799, 19472,
36930, 39095]` against a per-rank entry budget of 32,768 — a max/min ratio of **2.15x**.
Consequently:

- critical-rank `n2` ≈ 5.71 ms, light-rank `n2` ≈ 2.81 ms;
- light ranks wait up to ~3.1 ms at the post-GEMM barrier;
- the whole region runs at the hottest rank's pace.

That 3.1 ms barrier wait is a *symptom*, not 3.1 ms of directly recoverable latency. The
balance model — fitted at 0.136507 µs per entry from rank 3's measured 5,712 µs `n2` — puts
the recoverable amount at about **1.24 ms, or 15.7% of the region**, from moving the hottest
rank toward the mean.

The load-balancing estimate is not a kernel result: a B=4 slot
budget proved insufficient (remote expert-segment counts by destination are
`[4, 0, 2, 1, 2, 5, 1, 1]`, so rank 5 needs a fifth), placement adds ~13% dispatch fan-out,
and remote-weight movement may be expensive. It is a route-derived upper-bound model that
needs same-run graph A/B confirmation once an implementation exists.

### Destination-sort scope

The destination sort runs on the critical rank. One CTA previously handled the expert
histogram while the others became idle. `pf4h` parallelizes that work without changing the
push, GEMM, synchronization, or combine. Cross-rank GEMM load imbalance remains the largest
measured prefill cost.
