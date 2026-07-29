# The decode path: `k0d_mega`

Decode is B=64 tokens per rank, 8 ranks, top-k 8, 32 local experts, hidden 7168,
intermediate 2048, FP8 E4M3 block-scaled weights.

## Composition

Five graph nodes, against production's eight:

```
BF16 -> FP8 quantization
  -> k0d_mega            (push dispatch + rendezvous + destination sort, fused)
  -> n2_phase1           (W13)
  -> n2_phase2           (W2, write-once)
  -> k0d_combine         (rendezvous + owner-pull reduction, fused)
```

Wired in [`harness/region_ab.py`](../harness/region_ab.py) as `k0d_mega_body`:

```python
def k0d_mega_body(stream):
    _quant(stream)
    _k0d_mega_ingress(stream)
    _n2_nozero_pd(stream)
    _k0d_combine_pd(stream)
```

### 1. Quantization

[`kernels/decode/k0_region_kernels.hip`](../kernels/decode/k0_region_kernels.hip),
`k0_region_quant_to_shmem`.

Converts the 64 local decode tokens from BF16 into the FP8 K128 representation and the
scales consumed by the dispatch and the GEMM, writing both directly into the symmetric
staging buffers.

This stays a separate graph node deliberately. Fusing decode quantization into a single
CTA alongside the ingress was built and measured, and it was substantially slower.

### 2. `k0d_mega` — fused ingress

[`kernels/decode/k0d_mega.hip`](../kernels/decode/k0d_mega.hip). Launch: 16 CTAs x 256
threads. It fuses what was previously `k0d_push` followed by `k0d_dsort` into one
persistent grid.

**M1, push dispatch.** Each wavefront handles one token and its eight top-k routes:
load the eight expert ids and weights; deduplicate routes that share a destination GPU;
atomically allocate one receive row on each distinct destination; push the token's FP8
activation, scales, expert ids and route weights straight into that GPU's symmetric
buffers; record the metadata the owner-pull combine will need. The important property is
that one activation row is sent once per destination GPU, not once per expert slot.

**M2–M3, publish and rendezvous.** After all local CTAs have posted their remote writes,
each CTA issues a system release fence, the final CTA publishes completion to every rank's
doorbell, every CTA waits until all source ranks have completed the current epoch, and an
acquire fence makes the payload visible. There is no standalone barrier kernel, but the
synchronization has not disappeared — it is folded in here. The wait doubles as the
cross-epoch guard that makes it safe to reuse the receive buffers on the next replay.

**M4, destination-side metadata.** CTA 0 counts rows per local expert, computes the
BM32-padded expert segments, seeds the scatter cursors, and produces `sei` and `nvi`.
CTA 1 builds `pull_ptr` / `pull_src` for the combine. CTAs 2–15 clear the live `part` rows
and transpose the FP8 scales.

**M5–M7, sort and finish.** After an internal grid barrier: pad to BM32, scatter the
received `(row, slot)` pairs and weights into `sti` and `swt`, reset the allocator, advance
the monotonic epoch. The sort uses atomic cursors, so within-expert row order is
nondeterministic — which is why acceptance is tolerance-based rather than bitwise.

### 3–4. The GEMM

[`n2_phase1.cpp`](../kernels/gemm/n2_phase1.cpp) consumes the sorted activation rows and
runs the first expert projection with W13, producing the quantized intermediate activation
and its scales (`a2q`, `dq2`). [`n2_phase2.cpp`](../kernels/gemm/n2_phase2.cpp) runs the W2
projection, applies the route weights, and writes each rank's partial BF16 output into the
symmetric `part` buffer.

Splitting the GEMM this way is what makes W2 write-once: a split-K W2 writes every output
element eight times through BF16 atomics, and phase 2 replaces that with split-N full-K,
reducing all 16 K128 groups in registers. Together the two phases are the bulk of the
region's compute.

### 5. `k0d_combine` — owner-pull reduction

[`kernels/decode/k0d_combine.hip`](../kernels/decode/k0d_combine.hip). Launch: 16 CTAs x
256 threads.

Its head contains an in-kernel all-rank rendezvous proving that every GPU's phase-2 GEMM
has completed. Each token owner then reads the list of ranks and rows that produced partial
results, pulls those BF16 rows directly over P2P, accumulates them in FP32 in producer
order, and writes the final BF16 token output. Direct peer reads, no staging copy. As with
the ingress, there is no standalone post-GEMM barrier node: the rendezvous is folded into
the combine head, which is also where MoRI puts its own.

## Results

Five independent processes, `production,k0d_sq_b2f,k0d_mega`, graph mode, 1,109,880 rows
gated per run, 100% gate pass rate.

| metric | production | `k0d_mega` |
|---|---:|---:|
| warm aligned median | 562.51 µs | **550.34 µs** |
| warm paired p50 ratio | — | **0.9781** |
| cold aligned median | 553.64 µs | 553.75 µs |
| paired p95 ratio | — | 1.0313 |

In-graph device phase profile (`s_memrealtime` markers, marker overhead subtracted):

| phase | custom | production |
|---|---:|---:|
| quant | 11.81 µs | included in `fmoe` |
| ingress | 33.27 µs | 28.55 µs |
| GEMM | 377.88 µs | 422.85 µs |
| combine | 26.27 µs | 51.11 µs |
| region total | 511.33 µs | 520.27 µs |

## The honest verdict

**`k0d_mega` is the best custom decode path here, and it is at parity with production, not
a confirmed win.**

- The warm median is nominally about 2.2% faster.
- Cold performance is tied (1.0002x).
- Four of the five paired processes straddle 1.0.
- The p95 ratio is above 1.0.

Where the remaining deficit sits, and what is worth attacking:

- **The GEMM is the largest absolute cost** — 378 µs, 73.9% of the custom region — but it is
  also where the custom path is *ahead* of production at the region level (377.88 vs 422.85).
  At the kernel level, measured in isolation, AITER's `fmoe` is still faster than our
  single-kernel GEMM.
- **The ingress is 4.72 µs slower than production's, not faster.** (An earlier write-up
  states this with the sign reversed in prose while its own table is correct; the table is
  right and the custom ingress is the slower one.)
- **The combine is the clearest win**: 26.27 µs against production's 51.11 µs for combine
  plus barrier.
- **The ingress is also the variance source** — it correlates with the region's end-to-end
  variance at r² ≈ 0.885.

The dominant microarchitectural fact behind the ingress cost: on this stack a single
wavefront's remote xGMI memory operations — atomics and posted stores alike — are delivered
at roughly 0.9 µs per sequential instruction, while operations spread across parallel lanes
of one instruction overlap (8 in about 0.7 µs). That explains both the serial 8-poke
doorbell fan-out (6.27 µs serial versus 0.70 µs lane-parallel, an 8.9x mechanism speedup
that ports to three protocol sites) and the M1 push sweep, where each token-warp issues
roughly 34 sequential remote instructions and the fabric sits largely idle while warps
serialize on delivery.

## What is deliberately not here

An experimental successor, `k0d2_mega`, exists in the working tree it came from and is
**not** carried into this repository. It bundles several changes at once and has neither a
complete integration nor an A/B result. `k0d_mega` is the one that was measured.
