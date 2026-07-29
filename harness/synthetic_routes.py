#!/usr/bin/env python3
"""Deterministic synthetic top-k routes for the B64 TP1/DP8/EP8 harness.

These fixtures replace router outputs only.  Hidden states, expert weights, and
all production/custom kernels remain unchanged in the benchmark harness.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Tuple

import numpy as np


FAMILIES = (
    "balanced4",
    "skewed_hot",
    "many_empty",
    "high_fanout8",
    "low_fanout1",
)


def _pairs_for_token(family: str, g: int, world: int, experts_per_rank: int):
    q = g // world
    pairs = []
    if family == "balanced4":
        destinations = [(g + j) % world for j in range(4)]
        for j, destination in enumerate(destinations[:3] if g < 19 else destinations):
            local = 2 * ((q + 4 * j) % (experts_per_rank // 2))
            pairs.extend(((destination, local), (destination, local + 1)))
        if g < 19:
            for j in range(2):
                destination = destinations[j]
                local = 2 * ((q + 4 * j) % (experts_per_rank // 2))
                pairs.append((destination, (local + 2) % experts_per_rank))
    elif family == "skewed_hot":
        destinations = range(4) if g < 384 else range(4, 8)
        for j, destination in enumerate(destinations):
            pairs.extend(((destination, 0), (destination, 1 + ((g + 7 * j) % 31))))
    elif family == "many_empty":
        for j in range(4):
            destination = (g + j) % world
            local = 2 * ((q + j) % 2)
            pairs.extend(((destination, local), (destination, local + 1)))
    elif family == "high_fanout8":
        for destination in range(world):
            pairs.append((destination, (g + 5 * destination) % experts_per_rank))
    elif family == "low_fanout1":
        destination = g % world
        base = 8 * (q % 4)
        pairs.extend((destination, base + j) for j in range(8))
    else:
        raise ValueError(f"unknown synthetic route {family!r}; choose one of {FAMILIES}")
    return pairs


def route_stats(
    ids: np.ndarray,
    weights: np.ndarray,
    *,
    world: int = 8,
    experts_per_rank: int = 32,
    block_m: int = 32,
) -> Dict[str, object]:
    ids = np.asarray(ids)
    weights = np.asarray(weights)
    if ids.ndim != 3:
        raise ValueError(f"ids must be [world,tokens,topk], got {ids.shape}")
    if weights.shape != ids.shape:
        raise ValueError(f"weights shape {weights.shape} != ids shape {ids.shape}")
    if ids.dtype != np.int32:
        raise TypeError(f"ids must be int32, got {ids.dtype}")
    if weights.dtype != np.float32:
        raise TypeError(f"weights must be float32, got {weights.dtype}")
    if not ids.flags.c_contiguous or not weights.flags.c_contiguous:
        raise ValueError("ids and weights must be C-contiguous")

    global_experts = world * experts_per_rank
    flat = ids.reshape(-1, ids.shape[-1])
    flat_w = weights.reshape(-1, weights.shape[-1])
    if int(flat.min()) < 0 or int(flat.max()) >= global_experts:
        raise ValueError(f"expert ids must be in [0,{global_experts})")
    if any(np.unique(row).size != row.size for row in flat):
        raise ValueError("each token must route to unique experts")
    if not np.isfinite(flat_w).all() or not (flat_w > 0).all():
        raise ValueError("weights must be finite and strictly positive")

    destinations = flat // experts_per_rank
    fanouts = np.asarray([np.unique(row).size for row in destinations], dtype=np.int32)
    fanout_values, fanout_counts = np.unique(fanouts, return_counts=True)
    expert_counts = np.bincount(flat.reshape(-1), minlength=global_experts).reshape(
        world, experts_per_rank
    )
    assignments = expert_counts.sum(axis=1)
    arrival_rows = np.zeros((world,), dtype=np.int64)
    for token_destinations in destinations:
        arrival_rows[np.unique(token_destinations)] += 1
    padded = np.asarray(
        [
            sum(
                int(math.ceil(int(count) / block_m) * block_m)
                for count in expert_counts[rank]
                if count
            )
            for rank in range(world)
        ],
        dtype=np.int64,
    )
    ids_digest = hashlib.sha256(ids.tobytes(order="C")).hexdigest()
    fixture_digest = hashlib.sha256()
    fixture_digest.update(ids.tobytes(order="C"))
    fixture_digest.update(weights.tobytes(order="C"))
    return {
        "shape": list(ids.shape),
        "num_tokens_global": int(flat.shape[0]),
        "topk": int(flat.shape[1]),
        "expert_id_min": int(flat.min()),
        "expert_id_max": int(flat.max()),
        "unique_ids_per_token": True,
        "weight_sum_min": float(flat_w.sum(axis=1).min()),
        "weight_sum_max": float(flat_w.sum(axis=1).max()),
        "fanout_histogram": {
            str(int(value)): int(count) for value, count in zip(fanout_values, fanout_counts)
        },
        "fanout_mean": float(fanouts.mean()),
        "fanout_max": int(fanouts.max()),
        "arrival_rows_per_rank": arrival_rows.astype(int).tolist(),
        "expert_assignments_per_rank": assignments.astype(int).tolist(),
        "active_experts_per_rank": (expert_counts > 0).sum(axis=1).astype(int).tolist(),
        "empty_experts_per_rank": (expert_counts == 0).sum(axis=1).astype(int).tolist(),
        "expert_count_min_active": int(expert_counts[expert_counts > 0].min()),
        "expert_count_max": int(expert_counts.max()),
        "padded_rows_per_rank": padded.astype(int).tolist(),
        "padded_blocks_per_rank": (padded // block_m).astype(int).tolist(),
        "padding_rows_per_rank": (padded - assignments).astype(int).tolist(),
        "padding_ratio_global": float((padded.sum() - assignments.sum()) / padded.sum()),
        "total_assignments": int(assignments.sum()),
        "route_ids_sha256": ids_digest,
        "generated_fixture_sha256": fixture_digest.hexdigest(),
        # Kept for compatibility with the first synthetic campaign's JSON reader.
        "sha256": fixture_digest.hexdigest(),
    }


def generate_synthetic_route(
    family: str,
    *,
    seed: int = 0,
    world: int = 8,
    tokens_per_rank: int = 64,
    topk: int = 8,
    experts_per_rank: int = 32,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Return global ``[world,tokens_per_rank,topk]`` IDs, weights, and stats."""
    if world != 8 or topk != 8 or experts_per_rank != 32:
        raise ValueError("synthetic fixtures are intentionally pinned to WORLD=8, TOPK=8, E=32")
    if family not in FAMILIES:
        raise ValueError(f"unknown synthetic route {family!r}; choose one of {FAMILIES}")

    num_tokens = world * tokens_per_rank
    ids = np.empty((num_tokens, topk), dtype=np.int32)
    weights = np.empty((num_tokens, topk), dtype=np.float32)
    base_weights = np.asarray([16, 14, 12, 10, 8, 6, 4, 2], dtype=np.float32)
    base_weights /= base_weights.sum(dtype=np.float32)

    for g in range(num_tokens):
        pairs = _pairs_for_token(family, g, world, experts_per_rank)
        if len(pairs) != topk or len(set(pairs)) != topk:
            raise AssertionError(f"{family} token {g} produced invalid pairs: {pairs}")
        # A per-token permutation avoids making destination groups artificially contiguous in
        # top-k slot order. Seed affects layout only, not the fixture's load/fanout statistics.
        token_seed = (int(seed) * 0x9E3779B1 + g * 0x85EBCA77) & 0xFFFFFFFF
        order = np.random.default_rng(token_seed).permutation(topk)
        expert_ids = np.asarray(
            [destination * experts_per_rank + local for destination, local in pairs],
            dtype=np.int32,
        )
        ids[g] = expert_ids[order]
        weights[g] = base_weights

    ids = np.ascontiguousarray(ids.reshape(world, tokens_per_rank, topk))
    weights = np.ascontiguousarray(weights.reshape(world, tokens_per_rank, topk))
    stats = route_stats(ids, weights, world=world, experts_per_rank=experts_per_rank)
    stats.update(
        {
            "family": family,
            "seed": int(seed),
            "production_like": family == "balanced4",
            "fixture_class": (
                "production-shaped"
                if family == "balanced4"
                else "group-router stress"
                if family in {"skewed_hot", "many_empty"}
                else "non-production topology stress"
            ),
            "stress_only": family in {"high_fanout8", "low_fanout1"},
        }
    )
    return ids, weights, stats
