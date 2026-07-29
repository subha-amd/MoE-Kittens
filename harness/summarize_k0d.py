#!/usr/bin/env python3
"""Strictly validate and summarize a five-run k0d decode campaign.

Usage:
    python3 summarize_k0d.py RUN1 RUN2 RUN3 RUN4 RUN5 [--out summary.json]

Each argument is an explicit campaign run directory (a shell glob is fine if it
expands to exactly five arguments). The command reads but never changes run
artifacts, emits the summary JSON on stdout, and optionally creates ``--out``.
An existing output file is never overwritten.
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path


RUN_COUNT = 5
RANK_COUNT = 8
CONDITIONS = ("warm", "cold")
K0D_ARMS = ("k0d_s", "k0d_sq", "k0d_sq_b2f", "k0d_f")
K0D_MODULES = ("k0d_fused", "k0d_barrier", "k0d_gather_zero", "k0d_combine")
RANK_NAME_RE = re.compile(r"k0pf_prefill_rank([0-9]+)\.json\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MAX_ABS = 0.02
MAX_REL_L2 = 0.01


class CampaignError(ValueError):
    """A fail-closed campaign validation error."""


def _mapping(value, label):
    if type(value) is not dict:
        raise CampaignError(f"{label}: expected JSON object")
    return value


def _list(value, label):
    if type(value) is not list:
        raise CampaignError(f"{label}: expected JSON array")
    return value


def _field(mapping, key, label):
    obj = _mapping(mapping, label)
    if key not in obj:
        raise CampaignError(f"{label}: missing {key!r}")
    return obj[key]


def _nested(mapping, label, *keys):
    value = mapping
    current = label
    for key in keys:
        value = _field(value, key, current)
        current = f"{current}.{key}"
    return value


def _true(value, label):
    if value is not True:
        raise CampaignError(f"{label}: expected true, got {value!r}")


def _int(value, expected, label):
    if type(value) is not int or value != expected:
        raise CampaignError(f"{label}: expected integer {expected}, got {value!r}")


def _zero(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value != 0:
        raise CampaignError(f"{label}: expected numeric zero, got {value!r}")


def _number(value, label, *, positive=False):
    if type(value) not in (int, float):
        raise CampaignError(f"{label}: expected finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise CampaignError(f"{label}: expected {qualifier} number, got {value!r}")
    return result


def _same_keys(mapping, expected, label):
    actual = set(_mapping(mapping, label))
    wanted = set(expected)
    if actual != wanted:
        raise CampaignError(
            f"{label}: keys differ; missing={sorted(wanted - actual)} "
            f"extra={sorted(actual - wanted)}"
        )


def _canonical_json(value, label):
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"{label}: is not canonical finite JSON: {exc}") from exc


def _correctness_metrics(gate, label):
    gate = _mapping(gate, label)
    max_abs = _number(_field(gate, "max_abs", label), f"{label}.max_abs")
    rel_l2 = _number(_field(gate, "rel_L2", label), f"{label}.rel_L2")
    _int(_field(gate, "nonfinite", label), 0, f"{label}.nonfinite")
    if max_abs < 0 or max_abs > MAX_ABS:
        raise CampaignError(f"{label}.max_abs: {max_abs} exceeds {MAX_ABS}")
    if rel_l2 < 0 or rel_l2 > MAX_REL_L2:
        raise CampaignError(f"{label}.rel_L2: {rel_l2} exceeds {MAX_REL_L2}")


def _load_run(directory):
    if not directory.is_dir():
        raise CampaignError(f"{directory}: not a run directory")

    rank_files = sorted(directory.glob("k0pf_prefill_rank*.json"))
    if len(rank_files) != RANK_COUNT:
        raise CampaignError(
            f"{directory}: expected exactly {RANK_COUNT} rank JSONs, "
            f"found {len(rank_files)}"
        )

    ranks = {}
    paths = {}
    for path in rank_files:
        match = RANK_NAME_RE.fullmatch(path.name)
        if match is None:
            raise CampaignError(f"{path}: malformed rank JSON filename")
        rank = int(match.group(1))
        if rank in ranks:
            raise CampaignError(f"{directory}: duplicate rank {rank} JSON")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CampaignError(f"{path}: cannot read valid JSON: {exc}") from exc
        payload = _mapping(payload, str(path))
        _int(_field(payload, "rank", str(path)), rank, f"{path}.rank")
        ranks[rank] = payload
        paths[rank] = path

    expected_ranks = set(range(RANK_COUNT))
    if set(ranks) != expected_ranks:
        raise CampaignError(
            f"{directory}: expected ranks 0..7, got {sorted(ranks)}"
        )
    return {"directory": directory, "ranks": ranks, "paths": paths}


def _validate_hash_map(value, label):
    hashes = _mapping(value, label)
    _same_keys(hashes, K0D_MODULES, label)
    for module in K0D_MODULES:
        digest = hashes[module]
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            raise CampaignError(
                f"{label}.{module}: expected lowercase SHA-256 hex digest"
            )
    return hashes


def _validate_config(config, label):
    config = _mapping(config, label)
    _int(_field(config, "T", label), 64, f"{label}.T")
    _int(_field(config, "maxtok", label), 512, f"{label}.maxtok")
    _int(_field(config, "maxtok_prod", label), 512, f"{label}.maxtok_prod")
    _int(_field(config, "incl_plan", label), 1, f"{label}.incl_plan")
    _int(_field(config, "ntimed", label), 100, f"{label}.ntimed")
    _int(_field(config, "nuntimed", label), 20, f"{label}.nuntimed")

    arms = _list(_field(config, "arms", label), f"{label}.arms")
    if not arms or any(type(arm) is not str or not arm for arm in arms):
        raise CampaignError(f"{label}.arms: expected non-empty arm names")
    if len(set(arms)) != len(arms):
        raise CampaignError(f"{label}.arms: duplicate arm name")
    if arms.count("production") != 1:
        raise CampaignError(f"{label}.arms: require exactly one production arm")
    candidates = [arm for arm in arms if arm != "production"]
    if not candidates:
        raise CampaignError(f"{label}.arms: require at least one k0d candidate")
    unknown = [arm for arm in candidates if arm not in K0D_ARMS]
    if unknown:
        raise CampaignError(f"{label}.arms: non-k0d candidates {unknown}")
    return list(arms), candidates


def _validate_k0d_identity(payload, label, candidates):
    weight_storage = _mapping(
        _field(payload, "weight_storage", label), f"{label}.weight_storage"
    )
    for key in (
        "shared_between_production_and_candidate",
        "w13_w1_same_ptr",
        "w2_w2c_same_ptr",
    ):
        _true(
            _field(weight_storage, key, f"{label}.weight_storage"),
            f"{label}.weight_storage.{key}",
        )
    _int(
        _field(weight_storage, "duplicate_bytes_removed", f"{label}.weight_storage"),
        1409286144,
        f"{label}.weight_storage.duplicate_bytes_removed",
    )

    k0d = _mapping(_field(payload, "k0d", label), f"{label}.k0d")
    for key in ("requested", "b64_only", "shape_ok", "modules_loaded"):
        _true(_field(k0d, key, f"{label}.k0d"), f"{label}.k0d.{key}")

    shape = _mapping(
        _field(k0d, "shape_preflight", f"{label}.k0d"),
        f"{label}.k0d.shape_preflight",
    )
    expected_shape = {
        "world": 8,
        "T": 64,
        "T_LOC_MAX": 512,
        "PADMAX": 5088,
        "MROWS": 512,
        "MAXTOK": 512,
        "MAXTOK_PROD": 512,
        "NCHUNK": 16,
    }
    for key, expected in expected_shape.items():
        _int(
            _field(shape, key, f"{label}.k0d.shape_preflight"),
            expected,
            f"{label}.k0d.shape_preflight.{key}",
        )
    shape_expected = _mapping(
        _field(k0d, "shape_expected", f"{label}.k0d"),
        f"{label}.k0d.shape_expected",
    )
    for key, expected in expected_shape.items():
        _int(
            _field(shape_expected, key, f"{label}.k0d.shape_expected"),
            expected,
            f"{label}.k0d.shape_expected.{key}",
        )

    registered = _list(
        _field(k0d, "arms_registered", f"{label}.k0d"),
        f"{label}.k0d.arms_registered",
    )
    missing = set(candidates) - set(registered)
    if missing:
        raise CampaignError(
            f"{label}.k0d.arms_registered: missing candidates {sorted(missing)}"
        )

    route_contract = _mapping(
        _field(k0d, "route_input_contract", f"{label}.k0d"),
        f"{label}.k0d.route_input_contract",
    )
    for key in ("live_topk_is_symmetric", "ids_same_ptr", "weights_same_ptr"):
        _true(
            _field(route_contract, key, f"{label}.k0d.route_input_contract"),
            f"{label}.k0d.route_input_contract.{key}",
        )
    route_setup = _mapping(
        _field(k0d, "route_swap_setup", f"{label}.k0d"),
        f"{label}.k0d.route_swap_setup",
    )
    _true(
        _field(route_setup, "ready", f"{label}.k0d.route_swap_setup"),
        f"{label}.k0d.route_swap_setup.ready",
    )


def _validate_foundation_gates(payload, label):
    plan = _mapping(
        _field(payload, "k0pf_plan_equivalence", label),
        f"{label}.k0pf_plan_equivalence",
    )
    for key in ("pass_all_ranks", "local_pass"):
        _true(
            _field(plan, key, f"{label}.k0pf_plan_equivalence"),
            f"{label}.k0pf_plan_equivalence.{key}",
        )
    _zero(_field(plan, "k0pf_perr", f"{label}.k0pf_plan_equivalence"),
          f"{label}.k0pf_plan_equivalence.k0pf_perr")
    _zero(_field(plan, "frozen_perr", f"{label}.k0pf_plan_equivalence"),
          f"{label}.k0pf_plan_equivalence.frozen_perr")
    diffs = _mapping(
        _field(plan, "diffs", f"{label}.k0pf_plan_equivalence"),
        f"{label}.k0pf_plan_equivalence.diffs",
    )
    if not diffs:
        raise CampaignError(f"{label}.k0pf_plan_equivalence.diffs: empty")
    for key, value in diffs.items():
        _zero(value, f"{label}.k0pf_plan_equivalence.diffs.{key}")

    primitives = _mapping(
        _field(payload, "k0pf_primitives", label), f"{label}.k0pf_primitives"
    )
    for name, zero_fields in (
        ("quant", ("byte_diffs", "scale_diffs")),
        ("gather", ("byte_diffs", "scale_diffs")),
        ("combine", ("bf16_bit_diffs",)),
    ):
        primitive = _mapping(
            _field(primitives, name, f"{label}.k0pf_primitives"),
            f"{label}.k0pf_primitives.{name}",
        )
        _true(
            _field(primitive, "pass_all_ranks", f"{label}.k0pf_primitives.{name}"),
            f"{label}.k0pf_primitives.{name}.pass_all_ranks",
        )
        for key in zero_fields:
            _zero(
                _field(primitive, key, f"{label}.k0pf_primitives.{name}"),
                f"{label}.k0pf_primitives.{name}.{key}",
            )


def _validate_eager_and_graph(payload, label, arms, candidates):
    for key in (
        "graph_capture_ok",
        "eager_all_pass",
        "control_fails",
        "stress_all_pass",
        "xstress_pass",
        "k0d_epoch_ok",
        "k0d_route_swap_ok",
        "timing_gate_ok",
    ):
        _true(_field(payload, key, label), f"{label}.{key}")
    _zero(_field(payload, "pperr_after_stress", label),
          f"{label}.pperr_after_stress")
    if "k0d_sq_b2f" in candidates:
        _true(
            _field(payload, "k0d_sq_b2f_wait_control_ok", label),
            f"{label}.k0d_sq_b2f_wait_control_ok",
        )
        wait_control = _mapping(
            _field(payload, "k0d_sq_b2f_wait_control", label),
            f"{label}.k0d_sq_b2f_wait_control",
        )
        for key in ("pass_all_ranks", "local_pass"):
            _true(
                _field(wait_control, key, f"{label}.k0d_sq_b2f_wait_control"),
                f"{label}.k0d_sq_b2f_wait_control.{key}",
            )
        pperr = _field(
            wait_control, "pperr", f"{label}.k0d_sq_b2f_wait_control"
        )
        if type(pperr) is not int or (pperr & 8) == 0:
            raise CampaignError(
                f"{label}.k0d_sq_b2f_wait_control.pperr: expected timeout bit 8, "
                f"got {pperr!r}"
            )
        _int(
            _field(
                wait_control,
                "expected_timeout_bit",
                f"{label}.k0d_sq_b2f_wait_control",
            ),
            8,
            f"{label}.k0d_sq_b2f_wait_control.expected_timeout_bit",
        )

    eager = _mapping(_field(payload, "eager", label), f"{label}.eager")
    graph = _mapping(_field(payload, "graph", label), f"{label}.graph")
    _same_keys(eager, arms, f"{label}.eager")
    _same_keys(graph, arms, f"{label}.graph")
    for arm in arms:
        eager_arm = _mapping(_field(eager, arm, f"{label}.eager"),
                             f"{label}.eager.{arm}")
        for key in ("pass", "local_pass"):
            _true(_field(eager_arm, key, f"{label}.eager.{arm}"),
                  f"{label}.eager.{arm}.{key}")
        _zero(_field(eager_arm, "plan_err", f"{label}.eager.{arm}"),
              f"{label}.eager.{arm}.plan_err")
        _zero(_field(eager_arm, "pperr", f"{label}.eager.{arm}"),
              f"{label}.eager.{arm}.pperr")
        _correctness_metrics(eager_arm, f"{label}.eager.{arm}")

        graph_arm = _mapping(_field(graph, arm, f"{label}.graph"),
                             f"{label}.graph.{arm}")
        for key in ("pass", "local_pass", "stress_pass"):
            _true(_field(graph_arm, key, f"{label}.graph.{arm}"),
                  f"{label}.graph.{arm}.{key}")
        for key in ("plan_err", "pperr", "stress_plan_err", "stress_pperr"):
            _zero(_field(graph_arm, key, f"{label}.graph.{arm}"),
                  f"{label}.graph.{arm}.{key}")
        _correctness_metrics(graph_arm, f"{label}.graph.{arm}")

    capture = _mapping(
        _field(payload, "plan_capture", label), f"{label}.plan_capture"
    )
    _same_keys(capture, candidates, f"{label}.plan_capture")
    for arm in candidates:
        arm_gate = _mapping(
            _field(capture, arm, f"{label}.plan_capture"),
            f"{label}.plan_capture.{arm}",
        )
        for key in ("pass_all_ranks", "local_pass"):
            _true(_field(arm_gate, key, f"{label}.plan_capture.{arm}"),
                  f"{label}.plan_capture.{arm}.{key}")
        for key in ("plan_err", "pperr"):
            _zero(_field(arm_gate, key, f"{label}.plan_capture.{arm}"),
                  f"{label}.plan_capture.{arm}.{key}")
        rel_l2 = _number(
            _field(arm_gate, "rel_L2", f"{label}.plan_capture.{arm}"),
            f"{label}.plan_capture.{arm}.rel_L2",
        )
        if rel_l2 < 0 or rel_l2 > MAX_REL_L2:
            raise CampaignError(
                f"{label}.plan_capture.{arm}.rel_L2: {rel_l2} exceeds {MAX_REL_L2}"
            )


def _validate_k0d_primitives(payload, label, candidates):
    primitives = _mapping(
        _field(payload, "k0d_primitives", label), f"{label}.k0d_primitives"
    )
    _true(_field(primitives, "enabled", f"{label}.k0d_primitives"),
          f"{label}.k0d_primitives.enabled")
    expected_order = [arm for arm in K0D_ARMS if arm in candidates]
    primitive_arms = _list(
        _field(primitives, "arms", f"{label}.k0d_primitives"),
        f"{label}.k0d_primitives.arms",
    )
    if primitive_arms != expected_order:
        raise CampaignError(
            f"{label}.k0d_primitives.arms: expected {expected_order}, "
            f"got {primitive_arms!r}"
        )

    specs = {
        "fused_plan_quant": ("plan_err", "pperr"),
        "gather_zero": ("byte_diffs", "scale_diffs", "pperr"),
        "combine_real_n2_partial": ("bf16_bit_diffs", "pperr"),
    }
    for name, zero_fields in specs.items():
        gate = _mapping(
            _field(primitives, name, f"{label}.k0d_primitives"),
            f"{label}.k0d_primitives.{name}",
        )
        for key in ("pass_all_ranks", "local_pass"):
            _true(_field(gate, key, f"{label}.k0d_primitives.{name}"),
                  f"{label}.k0d_primitives.{name}.{key}")
        for key in zero_fields:
            _zero(_field(gate, key, f"{label}.k0d_primitives.{name}"),
                  f"{label}.k0d_primitives.{name}.{key}")
    _true(
        _nested(primitives, f"{label}.k0d_primitives", "gather_zero", "part_live_zero"),
        f"{label}.k0d_primitives.gather_zero.part_live_zero",
    )
    plan_diffs = _mapping(
        _nested(primitives, f"{label}.k0d_primitives", "fused_plan_quant", "diffs"),
        f"{label}.k0d_primitives.fused_plan_quant.diffs",
    )
    if not plan_diffs:
        raise CampaignError(
            f"{label}.k0d_primitives.fused_plan_quant.diffs: empty"
        )
    for key, value in plan_diffs.items():
        _zero(value, f"{label}.k0d_primitives.fused_plan_quant.diffs.{key}")


def _validate_epoch(payload, label, candidates):
    epoch = _mapping(_field(payload, "k0d_epoch", label), f"{label}.k0d_epoch")
    for key in (
        "pass_all_ranks",
        "local_pass",
        "noquant_downstream_control_16_fail",
    ):
        _true(_field(epoch, key, f"{label}.k0d_epoch"),
              f"{label}.k0d_epoch.{key}")
    epoch_arms = _mapping(
        _field(epoch, "arms", f"{label}.k0d_epoch"), f"{label}.k0d_epoch.arms"
    )
    _same_keys(epoch_arms, candidates, f"{label}.k0d_epoch.arms")
    for arm in candidates:
        arm_gate = _mapping(
            _field(epoch_arms, arm, f"{label}.k0d_epoch.arms"),
            f"{label}.k0d_epoch.arms.{arm}",
        )
        for key in ("candidate_16_pass", "state_delta_ok"):
            _true(_field(arm_gate, key, f"{label}.k0d_epoch.arms.{arm}"),
                  f"{label}.k0d_epoch.arms.{arm}.{key}")
        replays = _list(
            _field(arm_gate, "replays", f"{label}.k0d_epoch.arms.{arm}"),
            f"{label}.k0d_epoch.arms.{arm}.replays",
        )
        if len(replays) != 16:
            raise CampaignError(
                f"{label}.k0d_epoch.arms.{arm}.replays: expected 16, "
                f"got {len(replays)}"
            )
        for expected_rep, replay in enumerate(replays):
            replay_label = f"{label}.k0d_epoch.arms.{arm}.replays[{expected_rep}]"
            replay = _mapping(replay, replay_label)
            _int(_field(replay, "rep", replay_label), expected_rep,
                 f"{replay_label}.rep")
            for key in ("control_failed", "candidate_pass"):
                _true(_field(replay, key, replay_label), f"{replay_label}.{key}")
            for key in ("plan_err", "pperr"):
                _zero(_field(replay, key, replay_label), f"{replay_label}.{key}")
            rel_l2 = _number(
                _field(replay, "candidate_rel_L2", replay_label),
                f"{replay_label}.candidate_rel_L2",
            )
            max_abs = _number(
                _field(replay, "candidate_max_abs", replay_label),
                f"{replay_label}.candidate_max_abs",
            )
            if rel_l2 < 0 or rel_l2 > MAX_REL_L2:
                raise CampaignError(
                    f"{replay_label}.candidate_rel_L2: {rel_l2} exceeds {MAX_REL_L2}"
                )
            if max_abs < 0 or max_abs > MAX_ABS:
                raise CampaignError(
                    f"{replay_label}.candidate_max_abs: {max_abs} exceeds {MAX_ABS}"
                )

        if arm == "k0d_f":
            delta = _mapping(
                _field(arm_gate, "state_delta", f"{label}.k0d_epoch.arms.{arm}"),
                f"{label}.k0d_epoch.arms.{arm}.state_delta",
            )
            _int(_field(delta, "quant_count", f"{label}.k0d_epoch.arms.{arm}.state_delta"),
                 16, f"{label}.k0d_epoch.arms.{arm}.state_delta.quant_count")
            for key in ("arr_count", "don_count"):
                _int(
                    _field(delta, key, f"{label}.k0d_epoch.arms.{arm}.state_delta"),
                    256,
                    f"{label}.k0d_epoch.arms.{arm}.state_delta.{key}",
                )
            for key in ("quant_flags", "comb_flags", "arr_flags"):
                values = _list(
                    _field(delta, key, f"{label}.k0d_epoch.arms.{arm}.state_delta"),
                    f"{label}.k0d_epoch.arms.{arm}.state_delta.{key}",
                )
                if values != [16] * RANK_COUNT:
                    raise CampaignError(
                        f"{label}.k0d_epoch.arms.{arm}.state_delta.{key}: "
                        f"expected {[16] * RANK_COUNT}, got {values!r}"
                    )
        elif arm == "k0d_sq_b2f":
            delta = _mapping(
                _field(arm_gate, "state_delta", f"{label}.k0d_epoch.arms.{arm}"),
                f"{label}.k0d_epoch.arms.{arm}.state_delta",
            )
            _int(
                _field(
                    delta, "b1_count", f"{label}.k0d_epoch.arms.{arm}.state_delta"
                ),
                16,
                f"{label}.k0d_epoch.arms.{arm}.state_delta.b1_count",
            )
            _int(
                _field(
                    delta, "arr_count", f"{label}.k0d_epoch.arms.{arm}.state_delta"
                ),
                256,
                f"{label}.k0d_epoch.arms.{arm}.state_delta.arr_count",
            )
            for key in ("b1_flags", "arr_flags"):
                values = _list(
                    _field(
                        delta, key, f"{label}.k0d_epoch.arms.{arm}.state_delta"
                    ),
                    f"{label}.k0d_epoch.arms.{arm}.state_delta.{key}",
                )
                if values != [16] * RANK_COUNT:
                    raise CampaignError(
                        f"{label}.k0d_epoch.arms.{arm}.state_delta.{key}: "
                        f"expected {[16] * RANK_COUNT}, got {values!r}"
                    )


def _validate_route_swap(payload, label, candidates):
    route = _mapping(
        _field(payload, "k0d_route_swap", label), f"{label}.k0d_route_swap"
    )
    for key in (
        "pass_all_ranks",
        "local_pass",
        "route_inputs_distinct",
        "metadata_ok",
    ):
        _true(_field(route, key, f"{label}.k0d_route_swap"),
              f"{label}.k0d_route_swap.{key}")
    _correctness_metrics(
        _field(route, "production_alt_gate", f"{label}.k0d_route_swap"),
        f"{label}.k0d_route_swap.production_alt_gate",
    )
    _correctness_metrics(
        _field(route, "production_restore_gate", f"{label}.k0d_route_swap"),
        f"{label}.k0d_route_swap.production_restore_gate",
    )
    route_arms = _mapping(
        _field(route, "arms", f"{label}.k0d_route_swap"),
        f"{label}.k0d_route_swap.arms",
    )
    _same_keys(route_arms, candidates, f"{label}.k0d_route_swap.arms")
    for arm in candidates:
        arm_gate = _mapping(
            _field(route_arms, arm, f"{label}.k0d_route_swap.arms"),
            f"{label}.k0d_route_swap.arms.{arm}",
        )
        for key in ("pass_local", "metadata_changed", "restore_pass"):
            _true(_field(arm_gate, key, f"{label}.k0d_route_swap.arms.{arm}"),
                  f"{label}.k0d_route_swap.arms.{arm}.{key}")
        for key in ("plan_err", "pperr", "restore_plan_err", "restore_pperr"):
            _zero(_field(arm_gate, key, f"{label}.k0d_route_swap.arms.{arm}"),
                  f"{label}.k0d_route_swap.arms.{arm}.{key}")
        for gate_name in ("alt_corpus_gate", "vs_production", "restore_gate"):
            _correctness_metrics(
                _field(arm_gate, gate_name, f"{label}.k0d_route_swap.arms.{arm}"),
                f"{label}.k0d_route_swap.arms.{arm}.{gate_name}",
            )


def _validate_timing(payload, label, arms, candidates, ntimed):
    signature = {}
    for condition in CONDITIONS:
        gate_name = f"timed_replay_gate_{condition}"
        gate = _mapping(_field(payload, gate_name, label), f"{label}.{gate_name}")
        for key in ("pass_all_ranks", "local_pass"):
            _true(_field(gate, key, f"{label}.{gate_name}"),
                  f"{label}.{gate_name}.{key}")
        gate_arms = _mapping(
            _field(gate, "arms", f"{label}.{gate_name}"),
            f"{label}.{gate_name}.arms",
        )
        _same_keys(gate_arms, arms, f"{label}.{gate_name}.arms")
        for arm in arms:
            arm_gate = _mapping(
                _field(gate_arms, arm, f"{label}.{gate_name}.arms"),
                f"{label}.{gate_name}.arms.{arm}",
            )
            _int(_field(arm_gate, "replays", f"{label}.{gate_name}.arms.{arm}"),
                 ntimed, f"{label}.{gate_name}.arms.{arm}.replays")
            for key in ("failures", "max_nonfinite", "max_pperr", "max_plan_err"):
                _zero(_field(arm_gate, key, f"{label}.{gate_name}.arms.{arm}"),
                      f"{label}.{gate_name}.arms.{arm}.{key}")
            rel_l2 = _number(
                _field(arm_gate, "worst_rel_L2", f"{label}.{gate_name}.arms.{arm}"),
                f"{label}.{gate_name}.arms.{arm}.worst_rel_L2",
            )
            max_abs = _number(
                _field(arm_gate, "worst_max_abs", f"{label}.{gate_name}.arms.{arm}"),
                f"{label}.{gate_name}.arms.{arm}.worst_max_abs",
            )
            if rel_l2 < 0 or rel_l2 > MAX_REL_L2:
                raise CampaignError(
                    f"{label}.{gate_name}.arms.{arm}.worst_rel_L2: "
                    f"{rel_l2} exceeds {MAX_REL_L2}"
                )
            if max_abs < 0 or max_abs > MAX_ABS:
                raise CampaignError(
                    f"{label}.{gate_name}.arms.{arm}.worst_max_abs: "
                    f"{max_abs} exceeds {MAX_ABS}"
                )

        timing_name = f"timing_us_{condition}"
        ratio_name = f"ratios_{condition}"
        timing = _mapping(_field(payload, timing_name, label), f"{label}.{timing_name}")
        ratios = _mapping(_field(payload, ratio_name, label), f"{label}.{ratio_name}")
        _same_keys(timing, arms, f"{label}.{timing_name}")
        for arm in arms:
            values = _mapping(
                _field(timing, arm, f"{label}.{timing_name}"),
                f"{label}.{timing_name}.{arm}",
            )
            p50 = _number(_field(values, "p50", f"{label}.{timing_name}.{arm}"),
                          f"{label}.{timing_name}.{arm}.p50", positive=True)
            p95 = _number(_field(values, "p95", f"{label}.{timing_name}.{arm}"),
                          f"{label}.{timing_name}.{arm}.p95", positive=True)
            if p95 < p50:
                raise CampaignError(
                    f"{label}.{timing_name}.{arm}: p95 {p95} is below p50 {p50}"
                )
        for arm in candidates:
            key = f"{arm}/production"
            ratio = _mapping(_field(ratios, key, f"{label}.{ratio_name}"),
                             f"{label}.{ratio_name}.{key}")
            p50 = _number(_field(ratio, "p50", f"{label}.{ratio_name}.{key}"),
                          f"{label}.{ratio_name}.{key}.p50", positive=True)
            p95 = _number(_field(ratio, "p95", f"{label}.{ratio_name}.{key}"),
                          f"{label}.{ratio_name}.{key}.p95", positive=True)
            if p95 < p50:
                raise CampaignError(
                    f"{label}.{ratio_name}.{key}: p95 {p95} is below p50 {p50}"
                )
        if "k0d_sq" in candidates and "k0d_sq_b2f" in candidates:
            key = "k0d_sq_b2f/k0d_sq"
            ratio = _mapping(
                _field(ratios, key, f"{label}.{ratio_name}"),
                f"{label}.{ratio_name}.{key}",
            )
            p50 = _number(
                _field(ratio, "p50", f"{label}.{ratio_name}.{key}"),
                f"{label}.{ratio_name}.{key}.p50",
                positive=True,
            )
            p95 = _number(
                _field(ratio, "p95", f"{label}.{ratio_name}.{key}"),
                f"{label}.{ratio_name}.{key}.p95",
                positive=True,
            )
            if p95 < p50:
                raise CampaignError(
                    f"{label}.{ratio_name}.{key}: p95 {p95} is below p50 {p50}"
                )
        signature[condition] = {
            "timing": _canonical_json(timing, f"{label}.{timing_name}"),
            "ratios": _canonical_json(
                {f"{arm}/production": ratios[f"{arm}/production"]
                 for arm in candidates},
                f"{label}.{ratio_name}",
            ),
        }
    return signature


def _validate_artifact(payload, label, arms, candidates, ntimed):
    _validate_k0d_identity(payload, label, candidates)
    _validate_foundation_gates(payload, label)
    _validate_eager_and_graph(payload, label, arms, candidates)
    _validate_k0d_primitives(payload, label, candidates)
    _validate_epoch(payload, label, candidates)
    _validate_route_swap(payload, label, candidates)
    return _validate_timing(payload, label, arms, candidates, ntimed)


def _stats(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def build_summary(directory_args):
    if len(directory_args) != RUN_COUNT:
        raise CampaignError(
            f"expected exactly {RUN_COUNT} run directories, got {len(directory_args)}"
        )
    directories = []
    for raw in directory_args:
        try:
            directory = Path(raw).expanduser().resolve(strict=True)
        except OSError as exc:
            raise CampaignError(f"{raw}: cannot resolve run directory: {exc}") from exc
        directories.append(directory)
    if len(set(directories)) != RUN_COUNT:
        raise CampaignError("run directories must be five distinct paths")

    runs = [_load_run(directory) for directory in directories]
    first_label = str(runs[0]["paths"][0])
    first_payload = runs[0]["ranks"][0]
    config = _mapping(_field(first_payload, "config", first_label),
                      f"{first_label}.config")
    arms, candidates = _validate_config(config, f"{first_label}.config")
    config_signature = _canonical_json(config, f"{first_label}.config")
    source_hashes = _validate_hash_map(
        _nested(first_payload, first_label, "k0d", "source_sha256"),
        f"{first_label}.k0d.source_sha256",
    )
    hsaco_hashes = _validate_hash_map(
        _nested(first_payload, first_label, "k0d", "hsaco_sha256"),
        f"{first_label}.k0d.hsaco_sha256",
    )
    source_signature = _canonical_json(
        source_hashes, f"{first_label}.k0d.source_sha256"
    )
    hsaco_signature = _canonical_json(
        hsaco_hashes, f"{first_label}.k0d.hsaco_sha256"
    )
    ntimed = config["ntimed"]

    for run in runs:
        rank0_signature = None
        for rank in range(RANK_COUNT):
            path = run["paths"][rank]
            label = str(path)
            payload = run["ranks"][rank]
            artifact_config = _mapping(
                _field(payload, "config", label), f"{label}.config"
            )
            if _canonical_json(artifact_config, f"{label}.config") != config_signature:
                raise CampaignError(f"{label}.config: differs from {first_label}.config")
            source = _validate_hash_map(
                _nested(payload, label, "k0d", "source_sha256"),
                f"{label}.k0d.source_sha256",
            )
            if _canonical_json(source, f"{label}.k0d.source_sha256") != source_signature:
                raise CampaignError(
                    f"{label}.k0d.source_sha256: differs from {first_label}"
                )
            hsaco = _validate_hash_map(
                _nested(payload, label, "k0d", "hsaco_sha256"),
                f"{label}.k0d.hsaco_sha256",
            )
            if _canonical_json(hsaco, f"{label}.k0d.hsaco_sha256") != hsaco_signature:
                raise CampaignError(
                    f"{label}.k0d.hsaco_sha256: differs from {first_label}"
                )
            signature = _validate_artifact(
                payload, label, arms, candidates, ntimed
            )
            if rank == 0:
                rank0_signature = signature
            elif signature != rank0_signature:
                raise CampaignError(
                    f"{label}: aligned-MAX timing/ratio payload differs from rank 0 "
                    f"in {run['directory']}"
                )

    output = {
        "schema": "k0d_campaign_summary_v1",
        "run_dirs": [str(run["directory"]) for run in runs],
        "run_count": RUN_COUNT,
        "rank_count": RANK_COUNT,
        "timing_source_rank": 0,
        "rank_reduction": "aligned MAX(rank)",
        "ep_size": 8,
        "arms": arms,
        "candidates": candidates,
        "config": config,
        "hashes": {
            "source_sha256": source_hashes,
            "hsaco_sha256": hsaco_hashes,
        },
        "gates": {
            "all_required_gates_pass": True,
            "validated_rank_jsons": RUN_COUNT * RANK_COUNT,
        },
        "conditions": {},
        "comparisons": {},
    }
    for condition in CONDITIONS:
        condition_output = {}
        for arm in arms:
            p50s = [
                run["ranks"][0][f"timing_us_{condition}"][arm]["p50"]
                for run in runs
            ]
            p95s = [
                run["ranks"][0][f"timing_us_{condition}"][arm]["p95"]
                for run in runs
            ]
            arm_output = {"p50_us": _stats(p50s), "p95_us": _stats(p95s)}
            if arm != "production":
                ratio_key = f"{arm}/production"
                arm_output["ratio_p50"] = _stats(
                    run["ranks"][0][f"ratios_{condition}"][ratio_key]["p50"]
                    for run in runs
                )
                arm_output["ratio_p95"] = _stats(
                    run["ranks"][0][f"ratios_{condition}"][ratio_key]["p95"]
                    for run in runs
                )
            condition_output[arm] = arm_output
        output["conditions"][condition] = condition_output
    if "k0d_sq" in candidates and "k0d_sq_b2f" in candidates:
        comparison = {}
        for condition in CONDITIONS:
            ratio_key = "k0d_sq_b2f/k0d_sq"
            delta_key = f"delta_k0d_sq_b2f_vs_sq_us_{condition}"
            comparison[condition] = {
                "ratio_p50": _stats(
                    run["ranks"][0][f"ratios_{condition}"][ratio_key]["p50"]
                    for run in runs
                ),
                "ratio_p95": _stats(
                    run["ranks"][0][f"ratios_{condition}"][ratio_key]["p95"]
                    for run in runs
                ),
                "sq_minus_b2f_p50_us": _stats(
                    run["ranks"][0][delta_key]["p50"] for run in runs
                ),
            }
        output["comparisons"]["k0d_sq_b2f/k0d_sq"] = comparison
    return output


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dirs",
        metavar="RUN_DIR",
        nargs=RUN_COUNT,
        help="explicit run directories (exactly five)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="also create this JSON file; an existing path is never overwritten",
    )
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = build_summary(args.run_dirs)
        rendered = json.dumps(
            summary, allow_nan=False, indent=2, sort_keys=True
        ) + "\n"
        if args.out is not None:
            output_path = args.out.expanduser()
            try:
                with output_path.open("x", encoding="utf-8") as stream:
                    stream.write(rendered)
            except OSError as exc:
                raise CampaignError(f"{output_path}: cannot create --out: {exc}") from exc
    except CampaignError as exc:
        parser.error(str(exc))
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
