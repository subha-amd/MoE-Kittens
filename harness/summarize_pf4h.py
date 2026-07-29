#!/usr/bin/env python3
"""Per-rank in-graph phase attribution for the pf4h destination-sort arm.

The campaign summary provides aligned-MAX p50/p95 and arm-versus-production ratios. This
script reports per-rank boundary phases recorded when K0_DECODE_PHASE_PROFILE=1, including
the destination sort's count phase.

Usage:
    summarize_pf4h.py <run-directory-glob> [...]
"""

import argparse
import glob
import json
from pathlib import Path


def load_runs(patterns):
    runs = []
    for directory in sorted({Path(p) for pat in patterns for p in glob.glob(pat)}):
        rank_files = sorted(directory.glob("k0pf_prefill_rank*.json"))
        if len(rank_files) != 8:
            continue
        ranks = [json.loads(path.read_text()) for path in rank_files]
        ranks.sort(key=lambda r: r["rank"])
        runs.append((directory, ranks))
    if not runs:
        raise SystemExit("no complete 8-rank runs matched")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+")
    args = ap.parse_args()
    runs = load_runs(args.patterns)

    out = {"runs": [], "gates": {}, "region": {}, "phases": {}}
    gate_keys = (
        "timing_gate_ok", "eager_all_pass", "stress_all_pass", "xstress_pass",
        "k0pf3_epoch_ok",
    )
    gates = {k: True for k in gate_keys}
    gates["pf3_primitives"] = True
    gates["pperr_after_stress_zero"] = True

    for directory, ranks in runs:
        out["runs"].append(str(directory))
        for k in gate_keys:
            gates[k] = bool(gates[k] and all(bool(r.get(k)) for r in ranks))
        gates["pf3_primitives"] = bool(
            gates["pf3_primitives"]
            and all(r.get("pf3_primitives", {}).get("pass_all_ranks", True) for r in ranks)
        )
        gates["pperr_after_stress_zero"] = bool(
            gates["pperr_after_stress_zero"]
            and all(int(r.get("pperr_after_stress", 0)) == 0 for r in ranks)
        )
    out["gates"] = gates

    arms = runs[0][1][0]["config"]["arms"]
    for condition in ("warm", "cold"):
        cond = {}
        for arm in arms:
            cond[arm] = {
                "p50_us": [rs[0][f"timing_us_{condition}"][arm]["p50"] for _, rs in runs],
                "p95_us": [rs[0][f"timing_us_{condition}"][arm]["p95"] for _, rs in runs],
            }
        ratios = {}
        for key in runs[0][1][0].get(f"ratios_{condition}", {}):
            ratios[key] = {
                "p50": [rs[0][f"ratios_{condition}"][key]["p50"] for _, rs in runs],
                "p95": [rs[0][f"ratios_{condition}"][key]["p95"] for _, rs in runs],
            }
        cond["ratios"] = ratios
        out["region"][condition] = cond

    # Per-rank boundary phases. Present only when K0_DECODE_PHASE_PROFILE=1 was set.
    for directory, ranks in runs:
        prof = ranks[0].get("decode_device_phase_profile")
        if not prof:
            continue
        run_phases = {}
        for arm in prof["arms"]:
            run_phases[arm] = {
                "stage_names": prof["arms"][arm]["stage_names"],
                "per_rank_p50_us": [
                    r["decode_device_phase_profile"]["arms"][arm]["corrected_local_p50_us"]
                    for r in ranks
                ],
                "per_rank_p95_us": [
                    r["decode_device_phase_profile"]["arms"][arm]["corrected_local_p95_us"]
                    for r in ranks
                ],
            }
        run_phases["_t_loc_per_rank"] = [int(r["T_loc"]) for r in ranks if "T_loc" in r]
        out["phases"][str(directory)] = run_phases

    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
