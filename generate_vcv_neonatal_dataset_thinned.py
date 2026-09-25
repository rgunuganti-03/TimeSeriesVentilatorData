"""
generate_vcv_neonatal_dataset_thinned.py
------------------------------------------
Thinned dataset generation for VCV, neonatal population (Normal Neonate,
RDS). Parallel to generate_vcv_dataset_thinned.py rather than an
extension of it, per the project's own architectural decision
(ARCHITECTURE.md, 1a: "Neonatal dataset generation will use parallel,
population-specific scripts rather than sharing the adult thinned
grid").

LOOSE-COUPLING DESIGN NOTE
This script imports CONDITION_TIERS, NEONATE_CONDITION_WEIGHT_KG,
RECRUITMENT_SLOPES, and _make_scenario_id directly from
generator/vcv_generator.py. Nothing about which conditions are neonatal,
what they weigh, their compliance/resistance sweep ranges, their
n_cycles, their recruitment slopes, their scenario-ID formatting, or the
underlying physics is duplicated here -- so a change to any of those in
vcv_generator.py or conditions.py is picked up automatically on the next
run, with no edit to this file. The only local design decision is the
THINNED ventilator-side sweep grid itself, and even that is shared with
the other four neonatal scripts via generator/neonatal_thinning.py
rather than copied five times.

Run: python generate_vcv_neonatal_dataset_thinned.py
"""

import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator.vcv_generator import (
    generate_breath_cycles,
    CONDITION_TIERS,
    NEONATE_CONDITION_WEIGHT_KG,
    RECRUITMENT_SLOPES,
    _make_scenario_id,
)
from generator.neonatal_thinning import (
    neonatal_tiers,
    mechanics_grid,
    TIDAL_VOLUME_ML_PER_KG,
    RESPIRATORY_RATE,
    PEEP_CMH2O,
    IE_RATIO,
    FLOW_PATTERN,
)

OUTPUT_DIR = Path("data/exports/vcv_neonatal")

# Ventilator-side thinned sweep grid. Values and rationale live in
# generator/neonatal_thinning.py, shared across all five neonatal
# scripts; only the *assembly* into this mode's specific grid dict is
# local.
THINNED_NEONATAL_GRID = {
    "tidal_volume_ml_per_kg": TIDAL_VOLUME_ML_PER_KG,
    "respiratory_rate":       RESPIRATORY_RATE,
    "peep_cmH2O":             PEEP_CMH2O,
    "ie_ratio":                IE_RATIO,
    "flow_pattern":            FLOW_PATTERN,
}


# ---------------------------------------------------------------------------
# Per-mechanics-point sweep
# ---------------------------------------------------------------------------

def _generate_thinned_dataset(condition_name: str,
                               compliance_ml_per_cmH2O: float,
                               resistance_cmH2O_L_s: float,
                               n_cycles: int) -> list:
    """
    Sweep the neonatal-thinned VCV grid for one condition + mechanics
    pair. weight_kg is resolved per-condition from the IMPORTED
    NEONATE_CONDITION_WEIGHT_KG -- never hardcoded here -- so a revised
    neonatal weight in vcv_generator.py is picked up automatically.

    Returns a list of scenario dicts -- same structure as the adult
    generate_vcv_dataset_thinned.py's own _generate_thinned_dataset().
    """
    scenarios = []
    weight_kg = NEONATE_CONDITION_WEIGHT_KG[condition_name]
    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.0)

    keys   = ["tidal_volume_ml_per_kg", "respiratory_rate",
              "peep_cmH2O", "ie_ratio", "flow_pattern"]
    values = [THINNED_NEONATAL_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        vt_per_kg, rr, peep, ie, pattern = combo

        vt_mL = vt_per_kg * weight_kg

        params = {
            "respiratory_rate":        rr,
            "tidal_volume_ml":         vt_mL,
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "flow_pattern":            pattern,
            "condition":               condition_name,
            "population":              "neonate",
            "weight_kg":               weight_kg,
            "recruitment_slope":       rec_slope,
        }

        scenario_id = _make_scenario_id(condition_name, params)

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles)
        except Exception as e:
            scenarios.append({
                "scenario_id":    scenario_id,
                "condition":      condition_name,
                "params":         params,
                "metrics":        {},
                "is_valid":       False,
                "invalid_reason": f"Generator error: {e}",
                "generated_at":   datetime.now(timezone.utc).isoformat(),
            })
            continue

        metrics = {
            "ppeak_cmH2O":     result["ppeak_cmH2O"],
            "pplat_cmH2O":     result["pplat_cmH2O"],
            "driving_p_cmH2O": result["driving_p_cmH2O"],
            "mean_paw_cmH2O":  result["mean_paw_cmH2O"],
            "auto_peep_cmH2O": result["auto_peep_cmH2O"],
            "delivered_vt_ml": result["delivered_vt_ml"],
            "minute_vent_l":   result["minute_vent_l"],
        }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "is_valid":       result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "generated_at":   datetime.now(timezone.utc).isoformat(),
        })

    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    run_ts    = datetime.now(timezone.utc).isoformat()

    tiers = neonatal_tiers(CONDITION_TIERS, NEONATE_CONDITION_WEIGHT_KG)
    if not tiers:
        raise RuntimeError(
            "No neonatal tiers found in CONDITION_TIERS -- check that "
            "generator/vcv_generator.py's CONDITION_TIERS and "
            "NEONATE_CONDITION_WEIGHT_KG both list the same condition names."
        )

    combos_per_point = 1
    for v in THINNED_NEONATAL_GRID.values():
        combos_per_point *= len(v)

    manifest_rows = []
    log_tiers     = []
    grand_total = grand_valid = grand_invalid = 0

    print("=" * 70)
    print("  VCV Neonatal Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Tiers      : {[t['name'] for t in tiers]}")
    print(f"  Grid size  : {combos_per_point} combinations per mechanics point")
    print("=" * 70)
    sys.stdout.flush()

    for tier in tiers:
        tier_name  = tier["name"]
        n_cycles   = tier["n_cycles"]
        mechanics  = mechanics_grid(tier)
        tier_start = time.perf_counter()
        tier_total = tier_valid = tier_invalid = 0

        print(f"\n  [{tier_name}]")
        print(f"    Mechanics pairs : {len(mechanics)}")
        print(f"    Cycles/scenario : {n_cycles}")
        sys.stdout.flush()

        for C, R in mechanics:
            scenarios = _generate_thinned_dataset(
                condition_name           = tier_name,
                compliance_ml_per_cmH2O = C,
                resistance_cmH2O_L_s    = R,
                n_cycles                 = n_cycles,
            )

            for s in scenarios:
                tier_total += 1
                p = s["params"]
                m = s["metrics"]

                if s["is_valid"]:
                    tier_valid += 1
                else:
                    tier_invalid += 1

                manifest_rows.append({
                    "scenario_id":              s["scenario_id"],
                    "condition":                s["condition"],
                    "population":               "neonate",
                    "weight_kg":                p["weight_kg"],
                    "generated_at":             s["generated_at"],
                    "is_valid":                 s["is_valid"],
                    "invalid_reason":           s["invalid_reason"],
                    "compliance_ml_per_cmH2O":  p["compliance_ml_per_cmH2O"],
                    "resistance_cmH2O_L_s":     p["resistance_cmH2O_L_s"],
                    "tidal_volume_ml":          p["tidal_volume_ml"],
                    "respiratory_rate":         p["respiratory_rate"],
                    "peep_cmH2O":               p["peep_cmH2O"],
                    "ie_ratio":                 p["ie_ratio"],
                    "flow_pattern":             p["flow_pattern"],
                    "ppeak_cmH2O":              m.get("ppeak_cmH2O", ""),
                    "pplat_cmH2O":              m.get("pplat_cmH2O", ""),
                    "driving_p_cmH2O":          m.get("driving_p_cmH2O", ""),
                    "mean_paw_cmH2O":           m.get("mean_paw_cmH2O", ""),
                    "auto_peep_cmH2O":          m.get("auto_peep_cmH2O", ""),
                    "delivered_vt_ml":          m.get("delivered_vt_ml", ""),
                    "minute_vent_l":            m.get("minute_vent_l", ""),
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        log_tiers.append({
            "condition":        tier_name,
            "mechanics_pairs":  len(mechanics),
            "n_cycles":         n_cycles,
            "tier_total":       tier_total,
            "tier_valid":       tier_valid,
            "tier_invalid":     tier_invalid,
            "valid_pct":        round(100 * tier_valid / tier_total, 1) if tier_total else 0,
            "elapsed_s":        round(tier_elapsed, 1),
        })

        print(f"    Valid    : {tier_valid:,}  ({100*tier_valid/tier_total:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({100*tier_invalid/tier_total:.1f}%)")
        print(f"    Time     : {tier_elapsed:.1f}s")
        sys.stdout.flush()

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "vcv_neonatal_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                 "VCV",
        "population":           "neonate",
        "dataset_type":         "thinned",
        "generated_at":         run_ts,
        "output_dir":           str(OUTPUT_DIR.resolve()),
        "neonatal_conditions":  [t["name"] for t in tiers],
        "thinned_grid":         THINNED_NEONATAL_GRID,
        "combos_per_point":     combos_per_point,
        "grand_total":          grand_total,
        "grand_valid":          grand_valid,
        "grand_invalid":        grand_invalid,
        "valid_pct":            round(100 * grand_valid / grand_total, 1) if grand_total else 0,
        "total_elapsed_s":      round(run_elapsed, 1),
        "tiers":                log_tiers,
    }
    log_path = OUTPUT_DIR / "vcv_neonatal_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  VCV Neonatal Thinned Dataset Generation Complete")
    print("  " + "-" * 40)
    print(f"  Total scenarios  : {grand_total:,}")
    print(f"  Valid            : {grand_valid:,}  "
          f"({100*grand_valid/grand_total:.1f}%)")
    print(f"  Invalid          : {grand_invalid:,}  "
          f"({100*grand_invalid/grand_total:.1f}%)")
    print(f"  Manifest         : {manifest_path}")
    print(f"  Log              : {log_path}")
    print(f"  Total time       : {run_elapsed:.1f}s  "
          f"({run_elapsed/60:.1f} min)")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    run()
