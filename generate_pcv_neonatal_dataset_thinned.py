"""
generate_pcv_neonatal_dataset_thinned.py
------------------------------------------
Thinned dataset generation for PCV, neonatal population (Normal Neonate,
RDS). Parallel to generate_pcv_dataset_thinned.py rather than an
extension of it, per ARCHITECTURE.md 1a's architectural decision.

See generate_vcv_neonatal_dataset_thinned.py's module docstring for the
full loose-coupling rationale -- the same design applies here: tiers,
weights, recruitment slopes, and scenario-ID formatting are all imported
from generator/pcv_generator.py rather than duplicated; only the
ventilator-side THINNED grid is a local decision, and it is shared with
the other four neonatal scripts via generator/neonatal_thinning.py.

Run: python generate_pcv_neonatal_dataset_thinned.py
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
from generator.pcv_generator import (
    generate_breath_cycles,
    CONDITION_TIERS,
    NEONATE_CONDITION_WEIGHT_KG,
    RECRUITMENT_SLOPES,
    _make_scenario_id,
)
from generator.neonatal_thinning import (
    neonatal_tiers,
    mechanics_grid,
    INSP_PRESSURE_CMH2O,
    RESPIRATORY_RATE,
    PEEP_CMH2O,
    IE_RATIO,
    RISE_TIME_S,
)

OUTPUT_DIR = Path("data/exports/pcv_neonatal")

THINNED_NEONATAL_GRID = {
    "insp_pressure_cmH2O": INSP_PRESSURE_CMH2O,
    "respiratory_rate":    RESPIRATORY_RATE,
    "peep_cmH2O":          PEEP_CMH2O,
    "ie_ratio":            IE_RATIO,
    "rise_time_s":         [RISE_TIME_S],
}


# ---------------------------------------------------------------------------
# Per-mechanics-point sweep
# ---------------------------------------------------------------------------

def _generate_thinned_dataset(condition_name: str,
                               compliance_ml_per_cmH2O: float,
                               resistance_cmH2O_L_s: float,
                               n_cycles: int) -> list:
    """
    Sweep the neonatal-thinned PCV grid for one condition + mechanics
    pair. weight_kg comes from the IMPORTED NEONATE_CONDITION_WEIGHT_KG.
    """
    scenarios = []
    weight_kg = NEONATE_CONDITION_WEIGHT_KG[condition_name]
    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.0)

    keys   = ["insp_pressure_cmH2O", "respiratory_rate",
              "peep_cmH2O", "ie_ratio", "rise_time_s"]
    values = [THINNED_NEONATAL_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        p_insp, rr, peep, ie, t_rise = combo

        params = {
            "respiratory_rate":        rr,
            "insp_pressure_cmH2O":     p_insp,
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "rise_time_s":             t_rise,
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
            "ppeak_cmH2O":         result["ppeak_cmH2O"],
            "delivered_vt_ml":     result["delivered_vt_ml"],
            "driving_p_cmH2O":     result["driving_p_cmH2O"],
            "mean_paw_cmH2O":      result["mean_paw_cmH2O"],
            "auto_peep_cmH2O":     result["auto_peep_cmH2O"],
            "fill_fraction":       result["fill_fraction"],
            "minute_vent_l":       result["minute_vent_l"],
            "time_to_peak_flow_s": result["time_to_peak_flow_s"],
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
            "generator/pcv_generator.py's CONDITION_TIERS and "
            "NEONATE_CONDITION_WEIGHT_KG both list the same condition names."
        )

    combos_per_point = 1
    for v in THINNED_NEONATAL_GRID.values():
        combos_per_point *= len(v)

    manifest_rows = []
    log_tiers     = []
    grand_total = grand_valid = grand_invalid = 0

    print("=" * 70)
    print("  PCV Neonatal Thinned Dataset Generation")
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
                    "insp_pressure_cmH2O":      p["insp_pressure_cmH2O"],
                    "respiratory_rate":         p["respiratory_rate"],
                    "peep_cmH2O":               p["peep_cmH2O"],
                    "ie_ratio":                 p["ie_ratio"],
                    "rise_time_s":              p["rise_time_s"],
                    "ppeak_cmH2O":              m.get("ppeak_cmH2O", ""),
                    "delivered_vt_ml":          m.get("delivered_vt_ml", ""),
                    "driving_p_cmH2O":          m.get("driving_p_cmH2O", ""),
                    "mean_paw_cmH2O":           m.get("mean_paw_cmH2O", ""),
                    "auto_peep_cmH2O":          m.get("auto_peep_cmH2O", ""),
                    "fill_fraction":            m.get("fill_fraction", ""),
                    "minute_vent_l":            m.get("minute_vent_l", ""),
                    "time_to_peak_flow_s":      m.get("time_to_peak_flow_s", ""),
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

    manifest_path = OUTPUT_DIR / "pcv_neonatal_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                 "PCV",
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
    log_path = OUTPUT_DIR / "pcv_neonatal_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  PCV Neonatal Thinned Dataset Generation Complete")
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
