"""
generate_psv_neonatal_dataset_thinned.py
------------------------------------------
Thinned dataset generation for PSV, neonatal population (Normal Neonate,
RDS). Parallel to generate_psv_dataset_thinned.py rather than an
extension of it, per ARCHITECTURE.md 1a's architectural decision.

See generate_vcv_neonatal_dataset_thinned.py's module docstring for the
full loose-coupling rationale. PSV additionally carries patient-effort
parameters and stochastic breath-to-breath variability -- both grid
dimensions and the per-scenario seed strategy mirror the adult PSV
thinned script's own conventions.

Run: python generate_psv_neonatal_dataset_thinned.py
"""

import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator.psv_generator import (
    generate_breath_cycles,
    CONDITION_TIERS,
    NEONATE_CONDITION_WEIGHT_KG,
    RECRUITMENT_SLOPES,
    _make_scenario_id,
)
from generator.neonatal_thinning import (
    neonatal_tiers,
    mechanics_grid,
    dominant_dyssync,
    PRESSURE_SUPPORT_CMH2O,
    PEEP_CMH2O,
    FLOW_CYCLE_THRESHOLD,
    TRIGGER_THRESHOLD_CMH2O,
    RISE_TIME_S,
    PMUS_PEAK_CMH2O,
    EFFORT_RATE_PER_MIN,
    EFFORT_DURATION_S,
    PMUS_CV,
)

OUTPUT_DIR = Path("data/exports/psv_neonatal")
BASE_SEED  = 42  # matches the dashboard's own default PSV/SIMV seed

THINNED_NEONATAL_GRID = {
    "pressure_support_cmH2O":   PRESSURE_SUPPORT_CMH2O,
    "peep_cmH2O":               PEEP_CMH2O,
    "flow_cycle_threshold":     FLOW_CYCLE_THRESHOLD,
    "trigger_threshold_cmH2O": [TRIGGER_THRESHOLD_CMH2O],
    "rise_time_s":              [RISE_TIME_S],
    "pmus_peak_cmH2O":          PMUS_PEAK_CMH2O,
    "effort_rate_per_min":     [EFFORT_RATE_PER_MIN],
    "effort_duration_s":       [EFFORT_DURATION_S],
    "pmus_cv":                 [PMUS_CV],
}


# ---------------------------------------------------------------------------
# Per-mechanics-point sweep
# ---------------------------------------------------------------------------

def _generate_thinned_dataset(condition_name: str,
                               compliance_ml_per_cmH2O: float,
                               resistance_cmH2O_L_s: float,
                               n_cycles: int,
                               seed: int = BASE_SEED) -> list:
    """
    Sweep the neonatal-thinned PSV grid for one condition + mechanics
    pair. weight_kg comes from the IMPORTED NEONATE_CONDITION_WEIGHT_KG.
    Each scenario gets a deterministic seed derived from a base RNG,
    matching the seeding strategy documented for the adult PSV thinned
    script (CR0013): any scenario can be regenerated identically from
    its manifest row's params + seed.
    """
    scenarios = []
    rng_base  = np.random.default_rng(seed)
    weight_kg = NEONATE_CONDITION_WEIGHT_KG[condition_name]
    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.0)

    keys   = ["pressure_support_cmH2O", "peep_cmH2O", "flow_cycle_threshold",
              "trigger_threshold_cmH2O", "rise_time_s", "pmus_peak_cmH2O",
              "effort_rate_per_min", "effort_duration_s", "pmus_cv"]
    values = [THINNED_NEONATAL_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        g = dict(zip(keys, combo))

        params = {
            "condition":                condition_name,
            "population":               "neonate",
            "weight_kg":                weight_kg,
            "compliance_ml_per_cmH2O":  compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":     resistance_cmH2O_L_s,
            "recruitment_slope":        rec_slope,
            **g,
        }

        scenario_seed = int(rng_base.integers(0, 2**31))
        scenario_id   = _make_scenario_id(condition_name, params)

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles,
                                             seed=scenario_seed)
        except Exception as e:
            scenarios.append({
                "scenario_id":     scenario_id,
                "condition":       condition_name,
                "seed":            scenario_seed,
                "params":          params,
                "metrics":         {},
                "is_valid":        False,
                "invalid_reason":  f"Generator error: {e}",
                "dyssync_labels":  [],
                "generated_at":    datetime.now(timezone.utc).isoformat(),
            })
            continue

        metrics = {
            "ppeak_cmH2O":                  result.get("ppeak_cmH2O", ""),
            "delivered_vt_ml":              result.get("delivered_vt_ml", ""),
            "patient_vt_ml":                result.get("patient_vt_ml", ""),
            "driving_p_cmH2O":              result.get("driving_p_cmH2O", ""),
            "mean_paw_cmH2O":               result.get("mean_paw_cmH2O", ""),
            "auto_peep_cmH2O":              result.get("auto_peep_cmH2O", ""),
            "total_peep_cmH2O":             result.get("total_peep_cmH2O", ""),
            "fill_fraction":                result.get("fill_fraction", ""),
            "minute_vent_l":                result.get("minute_vent_l", ""),
            "pres_peak_cmH2O":              result.get("pres_peak_cmH2O", ""),
            "pel_end_insp_cmH2O":           result.get("pel_end_insp_cmH2O", ""),
            "stress_index":                 result.get("stress_index", ""),
            "pres_pel_ratio":                result.get("pres_pel_ratio", ""),
            "triggered_breath_rate":        result.get("triggered_breath_rate", ""),
            "ineffective_trigger_fraction": result.get("ineffective_trigger_fraction", ""),
        }

        scenarios.append({
            "scenario_id":     scenario_id,
            "condition":       condition_name,
            "seed":            scenario_seed,
            "params":          params,
            "metrics":         metrics,
            "is_valid":        result.get("is_valid", True),
            "invalid_reason":  result.get("invalid_reason", ""),
            "dyssync_labels":  result.get("breath_dyssynchrony_labels", []),
            "generated_at":    datetime.now(timezone.utc).isoformat(),
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
            "generator/psv_generator.py's CONDITION_TIERS and "
            "NEONATE_CONDITION_WEIGHT_KG both list the same condition names."
        )

    combos_per_point = 1
    for v in THINNED_NEONATAL_GRID.values():
        combos_per_point *= len(v)

    manifest_rows = []
    log_tiers     = []
    grand_total = grand_valid = grand_invalid = 0

    print("=" * 70)
    print("  PSV Neonatal Thinned Dataset Generation")
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
        tier_dyssync_total = {}

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
                labels = s["dyssync_labels"]

                if s["is_valid"]:
                    tier_valid += 1
                else:
                    tier_invalid += 1

                for label in labels:
                    tier_dyssync_total[label] = tier_dyssync_total.get(label, 0) + 1

                manifest_rows.append({
                    "scenario_id":                  s["scenario_id"],
                    "condition":                    s["condition"],
                    "population":                   "neonate",
                    "weight_kg":                    p["weight_kg"],
                    "generated_at":                 s["generated_at"],
                    "seed":                         s["seed"],
                    "is_valid":                     s["is_valid"],
                    "invalid_reason":               s["invalid_reason"],
                    "compliance_ml_per_cmH2O":      p["compliance_ml_per_cmH2O"],
                    "resistance_cmH2O_L_s":         p["resistance_cmH2O_L_s"],
                    "pressure_support_cmH2O":       p["pressure_support_cmH2O"],
                    "peep_cmH2O":                   p["peep_cmH2O"],
                    "flow_cycle_threshold":         p["flow_cycle_threshold"],
                    "trigger_threshold_cmH2O":      p["trigger_threshold_cmH2O"],
                    "rise_time_s":                  p["rise_time_s"],
                    "pmus_peak_cmH2O":               p["pmus_peak_cmH2O"],
                    "effort_rate_per_min":           p["effort_rate_per_min"],
                    "effort_duration_s":             p["effort_duration_s"],
                    "pmus_cv":                       p["pmus_cv"],
                    **m,
                    "breath_dyssynchrony_labels":   ";".join(labels),
                    "dominant_dyssync":             dominant_dyssync(labels),
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        log_tiers.append({
            "condition":         tier_name,
            "mechanics_pairs":   len(mechanics),
            "n_cycles":          n_cycles,
            "tier_total":        tier_total,
            "tier_valid":        tier_valid,
            "tier_invalid":      tier_invalid,
            "valid_pct":         round(100 * tier_valid / tier_total, 1) if tier_total else 0,
            "elapsed_s":         round(tier_elapsed, 1),
            "dyssync_totals":    tier_dyssync_total,
        })

        print(f"    Valid    : {tier_valid:,}  ({100*tier_valid/tier_total:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({100*tier_invalid/tier_total:.1f}%)")
        print(f"    Time     : {tier_elapsed:.1f}s")
        sys.stdout.flush()

    manifest_path = OUTPUT_DIR / "psv_neonatal_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                 "PSV",
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
    log_path = OUTPUT_DIR / "psv_neonatal_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  PSV Neonatal Thinned Dataset Generation Complete")
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
