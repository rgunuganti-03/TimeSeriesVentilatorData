"""
generate_prvc_neonatal_dataset_thinned.py
--------------------------------------------
Thinned dataset generation for PRVC, neonatal population (Normal
Neonate, RDS). Parallel to generate_prvc_dataset_thinned.py rather than
an extension of it, per ARCHITECTURE.md 1a's architectural decision.

See generate_vcv_neonatal_dataset_thinned.py's module docstring for the
full loose-coupling rationale. pressure_ceiling_cmH2O is PRVC's own
signature parameter and is thinned least aggressively of any dimension
here, mirroring how the adult PRVC thinned script treats its own
pressure_ceiling dimension. Algorithm-control constants specific to
PRVC's outer adaptive loop (adaptation step, VT tolerance) are NOT
swept or re-declared here -- they are read back from each scenario's own
result dict, exactly like every other metric, so a tuning change to
those constants in prvc_generator.py needs no edit here either.

Run: python generate_prvc_neonatal_dataset_thinned.py
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
from generator.prvc_generator import (
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
    PRESSURE_CEILING_CMH2O,
)

OUTPUT_DIR = Path("data/exports/prvc_neonatal")

THINNED_NEONATAL_GRID = {
    "pressure_ceiling_cmH2O":  PRESSURE_CEILING_CMH2O,
    "tidal_volume_ml_per_kg":  TIDAL_VOLUME_ML_PER_KG,
    "respiratory_rate":        RESPIRATORY_RATE,
    "peep_cmH2O":              PEEP_CMH2O,
    "ie_ratio":                 IE_RATIO,
    "flow_pattern":             FLOW_PATTERN,
}


# ---------------------------------------------------------------------------
# Per-mechanics-point sweep
# ---------------------------------------------------------------------------

def _generate_thinned_dataset(condition_name: str,
                               compliance_ml_per_cmH2O: float,
                               resistance_cmH2O_L_s: float,
                               n_cycles: int) -> list:
    """
    Sweep the neonatal-thinned PRVC grid for one condition + mechanics
    pair. weight_kg comes from the IMPORTED NEONATE_CONDITION_WEIGHT_KG.
    """
    scenarios = []
    weight_kg = NEONATE_CONDITION_WEIGHT_KG[condition_name]
    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.0)

    keys   = ["pressure_ceiling_cmH2O", "tidal_volume_ml_per_kg",
              "respiratory_rate", "peep_cmH2O", "ie_ratio", "flow_pattern"]
    values = [THINNED_NEONATAL_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        g = dict(zip(keys, combo))
        vt_mL = g["tidal_volume_ml_per_kg"] * weight_kg

        params = {
            "condition":                condition_name,
            "population":               "neonate",
            "weight_kg":                weight_kg,
            "compliance_ml_per_cmH2O":  compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":     resistance_cmH2O_L_s,
            "recruitment_slope":        rec_slope,
            "pressure_ceiling_cmH2O":   g["pressure_ceiling_cmH2O"],
            "tidal_volume_ml":          vt_mL,
            "respiratory_rate":         g["respiratory_rate"],
            "peep_cmH2O":               g["peep_cmH2O"],
            "ie_ratio":                 g["ie_ratio"],
            "flow_pattern":             g["flow_pattern"],
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

        # Pulled defensively via .get() rather than a hardcoded key list:
        # PRVC's per-breath convergence bookkeeping (adaptation step, VT
        # tolerance, test-breath plateau, breaths-to-converge, converged,
        # ceiling-limited) is echoed back on the result dict itself, so
        # whatever prvc_generator.py currently reports is carried straight
        # into the manifest without this script needing to know the exact
        # field set in advance.
        metrics = {
            "ppeak_cmH2O":              result.get("ppeak_cmH2O", ""),
            "delivered_vt_ml":          result.get("delivered_vt_ml", ""),
            "driving_p_cmH2O":          result.get("driving_p_cmH2O", ""),
            "mean_paw_cmH2O":           result.get("mean_paw_cmH2O", ""),
            "auto_peep_cmH2O":          result.get("auto_peep_cmH2O", ""),
            "minute_vent_l":            result.get("minute_vent_l", ""),
            "adaptation_step_cmH2O":    result.get("adaptation_step_cmH2O", ""),
            "vt_tolerance_frac":        result.get("vt_tolerance_frac", ""),
            "test_breath_plateau_cmH2O": result.get("test_breath_plateau_cmH2O", ""),
            "breaths_to_converge":      result.get("breaths_to_converge", ""),
            "converged":                result.get("converged", ""),
            "ceiling_limited":          result.get("ceiling_limited", ""),
        }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "is_valid":       result.get("is_valid", True),
            "invalid_reason": result.get("invalid_reason", ""),
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
            "generator/prvc_generator.py's CONDITION_TIERS and "
            "NEONATE_CONDITION_WEIGHT_KG both list the same condition names."
        )

    combos_per_point = 1
    for v in THINNED_NEONATAL_GRID.values():
        combos_per_point *= len(v)

    manifest_rows = []
    log_tiers     = []
    grand_total = grand_valid = grand_invalid = 0
    grand_converged = grand_ceiling_limited = 0

    print("=" * 70)
    print("  PRVC Neonatal Thinned Dataset Generation")
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
        tier_converged = tier_ceiling_limited = 0

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

                if m.get("converged"):
                    tier_converged += 1
                if m.get("ceiling_limited"):
                    tier_ceiling_limited += 1

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
                    "pressure_ceiling_cmH2O":   p["pressure_ceiling_cmH2O"],
                    "tidal_volume_ml":          p["tidal_volume_ml"],
                    "respiratory_rate":         p["respiratory_rate"],
                    "peep_cmH2O":               p["peep_cmH2O"],
                    "ie_ratio":                 p["ie_ratio"],
                    "flow_pattern":             p["flow_pattern"],
                    **m,
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid
        grand_converged += tier_converged
        grand_ceiling_limited += tier_ceiling_limited

        log_tiers.append({
            "condition":         tier_name,
            "mechanics_pairs":   len(mechanics),
            "n_cycles":          n_cycles,
            "tier_total":        tier_total,
            "tier_valid":        tier_valid,
            "tier_invalid":      tier_invalid,
            "valid_pct":         round(100 * tier_valid / tier_total, 1) if tier_total else 0,
            "converged_pct":     round(100 * tier_converged / tier_total, 1) if tier_total else 0,
            "ceiling_limited_pct": round(100 * tier_ceiling_limited / tier_total, 1) if tier_total else 0,
            "elapsed_s":         round(tier_elapsed, 1),
        })

        print(f"    Valid    : {tier_valid:,}  ({100*tier_valid/tier_total:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({100*tier_invalid/tier_total:.1f}%)")
        print(f"    Time     : {tier_elapsed:.1f}s")
        sys.stdout.flush()

    manifest_path = OUTPUT_DIR / "prvc_neonatal_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                 "PRVC",
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
        "grand_converged":      grand_converged,
        "grand_ceiling_limited": grand_ceiling_limited,
        "valid_pct":            round(100 * grand_valid / grand_total, 1) if grand_total else 0,
        "total_elapsed_s":      round(run_elapsed, 1),
        "tiers":                log_tiers,
    }
    log_path = OUTPUT_DIR / "prvc_neonatal_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  PRVC Neonatal Thinned Dataset Generation Complete")
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
