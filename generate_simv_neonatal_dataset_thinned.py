"""
generate_simv_neonatal_dataset_thinned.py
--------------------------------------------
Thinned dataset generation for SIMV, neonatal population (Normal
Neonate, RDS). Parallel to generate_simv_dataset_thinned.py rather than
an extension of it, per ARCHITECTURE.md 1a's architectural decision.

See generate_vcv_neonatal_dataset_thinned.py's module docstring for the
full loose-coupling rationale. SIMV carries the same THINNED_SHARED_GRID
/ THINNED_VC_GRID / THINNED_PC_GRID split the adult SIMV thinned script
uses, for the same reason: the mandatory-mode axis (VC vs. PC) changes
which ventilator settings apply. f_window -- SIMV's own signature
parameter -- is reused directly from generator/neonatal_thinning.py's
shared grid (its physiological meaning as a synchronization window
doesn't change with population), while SIMV's mandatory (backup) rate
gets its own neonatal-specific bookends, distinct from the fully-
controlling respiratory_rate VCV/PCV/PRVC sweep, for the same reason the
adult SIMV thinned script gives its own RR dimension a different range
than its siblings.

Run: python generate_simv_neonatal_dataset_thinned.py
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
from generator.simv_generator import (
    generate_breath_cycles,
    CONDITION_TIERS,
    NEONATE_CONDITION_WEIGHT_KG,
    RECRUITMENT_SLOPES,
    _make_scenario_id,
)
from generator.neonatal_thinning import (
    neonatal_tiers,
    mechanics_grid,
    SIMV_BACKUP_RATE,
    PEEP_CMH2O,
    IE_RATIO,
    RISE_TIME_S,
    PRESSURE_SUPPORT_CMH2O,
    FLOW_CYCLE_THRESHOLD,
    TRIGGER_THRESHOLD_CMH2O,
    PMUS_PEAK_CMH2O,
    EFFORT_RATE_PER_MIN,
    EFFORT_DURATION_S,
    PMUS_CV,
    TIDAL_VOLUME_ML_PER_KG,
    FLOW_PATTERN,
    INSP_PRESSURE_CMH2O,
)

OUTPUT_DIR = Path("data/exports/simv_neonatal")
BASE_SEED  = 42  # matches the dashboard's own default PSV/SIMV seed

# f_window kept as an ASSUMPTION placeholder here: no neonatal-specific
# vendor guidance was found during the project's literature-grounding
# pass (that pass covered adult ventilators only), so the same
# physiologically-defined window fractions the adult SIMV thinned script
# uses are reused directly -- f_window describes a fraction of the
# breath cycle, not an absolute time, so it does not need a
# population-specific rescale the way e.g. rise_time_s does.
F_WINDOW = [0.15, 0.25, 0.30]

THINNED_SHARED_GRID = {
    "respiratory_rate":         SIMV_BACKUP_RATE,
    # Thinned to PEEP_CMH2O's lower (standard, Normal-Neonate-preset)
    # value rather than kept as a two-value dimension -- matching how
    # adult SIMV's own thinned script drops PEEP to a single value to
    # offset the extra combinatorial cost of the mandatory-mode axis
    # (CR0021). Still sourced from the one shared constant, not a new
    # magic number.
    "peep_cmH2O":               [PEEP_CMH2O[0]],
    "ie_ratio":                  IE_RATIO,
    "rise_time_s":               [RISE_TIME_S],
    "f_window":                  F_WINDOW,
    "pressure_support_cmH2O":    PRESSURE_SUPPORT_CMH2O,
    "flow_cycle_threshold":      FLOW_CYCLE_THRESHOLD,
    "trigger_threshold_cmH2O": [TRIGGER_THRESHOLD_CMH2O],
    "pmus_peak_cmH2O":           PMUS_PEAK_CMH2O,
    "effort_rate_per_min":      [EFFORT_RATE_PER_MIN],
    "effort_duration_s":        [EFFORT_DURATION_S],
    "pmus_cv":                  [PMUS_CV],
}

THINNED_VC_GRID = {
    "tidal_volume_ml_per_kg": TIDAL_VOLUME_ML_PER_KG,
    "flow_pattern":           FLOW_PATTERN,
}

THINNED_PC_GRID = {
    "insp_pressure_cmH2O": INSP_PRESSURE_CMH2O,
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
    Sweep the neonatal-thinned SIMV grid (shared x VC, then shared x PC)
    for one condition + mechanics pair. weight_kg comes from the
    IMPORTED NEONATE_CONDITION_WEIGHT_KG.
    """
    scenarios = []
    rng_base  = np.random.default_rng(seed)
    weight_kg = NEONATE_CONDITION_WEIGHT_KG[condition_name]
    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.0)

    shared_keys = list(THINNED_SHARED_GRID.keys())
    shared_vals = [THINNED_SHARED_GRID[k] for k in shared_keys]

    def _run_combo(base_params: dict):
        scenario_seed = int(rng_base.integers(0, 2**31))
        scenario_id   = _make_scenario_id(condition_name, base_params)

        try:
            result = generate_breath_cycles(base_params, n_cycles=n_cycles,
                                             seed=scenario_seed)
        except Exception as e:
            scenarios.append({
                "scenario_id":    scenario_id,
                "condition":      condition_name,
                "seed":           scenario_seed,
                "params":         base_params,
                "metrics":        {},
                "is_valid":       False,
                "invalid_reason": f"Generator error: {e}",
                "generated_at":   datetime.now(timezone.utc).isoformat(),
            })
            return

        metrics = {
            "ppeak_cmH2O":                     result.get("ppeak_cmH2O", ""),
            "delivered_vt_ml":                 result.get("delivered_vt_ml", ""),
            "patient_vt_ml":                   result.get("patient_vt_ml", ""),
            "driving_p_cmH2O":                 result.get("driving_p_cmH2O", ""),
            "mean_paw_cmH2O":                  result.get("mean_paw_cmH2O", ""),
            "auto_peep_cmH2O":                 result.get("auto_peep_cmH2O", ""),
            "minute_vent_l":                   result.get("minute_vent_l", ""),
            "mandatory_synchronized_fraction": result.get("mandatory_synchronized_fraction", ""),
            "n_spontaneous_breaths":           result.get("n_spontaneous_breaths", ""),
        }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "seed":           scenario_seed,
            "params":         base_params,
            "metrics":        metrics,
            "is_valid":       result.get("is_valid", True),
            "invalid_reason": result.get("invalid_reason", ""),
            "generated_at":   datetime.now(timezone.utc).isoformat(),
        })

    for shared_combo in itertools.product(*shared_vals):
        g = dict(zip(shared_keys, shared_combo))

        base = {
            "condition":                condition_name,
            "population":               "neonate",
            "weight_kg":                weight_kg,
            "compliance_ml_per_cmH2O":  compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":     resistance_cmH2O_L_s,
            "recruitment_slope":        rec_slope,
            **g,
        }

        # --- VC-mandatory sub-sweep ---------------------------------
        for vt_per_kg, pattern in itertools.product(
                THINNED_VC_GRID["tidal_volume_ml_per_kg"],
                THINNED_VC_GRID["flow_pattern"]):
            p = {
                **base,
                "mandatory_mode":   "VC",
                "tidal_volume_ml":  vt_per_kg * weight_kg,
                "flow_pattern":     pattern,
            }
            _run_combo(p)

        # --- PC-mandatory sub-sweep ---------------------------------
        for insp_p in THINNED_PC_GRID["insp_pressure_cmH2O"]:
            p = {
                **base,
                "mandatory_mode":       "PC",
                "insp_pressure_cmH2O": insp_p,
            }
            _run_combo(p)

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
            "generator/simv_generator.py's CONDITION_TIERS and "
            "NEONATE_CONDITION_WEIGHT_KG both list the same condition names."
        )

    shared_combos = 1
    for v in THINNED_SHARED_GRID.values():
        shared_combos *= len(v)
    vc_combos = len(THINNED_VC_GRID["tidal_volume_ml_per_kg"]) * len(THINNED_VC_GRID["flow_pattern"])
    pc_combos = len(THINNED_PC_GRID["insp_pressure_cmH2O"])
    combos_per_point = shared_combos * (vc_combos + pc_combos)

    manifest_rows = []
    log_tiers     = []
    grand_total = grand_valid = grand_invalid = 0

    print("=" * 70)
    print("  SIMV Neonatal Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Tiers      : {[t['name'] for t in tiers]}")
    print(f"  Grid size  : {combos_per_point:,} combinations per mechanics "
          f"point (shared {shared_combos} x [VC {vc_combos} + PC {pc_combos}])")
    print("=" * 70)
    sys.stdout.flush()

    for tier in tiers:
        tier_name  = tier["name"]
        n_cycles   = tier["n_cycles"]
        mechanics  = mechanics_grid(tier)
        tier_start = time.perf_counter()
        tier_total = tier_valid = tier_invalid = 0
        tier_vc = tier_pc = 0
        sync_fracs = []
        n_spont_list = []

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
                    if isinstance(m.get("mandatory_synchronized_fraction"), (int, float)):
                        sync_fracs.append(m["mandatory_synchronized_fraction"])
                    if isinstance(m.get("n_spontaneous_breaths"), (int, float)):
                        n_spont_list.append(m["n_spontaneous_breaths"])
                else:
                    tier_invalid += 1

                if p.get("mandatory_mode") == "VC":
                    tier_vc += 1
                else:
                    tier_pc += 1

                manifest_rows.append({
                    "scenario_id":              s["scenario_id"],
                    "condition":                s["condition"],
                    "population":               "neonate",
                    "weight_kg":                p["weight_kg"],
                    "generated_at":             s["generated_at"],
                    "seed":                     s["seed"],
                    "is_valid":                 s["is_valid"],
                    "invalid_reason":           s["invalid_reason"],
                    "compliance_ml_per_cmH2O":  p["compliance_ml_per_cmH2O"],
                    "resistance_cmH2O_L_s":     p["resistance_cmH2O_L_s"],
                    "mandatory_mode":           p.get("mandatory_mode", ""),
                    "respiratory_rate":         p["respiratory_rate"],
                    "peep_cmH2O":               p["peep_cmH2O"],
                    "ie_ratio":                 p["ie_ratio"],
                    "rise_time_s":              p["rise_time_s"],
                    "f_window":                 p["f_window"],
                    "pressure_support_cmH2O":   p["pressure_support_cmH2O"],
                    "flow_cycle_threshold":     p["flow_cycle_threshold"],
                    "trigger_threshold_cmH2O":  p["trigger_threshold_cmH2O"],
                    "pmus_peak_cmH2O":          p["pmus_peak_cmH2O"],
                    "effort_rate_per_min":      p["effort_rate_per_min"],
                    "effort_duration_s":        p["effort_duration_s"],
                    "pmus_cv":                  p["pmus_cv"],
                    "tidal_volume_ml":          p.get("tidal_volume_ml", ""),
                    "flow_pattern":             p.get("flow_pattern", ""),
                    "insp_pressure_cmH2O":      p.get("insp_pressure_cmH2O", ""),
                    **m,
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        mean_sync  = float(np.mean(sync_fracs)) if sync_fracs else None
        mean_spont = float(np.mean(n_spont_list)) if n_spont_list else None

        log_tiers.append({
            "condition":                   tier_name,
            "mechanics_pairs":             len(mechanics),
            "n_cycles":                    n_cycles,
            "tier_total":                  tier_total,
            "tier_valid":                  tier_valid,
            "tier_invalid":                tier_invalid,
            "tier_vc":                     tier_vc,
            "tier_pc":                     tier_pc,
            "mean_synchronized_fraction":  round(mean_sync, 3) if mean_sync is not None else None,
            "mean_spontaneous_breaths":    round(mean_spont, 2) if mean_spont is not None else None,
            "valid_pct":                   round(100 * tier_valid / tier_total, 1) if tier_total else 0,
            "elapsed_s":                   round(tier_elapsed, 1),
        })

        print(f"    Valid    : {tier_valid:,}  ({100*tier_valid/tier_total:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({100*tier_invalid/tier_total:.1f}%)")
        print(f"    VC / PC  : {tier_vc:,} / {tier_pc:,}")
        print(f"    Time     : {tier_elapsed:.1f}s")
        sys.stdout.flush()

    manifest_path = OUTPUT_DIR / "simv_neonatal_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                 "SIMV",
        "population":           "neonate",
        "dataset_type":         "thinned",
        "generated_at":         run_ts,
        "output_dir":           str(OUTPUT_DIR.resolve()),
        "neonatal_conditions":  [t["name"] for t in tiers],
        "thinned_shared_grid":  THINNED_SHARED_GRID,
        "thinned_vc_grid":      THINNED_VC_GRID,
        "thinned_pc_grid":      THINNED_PC_GRID,
        "combos_per_point":     combos_per_point,
        "grand_total":          grand_total,
        "grand_valid":          grand_valid,
        "grand_invalid":        grand_invalid,
        "valid_pct":            round(100 * grand_valid / grand_total, 1) if grand_total else 0,
        "total_elapsed_s":      round(run_elapsed, 1),
        "total_elapsed_min":    round(run_elapsed / 60, 1),
        "tiers":                log_tiers,
        "notes": [
            "Grid size and mandatory-backup-rate bookends are a starting "
            "point, not a measured runtime target the way adult SIMV's "
            "were (CR0021 timed a real 60-scenario sample before finalizing "
            "its grid). Time a small sample here the same way before a full "
            "overnight run and thin THINNED_SHARED_GRID further if the "
            "projected runtime runs long.",
        ],
    }
    log_path = OUTPUT_DIR / "simv_neonatal_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print("  SIMV Neonatal Thinned Dataset Generation Complete")
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
