"""
generate_prvc_dataset_thinned.py
----------------------------------
Self-contained PRVC dataset generator with the thinned parameter grid
built in. Sweeps only the kept parameter combinations -- no intermediate
full dataset, no HDF5, no post-processing step.

Run from the project root:
    python generate_prvc_dataset_thinned.py

For an overnight run:
    nohup python -u generate_prvc_dataset_thinned.py > prvc_thinned.log 2>&1 &

Monitor progress:
    tail -f prvc_thinned.log

Output:
    data/exports/prvc/
        prvc_manifest_thinned.csv   -- one row per scenario (valid + invalid)
        prvc_generation_log.json    -- run summary: counts, timing, config

Thinned parameter grid (rationale documented below):

    Tidal volume target : 4, 6, 10 mL/kg IBW
        6 is the ARDSnet lung-protective anchor; 4 and 10 bound the
        protective-to-upper-normal range. Dropped 8 as a redundant
        interpolation between 6 and 10 -- the staircase's qualitative
        behavior at 8 mL/kg tracks smoothly between the 6 and 10 mL/kg
        cases rather than revealing anything distinct.

    Respiratory rate : 8, 16, 24, 30 bpm
        Same rationale and exact values as pcv_generator's thinning --
        adjacent 4 bpm steps change fill fraction and I:E-driven timing
        only marginally. Kept values span distinct fill-fraction bands
        and matter equally to PRVC's test-breath and PC-breath physics.

    PEEP : 0, 8, 16, 20 cmH2O
        Matches pcv_generator's thinning exactly. Spans no-PEEP,
        moderate, high, and maximum clinical ranges -- coarse enough to
        still show the PEEP-recruitment effect on converged working
        pressure (see PRVC_PARAMETER_GRID.md, Moderate ARDS recruitment
        slope 0.90) without the full 6-point resolution.

    I:E ratio : 1:1, 1:2, 1:3 (all kept)
        Each ratio changes inspiratory time non-linearly, which affects
        both the VC test breath's flow rate (and thus its plateau
        pressure reading) and every subsequent PC breath's fill
        fraction. All three must be kept, matching vcv/pcv precedent.

    Pressure ceiling : 15, 20, 25, 35 cmH2O above PEEP
        PRVC-unique dimension with no vcv/pcv/psv analogue -- this is
        the single parameter that determines whether ceiling-limited
        non-convergence appears at all (see PRVC_PARAMETER_GRID.md).
        Given that outsized diagnostic importance, thinned least
        aggressively of the five swept dimensions: only 30 dropped (a
        redundant interpolation between 25 and 35), keeping the full
        tight-to-generous span intact.

    adaptation_step_cmH2O and vt_tolerance_frac are NOT swept here --
    both are fixed uniform constants per the project decision documented
    in PRVC_PARAMETER_GRID.md (2.0 cmH2O and 0.10 respectively). See that
    doc for the C_threshold analysis behind treating these as one
    deployed-device algorithm rather than a per-condition grid axis.

Expected combinations per mechanics point:
    3 VT x 4 RR x 4 PEEP x 3 IE x 4 CEIL = 576 (vs 2,520 in the full grid)
    Reduction: 77.1% fewer scenarios per mechanics point

Runtime note:
    prvc_generator uses explicit Euler multi-compartment integration (no
    scipy.integrate dependency) -- no ODE solver overhead per breath.
    Estimated total runtime across all 137 mechanics points: ~4-5 hours.
    COPD and Bronchospasm use 25 cycles/scenario (vs 12 for all other
    tiers) for auto-PEEP and pressure-staircase steady state, roughly
    doubling their per-scenario cost -- they dominate total runtime.
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
from generator.prvc_generator import (
    ADAPTATION_STEP_CMH2O_DEFAULT,
    IBW_KG,
    PPEAK_MAX_CMHH2O,
    VT_MAX_ML,
    VT_MIN_ML,
    VT_TOLERANCE_FRAC_DEFAULT,
    _make_scenario_id,
    generate_breath_cycles,
)


# ---------------------------------------------------------------------------
# Thinned parameter grid
# ---------------------------------------------------------------------------

THINNED_PARAMETER_GRID = {
    "vt_target_ml_per_kg":     [4, 6, 10],
    "respiratory_rate":        [8, 16, 24, 30],
    "peep_cmH2O":              [0, 8, 16, 20],
    "ie_ratio":                [1.0, 0.5, 0.33],
    "pressure_ceiling_cmH2O":  [15, 20, 25, 35],
}

# ---------------------------------------------------------------------------
# Condition tier definitions -- identical to full generator
# ---------------------------------------------------------------------------

CONDITION_TIERS = [
    {"name": "Normal",         "compliance_range": (60, 100), "compliance_step": 10,
     "resistance_range": (8, 12),   "resistance_step": 1, "n_cycles": 12},
    {"name": "Mild ARDS",      "compliance_range": (40, 55),  "compliance_step": 5,
     "resistance_range": (10, 14),  "resistance_step": 2, "n_cycles": 12},
    {"name": "Moderate ARDS",  "compliance_range": (28, 40),  "compliance_step": 4,
     "resistance_range": (12, 16),  "resistance_step": 2, "n_cycles": 12},
    {"name": "Severe ARDS",    "compliance_range": (15, 28),  "compliance_step": 4,
     "resistance_range": (14, 20),  "resistance_step": 3, "n_cycles": 12},
    {"name": "COPD",           "compliance_range": (80, 150), "compliance_step": 20,
     "resistance_range": (18, 35),  "resistance_step": 5, "n_cycles": 25},
    {"name": "Bronchospasm",   "compliance_range": (60, 90),  "compliance_step": 10,
     "resistance_range": (25, 50),  "resistance_step": 5, "n_cycles": 25},
    {"name": "Pneumonia",      "compliance_range": (40, 65),  "compliance_step": 5,
     "resistance_range": (10, 16),  "resistance_step": 2, "n_cycles": 12},
]

OUTPUT_DIR = Path("data/exports/prvc")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mechanics_grid(tier: dict) -> list:
    c_start, c_stop = tier["compliance_range"]
    r_start, r_stop = tier["resistance_range"]
    compliances = np.arange(
        c_start, c_stop + tier["compliance_step"], tier["compliance_step"]
    ).tolist()
    resistances = np.arange(
        r_start, r_stop + tier["resistance_step"], tier["resistance_step"]
    ).tolist()
    return [(float(c), float(r)) for c in compliances for r in resistances]


def _generate_thinned_dataset(
    condition_name:          str,
    compliance_ml_per_cmH2O: float,
    resistance_cmH2O_L_s:    float,
    n_cycles:                int = 12,
) -> list:
    """
    Sweep the thinned PRVC parameter grid for one condition + mechanics
    pair. Returns a list of scenario dicts -- metrics only, no waveform
    arrays (regeneratable on demand via generate_breath_cycles(params,
    seed=seed), since prvc_generator has no stochastic elements).
    """
    scenarios = []

    keys = ["vt_target_ml_per_kg", "respiratory_rate",
             "peep_cmH2O", "ie_ratio", "pressure_ceiling_cmH2O"]
    values = [THINNED_PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        vt_per_kg, rr, peep, ie, ceiling = combo
        vt_mL = vt_per_kg * IBW_KG

        params = {
            "vt_target_ml":            vt_mL,
            "respiratory_rate":        rr,
            "peep_cmH2O":              peep,
            "ie_ratio":                ie,
            "pressure_ceiling_cmH2O":  ceiling,
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "condition":               condition_name,
            "adaptation_step_cmH2O":   ADAPTATION_STEP_CMH2O_DEFAULT,
            "vt_tolerance_frac":       VT_TOLERANCE_FRAC_DEFAULT,
        }

        scenario_id = _make_scenario_id(
            condition_name, compliance_ml_per_cmH2O, resistance_cmH2O_L_s, params
        )
        seed = abs(hash((condition_name, compliance_ml_per_cmH2O,
                          resistance_cmH2O_L_s, tuple(combo)))) % (2**31)

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles, seed=seed)
        except Exception as exc:
            scenarios.append({
                "scenario_id":    scenario_id,
                "condition":      condition_name,
                "params":         params,
                "metrics":        {},
                "is_valid":       False,
                "invalid_reason": f"Generator error: {exc}",
                "generated_at":   datetime.now(timezone.utc).isoformat(),
                "seed":           seed,
            })
            continue

        metrics = {
            "ppeak_cmH2O":               result["ppeak_cmH2O"],
            "delivered_vt_ml":           result["delivered_vt_ml"],
            "driving_p_cmH2O":           result["driving_p_cmH2O"],
            "mean_paw_cmH2O":            result["mean_paw_cmH2O"],
            "auto_peep_cmH2O":           result["auto_peep_cmH2O"],
            "fill_fraction":             result["fill_fraction"],
            "minute_vent_l":             result["minute_vent_l"],
            "test_breath_plateau_cmH2O": result["test_breath_plateau_cmH2O"],
            "breaths_to_converge":       result["breaths_to_converge"],
            "converged":                 result["converged"],
            "ceiling_limited":           result["ceiling_limited"],
        }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "is_valid":       result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "generated_at":   datetime.now(timezone.utc).isoformat(),
            "seed":           seed,
        })

    return scenarios


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    run_ts = datetime.now(timezone.utc).isoformat()

    manifest_rows = []
    log_tiers = []

    grand_total = 0
    grand_valid = 0
    grand_invalid = 0
    grand_converged = 0
    grand_ceiling_limited = 0

    combos_per_point = (
        len(THINNED_PARAMETER_GRID["vt_target_ml_per_kg"])
        * len(THINNED_PARAMETER_GRID["respiratory_rate"])
        * len(THINNED_PARAMETER_GRID["peep_cmH2O"])
        * len(THINNED_PARAMETER_GRID["ie_ratio"])
        * len(THINNED_PARAMETER_GRID["pressure_ceiling_cmH2O"])
    )

    print("=" * 70)
    print("  PRVC Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Grid size  : {combos_per_point} combinations per mechanics point")
    print("  Note       : Explicit Euler multi-compartment -- no scipy dependency")
    print("  Tip        : Run with nohup -u and tail -f prvc_thinned.log")
    print("=" * 70)
    sys.stdout.flush()

    for tier in CONDITION_TIERS:
        tier_name = tier["name"]
        mechanics = _mechanics_grid(tier)
        n_cycles = tier["n_cycles"]
        tier_start = time.perf_counter()

        tier_total = 0
        tier_valid = 0
        tier_invalid = 0
        tier_converged = 0
        tier_ceiling_limited = 0

        print(f"\n  [{tier_name}]")
        print(f"    Mechanics pairs : {len(mechanics)}")
        print(f"    Cycles/scenario : {n_cycles}")
        print(f"    Total scenarios : {len(mechanics) * combos_per_point:,} (estimated)")
        sys.stdout.flush()

        for C, R in mechanics:
            scenarios = _generate_thinned_dataset(
                condition_name=tier_name,
                compliance_ml_per_cmH2O=C,
                resistance_cmH2O_L_s=R,
                n_cycles=n_cycles,
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
                    "scenario_id":               s["scenario_id"],
                    "condition":                 s["condition"],
                    "generated_at":              s["generated_at"],
                    "is_valid":                  s["is_valid"],
                    "invalid_reason":            s["invalid_reason"],
                    "compliance_ml_per_cmH2O":   p["compliance_ml_per_cmH2O"],
                    "resistance_cmH2O_L_s":      p["resistance_cmH2O_L_s"],
                    "vt_target_ml":              p["vt_target_ml"],
                    "respiratory_rate":          p["respiratory_rate"],
                    "peep_cmH2O":                p["peep_cmH2O"],
                    "ie_ratio":                  p["ie_ratio"],
                    "pressure_ceiling_cmH2O":    p["pressure_ceiling_cmH2O"],
                    "adaptation_step_cmH2O":     p["adaptation_step_cmH2O"],
                    "vt_tolerance_frac":         p["vt_tolerance_frac"],
                    "ppeak_cmH2O":               m.get("ppeak_cmH2O", ""),
                    "delivered_vt_ml":           m.get("delivered_vt_ml", ""),
                    "driving_p_cmH2O":           m.get("driving_p_cmH2O", ""),
                    "mean_paw_cmH2O":            m.get("mean_paw_cmH2O", ""),
                    "auto_peep_cmH2O":           m.get("auto_peep_cmH2O", ""),
                    "fill_fraction":             m.get("fill_fraction", ""),
                    "minute_vent_l":             m.get("minute_vent_l", ""),
                    "test_breath_plateau_cmH2O": m.get("test_breath_plateau_cmH2O", ""),
                    "breaths_to_converge":       m.get("breaths_to_converge", ""),
                    "converged":                 m.get("converged", ""),
                    "ceiling_limited":           m.get("ceiling_limited", ""),
                    "seed":                      s["seed"],
                })

        tier_elapsed = time.perf_counter() - tier_start
        valid_pct = 100 * tier_valid / tier_total if tier_total > 0 else 0
        invalid_pct = 100 * tier_invalid / tier_total if tier_total > 0 else 0
        converged_pct = 100 * tier_converged / tier_total if tier_total > 0 else 0
        ceiling_pct = 100 * tier_ceiling_limited / tier_total if tier_total > 0 else 0

        print(f"    Total           : {tier_total:,}")
        print(f"    Valid           : {tier_valid:,}  ({valid_pct:.1f}%)")
        print(f"    Invalid         : {tier_invalid:,}  ({invalid_pct:.1f}%)")
        print(f"    Converged       : {tier_converged:,}  ({converged_pct:.1f}%)")
        print(f"    Ceiling-limited : {tier_ceiling_limited:,}  ({ceiling_pct:.1f}%)")
        print(f"    Time            : {tier_elapsed:.1f}s")
        sys.stdout.flush()

        log_tiers.append({
            "condition": tier_name,
            "mechanics_pairs": len(mechanics),
            "n_cycles": n_cycles,
            "tier_total": tier_total,
            "tier_valid": tier_valid,
            "tier_invalid": tier_invalid,
            "tier_converged": tier_converged,
            "tier_ceiling_limited": tier_ceiling_limited,
            "valid_pct": round(valid_pct, 1),
            "converged_pct": round(converged_pct, 1),
            "ceiling_limited_pct": round(ceiling_pct, 1),
            "elapsed_s": round(tier_elapsed, 1),
        })

        grand_total += tier_total
        grand_valid += tier_valid
        grand_invalid += tier_invalid
        grand_converged += tier_converged
        grand_ceiling_limited += tier_ceiling_limited

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "prvc_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode": "PRVC",
        "dataset_type": "thinned",
        "generated_at": run_ts,
        "ibw_kg": IBW_KG,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "thinned_grid": THINNED_PARAMETER_GRID,
        "combos_per_point": combos_per_point,
        "adaptation_step_cmH2O": ADAPTATION_STEP_CMH2O_DEFAULT,
        "vt_tolerance_frac": VT_TOLERANCE_FRAC_DEFAULT,
        "grand_total": grand_total,
        "grand_valid": grand_valid,
        "grand_invalid": grand_invalid,
        "grand_converged": grand_converged,
        "grand_ceiling_limited": grand_ceiling_limited,
        "valid_pct": round(100 * grand_valid / grand_total, 1) if grand_total > 0 else 0,
        "converged_pct": round(100 * grand_converged / grand_total, 1) if grand_total > 0 else 0,
        "ceiling_limited_pct": round(100 * grand_ceiling_limited / grand_total, 1)
                                if grand_total > 0 else 0,
        "total_elapsed_s": round(run_elapsed, 1),
        "total_elapsed_min": round(run_elapsed / 60, 1),
        "tiers": log_tiers,
        "validity_thresholds": {
            "ppeak_max_cmH2O": PPEAK_MAX_CMHH2O,
            "vt_min_mL": VT_MIN_ML,
            "vt_max_mL": VT_MAX_ML,
        },
        "notes": [
            "COPD and Bronchospasm use 25 cycles/scenario for auto-PEEP and "
            "pressure-staircase steady state; all other tiers use 12 "
            "cycles/scenario",
            "adaptation_step_cmH2O and vt_tolerance_frac are fixed uniform "
            "constants, not swept -- see PRVC_PARAMETER_GRID.md",
            "Breath 1 in every scenario is a volume-controlled test breath "
            "(AutoFlow-style bootstrap); it is excluded from convergence "
            "tracking and always delivers ~vt_target by construction",
            "Ceiling-limited non-convergence is a deliberately retained, "
            "labeled outcome (see converged / ceiling_limited columns), not "
            "an invalidity condition on its own",
            "Seeds are deterministic; prvc_generator has no stochastic "
            "elements (purely mandatory mode), so identical params always "
            "reproduce identical waveforms regardless of seed value",
            "Full waveform arrays (including pressure_trajectory and "
            "delivered_vt_trajectory) are not persisted in the manifest -- "
            "regenerate via generate_breath_cycles(params, seed=seed) using "
            "the params implied by each row plus its recorded seed",
        ],
    }
    log_path = OUTPUT_DIR / "prvc_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # --- Final summary ------------------------------------------------
    valid_pct_grand = 100 * grand_valid / grand_total if grand_total > 0 else 0
    converged_pct_grand = 100 * grand_converged / grand_total if grand_total > 0 else 0
    ceiling_pct_grand = 100 * grand_ceiling_limited / grand_total if grand_total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  PRVC Thinned Dataset Generation Complete")
    print(f"  {'─' * 40}")
    print(f"  Total scenarios  : {grand_total:,}")
    print(f"  Valid            : {grand_valid:,}  ({valid_pct_grand:.1f}%)")
    print(f"  Invalid          : {grand_invalid:,}  ({100 - valid_pct_grand:.1f}%)")
    print(f"  Converged        : {grand_converged:,}  ({converged_pct_grand:.1f}%)")
    print(f"  Ceiling-limited  : {grand_ceiling_limited:,}  ({ceiling_pct_grand:.1f}%)")
    print(f"  Manifest         : {manifest_path}")
    print(f"  Log              : {log_path}")
    print(f"  Total time       : {run_elapsed:.1f}s  ({run_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    # Per-condition summary table
    print(f"\n  {'Condition':<16} {'Total':>8} {'Valid':>8} {'Valid%':>7} "
          f"{'Conv%':>7} {'Ceiling%':>9}")
    print(f"  {'─' * 65}")
    for t in log_tiers:
        print(f"  {t['condition']:<16} {t['tier_total']:>8,} {t['tier_valid']:>8,} "
              f"{t['valid_pct']:>6.1f}% {t['converged_pct']:>6.1f}% "
              f"{t['ceiling_limited_pct']:>8.1f}%")
    print()


if __name__ == "__main__":
    run()
