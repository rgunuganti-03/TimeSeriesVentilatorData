"""
generate_vcv_dataset_thinned.py
--------------------------------
Self-contained VCV dataset generator with the thinned parameter grid
built in. Sweeps only the kept parameter combinations — no intermediate
full dataset, no HDF5, no post-processing step.

Run from the project root:
    python generate_vcv_dataset_thinned.py

Output:
    data/exports/vcv/
        vcv_manifest_thinned.csv    — one row per scenario (valid + invalid)
        vcv_generation_log.json     — run summary: counts, timing, config

Thinned parameter grid (rationale documented below):

    Respiratory rate : 8, 16, 24, 30 bpm
        Adjacent 4-bpm steps produce minimal waveform shape change at
        fixed mechanics. Kept values span slow, moderate, and fast regimes.

    PEEP             : 0, 8, 16, 20 cmH2O
        PEEP shifts the pressure baseline vertically without changing
        elastic or resistive waveform shape. Kept values cover no PEEP,
        moderate, high, and maximum clinical ranges.

    Tidal volume     : 4, 6, 10 mL/kg IBW  (280, 420, 700 mL at 70 kg)
        4 = ultra-protective, 6 = ARDSNet standard, 10 = upper standard.
        8 mL/kg dropped — sits between 6 and 10 without a distinct
        clinical strategy.

    I:E ratio        : 1:1, 1:2, 1:3  (all kept)
        Each ratio represents a fundamentally different inspiratory time
        allocation. No intermediate values exist in the grid.

    Flow pattern     : square, decelerating  (both kept)
        Discrete waveform shapes — not a continuous parameter.

Expected combinations per mechanics point:
    4 RR × 4 PEEP × 3 TV × 3 IE × 2 flow = 288 (vs 1,008 in full grid)
    Reduction: 71.4% fewer scenarios per mechanics point
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
from generator.vcv_generator import generate_breath_cycles, IBW_KG
from generator.vcv_generator import _validate_params, _make_scenario_id


# ---------------------------------------------------------------------------
# Thinned parameter grid
# ---------------------------------------------------------------------------

THINNED_PARAMETER_GRID = {
    "tidal_volume_mL_per_kg": [4, 6, 10],
    "respiratory_rate":       [8, 16, 24, 30],
    "peep_cmH2O":             [0, 8, 16, 20],
    "ie_ratio":               [1.0, 0.5, 0.33],
    "flow_pattern":           ["square", "decelerating"],
}

# ---------------------------------------------------------------------------
# Condition tier definitions — identical to full generator
# ---------------------------------------------------------------------------

CONDITION_TIERS = [
    {
        "name":             "Normal",
        "compliance_range": (50, 100),
        "compliance_step":  10,
        "resistance_range": (2, 5),
        "resistance_step":  1,
    },
    {
        "name":             "Mild ARDS",
        "compliance_range": (30, 50),
        "compliance_step":  5,
        "resistance_range": (3, 6),
        "resistance_step":  1,
    },
    {
        "name":             "Moderate ARDS",
        "compliance_range": (20, 30),
        "compliance_step":  5,
        "resistance_range": (4, 8),
        "resistance_step":  2,
    },
    {
        "name":             "Severe ARDS",
        "compliance_range": (10, 20),
        "compliance_step":  5,
        "resistance_range": (5, 10),
        "resistance_step":  2,
    },
    {
        "name":             "COPD",
        "compliance_range": (40, 80),
        "compliance_step":  10,
        "resistance_range": (10, 20),
        "resistance_step":  2,
    },
    {
        "name":             "Bronchospasm",
        "compliance_range": (40, 70),
        "compliance_step":  10,
        "resistance_range": (15, 30),
        "resistance_step":  5,
    },
    {
        "name":             "Pneumonia",
        "compliance_range": (25, 45),
        "compliance_step":  5,
        "resistance_range": (4, 8),
        "resistance_step":  2,
    },
]

N_CYCLES   = 10
OUTPUT_DIR = Path("data/exports/vcv")


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
    compliance_mL_per_cmH2O: float,
    resistance_cmH2O_L_s:    float,
    n_cycles:                int = 10,
) -> list:
    """
    Sweep the thinned parameter grid for one condition + mechanics pair.
    Returns a list of scenario dicts — same structure as generate_dataset()
    in vcv_generator.py.
    """
    scenarios = []

    keys   = ["tidal_volume_mL_per_kg", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "flow_pattern"]
    values = [THINNED_PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        vt_per_kg, rr, peep, ie, pattern = combo

        vt_mL = vt_per_kg * IBW_KG

        params = {
            "respiratory_rate":        rr,
            "tidal_volume_mL":         vt_mL,
            "compliance_mL_per_cmH2O": compliance_mL_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "flow_pattern":            pattern,
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
            "delivered_vt_mL": result["delivered_vt_mL"],
            "minute_vent_L":   result["minute_vent_L"],
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

    manifest_rows = []
    log_tiers     = []

    grand_total   = 0
    grand_valid   = 0
    grand_invalid = 0

    combos_per_point = (
        len(THINNED_PARAMETER_GRID["tidal_volume_mL_per_kg"])
        * len(THINNED_PARAMETER_GRID["respiratory_rate"])
        * len(THINNED_PARAMETER_GRID["peep_cmH2O"])
        * len(THINNED_PARAMETER_GRID["ie_ratio"])
        * len(THINNED_PARAMETER_GRID["flow_pattern"])
    )

    print("=" * 70)
    print("  VCV Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Cycles     : {N_CYCLES} per scenario")
    print(f"  Grid size  : {combos_per_point} combinations per mechanics point")
    print("=" * 70)

    for tier in CONDITION_TIERS:
        tier_name  = tier["name"]
        mechanics  = _mechanics_grid(tier)
        tier_start = time.perf_counter()

        tier_total   = 0
        tier_valid   = 0
        tier_invalid = 0

        print(f"\n  [{tier_name}]")
        print(f"    Mechanics pairs : {len(mechanics)}")

        for C, R in mechanics:
            scenarios = _generate_thinned_dataset(
                condition_name           = tier_name,
                compliance_mL_per_cmH2O = C,
                resistance_cmH2O_L_s    = R,
                n_cycles                 = N_CYCLES,
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
                    "scenario_id":             s["scenario_id"],
                    "condition":               s["condition"],
                    "generated_at":            s["generated_at"],
                    "is_valid":                s["is_valid"],
                    "invalid_reason":          s["invalid_reason"],
                    "compliance_mL_per_cmH2O": p["compliance_mL_per_cmH2O"],
                    "resistance_cmH2O_L_s":    p["resistance_cmH2O_L_s"],
                    "respiratory_rate":        p["respiratory_rate"],
                    "tidal_volume_mL":         p["tidal_volume_mL"],
                    "tidal_volume_mL_per_kg":  p["tidal_volume_mL"] / IBW_KG,
                    "ie_ratio":                p["ie_ratio"],
                    "peep_cmH2O":              p["peep_cmH2O"],
                    "flow_pattern":            p["flow_pattern"],
                    "ppeak_cmH2O":             m.get("ppeak_cmH2O",     ""),
                    "pplat_cmH2O":             m.get("pplat_cmH2O",     ""),
                    "driving_p_cmH2O":         m.get("driving_p_cmH2O", ""),
                    "mean_paw_cmH2O":          m.get("mean_paw_cmH2O",  ""),
                    "auto_peep_cmH2O":         m.get("auto_peep_cmH2O", ""),
                    "delivered_vt_mL":         m.get("delivered_vt_mL", ""),
                    "minute_vent_L":           m.get("minute_vent_L",   ""),
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        tier_log = {
            "condition":       tier_name,
            "mechanics_pairs": len(mechanics),
            "total":           tier_total,
            "valid":           tier_valid,
            "invalid":         tier_invalid,
            "valid_pct":       round(100 * tier_valid / tier_total, 1)
                               if tier_total > 0 else 0,
            "elapsed_s":       round(tier_elapsed, 1),
        }
        log_tiers.append(tier_log)

        print(f"    Total    : {tier_total:,}")
        print(f"    Valid    : {tier_valid:,}  "
              f"({100*tier_valid/tier_total:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  "
              f"({100*tier_invalid/tier_total:.1f}%)")
        print(f"    Time     : {tier_elapsed:.1f}s")

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "vcv_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":              "VCV",
        "dataset_type":      "thinned",
        "generated_at":      run_ts,
        "n_cycles":          N_CYCLES,
        "ibw_kg":            IBW_KG,
        "output_dir":        str(OUTPUT_DIR.resolve()),
        "thinned_grid":      THINNED_PARAMETER_GRID,
        "combos_per_point":  combos_per_point,
        "grand_total":       grand_total,
        "grand_valid":       grand_valid,
        "grand_invalid":     grand_invalid,
        "valid_pct":         round(100 * grand_valid / grand_total, 1)
                             if grand_total > 0 else 0,
        "total_elapsed_s":   round(run_elapsed, 1),
        "tiers":             log_tiers,
    }
    log_path = OUTPUT_DIR / "vcv_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # --- Final summary ----------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  VCV Thinned Dataset Generation Complete")
    print(f"  {'─' * 40}")
    print(f"  Total scenarios  : {grand_total:,}")
    print(f"  Valid            : {grand_valid:,}  "
          f"({100*grand_valid/grand_total:.1f}%)")
    print(f"  Invalid          : {grand_invalid:,}  "
          f"({100*grand_invalid/grand_total:.1f}%)")
    print(f"  Manifest         : {manifest_path}")
    print(f"  Log              : {log_path}")
    print(f"  Total time       : {run_elapsed:.1f}s  "
          f"({run_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
