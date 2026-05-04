"""
generate_pcv_dataset_thinned.py
--------------------------------
Self-contained PCV dataset generator with the thinned parameter grid
built in. Sweeps only the kept parameter combinations — no intermediate
full dataset, no HDF5, no post-processing step.

Run from the project root:
    python generate_pcv_dataset_thinned.py

For an overnight run:
    nohup python -u generate_pcv_dataset_thinned.py > pcv_thinned.log 2>&1 &

Monitor progress:
    tail -f pcv_thinned.log

Output:
    data/exports/pcv/
        pcv_manifest_thinned.csv    — one row per scenario (valid + invalid)
        pcv_generation_log.json     — run summary: counts, timing, config

Thinned parameter grid (rationale documented below):

    Respiratory rate : 8, 16, 24, 30 bpm
        Adjacent 4-bpm steps produce minimal waveform shape change at
        fixed mechanics. In PCV, RR also affects fill fraction through
        inspiratory time — kept values span distinct fill fraction bands.

    PEEP             : 0, 8, 16, 20 cmH2O
        PIP = PEEP + insp_pressure, so adjacent PEEP values at fixed
        insp_pressure produce nearly identical waveform shapes with only
        a vertical offset. Kept values cover no PEEP, moderate, high,
        and maximum clinical ranges.

    Inspiratory pressure : 5, 15, 25, 35 cmH2O above PEEP
        Directly scales delivered tidal volume (VT = P * C * fill_frac).
        Each kept value represents a distinct clinical support level:
        5 (minimal), 15 (moderate), 25 (high), 35 (maximum).
        Dropped 10, 20, 30 as intermediate values.

    I:E ratio        : 1:1, 1:2, 1:3  (all kept)
        Each ratio changes inspiratory time and fill fraction in a
        meaningful non-linear way. All three must be kept.

    Rise time        : 0.0, 0.2, 0.4 seconds
        0.0 = square wave step (maximum initial flow)
        0.2 = moderate ramp
        0.4 = slowest clinical ramp
        Dropped 0.1 — sits between 0.0 and 0.2 without a distinct
        clinical strategy.

Expected combinations per mechanics point:
    4 RR × 4 PEEP × 4 P × 3 IE × 3 RT = 576 (vs 3,528 in full grid)
    Reduction: 83.7% fewer scenarios per mechanics point

Runtime note:
    PCV uses scipy.integrate.solve_ivp — slower than VCV.
    Estimated runtime: 2-3 hours (vs 8-10 hours for full grid).
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
from generator.pcv_generator import generate_breath_cycles, IBW_KG
from generator.pcv_generator import _make_scenario_id


# ---------------------------------------------------------------------------
# Thinned parameter grid
# ---------------------------------------------------------------------------

THINNED_PARAMETER_GRID = {
    "insp_pressure_cmH2O": [5, 15, 25, 35],
    "respiratory_rate":    [8, 16, 24, 30],
    "peep_cmH2O":          [0, 8, 16, 20],
    "ie_ratio":            [1.0, 0.5, 0.33],
    "rise_time_s":         [0.0, 0.2, 0.4],
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
OUTPUT_DIR = Path("data/exports/pcv")


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
    Sweep the thinned PCV parameter grid for one condition + mechanics pair.
    Returns a list of scenario dicts.
    """
    scenarios = []

    keys   = ["insp_pressure_cmH2O", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "rise_time_s"]
    values = [THINNED_PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        p_insp, rr, peep, ie, t_rise = combo

        params = {
            "respiratory_rate":        rr,
            "insp_pressure_cmH2O":     p_insp,
            "compliance_mL_per_cmH2O": compliance_mL_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "rise_time_s":             t_rise,
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
            "delivered_vt_mL":     result["delivered_vt_mL"],
            "driving_p_cmH2O":     result["driving_p_cmH2O"],
            "mean_paw_cmH2O":      result["mean_paw_cmH2O"],
            "auto_peep_cmH2O":     result["auto_peep_cmH2O"],
            "fill_fraction":       result["fill_fraction"],
            "minute_vent_L":       result["minute_vent_L"],
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

    manifest_rows = []
    log_tiers     = []

    grand_total   = 0
    grand_valid   = 0
    grand_invalid = 0

    combos_per_point = (
        len(THINNED_PARAMETER_GRID["insp_pressure_cmH2O"])
        * len(THINNED_PARAMETER_GRID["respiratory_rate"])
        * len(THINNED_PARAMETER_GRID["peep_cmH2O"])
        * len(THINNED_PARAMETER_GRID["ie_ratio"])
        * len(THINNED_PARAMETER_GRID["rise_time_s"])
    )

    print("=" * 70)
    print("  PCV Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Cycles     : {N_CYCLES} per scenario")
    print(f"  Grid size  : {combos_per_point} combinations per mechanics point")
    print("  Note       : ODE solver — estimated 2-3 hours total runtime")
    print("  Tip        : Run with nohup -u and tail -f pcv_thinned.log")
    print("=" * 70)
    sys.stdout.flush()

    for tier in CONDITION_TIERS:
        tier_name  = tier["name"]
        mechanics  = _mechanics_grid(tier)
        tier_start = time.perf_counter()

        tier_total   = 0
        tier_valid   = 0
        tier_invalid = 0

        print(f"\n  [{tier_name}]")
        print(f"    Mechanics pairs : {len(mechanics)}")
        sys.stdout.flush()

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
                    "insp_pressure_cmH2O":     p["insp_pressure_cmH2O"],
                    "ie_ratio":                p["ie_ratio"],
                    "peep_cmH2O":              p["peep_cmH2O"],
                    "rise_time_s":             p["rise_time_s"],
                    "ppeak_cmH2O":             m.get("ppeak_cmH2O",         ""),
                    "delivered_vt_mL":         m.get("delivered_vt_mL",     ""),
                    "driving_p_cmH2O":         m.get("driving_p_cmH2O",     ""),
                    "mean_paw_cmH2O":          m.get("mean_paw_cmH2O",      ""),
                    "auto_peep_cmH2O":         m.get("auto_peep_cmH2O",     ""),
                    "fill_fraction":           m.get("fill_fraction",        ""),
                    "minute_vent_L":           m.get("minute_vent_L",        ""),
                    "time_to_peak_flow_s":     m.get("time_to_peak_flow_s",  ""),
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        # ETA based on measured throughput
        elapsed_so_far = time.perf_counter() - run_start
        rate_per_s     = grand_total / elapsed_so_far if elapsed_so_far > 0 else 1
        remaining      = sum(
            len(_mechanics_grid(t)) * combos_per_point
            for t in CONDITION_TIERS[CONDITION_TIERS.index(tier) + 1:]
        )
        eta_s = remaining / rate_per_s if rate_per_s > 0 else 0

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
        if eta_s > 0:
            print(f"    ETA      : ~{eta_s/60:.0f} min remaining")
        sys.stdout.flush()

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "pcv_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":             "PCV",
        "dataset_type":     "thinned",
        "generated_at":     run_ts,
        "n_cycles":         N_CYCLES,
        "ibw_kg":           IBW_KG,
        "output_dir":       str(OUTPUT_DIR.resolve()),
        "thinned_grid":     THINNED_PARAMETER_GRID,
        "combos_per_point": combos_per_point,
        "grand_total":      grand_total,
        "grand_valid":      grand_valid,
        "grand_invalid":    grand_invalid,
        "valid_pct":        round(100 * grand_valid / grand_total, 1)
                            if grand_total > 0 else 0,
        "total_elapsed_s":  round(run_elapsed, 1),
        "tiers":            log_tiers,
    }
    log_path = OUTPUT_DIR / "pcv_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  PCV Thinned Dataset Generation Complete")
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
    sys.stdout.flush()


if __name__ == "__main__":
    run()
