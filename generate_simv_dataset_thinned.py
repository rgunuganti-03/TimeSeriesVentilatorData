"""
generate_simv_dataset_thinned.py
-----------------------------------
Self-contained SIMV dataset generator with the thinned parameter grid
built in. Sweeps only the kept parameter combinations -- no intermediate
full dataset, no HDF5, no post-processing step. Metrics only, no waveform
arrays in the manifest (regeneratable on demand via
generate_breath_cycles(params, seed=seed) -- same "no waveform storage"
convention as generate_psv_dataset_thinned.py / generate_prvc_dataset_thinned.py,
since SIMV, like PSV, has stochastic patient-effort draws and needs the
deterministic seed for reproducibility rather than storing every array).

Run from the project root:
    python generate_simv_dataset_thinned.py

For an overnight run (SIMV's per-scenario cost is higher than its sibling
engines -- see runtime note below):
    nohup python -u generate_simv_dataset_thinned.py > simv_thinned.log 2>&1 &

Monitor progress:
    tail -f simv_thinned.log

Output:
    data/exports/simv/
        simv_manifest_thinned.csv   -- one row per scenario (valid + invalid)
        simv_generation_log.json    -- run summary: counts, timing, config

Why SIMV's thinning has an extra axis its siblings don't
-----------------------------------------------------------
Every other thinned script sweeps one control regime. SIMV sweeps two
(mandatory_mode = VC or PC) *and* the spontaneous-breath / patient-effort
dimensions PSV needs, *and* its own mode-defining synchronization-window
parameter -- three axes of complexity stacked on top of each other. Left
unthinned this explodes combinatorially (the full PARAMETER_GRID in
generator/simv_generator.py is intentionally left at full resolution for
documentation/completeness, not for direct sweeping). The thinning below
is therefore more aggressive on the dimensions with weak sibling precedent
for "keep everything" (effort_rate_per_min, pmus_cv, effort_duration_s,
ie_ratio-for-this-file-only) while preserving full sibling-precedent
treatment for the dimensions with a documented reason to keep multiple
values (f_window, pressure_support_cmH2O, flow_cycle_threshold).

Thinned parameter grid (rationale documented below)
-----------------------------------------------------

    Shared (both mandatory sub-modes):

    Respiratory rate (mandatory backup rate) : 4, 12 bpm
        SIMV's own PARAMETER_GRID already scopes this to the clinically
        relevant weaning range (4-12 bpm) rather than vcv/pcv's full 8-30
        bpm CMV range -- see SIMV_CONTROL_LOOP.md / SIMV grounding doc.
        Thinned further to the two clinical bookends: near-extubation
        endpoint (4) and typical initiation (12). The 6/8/10 midpoints
        interpolate smoothly between these two regimes in T_mand and
        window timing without revealing a distinct clinical strategy.

    PEEP : 8 cmH2O (single value)
        PEEP shifts the pressure baseline vertically without changing
        elastic or resistive waveform shape -- the same rationale
        vcv/pcv/prvc use to justify their own PEEP thinning, taken one
        step further here. Measuring actual per-scenario cost put the
        full two-bookend-value grid at roughly 17 hours end-to-end (see
        runtime note below) -- PEEP was the dimension with the weakest
        "shape changes, not just shifts" case for keeping multiple values,
        so it absorbed the cut needed to bring the run back to the
        sibling scripts' 8-10 hour overnight precedent. A single
        mid-clinical value (8 cmH2O, the low end of typical ARDS/COPD PEEP
        titration) is used rather than a bookend so the kept value isn't
        itself an edge case.

    I:E ratio : 1:1, 1:2, 1:3 (all kept)
        Matches vcv/pcv/prvc precedent exactly -- each ratio represents a
        fundamentally different mandatory-breath inspiratory time
        allocation with no redundant interpolation between the three.

    Rise time : 0.1 s (single value)
        Matches psv_generator's own thinned-script precedent exactly:
        instantaneous (0.0) and slow (0.4) contribute minimal additional
        waveform diversity at dataset scale; the full range remains
        available via generator/simv_generator.py's PARAMETER_GRID for
        targeted runs.

    Synchronization window (f_window) : 0.15, 0.25, 0.30
        SIMV's mode-defining parameter -- thinned least aggressively of
        any dimension in this file (same treatment prvc_generator's own
        thinned script gives pressure_ceiling, its equivalent signature
        parameter). Per the project's literature-grounding pass, no
        single vendor value exists (Servo frames the window as "first 90%
        of breath cycle time," Drager as "~20% of expiratory time");
        keeping three points spanning the recommended 0.15-0.30 tunable
        range preserves that vendor-variation signal rather than
        collapsing to one default.

    Pressure support (spontaneous breaths) : 5, 12, 20 cmH2O
        Identical values and rationale to psv_generator's own thinned
        script: weaning (5), standard support (12), high support (20).
        Intermediate 8/16 produce in-between waveforms with no distinct
        clinical strategy.

    Flow-cycle threshold (spontaneous breaths) : 0.25, 0.40, 0.65
        All three kept -- same "cannot be thinned without losing entire
        waveform-morphology regimes" reasoning psv_generator's thinned
        script applies to its own FCT dimension, now using this project's
        literature-refined SIMV defaults (restrictive mid-range ~0.25-0.40,
        obstructive high ~0.65) rather than PSV's original ~0.10 low
        anchor, which the grounding doc found unrepresentative for ARDS.

    Trigger threshold : 1.5 cmH2O (single value)
        Matches psv_generator's own thinned-script precedent and
        rationale exactly: threshold mainly gates ineffective triggering
        via the auto-PEEP interaction, already covered by varying
        resistance/condition across the mechanics grid.

    Pmus peak : 5, 20 cmH2O
        Weak vs. strong effort bookends -- cut from three points to two
        (vs. psv_generator's own three-point thinning) to offset the
        extra mandatory_mode multiplier this file carries that psv's
        thinned script doesn't.

    Effort rate, effort duration, Pmus CV : single representative value
        each (16 breaths/min, 0.7 s, 0.20)
        These are the dimensions with the weakest "must keep multiple
        values" case even in psv_generator's own thinned script, and the
        first ones cut further here to keep the combined
        (shared x mode-specific) combinatorial count in the same rough
        order of magnitude as the sibling thinned scripts despite SIMV's
        extra mandatory_mode axis.

    Mandatory-mode-specific:

    Tidal volume (SIMV-VC) : 4, 6, 10 mL/kg IBW
        Identical values and rationale to vcv_generator's / prvc_generator's
        own thinned scripts: 4 = ultra-protective, 6 = ARDSNet standard,
        10 = upper standard; 8 dropped as a redundant interpolation.

    Flow pattern (SIMV-VC) : square, decelerating (both kept)
        Discrete waveform shapes, not a continuous parameter -- matches
        vcv_generator precedent.

    Inspiratory pressure (SIMV-PC) : 10, 20, 35 cmH2O above PEEP
        Low/mid/clinical-ceiling bookends, cut from the full 6-value grid
        to 3 -- the ceiling value (35) is kept explicitly since it
        coincides with INSP_PRESSURE_MAX_CMHH2O, the validity-filter
        threshold, so the thinned sweep still samples right at the
        boundary the way vcv/pcv's thinned scripts keep their own extremes.

Expected combinations per mechanics point
--------------------------------------------
    Shared   : 2 RR x 1 PEEP x 3 IE x 1 rise x 3 f_window x 3 PS x 3 FCT
               x 1 trigger x 2 pmus x 1 rate x 1 dur x 1 cv = 324
    SIMV-VC  : 324 x (3 VT x 2 flow_pattern)          =  1,944
    SIMV-PC  : 324 x (3 insp_pressure)                =    972
    Combined (VC + PC) per mechanics point            =  2,916

n_cycles per condition tier, and measured runtime
------------------------------------------------------
SIMV's mandatory cycle time (T_mand = 60/RR, RR in the 4-12 bpm range used
here) spans 5-15 s -- substantially longer than vcv/pcv/psv/prvc's typical
2-7.5 s mandatory cycle at their own thinned RR ranges, and each cycle
additionally simulates however many spontaneous breaths the effort-rate
schedule interleaves. To keep total simulated auto-PEEP-relevant time (and
per-scenario runtime) comparable to the sibling scripts rather than several
times longer, n_cycles is set lower than psv/prvc's 12/25 convention: 6
mandatory cycles for most conditions, 10 for COPD/Bronchospasm (still the
two tiers needing extra cycles to reach auto-PEEP steady state).

Measured directly (60-scenario timed sample, mixed VC/PC/RR): ~82 ms/
scenario at n_cycles=6. Across all 7 tiers' full mechanics grids (~126
total (compliance, resistance) pairs) at 2,916 combos/point, that's
roughly 367,000 total scenarios and ~8-9 hours end-to-end -- in line with
the vcv/pcv full-dataset precedent (8-10 hours) despite SIMV's extra
mandatory_mode axis, which is the reason PEEP above was thinned to a
single value rather than the two-bookend treatment an earlier draft of
this grid used (that version measured out closer to 17 hours).
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
from generator.simv_generator import generate_breath_cycles, IBW_KG
from generator.simv_generator import _make_scenario_id


# ---------------------------------------------------------------------------
# Thinned parameter grid
# ---------------------------------------------------------------------------

THINNED_SHARED_GRID = {
    "respiratory_rate":         [4.0, 12.0],
    "peep_cmH2O":                [8.0],
    "ie_ratio":                  [1.0, 0.5, 0.33],
    "rise_time_s":               [0.1],
    "f_window":                  [0.15, 0.25, 0.30],
    "pressure_support_cmH2O":    [5.0, 12.0, 20.0],
    "flow_cycle_threshold":      [0.25, 0.40, 0.65],
    "trigger_threshold_cmH2O":  [1.5],
    "pmus_peak_cmH2O":           [5.0, 20.0],
    "effort_rate_per_min":       [16.0],
    "effort_duration_s":         [0.7],
    "pmus_cv":                   [0.20],
}

THINNED_VC_GRID = {
    "tidal_volume_ml_per_kg": [4, 6, 10],
    "flow_pattern":           ["square", "decelerating"],
}

THINNED_PC_GRID = {
    "insp_pressure_cmH2O": [10.0, 20.0, 35.0],
}

# ---------------------------------------------------------------------------
# Condition tier definitions -- mechanics grid identical to the shared
# CONDITION_TIERS used across vcv/pcv/psv/prvc's own thinned scripts
# (corrected resistance floors); n_cycles adapted per the runtime note above.
# ---------------------------------------------------------------------------

CONDITION_TIERS = [
    {"name": "Normal",         "compliance_range": (60, 100), "compliance_step": 10,
     "resistance_range": (8, 12),   "resistance_step": 1, "n_cycles": 6},
    {"name": "Mild ARDS",      "compliance_range": (40, 55),  "compliance_step": 5,
     "resistance_range": (10, 14),  "resistance_step": 2, "n_cycles": 6},
    {"name": "Moderate ARDS",  "compliance_range": (28, 40),  "compliance_step": 4,
     "resistance_range": (12, 16),  "resistance_step": 2, "n_cycles": 6},
    {"name": "Severe ARDS",    "compliance_range": (15, 28),  "compliance_step": 4,
     "resistance_range": (14, 20),  "resistance_step": 3, "n_cycles": 6},
    {"name": "COPD",           "compliance_range": (80, 150), "compliance_step": 20,
     "resistance_range": (18, 35),  "resistance_step": 5, "n_cycles": 10},
    {"name": "Bronchospasm",   "compliance_range": (60, 90),  "compliance_step": 10,
     "resistance_range": (25, 50),  "resistance_step": 5, "n_cycles": 10},
    {"name": "Pneumonia",      "compliance_range": (40, 65),  "compliance_step": 5,
     "resistance_range": (10, 16),  "resistance_step": 2, "n_cycles": 6},
]

OUTPUT_DIR = Path("data/exports/simv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mechanics_grid(tier: dict) -> list:
    """Build the (C, R) mechanics grid for one condition tier."""
    c_start, c_stop = tier["compliance_range"]
    r_start, r_stop = tier["resistance_range"]
    compliances = np.arange(
        c_start, c_stop + tier["compliance_step"], tier["compliance_step"]
    ).tolist()
    resistances = np.arange(
        r_start, r_stop + tier["resistance_step"], tier["resistance_step"]
    ).tolist()
    return [(float(c), float(r)) for c in compliances for r in resistances]


def _make_deterministic_seed(condition: str, C: float, R: float, mode: str,
                              shared_combo: tuple, mode_combo: tuple) -> int:
    """
    Compute a deterministic seed from scenario parameters so that the same
    scenario always produces the same stochastic Pmus/effort draws --
    same pattern as generate_psv_dataset_thinned.py / prvc's thinned script.
    """
    key = (condition, round(C, 1), round(R, 1), mode,
           tuple(round(v, 3) if isinstance(v, float) else v for v in shared_combo),
           tuple(round(v, 3) if isinstance(v, float) else v for v in mode_combo))
    return int(abs(hash(key))) % (2 ** 31)


def _generate_thinned_dataset(condition_name: str, compliance_ml_per_cmH2O: float,
                               resistance_cmH2O_L_s: float, n_cycles: int) -> list:
    """
    Sweep the thinned SIMV grid for one condition + mechanics pair, across
    both mandatory sub-modes. Returns a list of scenario dicts -- metrics
    only, no waveform arrays (regeneratable on demand via
    generate_breath_cycles(params, seed=seed)).
    """
    scenarios = []

    shared_keys = list(THINNED_SHARED_GRID.keys())
    shared_values = [THINNED_SHARED_GRID[k] for k in shared_keys]

    for mode, mode_grid in (("VC", THINNED_VC_GRID), ("PC", THINNED_PC_GRID)):
        mode_keys = list(mode_grid.keys())
        mode_values = [mode_grid[k] for k in mode_keys]

        for shared_combo in itertools.product(*shared_values):
            for mode_combo in itertools.product(*mode_values):
                params = dict(zip(shared_keys, shared_combo))
                params.update(dict(zip(mode_keys, mode_combo)))
                params["mandatory_mode"] = mode
                params["condition"] = condition_name
                params["compliance_ml_per_cmH2O"] = compliance_ml_per_cmH2O
                params["resistance_cmH2O_L_s"] = resistance_cmH2O_L_s
                if mode == "VC":
                    params["tidal_volume_ml"] = (
                        params.pop("tidal_volume_ml_per_kg") * IBW_KG
                    )

                scenario_id = _make_scenario_id(condition_name, params)
                seed = _make_deterministic_seed(
                    condition_name, compliance_ml_per_cmH2O, resistance_cmH2O_L_s,
                    mode, shared_combo, mode_combo,
                )

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
                    "n_compartments":                  result["n_compartments"],
                    "n_mandatory_breaths":              result["n_mandatory_breaths"],
                    "n_spontaneous_breaths":            result["n_spontaneous_breaths"],
                    "mandatory_synchronized_fraction":  result["mandatory_synchronized_fraction"],
                    "mandatory_delivered_vt_ml":        result["mandatory_delivered_vt_ml"],
                    "spontaneous_delivered_vt_ml":      result["spontaneous_delivered_vt_ml"],
                    "ppeak_cmH2O":                       result["ppeak_cmH2O"],
                    "driving_p_cmH2O":                   result["driving_p_cmH2O"],
                    "mean_paw_cmH2O":                    result["mean_paw_cmH2O"],
                    "auto_peep_cmH2O":                   result["auto_peep_cmH2O"],
                    "minute_vent_l":                     result["minute_vent_l"],
                    "ineffective_trigger_fraction":      result["ineffective_trigger_fraction"],
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
    run_ts    = datetime.now(timezone.utc).isoformat()

    manifest_rows = []
    log_tiers     = []

    grand_total   = 0
    grand_valid   = 0
    grand_invalid = 0

    shared_combos_per_point = 1
    for v in THINNED_SHARED_GRID.values():
        shared_combos_per_point *= len(v)
    vc_combos_per_point = shared_combos_per_point
    for v in THINNED_VC_GRID.values():
        vc_combos_per_point *= len(v)
    pc_combos_per_point = shared_combos_per_point
    for v in THINNED_PC_GRID.values():
        pc_combos_per_point *= len(v)
    combos_per_point = vc_combos_per_point + pc_combos_per_point

    print("=" * 70)
    print("  SIMV Thinned Dataset Generation")
    print(f"  Started    : {run_ts}")
    print(f"  Output dir : {OUTPUT_DIR.resolve()}")
    print(f"  Grid size  : {combos_per_point:,} combinations per mechanics "
          f"point (VC {vc_combos_per_point:,} + PC {pc_combos_per_point:,})")
    print("=" * 70)
    sys.stdout.flush()

    for tier in CONDITION_TIERS:
        tier_name  = tier["name"]
        n_cycles   = tier["n_cycles"]
        mechanics  = _mechanics_grid(tier)
        tier_start = time.perf_counter()

        tier_total   = 0
        tier_valid   = 0
        tier_invalid = 0
        tier_vc      = 0
        tier_pc      = 0
        sync_fracs   = []
        n_spont_list = []

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
                    if m.get("mandatory_synchronized_fraction") is not None:
                        sync_fracs.append(m["mandatory_synchronized_fraction"])
                    if m.get("n_spontaneous_breaths") is not None:
                        n_spont_list.append(m["n_spontaneous_breaths"])
                else:
                    tier_invalid += 1

                if p.get("mandatory_mode") == "VC":
                    tier_vc += 1
                else:
                    tier_pc += 1

                manifest_rows.append({
                    "scenario_id":                      s["scenario_id"],
                    "condition":                         s["condition"],
                    "generated_at":                      s["generated_at"],
                    "is_valid":                          s["is_valid"],
                    "invalid_reason":                    s["invalid_reason"],
                    "seed":                               s.get("seed", ""),
                    "mandatory_mode":                    p.get("mandatory_mode", ""),
                    "compliance_ml_per_cmH2O":           p.get("compliance_ml_per_cmH2O", ""),
                    "resistance_cmH2O_L_s":              p.get("resistance_cmH2O_L_s", ""),
                    "respiratory_rate":                  p.get("respiratory_rate", ""),
                    "peep_cmH2O":                         p.get("peep_cmH2O", ""),
                    "ie_ratio":                           p.get("ie_ratio", ""),
                    "rise_time_s":                        p.get("rise_time_s", ""),
                    "f_window":                           p.get("f_window", ""),
                    "tidal_volume_ml":                   p.get("tidal_volume_ml", ""),
                    "flow_pattern":                       p.get("flow_pattern", ""),
                    "insp_pressure_cmH2O":               p.get("insp_pressure_cmH2O", ""),
                    "pressure_support_cmH2O":            p.get("pressure_support_cmH2O", ""),
                    "flow_cycle_threshold":               p.get("flow_cycle_threshold", ""),
                    "trigger_threshold_cmH2O":            p.get("trigger_threshold_cmH2O", ""),
                    "pmus_peak_cmH2O":                    p.get("pmus_peak_cmH2O", ""),
                    "effort_rate_per_min":                p.get("effort_rate_per_min", ""),
                    "effort_duration_s":                  p.get("effort_duration_s", ""),
                    "pmus_cv":                             p.get("pmus_cv", ""),
                    "n_compartments":                     m.get("n_compartments", ""),
                    "n_mandatory_breaths":                m.get("n_mandatory_breaths", ""),
                    "n_spontaneous_breaths":              m.get("n_spontaneous_breaths", ""),
                    "mandatory_synchronized_fraction":    m.get("mandatory_synchronized_fraction", ""),
                    "mandatory_delivered_vt_ml":          m.get("mandatory_delivered_vt_ml", ""),
                    "spontaneous_delivered_vt_ml":        m.get("spontaneous_delivered_vt_ml", ""),
                    "ppeak_cmH2O":                         m.get("ppeak_cmH2O", ""),
                    "driving_p_cmH2O":                     m.get("driving_p_cmH2O", ""),
                    "mean_paw_cmH2O":                      m.get("mean_paw_cmH2O", ""),
                    "auto_peep_cmH2O":                     m.get("auto_peep_cmH2O", ""),
                    "minute_vent_l":                       m.get("minute_vent_l", ""),
                    "ineffective_trigger_fraction":        m.get("ineffective_trigger_fraction", ""),
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        valid_pct   = 100 * tier_valid   / tier_total if tier_total > 0 else 0
        invalid_pct = 100 * tier_invalid / tier_total if tier_total > 0 else 0
        mean_sync   = float(np.mean(sync_fracs)) if sync_fracs else 0.0
        mean_spont  = float(np.mean(n_spont_list)) if n_spont_list else 0.0

        print(f"    Total    : {tier_total:,}  (VC {tier_vc:,} / PC {tier_pc:,})")
        print(f"    Valid    : {tier_valid:,}  ({valid_pct:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({invalid_pct:.1f}%)")
        print(f"    Mean synchronized fraction : {mean_sync:.2f}")
        print(f"    Mean spontaneous breaths/scenario : {mean_spont:.1f}")
        print(f"    Time     : {tier_elapsed:.1f}s")
        elapsed_so_far = time.perf_counter() - run_start
        avg_per_tier = elapsed_so_far / (CONDITION_TIERS.index(tier) + 1)
        remaining_tiers = len(CONDITION_TIERS) - (CONDITION_TIERS.index(tier) + 1)
        eta_s = avg_per_tier * remaining_tiers
        print(f"    ETA remaining : {eta_s / 60:.1f} min "
              f"({remaining_tiers} tiers left)")
        sys.stdout.flush()

        log_tiers.append({
            "condition":               tier_name,
            "mechanics_pairs":         len(mechanics),
            "n_cycles":                n_cycles,
            "tier_total":              tier_total,
            "tier_valid":              tier_valid,
            "tier_invalid":            tier_invalid,
            "tier_vc":                 tier_vc,
            "tier_pc":                 tier_pc,
            "mean_synchronized_fraction": round(mean_sync, 3),
            "mean_spontaneous_breaths":   round(mean_spont, 2),
            "valid_pct":               round(valid_pct, 1),
            "elapsed_s":               round(tier_elapsed, 1),
        })

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "simv_manifest_thinned.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                "SIMV",
        "dataset_type":        "thinned",
        "generated_at":        run_ts,
        "ibw_kg":              IBW_KG,
        "output_dir":          str(OUTPUT_DIR.resolve()),
        "thinned_shared_grid": THINNED_SHARED_GRID,
        "thinned_vc_grid":     THINNED_VC_GRID,
        "thinned_pc_grid":     THINNED_PC_GRID,
        "vc_combos_per_point": vc_combos_per_point,
        "pc_combos_per_point": pc_combos_per_point,
        "combos_per_point":    combos_per_point,
        "grand_total":         grand_total,
        "grand_valid":         grand_valid,
        "grand_invalid":       grand_invalid,
        "valid_pct":           round(100 * grand_valid / grand_total, 1)
                               if grand_total > 0 else 0,
        "total_elapsed_s":     round(run_elapsed, 1),
        "total_elapsed_min":   round(run_elapsed / 60, 1),
        "tiers":               log_tiers,
        "notes": [
            "COPD and Bronchospasm use 10 mandatory cycles/scenario for "
            "auto-PEEP steady state; all others use 6",
            "n_cycles counts mandatory macro-cycles, not total breaths -- "
            "spontaneous breaths interleave per the synchronization window "
            "and are not fixed in advance (see n_spontaneous_breaths)",
            "Seeds are deterministic: same params always produce the same "
            "stochastic patient-effort draws",
            "Invalid scenarios stored in manifest only (no waveform data)",
            "Both mandatory sub-modes (VC, PC) swept for every condition "
            "and mechanics pair -- see tier_vc / tier_pc counts",
            "Multi-compartment lung model: COPD=3, Pneumonia=3, "
            "ARDS tiers=2, Bronchospasm=2, Normal=1",
        ],
    }
    log_path = OUTPUT_DIR / "simv_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # --- Final summary ------------------------------------------------------
    valid_pct_grand = 100 * grand_valid / grand_total if grand_total > 0 else 0

    print(f"\n{'=' * 70}")
    print("  SIMV Thinned Dataset Generation Complete")
    print(f"  {'─' * 40}")
    print(f"  Total scenarios  : {grand_total:,}")
    print(f"  Valid            : {grand_valid:,}  ({valid_pct_grand:.1f}%)")
    print(f"  Invalid          : {grand_invalid:,}  "
          f"({100 - valid_pct_grand:.1f}%)")
    print(f"  Manifest columns : {len(manifest_rows[0]) if manifest_rows else 0}")
    print(f"  Manifest         : {manifest_path}")
    print(f"  Log              : {log_path}")
    print(f"  Total time       : {run_elapsed:.1f}s  "
          f"({run_elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")

    print(f"\n  {'Condition':<20} {'Total':>8} {'Valid':>8} "
          f"{'Valid%':>7} {'SyncFrac':>9} {'Spont/scn':>10}")
    print(f"  {'─' * 66}")
    for t in log_tiers:
        print(f"  {t['condition']:<20} {t['tier_total']:>8,} "
              f"{t['tier_valid']:>8,} {t['valid_pct']:>6.1f}% "
              f"{t['mean_synchronized_fraction']:>9.2f} "
              f"{t['mean_spontaneous_breaths']:>10.1f}")
    print()
    sys.stdout.flush()


if __name__ == "__main__":
    run()
