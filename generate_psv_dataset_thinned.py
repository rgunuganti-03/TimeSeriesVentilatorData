"""
generate_psv_dataset_thinned.py
--------------------------------
Self-contained PSV dataset generator with the thinned parameter grid
built in. Sweeps the kept ventilator-side and patient-side parameter
combinations — no intermediate full dataset, no post-processing step.

Run from the project root:
    python generate_psv_dataset_thinned.py

Output:
    data/exports/psv/
        psv_manifest_thinned.csv    — one row per scenario (valid + invalid)
        psv_generation_log.json     — run summary: counts, timing, config

PSV parameter space overview
-----------------------------
PSV has two independent parameter groups that are both swept:

    Ventilator-side (what the clinician controls):
        pressure_support_cmH2O  — PS level above PEEP
        peep_cmH2O              — extrinsic end-expiratory pressure
        flow_cycle_threshold    — fraction of peak flow at which Ti ends
        trigger_threshold_cmH2O — effort required to trigger
        rise_time_s             — ramp from PEEP to PIP

    Patient-side (patient physiology and effort model):
        pmus_peak_cmH2O         — mean peak inspiratory muscle effort
        effort_rate_per_min     — patient's neural respiratory rate
        effort_duration_s       — duration of each inspiratory effort
        pmus_cv                 — breath-to-breath effort variability (CV)

Thinned ventilator-side grid rationale
----------------------------------------
    pressure_support: [5, 12, 20] cmH2O
        Covers weaning (5), standard support (12), and high support (20).
        Intermediate values (8, 16) produce waveforms between these anchors
        with no distinct clinical strategy.

    peep: [0, 8, 15] cmH2O
        No PEEP, therapeutic mid-range (ARDS/COPD management), and high
        (severe ARDS). Covers the auto-PEEP interaction space for obstructive
        conditions without the intermediate 5 and 10 cmH2O values.

    flow_cycle_threshold: [0.10, 0.25, 0.40]
        All three values kept — each produces a qualitatively different
        waveform morphology: delayed cycling (0.10), nominal synchronous
        (0.25), and premature cycling (0.40). Cannot be thinned without
        losing entire dyssynchrony subtypes from the dataset.

    trigger_threshold: [1.5] cmH2O
        Single representative value. The threshold primarily gates
        ineffective triggering via the auto-PEEP interaction, which is
        already covered by varying resistance and COPD mechanics.
        The full DATASET_GRID value [0.5, 2.0] is archived for extended runs.

    rise_time: [0.1] s
        Single intermediate value. Instantaneous (0.0) and slow (0.4)
        ramp edges contribute minimal additional waveform diversity at the
        scale of dataset coverage; both are available in full DATASET_GRID.

Thinned patient-side grid rationale
--------------------------------------
    pmus_peak: [5, 12, 20] cmH2O
        Weak (recovering or over-sedated), moderate (comfortable wean),
        and strong (distressed or high drive) effort. The three values
        span the clinically meaningful range without redundancy.

    effort_rate: [14, 22, 30] breaths/min
        Slow (comfortable), normal, and rapid (distress/high drive).
        Adjacent values at 16, 20, 24 produce similar auto-PEEP accumulation
        and Vt patterns.

    effort_duration: [0.6, 1.0] s
        Short neural Ti (interaction with FCT → premature cycling risk) and
        long neural Ti (interaction with FCT → delayed cycling risk). These
        two values capture the key dyssynchrony-relevant extremes.

    pmus_cv: [0.20]
        Single value representing physiologically realistic breath-to-breath
        variability. Low (0.15) and high (0.35) CV are archived for targeted
        variability studies in the extended dataset.

Combinations per mechanics point:
    3 PS × 3 PEEP × 3 FCT × 1 threshold × 1 rise
    × 3 Pmus × 3 effort_rate × 2 effort_dur × 1 CV
    = 27 (ventilator) × 18 (patient) = 486 per mechanics point

PSV-specific dataset properties
---------------------------------
    - Breath-to-breath Vt variability is a feature, not noise
    - Dyssynchrony labels are stored per scenario (counts by subtype)
    - Auto-PEEP builds over multiple cycles: COPD uses 25 cycles,
      Bronchospasm uses 20 cycles, all others use 12 cycles
    - Seeds are deterministic per scenario for reproducibility
    - Pressure decomposition metrics stored: Pres, Pel, PEEP_total
    - Patient Vt ≤ delivered Vt (circuit + leak corrections applied)
    - ETT complications (cuff leak, partial obstruction) are out of scope
      for this thinned sweep — regenerate specific scenarios on demand

Extending the dataset
-----------------------
    To add ETT complications: instantiate scenarios with
        "ett_complication": "cuff_leak" | "partial_obstruction"
    and vary cuff_leak_fraction / obstruction_R_multiplier.

    To add SBT temporal sequences: call generate_sbt_sequence() from
    psv_generator.py with a subset of weaning-relevant params and store
    the rrsb_trajectory and per-window waveforms separately in a
    dedicated SBT manifest and HDF5 group.
"""

import itertools
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator.psv_generator import (
    IBW_KG,
    PPEAK_MAX_CMHH2O,
    PS_MAX_CMHH2O,
    VT_MAX_ML,
    VT_MIN_ML,
    _make_scenario_id,
    generate_breath_cycles,
)


# ---------------------------------------------------------------------------
# Thinned Parameter Grid — PSV
# ---------------------------------------------------------------------------

PSV_THINNED_GRID = {
    # ---- Ventilator-side ----
    "pressure_support_cmH2O":  [5, 12, 20],
    "peep_cmH2O":               [0, 8, 15],
    "flow_cycle_threshold":     [0.10, 0.25, 0.40],
    "trigger_threshold_cmH2O": [1.5],
    "rise_time_s":              [0.1],
    # ---- Patient-side ----
    "pmus_peak_cmH2O":         [5, 12, 20],
    "effort_rate_per_min":      [14, 22, 30],
    "effort_duration_s":        [0.6, 1.0],
    "pmus_cv":                  [0.20],
}

# Dyssynchrony label set — all known subtypes from _classify_dyssynchrony
DYSSYNC_SUBTYPES = [
    "synchronous",
    "ineffective_trigger",
    "delayed_cycling",
    "premature_cycling",
    "double_trigger",
    "flow_starvation",
    "reverse_trigger",
]


# ---------------------------------------------------------------------------
# Condition Tier Definitions
# ---------------------------------------------------------------------------
# Each tier defines:
#   name             : condition label (must match COMPARTMENT_PROFILES key)
#   compliance_range : (min, max) mL/cmH2O — thinned C sweep
#   compliance_step  : step size for np.arange
#   resistance_range : (min, max) cmH2O/L/s — thinned R sweep
#   resistance_step  : step size for np.arange
#   n_cycles         : cycles per scenario (more for high-R conditions to
#                      allow auto-PEEP to reach steady state)
#   default_effort   : representative patient effort for condition notes
#
# Resistance values include ETT contribution (~5–7 cmH2O/L/s for 7.5 mm ID).
# A resistance of 2 cmH2O/L/s is physiologically unrealistic for any
# intubated patient and is excluded from all tier ranges.

CONDITION_TIERS = [
    {
        "name":             "Normal",
        "compliance_range": (60, 100),
        "compliance_step":  10,
        "resistance_range": (8, 12),
        "resistance_step":   1,
        "n_cycles":         12,
        # Normal weaning patients: comfortable low-support PSV
        "default_effort":   {"pmus_peak": 8, "effort_rate": 18, "effort_dur": 0.80},
    },
    {
        "name":             "Mild ARDS",
        "compliance_range": (40, 55),
        "compliance_step":   5,
        "resistance_range": (10, 14),
        "resistance_step":   2,
        "n_cycles":         12,
        # Mild ARDS: moderately elevated drive, P-SILI risk at high Pmus
        "default_effort":   {"pmus_peak": 12, "effort_rate": 24, "effort_dur": 0.70},
    },
    {
        "name":             "Moderate ARDS",
        "compliance_range": (28, 40),
        "compliance_step":   4,
        "resistance_range": (12, 16),
        "resistance_step":   2,
        "n_cycles":         12,
        # Moderate ARDS: high respiratory drive, double-trigger risk
        "default_effort":   {"pmus_peak": 15, "effort_rate": 28, "effort_dur": 0.65},
    },
    {
        "name":             "Severe ARDS",
        "compliance_range": (15, 28),
        "compliance_step":   4,
        "resistance_range": (14, 20),
        "resistance_step":   3,
        "n_cycles":         12,
        # Severe ARDS: usually paralysed; model early awakening / light sedation
        "default_effort":   {"pmus_peak": 6, "effort_rate": 18, "effort_dur": 0.80},
    },
    {
        "name":             "COPD",
        "compliance_range": (80, 150),
        "compliance_step":  20,
        "resistance_range": (18, 35),
        "resistance_step":   5,
        "n_cycles":         25,   # Extra cycles for auto-PEEP steady state
        # COPD: elevated drive (must overcome auto-PEEP), high ineffective rate
        "default_effort":   {"pmus_peak": 14, "effort_rate": 26, "effort_dur": 0.75},
    },
    {
        "name":             "Bronchospasm",
        "compliance_range": (60, 90),
        "compliance_step":  10,
        "resistance_range": (25, 50),
        "resistance_step":   5,
        "n_cycles":         20,   # Elevated R → more auto-PEEP buildup needed
        # Bronchospasm: recovery phase only; acute phase uses mandatory modes
        "default_effort":   {"pmus_peak": 12, "effort_rate": 22, "effort_dur": 0.75},
    },
    {
        "name":             "Pneumonia",
        "compliance_range": (40, 65),
        "compliance_step":   5,
        "resistance_range": (10, 16),
        "resistance_step":   2,
        "n_cycles":         12,
        # Pneumonia: similar to mild ARDS; secretions drive elevated resistance
        "default_effort":   {"pmus_peak": 11, "effort_rate": 23, "effort_dur": 0.75},
    },
]

OUTPUT_DIR = Path("data/exports/psv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mechanics_grid(tier: dict) -> list:
    """
    Build the (C, R) mechanics grid for one condition tier.
    Returns a list of (compliance, resistance) float tuples.
    """
    c_start, c_stop = tier["compliance_range"]
    r_start, r_stop = tier["resistance_range"]
    compliances = np.arange(
        c_start, c_stop + tier["compliance_step"], tier["compliance_step"]
    ).tolist()
    resistances = np.arange(
        r_start, r_stop + tier["resistance_step"], tier["resistance_step"]
    ).tolist()
    return [(float(c), float(r)) for c in compliances for r in resistances]


def _ventilator_combos() -> list:
    """
    Build all ventilator-side parameter combinations from PSV_THINNED_GRID.
    Returns a list of dicts, one per combination.
    """
    keys = [
        "pressure_support_cmH2O",
        "peep_cmH2O",
        "flow_cycle_threshold",
        "trigger_threshold_cmH2O",
        "rise_time_s",
    ]
    combos = list(itertools.product(*[PSV_THINNED_GRID[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _patient_combos() -> list:
    """
    Build all patient-side parameter combinations from PSV_THINNED_GRID.
    Returns a list of dicts, one per combination.
    """
    keys = [
        "pmus_peak_cmH2O",
        "effort_rate_per_min",
        "effort_duration_s",
        "pmus_cv",
    ]
    combos = list(itertools.product(*[PSV_THINNED_GRID[k] for k in keys]))
    return [dict(zip(keys, combo)) for combo in combos]


def _make_deterministic_seed(condition: str, C: float, R: float,
                              vent: dict, patient: dict) -> int:
    """
    Compute a deterministic seed from scenario parameters so that the same
    scenario always produces the same stochastic Pmus draws.
    Uses Python's hash() with a fixed string representation.
    """
    key = (
        condition,
        round(C, 1), round(R, 1),
        round(vent["pressure_support_cmH2O"], 1),
        round(vent["peep_cmH2O"], 1),
        round(vent["flow_cycle_threshold"], 3),
        round(vent["trigger_threshold_cmH2O"], 2),
        round(vent["rise_time_s"], 2),
        round(patient["pmus_peak_cmH2O"], 1),
        round(patient["effort_rate_per_min"], 1),
        round(patient["effort_duration_s"], 2),
        round(patient["pmus_cv"], 3),
    )
    return int(abs(hash(key))) % (2 ** 31)


def _dyssynchrony_counts(labels: list) -> dict:
    """
    Summarize a list of per-breath dyssynchrony labels into a dict of counts.
    All known subtypes are present as keys (value = 0 if not observed).
    """
    counts = Counter(labels)
    return {subtype: counts.get(subtype, 0) for subtype in DYSSYNC_SUBTYPES}


def _generate_psv_thinned(condition_name: str,
                            compliance_mL_per_cmH2O: float,
                            resistance_cmH2O_L_s: float,
                            n_cycles: int) -> list:
    """
    Sweep the PSV thinned grid for one condition + mechanics pair.

    Iterates over all (ventilator_combo, patient_combo) pairs from
    PSV_THINNED_GRID. For each combination, calls generate_breath_cycles
    with a deterministic seed and collects all metrics and dyssynchrony
    label counts.

    Parameters
    ----------
    condition_name           : str
    compliance_mL_per_cmH2O : float
    resistance_cmH2O_L_s    : float
    n_cycles                 : int — more for COPD/Bronchospasm auto-PEEP

    Returns
    -------
    list of dicts — one per (ventilator, patient) parameter combination
    """
    scenarios = []
    vent_combos    = _ventilator_combos()
    patient_combos = _patient_combos()

    for vent, patient in itertools.product(vent_combos, patient_combos):

        params = {
            # Mechanics
            "compliance_mL_per_cmH2O": compliance_mL_per_cmH2O,
            "resistance_cmH2O_L_s":     resistance_cmH2O_L_s,
            "condition":                condition_name,
            # Ventilator-side
            **vent,
            # Patient-side
            **patient,
        }

        seed = _make_deterministic_seed(
            condition_name, compliance_mL_per_cmH2O, resistance_cmH2O_L_s,
            vent, patient
        )

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles, seed=seed)
        except Exception as exc:
            # Catch generator errors (e.g. unexpected numeric overflow)
            # and record as invalid rather than crashing the entire run
            scenario_id = _make_scenario_id(condition_name, params)
            scenarios.append({
                "scenario_id":    scenario_id,
                "condition":      condition_name,
                "params":         params,
                "metrics":        {},
                "dyssync_labels": [],
                "is_valid":       False,
                "invalid_reason": f"Generator error: {exc}",
                "generated_at":   datetime.now(timezone.utc).isoformat(),
                "seed":           seed,
            })
            continue

        scenario_id = _make_scenario_id(condition_name, params)
        metrics = {
            "ppeak_cmH2O":                  result.get("ppeak_cmH2O",                  ""),
            "delivered_vt_mL":              result.get("delivered_vt_mL",              ""),
            "patient_vt_mL":                result.get("patient_vt_mL",                ""),
            "driving_p_cmH2O":              result.get("driving_p_cmH2O",              ""),
            "mean_paw_cmH2O":               result.get("mean_paw_cmH2O",               ""),
            "auto_peep_cmH2O":              result.get("auto_peep_cmH2O",              ""),
            "total_peep_cmH2O":             result.get("total_peep_cmH2O",             ""),
            "fill_fraction":                result.get("fill_fraction",                ""),
            "minute_vent_L":                result.get("minute_vent_L",                ""),
            "pres_peak_cmH2O":              result.get("pres_peak_cmH2O",              ""),
            "pel_end_insp_cmH2O":           result.get("pel_end_insp_cmH2O",           ""),
            "stress_index":                 result.get("stress_index",                 ""),
            "pres_pel_ratio":               result.get("pres_pel_ratio",               ""),
            "triggered_breath_rate":        result.get("triggered_breath_rate",        ""),
            "ineffective_trigger_fraction": result.get("ineffective_trigger_fraction", ""),
        }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "dyssync_labels": result.get("breath_dyssynchrony_labels", []),
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

    vent_combos_per_point    = len(_ventilator_combos())
    patient_combos_per_point = len(_patient_combos())
    combos_per_point         = vent_combos_per_point * patient_combos_per_point

    manifest_rows = []
    log_tiers     = []

    grand_total   = 0
    grand_valid   = 0
    grand_invalid = 0

    print("=" * 70)
    print("  PSV Thinned Dataset Generation")
    print(f"  Started        : {run_ts}")
    print(f"  Output dir     : {OUTPUT_DIR.resolve()}")
    print(f"  Vent combos    : {vent_combos_per_point} per mechanics point")
    print(f"  Patient combos : {patient_combos_per_point} per mechanics point")
    print(f"  Total combos   : {combos_per_point} per mechanics point")
    print("  Note           : Event-driven ODE — longer runtime than VCV/PCV")
    print("  Tip            : Run with nohup and tail -f psv_thinned.log")
    print("=" * 70)
    sys.stdout.flush()

    for tier in CONDITION_TIERS:
        tier_name  = tier["name"]
        mechanics  = _mechanics_grid(tier)
        n_cycles   = tier["n_cycles"]
        tier_start = time.perf_counter()

        tier_total   = 0
        tier_valid   = 0
        tier_invalid = 0

        # Per-tier dyssynchrony accumulators (for log summary)
        tier_dyssync_total = Counter()

        print(f"\n  [{tier_name}]")
        print(f"    Mechanics pairs : {len(mechanics)}")
        print(f"    Cycles/scenario : {n_cycles}")
        print(f"    Total scenarios : "
              f"{len(mechanics) * combos_per_point:,} (estimated)")
        sys.stdout.flush()

        for C, R in mechanics:
            scenarios = _generate_psv_thinned(
                condition_name           = tier_name,
                compliance_mL_per_cmH2O = C,
                resistance_cmH2O_L_s    = R,
                n_cycles                 = n_cycles,
            )

            for s in scenarios:
                tier_total += 1
                p  = s["params"]
                m  = s["metrics"]
                dc = _dyssynchrony_counts(s["dyssync_labels"])

                if s["is_valid"]:
                    tier_valid += 1
                else:
                    tier_invalid += 1

                for subtype, count in dc.items():
                    tier_dyssync_total[subtype] += count

                manifest_rows.append({
                    # ---- Identity ----------------------------------------
                    "scenario_id":                  s["scenario_id"],
                    "condition":                    s["condition"],
                    "generated_at":                 s["generated_at"],
                    "seed":                         s["seed"],
                    "is_valid":                     s["is_valid"],
                    "invalid_reason":               s["invalid_reason"],
                    # ---- Mechanics ---------------------------------------
                    "compliance_mL_per_cmH2O":      p["compliance_mL_per_cmH2O"],
                    "resistance_cmH2O_L_s":          p["resistance_cmH2O_L_s"],
                    # ---- Ventilator-side parameters ----------------------
                    "pressure_support_cmH2O":        p["pressure_support_cmH2O"],
                    "peep_cmH2O":                    p["peep_cmH2O"],
                    "flow_cycle_threshold":           p["flow_cycle_threshold"],
                    "trigger_threshold_cmH2O":        p["trigger_threshold_cmH2O"],
                    "rise_time_s":                   p["rise_time_s"],
                    # ---- Patient-side parameters -------------------------
                    "pmus_peak_cmH2O":               p["pmus_peak_cmH2O"],
                    "effort_rate_per_min":            p["effort_rate_per_min"],
                    "effort_duration_s":              p["effort_duration_s"],
                    "pmus_cv":                        p["pmus_cv"],
                    # ---- Scalar metrics (blank if invalid) ---------------
                    "ppeak_cmH2O":                   m.get("ppeak_cmH2O",                  ""),
                    "delivered_vt_mL":               m.get("delivered_vt_mL",              ""),
                    "patient_vt_mL":                 m.get("patient_vt_mL",                ""),
                    "driving_p_cmH2O":               m.get("driving_p_cmH2O",              ""),
                    "mean_paw_cmH2O":                m.get("mean_paw_cmH2O",               ""),
                    "auto_peep_cmH2O":               m.get("auto_peep_cmH2O",              ""),
                    "total_peep_cmH2O":              m.get("total_peep_cmH2O",             ""),
                    "fill_fraction":                  m.get("fill_fraction",                ""),
                    "minute_vent_L":                  m.get("minute_vent_L",                ""),
                    "pres_peak_cmH2O":               m.get("pres_peak_cmH2O",              ""),
                    "pel_end_insp_cmH2O":            m.get("pel_end_insp_cmH2O",           ""),
                    "stress_index":                   m.get("stress_index",                 ""),
                    "pres_pel_ratio":                 m.get("pres_pel_ratio",               ""),
                    "triggered_breath_rate":          m.get("triggered_breath_rate",        ""),
                    "ineffective_trigger_fraction":   m.get("ineffective_trigger_fraction", ""),
                    # ---- Dyssynchrony label counts -----------------------
                    "n_synchronous":                 dc["synchronous"],
                    "n_ineffective_trigger":         dc["ineffective_trigger"],
                    "n_delayed_cycling":             dc["delayed_cycling"],
                    "n_premature_cycling":           dc["premature_cycling"],
                    "n_double_trigger":              dc["double_trigger"],
                    "n_flow_starvation":             dc["flow_starvation"],
                    "n_reverse_trigger":             dc["reverse_trigger"],
                })

        tier_elapsed = time.perf_counter() - tier_start
        grand_total   += tier_total
        grand_valid   += tier_valid
        grand_invalid += tier_invalid

        valid_pct   = 100 * tier_valid   / tier_total if tier_total > 0 else 0
        invalid_pct = 100 * tier_invalid / tier_total if tier_total > 0 else 0

        print(f"    Total    : {tier_total:,}")
        print(f"    Valid    : {tier_valid:,}  ({valid_pct:.1f}%)")
        print(f"    Invalid  : {tier_invalid:,}  ({invalid_pct:.1f}%)")
        print(f"    Dominant dyssync: "
              f"{tier_dyssync_total.most_common(2)}")
        print(f"    Time     : {tier_elapsed:.1f}s")
        sys.stdout.flush()

        log_tiers.append({
            "condition":           tier_name,
            "mechanics_pairs":     len(mechanics),
            "n_cycles":            n_cycles,
            "tier_total":          tier_total,
            "tier_valid":          tier_valid,
            "tier_invalid":        tier_invalid,
            "valid_pct":           round(valid_pct, 1),
            "dyssync_totals":      dict(tier_dyssync_total),
            "elapsed_s":           round(tier_elapsed, 1),
        })

    # --- Write manifest ---------------------------------------------------
    manifest_path = OUTPUT_DIR / "psv_manifest_thinned.csv"
    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_path, index=False)

    # --- Write generation log --------------------------------------------
    run_elapsed = time.perf_counter() - run_start
    log = {
        "mode":                    "PSV",
        "dataset_type":            "thinned",
        "generated_at":            run_ts,
        "ibw_kg":                  IBW_KG,
        "output_dir":              str(OUTPUT_DIR.resolve()),
        "thinned_grid":            PSV_THINNED_GRID,
        "ventilator_combos_per_point":   vent_combos_per_point,
        "patient_combos_per_point":      patient_combos_per_point,
        "total_combos_per_point":        combos_per_point,
        "grand_total":             grand_total,
        "grand_valid":             grand_valid,
        "grand_invalid":           grand_invalid,
        "valid_pct":               round(100 * grand_valid / grand_total, 1)
                                   if grand_total > 0 else 0,
        "total_elapsed_s":         round(run_elapsed, 1),
        "total_elapsed_min":       round(run_elapsed / 60, 1),
        "tiers":                   log_tiers,
        "validity_thresholds": {
            "ppeak_max_cmH2O":        PPEAK_MAX_CMHH2O,
            "ps_max_cmH2O":           PS_MAX_CMHH2O,
            "vt_min_mL":              VT_MIN_ML,
            "vt_max_mL":              VT_MAX_ML,
        },
        "dyssync_subtypes_tracked": DYSSYNC_SUBTYPES,
        "notes": [
            "COPD uses 25 cycles/scenario for auto-PEEP steady state",
            "Bronchospasm uses 20 cycles/scenario",
            "All others use 12 cycles/scenario",
            "Seeds are deterministic: same params always produce same waveform",
            "Invalid scenarios stored in manifest only (no waveform data)",
            "Pressure decomposition arrays available on regeneration: "
            "pressure_resistive, pressure_elastic, pressure_total_peep",
            "Multi-compartment lung model: COPD=3, Pneumonia=3, "
            "ARDS=2, others=1",
        ],
    }
    log_path = OUTPUT_DIR / "psv_generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    # --- Final summary ----------------------------------------------------
    valid_pct_grand = 100 * grand_valid / grand_total if grand_total > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  PSV Thinned Dataset Generation Complete")
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

    # Per-condition summary table
    print(f"\n  {'Condition':<20} {'Total':>8} {'Valid':>8} "
          f"{'Valid%':>7} {'Top dyssync':>20}")
    print(f"  {'─' * 67}")
    for t in log_tiers:
        top = (sorted(t["dyssync_totals"].items(),
                       key=lambda x: -x[1])[:1] or [("—", 0)])[0]
        print(f"  {t['condition']:<20} {t['tier_total']:>8,} "
              f"{t['tier_valid']:>8,} "
              f"{t['valid_pct']:>6.1f}% "
              f"  {top[0]}")
    print()


if __name__ == "__main__":
    run()
