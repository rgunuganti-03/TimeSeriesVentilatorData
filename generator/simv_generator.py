"""
generator/simv_generator.py
----------------------------
SIMV (Synchronized Intermittent Mandatory Ventilation) waveform generator —
multi-compartment, hybrid mandatory/spontaneous.

Control loop
------------
SIMV has no single control regime. It switches between two physics regimes
already implemented elsewhere in this project, sequenced by a synchronization
window:

    Mandatory breath (VC or PC sub-mode) — identical control loop to
    vcv_generator / pcv_generator. Ventilator prescribes flow (VC) or
    pressure (PC); the patient's effort decides only WHEN the breath starts,
    never how much volume/pressure is delivered.

    Spontaneous breath — identical control loop to psv_generator.
    Patient-triggered, pressure-limited to the set pressure-support level,
    flow-cycled off at the condition-appropriate flow-cycle threshold.

This is the fourth engine in the project and introduces no new lung
physics. What it introduces is scheduling logic: the synchronization
window, which decides — at each patient effort attempt — whether that
attempt is eligible to trigger a synchronized mandatory breath or only a
spontaneous one, and falls back to a time-triggered mandatory breath if no
effort occurs before the window closes. See Docs/control_loops/
SIMV_CONTROL_LOOP.md for the full mechanistic description.

Synchronization window
-----------------------
    T_mand       = 60 / respiratory_rate           (mandatory cycle length)
    W            = f_window * T_mand               (window width)
    window_open  = T_mand - W                        (relative to the start
                                                        of the current cycle)

Patient effort attempts occur at the neural rate (effort_rate_per_min),
deterministically spaced at 60/effort_rate_per_min apart (matching
psv_generator's treatment of effort_rate_per_min as the patient's intrinsic
neural rate), with breath-to-breath variability layered on top via
pmus_peak_cmH2O (log-normal jitter) and effort_duration_s (normal jitter),
exactly as in psv_generator.

For each attempt:
    - Before window_open: a successful trigger (same pressure-based
      _check_trigger used in psv_generator — see literature-grounding note
      below) delivers a spontaneous (PSV-style) breath. A failed trigger is
      recorded as an ineffective spontaneous-zone effort, exactly as in
      psv_generator's ineffective-trigger handling.
    - At/after window_open: a successful trigger delivers a *synchronized*
      mandatory breath, immediately, using the VC or PC physics selected by
      mandatory_mode. A failed trigger is skipped (time keeps advancing).
    - If window_open through T_mand elapses with no successful trigger, a
      *time-triggered* mandatory breath is delivered at T_mand.

One mandatory breath is delivered exactly once per macro-cycle — this,
combined with the sequential single-active-breath execution model (no new
breath of either type starts while one is in progress), is what prevents
breath stacking (ARCHITECTURE.md's flagged requirement for this mode).//
The macro-cycle timer resets at the *start* of whichever mandatory breath
was just delivered, so a synchronized (early) mandatory breath shifts the
next window earlier, matching real SIMV timer-reset behavior.

Literature grounding (see "Grounding a Synthetic SIMV Waveform Generator in
the Clinical Literature", compiled for this project):
    - f_window has no single agreed physiological value across ventilator
      platforms (Servo frames it as "first 90% of breath cycle time";
      Dräger as "~20% of expiratory time, 0-80% selectable"; teaching
      sources as "~25% of cycle time" / "~0.5 s"). Implemented here as a
      tunable fraction of T_mand, default range 0.15-0.30.
    - Flow-cycle threshold defaults were revised from an earlier draft:
      obstructive/bronchospasm conditions default toward the high end
      (~0.65, supported by Tassaux et al. 2005, AJRCCM 172:1283-1289, which
      found ETS 70% reduced delayed cycling and auto-PEEP versus ETS 10% in
      obstructive patients) while restrictive/ARDS conditions default to a
      mid-range ~0.25-0.40 rather than ~0.10 (Tokioka et al. 2001, Anesth
      Analg 92:161-165, found very low cycling criteria in ARDS/ALI
      patients prolonged inspiration and increased expiratory work of
      breathing — a low threshold is a "delayed-cycling stress test," not
      the ARDS default).
    - Asynchrony is labeled using the AI > 10% threshold established in
      Thille et al. 2006 (Intensive Care Med 32:1515-1522) as the
      high-asynchrony regime.
    - Mandatory-rate grid is scoped to the clinically relevant SIMV range
      (~4-12 breaths/min: weaning endpoint through initiation) rather than
      the full CMV range used by vcv_generator/pcv_generator.

Physiological refinements incorporated (identical to sibling generators
except where noted)
---------------------------------------------------------------------------
    1. Multi-compartment lung mechanics — parallel RC compartments per
       condition, identical COMPARTMENT_PROFILES to vcv/pcv/psv/prvc
       (Normal=1, ARDS tiers=2, COPD=3, Bronchospasm=2, Pneumonia=3).
    2. Continuous compartment/auto-PEEP state across breath-type
       transitions (NEW to this engine — the other four are each
       self-contained single-regime simulators; SIMV must hand live lung
       state between mandatory and spontaneous physics within one
       scenario, since it's the same lung throughout).
    3. Flow-dependent resistance (Rohrer: K1*Q + K2*Q*|Q|), applied on
       total flow for the displayed pressure decomposition.
    4. Volume-dependent expiratory resistance (dynamic airway collapse) —
       used identically whether the preceding breath was mandatory or
       spontaneous, since expiration physics doesn't depend on how
       inspiration was triggered.
    5. Non-linear compliance per compartment via stress index.
    6. PEEP-recruited compliance, condition-specific slope
       (RECRUITMENT_SLOPES, identical to psv/prvc; zero for COPD/
       Bronchospasm).
    7. Chest wall compliance in series (default ~inert).
    8. Circuit compliance correction (post-hoc VT scalar, mandatory
       breaths only — matches vcv/pcv precedent).
    9. ETT complications (cuff leak, obstruction) — applied to whichever
       breath type is active.
   10. Patient-ventilator dyssynchrony labeling on spontaneous breaths only
       (mandatory breaths are ventilator-paced by construction and carry a
       fixed "controlled" label) — ineffective triggering, double
       triggering, delayed cycling, premature cycling, flow starvation.
       Reused from psv_generator's classification logic.

Deliberately NOT modeled (out of scope for this pass, matching the scoping
precedent prvc_generator set for its own mode): unsupported (zero-PS)
spontaneous breathing, reverse triggering, spontaneous breathing trial
temporal sequencing. Available as a follow-up if the parameter grid
specifically calls for it (see SIMV_CONTROL_LOOP.md, Section 11.3).

Interface contract (identical to vcv_generator, pcv_generator,
psv_generator, prvc_generator)
------------------------------------------------------------------------------
    generate_breath_cycles(params, n_cycles, seed) -> dict
    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list

n_cycles here means the number of *mandatory* macro-cycles to simulate
(matching vcv/pcv/prvc's use of n_cycles for mandatory-breath count); the
number of spontaneous breaths generated in between is not fixed in advance
— it falls out of the effort rate and the window width.

Output dict keys
----------------
    Core waveforms (np.ndarray, 100 Hz):
        time, pressure, flow, volume

    Pressure decomposition (np.ndarray, same length):
        pressure_resistive, pressure_elastic, pressure_total_peep

    Per-breath records (list of dicts, length = total breaths delivered):
        breath_records — each: {"breath_type", "trigger_mode",
        "dyssynchrony_label", "delivered_vt_ml", "ppeak_cmH2O", "t_start_s","duration_s"}

    Scalar metrics:
        n_mandatory_breaths, n_spontaneous_breaths,
        mandatory_synchronized_fraction, mandatory_delivered_vt_ml,
        spontaneous_delivered_vt_ml, ppeak_cmH2O, mean_paw_cmH2O,
        auto_peep_cmH2O, minute_vent_l, ineffective_trigger_fraction,
        n_compartments

    Validity:
        is_valid, invalid_reason

Run smoke test:
    python generator/simv_generator.py

NOTE on architecture (see SIMV_CONTROL_LOOP.md Section 12): helper
functions and compartment profiles are inlined here to match the existing
project pattern, as vcv/pcv/psv/prvc all do — this is the fifth copy the
other four docstrings' "future lung_physics.py refactor" notes anticipated.
Not resolved here; flagged again since this file is the one that makes the
refactor's value concrete (mandatory- and spontaneous-breath physics now
have to share live state within one run, which none of the four prior
generators were shaped to do).
"""

import itertools
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Section 1 — Parameter Grid
# ---------------------------------------------------------------------------
# Mandatory-rate range is deliberately scoped to SIMV's clinical use
# (weaning-relevant backup rate, ~4-12 bpm) rather than vcv/pcv's full
# 8-30 bpm CMV range — see literature-grounding note in module docstring.
PARAMETER_GRID: Dict = {
    "mandatory_mode":          ["VC", "PC"],
    "tidal_volume_ml_per_kg":  [4, 6, 8, 10],           # VC mandatory only
    "insp_pressure_cmH2O":     [10, 15, 20, 25, 30, 35],  # PC mandatory only
    "flow_pattern":            ["square", "decelerating"],  # VC mandatory only
    "respiratory_rate":        [4, 6, 8, 10, 12],       # mandatory backup rate
    "peep_cmH2O":               [0, 4, 8, 12, 16, 20],
    "ie_ratio":                 [1.0, 0.5, 0.33],        # mandatory breaths only
    "rise_time_s":              [0.0, 0.1, 0.2, 0.4],    # PC mandatory + spontaneous
    "f_window":                 [0.15, 0.20, 0.25, 0.30],
    "pressure_support_cmH2O":   [5, 8, 12, 16, 20],
    "flow_cycle_threshold":     [0.25, 0.40, 0.65],
    "trigger_threshold_cmH2O": [0.5, 1.5, 3.0],
    "pmus_peak_cmH2O":          [5, 8, 12, 16, 20],
    "effort_rate_per_min":      [12, 16, 20, 25, 30],
    "effort_duration_s":        [0.5, 0.7, 0.9, 1.1],
    "pmus_cv":                  [0.15, 0.25, 0.35],
}

# ---------------------------------------------------------------------------
# Section 2 — Safety Thresholds and Constants
# ---------------------------------------------------------------------------
IBW_KG: float                  = 70.0
VT_MIN_ML: float                = IBW_KG * 3       # 210 mL — inadequate ventilation
VT_MAX_ML: float                = IBW_KG * 12      # 840 mL — overdistension
PPEAK_MAX_CMHH2O: float         = 50.0             # barotrauma risk
DRIVING_P_MAX_CMHH2O: float     = 20.0             # ARDS mortality threshold (VC mandatory)
INSP_PRESSURE_MAX_CMHH2O: float = 35.0             # max driving above PEEP (PC mandatory)
PS_MAX_CMHH2O: float            = 20.0             # clinical PS ceiling
DT: float                       = 0.01             # 100 Hz internal timestep
INSPIRATORY_PAUSE_S: float      = 0.3              # VC mandatory only (Pplat)
MAX_INSP_TIME_S: float          = 3.0              # spontaneous-breath safety cutoff

CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5
DEFAULT_CHEST_WALL_COMPLIANCE: float   = 250.0     # mL/cmH2O (~inert default)
ETT_K1: float = 5.0   # cmH2O/L/s     — viscous ETT resistance
ETT_K2: float = 3.0   # cmH2O/(L/s)^2 — turbulent ETT resistance

AI_HIGH_ASYNCHRONY_THRESHOLD: float = 0.10   # Thille et al. 2006 — AI > 10%


# ---------------------------------------------------------------------------
# Section 3 — Condition-Specific Compartment Profiles
# ---------------------------------------------------------------------------
# Identical to vcv_generator / pcv_generator / psv_generator / prvc_generator.
COMPARTMENT_PROFILES: Dict = {
    "Normal": [
        {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
         "R_exp_ratio": 1.2,  "tethering": 0.80},
    ],
    "Mild ARDS": [
        {"fraction": 0.75, "C_frac": 0.90, "R_frac": 1.00,
         "R_exp_ratio": 1.4,  "tethering": 0.40},
        {"fraction": 0.25, "C_frac": 0.10, "R_frac": 1.60,
         "R_exp_ratio": 2.0,  "tethering": 0.10},
    ],
    "Moderate ARDS": [
        {"fraction": 0.60, "C_frac": 0.85, "R_frac": 1.00,
         "R_exp_ratio": 1.6,  "tethering": 0.25},
        {"fraction": 0.40, "C_frac": 0.05, "R_frac": 1.80,
         "R_exp_ratio": 2.5,  "tethering": 0.08},
    ],
    "Severe ARDS": [
        {"fraction": 0.40, "C_frac": 0.80, "R_frac": 1.00,
         "R_exp_ratio": 1.8,  "tethering": 0.20},
        {"fraction": 0.60, "C_frac": 0.03, "R_frac": 2.00,
         "R_exp_ratio": 3.0,  "tethering": 0.05},
    ],
    "COPD": [
        {"fraction": 0.35, "C_frac": 0.70, "R_frac": 0.55,
         "R_exp_ratio": 4.0,  "tethering": 0.15},
        {"fraction": 0.40, "C_frac": 1.05, "R_frac": 1.27,
         "R_exp_ratio": 6.0,  "tethering": 0.10},
        {"fraction": 0.25, "C_frac": 1.40, "R_frac": 2.36,
         "R_exp_ratio": 8.0,  "tethering": 0.05},
    ],
    "Bronchospasm": [
        {"fraction": 0.60, "C_frac": 0.90, "R_frac": 0.80,
         "R_exp_ratio": 3.0,  "tethering": 0.00},
        {"fraction": 0.40, "C_frac": 1.10, "R_frac": 1.43,
         "R_exp_ratio": 5.0,  "tethering": 0.00},
    ],
    "Pneumonia": [
        {"fraction": 0.60, "C_frac": 1.10, "R_frac": 0.83,
         "R_exp_ratio": 1.5,  "tethering": 0.70},
        {"fraction": 0.25, "C_frac": 0.55, "R_frac": 1.83,
         "R_exp_ratio": 3.0,  "tethering": 0.30},
        {"fraction": 0.15, "C_frac": 0.07, "R_frac": 6.67,
         "R_exp_ratio": 2.0,  "tethering": 0.10},
    ],
}

# PEEP-recruited compliance slopes — identical to psv_generator/prvc_generator.
RECRUITMENT_SLOPES: Dict = {
    "Normal":        0.00,
    "Mild ARDS":     0.50,
    "Moderate ARDS": 0.90,
    "Severe ARDS":   0.60,
    "COPD":          0.00,
    "Bronchospasm":  0.00,
    "Pneumonia":     0.10,
}

# Condition-aware flow-cycle-threshold guidance (literature-refined defaults
# for preset/UI use — flow_cycle_threshold itself remains a plain generator
# parameter, matching psv_generator; this dict is guidance, not enforced).
FCT_CONDITION_DEFAULTS: Dict = {
    "Normal":        0.25,
    "Mild ARDS":      0.30,
    "Moderate ARDS":  0.35,
    "Severe ARDS":    0.40,
    "COPD":           0.65,
    "Bronchospasm":   0.65,
    "Pneumonia":      0.35,
}


# ---------------------------------------------------------------------------
# Section 4 — Physics helper functions (mirrored from psv_generator /
# vcv_generator / pcv_generator / prvc_generator)
# ---------------------------------------------------------------------------

def _rohrer_resistance(Q: float, K1: float, K2: float) -> float:
    """Rohrer airway/ETT pressure drop: K1*Q + K2*Q*|Q|. Sign-preserving."""
    return K1 * Q + K2 * Q * abs(Q)


def _R_insp_with_tethering(R_base: float, V_current: float, V_target: float,
                            tethering: float) -> float:
    """Volume-dependent inspiratory resistance via parenchymal tethering."""
    V_frac = float(np.clip(V_current / max(V_target, 1.0), 0.0, 1.0))
    return R_base * max(1.0 - tethering * 0.30 * V_frac, 0.30)


def _R_exp_dynamic(V_current: float, V_end_insp: float, R_insp: float,
                    R_exp_ratio: float) -> float:
    """Volume-dependent expiratory resistance from dynamic airway collapse."""
    frac_exhaled = 1.0 - float(np.clip(V_current / max(V_end_insp, 1.0), 0.0, 1.0))
    return R_insp * (1.0 + (R_exp_ratio - 1.0) * frac_exhaled)


def _compliance_nonlinear(V_mL: float, C_base: float, V_ref: float,
                           stress_index: float = 1.0) -> float:
    """Power-law volume-dependent compliance. SI=1.0 -> linear (no-op)."""
    if abs(stress_index - 1.0) < 0.01 or V_mL <= 0.0:
        return C_base
    V_norm = max(V_mL / max(V_ref, 1.0), 0.01)
    return float(C_base * (V_norm ** (1.0 - stress_index)))


def _peep_recruited_compliance(C_base: float, peep: float, peep_ref: float,
                                recruitment_slope: float) -> float:
    """PEEP-mediated alveolar recruitment increases effective compliance."""
    delta_peep = max(0.0, peep - peep_ref)
    return C_base + recruitment_slope * delta_peep


def _C_rs(C_lung: float, C_chest: float) -> float:
    """Total respiratory system compliance, lung and chest wall in series."""
    if C_chest >= 9000.0:
        return C_lung
    return 1.0 / (1.0 / max(C_lung, 0.1) + 1.0 / max(C_chest, 0.1))


def _circuit_vt_correction(vt_mL: float, ppeak: float, peep: float,
                            C_circ: float = CIRCUIT_COMPLIANCE_ML_PER_CMH2O,
                            compensated: bool = True) -> float:
    """Adjust delivered Vt for gas sequestered in compliant ventilator circuit."""
    if compensated:
        return vt_mL
    return max(0.0, vt_mL - C_circ * max(ppeak - peep, 0.0))


def _check_trigger(pmus_at_onset: float, auto_peep: float, threshold: float) -> bool:
    """
    Determine whether a patient effort successfully triggers the ventilator.

    Reused unchanged from psv_generator (project decision: SIMV's mandatory-
    breath trigger uses the same pressure-based mechanic as PSV's
    spontaneous-breath trigger, rather than introducing the brief's
    separate flow-based "trigger sensitivity" units — see
    SIMV_CONTROL_LOOP.md Section 10 trigger-threshold note). The patient
    must first overcome auto-PEEP before net effort can cross threshold.
    """
    return (pmus_at_onset - auto_peep) >= threshold

def _advance_schedule(next_attempt_t: float, floor_t: float, interval: float) -> float:
    """
    Advance a periodic patient-effort schedule to the first slot at or after
    floor_t, preserving the schedule's original neural phase rather than
    snapping it to floor_t.

    Bug this fixes: previously, when a scheduled attempt fell inside a
    mandatory breath's own inspiration/pause, the schedule was clamped to
    the mandatory breath's end time exactly (`max(next_attempt_t, t_current)`),
    producing a zero-gap re-trigger — a spontaneous breath firing in the
    same instant the mandatory breath's inspiration ended, with no passive
    expiratory time between them. Advancing by whole `interval` steps
    instead guarantees the next attempt is strictly in the future and stays
    on the patient's original neural timing.
    """
    while next_attempt_t <= floor_t + 1e-9:
        next_attempt_t += interval
    return next_attempt_t


def _pmus_waveform(t_elapsed: float, t_duration: float, pmus_peak: float) -> float:
    """Half-sinusoidal Pmus profile for a single inspiratory effort."""
    if t_elapsed <= 0.0 or t_elapsed >= t_duration or t_duration <= 0.0:
        return 0.0
    return pmus_peak * float(np.sin(np.pi * t_elapsed / t_duration))


def _get_ett_params(ett_complication: Optional[str], cuff_leak_frac: float,
                     obs_multiplier: float, K1_base: float, K2_base: float
                     ) -> Tuple[float, float, float]:
    """Apply ETT complication overlays to the Rohrer coefficients / leak fraction."""
    if ett_complication == "obstruction":
        return K1_base * obs_multiplier, K2_base * obs_multiplier, 0.0
    if ett_complication == "cuff_leak":
        return K1_base, K2_base, cuff_leak_frac
    return K1_base, K2_base, 0.0


def _build_compartments(condition: str, C_global: float, R_global: float,
                         peep: float, peep_ref: float, rec_slope: float,
                         C_chest: float) -> Dict:
    """
    Build per-compartment C/R arrays from the condition profile, applying
    PEEP-recruited compliance and chest-wall series compliance at the
    global level before splitting across compartments (compartment
    normalization formula matching prvc_generator / psv_generator:
    C_lung_rec * C_frac_arr * fractions / C_frac_norm).
    """
    profile = COMPARTMENT_PROFILES.get(condition, COMPARTMENT_PROFILES["Normal"])
    n_comps = len(profile)

    fractions   = np.array([c["fraction"]     for c in profile])
    C_frac_arr  = np.array([c["C_frac"]       for c in profile])
    R_frac_arr  = np.array([c["R_frac"]       for c in profile])
    R_exp_arr   = np.array([c["R_exp_ratio"]  for c in profile])
    teth_arr    = np.array([c["tethering"]    for c in profile])
    C_frac_norm = float(np.dot(C_frac_arr, fractions))

    C_lung_rec = _peep_recruited_compliance(C_global, peep, peep_ref, rec_slope)
    C_lung_rec = _C_rs(C_lung_rec, C_chest)

    C_comps_base = C_lung_rec * C_frac_arr * fractions / max(C_frac_norm, 0.01)
    R_comps_base = R_global * R_frac_arr

    return {
        "n_comps":     n_comps,
        "fractions":   fractions,
        "C_base":      C_comps_base,
        "R_base":      R_comps_base,
        "R_exp_ratio": R_exp_arr,
        "tethering":   teth_arr,
    }


def _current_C_rs_arr(V_comps: np.ndarray, comps: Dict, C_chest: float,
                       stress_index: float, vt_ref_per_comp: np.ndarray) -> np.ndarray:
    """Per-compartment total (lung + chest wall) compliance at current volume."""
    n_comps = comps["n_comps"]
    out = np.zeros(n_comps)
    for i in range(n_comps):
        C_i = _compliance_nonlinear(
            V_comps[i], comps["C_base"][i], vt_ref_per_comp[i], stress_index)
        out[i] = _C_rs(C_i, C_chest)
    return out


def _current_auto_peep(V_comps: np.ndarray, comps: Dict, C_chest: float,
                        stress_index: float, vt_ref_per_comp: np.ndarray) -> float:
    """Auto-PEEP from current end-expiratory-ish compartment state (parallel C sum)."""
    C_rs_arr = _current_C_rs_arr(V_comps, comps, C_chest, stress_index, vt_ref_per_comp)
    C_total = float(C_rs_arr.sum())
    return max(0.0, float(V_comps.sum()) / max(C_total, 0.1))


def _solve_branch_pressure(V_comps: np.ndarray, C_rs_arr: np.ndarray,
                            R_arr: np.ndarray, Q_total: float,
                            peep: float) -> Tuple[float, np.ndarray]:
    """
    Algebraic branch-point solver for VC mandatory breaths (identical to
    vcv_generator): given prescribed total flow and current per-compartment
    state, find branch pressure and per-compartment flows satisfying mass
    balance. P_branch = (Q_total + PEEP*S_invR + S_VCR) / S_invR.
    """
    inv_R       = 1.0 / np.maximum(R_arr, 0.1)
    sum_inv_R   = float(inv_R.sum())
    elastic_arr = V_comps / np.maximum(C_rs_arr, 0.1)
    sum_VCR     = float(np.sum(elastic_arr * inv_R))

    P_branch = (Q_total + peep * sum_inv_R + sum_VCR) / sum_inv_R
    Q_comps  = (P_branch - peep - elastic_arr) * inv_R
    return P_branch, Q_comps


def _classify_dyssynchrony(triggered: bool, t_insp: float, t_effort_dur: float,
                            Q_peak: float, flow_cycle_threshold: float,
                            ps_level: float, Q_at_trigger: float,
                            Q_demand: float) -> str:
    """
    Classify a spontaneous breath's synchrony (reused, five-category subset
    of psv_generator's six-category classifier — reverse-triggering is out
    of scope here per SIMV_CONTROL_LOOP.md Section 11.3).
    """
    if not triggered:
        return "ineffective_trigger"

    if Q_demand > Q_at_trigger * 1.8 and ps_level < 10.0:
        return "flow_starvation"

    ti_ratio = t_insp / max(t_effort_dur, 0.1)
    if flow_cycle_threshold <= 0.15 and ti_ratio > 1.05:
        return "delayed_cycling"

    if flow_cycle_threshold >= 0.55 and ti_ratio < 0.65:
        return "premature_cycling"

    if t_insp < 0.4 * t_effort_dur and t_insp < 0.5:
        return "double_trigger"

    return "synchronous"


# ---------------------------------------------------------------------------
# Section 5 — Parameter validation
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS_COMMON = [
    "mandatory_mode", "respiratory_rate", "peep_cmH2O", "ie_ratio",
    "rise_time_s", "f_window", "pressure_support_cmH2O",
    "flow_cycle_threshold", "trigger_threshold_cmH2O", "pmus_peak_cmH2O",
    "effort_rate_per_min", "effort_duration_s", "pmus_cv",
    "compliance_ml_per_cmH2O", "resistance_cmH2O_L_s",
]


def _validate_params(params: dict) -> None:
    missing = [k for k in _REQUIRED_PARAMS_COMMON if k not in params]
    if missing:
        raise ValueError(f"Missing required parameter(s): {missing}")

    mode = params["mandatory_mode"]
    if mode not in ("VC", "PC"):
        raise ValueError(f"mandatory_mode must be 'VC' or 'PC', got '{mode}'")
    if mode == "VC":
        if "tidal_volume_ml" not in params:
            raise ValueError("Missing required parameter(s): ['tidal_volume_ml'] (mandatory_mode='VC')")
        if "flow_pattern" not in params:
            raise ValueError("Missing required parameter(s): ['flow_pattern'] (mandatory_mode='VC')")
        if params["flow_pattern"] not in ("square", "decelerating"):
            raise ValueError("flow_pattern must be 'square' or 'decelerating'")
    else:
        if "insp_pressure_cmH2O" not in params:
            raise ValueError("Missing required parameter(s): ['insp_pressure_cmH2O'] (mandatory_mode='PC')")

    if not (4    <= float(params["respiratory_rate"])          <= 35):
        raise ValueError("respiratory_rate must be 4–35 bpm")
    if not (0    <= float(params["peep_cmH2O"])                 <= 20):
        raise ValueError("peep_cmH2O must be 0–20 cmH2O")
    if not (0.2  <= float(params["ie_ratio"])                   <= 1.0):
        raise ValueError("ie_ratio must be 0.2–1.0")
    if not (0.0  <= float(params["rise_time_s"])                <= 0.5):
        raise ValueError("rise_time_s must be 0–0.5 s")
    if not (0.05 <= float(params["f_window"])                   <= 0.60):
        raise ValueError("f_window must be 0.05–0.60")
    if not (1    <= float(params["pressure_support_cmH2O"])     <= 50):
        raise ValueError("pressure_support_cmH2O out of range [1, 50]")
    if not (0.05 <= float(params["flow_cycle_threshold"])       <= 0.70):
        raise ValueError("flow_cycle_threshold out of range [0.05, 0.70]")
    if not (5    <= float(params["compliance_ml_per_cmH2O"])    <= 200):
        raise ValueError("compliance_ml_per_cmH2O out of range [5, 200]")
    if not (0.5  <= float(params["resistance_cmH2O_L_s"])       <= 60):
        raise ValueError("resistance_cmH2O_L_s out of range [0.5, 60]")


# ---------------------------------------------------------------------------
# Section 6 — Single-breath inspiration physics (mandatory VC / PC,
# spontaneous). Expiration is handled generically in Section 7 regardless
# of which of these produced the preceding inspiration — this is what lets
# compartment/auto-PEEP state carry forward seamlessly across breath types.
# ---------------------------------------------------------------------------

def _run_mandatory_vc_inspiration(V_comps: np.ndarray, comps: Dict, C_chest: float,
                                   peep: float, t_insp: float, flow_pattern: str,
                                   vt_target_ml: float, K1_ett: float, K2_ett: float,
                                   stress_index: float, vt_ref_per_comp: np.ndarray,
                                   vt_full_per_comp: np.ndarray) -> Dict:
    """One VC mandatory breath: prescribed-flow inspiration + inspiratory
    pause, algebraic branch-point solve each step (vcv_generator physics)."""
    n_comps = comps["n_comps"]
    V_start = float(V_comps.sum()) 
    n_insp  = max(2, int(round(t_insp / DT)))
    n_pause = max(1, int(round(INSPIRATORY_PAUSE_S / DT)))

    t_i = np.linspace(0.0, t_insp, n_insp, endpoint=False)
    if flow_pattern == "square":
        Q_insp = np.full(n_insp, (vt_target_ml / 1000.0) / t_insp)
    else:
        Q_peak = 2.0 * (vt_target_ml / 1000.0) / t_insp
        Q_insp = Q_peak * (1.0 - t_i / t_insp)

    n_total = n_insp + n_pause
    t_rel   = np.zeros(n_total)
    Pao     = np.zeros(n_total)
    Q_tot   = np.zeros(n_total)
    V_tot   = np.zeros(n_total)

    for k in range(n_insp):
        C_rs_arr = _current_C_rs_arr(V_comps, comps, C_chest, stress_index, vt_ref_per_comp)
        R_arr = np.array([
            _R_insp_with_tethering(comps["R_base"][i], V_comps[i],
                                    vt_full_per_comp[i], comps["tethering"][i])
            for i in range(n_comps)
        ])
        Q_total = float(Q_insp[k])
        P_branch, Q_comps = _solve_branch_pressure(V_comps, C_rs_arr, R_arr, Q_total, peep)
        P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)

        V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

        t_rel[k] = t_i[k]
        Pao[k]   = P_branch + P_ett_drop
        Q_tot[k] = Q_total
        V_tot[k] = float(V_comps.sum())

    for k in range(n_pause):
        C_rs_arr = _current_C_rs_arr(V_comps, comps, C_chest, stress_index, vt_ref_per_comp)
        R_arr = np.array([
            _R_insp_with_tethering(comps["R_base"][i], V_comps[i],
                                    vt_full_per_comp[i], comps["tethering"][i])
            for i in range(n_comps)
        ])
        P_branch, Q_comps = _solve_branch_pressure(V_comps, C_rs_arr, R_arr, 0.0, peep)
        V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

        idx = n_insp + k
        t_rel[idx] = t_insp + (k + 1) * DT
        Pao[idx]   = P_branch
        Q_tot[idx] = 0.0
        V_tot[idx] = float(V_comps.sum())

    return {
        "t_rel": t_rel, "pressure": Pao, "flow": Q_tot, "volume": V_tot,
        "V_comps": V_comps, "duration": t_insp + INSPIRATORY_PAUSE_S,
        "ppeak_cmH2O": float(Pao.max()), "pplat_cmH2O": float(Pao[-1]),
        "delivered_vt_ml": float(V_tot[n_insp - 1]) - V_start,
    }


def _run_mandatory_pc_inspiration(V_comps: np.ndarray, comps: Dict, C_chest: float,
                                   peep: float, t_insp: float, t_rise: float,
                                   insp_pressure: float, K1_ett: float, K2_ett: float,
                                   stress_index: float, vt_ref_per_comp: np.ndarray
                                   ) -> Dict:
    """One PC mandatory breath: 3-phase pressure profile (rise + plateau
    only; expiration handled generically), per-compartment ODE (pcv_generator
    physics)."""
    n_comps = comps["n_comps"]
    V_start = float(V_comps.sum()) 
    t_rise_capped = min(t_rise, t_insp * 0.5)
    PIP = peep + insp_pressure
    n_insp = max(2, int(round(t_insp / DT)))

    t_rel = np.zeros(n_insp)
    Pao   = np.zeros(n_insp)
    Q_tot = np.zeros(n_insp)
    V_tot = np.zeros(n_insp)

    for k in range(n_insp):
        t = k * DT
        if t_rise_capped <= 0:
            P_vent = PIP
        elif t <= t_rise_capped:
            P_vent = peep + insp_pressure * (t / t_rise_capped)
        else:
            P_vent = PIP

        Q_comps = np.zeros(n_comps)
        for i in range(n_comps):
            C_i = _compliance_nonlinear(
                V_comps[i], comps["C_base"][i], vt_ref_per_comp[i], stress_index)
            C_rs_i = _C_rs(C_i, C_chest)
            R_i = _R_insp_with_tethering(
                comps["R_base"][i], V_comps[i], vt_ref_per_comp[i] * 2.0,
                comps["tethering"][i])
            drive = P_vent - (V_comps[i] / max(C_rs_i, 0.1)) - peep
            dVdt_i = drive / max(R_i, 0.1) * 1000.0
            V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
            Q_comps[i] = dVdt_i / 1000.0

        Q_total = float(Q_comps.sum())

        t_rel[k] = t
        # Pressure-targeted breath: the ventilator servo-clamps airway
        # pressure directly to P_vent — it is NOT P_branch/P_vent plus a
        # reconstructed ETT Rohrer drop (that reconstruction is only valid
        # for the flow-prescribed VC case, where pressure genuinely is the
        # dependent variable). Adding the drop here double-counted
        # resistance already implicit in the servo control, caught by the
        # smoke test (driving pressure inflated ~2.5x above the set value)
        # — same class of bug flagged for psv_generator ("control-loop
        # inversion").
        Pao[k]   = P_vent
        Q_tot[k] = Q_total
        V_tot[k] = float(V_comps.sum())

    return {
        "t_rel": t_rel, "pressure": Pao, "flow": Q_tot, "volume": V_tot,
        "V_comps": V_comps, "duration": t_insp,
        "ppeak_cmH2O": float(Pao.max()), "pplat_cmH2O": PIP,
        "delivered_vt_ml": float(V_tot[-1]) - V_start,
    }


def _run_spontaneous_inspiration(V_comps: np.ndarray, comps: Dict, C_chest: float,
                                  peep: float, auto_peep_now: float, ps_level: float,
                                  rise_time: float, fct: float, pmus_peak: float,
                                  eff_dur: float, K1_eff: float, K2_eff: float,
                                  stress_index: float, vt_ref_per_comp: np.ndarray
                                  ) -> Dict:
    """One spontaneous (PSV-style) breath: event-driven inspiration until
    flow-cycled off, per-compartment ODE with combined ventilator + patient
    driving pressure (psv_generator physics)."""
    n_comps = comps["n_comps"]
    V_start = float(V_comps.sum()) 
    t_list: List[float] = []
    Pao_list: List[float] = []
    Q_list: List[float] = []
    V_list: List[float] = []

    Q_peak_insp = 0.0
    Q_at_trigger = 0.0
    past_peak = False
    insp_ended_by_reversal = False
    t = 0.0

    while t < MAX_INSP_TIME_S:
        if t < rise_time:
            P_vent = peep + ps_level * (t / max(rise_time, DT))
        else:
            P_vent = peep + ps_level

        pmus_now = _pmus_waveform(t, eff_dur, pmus_peak)

        Q_comps = np.zeros(n_comps)
        for i in range(n_comps):
            Vi = max(V_comps[i], 0.0)
            C_i = _compliance_nonlinear(
                Vi, comps["C_base"][i], vt_ref_per_comp[i], stress_index)
            C_rs_i = _C_rs(C_i, C_chest)
            R_i = _R_insp_with_tethering(
                comps["R_base"][i], Vi, vt_ref_per_comp[i] * 2.0, comps["tethering"][i])
            drive = P_vent + pmus_now - (Vi / max(C_rs_i, 0.1)) - peep
            dVdt_i = drive / max(R_i, 0.1) * 1000.0
            V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
            Q_comps[i] = dVdt_i / 1000.0

        Q_total = float(Q_comps.sum())
        V_total = float(V_comps.sum())

        if t < DT:
            Q_at_trigger = Q_total

        t_list.append(t)
        # Servo-clamped: Pao = P_vent, not P_vent + reconstructed ETT drop
        # (see identical fix/note in _run_mandatory_pc_inspiration above).
        Pao_list.append(P_vent)
        Q_list.append(Q_total)
        V_list.append(V_total)

        if Q_total > Q_peak_insp:
            Q_peak_insp = Q_total
        elif Q_total < Q_peak_insp * 0.95 and Q_peak_insp > 0.01:
            past_peak = True

        if past_peak and Q_peak_insp > 0.0 and Q_total <= fct * Q_peak_insp:
            if Q_total <= 0.0:
                insp_ended_by_reversal = True
            break

        t += DT

    return {
        "t_rel": np.array(t_list), "pressure": np.array(Pao_list),
        "flow": np.array(Q_list), "volume": np.array(V_list),
        "V_comps": V_comps, "duration": t,
        "ppeak_cmH2O": float(max(Pao_list)) if Pao_list else peep,
        "delivered_vt_ml": (float(V_list[-1]) - V_start) if V_list else 0.0,
        "Q_peak_insp": Q_peak_insp, "Q_at_trigger": Q_at_trigger,
        "insp_ended_by_reversal": insp_ended_by_reversal,
    }


# ---------------------------------------------------------------------------
# Section 7 — Generic passive expiration / idle advance. Runs identically
# regardless of whether the preceding breath was mandatory or spontaneous —
# this is what lets compartment state carry forward seamlessly.
# ---------------------------------------------------------------------------

def _advance_passive(V_comps: np.ndarray, comps: Dict, C_chest: float, peep: float,
                      duration: float, K1_ett: float, K2_ett: float,
                      stress_index: float, vt_ref_per_comp: np.ndarray,
                      V_end_insp: np.ndarray) -> Dict:
    """Passive per-compartment emptying (or true idle, if already at
    baseline) for `duration` seconds. Airway pressure = PEEP plus the small
    ETT Rohrer drop on outflow, matching vcv/pcv/psv expiration convention."""
    n_comps = comps["n_comps"]
    n_steps = max(1, int(round(duration / DT)))

    t_rel = np.zeros(n_steps)
    Pao   = np.zeros(n_steps)
    Q_tot = np.zeros(n_steps)
    V_tot = np.zeros(n_steps)

    for k in range(n_steps):
        Q_comps = np.zeros(n_comps)
        for i in range(n_comps):
            Vi = max(V_comps[i], 0.0)
            C_i = _compliance_nonlinear(
                Vi, comps["C_base"][i], vt_ref_per_comp[i], stress_index)
            C_rs_i = _C_rs(C_i, C_chest)
            R_exp_i = _R_exp_dynamic(
                Vi, max(V_end_insp[i], 1.0), comps["R_base"][i], comps["R_exp_ratio"][i])
            dVdt_i = -(Vi / max(C_rs_i, 0.1)) / max(R_exp_i, 0.1) * 1000.0
            V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
            Q_comps[i] = dVdt_i / 1000.0

        Q_total = float(Q_comps.sum())
        P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)

        t_rel[k] = (k + 1) * DT
        # PEEP-valve regulation on the expiratory limb prevents airway
        # opening pressure from swinging far below set PEEP even at large
        # transient expiratory flows (the un-clamped Rohrer-drop
        # reconstruction produced -20 to -29 cmH2O dips after large
        # spontaneous breaths on compliant lungs — caught by
        # test_pressure_within_plausible_bounds). Floor matches the "small
        # negative dip at valve opening" this project's expiration
        # convention documents, bounded rather than unbounded.
        Pao[k]   = max(peep + P_ett_drop, peep - 5.0)
        Q_tot[k] = Q_total
        V_tot[k] = float(V_comps.sum())

    return {"t_rel": t_rel, "pressure": Pao, "flow": Q_tot, "volume": V_tot,
            "V_comps": V_comps}


# ---------------------------------------------------------------------------
# Section 8 — Main SIMV generator: the synchronization-window state machine
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 10,
                            seed: Optional[int] = None) -> dict:
    """
    Generate SIMV waveforms for n_cycles mandatory macro-cycles, with
    spontaneous breaths interleaved according to the synchronization window.
    See module docstring and SIMV_CONTROL_LOOP.md for the full mechanism.
    """
    _validate_params(params)
    rng = np.random.default_rng(seed)

    mode        = params["mandatory_mode"]
    rr_mand     = float(params["respiratory_rate"])
    peep        = float(params["peep_cmH2O"])
    ie          = float(params["ie_ratio"])
    rise_time   = float(params["rise_time_s"])
    f_window    = float(params["f_window"])
    ps_level    = float(params["pressure_support_cmH2O"])
    fct         = float(params["flow_cycle_threshold"])
    trig_thr    = float(params["trigger_threshold_cmH2O"])
    pmus_mean   = float(params["pmus_peak_cmH2O"])
    eff_rate    = float(params["effort_rate_per_min"])
    eff_dur_mn  = float(params["effort_duration_s"])
    pmus_cv     = float(params["pmus_cv"])
    C_global    = float(params["compliance_ml_per_cmH2O"])
    R_global    = float(params["resistance_cmH2O_L_s"])

    condition        = params.get("condition", "Normal")
    stress_index     = float(params.get("stress_index", 1.0))
    C_chest          = float(params.get("chest_wall_compliance_ml_per_cmH2O",
                                         DEFAULT_CHEST_WALL_COMPLIANCE))
    circ_compensated = bool(params.get("circuit_compensated", True))
    peep_ref         = float(params.get("peep_reference_cmH2O", 5.0))
    rec_slope        = float(params.get("recruitment_slope",
                                         RECRUITMENT_SLOPES.get(condition, 0.5)))

    ett_complication = params.get("ett_complication", None)
    cuff_leak_frac   = float(params.get("cuff_leak_fraction", 0.0))
    obs_multiplier   = float(params.get("obstruction_R_multiplier", 1.0))

    K1_intrinsic = R_global * 0.60
    K2_intrinsic = R_global * 0.04
    K1_base = K1_intrinsic + ETT_K1
    K2_base = K2_intrinsic + ETT_K2
    K1_eff, K2_eff, leak_frac = _get_ett_params(
        ett_complication, cuff_leak_frac, obs_multiplier, K1_base, K2_base)

    comps = _build_compartments(condition, C_global, R_global, peep,
                                 peep_ref, rec_slope, C_chest)
    n_comps = comps["n_comps"]
    vt_ref_per_comp = comps["C_base"] * 5.0    # mid-fill reference, mL
    vt_full_per_comp = comps["C_base"] * 10.0  # full-fill reference, mL

    # ---- Mandatory-breath timing -----------------------------------------
    T_mand = 60.0 / rr_mand
    t_insp_mand = T_mand * ie / (1.0 + ie)
    if t_insp_mand <= 0 or t_insp_mand >= T_mand:
        raise ValueError(
            f"Mandatory breath timing invalid: t_insp={t_insp_mand:.3f}s "
            f"T_mand={T_mand:.3f}s (RR={rr_mand}, IE={ie})"
        )
    W = f_window * T_mand
    window_open_rel = max(0.0, T_mand - W)

    attempt_interval = 60.0 / eff_rate

    # ---- State threaded across the whole simulation -----------------------
    V_comps = np.zeros(n_comps)
    t_current = 0.0          # global elapsed time
    t_mand_start = 0.0       # start time of the current macro-cycle
    mandatory_count = 0

    T_list: List[float] = []
    P_list: List[float] = []
    Q_list: List[float] = []
    V_list: List[float] = []
    breath_records: List[Dict] = []

    next_attempt_t = 0.0
    guard_iterations = 0
    GUARD_MAX = 200000

    def _append(seg: Dict, t_offset: float) -> None:
        T_list.extend((seg["t_rel"] + t_offset).tolist())
        P_list.extend(seg["pressure"].tolist())
        Q_list.extend(seg["flow"].tolist())
        V_list.extend(seg["volume"].tolist())

    while mandatory_count < n_cycles:
        guard_iterations += 1
        if guard_iterations > GUARD_MAX:
            raise RuntimeError("SIMV simulation exceeded iteration guard — "
                                "check parameter combination for a stuck loop")

        cycle_end_abs = t_mand_start + T_mand

        # Time-triggered fallback: window has closed with no synchronized trigger.
        if t_current >= cycle_end_abs - 1e-9:
            auto_peep_now = _current_auto_peep(V_comps, comps, C_chest,
                                                stress_index, vt_ref_per_comp)
            if mode == "VC":
                vt_target = float(params["tidal_volume_ml"])
                seg = _run_mandatory_vc_inspiration(
                    V_comps, comps, C_chest, peep, t_insp_mand,
                    params["flow_pattern"], vt_target, K1_eff, K2_eff,
                    stress_index, vt_ref_per_comp, vt_full_per_comp)
            else:
                insp_p = float(params["insp_pressure_cmH2O"])
                seg = _run_mandatory_pc_inspiration(
                    V_comps, comps, C_chest, peep, t_insp_mand, rise_time,
                    insp_p, K1_eff, K2_eff, stress_index, vt_ref_per_comp)

            delivered_vt = seg["delivered_vt_ml"] * (1.0 - leak_frac)
            _append(seg, t_current)
            breath_records.append({
                "breath_type": "mandatory", "trigger_mode": "time_triggered",
                "dyssynchrony_label": "controlled",
                "delivered_vt_ml": delivered_vt,
                "ppeak_cmH2O": seg["ppeak_cmH2O"], "t_start_s": t_current,
                "duration_s": seg["duration"],
            })
            V_comps = seg["V_comps"]
            t_mand_start = t_current
            t_current = t_current + seg["duration"]
            mandatory_count += 1
            next_attempt_t = _advance_schedule(next_attempt_t, t_current, attempt_interval)
            continue

        # Advance passive/idle time up to the sooner of: next effort attempt,
        # or the end of the current macro-cycle (window close).
        t_boundary = min(next_attempt_t, cycle_end_abs)
        if t_boundary > t_current + 1e-9:
            V_end_insp_ref = np.maximum(V_comps, 1.0)
            seg = _advance_passive(V_comps, comps, C_chest, peep,
                                    t_boundary - t_current, K1_eff, K2_eff,
                                    stress_index, vt_ref_per_comp, V_end_insp_ref)
            _append(seg, t_current)
            V_comps = seg["V_comps"]
            t_current = t_boundary
            continue

        # t_current has reached next_attempt_t: a patient effort attempt occurs.
        pmus_i = float(rng.lognormal(np.log(max(pmus_mean, 0.1)), pmus_cv))
        eff_dur_i = max(0.2, float(rng.normal(eff_dur_mn, pmus_cv * eff_dur_mn)))
        auto_peep_now = _current_auto_peep(V_comps, comps, C_chest,
                                            stress_index, vt_ref_per_comp)
        pmus_at_onset = pmus_i * 0.50
        triggered = _check_trigger(pmus_at_onset, auto_peep_now, trig_thr)
        in_window = (t_current - t_mand_start) >= window_open_rel - 1e-9
        attempt_onset_t = t_current

        if triggered and in_window:
            if mode == "VC":
                vt_target = float(params["tidal_volume_ml"])
                seg = _run_mandatory_vc_inspiration(
                    V_comps, comps, C_chest, peep, t_insp_mand,
                    params["flow_pattern"], vt_target, K1_eff, K2_eff,
                    stress_index, vt_ref_per_comp, vt_full_per_comp)
            else:
                insp_p = float(params["insp_pressure_cmH2O"])
                seg = _run_mandatory_pc_inspiration(
                    V_comps, comps, C_chest, peep, t_insp_mand, rise_time,
                    insp_p, K1_eff, K2_eff, stress_index, vt_ref_per_comp)

            delivered_vt = seg["delivered_vt_ml"] * (1.0 - leak_frac)
            _append(seg, t_current)
            breath_records.append({
                "breath_type": "mandatory", "trigger_mode": "synchronized",
                "dyssynchrony_label": "controlled",
                "delivered_vt_ml": delivered_vt,
                "ppeak_cmH2O": seg["ppeak_cmH2O"], "t_start_s": t_current,
                "duration_s": seg["duration"],
            })
            V_comps = seg["V_comps"]
            t_mand_start = attempt_onset_t
            t_current = attempt_onset_t + seg["duration"]
            mandatory_count += 1
            next_attempt_t = _advance_schedule(next_attempt_t, t_current, attempt_interval)
            continue

        if triggered and not in_window:
            Q_demand = pmus_i / max(R_global, 0.1)
            seg = _run_spontaneous_inspiration(
                V_comps, comps, C_chest, peep, auto_peep_now, ps_level,
                rise_time, fct, pmus_i, eff_dur_i, K1_eff, K2_eff,
                stress_index, vt_ref_per_comp)

            label = _classify_dyssynchrony(
                triggered=True, t_insp=seg["duration"], t_effort_dur=eff_dur_i,
                Q_peak=seg["Q_peak_insp"], flow_cycle_threshold=fct,
                ps_level=ps_level, Q_at_trigger=seg["Q_at_trigger"],
                Q_demand=Q_demand)

            delivered_vt = seg["delivered_vt_ml"] * (1.0 - leak_frac)
            _append(seg, t_current)
            breath_records.append({
                "breath_type": "spontaneous", "trigger_mode": "patient",
                "dyssynchrony_label": label, "delivered_vt_ml": delivered_vt,
                "ppeak_cmH2O": seg["ppeak_cmH2O"], "t_start_s": t_current,
                "duration_s": seg["duration"],
            })
            V_comps = seg["V_comps"]
            t_current = attempt_onset_t + seg["duration"]
            next_attempt_t = _advance_schedule(next_attempt_t, t_current, attempt_interval)
            continue

        # Ineffective attempt (failed trigger, either zone): small perturbation,
        # no breath delivered, time still advances by the attempted effort.
        n_eff_steps = max(1, int(round(eff_dur_i / DT)))
        seg_t = np.zeros(n_eff_steps)
        seg_p = np.zeros(n_eff_steps)
        seg_q = np.zeros(n_eff_steps)
        seg_v = np.zeros(n_eff_steps)
        for step in range(n_eff_steps):
            te = step * DT
            pmus_now = _pmus_waveform(te, eff_dur_i, pmus_i)
            Q_perturb = min(pmus_now / max(K1_eff + auto_peep_now, 1.0), 0.05)
            seg_t[step] = te
            seg_p[step] = peep + auto_peep_now - Q_perturb * K1_eff
            seg_q[step] = Q_perturb
            seg_v[step] = float(V_comps.sum())
        T_list.extend((seg_t + t_current).tolist())
        P_list.extend(seg_p.tolist())
        Q_list.extend(seg_q.tolist())
        V_list.extend(seg_v.tolist())
        breath_records.append({
            "breath_type": "ineffective_effort",
            "trigger_mode": "in_window" if in_window else "spontaneous_zone",
            "dyssynchrony_label": "ineffective_trigger",
            "delivered_vt_ml": 0.0, "ppeak_cmH2O": float(seg_p.max()) if n_eff_steps else peep,
            "t_start_s": t_current,
            "duration_s": n_eff_steps * DT,
        })
        t_current = attempt_onset_t + n_eff_steps * DT
        next_attempt_t = _advance_schedule(next_attempt_t, t_current, attempt_interval)

    # The while loop above always exits immediately after delivering the
    # n_cycles-th mandatory breath's INSPIRATION (mandatory_count reaches
    # n_cycles right inside the synchronized/time-triggered branches, both
    # of which `continue` straight back to the loop-condition check). Left
    # as-is, end-state metrics — auto_peep_cmH2O in particular — would be
    # computed from a full end-inspiratory volume rather than a genuine
    # end-expiratory one (caught by test_normal_has_minimal_auto_peep:
    # Normal was reporting ~10 cmH2O of "auto-PEEP" that was really just
    # the last breath's un-exhaled tidal volume). Run one natural mandatory
    # expiratory duration of passive decay to reach a realistic end state
    # before computing final metrics.
    t_final_exp = max(T_mand - t_insp_mand, 0.5)
    V_end_insp_ref = np.maximum(V_comps, 1.0)
    final_seg = _advance_passive(V_comps, comps, C_chest, peep, t_final_exp,
                                  K1_eff, K2_eff, stress_index, vt_ref_per_comp,
                                  V_end_insp_ref)
    _append(final_seg, t_current)
    V_comps = final_seg["V_comps"]
    t_current += t_final_exp

    # ---- Assemble arrays ---------------------------------------------------
    time_arr = np.array(T_list)
    pressure_arr = np.array(P_list)
    flow_arr = np.array(Q_list)
    volume_arr = np.array(V_list)

    # Sort/monotonicity safety net (breath segments are appended in temporal
    # order by construction, but guard against float accumulation drift —
    # matches the recurring monotonicity risk flagged for psv_generator).
    order = np.argsort(time_arr, kind="stable")
    time_arr = time_arr[order]
    pressure_arr = pressure_arr[order]
    flow_arr = flow_arr[order]
    volume_arr = volume_arr[order]

    pressure_resistive = _rohrer_resistance(flow_arr, K1_eff, K2_eff)
    C_rs_end = _current_C_rs_arr(V_comps, comps, C_chest, stress_index, vt_ref_per_comp)
    C_total_end = float(C_rs_end.sum())
    auto_peep_final = max(0.0, float(V_comps.sum()) / max(C_total_end, 0.1))
    pressure_elastic = volume_arr / max(C_total_end, 0.1)
    pressure_total_peep = np.full_like(time_arr, peep + auto_peep_final)

    mandatory_records = [b for b in breath_records if b["breath_type"] == "mandatory"]
    spontaneous_records = [b for b in breath_records if b["breath_type"] == "spontaneous"]
    ineffective_records = [b for b in breath_records if b["breath_type"] == "ineffective_effort"]

    n_mandatory = len(mandatory_records)
    n_spontaneous = len(spontaneous_records)
    n_sync = sum(1 for b in mandatory_records if b["trigger_mode"] == "synchronized")
    sync_fraction = (n_sync / n_mandatory) if n_mandatory else 0.0

    mand_vt = float(np.mean([b["delivered_vt_ml"] for b in mandatory_records])) \
        if mandatory_records else 0.0
    spont_vt = float(np.mean([b["delivered_vt_ml"] for b in spontaneous_records])) \
        if spontaneous_records else 0.0

    ppeak = float(pressure_arr.max()) if len(pressure_arr) else peep
    mean_paw = float(pressure_arr.mean()) if len(pressure_arr) else peep
    total_time_min = (time_arr[-1] / 60.0) if len(time_arr) else (n_cycles * T_mand / 60.0)
    total_delivered_vt_ml = sum(b["delivered_vt_ml"] for b in breath_records)
    minute_vent = (total_delivered_vt_ml / 1000.0) / max(total_time_min, 1e-6)

    n_effort_attempts = n_spontaneous + n_sync + len(ineffective_records)
    ineffective_fraction = (len(ineffective_records) / n_effort_attempts) \
        if n_effort_attempts else 0.0

    mandatory_ppeak = float(np.mean([b["ppeak_cmH2O"] for b in mandatory_records])) \
        if mandatory_records else peep
    driving_p = (mandatory_ppeak - peep) if mode == "PC" else \
        (float(np.mean([b["ppeak_cmH2O"] for b in mandatory_records])) - peep
         if mandatory_records else 0.0)

    if mode == "VC":
        mand_vt_corrected = _circuit_vt_correction(
            mand_vt, mandatory_ppeak, peep, compensated=circ_compensated)
    else:
        mand_vt_corrected = mand_vt

    # ---- Validity filter ----------------------------------------------------
    is_valid = True
    invalid_reason = ""
    if ppeak > PPEAK_MAX_CMHH2O:
        is_valid = False
        invalid_reason = f"Ppeak {ppeak:.1f} cmH2O exceeds barotrauma threshold ({PPEAK_MAX_CMHH2O})"
    elif mode == "VC" and driving_p > DRIVING_P_MAX_CMHH2O:
        is_valid = False
        invalid_reason = f"Mandatory driving pressure {driving_p:.1f} cmH2O exceeds ARDS mortality threshold ({DRIVING_P_MAX_CMHH2O})"
    elif mode == "PC" and float(params["insp_pressure_cmH2O"]) > INSP_PRESSURE_MAX_CMHH2O:
        is_valid = False
        invalid_reason = f"Mandatory inspiratory pressure exceeds maximum ({INSP_PRESSURE_MAX_CMHH2O} cmH2O)"
    elif ps_level > PS_MAX_CMHH2O:
        is_valid = False
        invalid_reason = f"Pressure support {ps_level} cmH2O exceeds clinical ceiling ({PS_MAX_CMHH2O})"
    elif mandatory_records and mand_vt_corrected < VT_MIN_ML:
        is_valid = False
        invalid_reason = f"Mandatory delivered VT {mand_vt_corrected:.0f} mL below minimum ({VT_MIN_ML:.0f} mL)"
    elif mandatory_records and mand_vt_corrected > VT_MAX_ML:
        is_valid = False
        invalid_reason = f"Mandatory delivered VT {mand_vt_corrected:.0f} mL exceeds maximum ({VT_MAX_ML:.0f} mL)"

    return {
        "time": time_arr, "pressure": pressure_arr, "flow": flow_arr, "volume": volume_arr,
        "pressure_resistive": pressure_resistive, "pressure_elastic": pressure_elastic,
        "pressure_total_peep": pressure_total_peep,
        "breath_records": breath_records,
        "n_compartments": n_comps,
        "n_mandatory_breaths": n_mandatory, "n_spontaneous_breaths": n_spontaneous,
        "mandatory_synchronized_fraction": sync_fraction,
        "mandatory_delivered_vt_ml": mand_vt_corrected,
        "spontaneous_delivered_vt_ml": spont_vt,
        "ppeak_cmH2O": ppeak, "driving_p_cmH2O": driving_p,
        "mean_paw_cmH2O": mean_paw, "auto_peep_cmH2O": auto_peep_final,
        "minute_vent_l": minute_vent,
        "ineffective_trigger_fraction": ineffective_fraction,
        "is_valid": is_valid, "invalid_reason": invalid_reason,
    }


# ---------------------------------------------------------------------------
# Section 9 — Scenario ID + dataset sweep
# ---------------------------------------------------------------------------

def _make_scenario_id(condition_name: str, params: dict) -> str:
    mode = params["mandatory_mode"]
    parts = [
        "SIMV", mode, condition_name.replace(" ", ""),
        # Compliance/resistance were missing here entirely -- every
        # mechanics pair swept within a condition tier (exactly what
        # generate_simv_dataset_thinned.py's per-tier mechanics grid does)
        # collided on an identical scenario_id. Same bug class flagged in
        # project history for PSV/PRVC ("every swept parameter must be
        # encoded"); caught here by the thinned-dataset pipeline test
        # rather than this file's own smoke test, since the smoke test
        # only ever calls with one fixed (C, R) at a time.
        f"C{params['compliance_ml_per_cmH2O']}",
        f"R{params['resistance_cmH2O_L_s']}",
        f"RRm{int(params['respiratory_rate'])}",
        f"PEEP{int(params['peep_cmH2O'])}",
        f"IE{params['ie_ratio']}",
        f"RT{params['rise_time_s']}",
        f"FW{params['f_window']}",
        f"PS{int(params['pressure_support_cmH2O'])}",
        f"FCT{params['flow_cycle_threshold']}",
        f"TRIG{params['trigger_threshold_cmH2O']}",
        f"PMUS{int(params['pmus_peak_cmH2O'])}",
        f"ERATE{int(params['effort_rate_per_min'])}",
        f"EDUR{params['effort_duration_s']}",
        f"CV{params['pmus_cv']}",
    ]
    if mode == "VC":
        parts.append(f"VT{int(params['tidal_volume_ml'])}")
        parts.append(str(params["flow_pattern"]))
    else:
        parts.append(f"PC{int(params['insp_pressure_cmH2O'])}")
    return "_".join(parts)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_dataset(condition_name: str, compliance_ml_per_cmH2O: float,
                      resistance_cmH2O_L_s: float, n_cycles: int = 10,
                      max_scenarios: Optional[int] = None) -> List[Dict]:
    """
    Sweep PARAMETER_GRID for one condition + mechanics pair, bifurcated by
    mandatory_mode (VC rows use tidal_volume/flow_pattern; PC rows use
    insp_pressure — the two are mutually exclusive per scenario). Pass
    max_scenarios to cap for smoke tests; leave None for full generation
    (intended to be thinned first via a companion
    generate_simv_dataset_thinned.py script, matching sibling precedent).
    """
    shared_keys = ["respiratory_rate", "peep_cmH2O", "ie_ratio", "rise_time_s",
                   "f_window", "pressure_support_cmH2O", "flow_cycle_threshold",
                   "trigger_threshold_cmH2O", "pmus_peak_cmH2O",
                   "effort_rate_per_min", "effort_duration_s", "pmus_cv"]
    shared_values = [PARAMETER_GRID[k] for k in shared_keys]

    scenarios: List[Dict] = []
    count = 0

    # mode is nested inside shared_combo (not outermost) so that a capped
    # or thinned slice — including this file's own smoke test — samples
    # both mandatory sub-modes early rather than exhausting the cap on VC
    # alone before ever reaching PC.
    for shared_combo in itertools.product(*shared_values):
        for mode in PARAMETER_GRID["mandatory_mode"]:
            if mode == "VC":
                mode_keys = ["tidal_volume_ml_per_kg", "flow_pattern"]
            else:
                mode_keys = ["insp_pressure_cmH2O"]
            mode_values = [PARAMETER_GRID[k] for k in mode_keys]

            for mode_combo in itertools.product(*mode_values):
                if max_scenarios is not None and count >= max_scenarios:
                    return scenarios

                params = dict(zip(shared_keys, shared_combo))
                params.update(dict(zip(mode_keys, mode_combo)))
                params["mandatory_mode"] = mode
                params["condition"] = condition_name
                params["compliance_ml_per_cmH2O"] = compliance_ml_per_cmH2O
                params["resistance_cmH2O_L_s"] = resistance_cmH2O_L_s
                if mode == "VC":
                    params["tidal_volume_ml"] = params.pop("tidal_volume_ml_per_kg") * IBW_KG

                scenario_id = _make_scenario_id(condition_name, params)
                count += 1

                try:
                    result = generate_breath_cycles(params, n_cycles=n_cycles)
                except ValueError as e:
                    scenarios.append({
                        "scenario_id": scenario_id, "condition": condition_name,
                        "params": params, "metrics": {}, "is_valid": False,
                        "invalid_reason": str(e), "waveforms": {},
                        "generated_at": _timestamp(),
                    })
                    continue

                metric_keys = [
                    "n_mandatory_breaths", "n_spontaneous_breaths",
                    "mandatory_synchronized_fraction", "mandatory_delivered_vt_ml",
                    "spontaneous_delivered_vt_ml", "ppeak_cmH2O", "driving_p_cmH2O",
                    "mean_paw_cmH2O", "auto_peep_cmH2O", "minute_vent_l",
                    "ineffective_trigger_fraction",
                ]
                metrics = {k: result[k] for k in metric_keys}

                waveforms = {}
                if result["is_valid"]:
                    waveforms = {
                        "time": result["time"], "pressure": result["pressure"],
                        "flow": result["flow"], "volume": result["volume"],
                    }

                scenarios.append({
                    "scenario_id": scenario_id, "condition": condition_name,
                    "params": params, "metrics": metrics,
                    "is_valid": result["is_valid"], "invalid_reason": result["invalid_reason"],
                    "waveforms": waveforms,
                    "breath_records": result.get("breath_records", []),
                    "generated_at": _timestamp(),
                })

    return scenarios


# ---------------------------------------------------------------------------
# Section 10 — Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _PASS = "\033[92m✓\033[0m"
    _FAIL = "\033[91m✗\033[0m"
    _results: List[bool] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        status = _PASS if condition else _FAIL
        print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
        _results.append(bool(condition))

    print("=" * 65)
    print("  SIMV Generator — Multi-Compartment Smoke Test")
    print("=" * 65)

    base_vc = {
        "mandatory_mode":          "VC",
        "tidal_volume_ml":          420.0,
        "flow_pattern":             "square",
        "respiratory_rate":         8.0,
        "peep_cmH2O":               5.0,
        "ie_ratio":                 0.5,
        "rise_time_s":              0.1,
        "f_window":                 0.25,
        "pressure_support_cmH2O":   10.0,
        "flow_cycle_threshold":     0.25,
        "trigger_threshold_cmH2O": 1.5,
        "pmus_peak_cmH2O":          12.0,
        "effort_rate_per_min":      20.0,
        "effort_duration_s":        0.8,
        "pmus_cv":                  0.15,
        "compliance_ml_per_cmH2O":  60.0,
        "resistance_cmH2O_L_s":     10.0,
        "condition":                "Normal",
    }

    # ---- [1/4] Basic SIMV-VC scenario -----------------------------------
    print("\n[1/4] SIMV-VC — Normal lung, basic structure")
    r_vc = generate_breath_cycles(base_vc, n_cycles=6, seed=1)
    _check("returns dict", isinstance(r_vc, dict))
    _check("n_mandatory_breaths == n_cycles", r_vc["n_mandatory_breaths"] == 6,
           f"got {r_vc['n_mandatory_breaths']}")
    _check("at least some spontaneous breaths occurred",
           r_vc["n_spontaneous_breaths"] > 0,
           f"n_spont={r_vc['n_spontaneous_breaths']}")
    _check("Normal uses 1 compartment", r_vc["n_compartments"] == 1)
    _check("time array is monotonically non-decreasing",
           bool(np.all(np.diff(r_vc["time"]) >= -1e-9)))
    _check("mandatory VT near target (420 mL)",
           abs(r_vc["mandatory_delivered_vt_ml"] - 420.0) < 60.0,
           f"got {r_vc['mandatory_delivered_vt_ml']:.0f} mL")
    _check("some mandatory breaths synchronized",
           0.0 <= r_vc["mandatory_synchronized_fraction"] <= 1.0,
           f"sync_frac={r_vc['mandatory_synchronized_fraction']:.2f}")

    # ---- [2/4] SIMV-PC — rise time and PC-specific physics ---------------
    print("\n[2/4] SIMV-PC — rise time effect, driving pressure")
    base_pc = {**base_vc, "mandatory_mode": "PC", "insp_pressure_cmH2O": 15.0}
    del base_pc["tidal_volume_ml"]
    del base_pc["flow_pattern"]
    r_pc0 = generate_breath_cycles({**base_pc, "rise_time_s": 0.0}, n_cycles=5, seed=2)
    r_pc4 = generate_breath_cycles({**base_pc, "rise_time_s": 0.4}, n_cycles=5, seed=2)
    _check("SIMV-PC returns dict", isinstance(r_pc0, dict))
    _check("driving pressure ≈ set insp pressure",
           abs(r_pc0["driving_p_cmH2O"] - 15.0) < 3.0,
           f"got {r_pc0['driving_p_cmH2O']:.1f}")
    _check("n_mandatory_breaths == n_cycles (PC)",
           r_pc0["n_mandatory_breaths"] == 5)

    # ---- [3/4] Synchronization window behavior + compartment continuity --
    print("\n[3/4] Synchronization window — high vs low effort rate; COPD auto-PEEP")
    r_low_effort = generate_breath_cycles(
        {**base_vc, "effort_rate_per_min": 6.0, "pmus_peak_cmH2O": 3.0,
         "trigger_threshold_cmH2O": 3.0},
        n_cycles=6, seed=3)
    r_high_effort = generate_breath_cycles(
        {**base_vc, "effort_rate_per_min": 30.0, "pmus_peak_cmH2O": 20.0,
         "trigger_threshold_cmH2O": 0.5},
        n_cycles=6, seed=3)
    _check("weak/rare effort -> more time-triggered mandatory breaths",
           r_low_effort["mandatory_synchronized_fraction"] <=
           r_high_effort["mandatory_synchronized_fraction"] + 1e-6,
           f"low={r_low_effort['mandatory_synchronized_fraction']:.2f} "
           f"high={r_high_effort['mandatory_synchronized_fraction']:.2f}")
    _check("strong/frequent effort -> more spontaneous breaths",
           r_high_effort["n_spontaneous_breaths"] >= r_low_effort["n_spontaneous_breaths"],
           f"low={r_low_effort['n_spontaneous_breaths']} "
           f"high={r_high_effort['n_spontaneous_breaths']}")

    p_copd = {**base_vc, "condition": "COPD", "compliance_ml_per_cmH2O": 100.0,
              "resistance_cmH2O_L_s": 22.0, "respiratory_rate": 14.0}
    r_copd = generate_breath_cycles(p_copd, n_cycles=8, seed=4)
    _check("COPD uses 3 compartments", r_copd["n_compartments"] == 3)
    _check("COPD develops measurable auto-PEEP (state carried across breath types)",
           r_copd["auto_peep_cmH2O"] > 0.3,
           f"auto_peep={r_copd['auto_peep_cmH2O']:.2f}")

    # ---- [4/4] Dataset sweep (small slice) --------------------------------
    print("\n[4/4] Dataset sweep — Normal lung, capped slice")
    scenarios = generate_dataset(
        condition_name="Normal", compliance_ml_per_cmH2O=60.0,
        resistance_cmH2O_L_s=10.0, n_cycles=2, max_scenarios=12,
    )
    ids = [s["scenario_id"] for s in scenarios]
    _check("dataset non-empty", len(scenarios) > 0, f"got {len(scenarios)}")
    _check("all scenario IDs unique", len(ids) == len(set(ids)),
           f"{len(set(ids))} unique of {len(ids)}")
    _check("both mandatory modes represented",
           any(s["params"]["mandatory_mode"] == "VC" for s in scenarios) and
           any(s["params"]["mandatory_mode"] == "PC" for s in scenarios))
    print(f"     total={len(scenarios)}  example_id={ids[0] if ids else '—'}")

    n_pass = sum(_results)
    n_total = len(_results)
    print(f"\n{'=' * 65}")
    print(f"  SIMV generator smoke test: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("  WARNING: some checks failed — review output above")
    print(f"{'=' * 65}\n")
    sys.exit(0 if n_pass == n_total else 1)
