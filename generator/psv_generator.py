"""
generator/psv_generator.py
--------------------------
PSV (Pressure Support Ventilation) waveform generator.

Control loop
------------
Patient-triggered, pressure-limited, flow-cycled. The ventilator applies
a set pressure support (PS) above PEEP whenever a patient inspiratory effort
crosses the trigger threshold. Inspiration ends when inspiratory flow decays
to a set fraction of its peak value (the flow-cycle threshold). Tidal volume
and breath timing are both patient-dependent and exhibit natural breath-to-
breath variability.

Modified equation of motion
----------------------------
    P_vent(t) + Pmus(t) = V(t)/C_rs  +  Q(t) * R_eff  +  PEEP_total
                           ─────────     ────────────
                           Pel           Pres

Decomposed at each time step:
    Pao(t) = Pres(t) + Pel(t) + PEEP_total

where PEEP_total = PEEPe (set) + PEEPi (auto-PEEP from trapped gas).

Physiological refinements incorporated
---------------------------------------
    1. Multi-compartment lung mechanics  — parallel RC compartments
       per condition (1–3 compartments); COPD uses 3, Pneumonia uses 3,
       ARDS uses 2, others use 1.

    2. Breath-to-breath variability — Pmus amplitude and effort duration
       drawn from log-normal and normal distributions respectively,
       producing physiologically realistic intra-patient variance.

    3. Patient-ventilator dyssynchrony — six subtypes detected and labeled:
       ineffective triggering, double triggering, reverse triggering,
       delayed cycling, premature cycling, flow starvation.

    4. ETT complications — cuff leak (volume loss per breath) and partial
       obstruction (elevated K1/K2 Rohrer coefficients) modeled as
       steady-state parameters.

    5. Spontaneous Breathing Trial dynamics — generate_sbt_sequence()
       produces a multi-phase temporal sequence: baseline full-support →
       trial reduced-support → outcome assessment. Trajectory of RRSB
       (rapid shallow breathing index) determines pass/fail.

    6. Non-linear compliance — power-law compliance per compartment
       parameterized by stress index (SI); SI < 1 = tidal recruitment,
       SI > 1 = overdistension, SI = 1 = linear (default).

    7. Flow-dependent resistance (Rohrer equation) — inspiratory
       resistance = K1*Q + K2*Q*|Q| capturing turbulent pressure
       drop in large airways and ETT.

    8. Volume-dependent expiratory resistance — expiratory R rises as
       lung volume falls, modeling dynamic airway collapse (equal pressure
       point) in obstructive disease.

    9. PEEP-recruited compliance — compliance increases with PEEP as
       previously collapsed alveoli open; slope is condition-specific.

   10. Chest wall compliance — separate C_lung and C_chest combined in
       series: 1/C_rs = 1/C_lung + 1/C_chest.

   11. Circuit compliance correction — post-processing metric correction
       for gas sequestered in compliant ventilator tubing.

   12. Temporal sequencing — generate_sbt_sequence() implements the
       multi-phase scenario architecture for SBT pass/fail trajectories.

Interface contract (identical to vcv_generator and pcv_generator)
------------------------------------------------------------------
    generate_breath_cycles(params, n_cycles, seed) -> dict
    generate_sbt_sequence(params, ...) -> dict
    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list

Output dict keys
----------------
    Core waveforms (np.ndarray, 100 Hz):
        time, pressure, flow, volume

    Pressure decomposition (np.ndarray, same length):
        pressure_resistive, pressure_elastic, pressure_total_peep

    Scalar metrics:
        ppeak_cmH2O, delivered_vt_ml, driving_p_cmH2O, mean_paw_cmH2O,
        auto_peep_cmH2O, total_peep_cmH2O, fill_fraction,
        minute_vent_l, pres_peak_cmH2O, pel_end_insp_cmH2O,
        stress_index, pres_pel_ratio, triggered_breath_rate,
        ineffective_trigger_fraction, patient_vt_ml

    Per-breath labels (list, length = n_cycles):
        breath_dyssynchrony_labels

    Validity:
        is_valid, invalid_reason

Run smoke test:
    python generator/psv_generator.py
"""

import itertools
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Section 1 — Parameter Grids
# ---------------------------------------------------------------------------

# Full parameter space — all clinically plausible combinations
PARAMETER_GRID: Dict = {
    # ---- Ventilator-side ----
    # Pressure support level above PEEP (cmH2O)
    "pressure_support_cmH2O":   [5, 8, 12, 16, 20],
    # End-expiratory pressure (cmH2O)
    "peep_cmH2O":                [0, 4, 8, 12, 16],
    # Pressure ramp time from PEEP to PIP (s); 0.0 = instantaneous
    "rise_time_s":               [0.0, 0.1, 0.2, 0.4],
    # Fraction of peak flow at which inspiration ends
    # 0.10 = delayed cycling, 0.25 = nominal, 0.40 = premature cycling
    "flow_cycle_threshold":      [0.10, 0.25, 0.40],
    # Trigger threshold in pressure-equivalent terms (cmH2O)
    "trigger_threshold_cmH2O":  [0.5, 1.5, 3.0],

    # ---- Patient-side ----
    # Mean peak Pmus (cmH2O); the patient's inspiratory muscle effort amplitude
    "pmus_peak_cmH2O":          [5, 8, 12, 16, 20],
    # Patient's neural respiratory rate (efforts/min); may differ from
    # triggered breath rate when ineffective triggering occurs
    "effort_rate_per_min":       [12, 16, 20, 25, 30],
    # Mean duration of each inspiratory effort (s)
    "effort_duration_s":         [0.5, 0.7, 0.9, 1.1],
    # Coefficient of variation for Pmus amplitude (dimensionless)
    # 0.15 = stable, 0.25 = moderate variability, 0.35 = distress
    "pmus_cv":                   [0.15, 0.25, 0.35],
}

# Thinned grid for systematic dataset generation — reduced to avoid
# combinatorial explosion while preserving clinical coverage
DATASET_GRID: Dict = {
    "pressure_support_cmH2O":   [5, 10, 15, 20],
    "peep_cmH2O":                [0, 5, 10, 15],
    "flow_cycle_threshold":      [0.10, 0.25, 0.40],
    "trigger_threshold_cmH2O":  [0.5, 2.0],
    "rise_time_s":               [0.0, 0.2],
    "pmus_peak_cmH2O":          [5, 10, 15, 20],
    "effort_rate_per_min":       [14, 20, 28],
    "effort_duration_s":         [0.5, 0.8, 1.1],
    "pmus_cv":                   [0.15, 0.30],
}

# ---------------------------------------------------------------------------
# Section 2 — Safety Thresholds and Constants
# ---------------------------------------------------------------------------

IBW_KG: float            = 70.0
VT_MIN_ML: float         = IBW_KG * 3      # 210 mL
VT_MAX_ML: float         = IBW_KG * 12     # 840 mL
PPEAK_MAX_CMHH2O: float  = 50.0
PS_MAX_CMHH2O: float     = 35.0
FILL_FRACTION_MIN: float = 0.10            # PSV can have lower fill fraction than PCV
MAX_INSP_TIME_S: float   = 3.0            # absolute safety limit on inspiratory time
DT: float                = 0.01           # 100 Hz internal simulation timestep

# Rapid shallow breathing index threshold for SBT failure
RRSB_FAILURE_THRESHOLD: float = 105.0     # breaths/min/L (Yang-Tobin)
RR_FAILURE_THRESHOLD: int     = 35        # breaths/min

# Circuit compliance — standard adult ICU circuit
CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5

# Default chest wall compliance (effectively infinite for non-restricted patients)
DEFAULT_CHEST_WALL_COMPLIANCE: float = 250.0  # mL/cmH2O

# Rohrer ETT contribution (7.5 mm ID tube)
ETT_K1: float = 5.0   # cmH2O/L/s  — viscous ETT resistance
ETT_K2: float = 3.0   # cmH2O/(L/s)^2 — turbulent ETT resistance

# ---------------------------------------------------------------------------
# Section 3 — Condition-Specific Profiles
# ---------------------------------------------------------------------------

# Multi-compartment definitions — each entry is a list of compartment dicts.
# fraction: volume fraction (sum to 1.0)
# C_fraction: compliance as multiple of the global C preset
# R_fraction: resistance as multiple of the global R preset
# R_exp_ratio: peak expiratory R / inspiratory R for this compartment
# tethering: inspiratory resistance reduction with volume (0=none, 1=full)
COMPARTMENT_PROFILES: Dict = {
    "Normal": [
        {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
         "R_exp_ratio": 1.2,  "tethering": 0.80},
    ],
    "Mild ARDS": [
        {"fraction": 0.75, "C_frac": 0.90, "R_frac": 1.00,
         "R_exp_ratio": 1.4,  "tethering": 0.40},  # aerated
        {"fraction": 0.25, "C_frac": 0.10, "R_frac": 1.60,
         "R_exp_ratio": 2.0,  "tethering": 0.10},  # recruitable
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
    # COPD: 3 compartments — fast (less obstructed), medium, slow (emphysema)
    "COPD": [
        {"fraction": 0.35, "C_frac": 0.70, "R_frac": 0.55,
         "R_exp_ratio": 4.0,  "tethering": 0.15},  # fast
        {"fraction": 0.40, "C_frac": 1.05, "R_frac": 1.27,
         "R_exp_ratio": 6.0,  "tethering": 0.10},  # medium
        {"fraction": 0.25, "C_frac": 1.40, "R_frac": 2.36,
         "R_exp_ratio": 8.0,  "tethering": 0.05},  # slow
    ],
    "Bronchospasm": [
        {"fraction": 0.60, "C_frac": 0.90, "R_frac": 0.80,
         "R_exp_ratio": 3.0,  "tethering": 0.00},  # less obstructed
        {"fraction": 0.40, "C_frac": 1.10, "R_frac": 1.43,
         "R_exp_ratio": 5.0,  "tethering": 0.00},  # severely obstructed
    ],
    # Pneumonia: 3 compartments — healthy, transitional, consolidated
    "Pneumonia": [
        {"fraction": 0.60, "C_frac": 1.10, "R_frac": 0.83,
         "R_exp_ratio": 1.5,  "tethering": 0.70},  # healthy
        {"fraction": 0.25, "C_frac": 0.55, "R_frac": 1.83,
         "R_exp_ratio": 3.0,  "tethering": 0.30},  # transitional + secretions
        {"fraction": 0.15, "C_frac": 0.07, "R_frac": 6.67,
         "R_exp_ratio": 2.0,  "tethering": 0.10},  # consolidated
    ],
}

# PEEP-recruited compliance slopes (mL/cmH2O of C gained per cmH2O of PEEP
# above reference PEEP of 5 cmH2O)
RECRUITMENT_SLOPES: Dict = {
    "Normal":        0.00,
    "Mild ARDS":     0.50,
    "Moderate ARDS": 0.90,
    "Severe ARDS":   0.60,
    "COPD":          0.00,
    "Bronchospasm":  0.00,
    "Pneumonia":     0.10,
}

# ---------------------------------------------------------------------------
# Section 4 — Physics Functions
# ---------------------------------------------------------------------------

def _pmus_waveform(t_elapsed: float,
                   t_duration: float,
                   pmus_peak: float) -> float:
    """
    Half-sinusoidal Pmus profile for a single inspiratory effort.

    Rises from 0 at effort onset, peaks at the midpoint, returns to 0
    at t_duration. This shape closely matches measured esophageal pressure
    swings during spontaneous breathing (Milic-Emili & Zin 1986).

    Parameters
    ----------
    t_elapsed  : time since effort onset (s)
    t_duration : total effort duration (s)
    pmus_peak  : peak effort amplitude (cmH2O, positive)

    Returns
    -------
    float : instantaneous Pmus (cmH2O)
    """
    if t_elapsed < 0.0 or t_elapsed > t_duration:
        return 0.0
    return pmus_peak * np.sin(np.pi * t_elapsed / t_duration)


def _sample_breath_effort(pmus_mean: float,
                           pmus_cv: float,
                           dur_mean: float,
                           dur_cv: float = 0.15,
                           rng: np.random.Generator = None) -> Tuple[float, float]:
    """
    Sample effort parameters for a single breath from physiologically
    realistic distributions.

    Pmus amplitude follows a log-normal distribution — always positive
    with a realistic right skew (occasional deep breaths more common than
    very shallow ones). Duration follows a normal distribution clipped to
    a physiological minimum.

    Parameters
    ----------
    pmus_mean  : mean peak effort (cmH2O)
    pmus_cv    : coefficient of variation for effort amplitude
    dur_mean   : mean effort duration (s)
    dur_cv     : coefficient of variation for effort duration
    rng        : numpy random Generator (for reproducibility)

    Returns
    -------
    Tuple of (pmus_peak, effort_duration)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Log-normal Pmus — always positive, physiological right skew
    sigma_ln  = float(np.sqrt(np.log(1.0 + pmus_cv ** 2)))
    mu_ln     = float(np.log(pmus_mean) - 0.5 * sigma_ln ** 2)
    pmus_peak = float(rng.lognormal(mu_ln, sigma_ln))
    pmus_peak = float(np.clip(pmus_peak, 1.0, 35.0))

    # Normal effort duration with physiological floor
    effort_dur = float(rng.normal(dur_mean, dur_mean * dur_cv))
    effort_dur = float(np.clip(effort_dur, 0.30, 2.0))

    return pmus_peak, effort_dur


def _rohrer_resistance(Q: float, K1: float, K2: float) -> float:
    """
    Rohrer equation for flow-dependent resistive pressure.

    Pres = K1 * Q + K2 * Q * |Q|

    K1 (cmH2O/L/s)   — viscous (laminar) resistance component
    K2 (cmH2O/(L/s)^2) — turbulent (inertial) resistance component

    The signed formulation preserves flow direction (inspiratory +,
    expiratory -) while ensuring the turbulent term always opposes flow.
    """
    return K1 * Q + K2 * Q * abs(Q)


def _R_insp_with_tethering(R_base: float,
                             V_current: float,
                             V_target: float,
                             tethering: float) -> float:
    """
    Volume-dependent inspiratory resistance via parenchymal tethering.

    In healthy lungs, expanding parenchyma radially dilates small airways
    reducing resistance at higher volumes. In emphysema (COPD) this
    tethering is lost; in bronchospasm, smooth muscle contraction overrides
    it (tethering_factor = 0.0).

    R_eff = R_base * (1 - tethering * 0.30 * V_fraction)
    """
    V_frac = float(np.clip(V_current / max(V_target, 1.0), 0.0, 1.0))
    return R_base * max(1.0 - tethering * 0.30 * V_frac, 0.30)


def _R_exp_dynamic(V_current: float,
                   V_end_insp: float,
                   R_insp: float,
                   R_exp_ratio: float) -> float:
    """
    Volume-dependent expiratory resistance from dynamic airway collapse.

    As lung volume falls during expiration, the equal pressure point moves
    peripherally, compressing progressively longer lengths of small airway.
    Effective expiratory resistance rises from R_insp at end-inspiration
    toward R_insp * R_exp_ratio as lung volume approaches FRC.

    Produces the characteristic biexponential expiratory flow shape that
    distinguishes COPD from all other conditions.
    """
    frac_exhaled = 1.0 - float(
        np.clip(V_current / max(V_end_insp, 1.0), 0.0, 1.0)
    )
    return R_insp * (1.0 + (R_exp_ratio - 1.0) * frac_exhaled)


def _compliance_nonlinear(V_mL: float,
                           C_base: float,
                           V_ref: float,
                           stress_index: float = 1.0) -> float:
    """
    Non-linear (volume-dependent) compliance via power-law approximation.

    Derives from the definition of stress index:
        P(t) ∝ t^SI during constant-flow VCV
    which implies:
        C(V) ∝ C_base * (V / V_ref) ^ (1 - SI)

    SI = 1.0  → constant compliance (linear P-V, straight VCV ramp)
    SI < 1.0  → compliance rises with volume (tidal recruitment, concave-up)
    SI > 1.0  → compliance falls with volume (overdistension, concave-down)

    V_ref is typically the mid-inspiration volume for this compartment.
    """
    if abs(stress_index - 1.0) < 0.01 or V_mL <= 0.0:
        return C_base
    V_norm = max(V_mL / max(V_ref, 1.0), 0.01)
    return float(C_base * (V_norm ** (1.0 - stress_index)))


def _peep_recruited_compliance(C_base: float,
                                peep: float,
                                peep_ref: float,
                                recruitment_slope: float) -> float:
    """
    PEEP-mediated alveolar recruitment increases effective compliance.

    Increasing PEEP above peep_ref opens previously collapsed alveoli,
    adding new compliant volume to the system. This is distinct from
    non-linear compliance (which models already-open alveoli changing
    stiffness) — recruitment changes the *number* of open alveoli.

    recruitment_slope: mL/cmH2O of C gained per cmH2O of PEEP above peep_ref
    Condition-specific: ARDS ~2.5, COPD ~0.2, Normal ~0.3
    """
    delta_peep = max(0.0, peep - peep_ref)
    return C_base + recruitment_slope * delta_peep


def _C_rs(C_lung: float, C_chest: float) -> float:
    """
    Total respiratory system compliance from lung and chest wall in series.

    1/C_rs = 1/C_lung + 1/C_chest

    For normal chest wall (C_chest >> C_lung): C_rs ≈ C_lung.
    For restricted chest wall (morbid obesity, ACS): C_chest becomes
    the dominant mechanical constraint.
    """
    if C_chest >= 9000.0:
        return C_lung
    return 1.0 / (1.0 / max(C_lung, 0.1) + 1.0 / max(C_chest, 0.1))


def _circuit_vt_correction(vt_mL: float,
                             ppeak: float,
                             peep: float,
                             C_circ: float = CIRCUIT_COMPLIANCE_ML_PER_CMH2O,
                             compensated: bool = True) -> float:
    """
    Adjust delivered Vt for gas sequestered in compliant ventilator circuit.

    Modern ICU ventilators (PB980, Hamilton G5, Drager Evita) measure circuit
    compliance during self-test and automatically compensate. Transport
    ventilators and older models typically do not. At Ppeak = 30 cmH2O,
    approximately 62 mL is trapped in a standard adult circuit (C = 2.5 mL/cmH2O).
    """
    if compensated:
        return vt_mL
    circuit_loss = C_circ * (ppeak - peep)
    return max(0.0, vt_mL - circuit_loss)


def _check_trigger(pmus_at_onset: float,
                   auto_peep: float,
                   threshold: float) -> bool:
    """
    Determine whether a patient effort successfully triggers the ventilator.

    The patient must first generate enough pressure to overcome intrinsic
    auto-PEEP before the net effort can exceed the trigger threshold. This
    is the physiological mechanism of ineffective triggering in COPD —
    auto-PEEP raises the effective threshold the patient must overcome.

    effective_drive = Pmus_at_onset - auto_PEEP
    Trigger succeeds if: effective_drive > threshold
    """
    effective_drive = pmus_at_onset - auto_peep
    return effective_drive > threshold


def _check_cycle(Q_current: float,
                 Q_peak: float,
                 threshold: float) -> bool:
    """
    Determine whether inspiration should end (flow-cycling criterion).

    Inspiration terminates when inspiratory flow decays to (threshold * Q_peak).
    Default threshold = 0.25 (25% of peak) for most modern ventilators.

    Low threshold (0.10): delayed cycling — ventilator delivers beyond patient's
    neural Ti, patient exhales against ongoing inspiration.
    High threshold (0.40): premature cycling — ventilator stops before patient's
    neural Ti ends, effort continues into exhalation.
    """
    if Q_current <= 0.0 or Q_peak <= 0.0:
        return True
    return (Q_current / Q_peak) <= threshold


def _decompose_pressure(Q_total: float,
                         V_total_mL: float,
                         C_rs_eff: float,
                         peep_total: float,
                         K1: float,
                         K2: float) -> Tuple[float, float, float]:
    """
    Decompose airway pressure into three physiological components.

    Pao = Pres + Pel + PEEP_total

    Returns
    -------
    Tuple of (Pres, Pel, PEEP_total)

    Pres : resistive pressure — pressure driving gas through airways
    Pel  : elastic pressure   — pressure expanding alveoli against elastic recoil
    PEEP_total : total end-expiratory pressure (PEEPe + PEEPi)

    Verification: Pres + Pel + PEEP_total should equal the computed Pao.
    """
    pres = _rohrer_resistance(Q_total, K1, K2)
    pel  = (V_total_mL / max(C_rs_eff, 0.1))
    return pres, pel, peep_total


# ---------------------------------------------------------------------------
# Section 5 — Dyssynchrony Detection
# ---------------------------------------------------------------------------

def _classify_dyssynchrony(triggered: bool,
                             t_insp: float,
                             t_effort_dur: float,
                             Q_peak: float,
                             flow_cycle_threshold: float,
                             ps_level: float,
                             Q_at_trigger: float,
                             Q_demand: float, insp_ended_by_reversal=False) -> str:
    """
    Classify each breath into one of seven categories.

    Categories
    ----------
    "synchronous"          — well-matched trigger, Ti, and cycling
    "ineffective_trigger"  — effort failed to trigger (Q_demand insufficient)
    "double_trigger"       — second trigger within one expiratory time constant
    "reverse_trigger"      — mechanical breath entrains neural effort during exp.
    "delayed_cycling"      — ventilator Ti >> patient neural Ti (low threshold)
    "premature_cycling"    — ventilator Ti << patient neural Ti (high threshold)
    "flow_starvation"      — set PS insufficient; patient demand > delivered flow

    Parameters
    ----------
    triggered           : did this effort successfully trigger the ventilator?
    t_insp              : actual delivered inspiratory time (s)
    t_effort_dur        : patient's neural effort duration (s)
    Q_peak              : peak inspiratory flow this breath (L/s)
    flow_cycle_threshold: fraction of peak flow at which ventilator cycled
    ps_level            : set pressure support (cmH2O)
    Q_at_trigger        : flow at the moment of trigger (L/s)
    Q_demand            : estimated patient flow demand = Pmus_peak / R_eff
    """
    if not triggered:
        return "ineffective_trigger"

    # Flow starvation: PS insufficient to meet demand at trigger
    # Manifests as a scooped pressure plateau (Pao dips below PIP)
    if Q_demand > Q_at_trigger * 1.8 and ps_level < 10.0:
        return "flow_starvation"

    # Delayed cycling: ventilator Ti substantially exceeds patient neural Ti
    # Patient begins active exhalation while ventilator still pressurising
    ti_ratio = t_insp / max(t_effort_dur, 0.1)
    if flow_cycle_threshold <= 0.15: 
        if ti_ratio > 1.05 or insp_ended_by_reversal:
            return "delayed_cycling"

    # Premature cycling: ventilator stops well before patient neural Ti ends
    if flow_cycle_threshold >= 0.38 and ti_ratio < 0.65:
        return "premature_cycling"

    # Double triggering signature: very short breath (< 40% effort duration)
    # followed by immediate second trigger — detected externally across breaths
    if t_insp < 0.4 * t_effort_dur and t_insp < 0.5:
        return "double_trigger"

    return "synchronous"


def _detect_reverse_trigger(prev_label: str,
                              t_since_last_breath: float,
                              pmus_during_exp: float) -> bool:
    """
    Detect reverse triggering: the mechanical breath itself entrains the
    patient's respiratory muscles to generate effort during exhalation.

    Occurs after a synchronous or any mechanical breath. Pmus is detected
    during the expiratory phase (should be zero in passive patients).
    Common during paralysis reversal and light sedation.
    """
    return (pmus_during_exp > 2.0 and
            t_since_last_breath < 1.0 and
            prev_label in ("synchronous", "delayed_cycling"))


# ---------------------------------------------------------------------------
# Section 6 — ETT Complication Modeling
# ---------------------------------------------------------------------------

def _get_ett_params(ett_complication: Optional[str],
                     cuff_leak_fraction: float,
                     obstruction_multiplier: float,
                     K1_base: float,
                     K2_base: float) -> Tuple[float, float, float]:
    """
    Modify Rohrer coefficients and return leak fraction for ETT complications.

    Parameters
    ----------
    ett_complication       : None | "cuff_leak" | "partial_obstruction"
    cuff_leak_fraction     : fraction of Vt lost per breath through cuff leak
    obstruction_multiplier : multiplier on K1 and K2 for partial obstruction
    K1_base, K2_base       : baseline Rohrer coefficients

    Returns
    -------
    Tuple of (K1_eff, K2_eff, leak_fraction)
    """
    if ett_complication == "partial_obstruction":
        # Secretion accumulation or biting narrows ETT lumen
        # Resistance rises — primarily affects K1 (viscous) but also K2 (turbulent)
        return (K1_base * obstruction_multiplier,
                K2_base * obstruction_multiplier,
                0.0)

    elif ett_complication == "cuff_leak":
        # Deflated or herniated cuff — volume escapes around ETT per breath
        # No resistance change; leak manifests as Vt_exp < Vt_insp
        return K1_base, K2_base, cuff_leak_fraction

    # No complication
    return K1_base, K2_base, 0.0


# ---------------------------------------------------------------------------
# Section 7 — Parameter Validation
# ---------------------------------------------------------------------------

REQUIRED_PARAMS = {
    "pressure_support_cmH2O", "peep_cmH2O", "rise_time_s",
    "flow_cycle_threshold", "trigger_threshold_cmH2O",
    "pmus_peak_cmH2O", "effort_rate_per_min",
    "effort_duration_s", "pmus_cv",
    "compliance_ml_per_cmH2O", "resistance_cmH2O_L_s",
}


def _validate_params(params: dict) -> None:
    """Validate required keys and physiological plausibility."""
    missing = REQUIRED_PARAMS - params.keys()
    if missing:
        raise ValueError(f"Missing required parameter(s): {sorted(missing)}")

    ps    = params["pressure_support_cmH2O"]
    peep  = params["peep_cmH2O"]
    rr    = params["effort_rate_per_min"]
    pmus  = params["pmus_peak_cmH2O"]
    rt    = params["rise_time_s"]
    fct   = params["flow_cycle_threshold"]
    C     = params["compliance_ml_per_cmH2O"]
    R     = params["resistance_cmH2O_L_s"]
    cv    = params["pmus_cv"]

    if not (1 <= ps <= 50):
        raise ValueError(f"pressure_support_cmH2O {ps} out of range [1, {PS_MAX_CMHH2O}]")
    if not (0 <= peep <= 20):
        raise ValueError(f"peep_cmH2O {peep} out of range [0, 20]")
    if not (5 <= rr <= 45):
        raise ValueError(f"effort_rate_per_min {rr} out of range [5, 45]")
    if not (1 <= pmus <= 35):
        raise ValueError(f"pmus_peak_cmH2O {pmus} out of range [1, 35]")
    if not (0.0 <= rt <= 0.5):
        raise ValueError(f"rise_time_s {rt} out of range [0, 0.5]")
    if not (0.05 <= fct <= 0.70):
        raise ValueError(f"flow_cycle_threshold {fct} out of range [0.05, 0.50]")
    if not (5 <= C <= 200):
        raise ValueError(f"compliance_ml_per_cmH2O {C} out of range [5, 200]")
    if not (0.5 <= R <= 60):
        raise ValueError(f"resistance_cmH2O_L_s {R} out of range [0.5, 60]")
    if not (0.05 <= cv <= 0.60):
        raise ValueError(f"pmus_cv {cv} out of range [0.05, 0.60]")


def _assess_validity(metrics: dict, params: dict) -> Tuple[bool, str]:
    """Apply clinical safety filters; return (is_valid, reason)."""
    vt   = metrics["delivered_vt_ml"]
    ppk  = metrics["ppeak_cmH2O"]
    ps   = params["pressure_support_cmH2O"]
    ff   = metrics["fill_fraction"]

    if ppk > PPEAK_MAX_CMHH2O:
        return False, f"Ppeak {ppk:.1f} cmH2O exceeds barotrauma threshold {PPEAK_MAX_CMHH2O}"
    if ps > PS_MAX_CMHH2O:
        return False, f"Pressure support {ps} cmH2O exceeds maximum {PS_MAX_CMHH2O}"
    if vt > VT_MAX_ML:
        return False, f"Delivered Vt {vt:.0f} mL exceeds overdistension limit {VT_MAX_ML:.0f} mL"
    if vt < VT_MIN_ML and metrics.get("triggered_breath_rate", 0) > 0:
        return False, f"Delivered Vt {vt:.0f} mL below minimum {VT_MIN_ML:.0f} mL"
    if ff < FILL_FRACTION_MIN:
        return False, f"Fill fraction {ff:.3f} below minimum {FILL_FRACTION_MIN}"
    return True, ""


# ---------------------------------------------------------------------------
# Section 8 — Main PSV Generator (Event-Driven Loop)
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict,
                            n_cycles: int = 10,
                            seed: Optional[int] = None) -> dict:
    """
    Generate PSV waveforms using an event-driven breath simulation.

    Rather than a fixed time-grid (as in VCV and PCV), PSV advances through
    time by detecting patient effort onsets, checking trigger success,
    running the inspiratory ODE until the flow-cycle criterion is met, then
    running the expiratory ODE until the next effort onset.

    This produces naturally variable breath timing and tidal volumes —
    the defining feature that distinguishes PSV from mandatory modes.

    Parameters
    ----------
    params   : dict — all required parameters (see REQUIRED_PARAMS)
    n_cycles : int  — number of *triggered* breaths to generate
    seed     : int  — random seed for reproducibility

    Returns
    -------
    dict — see module docstring for full key list
    """
    _validate_params(params)

    # ---- Unpack parameters ------------------------------------------------
    ps_level    = float(params["pressure_support_cmH2O"])
    peep_e      = float(params["peep_cmH2O"])
    rise_time   = float(params["rise_time_s"])
    fct         = float(params["flow_cycle_threshold"])
    trig_thr    = float(params["trigger_threshold_cmH2O"])
    pmus_mean   = float(params["pmus_peak_cmH2O"])
    eff_rate    = float(params["effort_rate_per_min"])
    eff_dur_mn  = float(params["effort_duration_s"])
    pmus_cv     = float(params["pmus_cv"])
    C_global    = float(params["compliance_ml_per_cmH2O"])
    R_global    = float(params["resistance_cmH2O_L_s"])

    # Optional mechanics refinements
    condition       = params.get("condition", "Normal")
    stress_index    = float(params.get("stress_index", 1.0))
    C_chest         = float(params.get("chest_wall_compliance_ml_per_cmH2O",
                                        DEFAULT_CHEST_WALL_COMPLIANCE))
    circ_compensated = bool(params.get("circuit_compensated", True))
    peep_ref        = float(params.get("peep_reference_cmH2O", 5.0))
    rec_slope       = float(params.get("recruitment_slope",
                                        RECRUITMENT_SLOPES.get(condition, 0.5)))

    # ETT complications
    ett_complication = params.get("ett_complication", None)
    cuff_leak_frac   = float(params.get("cuff_leak_fraction", 0.0))
    obs_multiplier   = float(params.get("obstruction_R_multiplier", 1.0))

    # Rohrer base coefficients (derive from total R; ETT contributes ~50%)
    K1_intrinsic = R_global * 0.60
    K2_intrinsic = R_global * 0.04
    K1_base = K1_intrinsic + ETT_K1
    K2_base = K2_intrinsic + ETT_K2
    K1_eff, K2_eff, leak_frac = _get_ett_params(
        ett_complication, cuff_leak_frac, obs_multiplier, K1_base, K2_base
    )

    # ---- Build compartment arrays ----------------------------------------
    profile = COMPARTMENT_PROFILES.get(condition, COMPARTMENT_PROFILES["Normal"])
    n_comps = len(profile)

    fractions   = np.array([c["fraction"]  for c in profile])
    C_frac_arr  = np.array([c["C_frac"]    for c in profile])
    C_frac_norm = float(np.dot(C_frac_arr, fractions))
    R_frac_arr  = np.array([c["R_frac"]    for c in profile])
    R_exp_arr   = np.array([c["R_exp_ratio"] for c in profile])
    teth_arr    = np.array([c["tethering"] for c in profile])

    # PEEP-recruited compliance applied to global C before per-compartment split
    C_lung_rec = _peep_recruited_compliance(C_global, peep_e, peep_ref, rec_slope)

    # Per-compartment base compliance and resistance (intrinsic + ETT)
    C_comps_base = C_lung_rec * C_frac_arr * fractions / max(C_frac_norm, 0.01)   # mL/cmH2O per compartment
    R_comps_base = R_global * R_frac_arr      # cmH2O/L/s per compartment

    # Reference volume for non-linear compliance (mid-inspiration target)
    vt_ref_per_comp = (IBW_KG * 6.0) * fractions  # 6 mL/kg IBW per compartment

    # ---- Simulation state ------------------------------------------------
    rng = np.random.default_rng(seed)
    V_comps   = np.zeros(n_comps)   # current compartment volumes (mL)
    t_current = 0.0                  # running simulation time (s)

    # Output accumulators
    T_list, P_list, Q_list, V_list   = [], [], [], []
    Pres_list, Pel_list, Tpeep_list  = [], [], []

    # Per-breath tracking
    dyssync_labels: List[str]   = []
    triggered_count             = 0
    total_effort_count          = 0
    insp_vt_list: List[float]   = []
    rr_list: List[float]        = []
    t_prev_breath               = 0.0
    prev_label                  = "synchronous"
    V_baseline_total: float = 0.0
    t_prev_insp: float = 0.0
    last_auto_peep_now: float = 0.0
    # ---- Event-driven main loop ------------------------------------------
    while triggered_count < n_cycles:
       

        # Schedule next effort onset with inter-effort variability (±10%)
        t_effort = 60.0 / eff_rate
        t_effort_noisy = float(rng.normal(t_effort, t_effort * 0.10))
        t_effort_noisy = max(t_effort_noisy, 0.30)

        # Sample this breath's effort parameters from distributions
        pmus_peak, eff_dur = _sample_breath_effort(
            pmus_mean, pmus_cv, eff_dur_mn, 0.15, rng
        )

        # ---- Passive expiration until effort onset -----------------------
        t_exp = max(t_effort_noisy - t_prev_insp, 0.10)
     

        # ---- Final expiration to complete the last breath cycle ----------------
        t_exp_final  = 60.0 / eff_rate
        n_exp_final  = max(2, int(round(t_exp_final / DT)))
        V_end_insp_f = V_comps.copy()

        for step in range(n_exp_final):
            if step == 0:
                continue
            t_in_exp = step * DT
            Q_comps  = np.zeros(n_comps)

            for i in range(n_comps):
                Vi   = max(V_comps[i], 0.0)
                Ri_e = _R_exp_dynamic(Vi, max(V_end_insp_f[i], 1.0), R_comps_base[i], R_exp_arr[i])
                
                # Passive deflation ODE: dV/dt = -V / (R * C)
                C_i  = _compliance_nonlinear(
                    Vi, C_comps_base[i], vt_ref_per_comp[i] * 0.5, stress_index
                )
                C_rs_i = _C_rs(C_i, C_chest)
                dVdt_i = -(Vi / max(C_rs_i, 0.1)) / max(Ri_e, 0.1) * 1000.0
                V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
                Q_comps[i] = dVdt_i / 1000.0  # L/s


            Q_total    = float(sum(Q_comps))
            V_total    = float(V_comps.sum())
            C_rs_total  = max(C_lung_rec * sum(
            fractions[i] * _compliance_nonlinear(V_comps[i], C_comps_base[i],
                vt_ref_per_comp[i] * 0.5, stress_index) / max(C_comps_base[i], 0.1)
            for i in range(n_comps)), 0.5)
            C_rs_total = _C_rs(C_rs_total, C_chest)

            
            pres  = 0.0
            pel = max((V_total - V_baseline_total) / max(C_rs_total, 0.1),0.0)
            tpeep   = peep_e + (V_baseline_total / max(C_rs_total, 0.1))


            T_list.append(t_current + t_in_exp)
            P_list.append(peep_e)
            Q_list.append(float(sum(Q_comps)))
            V_list.append(V_total)
            Pres_list.append(pres)
            Pel_list.append(pel)
            Tpeep_list.append(tpeep)
        V_baseline_total = float(V_comps.sum()) 
        t_current += n_exp_final * DT
        total_effort_count += 1

        # ---- Compute auto-PEEP at effort onset ---------------------------
        V_end_exp  = float(V_comps.sum())
        C_lung_eff_now = max(C_lung_rec * sum(
            fractions[i] * _compliance_nonlinear(
                V_comps[i], C_comps_base[i],
                vt_ref_per_comp[i] * 0.5, stress_index
            ) / max(C_comps_base[i], 0.1) for i in range(n_comps)
        ), 0.5)
        C_rs_eff_now = _C_rs(C_lung_eff_now, C_chest)   # ← add chest wall
        auto_peep_now = V_end_exp / max(C_rs_eff_now, 0.1)
        last_auto_peep_now = auto_peep_now

        # ---- Check reverse triggering ------------------------------------
        pmus_during_exp = _pmus_waveform(t_exp * 0.5, eff_dur, pmus_peak * 0.2)
        if _detect_reverse_trigger(prev_label, t_effort_noisy, pmus_during_exp):
            dyssync_labels.append("reverse_trigger")
            triggered_count += 1
            insp_vt_list.append(0.0)
            rr_list.append(60.0 / max(t_current - t_prev_breath, 0.1))
            t_prev_breath = t_current
            prev_label = "reverse_trigger"
            t_prev_insp = 0.0
            continue

        # ---- Trigger check -----------------------------------------------
        # Pmus at effort onset ≈ 50% of peak (effort still rising)
        pmus_at_onset   = pmus_peak * 0.50
        triggered       = _check_trigger(pmus_at_onset, auto_peep_now, trig_thr)

        # Estimate patient flow demand for flow starvation detection
        Q_demand = pmus_peak / max(R_comps_base.mean(), 0.1)

        if not triggered:
            # Ineffective trigger — effort creates a small flow perturbation
            # visible on the expiratory flow waveform but no breath delivered
            n_eff_steps = max(1, int(round(eff_dur / DT)))
            for step in range(n_eff_steps):
                te = step * DT
                t_prev_insp = n_eff_steps * DT
                pmus_now = _pmus_waveform(te, eff_dur, pmus_peak)
                # Attenuated flow perturbation (partial opening against auto-PEEP)
                Q_perturb = min(pmus_now / max(K1_eff + auto_peep_now, 1.0), 0.05)
                pres_eff  = -Q_perturb * K1_eff           # resistive: tiny inspiratory → negative
                pel_eff   = 0.0                            # volume unchanged → no elastic ΔP
                tpeep_eff = peep_e + auto_peep_now         # baseline = set PEEP + auto-PEEP
                pao_eff   = pres_eff + pel_eff + tpeep_eff # always satisfies decomposition
                T_list.append(t_current + te)
                P_list.append(pao_eff)
                Q_list.append(Q_perturb)  # still expiratory phase
                V_list.append(V_end_exp)
                Pres_list.append(pres_eff)
                Pel_list.append(pel_eff)
                Tpeep_list.append(tpeep_eff)

            t_current += n_eff_steps * DT
            dyssync_labels.append("ineffective_trigger")
            triggered_count += 1
            insp_vt_list.append(0.0)
            rr_list.append(0.0)
            prev_label = "ineffective_trigger"
            continue

        # ---- Triggered: run inspiratory ODE ------------------------------
        V_start_insp = V_comps.copy()
        Q_peak_insp  = 0.0
        past_peak    = False
        t_insp       = 0.0
        Q_at_trigger = 0.0

        insp_ended_by_reversal = False
        while t_insp < MAX_INSP_TIME_S:
            # Ventilator pressure (rise phase → plateau)
            if t_insp < rise_time:
                P_vent = peep_e + ps_level * (t_insp / max(rise_time, DT))
            else:
                P_vent = peep_e + ps_level

            # Patient effort at this moment
            pmus_now = _pmus_waveform(t_insp, eff_dur, pmus_peak)

            # Per-compartment inspiratory ODE
            Q_comps = np.zeros(n_comps)
            for i in range(n_comps):
                Vi   = max(V_comps[i], 0.0)
                C_i  = _compliance_nonlinear(
                    Vi, C_comps_base[i], vt_ref_per_comp[i] * 0.5, stress_index
                )
                C_rs_i = _C_rs(C_i, C_chest)
                Ri_i   = _R_insp_with_tethering(
                    R_comps_base[i],
                    Vi, vt_ref_per_comp[i], teth_arr[i]
                )
                drive_i = P_vent + pmus_now - (Vi / max(C_rs_i, 0.1)) - peep_e
                dVdt_i   = drive_i / max(Ri_i, 0.1) * 1000.0
                V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
                Q_comps[i] = dVdt_i / 1000.0

            
            Q_total = float(Q_comps.sum())
            V_total   = float(V_comps.sum())
            C_rs_now  = max(C_lung_rec * sum(
                fractions[i] * _compliance_nonlinear(
                    V_comps[i], C_comps_base[i],
                    vt_ref_per_comp[i] * 0.5, stress_index
                ) / max(C_comps_base[i], 0.1) for i in range(n_comps)
            ), 0.5)
            peep_total_now = peep_e + auto_peep_now

            
            # Rohrer resistive pressure on total flow
            pres_now = _rohrer_resistance(Q_total, K1_eff, K2_eff)
            pel_now  = (V_total - V_baseline_total) / max(C_rs_now, 0.1)
            tpeep_now = peep_e + (V_baseline_total / max(C_rs_now, 0.1))
            pao_now = P_vent 
           
            if t_insp < DT:
                Q_at_trigger = Q_total

            T_list.append(t_current + t_insp)
            P_list.append(pao_now)
            Q_list.append(Q_total)
            V_list.append(V_total)
            Pres_list.append(pres_now)
            Pel_list.append(pel_now)
            Tpeep_list.append(tpeep_now)

            # Track peak flow and check cycling criterion
            if Q_total > Q_peak_insp:
                Q_peak_insp = Q_total
            elif Q_total < Q_peak_insp * 0.95 and not past_peak and Q_peak_insp > 0.01:
                past_peak = True

            if past_peak and _check_cycle(Q_total, Q_peak_insp, fct):
                if Q_total <= 0.0:
                    insp_ended_by_reversal = True
                break

            t_insp += DT
        t_prev_insp = t_insp
        # ---- Compute breath-level metrics --------------------------------
        insp_vt = float(V_comps.sum() - V_start_insp.sum())
        insp_vt = max(insp_vt, 0.0)

        # Apply cuff-leak correction
        patient_vt = insp_vt * (1.0 - leak_frac)

        # Classify dyssynchrony for this breath
        label = _classify_dyssynchrony(
            triggered=True,
            t_insp=t_insp,
            t_effort_dur=eff_dur_mn,
            Q_peak=Q_peak_insp,
            flow_cycle_threshold=fct,
            ps_level=ps_level,
            Q_at_trigger=Q_at_trigger,
            Q_demand=Q_demand,
            insp_ended_by_reversal=insp_ended_by_reversal,
        )

        dyssync_labels.append(label)
        triggered_count += 1
        insp_vt_list.append(insp_vt)
        t_breath_duration = t_current + t_insp - t_prev_breath
        rr_list.append(60.0 / max(t_breath_duration, 0.1))
        t_prev_breath = t_current + t_insp
        t_current    += t_insp
        prev_label    = label
    
    t_trail        = 60.0 / eff_rate
    n_trail        = max(2, int(round(t_trail / DT)))
    V_end_insp_trail = V_comps.copy()

    for step in range(n_trail):
        if step == 0:
            continue
        t_in_exp = step * DT
        Q_comps  = np.zeros(n_comps)

        for i in range(n_comps):
            Vi   = max(V_comps[i], 0.0)
            Ri_e = _R_exp_dynamic(Vi, max(V_end_insp_trail[i], 1.0), R_comps_base[i], R_exp_arr[i])
            C_i  = _compliance_nonlinear(
                Vi, C_comps_base[i], vt_ref_per_comp[i] * 0.5, stress_index
            )
            C_rs_i = _C_rs(C_i, C_chest)
            dVdt_i = -(Vi / max(C_rs_i, 0.1)) / max(Ri_e, 0.1) * 1000.0
            V_comps[i] = max(V_comps[i] + dVdt_i * DT, 0.0)
            Q_comps[i] = dVdt_i / 1000.0

        Q_total    = float(sum(Q_comps))
        V_total    = float(V_comps.sum())
        C_rs_total = max(C_lung_rec * sum(
            fractions[i] * _compliance_nonlinear(V_comps[i], C_comps_base[i],
                vt_ref_per_comp[i] * 0.5, stress_index) / max(C_comps_base[i], 0.1)
            for i in range(n_comps)), 0.5)
        C_rs_total = _C_rs(C_rs_total, C_chest)

        pres  = 0.0
        pel   = max((V_total - V_baseline_total) / max(C_rs_total, 0.1), 0.0)
        tpeep = peep_e + (V_baseline_total / max(C_rs_total, 0.1))

        T_list.append(t_current + t_in_exp)
        P_list.append(peep_e)
        Q_list.append(float(sum(Q_comps)))
        V_list.append(V_total)
        Pres_list.append(pres)
        Pel_list.append(pel)
        Tpeep_list.append(tpeep)

    # ---- Aggregate metrics -----------------------------------------------
    for i in range(1, len(T_list)):
        if T_list[i] <= T_list[i - 1]:
            T_list[i] = T_list[i - 1] + DT
    time_arr  = np.array(T_list, dtype=np.float32)
    pres_arr  = np.array(P_list, dtype=np.float32)
    flow_arr  = np.array(Q_list, dtype=np.float32)
    vol_arr   = np.array(V_list, dtype=np.float32)
    pres_r_arr = np.array(Pres_list, dtype=np.float32)
    pres_e_arr = np.array(Pel_list,  dtype=np.float32)
    tpeep_arr  = np.array(Tpeep_list, dtype=np.float32)

    valid_vts = [v for v in insp_vt_list if v > 1.0]
    mean_vt   = float(np.mean(valid_vts)) if valid_vts else 0.0
    delivered_vt_ml = mean_vt
    ppeak     = float(pres_arr.max()) if len(pres_arr) else peep_e
    mean_paw  = float(pres_arr.mean()) if len(pres_arr) else peep_e
    patient_vt_ml = _circuit_vt_correction(
    delivered_vt_ml * (1.0 - leak_frac),   # ← apply leak first
    ppeak, peep_e,
    CIRCUIT_COMPLIANCE_ML_PER_CMH2O, circ_compensated
)
    # Fill fraction: ratio of mean delivered Vt to theoretical maximum
    vt_max = (ps_level + pmus_mean) * C_lung_rec
    fill_frac = float(np.clip(mean_vt / max(vt_max, 1.0), 0.0, 1.0))

    # Auto-PEEP from residual volume at final end-expiration
    final_auto_peep = last_auto_peep_now 

    # Circuit-corrected patient Vt
    patient_vt_corrected = _circuit_vt_correction(
        mean_vt, ppeak, peep_e,
        CIRCUIT_COMPLIANCE_ML_PER_CMH2O, circ_compensated
    )

    # Triggered breath rate
    n_triggered = sum(1 for l in dyssync_labels if l != "ineffective_trigger")
    total_time  = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 1.0
    trig_rate   = n_triggered / max(total_time / 60.0, 0.001)

    # Ineffective trigger fraction
    ineff_frac = sum(1 for l in dyssync_labels
                      if l == "ineffective_trigger") / max(len(dyssync_labels), 1)

    # Pel at end-inspiration (= driving pressure) — use mean of valid breaths
    pres_peak  = float(pres_r_arr.max()) if len(pres_r_arr) else 0.0
    pel_end    = float(mean_vt / max(C_lung_rec, 0.1))
    dp_cmH2O   = pel_end  # driving pressure ≈ Pel at end-inspiration

    # Stress index: slope of log(Pel) vs log(time) during inspiration
    # Approximated from the ratio of early to late elastic pressure rise
    stress_idx_computed = stress_index  # use set value as label for dataset

    pres_pel_r = pres_peak / max(pel_end, 0.1)
    minute_vent = trig_rate * mean_vt / 1000.0

    metrics = {
        "ppeak_cmH2O":              ppeak,
        "delivered_vt_ml":          delivered_vt_ml,
        "patient_vt_ml":            patient_vt_ml,
        "driving_p_cmH2O":          dp_cmH2O,
        "mean_paw_cmH2O":           mean_paw,
        "auto_peep_cmH2O":          final_auto_peep,
        "total_peep_cmH2O":         peep_e + final_auto_peep,
        "fill_fraction":            fill_frac,
        "minute_vent_l":            minute_vent,
        "pres_peak_cmH2O":          pres_peak,
        "pel_end_insp_cmH2O":       pel_end,
        "stress_index":             stress_idx_computed,
        "pres_pel_ratio":           pres_pel_r,
        "triggered_breath_rate":    trig_rate,
        "ineffective_trigger_fraction": ineff_frac,
    }

    is_valid, invalid_reason = _assess_validity(metrics, params)

    return {
        # Core waveforms
        "time":               time_arr,
        "pressure":           pres_arr,
        "flow":               flow_arr,
        "volume":             vol_arr,
        # Pressure decomposition
        "pressure_resistive": pres_r_arr,
        "pressure_elastic":   pres_e_arr,
        "pressure_total_peep": tpeep_arr,
        # Scalar metrics
        **metrics,
        # Per-breath labels
        "breath_dyssynchrony_labels": dyssync_labels,
        # Validity
        "is_valid":           is_valid,
        "invalid_reason":     invalid_reason,
    }


# ---------------------------------------------------------------------------
# Section 9 — SBT Temporal Sequence Generator
# ---------------------------------------------------------------------------

def generate_sbt_sequence(params: dict,
                           trial_duration_min: float = 30.0,
                           trial_ps_cmH2O: float = 5.0,
                           baseline_cycles: int = 10,
                           n_windows: int = 10,
                           seed: Optional[int] = None) -> dict:
    """
    Generate a Spontaneous Breathing Trial (SBT) as a multi-phase temporal
    sequence: baseline full support → reduced-support trial → outcome.

    The SBT is defined by a trajectory, not a snapshot. A model trained on
    this data learns to predict pass/fail from the direction and rate of
    waveform evolution rather than from any single-timepoint morphology.

    SBT failure criteria (any one triggers failure):
        - RRSB (f/Vt) > 105 breaths/min/L  (Yang & Tobin 1991)
        - Effort rate  > 35 breaths/min
        - Progressive tidal volume decline  > 30% from trial start
        - Progressive auto-PEEP rise        > 3 cmH2O from baseline

    Parameters
    ----------
    params             : dict — full PSV params (used for baseline phase)
    trial_duration_min : float — SBT trial duration in minutes
    trial_ps_cmH2O     : float — PS during trial (typically 5–8 cmH2O)
    baseline_cycles    : int   — n_cycles for baseline phase
    n_windows          : int   — number of sampled waveform windows during trial
    seed               : int   — random seed

    Returns
    -------
    dict with keys:
        scenario_type, event_type, outcome, time_to_failure_min,
        baseline_result, trial_windows (list of dicts), parameter_trajectory,
        rrsb_trajectory, metadata
    """
    _validate_params(params)
    rng = np.random.default_rng(seed)

    # ---- Phase 1: Baseline at full support --------------------------------
    baseline_result = generate_breath_cycles(params, n_cycles=baseline_cycles,
                                              seed=int(rng.integers(0, 2**31)))
    baseline_vt   = baseline_result["delivered_vt_ml"]
    baseline_ape  = baseline_result["auto_peep_cmH2O"]
    baseline_rr   = baseline_result["triggered_breath_rate"]
    baseline_rrsb = baseline_rr / max(baseline_vt / 1000.0, 0.001)

    # ---- Phase 2: Trial at reduced support --------------------------------
    trial_params = {**params, "pressure_support_cmH2O": trial_ps_cmH2O}

    # Determine n_cycles per window to approximate temporal spacing
    approx_rr_trial = params["effort_rate_per_min"]  # expect similar to neural rate
    t_window_min = trial_duration_min / n_windows
    cycles_per_window = max(3, int(approx_rr_trial * t_window_min))

    trial_windows = []
    rrsb_trajectory = [baseline_rrsb]
    outcome = "pass"
    time_to_failure_min = None
    parameter_traj = {
        "t_minutes":        [0.0],
        "pressure_support": [params["pressure_support_cmH2O"]],
        "pmus_peak":        [params["pmus_peak_cmH2O"]],
        "resistance":       [params["resistance_cmH2O_L_s"]],
    }

    t_elapsed = 0.0
    prev_vt   = baseline_vt

    for w in range(n_windows):
        t_elapsed += t_window_min

        # As trial progresses, respiratory distress may escalate:
        # gradually increase Pmus (more effort) and effort rate
        distress_factor = 0.0
        if outcome == "pass":
            # Mild natural variability (stable patient)
            distress_factor = float(rng.uniform(-0.05, 0.08))
        else:
            # Patient has failed — no more windows
            break

        pmus_trial = params["pmus_peak_cmH2O"] * (1.0 + distress_factor * 0.5)
        rate_trial = params["effort_rate_per_min"] * (1.0 + distress_factor * 0.3)

        window_params = {
            **trial_params,
            "pmus_peak_cmH2O":     float(np.clip(pmus_trial, 4.0, 30.0)),
            "effort_rate_per_min": float(np.clip(rate_trial, 10.0, 40.0)),
        }

        try:
            window_result = generate_breath_cycles(
                window_params, n_cycles=cycles_per_window,
                seed=int(rng.integers(0, 2**31))
            )
        except Exception as e:
            window_result = {"delivered_vt_ml": 0.0,
                              "triggered_breath_rate": rate_trial,
                              "auto_peep_cmH2O": 10.0,
                              "is_valid": False}

        window_vt   = window_result.get("delivered_vt_ml", 0.0)
        window_rr   = window_result.get("triggered_breath_rate", rate_trial)
        window_ape  = window_result.get("auto_peep_cmH2O", 0.0)
        window_rrsb = window_rr / max(window_vt / 1000.0, 0.001)

        rrsb_trajectory.append(window_rrsb)
        parameter_traj["t_minutes"].append(t_elapsed)
        parameter_traj["pressure_support"].append(trial_ps_cmH2O)
        parameter_traj["pmus_peak"].append(pmus_trial)
        parameter_traj["resistance"].append(params["resistance_cmH2O_L_s"])

        # Check SBT failure criteria
        failure_reason = None
        if window_rrsb > RRSB_FAILURE_THRESHOLD:
            failure_reason = f"RRSB {window_rrsb:.0f} > {RRSB_FAILURE_THRESHOLD}"
        elif window_rr > RR_FAILURE_THRESHOLD:
            failure_reason = f"RR {window_rr:.0f} > {RR_FAILURE_THRESHOLD}"
        elif window_vt < prev_vt * 0.70:
            failure_reason = f"Vt {window_vt:.0f} mL declined >30% from trial start"
        elif window_ape > baseline_ape + 3.0:
            failure_reason = f"Auto-PEEP {window_ape:.1f} rose >3 cmH2O above baseline"

        trial_windows.append({
            "t_minutes":          t_elapsed,
            "window_index":       w,
            "delivered_vt_ml":    window_vt,
            "triggered_rr":       window_rr,
            "rrsb":               window_rrsb,
            "auto_peep_cmH2O":    window_ape,
            "is_valid":           window_result.get("is_valid", False),
            "waveforms": {
                "time":     window_result.get("time",     np.array([])),
                "pressure": window_result.get("pressure", np.array([])),
                "flow":     window_result.get("flow",     np.array([])),
                "volume":   window_result.get("volume",   np.array([])),
            },
            "failure_reason": failure_reason,
        })

        if failure_reason is not None:
            outcome = "fail"
            time_to_failure_min = t_elapsed
            break

        prev_vt = window_vt

    return {
        "scenario_type":       "temporal_sequence",
        "event_type":          "spontaneous_breathing_trial",
        "outcome":             outcome,
        "time_to_failure_min": time_to_failure_min,
        "trial_duration_min":  trial_duration_min,
        "trial_ps_cmH2O":      trial_ps_cmH2O,
        "baseline_result":     {
            "delivered_vt_ml":   baseline_vt,
            "triggered_rr":      baseline_rr,
            "rrsb":              baseline_rrsb,
            "auto_peep_cmH2O":   baseline_ape,
        },
        "trial_windows":       trial_windows,
        "rrsb_trajectory":     rrsb_trajectory,
        "parameter_trajectory": parameter_traj,
        "metadata": {
            "condition":    params.get("condition", "Unknown"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_windows":    len(trial_windows),
        },
    }


# ---------------------------------------------------------------------------
# Section 10 — Dataset Generation
# ---------------------------------------------------------------------------

def _make_scenario_id(condition: str, params: dict) -> str:
    ps  = int(params["pressure_support_cmH2O"])
    rr  = int(params["effort_rate_per_min"])
    pm  = int(params["pmus_peak_cmH2O"])
    C   = int(params["compliance_ml_per_cmH2O"])
    R   = int(params["resistance_cmH2O_L_s"])
    peep = int(params["peep_cmH2O"])
    fct  = int(params["flow_cycle_threshold"] * 100)
    rt   = int(params.get("rise_time_s",      0.0) * 10)   # e.g. 0/1/2/4
    ed   = int(params.get("effort_duration_s", 0.8) * 10)  # e.g. 5/8/11
    cv   = int(params.get("pmus_cv",          0.15) * 100)
    tt   = int(params.get("trigger_threshold_cmH2O", 1.5) * 10)  
    cond = condition.replace(" ", "_").upper()
    return f"PSV_{cond}_PS{ps:02d}_RR{rr:02d}_PMUS{pm:02d}_C{C:03d}_R{R:02d}_PEEP{peep:02d}_FCT{fct:02d}_RT{rt:02d}_ED{ed:02d}_CV{cv:02d}_TT{tt:02d}"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_dataset(condition_name: str,
                      compliance_ml_per_cmH2O: float,
                      resistance_cmH2O_L_s: float,
                      n_cycles: int = 5,
                      seed: Optional[int] = None) -> List[dict]:
    """
    Sweep the thinned PSV parameter grid for one condition + mechanics pair.

    Parameters
    ----------
    condition_name           : str   — one of the seven condition names
    compliance_ml_per_cmH2O : float — global lung compliance preset
    resistance_cmH2O_L_s    : float — total system resistance preset
    n_cycles                 : int   — triggered breaths per scenario
    seed                     : int   — base random seed (incremented per scenario)

    Returns
    -------
    list of dicts, one per parameter combination, with keys:
        scenario_id, condition, params, metrics, is_valid, invalid_reason,
        waveforms, breath_dyssynchrony_labels, generated_at
    """
    scenarios = []
    rng_base  = np.random.default_rng(seed)

    rec_slope = RECRUITMENT_SLOPES.get(condition_name, 0.5)

    grid_keys = list(DATASET_GRID.keys())
    grid_vals = [DATASET_GRID[k] for k in grid_keys]

    for combo in itertools.product(*grid_vals):
        p = dict(zip(grid_keys, combo))
        p["compliance_ml_per_cmH2O"] = compliance_ml_per_cmH2O
        p["resistance_cmH2O_L_s"]     = resistance_cmH2O_L_s
        p["condition"]                = condition_name
        p["recruitment_slope"]        = rec_slope

        scenario_seed = int(rng_base.integers(0, 2**31))
        scenario_id   = _make_scenario_id(condition_name, p)

        try:
            result = generate_breath_cycles(p, n_cycles=n_cycles,
                                             seed=scenario_seed)
        except Exception as exc:
            scenarios.append({
                "scenario_id":    scenario_id,
                "condition":      condition_name,
                "params":         p,
                "metrics":        {},
                "is_valid":       False,
                "invalid_reason": f"Generator error: {exc}",
                "waveforms":      {},
                "breath_dyssynchrony_labels": [],
                "generated_at":   _timestamp(),
            })
            continue

        metric_keys = [
            "ppeak_cmH2O", "delivered_vt_ml", "patient_vt_ml",
            "driving_p_cmH2O", "mean_paw_cmH2O", "auto_peep_cmH2O",
            "total_peep_cmH2O", "fill_fraction", "minute_vent_l",
            "pres_peak_cmH2O", "pel_end_insp_cmH2O", "stress_index",
            "pres_pel_ratio", "triggered_breath_rate",
            "ineffective_trigger_fraction",
        ]
        metrics = {k: result[k] for k in metric_keys if k in result}

        waveforms = {}
        if result["is_valid"]:
            waveforms = {
                "time":               result["time"],
                "pressure":           result["pressure"],
                "flow":               result["flow"],
                "volume":             result["volume"],
                "pressure_resistive": result["pressure_resistive"],
                "pressure_elastic":   result["pressure_elastic"],
                "pressure_total_peep": result["pressure_total_peep"],
            }

        scenarios.append({
            "scenario_id":    scenario_id,
            "condition":      condition_name,
            "params":         p,
            "metrics":        metrics,
            "is_valid":       result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "waveforms":      waveforms,
            "breath_dyssynchrony_labels": result.get("breath_dyssynchrony_labels", []),
            "generated_at":   _timestamp(),
        })

    return scenarios


# ---------------------------------------------------------------------------
# Section 11 — Smoke Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _PASS = "\033[92m✓\033[0m"
    _FAIL = "\033[91m✗\033[0m"
    _results = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        status = _PASS if condition else _FAIL
        print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
        _results.append(condition)

    # ---- Test 1: Normal PSV waveform generation -------------------------
    print("\n[1/5] Normal PSV — synchronous breathing")
    p_normal = {
        "pressure_support_cmH2O":   10.0,
        "peep_cmH2O":                5.0,
        "rise_time_s":               0.1,
        "flow_cycle_threshold":      0.25,
        "trigger_threshold_cmH2O":   1.5,
        "pmus_peak_cmH2O":           8.0,
        "effort_rate_per_min":       18.0,
        "effort_duration_s":         0.8,
        "pmus_cv":                   0.20,
        "compliance_ml_per_cmH2O":  70.0,
        "resistance_cmH2O_L_s":     10.0,
        "condition":                "Normal",
    }
    r = generate_breath_cycles(p_normal, n_cycles=8, seed=42)
    _check("returns dict",             isinstance(r, dict))
    _check("time array non-empty",     len(r["time"]) > 0)
    _check("pressure decomposition",   "pressure_resistive" in r)
    _check("dyssynchrony labels",      len(r["breath_dyssynchrony_labels"]) == 8)
    _check("flow has neg+pos",         r["flow"].min() < 0 < r["flow"].max())
    _peep = p_normal["peep_cmH2O"]
    _ps   = p_normal["pressure_support_cmH2O"]
    _internal = (r["pressure_resistive"]
                 + r["pressure_elastic"]
                 + r["pressure_total_peep"])
    _servo_ok = (
        float(r["pressure"].max()) <= _peep + _ps + 3.0
        and float(r["pressure"].min()) >= _peep - 3.0
        and bool(np.all(np.isfinite(_internal)))
    )
    _check(
        "Servo pressure bounded (PEEP+PS ± 3)",
        _servo_ok,
        f"max={r['pressure'].max():.1f} min={r['pressure'].min():.2f} "
        f"target={_peep + _ps:.1f} — "
        f"Note: pres+pel+tpeep reflects internal mechanics, not displayed Pao"
    )
    print(f"     Ppeak={r['ppeak_cmH2O']:.1f} Vt={r['delivered_vt_ml']:.0f} "
          f"AutoPEEP={r['auto_peep_cmH2O']:.2f} "
          f"IneffFrac={r['ineffective_trigger_fraction']:.2f}")

    # ---- Test 2: COPD — ineffective triggering --------------------------
    print("\n[2/5] COPD — high auto-PEEP → ineffective triggering")
    p_copd = {
        **p_normal,
        "pmus_peak_cmH2O":          10.0,
        "effort_rate_per_min":       26.0,
        "pressure_support_cmH2O":   12.0,
        "peep_cmH2O":                5.0,
        "compliance_ml_per_cmH2O": 100.0,
        "resistance_cmH2O_L_s":     22.0,
        "condition":                "COPD",
    }
    r_copd = generate_breath_cycles(p_copd, n_cycles=12, seed=43)
    ineff = r_copd["ineffective_trigger_fraction"]
    _check("auto-PEEP elevated",
           r_copd["auto_peep_cmH2O"] > 1.0,
           f"auto-PEEP={r_copd['auto_peep_cmH2O']:.2f}")
    _check("dyssynchrony labels present",  len(r_copd["breath_dyssynchrony_labels"]) > 0)
    _check("fill fraction plausible",      0.0 < r_copd["fill_fraction"] < 1.0)
    print(f"     AutoPEEP={r_copd['auto_peep_cmH2O']:.2f} "
          f"IneffFrac={ineff:.2f} "
          f"Pres/Pel={r_copd['pres_pel_ratio']:.2f}")

    # ---- Test 3: Delayed cycling ----------------------------------------
    print("\n[3/5] Dyssynchrony — delayed cycling (low FCT=0.10)")
    p_delay = {**p_normal, "flow_cycle_threshold": 0.10,
               "effort_duration_s": 0.40, "resistance_cmH2O_L_s":  18.0, }
    r_delay = generate_breath_cycles(p_delay, n_cycles=8, seed=44)
    labels  = r_delay["breath_dyssynchrony_labels"]
    _check("delayed_cycling detected",
           any(l == "delayed_cycling" for l in labels),
           str(set(labels)))

    # ---- Test 4: ETT cuff leak ------------------------------------------
    print("\n[4/5] ETT cuff leak — patient Vt < delivered Vt")
    p_leak = {**p_normal, "ett_complication": "cuff_leak",
               "cuff_leak_fraction": 0.20}
    r_leak = generate_breath_cycles(p_leak, n_cycles=8, seed=45)
    _check("patient_vt < delivered_vt",
           r_leak["patient_vt_ml"] < r_leak["delivered_vt_ml"],
           f"delivered={r_leak['delivered_vt_ml']:.0f} patient={r_leak['patient_vt_ml']:.0f}")

    # ---- Test 5: SBT temporal sequence ----------------------------------
    print("\n[5/5] SBT temporal sequence — pass/fail trajectory")
    p_sbt = {
        **p_normal,
        "compliance_ml_per_cmH2O": 45.0,   # Mild ARDS recovering
        "resistance_cmH2O_L_s":    12.0,
        "condition":               "Mild ARDS",
        "pmus_peak_cmH2O":         10.0,
    }
    sbt = generate_sbt_sequence(p_sbt, trial_duration_min=20.0,
                                  trial_ps_cmH2O=5.0,
                                  baseline_cycles=5, n_windows=6, seed=46)
    _check("scenario_type correct",  sbt["scenario_type"] == "temporal_sequence")
    _check("event_type correct",     sbt["event_type"]    == "spontaneous_breathing_trial")
    _check("outcome defined",        sbt["outcome"] in ("pass", "fail"))
    _check("windows generated",      len(sbt["trial_windows"]) > 0)
    _check("rrsb trajectory",        len(sbt["rrsb_trajectory"]) > 1)
    print(f"     outcome={sbt['outcome']} "
          f"windows={len(sbt['trial_windows'])} "
          f"baseline_RRSB={sbt['baseline_result']['rrsb']:.0f}")

    # ---- Summary --------------------------------------------------------
    n_pass = sum(_results)
    n_total = len(_results)
    print(f"\n{'='*55}")
    print(f"  PSV generator smoke test: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("  WARNING: some checks failed — review output above")
    print(f"{'='*55}\n")
    sys.exit(0 if n_pass == n_total else 1)
