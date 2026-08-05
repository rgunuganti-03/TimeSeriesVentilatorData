"""
generator/prvc_generator.py
----------------------------
PRVC (Pressure-Regulated Volume Control) waveform generator.

Control loop
------------
Dual-control, breath-to-breath adaptive, pressure-limited, time-cycled
mandatory mode. Two nested loops:

  Inner loop (intra-breath) — identical structure to PCV: a constant
  working pressure P_work(n) is applied for breath n's inspiration
  (rise / plateau / passive expiratory decay). Volume and flow are
  dependent variables produced by the multi-compartment RC equation of
  motion, integrated with explicit Euler at 100 Hz per compartment.

  Outer loop (inter-breath) — after each breath, delivered tidal volume
  is measured and compared to VT_target. If the error exceeds
  vt_tolerance_frac, the working pressure for the next breath is stepped
  by +/- adaptation_step_cmH2O, clipped to [PEEP + 5, PEEP + pressure_ceiling].
  Breath 1 is a volume-controlled "test breath" (constant-flow, multi-
  compartment branch-point solve, with an inspiratory pause) whose
  plateau pressure seeds P_work(2) -- matching documented Servo/Dräger
  AutoFlow behavior rather than a blind assumed-compliance guess.

Modified equation of motion
----------------------------
    Breaths n >= 2 (pressure-prescribed, per compartment i):
        dV_i/dt = (P_work(n) - V_i/C_i - PEEP) / R_i * 1000   [mL/s]

    Breath 1 (flow-prescribed test breath, algebraic branch-point solve):
        Q_total = sum_i (Pao - V_i/C_i - PEEP) / R_i
        Pao(t)  = [Q_total + sum(V_i/(C_i*R_i)) + PEEP*sum(1/R_i)] / sum(1/R_i)
        Q_i(t)  = (Pao(t) - V_i/C_i - PEEP) / R_i

Physiological refinements incorporated
---------------------------------------
    1. Multi-compartment lung mechanics -- parallel RC compartments per
       condition, identical compartment counts/profiles to psv_generator
       (Normal=1, ARDS tiers=2, COPD=3, Pneumonia=3, Bronchospasm=2).

    2. Non-linear compliance -- power-law compliance per compartment
       parameterized by stress index, identical to psv_generator.

    3. Flow-dependent resistance (Rohrer) -- K1*Q + K2*Q*|Q| applied at
       the compartment level using that compartment's own flow and
       R_frac-scaled coefficients. NOTE: this is a documented
       simplification of PSV's system-level ETT Rohrer term -- treating
       each compartment as bearing its own share of ETT resistance
       rather than exactly modeling one shared conduit. Acceptable given
       the schematic nature of the simulation; flag for review if exact
       ETT-sharing physics is required later.

    4. Volume-dependent expiratory resistance -- expiratory R rises as
       lung volume falls (dynamic airway collapse), identical to
       psv_generator. Produces the biexponential expiratory decay in
       COPD.

    5. PEEP-recruited compliance -- condition-specific recruitment slope,
       identical to psv_generator (COPD/Bronchospasm = 0.0).

    6. Auto-PEEP carry-forward -- end-expiratory compartment volumes
       carry forward as the next breath's starting volumes (not reset to
       zero), producing breath-stacking / dynamic hyperinflation in
       COPD and Bronchospasm.

    7. Outer-loop multi-breath VT averaging -- the error signal driving
       the adaptation step is a moving average of the last
       `vt_averaging_window` breaths' delivered VT (default 2), not a
       single breath's, matching documented AutoFlow anti-hunting
       behavior. This is the PRVC-specific refinement with no VCV/PCV/PSV
       analogue.

    Deliberately NOT included (see PRVC parameter grid doc): breath-to-
    breath Pmus variability, patient-ventilator dyssynchrony, SBT
    sequencing, chest wall compliance, ETT complications, circuit
    compliance correction -- all optional/not-applicable for the purely
    mandatory PRVC mode as scoped. Available as no-op-by-default optional
    params for interface consistency with psv_generator.

adaptation_step_cmH2O and vt_tolerance_frac are UNIFORM constants across
all conditions in this implementation (see PARAMETER_GRID note) -- not
condition-specific -- reflecting a single deployed device algorithm that
does not know the patient's diagnosis in advance.

Interface contract (identical to vcv_generator, pcv_generator, psv_generator)
------------------------------------------------------------------------------
    generate_breath_cycles(params, n_cycles, seed) -> dict
    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list

Output dict keys
----------------
    Core waveforms (np.ndarray, 100 Hz):
        time, pressure, flow, volume

    Pressure decomposition (np.ndarray, same length):
        pressure_resistive, pressure_elastic, pressure_total_peep

    Per-breath trajectories (np.ndarray, length = n_cycles):
        pressure_trajectory, delivered_vt_trajectory

    Scalar metrics:
        ppeak_cmH2O, delivered_vt_ml, driving_p_cmH2O, mean_paw_cmH2O,
        auto_peep_cmH2O, fill_fraction, minute_vent_l,
        test_breath_plateau_cmH2O, breaths_to_converge, converged,
        ceiling_limited

    Validity:
        is_valid, invalid_reason

Run smoke test:
    python generator/prvc_generator.py
"""

import itertools
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Section 1 -- Parameter Grid
# ---------------------------------------------------------------------------

# Full parameter space -- ventilator-side. adaptation_step_cmH2O and
# vt_tolerance_frac are single-item lists deliberately: per project
# decision, these represent one uniform deployed-device algorithm and are
# NOT swept per condition (see module docstring and PRVC_PARAMETER_GRID
# doc for the C_threshold analysis behind this choice).
PARAMETER_GRID: Dict = {
    "vt_target_ml_per_kg":      [4, 6, 8, 10],
    "respiratory_rate":         [8, 12, 16, 20, 24, 28, 30],
    "peep_cmH2O":                [0, 4, 8, 12, 16, 20],
    "ie_ratio":                  [1.0, 0.5, 0.33],
    "pressure_ceiling_cmH2O":   [15, 20, 25, 30, 35],
    "adaptation_step_cmH2O":    [2.0],
    "vt_tolerance_frac":        [0.10],
}

# ---------------------------------------------------------------------------
# Section 2 -- Safety Thresholds and Constants
# ---------------------------------------------------------------------------

IBW_KG: float             = 70.0
VT_MIN_ML: float          = IBW_KG * 3       # 210 mL
VT_MAX_ML: float          = IBW_KG * 12      # 840 mL
PPEAK_MAX_CMHH2O: float   = 50.0
DT: float                 = 0.01             # 100 Hz internal simulation timestep

VT_MIN_ML_PER_KG_ADULT:    float = 3.0    # existing behavior, unchanged
VT_MAX_ML_PER_KG_ADULT:    float = 12.0
VT_MIN_ML_PER_KG_NEONATE:  float = 4.0    # lung-protective floor — Spaeth 2022 / neonatal consensus
VT_MAX_ML_PER_KG_NEONATE:  float = 8.0    # ceiling tighter than adult's 12x — ASSUMPTION, flag for review
NEONATE_IBW_KG_DEFAULT:    float = 3.0    # fallback only if weight_kg is somehow absent


# Rise time is fixed, not swept, for PRVC (brief's parameter table does
# not list rise time for PRVC -- see parameter grid doc section 1c).
RISE_TIME_S: float = 0.10

# Uniform outer-loop constants (defaults; overridable per params for
# testing, but the standard dataset generation uses these for every
# condition -- see PARAMETER_GRID note above).
ADAPTATION_STEP_CMH2O_DEFAULT: float = 2.0
VT_TOLERANCE_FRAC_DEFAULT: float     = 0.10
PRESSURE_FLOOR_ABOVE_PEEP: float     = 5.0   # never let working pressure collapse below this

# Outer-loop multi-breath damping (refinement #7 -- no VCV/PCV/PSV analogue)
VT_AVERAGING_WINDOW_DEFAULT: int = 2

# Test-breath bootstrap constants
C_ASSUMED_FALLBACK: float          = 30.0    # mL/cmH2O, closed-form fallback only
AUTOFLOW_TEST_BREATH_FRACTION: float = 0.75  # Dräger-documented re-application factor
TEST_BREATH_PAUSE_FRACTION: float    = 0.10  # Servo-S documented "10% pause time"

# Convergence stability: require this many consecutive in-tolerance
# breaths before declaring "converged" (rules out a lucky single crossing).
CONVERGENCE_STABILITY_BREATHS: int = 2

# Circuit compliance -- standard adult ICU circuit (optional refinement)
CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5

# Default chest wall compliance (effectively infinite / non-restrictive)
DEFAULT_CHEST_WALL_COMPLIANCE: float = 250.0  # mL/cmH2O

# Rohrer ETT contribution (7.5 mm ID tube) -- see refinement #3 note
ETT_K1: float = 5.0   # cmH2O/L/s
ETT_K2: float = 3.0   # cmH2O/(L/s)^2
# ---------------------------------------------------------------------------
# Section 2b — Neonatal population constants (only 3 — see CR0023)
# ---------------------------------------------------------------------------
NEONATE_PPEAK_MAX_CMHH2O:                float = 30.0  # neonatal barotrauma risk — MSD Manual PIP ranges
NEONATE_DEFAULT_CHEST_WALL_COMPLIANCE:   float = 12.0  # NOT ~inert — first-order term for this population
NEONATE_CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 0.6   # dedicated low-compliance neonatal circuit



def _neonate_or_adult(population: str, neonate_val, adult_val):
    """Return neonate_val if population == 'neonate', else adult_val.
    Works for any type — floats, None, whatever a given constant needs."""
    return neonate_val if population == "neonate" else adult_val
# ---------------------------------------------------------------------------
# Section 3 -- Condition-Specific Profiles
# ---------------------------------------------------------------------------
# Identical to psv_generator.py -- PRVC reuses the same compartment
# architecture (see PRVC parameter grid doc, section 1f).

COMPARTMENT_PROFILES: Dict = {
    "Normal": [
        {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
         "R_exp_ratio": 1.2, "tethering": 0.80},
    ],
    "Mild ARDS": [
        {"fraction": 0.75, "C_frac": 0.90, "R_frac": 1.00,
         "R_exp_ratio": 1.4, "tethering": 0.40},
        {"fraction": 0.25, "C_frac": 0.10, "R_frac": 1.60,
         "R_exp_ratio": 2.0, "tethering": 0.10},
    ],
    "Moderate ARDS": [
        {"fraction": 0.60, "C_frac": 0.85, "R_frac": 1.00,
         "R_exp_ratio": 1.6, "tethering": 0.25},
        {"fraction": 0.40, "C_frac": 0.05, "R_frac": 1.80,
         "R_exp_ratio": 2.5, "tethering": 0.08},
    ],
    "Severe ARDS": [
        {"fraction": 0.40, "C_frac": 0.80, "R_frac": 1.00,
         "R_exp_ratio": 1.8, "tethering": 0.20},
        {"fraction": 0.60, "C_frac": 0.03, "R_frac": 2.00,
         "R_exp_ratio": 3.0, "tethering": 0.05},
    ],
    "COPD": [
        {"fraction": 0.35, "C_frac": 0.70, "R_frac": 0.55,
         "R_exp_ratio": 4.0, "tethering": 0.15},
        {"fraction": 0.40, "C_frac": 1.05, "R_frac": 1.27,
         "R_exp_ratio": 6.0, "tethering": 0.10},
        {"fraction": 0.25, "C_frac": 1.40, "R_frac": 2.36,
         "R_exp_ratio": 8.0, "tethering": 0.05},
    ],
    "Bronchospasm": [
        {"fraction": 0.60, "C_frac": 0.90, "R_frac": 0.80,
         "R_exp_ratio": 3.0, "tethering": 0.00},
        {"fraction": 0.40, "C_frac": 1.10, "R_frac": 1.43,
         "R_exp_ratio": 5.0, "tethering": 0.00},
    ],
    "Pneumonia": [
        {"fraction": 0.60, "C_frac": 1.10, "R_frac": 0.83,
         "R_exp_ratio": 1.5, "tethering": 0.70},
        {"fraction": 0.25, "C_frac": 0.55, "R_frac": 1.83,
         "R_exp_ratio": 3.0, "tethering": 0.30},
        {"fraction": 0.15, "C_frac": 0.07, "R_frac": 6.67,
         "R_exp_ratio": 2.0, "tethering": 0.10},
    ],
    "Normal Neonate": [
    {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
     "R_exp_ratio": 1.2, "tethering": 0.80},   # identical shape to adult Normal
    ],
    "RDS": [
        {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
        "R_exp_ratio": 1.3, "tethering": 0.30},   # single compartment — OPEN DECISION, see below
    ],
    
}

RECRUITMENT_SLOPES: Dict = {
    "Normal":        0.00,
    "Mild ARDS":     0.50,
    "Moderate ARDS": 0.90,
    "Severe ARDS":   0.60,
    "COPD":          0.00,
    "Bronchospasm":  0.00,
    "Pneumonia":     0.10,
    "Normal Neonate":               0.30,   # ASSUMPTION — modest PEEP recruitment, like adult Normal
    "RDS":                          0.60,   # higher than adult ARDS — RDS is the textbook recruitable lung
}

# Condition tiers used by generate_dataset() / thinned-script style sweeps.
# Identical to the CONDITION_TIERS shared across vcv/pcv/psv thinned
# scripts (corrected resistance floors -- see parameter grid doc 1d).
CONDITION_TIERS: List[Dict] = [
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
    {"name": "Normal Neonate",              "compliance_range": (3, 6),    "compliance_step": 0.5,
    "resistance_range": (60, 100),  "resistance_step": 10, "n_cycles": 15},
    {"name": "RDS",                         "compliance_range": (0.4, 1.2), "compliance_step": 0.1,
    "resistance_range": (60, 100),  "resistance_step": 10, "n_cycles": 15},

]

REQUIRED_PARAMS = [
    "vt_target_ml", "respiratory_rate", "peep_cmH2O", "ie_ratio",
    "pressure_ceiling_cmH2O", "compliance_ml_per_cmH2O", "resistance_cmH2O_L_s",
]


# ---------------------------------------------------------------------------
# Section 4 -- Physics Functions
# ---------------------------------------------------------------------------

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


def _rohrer(Q_L_s: float, K1: float, K2: float) -> float:
    """Rohrer flow-dependent resistive pressure drop. Q in L/s, returns cmH2O."""
    return K1 * Q_L_s + K2 * Q_L_s * abs(Q_L_s)


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
    circuit_loss = C_circ * (ppeak - peep)
    return max(0.0, vt_mL - circuit_loss)


# ---------------------------------------------------------------------------
# Section 5 -- Compartment setup helper
# ---------------------------------------------------------------------------

def _build_compartments(condition: str, C_global: float, R_global: float,
                         peep: float, peep_ref: float, rec_slope: float,
                         C_chest: float) -> Dict:
    """
    Build per-compartment C/R arrays from the condition profile, applying
    PEEP-recruited compliance and chest-wall series compliance at the
    global level before splitting across compartments (matching the
    compartment normalization formula established in psv_generator:
    C_lung_rec * C_frac_arr * fractions / C_frac_norm).
    """
    profile = COMPARTMENT_PROFILES.get(condition, COMPARTMENT_PROFILES["Normal"])
    n_comps = len(profile)

    fractions = np.array([c["fraction"] for c in profile])
    C_frac_arr = np.array([c["C_frac"] for c in profile])
    R_frac_arr = np.array([c["R_frac"] for c in profile])
    R_exp_arr = np.array([c["R_exp_ratio"] for c in profile])
    teth_arr = np.array([c["tethering"] for c in profile])
    C_frac_norm = float(np.dot(C_frac_arr, fractions))

    C_lung_rec = _peep_recruited_compliance(C_global, peep, peep_ref, rec_slope)
    C_lung_rec = _C_rs(C_lung_rec, C_chest)

    C_comps_base = C_lung_rec * C_frac_arr * fractions / max(C_frac_norm, 0.01)
    R_comps_base = R_global * R_frac_arr

    return {
        "n_comps": n_comps,
        "fractions": fractions,
        "C_base": C_comps_base,
        "R_base": R_comps_base,
        "R_exp_ratio": R_exp_arr,
        "tethering": teth_arr,
    }


# ---------------------------------------------------------------------------
# Section 6 -- Test breath (volume-controlled bootstrap)
# ---------------------------------------------------------------------------

def _run_vc_test_breath(comps: Dict, vt_target_ml: float, peep: float,
                         t_insp: float, stress_index: float,
                         K1_frac: float, K2_frac: float) -> Tuple[np.ndarray, ...]:
    """
    Deliver a constant-total-flow, multi-compartment volume-controlled
    breath (algebraic branch-point solve each timestep), then compute the
    equilibrated plateau pressure via parallel-compliance summation
    (physically: at zero net flow, parallel compartments connected to a
    common airway equalize pressure by internal redistribution --
    pendelluft -- conserving total volume; for parallel compliances this
    equilibrium pressure is V_total / sum(C_i), the same relationship as
    capacitors in parallel. The transient redistribution itself is not
    simulated -- only the equilibrium result is used for P_work(2).)

    Returns (t, P, Q, V) arrays covering inspiration + a synthetic pause
    segment, plus the scalar plateau pressure and end-inspiratory volumes.
    """
    n_comps = comps["n_comps"]
    C_base = comps["C_base"]
    R_base = comps["R_base"]

    V = np.zeros(n_comps)
    Q_total_target = (vt_target_ml / 1000.0) / max(t_insp, 0.05)  # L/s

    n_steps = max(2, int(round(t_insp / DT)))
    t_list, P_list, Q_list, V_list = [], [], [], []
    t_now = 0.0

    for _ in range(n_steps):
        C_eff = np.array([
            _compliance_nonlinear(V[i], C_base[i], vt_target_ml * comps["fractions"][i] * 0.6,
                                   stress_index)
            for i in range(n_comps)
        ])
        C_eff = np.maximum(C_eff, 0.5)
        R_eff = R_base.copy()

        inv_R = 1.0 / np.maximum(R_eff, 0.1)
        Pao = (Q_total_target + np.sum(V / (C_eff * R_eff)) + peep * np.sum(inv_R)) / np.sum(inv_R)
        Q_i = (Pao - V / C_eff - peep) / R_eff  # L/s

        V = V + Q_i * DT * 1000.0
        V = np.maximum(V, 0.0)

        t_list.append(t_now)
        P_list.append(Pao)
        Q_list.append(float(np.sum(Q_i)))
        V_list.append(float(np.sum(V)))
        t_now += DT

    V_end_insp = V.copy()
    C_eff_end = np.array([
        _compliance_nonlinear(V_end_insp[i], C_base[i],
                               vt_target_ml * comps["fractions"][i] * 0.6, stress_index)
        for i in range(n_comps)
    ])
    C_eff_end = np.maximum(C_eff_end, 0.5)
    C_total_end = float(np.sum(C_eff_end))
    V_total_end = float(np.sum(V_end_insp))
    P_plat = peep + V_total_end / max(C_total_end, 0.5)

    # Synthetic pause segment for waveform continuity/realism
    pause_dur = t_insp * TEST_BREATH_PAUSE_FRACTION
    n_pause = max(1, int(round(pause_dur / DT)))
    for _ in range(n_pause):
        t_list.append(t_now)
        P_list.append(P_plat)
        Q_list.append(0.0)
        V_list.append(V_total_end)
        t_now += DT

    return (np.array(t_list), np.array(P_list), np.array(Q_list), np.array(V_list),
            P_plat, V_end_insp, t_now)


# ---------------------------------------------------------------------------
# Section 7 -- Single pressure-controlled breath (inner loop)
# ---------------------------------------------------------------------------

def _run_pc_breath(comps: Dict, V_start: np.ndarray, P_work: float, peep: float,
                    t_insp: float, t_exp: float, rise_time: float,
                    stress_index: float) -> Tuple[np.ndarray, ...]:
    """
    Run one full inspiration+expiration cycle at a fixed working pressure
    P_work, starting from V_start (auto-PEEP carry-forward), multi-
    compartment explicit Euler at 100 Hz.
    """
    n_comps = comps["n_comps"]
    C_base = comps["C_base"]
    R_base = comps["R_base"]
    R_exp_ratio = comps["R_exp_ratio"]
    tethering = comps["tethering"]
    fractions = comps["fractions"]

    V = V_start.copy()
    V_target_per_comp = np.maximum(V_start + 50.0 * fractions, 50.0)  # ref for tethering/nonlin

    t_list, P_list, Q_list, V_list = [], [], [], []
    t_now = 0.0

    # ---- Inspiration: rise + plateau ----
    n_insp = max(2, int(round(t_insp / DT)))
    rise_steps = max(1, int(round(rise_time / DT)))

    V_end_insp = None
    for step in range(n_insp):
        if step < rise_steps and rise_time > 1e-6:
            frac = (step + 1) / rise_steps
            P_applied = peep + frac * (P_work - peep)
        else:
            P_applied = P_work

        C_eff = np.array([
            _compliance_nonlinear(V[i], C_base[i], V_target_per_comp[i], stress_index)
            for i in range(n_comps)
        ])
        C_eff = np.maximum(C_eff, 0.5)
        R_eff = np.array([
            _R_insp_with_tethering(R_base[i], V[i], V_target_per_comp[i], tethering[i])
            for i in range(n_comps)
        ])

        Q_i = (P_applied - V / C_eff - peep) / np.maximum(R_eff, 0.1)  # L/s
        V = np.maximum(V + Q_i * DT * 1000.0, 0.0)

        t_list.append(t_now)
        P_list.append(P_applied)
        Q_list.append(float(np.sum(Q_i)))
        V_list.append(float(np.sum(V)))
        t_now += DT

    V_end_insp = V.copy()

    # ---- Expiration: passive, dynamic expiratory resistance ----
    n_exp = max(2, int(round(t_exp / DT)))
    for _ in range(n_exp):
        C_eff = np.array([
            _compliance_nonlinear(V[i], C_base[i], V_target_per_comp[i], stress_index)
            for i in range(n_comps)
        ])
        C_eff = np.maximum(C_eff, 0.5)
        R_insp_now = np.array([
            _R_insp_with_tethering(R_base[i], V[i], V_target_per_comp[i], tethering[i])
            for i in range(n_comps)
        ])
        R_exp_eff = np.array([
            _R_exp_dynamic(V[i], max(V_end_insp[i], 1.0), R_insp_now[i], R_exp_ratio[i])
            for i in range(n_comps)
        ])

        Q_i = (peep - V / C_eff - peep) / np.maximum(R_exp_eff, 0.1)  # L/s, negative (exhaling)
        V = np.maximum(V + Q_i * DT * 1000.0, 0.0)

        t_list.append(t_now)
        P_list.append(peep)
        Q_list.append(float(np.sum(Q_i)))
        V_list.append(float(np.sum(V)))
        t_now += DT

    return (np.array(t_list), np.array(P_list), np.array(Q_list), np.array(V_list),
            V_end_insp, V.copy(), t_now)


# ---------------------------------------------------------------------------
# Section 8 -- Validation
# ---------------------------------------------------------------------------

def _validate_params(params: Dict) -> None:
    missing = [k for k in REQUIRED_PARAMS if k not in params]
    if missing:
        raise ValueError(f"Missing required PRVC params: {missing}")
    if params["compliance_ml_per_cmH2O"] <= 0:
        raise ValueError("compliance_ml_per_cmH2O must be positive")
    if params["resistance_cmH2O_L_s"] <= 0:
        raise ValueError("resistance_cmH2O_L_s must be positive")
    if params["ie_ratio"] <= 0:
        raise ValueError("ie_ratio must be positive")


def _make_scenario_id(condition: str, C: float, R: float, params: Dict) -> str:
    ie = params.get("ie_ratio", 0.0)
    ie_tag = f"IE{round(ie * 100):03d}"
    return (f"PRVC_{condition.replace(' ', '')}_C{C:.0f}_R{R:.0f}_"
            f"VT{params.get('vt_target_ml', 0):.0f}_"
            f"RR{params.get('respiratory_rate', 0)}_"
            f"PEEP{params.get('peep_cmH2O', 0)}_"
            f"{ie_tag}_"
            f"CEIL{params.get('pressure_ceiling_cmH2O', 0)}")


# ---------------------------------------------------------------------------
# Section 9 -- Main generator
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: Dict, n_cycles: int = 12, seed: int = 0) -> Dict:
    """
    Generate a PRVC multi-breath sequence: one VC test breath followed by
    n_cycles-1 pressure-controlled breaths under outer-loop adaptive
    control. See module docstring for control loop details.
    """
    _validate_params(params)

    vt_target = float(params["vt_target_ml"])
    rr = float(params["respiratory_rate"])
    peep = float(params["peep_cmH2O"])
    ie_ratio = float(params["ie_ratio"])
    pressure_ceiling = float(params["pressure_ceiling_cmH2O"])
    C_global = float(params["compliance_ml_per_cmH2O"])
    R_global = float(params["resistance_cmH2O_L_s"])

    condition = params.get("condition", "Normal")
    population = params.get("population", "adult")
    weight_kg  = float(params.get("weight_kg", NEONATE_IBW_KG_DEFAULT if population == "neonate" else IBW_KG))
    if population == "neonate":
        weight = float(params.get("weight_kg", NEONATE_IBW_KG_DEFAULT))
        vt_min_ml = weight * VT_MIN_ML_PER_KG_NEONATE
        vt_max_ml = weight * VT_MAX_ML_PER_KG_NEONATE
    else:
        vt_min_ml = IBW_KG * VT_MIN_ML_PER_KG_ADULT   # identical to current VT_MIN_ML
        vt_max_ml = IBW_KG * VT_MAX_ML_PER_KG_ADULT
    adaptation_step = float(params.get("adaptation_step_cmH2O", ADAPTATION_STEP_CMH2O_DEFAULT))
    vt_tolerance_frac = float(params.get("vt_tolerance_frac", VT_TOLERANCE_FRAC_DEFAULT))
    rise_time = float(params.get("rise_time_s", RISE_TIME_S))
    vt_avg_window = int(params.get("vt_averaging_window", VT_AVERAGING_WINDOW_DEFAULT))
    use_vc_test_breath = bool(params.get("use_vc_test_breath", True))

    stress_index = float(params.get("stress_index", 1.0))
    C_chest          = float(params.get(
        "chest_wall_compliance_ml_per_cmH2O",
        _neonate_or_adult(population, NEONATE_DEFAULT_CHEST_WALL_COMPLIANCE, DEFAULT_CHEST_WALL_COMPLIANCE),
    ))
    ppeak_max = _neonate_or_adult(population, NEONATE_PPEAK_MAX_CMHH2O, PPEAK_MAX_CMHH2O)
    circuit_c = _neonate_or_adult(population, NEONATE_CIRCUIT_COMPLIANCE_ML_PER_CMH2O, CIRCUIT_COMPLIANCE_ML_PER_CMH2O)
    vt_min_ml = weight_kg * _neonate_or_adult(population, VT_MIN_ML_PER_KG_NEONATE, VT_MIN_ML_PER_KG_ADULT)
    peep_ref = float(params.get("peep_reference_cmH2O", 5.0))
    rec_slope = float(params.get("recruitment_slope", RECRUITMENT_SLOPES.get(condition, 0.5)))
    circ_compensated = bool(params.get("circuit_compensated", True))

    if rr <= 0:
        raise ValueError("respiratory_rate must be positive")
    t_cycle = 60.0 / rr
    t_insp = t_cycle * ie_ratio / (1.0 + ie_ratio)
    t_exp = t_cycle - t_insp

    comps = _build_compartments(condition, C_global, R_global, peep, peep_ref,
                                 rec_slope, C_chest)

    T_list, P_list, Q_list, V_list = [], [], [], []
    pressure_trajectory = np.zeros(n_cycles)
    delivered_vt_trajectory = np.zeros(n_cycles)

    t_offset = 0.0
    P_work = peep + (vt_target / C_ASSUMED_FALLBACK)  # closed-form fallback seed
    V_carry = np.zeros(comps["n_comps"])
    test_breath_plateau = None

    converged = False
    ceiling_limited = False
    breaths_to_converge: Optional[int] = None
    stable_count = 0
    recent_vts: List[float] = []

    for n in range(n_cycles):
        breath_num = n + 1
        is_maneuver_breath = (breath_num == 1 and use_vc_test_breath)

        if is_maneuver_breath:
            t, P, Q, V, P_plat, V_end_insp, dur = _run_vc_test_breath(
                comps, vt_target, peep, t_insp, stress_index,
                K1_frac=0.60, K2_frac=0.04,
            )
            test_breath_plateau = P_plat
            P_work_this_breath = P_plat  # what breath 1 actually operated at

            # Passive expiration to complete the cycle for waveform continuity
            n_exp = max(2, int(round(t_exp / DT)))
            V_exp = V_end_insp.copy()
            R_exp_arr = comps["R_exp_ratio"]
            R_base = comps["R_base"]
            for _ in range(n_exp):
                C_eff = np.array([
                    _compliance_nonlinear(V_exp[i], comps["C_base"][i],
                                           vt_target * comps["fractions"][i] * 0.6,
                                           stress_index)
                    for i in range(comps["n_comps"])
                ])
                C_eff = np.maximum(C_eff, 0.5)
                R_exp_eff = R_base * R_exp_arr
                Q_i = (peep - V_exp / C_eff - peep) / np.maximum(R_exp_eff, 0.1)
                V_exp = np.maximum(V_exp + Q_i * DT * 1000.0, 0.0)
                t = np.append(t, t[-1] + DT)
                P = np.append(P, peep)
                Q = np.append(Q, float(np.sum(Q_i)))
                V = np.append(V, float(np.sum(V_exp)))
            V_carry = V_exp.copy()
            delivered_vt = float(np.sum(V_end_insp))

            # Seed P_work(2) via the AutoFlow rule (special case -- not the
            # standard step/tolerance rule, which only governs breaths >= 2).
            driving_plat = max(P_plat - peep, 0.0)
            P_work = peep + AUTOFLOW_TEST_BREATH_FRACTION * driving_plat
            P_work = float(np.clip(P_work, peep + PRESSURE_FLOOR_ABOVE_PEEP,
                                    peep + pressure_ceiling))

        else:
            P_work_this_breath = P_work
            t, P, Q, V, V_end_insp, V_exp_end, dur = _run_pc_breath(
                comps, V_carry, P_work_this_breath, peep, t_insp, t_exp, rise_time, stress_index,
            )
            V_carry = V_exp_end.copy()
            delivered_vt = float(np.sum(V_end_insp))

        T_list.append(t + t_offset)
        P_list.append(P)
        Q_list.append(Q)
        V_list.append(V)
        t_offset += t[-1] + DT if len(t) else t_cycle

        pressure_trajectory[n] = P_work_this_breath
        delivered_vt_trajectory[n] = delivered_vt

        # ---- Outer loop: update P_work for the NEXT breath ----
        # The VC test breath delivers vt_target by construction (it's
        # flow-prescribed), so it is excluded from convergence/stability
        # tracking and from the standard step rule -- it hasn't been
        # adaptively controlled yet. Its seed for breath 2 was already set
        # above via the AutoFlow rule.
        if not is_maneuver_breath:
            recent_vts.append(delivered_vt)
            if len(recent_vts) > vt_avg_window:
                recent_vts.pop(0)
            avg_vt = float(np.mean(recent_vts))
            error_frac = (vt_target - avg_vt) / max(vt_target, 1.0)

            in_tolerance = abs(error_frac) <= vt_tolerance_frac
            if in_tolerance:
                stable_count += 1
                if stable_count >= CONVERGENCE_STABILITY_BREATHS and breaths_to_converge is None:
                    breaths_to_converge = breath_num
                    converged = True
            else:
                stable_count = 0

            if breath_num < n_cycles:
                if not in_tolerance:
                    step = adaptation_step if error_frac > 0 else -adaptation_step
                    P_work_next = P_work_this_breath + step
                else:
                    P_work_next = P_work_this_breath
                P_work_ceiling = peep + pressure_ceiling
                P_work_floor = peep + PRESSURE_FLOOR_ABOVE_PEEP
                P_work = float(np.clip(P_work_next, P_work_floor, P_work_ceiling))

    if not converged:
        final_vt = delivered_vt_trajectory[-1]
        final_error = (vt_target - final_vt) / max(vt_target, 1.0)
        at_ceiling = pressure_trajectory[-1] >= (peep + pressure_ceiling - 1e-6)
        if at_ceiling and final_error > vt_tolerance_frac:
            ceiling_limited = True

    time_arr = np.concatenate(T_list)
    pressure_arr = np.concatenate(P_list)
    flow_arr = np.concatenate(Q_list)
    volume_arr = np.concatenate(V_list)

    ppeak = float(np.max(pressure_arr))
    ppeak_final_breath = float(np.max(P_list[-1]))
    driving_p = float(pressure_trajectory[-1] - peep)
    mean_paw = float(np.mean(pressure_arr))
    delivered_vt_final = float(delivered_vt_trajectory[-1])
    delivered_vt_final = _circuit_vt_correction(
        delivered_vt_final, ppeak_final_breath, peep,
        C_circ=circuit_c, compensated=circ_compensated,
    )
    cuff_leak_frac = float(params.get("ett_cuff_leak_fraction", 0.0))
    delivered_vt_final = max(0.0, delivered_vt_final * (1.0 - cuff_leak_frac))
    auto_peep = float(np.mean(V_carry) / max(np.mean(comps["C_base"]), 0.5)) if n_cycles > 1 else 0.0
    minute_vent = delivered_vt_final * rr / 1000.0
    fill_fraction = float(np.clip(delivered_vt_final / max(vt_target, 1.0), 0.0, 1.5))

    population = params.get("population", "adult")
    weight_kg  = float(params.get("weight_kg",
                     NEONATE_IBW_KG_DEFAULT if population == "neonate" else IBW_KG))
    ppeak_max  = _neonate_or_adult(population, NEONATE_PPEAK_MAX_CMHH2O, PPEAK_MAX_CMHH2O)
    vt_min_ml  = weight_kg * _neonate_or_adult(
        population, VT_MIN_ML_PER_KG_NEONATE, VT_MIN_ML_PER_KG_ADULT)

    is_valid = True
    invalid_reason = None
    if ppeak > ppeak_max:
        is_valid = False
        invalid_reason = f"Ppeak {ppeak:.1f} exceeds barotrauma threshold {ppeak_max}"
    elif converged and delivered_vt_final < vt_min_ml:
        is_valid = False
        invalid_reason = f"Converged delivered Vt {delivered_vt_final:.0f} mL below minimum {vt_min_ml:.0f} mL"
    elif converged and population != "neonate" and delivered_vt_final > VT_MAX_ML:
        is_valid = False
        invalid_reason = f"Converged delivered Vt {delivered_vt_final:.0f} mL exceeds maximum {VT_MAX_ML:.0f} mL"
    # Ceiling-limited non-convergence is retained as a valid, labeled scenario
    # (see parameter grid doc 1e) -- not hard-invalidated.

    return {
        "time": time_arr,
        "pressure": pressure_arr,
        "flow": flow_arr,
        "volume": volume_arr,
        "pressure_trajectory": pressure_trajectory,
        "delivered_vt_trajectory": delivered_vt_trajectory,
        "ppeak_cmH2O": ppeak,
        "ppeak_final_breath_cmH2O": ppeak_final_breath, 
        "delivered_vt_ml": delivered_vt_final,
        "driving_p_cmH2O": driving_p,
        "mean_paw_cmH2O": mean_paw,
        "auto_peep_cmH2O": auto_peep,
        "fill_fraction": fill_fraction,
        "minute_vent_l": minute_vent,
        "test_breath_plateau_cmH2O": test_breath_plateau,
        "breaths_to_converge": breaths_to_converge,
        "converged": converged,
        "ceiling_limited": ceiling_limited,
        "is_valid": is_valid,
        "invalid_reason": invalid_reason,
    }


def generate_dataset(condition_name: str, compliance_ml_per_cmH2O: float,
                      resistance_cmH2O_L_s: float, n_cycles: int = 12,
                      max_scenarios: Optional[int] = None) -> List[Dict]:
    """
    Sweep PARAMETER_GRID's ventilator-side dimensions (excluding the
    uniform adaptation_step/vt_tolerance_frac) for one condition +
    mechanics pair. Full sweep is 4x7x6x3x5 = 2,520 combinations per
    mechanics point -- pass max_scenarios to cap this for smoke tests or
    quick checks; leave None for actual dataset generation.

    Returns
    -------
    list of dicts, one per parameter combination. Each dict contains:
        "scenario_id"    : str
        "condition"      : str
        "params"         : dict -- full parameter set used
        "metrics"        : dict -- derived clinical metrics (scalars only)
        "is_valid"       : bool
        "invalid_reason" : str or None
        "waveforms"      : dict -- time, pressure, flow, volume,
                            pressure_trajectory, delivered_vt_trajectory
                            (empty dict for invalid/errored scenarios)
        "generated_at"   : str -- ISO timestamp
    """
    keys = ["vt_target_ml_per_kg", "respiratory_rate", "peep_cmH2O",
            "ie_ratio", "pressure_ceiling_cmH2O"]
    combos = list(itertools.product(*[PARAMETER_GRID[k] for k in keys]))
    if max_scenarios is not None:
        combos = combos[:max_scenarios]

    scenarios = []
    for combo in combos:
        vent = dict(zip(keys, combo))
        params = {
            "vt_target_ml": vent["vt_target_ml_per_kg"] * IBW_KG,
            "respiratory_rate": vent["respiratory_rate"],
            "peep_cmH2O": vent["peep_cmH2O"],
            "ie_ratio": vent["ie_ratio"],
            "pressure_ceiling_cmH2O": vent["pressure_ceiling_cmH2O"],
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s": resistance_cmH2O_L_s,
            "condition": condition_name,
            "adaptation_step_cmH2O": PARAMETER_GRID["adaptation_step_cmH2O"][0],
            "vt_tolerance_frac": PARAMETER_GRID["vt_tolerance_frac"][0],
        }
        seed = abs(hash((condition_name, compliance_ml_per_cmH2O,
                          resistance_cmH2O_L_s, tuple(combo)))) % (2**31)
        scenario_id = _make_scenario_id(condition_name, compliance_ml_per_cmH2O,
                                         resistance_cmH2O_L_s, params)

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles, seed=seed)
        except Exception as exc:
            scenarios.append({
                "scenario_id": scenario_id,
                "condition": condition_name,
                "params": params,
                "metrics": {},
                "is_valid": False,
                "invalid_reason": f"Generator error: {exc}",
                "waveforms": {},
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            continue

        metrics = {
            "ppeak_cmH2O": result["ppeak_cmH2O"],
            "delivered_vt_ml": result["delivered_vt_ml"],
            "driving_p_cmH2O": result["driving_p_cmH2O"],
            "mean_paw_cmH2O": result["mean_paw_cmH2O"],
            "auto_peep_cmH2O": result["auto_peep_cmH2O"],
            "fill_fraction": result["fill_fraction"],
            "minute_vent_l": result["minute_vent_l"],
            "test_breath_plateau_cmH2O": result["test_breath_plateau_cmH2O"],
            "breaths_to_converge": result["breaths_to_converge"],
            "converged": result["converged"],
            "ceiling_limited": result["ceiling_limited"],
        }

        if result["is_valid"]:
            waveforms = {
                "time": result["time"],
                "pressure": result["pressure"],
                "flow": result["flow"],
                "volume": result["volume"],
                "pressure_trajectory": result["pressure_trajectory"],
                "delivered_vt_trajectory": result["delivered_vt_trajectory"],
            }
        else:
            waveforms = {}

        scenarios.append({
            "scenario_id": scenario_id,
            "condition": condition_name,
            "params": params,
            "metrics": metrics,
            "is_valid": result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "waveforms": waveforms,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    return scenarios


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _check(label: str, condition: bool, detail: str = "") -> bool:
    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    mark = f"{GREEN}PASS{RESET}" if condition else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    return condition


if __name__ == "__main__":
    all_pass = True

    base = {
        "vt_target_ml": 420.0,
        "respiratory_rate": 16.0,
        "peep_cmH2O": 5.0,
        "ie_ratio": 0.5,
        "pressure_ceiling_cmH2O": 25.0,
    }

    # ---- [1/4] Normal -- fast, clean convergence at low pressure ----
    print("\n[1/4] Normal -- fast convergence, low driving pressure")
    p_normal = {**base, "compliance_ml_per_cmH2O": 80.0, "resistance_cmH2O_L_s": 10.0,
                "condition": "Normal"}
    r_normal = generate_breath_cycles(p_normal, n_cycles=10, seed=1)
    all_pass &= _check("no exception / dict returned", isinstance(r_normal, dict))
    all_pass &= _check("waveform arrays non-empty", len(r_normal["time"]) > 0)
    all_pass &= _check("test breath plateau recorded",
                        r_normal["test_breath_plateau_cmH2O"] is not None)
    all_pass &= _check("converges", r_normal["converged"] is True,
                        f"breaths_to_converge={r_normal['breaths_to_converge']}")
    all_pass &= _check("low driving pressure at convergence",
                        r_normal["driving_p_cmH2O"] < 15.0,
                        f"driving_p={r_normal['driving_p_cmH2O']:.1f}")
    all_pass &= _check("is_valid", r_normal["is_valid"] is True, str(r_normal["invalid_reason"]))
    print(f"     Ppeak={r_normal['ppeak_cmH2O']:.1f} Vt={r_normal['delivered_vt_ml']:.0f} "
          f"driving_p={r_normal['driving_p_cmH2O']:.1f} "
          f"breaths_to_converge={r_normal['breaths_to_converge']}")

    # ---- [2/4] Severe ARDS -- staircase climbs, likely ceiling-limited ----
    print("\n[2/4] Severe ARDS -- climbing staircase, low ceiling")
    p_severe = {**base, "compliance_ml_per_cmH2O": 18.0, "resistance_cmH2O_L_s": 16.0,
                "condition": "Severe ARDS", "pressure_ceiling_cmH2O": 15.0}
    r_severe = generate_breath_cycles(p_severe, n_cycles=12, seed=2)
    all_pass &= _check("test breath (breath 1) plateau reveals high true pressure need",
                        r_severe["pressure_trajectory"][0] > p_severe["peep_cmH2O"] +
                        p_severe["pressure_ceiling_cmH2O"],
                        f"breath1={r_severe['pressure_trajectory'][0]:.1f} "
                        f"ceiling={p_severe['peep_cmH2O'] + p_severe['pressure_ceiling_cmH2O']:.1f}")
    all_pass &= _check("adaptive breaths (2+) are flat at the ceiling, not still climbing",
                        bool(np.allclose(r_severe["pressure_trajectory"][1:],
                                          r_severe["pressure_trajectory"][-1], atol=0.01)),
                        str(r_severe["pressure_trajectory"][1:]))
    all_pass &= _check("pinned at or near ceiling",
                        r_severe["pressure_trajectory"][-1] >=
                        p_severe["peep_cmH2O"] + p_severe["pressure_ceiling_cmH2O"] - 0.5,
                        f"final={r_severe['pressure_trajectory'][-1]:.1f}")
    all_pass &= _check("scenario retained (not hard-invalidated by ceiling failure)",
                        r_severe["is_valid"] is True or r_severe["ceiling_limited"],
                        str(r_severe["invalid_reason"]))
    print(f"     converged={r_severe['converged']} ceiling_limited={r_severe['ceiling_limited']} "
          f"final_Vt={r_severe['delivered_vt_trajectory'][-1]:.0f} target={p_severe['vt_target_ml']:.0f}")

    # ---- [3/4] Moderate ARDS -- PEEP sensitivity ----
    print("\n[3/4] Moderate ARDS -- PEEP sensitivity (recruitment slope 0.90)")
    p_mod_lowpeep = {**base, "compliance_ml_per_cmH2O": 32.0, "resistance_cmH2O_L_s": 14.0,
                      "condition": "Moderate ARDS", "peep_cmH2O": 5.0,
                      "pressure_ceiling_cmH2O": 30.0}
    p_mod_hipeep = {**p_mod_lowpeep, "peep_cmH2O": 15.0}
    r_mod_low = generate_breath_cycles(p_mod_lowpeep, n_cycles=12, seed=3)
    r_mod_hi = generate_breath_cycles(p_mod_hipeep, n_cycles=12, seed=3)
    all_pass &= _check("higher PEEP converges at lower or equal driving pressure",
                        r_mod_hi["driving_p_cmH2O"] <= r_mod_low["driving_p_cmH2O"] + 0.5,
                        f"low_peep_driving={r_mod_low['driving_p_cmH2O']:.1f} "
                        f"hi_peep_driving={r_mod_hi['driving_p_cmH2O']:.1f}")
    all_pass &= _check("breath 2 deliberately undershoots the test breath's plateau (AutoFlow rule)",
                        r_mod_low["pressure_trajectory"][1] < r_mod_low["pressure_trajectory"][0],
                        str(r_mod_low["pressure_trajectory"][:2]))
    all_pass &= _check("genuine multi-step climb from breath 2 to convergence",
                        r_mod_low["pressure_trajectory"][r_mod_low["breaths_to_converge"] - 1] >
                        r_mod_low["pressure_trajectory"][1] + 1.0,
                        str(r_mod_low["pressure_trajectory"]))
    all_pass &= _check("delivered Vt converges toward target (not stuck undershooting)",
                        abs(r_mod_low["delivered_vt_ml"] - p_mod_lowpeep["vt_target_ml"]) <
                        0.15 * p_mod_lowpeep["vt_target_ml"],
                        f"delivered={r_mod_low['delivered_vt_ml']:.0f} target={p_mod_lowpeep['vt_target_ml']:.0f}")
    print(f"     low PEEP: driving_p={r_mod_low['driving_p_cmH2O']:.1f} "
          f"converged={r_mod_low['converged']} | "
          f"high PEEP: driving_p={r_mod_hi['driving_p_cmH2O']:.1f} "
          f"converged={r_mod_hi['converged']}")

    # ---- [4/4] COPD -- multi-compartment integrity + auto-PEEP ----
    print("\n[4/4] COPD -- multi-compartment integrity, auto-PEEP")
    p_copd = {**base, "compliance_ml_per_cmH2O": 100.0, "resistance_cmH2O_L_s": 25.0,
              "condition": "COPD", "respiratory_rate": 22.0, "pressure_ceiling_cmH2O": 30.0}
    r_copd = generate_breath_cycles(p_copd, n_cycles=25, seed=4)
    all_pass &= _check("array lengths consistent",
                        len(r_copd["time"]) == len(r_copd["pressure"]) ==
                        len(r_copd["flow"]) == len(r_copd["volume"]))
    all_pass &= _check("time array strictly increasing",
                        bool(np.all(np.diff(r_copd["time"]) > 0)))
    all_pass &= _check("auto-PEEP develops", r_copd["auto_peep_cmH2O"] > 0.3,
                        f"auto_peep={r_copd['auto_peep_cmH2O']:.2f}")
    all_pass &= _check("pressure_trajectory length == n_cycles",
                        len(r_copd["pressure_trajectory"]) == 25)
    print(f"     auto_peep={r_copd['auto_peep_cmH2O']:.2f} "
          f"converged={r_copd['converged']} Ppeak={r_copd['ppeak_cmH2O']:.1f}")

    # ---- Dataset sweep smoke check ----
    print("\n[dataset] generate_dataset() smoke check (capped sample, not full sweep)")
    full_combo_count = (len(PARAMETER_GRID["vt_target_ml_per_kg"]) *
                         len(PARAMETER_GRID["respiratory_rate"]) *
                         len(PARAMETER_GRID["peep_cmH2O"]) *
                         len(PARAMETER_GRID["ie_ratio"]) *
                         len(PARAMETER_GRID["pressure_ceiling_cmH2O"]))
    ds = generate_dataset("Normal", 80.0, 10.0, n_cycles=8, max_scenarios=6)
    all_pass &= _check("dataset non-empty", len(ds) > 0, str(len(ds)))
    all_pass &= _check("dataset entries have metrics/waveforms structure",
                        all("metrics" in d and "waveforms" in d for d in ds))
    print(f"     generated {len(ds)} sample scenarios (full sweep would be "
          f"{full_combo_count} per mechanics point)")

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME SMOKE TESTS FAILED")
        sys.exit(1)
