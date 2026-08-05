"""
generator/pcv_generator.py
--------------------------
Pressure-Controlled Ventilation (PCV) waveform generator — multi-compartment.

Control loop
------------
The ventilator prescribes inspiratory pressure (a three-phase profile: linear
rise from PEEP to PIP, plateau at PIP, drop to PEEP). Flow and volume are
the dependent variables — they emerge from the interaction between the
applied pressure and the patient's lung mechanics. Tidal volume is NOT
guaranteed: it depends on compliance, resistance, inspiratory time, and
rise time.

This is the fundamental distinction from VCV and PSV:
    pcv_generator  — prescribes pressure, derives flow/volume   (THIS FILE)
    vcv_generator  — prescribes flow,     derives pressure
    psv_generator  — prescribes pressure with patient effort, flow-cycles

Ventilation mode: Pressure-Controlled Continuous Mandatory Ventilation (PC-CMV)

Multi-compartment lung model
----------------------------
The lung is represented as 1–3 parallel RC compartments per condition:

    Normal:        1 compartment
    Mild ARDS:     2 compartments (aerated + recruitable)
    Moderate ARDS: 2 compartments
    Severe ARDS:   2 compartments
    COPD:          3 compartments (fast / medium / slow)
    Bronchospasm:  1 compartment
    Pneumonia:     3 compartments (healthy / transitional / consolidated)

Governing physics
-----------------
Because pressure is prescribed at the airway opening (P_vent(t)), each
compartment's volume evolves under its OWN independent ODE driven by the
same forcing function:

    dV_i/dt = ( P_vent(t) - V_i/C_rs_i(V_i) - PEEP ) / R_i(V_i) * 1000   [mL/s]

The branch-point algebraic constraint that VCV needs is not needed here —
pressure is already common across all compartments by construction.
Each compartment integrates with its own time constant tau_i = R_i*C_i/1000,
so total inspiratory flow is naturally multi-exponential: fast compartments
fill quickly, slow ones lag. Total flow at the airway is the sum of
per-compartment flows.

Numerics
--------
Explicit forward Euler at DT = 0.01 s (100 Hz), per compartment per phase.
This is a deliberate change from the prior single-compartment PCV (which
used scipy.integrate.solve_ivp with RK45) for two reasons:
    1. It matches the PSV multi-compartment pattern exactly (consistency
       across the three engines).
    2. Volume-dependent compliance and resistance make solve_ivp's adaptive
       stepping awkward across phase boundaries (rise → plateau → expire);
       explicit Euler with a fixed 100 Hz grid handles non-linear / phase-
       discontinuous dynamics naturally.

At DT = 0.01 s the integration is stable for the time constants present
in all seven conditions (the slowest is COPD's slow compartment at
tau ≈ 2 s — Euler stability requires DT << tau, easily satisfied).

Three-phase ventilator pressure profile
---------------------------------------
    Phase 1 — Rise (0 → t_rise):
        P_vent(t) = PEEP + insp_pressure * (t / t_rise)
        Linear ramp from PEEP to PIP. t_rise is settable; 0 = instantaneous
        step (textbook PCV). Internally capped at 50% of t_insp.

    Phase 2 — Plateau (t_rise → t_insp):
        P_vent(t) = PEEP + insp_pressure  (= PIP)
        Held constant; lung volume approaches steady state V_ss = p_insp*C.

    Phase 3 — Expiration (t_insp → t_cycle):
        P_vent = PEEP. Compartments empty passively along their own
        time constants with volume-dependent expiratory R (dynamic
        airway collapse, gated to R_exp_ratio > 1 compartments).

Fill fraction (multi-compartment generalisation)
------------------------------------------------
In the single-compartment version, fill_fraction = 1 - exp(-t_plateau/tau),
analytic and exact. In multi-compartment each compartment has its own tau,
so the analytic single-exponential expression no longer applies. The metric
is now computed numerically:

    fill_fraction = delivered_VT / (insp_pressure * C_total)

where C_total = Sum_i C_rs_i is the total parallel compliance at end-
inspiration. This is the fraction of the steady-state aggregate volume
actually reached at end-Ti. A fill_fraction near 1.0 means the lung
equilibrated; well below 1.0 means the breath ended while gas was still
flowing — the PCV signature of insufficient inspiratory time for the
mechanics. Hardest to reach in COPD, bronchospasm, and severe pneumonia
because of long time constants.

Per-compartment residual volume carries forward between cycles, so multi-
cycle simulations model progressive air trapping in COPD/bronchospasm.

Physiological refinements incorporated
---------------------------------------
    1. Multi-compartment parallel RC mechanics (NEW)
    2. Flow-dependent ETT resistance (Rohrer K1*Q + K2*Q*|Q| on Q_total)
       — applied to the displayed pressure decomposition; the per-
       compartment ODE uses the per-compartment R directly.
    3. Volume-dependent expiratory resistance per compartment (dynamic
       collapse) — strong in COPD, mild in bronchospasm, ~inert elsewhere.
    4. Non-linear compliance per compartment via stress index — modeled
       internally; the stress index metric is NOT exposed in PCV because
       PCV's decelerating-flow profile breaks the constant-flow assumption
       behind the stress-index definition.
    5. PEEP-recruited compliance — applied to global C before split; zero
       for COPD/bronchospasm by default.
    6. Chest wall compliance — in series per compartment via C_rs
       (default ~inert).
    7. Circuit compliance — post-hoc VT scalar correction.

ETT complications (overlays):
    - ETT obstruction: multiplies Rohrer K1 and K2 (and the per-
      compartment R, since intrinsic airway R is fraction of R_global
      → the multiplier hits both the displayed ETT drop and the
      ODE's resistance term).
    - ETT cuff leak: volume-balance correction on delivered VT
      (does NOT affect cycling — PCV is time-cycled, so a leak is a
      measurement note, not a behavior change).

Interface contract (identical to vcv_generator and psv_generator)
------------------------------------------------------------------
    generate_breath_cycles(params, n_cycles) -> dict
    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list

Output dict keys
----------------
    Core waveforms (np.ndarray, 100 Hz):
        time, pressure, flow, volume

    Auxiliary waveforms:
        pressure_branch  : ventilator-side P_vent(t) (= pressure)
        pressure_ett     : ETT Rohrer drop on total flow at each step
        volume_per_comp  : (T, n_compartments) per-compartment volume

    Scalar metrics:
        ppeak_cmH2O, delivered_vt_ml, driving_p_cmH2O, mean_paw_cmH2O,
        auto_peep_cmH2O, fill_fraction, minute_vent_l, time_to_peak_flow_s,
        n_compartments

    Validity:
        is_valid, invalid_reason

Run smoke test:
    python generator/pcv_generator.py

NOTE: helper functions and compartment profiles are inlined here to match
the existing project pattern. A future refactor into a shared
generator/lung_physics.py module would let VCV / PCV / PSV literally call
the same functions instead of three copies that can drift.
"""

import itertools
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Section 1 — Parameter Grid (unchanged from prior PCV)
# ---------------------------------------------------------------------------
PARAMETER_GRID: Dict = {
    "insp_pressure_cmH2O": [5, 10, 15, 20, 25, 30, 35],  # cmH2O above PEEP
    "respiratory_rate":    [8, 12, 16, 20, 24, 28, 30],   # bpm
    "peep_cmH2O":          [0, 4, 8, 12, 16, 20],         # cmH2O
    "ie_ratio":            [1.0, 0.5, 0.33],              # 1:1, 1:2, 1:3
    "rise_time_s":         [0.0, 0.1, 0.2, 0.4],          # seconds
}


# ---------------------------------------------------------------------------
# Section 2 — Safety Thresholds and Constants
# ---------------------------------------------------------------------------
IBW_KG: float                     = 70.0
VT_MIN_ML: float                  = IBW_KG * 3       # 210 mL
VT_MAX_ML: float                  = IBW_KG * 12      # 840 mL
PPEAK_MAX_CMHH2O: float           = 50.0             # barotrauma risk
INSP_PRESSURE_MAX_CMHH2O: float   = 35.0             # max driving above PEEP
FILL_FRACTION_MIN: float          = 0.20             # below this is clinically void
DT: float                         = 0.01             # 100 Hz internal timestep

VT_MIN_ML_PER_KG_ADULT:    float = 3.0    # existing behavior, unchanged
VT_MAX_ML_PER_KG_ADULT:    float = 12.0
VT_MIN_ML_PER_KG_NEONATE:  float = 4.0    # lung-protective floor — Spaeth 2022 / neonatal consensus
VT_MAX_ML_PER_KG_NEONATE:  float = 8.0    # ceiling tighter than adult's 12x — ASSUMPTION, flag for review
NEONATE_IBW_KG_DEFAULT:    float = 3.0    # fallback only if weight_kg is somehow absent


CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5
DEFAULT_CHEST_WALL_COMPLIANCE: float   = 250.0       # mL/cmH2O (~inert default)
ETT_K1: float = 5.0   # cmH2O/L/s     — viscous ETT resistance
ETT_K2: float = 3.0   # cmH2O/(L/s)^2 — turbulent ETT resistance

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
# Section 3 — Condition-Specific Compartment Profiles
# ---------------------------------------------------------------------------
# Counts per user spec for VCV/PCV:
#     Normal: 1 | Mild/Mod/Severe ARDS: 2 | COPD: 3 | Bronchospasm: 1 | Pneumonia: 3
COMPARTMENT_PROFILES: Dict = {
    "Normal": [
        {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
         "R_exp_ratio": 1.2,  "tethering": 0.80},
    ],
    "Mild ARDS": [
        {"fraction": 0.75, "C_frac": 0.90, "R_frac": 1.00,
         "R_exp_ratio": 1.4,  "tethering": 0.40},   # aerated
        {"fraction": 0.25, "C_frac": 0.10, "R_frac": 1.60,
         "R_exp_ratio": 2.0,  "tethering": 0.10},   # recruitable
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
    # COPD: 3 compartments — fast / medium / slow (emphysema)
    "COPD": [
        {"fraction": 0.35, "C_frac": 0.70, "R_frac": 0.55,
         "R_exp_ratio": 4.0,  "tethering": 0.15},
        {"fraction": 0.40, "C_frac": 1.05, "R_frac": 1.27,
         "R_exp_ratio": 6.0,  "tethering": 0.10},
        {"fraction": 0.25, "C_frac": 1.40, "R_frac": 2.36,
         "R_exp_ratio": 8.0,  "tethering": 0.05},
    ],
    # Bronchospasm: 2 compartments — less obstructed + severely obstructed.
    # Matches the PSV bronchospasm profile, bringing the three engines into
    # structural alignment. With 1 compartment, bronchospasm only shows the
    # high-R lumped response. With 2, you get airway-heterogeneity effects:
    # the slow compartment lags during inspiration (pendelluft) and dominates
    # the late expiratory tail. tethering = 0 in both — smooth-muscle override
    # eliminates the volume-dependent airway widening.
    "Bronchospasm": [
        {"fraction": 0.60, "C_frac": 0.90, "R_frac": 0.80,
         "R_exp_ratio": 3.0,  "tethering": 0.00},   # less obstructed
        {"fraction": 0.40, "C_frac": 1.10, "R_frac": 1.43,
         "R_exp_ratio": 5.0,  "tethering": 0.00},   # severely obstructed
    ],
    # Pneumonia: 3 compartments — healthy / transitional / consolidated
    "Pneumonia": [
        {"fraction": 0.60, "C_frac": 1.10, "R_frac": 0.83,
         "R_exp_ratio": 1.5,  "tethering": 0.70},
        {"fraction": 0.25, "C_frac": 0.55, "R_frac": 1.83,
         "R_exp_ratio": 3.0,  "tethering": 0.30},
        {"fraction": 0.15, "C_frac": 0.07, "R_frac": 6.67,
         "R_exp_ratio": 2.0,  "tethering": 0.10},
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

# PEEP-recruited compliance slopes (mL/cmH2O of C gained per cmH2O of PEEP
# above reference PEEP of 5 cmH2O). Zero for obstructive (PEEP does not
# recruit obstructed lung; it counters auto-PEEP instead).
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


# ---------------------------------------------------------------------------
# Section 4 — Physics helper functions (mirrored from psv_generator)
# ---------------------------------------------------------------------------

def _rohrer_resistance(Q: float, K1: float, K2: float) -> float:
    """Rohrer ETT/airway pressure drop: K1*Q + K2*Q*|Q|. Sign-preserving."""
    return K1 * Q + K2 * Q * abs(Q)


def _R_insp_with_tethering(R_base: float,
                            V_current: float,
                            V_target: float,
                            tethering: float) -> float:
    """Inspiratory R with parenchymal tethering (loose in COPD, lost in broncho)."""
    V_frac = float(np.clip(V_current / max(V_target, 1.0), 0.0, 1.0))
    return R_base * max(1.0 - tethering * 0.30 * V_frac, 0.30)


def _R_exp_dynamic(V_current: float,
                    V_end_insp: float,
                    R_insp: float,
                    R_exp_ratio: float) -> float:
    """Expiratory R rising as compartment empties (dynamic airway collapse)."""
    frac_exhaled = 1.0 - float(np.clip(V_current / max(V_end_insp, 1.0), 0.0, 1.0))
    return R_insp * (1.0 + (R_exp_ratio - 1.0) * frac_exhaled)


def _compliance_nonlinear(V_mL: float,
                           C_base: float,
                           V_ref: float,
                           stress_index: float = 1.0) -> float:
    """Power-law non-linear C: C(V) = C_base * (V/V_ref)^(1-SI)."""
    if abs(stress_index - 1.0) < 0.01 or V_mL <= 0.0:
        return C_base
    V_norm = max(V_mL / max(V_ref, 1.0), 0.01)
    return float(C_base * (V_norm ** (1.0 - stress_index)))


def _peep_recruited_compliance(C_base: float,
                                peep: float,
                                peep_ref: float,
                                recruitment_slope: float) -> float:
    """C gain from PEEP-mediated alveolar recruitment above peep_ref."""
    delta_peep = max(0.0, peep - peep_ref)
    return C_base + recruitment_slope * delta_peep


def _C_rs(C_lung: float, C_chest: float) -> float:
    """Series combination of lung and chest-wall compliance."""
    if C_chest >= 9000.0:
        return C_lung
    return 1.0 / (1.0 / max(C_lung, 0.1) + 1.0 / max(C_chest, 0.1))


def _circuit_vt_correction(vt_mL: float,
                            ppeak: float,
                            peep: float,
                            C_circ: float = CIRCUIT_COMPLIANCE_ML_PER_CMH2O,
                            compensated: bool = True) -> float:
    """Subtract gas sequestered in compliant ventilator tubing."""
    if compensated:
        return vt_mL
    return max(0.0, vt_mL - C_circ * max(ppeak - peep, 0.0))


# ---------------------------------------------------------------------------
# Section 5 — Parameter validation
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS = [
    "respiratory_rate", "insp_pressure_cmH2O", "compliance_ml_per_cmH2O",
    "resistance_cmH2O_L_s", "ie_ratio", "peep_cmH2O", "rise_time_s",
]

def _validate_params(params: dict) -> None:

    missing = [k for k in _REQUIRED_PARAMS if k not in params]
    if missing:
        raise ValueError(f"Missing required parameter(s): {missing}")

    population = params.get("population", "adult")
    rr_lo, rr_hi = (20, 80)   if population == "neonate" else (5, 35)
    c_lo,  c_hi  = (0.3, 10)  if population == "neonate" else (5, 150)
    r_lo,  r_hi  = (40, 200)  if population == "neonate" else (0.5, 50)

    if not (rr_lo <= float(params["respiratory_rate"]) <= rr_hi):
        raise ValueError(f"respiratory_rate must be {rr_lo}–{rr_hi} bpm")
    if not (1    <= float(params["insp_pressure_cmH2O"])      <= 50):
        raise ValueError("insp_pressure_cmH2O must be 1–50 cmH2O")
    if not (c_lo <= float(params["compliance_ml_per_cmH2O"])  <= c_hi):
        raise ValueError(f"compliance_ml_per_cmH2O must be {c_lo}–{c_hi} mL/cmH2O")
    if not (r_lo <= float(params["resistance_cmH2O_L_s"])     <= r_hi):
        raise ValueError(f"resistance_cmH2O_L_s must be {r_lo}–{r_hi} cmH2O/L/s")
    if not (0.2  <= float(params["ie_ratio"])                 <= 1.0):
        raise ValueError("ie_ratio must be 0.2–1.0")
    if not (0    <= float(params["peep_cmH2O"])               <= 20):
        raise ValueError("peep_cmH2O must be 0–20 cmH2O")
    if not (0.0  <= float(params["rise_time_s"])              <= 0.4):
        raise ValueError("rise_time_s must be 0.0–0.4 s")


# ---------------------------------------------------------------------------
# Section 6 — Public interface: generate_breath_cycles
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate multi-compartment PCV waveforms for n_cycles breaths.

    Parameters (in `params`)
    ------------------------
    Required:
        respiratory_rate         : float — bpm (8–30)
        insp_pressure_cmH2O      : float — pressure above PEEP (driving Δ)
        compliance_ml_per_cmH2O  : float — global lung compliance
        resistance_cmH2O_L_s     : float — global airway resistance
        ie_ratio                 : float — insp fraction (1.0=1:1, 0.33=1:3)
        peep_cmH2O               : float — PEEP
        rise_time_s              : float — pressure rise time (0.0–0.4)

    Optional:
        condition                : str  — COMPARTMENT_PROFILES key (default "Normal")
        stress_index             : float — non-linear C per compartment (default 1.0)
        chest_wall_compliance_ml_per_cmH2O : float — default 250 (~inert)
        circuit_compensated      : bool  — default True
        peep_reference_cmH2O     : float — default 5.0
        recruitment_slope        : float — overrides RECRUITMENT_SLOPES[cond]
        ett_obstruction_multiplier : float — default 1.0
        ett_cuff_leak_fraction     : float — default 0.0

    Returns
    -------
    dict — see module docstring for full key list
    """
    _validate_params(params)

    rr        = float(params["respiratory_rate"])
    p_insp    = float(params["insp_pressure_cmH2O"])     # driving above PEEP
    C_global  = float(params["compliance_ml_per_cmH2O"])
    R_global  = float(params["resistance_cmH2O_L_s"])
    ie        = float(params["ie_ratio"])
    peep      = float(params["peep_cmH2O"])
    t_rise    = float(params["rise_time_s"])
    PIP       = peep + p_insp                            # absolute peak pressure

    # ---- Optional params -----------------------------------------------
    condition  = params.get("condition", "Normal")
    population = params.get("population", "adult")
    weight_kg  = float(params.get("weight_kg", NEONATE_IBW_KG_DEFAULT if population == "neonate" else IBW_KG))

    if population == "neonate":
        weight = float(params.get("weight_kg", NEONATE_IBW_KG_DEFAULT))
        vt_min_ml = weight * VT_MIN_ML_PER_KG_NEONATE
    else:
        vt_min_ml = IBW_KG * VT_MIN_ML_PER_KG_ADULT   # identical to current VT_MIN_ML

    if condition not in COMPARTMENT_PROFILES:
        condition = "Normal"

    stress_index     = float(params.get("stress_index", 1.0))
    C_chest          = float(params.get(
        "chest_wall_compliance_ml_per_cmH2O",
        _neonate_or_adult(population, NEONATE_DEFAULT_CHEST_WALL_COMPLIANCE, DEFAULT_CHEST_WALL_COMPLIANCE),
    ))
    ppeak_max = _neonate_or_adult(population, NEONATE_PPEAK_MAX_CMHH2O, PPEAK_MAX_CMHH2O)
    circuit_c = _neonate_or_adult(population, NEONATE_CIRCUIT_COMPLIANCE_ML_PER_CMH2O, CIRCUIT_COMPLIANCE_ML_PER_CMH2O)
    vt_min_ml = weight_kg * _neonate_or_adult(population, VT_MIN_ML_PER_KG_NEONATE, VT_MIN_ML_PER_KG_ADULT)
    circ_compensated = bool(params.get("circuit_compensated", True))
    peep_ref         = float(params.get("peep_reference_cmH2O", 5.0))
    rec_slope        = float(params.get("recruitment_slope",
                                          RECRUITMENT_SLOPES.get(condition, 0.0)))
    obs_mult         = float(params.get("ett_obstruction_multiplier", 1.0))
    cuff_leak_frac   = float(params.get("ett_cuff_leak_fraction", 0.0))

    # ---- Compartment arrays --------------------------------------------
    profile = COMPARTMENT_PROFILES[condition]
    n_comps = len(profile)

    fractions   = np.array([c["fraction"]    for c in profile])
    C_frac_arr  = np.array([c["C_frac"]      for c in profile])
    R_frac_arr  = np.array([c["R_frac"]      for c in profile])
    R_exp_arr   = np.array([c["R_exp_ratio"] for c in profile])
    teth_arr    = np.array([c["tethering"]   for c in profile])

    # PEEP-recruited compliance applied before per-compartment split
    C_lung_rec  = _peep_recruited_compliance(C_global, peep, peep_ref, rec_slope)
    C_frac_norm = float(np.dot(C_frac_arr, fractions))
    C_comps_base = C_lung_rec * C_frac_arr * fractions / max(C_frac_norm, 0.01)
    # Per-compartment R: scaled by R_frac, AND by the obstruction multiplier
    # (obstruction raises both ETT Rohrer terms and the airway R uniformly).
    R_comps_base = R_global * R_frac_arr * obs_mult

    # Reference volume per compartment for non-linear C
    # In PCV the steady-state volume per compartment ≈ p_insp * C_comps_base[i]
    # so mid-fill reference is half that.
    vt_full_per_comp = p_insp * C_comps_base                 # mL, full per-comp
    vt_ref_per_comp  = 0.5 * vt_full_per_comp                # mid-fill reference

    # ETT Rohrer coefficients for the displayed pressure decomposition
    K1_ett = ETT_K1 * obs_mult
    K2_ett = ETT_K2 * obs_mult

    # ---- Timing ---------------------------------------------------------
    t_cycle = 60.0 / rr
    t_insp  = t_cycle * ie / (1.0 + ie)
    t_exp   = t_cycle - t_insp
    if t_insp <= 0 or t_exp <= 0:
        raise ValueError(
            f"Timing invalid: t_insp={t_insp:.3f}s t_exp={t_exp:.3f}s"
        )
    # Cap rise time at 50% of inspiratory time so a plateau always exists
    t_rise = min(t_rise, t_insp * 0.5)

    n_insp  = max(2, int(round(t_insp / DT)))
    n_exp   = max(2, int(round(t_exp  / DT)))
    n_per   = n_insp + n_exp
    n_total = n_per * n_cycles

    # ---- Ventilator pressure profile (a function of in-cycle time) -----
    def vent_pressure(t_in_breath: float) -> float:
        if t_in_breath <= t_rise:
            if t_rise <= 0:
                return PIP
            return peep + p_insp * (t_in_breath / t_rise)
        elif t_in_breath <= t_insp:
            return PIP
        else:
            return peep

    # ---- Output arrays --------------------------------------------------
    time_arr     = np.zeros(n_total)
    pressure_arr = np.zeros(n_total)
    flow_arr     = np.zeros(n_total)
    volume_arr   = np.zeros(n_total)
    p_branch_arr = np.zeros(n_total)
    p_ett_arr    = np.zeros(n_total)
    vol_per_comp = np.zeros((n_total, n_comps))

    # ---- Per-compartment state (carries forward between cycles) ---------
    V_comps = np.zeros(n_comps)

    def _step_inspiration(V_state: np.ndarray, P_vent: float) -> Tuple[np.ndarray, float]:
        """One explicit-Euler step per compartment during inspiration."""
        Q_comps = np.zeros(n_comps)
        for i in range(n_comps):
            C_i    = _compliance_nonlinear(
                V_state[i], C_comps_base[i], vt_ref_per_comp[i], stress_index)
            C_rs_i = _C_rs(C_i, C_chest)
            R_i    = _R_insp_with_tethering(
                R_comps_base[i], V_state[i], vt_full_per_comp[i], teth_arr[i])
            drive  = P_vent - (V_state[i] / max(C_rs_i, 0.1)) - peep
            Q_comps[i] = drive / max(R_i, 0.1)          # L/s
        Q_total = float(Q_comps.sum())
        return Q_comps, Q_total

    def _step_expiration(V_state: np.ndarray, V_end_insp_state: np.ndarray
                          ) -> Tuple[np.ndarray, float]:
        """One explicit-Euler step per compartment during expiration."""
        Q_comps = np.zeros(n_comps)
        for i in range(n_comps):
            C_i      = _compliance_nonlinear(
                V_state[i], C_comps_base[i], vt_ref_per_comp[i], stress_index)
            C_rs_i   = _C_rs(C_i, C_chest)
            R_exp_i  = _R_exp_dynamic(
                V_state[i], V_end_insp_state[i],
                R_comps_base[i], R_exp_arr[i])
            elastic  = V_state[i] / max(C_rs_i, 0.1)
            Q_comps[i] = -elastic / max(R_exp_i, 0.1)    # L/s, negative
        Q_total = float(Q_comps.sum())
        return Q_comps, Q_total

    # ---- Main per-cycle loop --------------------------------------------
    t_cursor = 0.0
    for cycle in range(n_cycles):
        offset = cycle * n_per
        t0     = t_cursor

        # -- Inspiration (rise + plateau) ---------------------------------
        for k in range(n_insp):
            t_in_breath = (k + 1) * DT     # end-of-step time within cycle
            P_vent      = vent_pressure(t_in_breath)

            Q_comps, Q_total = _step_inspiration(V_comps, P_vent)
            V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

            P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)

            idx = offset + k
            time_arr[idx]     = t0 + k * DT
            pressure_arr[idx] = P_vent
            flow_arr[idx]     = Q_total
            volume_arr[idx]   = float(V_comps.sum())
            p_branch_arr[idx] = P_vent
            p_ett_arr[idx]    = P_ett_drop
            vol_per_comp[idx] = V_comps.copy()

        V_end_insp_per_comp = V_comps.copy()

        # -- Expiration ---------------------------------------------------
        for k in range(n_exp):
            Q_comps, Q_total = _step_expiration(V_comps, V_end_insp_per_comp)
            V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

            P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)

            idx = offset + n_insp + k
            time_arr[idx]     = t0 + t_insp + k * DT
            pressure_arr[idx] = peep                # valve open to PEEP
            flow_arr[idx]     = Q_total
            volume_arr[idx]   = float(V_comps.sum())
            p_branch_arr[idx] = peep
            p_ett_arr[idx]    = P_ett_drop
            vol_per_comp[idx] = V_comps.copy()
        t_cursor = t0 + t_insp + n_exp * DT

    # ---- Derived metrics from the LAST cycle ----------------------------
    last_s   = (n_cycles - 1) * n_per
    last_e   = last_s + n_per
    last_p   = pressure_arr[last_s:last_e]
    last_v   = volume_arr[last_s:last_e]
    last_f   = flow_arr[last_s:last_e]
    last_t   = time_arr[last_s:last_e]

    ppeak    = float(last_p.max())
    mean_paw = float(np.mean(last_p))

    # Delivered VT = end-inspiratory total volume minus cycle-start volume
    vt_raw       = float(last_v[n_insp - 1] - last_v[0])
    delivered_vt = _circuit_vt_correction(
        vt_raw, ppeak, peep, C_circ=circuit_c, compensated=circ_compensated
    )
    delivered_vt = max(0.0, delivered_vt * (1.0 - cuff_leak_frac))

    minute_vent = (rr * delivered_vt) / 1000.0

    # Auto-PEEP from end-expiratory residual volume
    C_rs_end = np.array([
        _C_rs(_compliance_nonlinear(V_comps[i], C_comps_base[i],
                                      vt_ref_per_comp[i], stress_index),
              C_chest)
        for i in range(n_comps)
    ])
    C_total_end = float(C_rs_end.sum())
    auto_peep   = max(0.0, float(V_comps.sum()) / max(C_total_end, 0.1))

    # Fill fraction (multi-compartment generalisation):
    # delivered_VT / (p_insp * C_total_at_end_inspiration)
    C_rs_end_insp = np.array([
        _C_rs(_compliance_nonlinear(V_end_insp_per_comp[i], C_comps_base[i],
                                      vt_ref_per_comp[i], stress_index),
              C_chest)
        for i in range(n_comps)
    ])
    C_total_at_end_insp = float(C_rs_end_insp.sum())
    expected_full_VT    = p_insp * C_total_at_end_insp     # mL at steady state
    fill_fraction = float(np.clip(
        vt_raw / max(expected_full_VT, 1.0), 0.0, 1.0))

    # Time to peak inspiratory flow (within last cycle)
    insp_flow         = last_f[:n_insp]
    peak_flow_idx     = int(np.argmax(insp_flow)) if insp_flow.size > 0 else 0
    time_to_peak_flow = float(peak_flow_idx * DT)

    # ---- Validity filter ------------------------------------------------
    is_valid       = True
    invalid_reason = ""


    if ppeak > ppeak_max:
        is_valid = False
        invalid_reason = (
            f"PPeak {ppeak:.1f} cmH2O exceeds barotrauma threshold "
            f"({ppeak_max} cmH2O)"
        )
    elif delivered_vt < vt_min_ml:
        is_valid = False
        invalid_reason = (
            f"Delivered VT {delivered_vt:.0f} mL below minimum "
            f"({vt_min_ml:.0f} mL = "
            f"{_neonate_or_adult(population, VT_MIN_ML_PER_KG_NEONATE, VT_MIN_ML_PER_KG_ADULT)} mL/kg)"
        )
    elif population != "neonate" and delivered_vt > VT_MAX_ML:
        is_valid = False
        invalid_reason = (
            f"Delivered VT {delivered_vt:.0f} mL exceeds maximum "
            f"({VT_MAX_ML:.0f} mL = 12 mL/kg IBW)"
        )
    elif p_insp > INSP_PRESSURE_MAX_CMHH2O:
        is_valid = False
        invalid_reason = (
            f"Inspiratory pressure {p_insp:.1f} cmH2O exceeds maximum "
            f"({INSP_PRESSURE_MAX_CMHH2O} cmH2O above PEEP)"
        )
    elif fill_fraction < FILL_FRACTION_MIN:
        is_valid = False
        invalid_reason = (
            f"Fill fraction {fill_fraction:.3f} below minimum "
            f"({FILL_FRACTION_MIN}) — lung barely fills at these mechanics "
            f"and inspiratory time"
        )

    return {
        # Core waveforms
        "time":                 time_arr,
        "pressure":             pressure_arr,
        "flow":                 flow_arr,
        "volume":               volume_arr,
        # Auxiliary
        "pressure_branch":      p_branch_arr,
        "pressure_ett":         p_ett_arr,
        "volume_per_comp":      vol_per_comp,
        # Derived metrics
        "ppeak_cmH2O":          round(ppeak,             2),
        "delivered_vt_ml":      round(delivered_vt,      2),
        "driving_p_cmH2O":      round(float(p_insp),     2),
        "mean_paw_cmH2O":       round(mean_paw,          2),
        "auto_peep_cmH2O":      round(auto_peep,         2),
        "fill_fraction":        round(fill_fraction,     4),
        "minute_vent_l":        round(minute_vent,       3),
        "time_to_peak_flow_s":  round(time_to_peak_flow, 4),
        "n_compartments":       n_comps,
        "condition":            condition,
        # Validity
        "is_valid":             is_valid,
        "invalid_reason":       invalid_reason,
    }


# ---------------------------------------------------------------------------
# Section 7 — Public interface: generate_dataset
# ---------------------------------------------------------------------------

def _make_scenario_id(condition: str, params: dict) -> str:
    cond_short = condition.replace(" ", "")
    return (
        f"PCV_{cond_short}"
        f"_C{int(round(params['compliance_ml_per_cmH2O'] * (10 if params.get('population') == 'neonate' else 1))):03d}"
        f"_R{int(round(params['resistance_cmH2O_L_s'])):03d}"
        f"_PI{int(round(params['insp_pressure_cmH2O'])):02d}"
        f"_RR{int(round(params['respiratory_rate'])):03d}"
        f"_PEEP{int(round(params['peep_cmH2O'])):02d}"
        f"_IE{int(round(params['ie_ratio'] * 100)):03d}"
        f"_RT{int(round(params['rise_time_s'] * 100)):02d}"
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_dataset(
    condition_name:           str,
    compliance_ml_per_cmH2O: float,
    resistance_cmH2O_L_s:    float,
    n_cycles:                 int = 10,
) -> list:
    """
    Sweep the full PCV parameter grid for one condition + mechanics pair.
    The `condition_name` selects which COMPARTMENT_PROFILE to use.
    """
    scenarios: List[dict] = []

    keys   = ["insp_pressure_cmH2O", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "rise_time_s"]
    values = [PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        p_insp, rr, peep, ie, t_rise = combo

        params = {
            "respiratory_rate":        rr,
            "insp_pressure_cmH2O":     p_insp,
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "rise_time_s":             t_rise,
            "condition":               condition_name,
        }

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles)
        except Exception as e:
            scenarios.append({
                "scenario_id":    _make_scenario_id(condition_name, params),
                "condition":      condition_name,
                "params":         params,
                "metrics":        {},
                "is_valid":       False,
                "invalid_reason": f"Generator error: {e}",
                "waveforms":      {},
                "generated_at":   _timestamp(),
            })
            continue

        metrics = {
            "ppeak_cmH2O":         result["ppeak_cmH2O"],
            "delivered_vt_ml":     result["delivered_vt_ml"],
            "driving_p_cmH2O":     result["driving_p_cmH2O"],
            "mean_paw_cmH2O":      result["mean_paw_cmH2O"],
            "auto_peep_cmH2O":     result["auto_peep_cmH2O"],
            "fill_fraction":       result["fill_fraction"],
            "minute_vent_l":       result["minute_vent_l"],
            "time_to_peak_flow_s": result["time_to_peak_flow_s"],
            "n_compartments":      result["n_compartments"],
        }
        waveforms = {
            "time":     result["time"],
            "pressure": result["pressure"],
            "flow":     result["flow"],
            "volume":   result["volume"],
        }

        scenarios.append({
            "scenario_id":    _make_scenario_id(condition_name, params),
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "is_valid":       result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "waveforms":      waveforms,
            "generated_at":   _timestamp(),
        })

    return scenarios


# ---------------------------------------------------------------------------
# Section 8 — Smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    _PASS = "\033[92m✓\033[0m"
    _FAIL = "\033[91m✗\033[0m"
    _results: List[bool] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        status = _PASS if condition else _FAIL
        print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
        _results.append(bool(condition))

    print("=" * 65)
    print("  PCV Generator — Multi-Compartment Smoke Test")
    print("=" * 65)

    base = {
        "respiratory_rate":         15,
        "insp_pressure_cmH2O":      12.0,
        "compliance_ml_per_cmH2O":  60.0,
        "resistance_cmH2O_L_s":      8.0,
        "ie_ratio":                  0.5,
        "peep_cmH2O":                5.0,
        "rise_time_s":               0.1,
        "condition":                 "Normal",
    }

    # ---- Test 1: all rise times, Normal lung ----------------------------
    print("\n[1/4] All rise times — Normal lung, single compartment")
    rise_times = [0.0, 0.1, 0.2, 0.4]
    results_by_rt = {}
    for rt in rise_times:
        p = {**base, "rise_time_s": rt}
        r = generate_breath_cycles(p, n_cycles=3)
        results_by_rt[rt] = r

    r0 = results_by_rt[0.0]
    _check("rise=0.0 returns dict",          isinstance(r0, dict))
    _check("Normal uses 1 compartment",      r0["n_compartments"] == 1)
    _check("volume_per_comp shape correct",  r0["volume_per_comp"].shape[1] == 1)
    _check("pressure decomposition present",
           "pressure_branch" in r0 and "pressure_ett" in r0)

    # Ppeak should be ~constant across rise times (always reaches PIP at plateau)
    ppeaks = [results_by_rt[rt]["ppeak_cmH2O"] for rt in rise_times]
    pip    = base["peep_cmH2O"] + base["insp_pressure_cmH2O"]
    _check("Ppeak ≈ PIP for all rise times (plateau always reached)",
           all(abs(pk - pip) < 0.5 for pk in ppeaks),
           f"PIP={pip:.1f}  Ppeaks={[round(p,2) for p in ppeaks]}")

    # time_to_peak_flow should increase monotonically with rise_time
    t2pks = [results_by_rt[rt]["time_to_peak_flow_s"] for rt in rise_times]
    _check("time_to_peak_flow strictly increases with rise_time",
           all(t2pks[i] <= t2pks[i+1] + 1e-9 for i in range(len(t2pks) - 1)),
           f"{[round(t,3) for t in t2pks]}")

    # Fill fractions stay similar (rise time changes shape, not steady state)
    fills = [results_by_rt[rt]["fill_fraction"] for rt in rise_times]
    _check("fill fraction similar across rise times (Δ < 0.10)",
           max(fills) - min(fills) < 0.10,
           f"{[round(f,3) for f in fills]}")

    for rt in rise_times:
        r = results_by_rt[rt]
        print(f"     rise={rt:.1f}s  Ppeak={r['ppeak_cmH2O']:5.1f}  "
              f"t_to_peak_flow={r['time_to_peak_flow_s']:.3f}s  "
              f"fill={r['fill_fraction']:.3f}  VT={r['delivered_vt_ml']:5.0f}")

    # ---- Test 2: physiology direction checks ----------------------------
    print("\n[2/4] Physiology direction checks across conditions")

    p_normal = {**base}
    r_normal = generate_breath_cycles(p_normal, n_cycles=3)

    # Higher resistance → lower fill fraction
    p_hi_R = {**base, "resistance_cmH2O_L_s": 30.0}
    r_hi_R = generate_breath_cycles(p_hi_R, n_cycles=3)
    _check("higher R → lower fill fraction",
           r_hi_R["fill_fraction"] < r_normal["fill_fraction"],
           f"R=8: ff={r_normal['fill_fraction']:.3f}  "
           f"R=30: ff={r_hi_R['fill_fraction']:.3f}")

    # Higher driving pressure → larger delivered VT (roughly linear in C)
    p_hi_P = {**base, "insp_pressure_cmH2O": 20.0}
    r_hi_P = generate_breath_cycles(p_hi_P, n_cycles=3)
    _check("higher insp_pressure → larger delivered VT",
           r_hi_P["delivered_vt_ml"] > r_normal["delivered_vt_ml"],
           f"P=12: VT={r_normal['delivered_vt_ml']:.0f}  "
           f"P=20: VT={r_hi_P['delivered_vt_ml']:.0f}")

    # COPD multi-compartment: low fill fraction at default RR
    p_copd = {**base, "condition": "COPD",
              "compliance_ml_per_cmH2O": 100.0,
              "resistance_cmH2O_L_s":     22.0,
              "respiratory_rate":         20,
              "insp_pressure_cmH2O":      15.0}
    r_copd_3  = generate_breath_cycles(p_copd, n_cycles=3)
    r_copd_10 = generate_breath_cycles(p_copd, n_cycles=10)
    _check("COPD uses 3 compartments", r_copd_3["n_compartments"] == 3)
    _check("COPD multi-cycle auto-PEEP grows (hyperinflation)",
           r_copd_10["auto_peep_cmH2O"] > r_copd_3["auto_peep_cmH2O"],
           f"3-cyc={r_copd_3['auto_peep_cmH2O']:.2f}  "
           f"10-cyc={r_copd_10['auto_peep_cmH2O']:.2f}")

    # Severe ARDS multi-compartment: high fill fraction (small total C, fast equilibration)
    p_ards = {**base, "condition": "Severe ARDS",
              "compliance_ml_per_cmH2O": 18.0,
              "resistance_cmH2O_L_s":    16.0,
              "insp_pressure_cmH2O":     10.0}
    r_ards = generate_breath_cycles(p_ards, n_cycles=3)
    _check("Severe ARDS uses 2 compartments", r_ards["n_compartments"] == 2)
    _check("Severe ARDS reaches high fill fraction (fast tau)",
           r_ards["fill_fraction"] > 0.80,
           f"fill={r_ards['fill_fraction']:.3f}")

    # Bronchospasm now 2 compartments → low fill fraction (high R, two long tau)
    p_broncho = {**base, "condition": "Bronchospasm",
                 "compliance_ml_per_cmH2O": 70.0,
                 "resistance_cmH2O_L_s":    35.0,
                 "insp_pressure_cmH2O":     15.0}
    r_broncho = generate_breath_cycles(p_broncho, n_cycles=3)
    _check("Bronchospasm uses 2 compartments", r_broncho["n_compartments"] == 2)
    _check("Bronchospasm fill fraction < Normal fill fraction (high R)",
           r_broncho["fill_fraction"] < r_normal["fill_fraction"],
           f"broncho={r_broncho['fill_fraction']:.3f}  "
           f"normal={r_normal['fill_fraction']:.3f}")

    # Pneumonia uses 3 compartments
    p_pneu = {**base, "condition": "Pneumonia",
              "compliance_ml_per_cmH2O": 50.0,
              "resistance_cmH2O_L_s":    12.0}
    r_pneu = generate_breath_cycles(p_pneu, n_cycles=3)
    _check("Pneumonia uses 3 compartments", r_pneu["n_compartments"] == 3)

    print(f"     Normal:   fill={r_normal['fill_fraction']:.3f}  "
          f"VT={r_normal['delivered_vt_ml']:5.0f}")
    print(f"     ARDS:     fill={r_ards['fill_fraction']:.3f}  "
          f"VT={r_ards['delivered_vt_ml']:5.0f}  nC=2")
    print(f"     COPD:     fill={r_copd_3['fill_fraction']:.3f}  "
          f"VT={r_copd_3['delivered_vt_ml']:5.0f}  "
          f"autoPEEP(10c)={r_copd_10['auto_peep_cmH2O']:.2f}  nC=3")
    print(f"     Broncho:  fill={r_broncho['fill_fraction']:.3f}  "
          f"VT={r_broncho['delivered_vt_ml']:5.0f}  nC=2")

    # ---- Test 3: validity filter ---------------------------------------
    print("\n[3/4] Validity filter")

    # Invalid — insp_pressure > 35
    p_hi_P = {**base, "insp_pressure_cmH2O": 40.0, "compliance_ml_per_cmH2O": 15.0}
    r_hi_P_inv = generate_breath_cycles(p_hi_P, n_cycles=2)
    _check("insp_pressure > 35 flagged invalid",
           (not r_hi_P_inv["is_valid"]) and "ressure" in r_hi_P_inv["invalid_reason"].lower(),
           f"{r_hi_P_inv['invalid_reason'][:60]}")

    # Invalid — PPeak > 50 (high pressure + high PEEP)
    p_hi_pk = {**base, "insp_pressure_cmH2O": 35.0, "peep_cmH2O": 20.0}
    r_hi_pk = generate_breath_cycles(p_hi_pk, n_cycles=2)
    _check("PPeak > 50 flagged invalid (high pressure + high PEEP)",
           (not r_hi_pk["is_valid"]) and "PPeak" in r_hi_pk["invalid_reason"],
           f"{r_hi_pk['invalid_reason'][:60]}")

    # Invalid — fill fraction < 0.20 (very high R + short t_insp on compliant lung
    # keeps VT above 210 so the fill filter fires before the low-VT filter)
    p_low_ff = {**base, "condition": "Bronchospasm",
                "compliance_ml_per_cmH2O": 100.0,
                "resistance_cmH2O_L_s":     50.0,
                "respiratory_rate":         28,
                "ie_ratio":                 0.33,
                "rise_time_s":              0.0,
                "insp_pressure_cmH2O":      30.0}
    r_low_ff = generate_breath_cycles(p_low_ff, n_cycles=2)
    _check("fill fraction below 0.20 flagged invalid",
           (not r_low_ff["is_valid"]) and "ill fraction" in r_low_ff["invalid_reason"],
           f"{r_low_ff['invalid_reason'][:60]}")

    # Invalid — VT > 12 mL/kg IBW (high pressure on compliant lung)
    p_hi_vt = {**base, "compliance_ml_per_cmH2O": 100.0,
               "insp_pressure_cmH2O": 30.0}
    r_hi_vt = generate_breath_cycles(p_hi_vt, n_cycles=2)
    _check("VT above 12 mL/kg flagged invalid",
           (not r_hi_vt["is_valid"]) and "VT" in r_hi_vt["invalid_reason"],
           f"{r_hi_vt['invalid_reason'][:60]}")

    # Valid — standard Normal-lung settings
    r_good = generate_breath_cycles(base, n_cycles=2)
    _check("standard Normal-lung scenario passes filter",
           r_good["is_valid"] and r_good["invalid_reason"] == "",
           f"Ppeak={r_good['ppeak_cmH2O']:.1f} "
           f"VT={r_good['delivered_vt_ml']:.0f} "
           f"fill={r_good['fill_fraction']:.3f}")

    # ---- Test 4: dataset sweep (small slice) ---------------------------
    print("\n[4/4] Dataset sweep — Normal lung, n_cycles=1")
    scenarios = generate_dataset(
        condition_name="Normal",
        compliance_ml_per_cmH2O=60.0,
        resistance_cmH2O_L_s=10.0,
        n_cycles=1,
    )
    total = len(scenarios)
    valid = sum(1 for s in scenarios if s["is_valid"])
    ids = [s["scenario_id"] for s in scenarios]

    # Grid product: 7 insp_pressure × 7 RR × 6 PEEP × 3 IE × 4 rise_time = 3528
    expected = 7 * 7 * 6 * 3 * 4

    _check("dataset non-empty",       total > 0)
    _check("scenario count == grid product",
           total == expected,
           f"got {total} expected {expected}")
    _check("all scenario IDs unique", len(ids) == len(set(ids)),
           f"{len(set(ids))} unique of {len(ids)}")
    _check("at least one valid scenario", valid > 0,
           f"{valid}/{total} valid")
    _check("valid scenarios carry metrics",
           all(s.get("metrics") for s in scenarios if s["is_valid"]))
    print(f"     total={total} valid={valid} invalid={total - valid}")
    print(f"     example_id={scenarios[0]['scenario_id']}")
    print(f"     example_metrics: Ppeak={scenarios[0]['metrics'].get('ppeak_cmH2O','—')}  "
          f"VT={scenarios[0]['metrics'].get('delivered_vt_ml','—')}  "
          f"fill={scenarios[0]['metrics'].get('fill_fraction','—')}")

    # ---- Test 5: Neonatal population branch ------------------------------
    print("\n[5/5] Neonatal population branch — weight scaling, leak")
    p_neo = {
        "respiratory_rate":         50,
        "insp_pressure_cmH2O":      10.0,
        "compliance_ml_per_cmH2O":  4.0,
        "resistance_cmH2O_L_s":     80,
        "ie_ratio":                 0.5,
        "peep_cmH2O":               5.0,
        "rise_time_s":              0.05,
        "condition":                "Normal Neonate",
        "population":               "neonate",
        "weight_kg":                3.0,
    }
    r_neo = generate_breath_cycles(p_neo, n_cycles=3)
    _check("neonate scenario returns dict", isinstance(r_neo, dict))
    _check("neonate scenario is valid",     r_neo["is_valid"], r_neo.get("invalid_reason", ""))

    p_underweight_vt = {**p_neo, "weight_kg": 10.0}
    r_under = generate_breath_cycles(p_underweight_vt, n_cycles=3)
    _check("VT floor scales with weight_kg",
           not r_under["is_valid"] and "minimum" in r_under.get("invalid_reason", "").lower())

    p_leak = {**p_neo, "ett_cuff_leak_fraction": 0.15}
    r_leak = generate_breath_cycles(p_leak, n_cycles=3)
    _check("leak reduces delivered_vt vs no-leak baseline",
           r_leak["delivered_vt_ml"] < r_neo["delivered_vt_ml"])

    # ---- Summary --------------------------------------------------------
    n_pass = sum(_results)
    n_total = len(_results)
    print(f"\n{'=' * 65}")
    print(f"  PCV generator smoke test: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("  WARNING: some checks failed — review output above")
    print(f"{'=' * 65}\n")
    sys.exit(0 if n_pass == n_total else 1)