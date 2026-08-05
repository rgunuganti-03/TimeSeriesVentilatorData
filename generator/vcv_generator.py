"""
generator/vcv_generator.py
--------------------------
Volume-Controlled Ventilation (VCV) waveform generator — multi-compartment.

Control loop
------------
The ventilator prescribes total inspiratory flow — either a constant square
profile or a linearly decelerating ramp. Pressure is the dependent variable:
the airway pressure goes wherever it must to deliver the set flow against
the patient's lung mechanics.

This is the fundamental distinction from PCV and PSV:
    vcv_generator  — prescribes flow,     derives pressure   (THIS FILE)
    pcv_generator  — prescribes pressure, derives flow/volume
    psv_generator  — prescribes pressure with patient effort, flow-cycles

Ventilation mode: Volume-Controlled Continuous Mandatory Ventilation (VC-CMV)

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

Compartments share a common branch-point pressure (P_branch) at the carina
and the ETT acts as a series resistor between the ventilator and the branch.

Governing physics — inspiration
-------------------------------
At each time step with prescribed total flow Q_total(t), the per-compartment
equation of motion gives:

    P_branch = V_i/C_rs_i + R_i * Q_i + PEEP        (one per compartment)

with the mass-balance constraint:

    Sum_i Q_i = Q_total

Solving the linear system algebraically:

    P_branch = ( Q_total + PEEP * S_invR + S_VCR ) / S_invR
    Q_i      = ( P_branch - PEEP - V_i/C_rs_i ) / R_i

where S_invR = Sum_i (1/R_i) and S_VCR = Sum_i V_i/(C_rs_i * R_i).

The displayed airway opening pressure (Pao) adds the ETT Rohrer drop in
series, applied to the total flow:

    Pao = P_branch + K1_ETT * Q_total + K2_ETT * Q_total * |Q_total|

Inspiratory pause (0.3 s)
-------------------------
During the pause Q_total = 0 (valve closed). The solver runs with the same
mass-balance constraint, allowing compartments with different time
constants to redistribute gas via pendelluft until V_i/C_rs_i equalizes
across compartments. Pplat is reported as P_branch at the end of pause
(which equals V_total/C_total_eff + PEEP if equilibrium is fully reached).

Expiration
----------
The ventilator opens to atmosphere (Pao = PEEP). Each compartment empties
passively along its own time constant, with volume-dependent expiratory
resistance applied per compartment:

    Q_i(t) = -( V_i / C_rs_i ) / R_exp_i(V_i)         (L/s, negative)
    V_i(t+dt) = V_i(t) + Q_i * 1000 * dt              (mL)

Because each compartment has a distinct time constant, total expiratory
flow is naturally multi-exponential — the biexponential expiratory limb
that distinguishes COPD emerges from compartment heterogeneity.

Per-compartment residual volume carries forward between cycles, so
multi-cycle simulations model progressive air trapping in COPD and
bronchospasm (the inter-cycle volume carry-forward already present in
the single-compartment version, now per-compartment).

Physiological refinements incorporated
---------------------------------------
    1. Multi-compartment parallel RC mechanics (NEW)
    2. Flow-dependent ETT resistance (Rohrer: K1*Q + K2*Q*|Q|)
    3. Volume-dependent expiratory resistance per compartment
       (dynamic airway collapse — strong in COPD, mild in bronchospasm,
       negligible in ARDS/pneumonia/normal by default profile values)
    4. Non-linear compliance per compartment via stress index
       (SI<1 = tidal recruitment, SI>1 = overdistension, SI=1 = linear)
       VCV is the mode where stress index is interpretable.
    5. PEEP-recruited compliance — applied to the global C before the
       compartment split; zero by default for COPD and bronchospasm
    6. Chest wall compliance — in series per compartment via C_rs
       (default ~inert; only material for obesity/ACS, not implemented)
    7. Circuit compliance — post-hoc VT scalar correction, gated by
       circuit_compensated flag (modern ICU vents auto-compensate)

ETT complications (overlays, not condition properties):
    - ETT obstruction: multiplies Rohrer K1 and K2
    - ETT cuff leak: volume-balance correction on delivered VT
      (does NOT affect cycling — VCV is time/volume-cycled, so a leak
      is a measurement note, not a behavior change)

Stress index
------------
With multi-compartment VCV under square flow, the aggregate pressure ramp
naturally curves even with linear per-compartment compliance, because fast
compartments fill first and slow ones late. The reported stress_index is
fit from the pressure ramp on square-flow inspiration only. With
stress_index parameter = 1.0 (default), the curve still encodes the
time-constant heterogeneity of the condition; with stress_index != 1.0,
the per-compartment compliance is additionally curved.

Interface contract (identical to pcv_generator and psv_generator)
------------------------------------------------------------------
    generate_breath_cycles(params, n_cycles) -> dict
    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list

Output dict keys
----------------
    Core waveforms (np.ndarray, 100 Hz):
        time, pressure, flow, volume

    Auxiliary waveforms (np.ndarray, same length):
        pressure_branch  : branch-point pressure (Pao minus ETT Rohrer drop)
        pressure_ett     : ETT Rohrer drop component
        volume_per_comp  : (n_cycles*N_per_cycle, n_compartments) array

    Scalar metrics:
        ppeak_cmH2O, pplat_cmH2O, driving_p_cmH2O, mean_paw_cmH2O,
        auto_peep_cmH2O, delivered_vt_ml, minute_vent_l,
        stress_index, n_compartments

    Validity:
        is_valid, invalid_reason

Run smoke test:
    python generator/vcv_generator.py

NOTE: helper functions and compartment profiles are inlined here to match
the existing project pattern (PSV inlines them too). A future refactor
into a shared generator/lung_physics.py module would let all three engines
literally call the same functions instead of three copies that can drift.
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
PARAMETER_GRID: Dict = {
    "tidal_volume_ml_per_kg":  [4, 6, 8, 10],
    "respiratory_rate":         [8, 12, 16, 20, 24, 28, 30],
    "peep_cmH2O":               [0, 4, 8, 12, 16, 20],
    "ie_ratio":                 [1.0, 0.5, 0.33],
    "flow_pattern":             ["square", "decelerating"],
}


# ---------------------------------------------------------------------------
# Section 2 — Safety Thresholds and Constants
# ---------------------------------------------------------------------------
IBW_KG: float                  = 70.0
VT_MIN_ML: float               = IBW_KG * 3      # 210 mL — inadequate ventilation
VT_MAX_ML: float               = IBW_KG * 12     # 840 mL — overdistension
PPEAK_MAX_CMHH2O: float        = 50.0            # barotrauma risk
DRIVING_P_MAX_CMHH2O: float    = 20.0            # ARDS mortality threshold
DT: float                      = 0.01            # 100 Hz internal timestep
INSPIRATORY_PAUSE_S: float     = 0.3             # standard 0.3 s pause

VT_MIN_ML_PER_KG_ADULT:    float = 3.0    # existing behavior, unchanged
VT_MAX_ML_PER_KG_ADULT:    float = 12.0
VT_MIN_ML_PER_KG_NEONATE:  float = 4.0    # lung-protective floor — Spaeth 2022 / neonatal consensus
VT_MAX_ML_PER_KG_NEONATE:  float = 8.0    # ceiling tighter than adult's 12x — ASSUMPTION, flag for review
NEONATE_IBW_KG_DEFAULT:    float = 3.0    # fallback only if weight_kg is somehow absent


# Circuit compliance — standard adult ICU circuit
CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5

# Default chest wall compliance (effectively infinite for non-restricted)
DEFAULT_CHEST_WALL_COMPLIANCE: float = 250.0     # mL/cmH2O

# Rohrer ETT contribution (7.5 mm ID tube)
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
#
# Each entry is a list of compartment dicts:
#   fraction      : volume fraction (sums to 1.0 across compartments)
#   C_frac        : compliance multiplier vs the global C preset
#   R_frac        : resistance multiplier vs the global R preset
#   R_exp_ratio   : peak expiratory R / inspiratory R for this compartment
#   tethering     : inspiratory R reduction with volume (0=none, 1=full)
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
    # COPD: 3 compartments — fast (less obstructed), medium, slow (emphysema)
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
    # Pneumonia: 3 compartments — healthy, transitional+secretions, consolidated
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
# recruit an obstructed lung; it counters auto-PEEP instead).
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
    """Rohrer airway/ETT pressure drop: K1*Q + K2*Q*|Q|. Sign-preserving."""
    return K1 * Q + K2 * Q * abs(Q)


def _R_insp_with_tethering(R_base: float,
                            V_current: float,
                            V_target: float,
                            tethering: float) -> float:
    """
    Volume-dependent inspiratory resistance via parenchymal tethering.

    In healthy lungs, expanding parenchyma radially dilates small airways
    and reduces R at higher volumes. In emphysema this tethering is lost;
    in bronchospasm smooth muscle override sets tethering = 0.
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
    R_exp rises from R_insp at end-inspiration toward R_insp * R_exp_ratio
    as the compartment approaches its FRC volume.
    """
    frac_exhaled = 1.0 - float(np.clip(V_current / max(V_end_insp, 1.0), 0.0, 1.0))
    return R_insp * (1.0 + (R_exp_ratio - 1.0) * frac_exhaled)


def _compliance_nonlinear(V_mL: float,
                           C_base: float,
                           V_ref: float,
                           stress_index: float = 1.0) -> float:
    """
    Non-linear compliance via power-law (Grasso/Ranieri stress index form):
        C(V) = C_base * (V/V_ref) ^ (1 - SI)

    SI = 1.0 → linear (default); SI < 1 → recruitment; SI > 1 → overdistension.
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
    PEEP-mediated recruitment increases effective compliance by adding
    previously collapsed alveoli (distinct from non-linear C which models
    already-open alveoli changing stiffness).
    """
    delta_peep = max(0.0, peep - peep_ref)
    return C_base + recruitment_slope * delta_peep


def _C_rs(C_lung: float, C_chest: float) -> float:
    """
    Series combination of lung and chest-wall compliance:
        1/C_rs = 1/C_lung + 1/C_chest
    For C_chest >> C_lung (default), C_rs ≈ C_lung.
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
    Subtract gas sequestered in compliant ventilator tubing.
    Modern ICU vents auto-compensate; transport/older do not.
    """
    if compensated:
        return vt_mL
    return max(0.0, vt_mL - C_circ * max(ppeak - peep, 0.0))


# ---------------------------------------------------------------------------
# Section 5 — VCV-specific multi-compartment branch-point solver
# ---------------------------------------------------------------------------

def _solve_branch_pressure(V_comps: np.ndarray,
                            C_rs_arr: np.ndarray,
                            R_arr: np.ndarray,
                            Q_total: float,
                            peep: float) -> Tuple[float, np.ndarray]:
    """
    Algebraic solver for VCV: given prescribed total flow Q_total and the
    current per-compartment state, find the branch-point pressure and the
    per-compartment flows that satisfy:

        P_branch = V_i/C_rs_i + R_i * Q_i + PEEP   (one per compartment)
        Sum_i Q_i = Q_total                         (mass balance)

    Returns
    -------
    P_branch : float
    Q_comps  : np.ndarray of shape (n_comps,)
    """
    inv_R       = 1.0 / np.maximum(R_arr, 0.1)
    sum_inv_R   = float(inv_R.sum())
    elastic_arr = V_comps / np.maximum(C_rs_arr, 0.1)   # V_i/C_rs_i
    sum_VCR     = float(np.sum(elastic_arr * inv_R))    # Sum V_i/(C_i*R_i)

    P_branch = (Q_total + peep * sum_inv_R + sum_VCR) / sum_inv_R
    Q_comps  = (P_branch - peep - elastic_arr) * inv_R
    return P_branch, Q_comps


# ---------------------------------------------------------------------------
# Section 6 — Parameter validation
# ---------------------------------------------------------------------------

_REQUIRED_PARAMS = [
    "respiratory_rate", "tidal_volume_ml", "compliance_ml_per_cmH2O",
    "resistance_cmH2O_L_s", "ie_ratio", "peep_cmH2O", "flow_pattern",
]

def _validate_params(params: dict) -> None:
    missing = [k for k in _REQUIRED_PARAMS if k not in params]
    if missing:
        raise ValueError(f"Missing required parameter(s): {missing}")
    if params["flow_pattern"] not in ("square", "decelerating"):
        raise ValueError(
            f"flow_pattern must be 'square' or 'decelerating', "
            f"got '{params['flow_pattern']}'"
        )

    population = params.get("population", "adult")
    rr_lo, rr_hi = (20, 80)   if population == "neonate" else (5, 35)
    c_lo,  c_hi  = (0.3, 10)  if population == "neonate" else (5, 150)
    r_lo,  r_hi  = (40, 200)  if population == "neonate" else (0.5, 50)
    v_lo,  v_hi  = (3, 50)  if population == "neonate" else (100, 1000)


    if not (rr_lo <= float(params["respiratory_rate"]) <= rr_hi):
        raise ValueError(f"respiratory_rate must be {rr_lo}–{rr_hi} bpm")
    if not (v_lo  <= float(params["tidal_volume_ml"]) <= v_hi):
        raise ValueError(f"tidal_volume_ml must be {v_lo}–{v_hi} mL")
    if not (c_lo <= float(params["compliance_ml_per_cmH2O"])  <= c_hi):
        raise ValueError(f"compliance_ml_per_cmH2O must be {c_lo}–{c_hi} mL/cmH2O")
    if not (r_lo <= float(params["resistance_cmH2O_L_s"])     <= r_hi):
        raise ValueError(f"resistance_cmH2O_L_s must be {r_lo}–{r_hi} cmH2O/L/s")
    if not (0.2  <= float(params["ie_ratio"])                 <= 1.0):
        raise ValueError("ie_ratio must be 0.2–1.0")
    if not (0    <= float(params["peep_cmH2O"])               <= 20):
        raise ValueError("peep_cmH2O must be 0–20 cmH2O")


# ---------------------------------------------------------------------------
# Section 7 — Stress-index estimation (square-flow inspiration only)
# ---------------------------------------------------------------------------

def _estimate_stress_index(t_insp: np.ndarray,
                            p_insp: np.ndarray,
                            peep: float) -> Optional[float]:
    """
    Fit P(t) - PEEP = a * t^b + c over the square-flow inspiratory ramp
    and return b as the stress index. None if the fit fails.
    """
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return None
    if len(t_insp) < 8:
        return None
    # Use middle 80% to avoid endpoint artifacts
    lo, hi = int(len(t_insp) * 0.10), int(len(t_insp) * 0.90)
    t_sub = t_insp[lo:hi] - t_insp[lo]
    p_sub = p_insp[lo:hi]
    if t_sub[-1] <= 0 or p_sub.max() - p_sub.min() < 0.5:
        return None
    try:
        def model(t, a, b, c):
            return a * np.power(np.maximum(t, 1e-4), b) + c
        popt, _ = curve_fit(
            model, t_sub, p_sub,
            p0=[1.0, 1.0, float(p_sub[0])],
            bounds=([0.001, 0.3, -50.0], [200.0, 3.0, 100.0]),
            maxfev=2000,
        )
        return float(popt[1])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 8 — Public interface: generate_breath_cycles
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate multi-compartment VCV waveforms for n_cycles breaths.

    Parameters (in `params`)
    ------------------------
    Required:
        respiratory_rate         : float — bpm (8–30)
        tidal_volume_ml          : float — target VT in mL
        compliance_ml_per_cmH2O  : float — global lung compliance
        resistance_cmH2O_L_s     : float — global airway resistance
        ie_ratio                 : float — insp fraction (1.0 = 1:1, 0.33 = 1:3)
        peep_cmH2O               : float — PEEP
        flow_pattern             : str   — "square" or "decelerating"

    Optional (multi-compartment / refinements):
        condition                : str   — one of COMPARTMENT_PROFILES keys
                                           (default "Normal")
        stress_index             : float — SI for non-linear C (default 1.0)
        chest_wall_compliance_ml_per_cmH2O : float — default 250 (~inert)
        circuit_compensated      : bool  — default True
        peep_reference_cmH2O     : float — default 5.0
        recruitment_slope        : float — overrides RECRUITMENT_SLOPES[cond]

    Optional (ETT complications — overlays):
        ett_obstruction_multiplier : float — multiplies K1 and K2 (default 1.0)
        ett_cuff_leak_fraction     : float — fraction of VT lost (default 0.0)

    Returns
    -------
    dict — see module docstring for full key list
    """
    _validate_params(params)

    # ---- Unpack required params ----------------------------------------
    rr        = float(params["respiratory_rate"])
    vt_target = float(params["tidal_volume_ml"])
    C_global  = float(params["compliance_ml_per_cmH2O"])
    R_global  = float(params["resistance_cmH2O_L_s"])
    ie        = float(params["ie_ratio"])
    peep      = float(params["peep_cmH2O"])
    pattern   = str(params["flow_pattern"])

    # ---- Optional params -----------------------------------------------
    condition = params.get("condition", "Normal")
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

    # ---- Build compartment arrays --------------------------------------
    profile = COMPARTMENT_PROFILES[condition]
    n_comps = len(profile)

    fractions   = np.array([c["fraction"]    for c in profile])
    C_frac_arr  = np.array([c["C_frac"]      for c in profile])
    R_frac_arr  = np.array([c["R_frac"]      for c in profile])
    R_exp_arr   = np.array([c["R_exp_ratio"] for c in profile])
    teth_arr    = np.array([c["tethering"]   for c in profile])

    # Apply PEEP-recruited compliance to global C before compartment split
    C_lung_rec  = _peep_recruited_compliance(C_global, peep, peep_ref, rec_slope)

    # Normalize so total parallel compliance equals C_lung_rec
    C_frac_norm = float(np.dot(C_frac_arr, fractions))
    C_comps_base = C_lung_rec * C_frac_arr * fractions / max(C_frac_norm, 0.01)
    R_comps_base = R_global * R_frac_arr

    # Reference volume for non-linear C (mid-inspiration target per compartment)
    vt_ref_per_comp = vt_target * 0.5 * fractions
    vt_full_per_comp = vt_target * fractions     # full target per compartment

    # ETT Rohrer coefficients (with obstruction multiplier)
    K1_ett = ETT_K1 * obs_mult
    K2_ett = ETT_K2 * obs_mult

    # ---- Timing ---------------------------------------------------------
    t_cycle = 60.0 / rr
    t_pause = INSPIRATORY_PAUSE_S
    t_insp  = (t_cycle - t_pause) * ie / (1.0 + ie)
    t_exp   = t_cycle - t_insp - t_pause
    if t_insp <= 0 or t_exp <= 0:
        raise ValueError(
            f"Timing invalid: t_insp={t_insp:.3f}s t_exp={t_exp:.3f}s "
            f"(RR={rr} bpm, IE={ie}, pause={t_pause}s)"
        )

    n_insp  = max(2, int(round(t_insp / DT)))
    n_pause = max(1, int(round(t_pause / DT)))
    n_exp   = max(2, int(round(t_exp / DT)))
    n_per   = n_insp + n_pause + n_exp
    n_total = n_per * n_cycles

    # ---- Inspiratory flow profile (prescribed) -------------------------
    t_i = np.linspace(0.0, t_insp, n_insp, endpoint=False)
    if pattern == "square":
        Q_insp = np.full(n_insp, (vt_target / 1000.0) / t_insp)   # L/s
    else:  # decelerating
        Q_peak = 2.0 * (vt_target / 1000.0) / t_insp
        Q_insp = Q_peak * (1.0 - t_i / t_insp)

    # ---- Output arrays -------------------------------------------------
    time_arr     = np.zeros(n_total)
    pressure_arr = np.zeros(n_total)
    flow_arr     = np.zeros(n_total)
    volume_arr   = np.zeros(n_total)
    p_branch_arr = np.zeros(n_total)
    p_ett_arr    = np.zeros(n_total)
    vol_per_comp = np.zeros((n_total, n_comps))

    # ---- Per-compartment state (carries forward between cycles) --------
    V_comps = np.zeros(n_comps)

    def _per_compartment_C_rs(V_state: np.ndarray) -> np.ndarray:
        out = np.zeros(n_comps)
        for i in range(n_comps):
            C_i = _compliance_nonlinear(
                V_state[i], C_comps_base[i], vt_ref_per_comp[i], stress_index)
            out[i] = _C_rs(C_i, C_chest)
        return out

    # ---- Main per-cycle loop --------------------------------------------
    t_cursor = 0.0 
    for cycle in range(n_cycles):
        offset = cycle * n_per
        t0     = t_cursor

        # -- Inspiration: prescribed Q_total, solve for P_branch & Q_i --
        for k in range(n_insp):
            Q_total = float(Q_insp[k])

            C_rs_arr = _per_compartment_C_rs(V_comps)
            R_arr    = np.array([
                _R_insp_with_tethering(
                    R_comps_base[i], V_comps[i],
                    vt_full_per_comp[i], teth_arr[i])
                for i in range(n_comps)
            ])

            P_branch, Q_comps = _solve_branch_pressure(
                V_comps, C_rs_arr, R_arr, Q_total, peep
            )
            P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)
            Pao        = P_branch + P_ett_drop

            # Forward Euler update of compartment volumes
            V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

            idx = offset + k
            time_arr[idx]      = t0 + t_i[k]
            pressure_arr[idx]  = Pao
            flow_arr[idx]      = Q_total
            volume_arr[idx]    = float(V_comps.sum())
            p_branch_arr[idx]  = P_branch
            p_ett_arr[idx]     = P_ett_drop
            vol_per_comp[idx]  = V_comps.copy()

        # End-inspiration snapshot (used for R_exp_dynamic during expiration)
        V_end_insp_per_comp = V_comps.copy()

        # -- Inspiratory pause: Q_total = 0, pendelluft to equilibrium --
        for k in range(n_pause):
            C_rs_arr = _per_compartment_C_rs(V_comps)
            R_arr    = np.array([
                _R_insp_with_tethering(
                    R_comps_base[i], V_comps[i],
                    vt_full_per_comp[i], teth_arr[i])
                for i in range(n_comps)
            ])

            P_branch, Q_comps = _solve_branch_pressure(
                V_comps, C_rs_arr, R_arr, Q_total=0.0, peep=peep
            )
            # Pao = P_branch during pause (no ETT drop because Q_total = 0)
            Pao = P_branch

            V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

            idx = offset + n_insp + k
            time_arr[idx]      = t0 + t_insp + (k + 1) * DT
            pressure_arr[idx]  = Pao
            flow_arr[idx]      = 0.0          # total flow at airway = 0
            volume_arr[idx]    = float(V_comps.sum())
            p_branch_arr[idx]  = P_branch
            p_ett_arr[idx]     = 0.0
            vol_per_comp[idx]  = V_comps.copy()

        # -- Expiration: passive per-compartment emptying ---------------
        # Each compartment empties driven by its own elastic recoil through
        # its dynamic expiratory resistance. Pao at the airway opening = PEEP
        # plus the ETT drop on outflow (small negative dip at valve opening).
       

        for k in range(n_exp):
            C_rs_arr = _per_compartment_C_rs(V_comps)
            R_exp_arr_now = np.array([
                _R_exp_dynamic(
                    V_comps[i], V_end_insp_per_comp[i],
                    R_comps_base[i], R_exp_arr[i])
                for i in range(n_comps)
            ])
            # Q_i (L/s, negative for outflow) = -(V_i/C_rs_i) / R_exp_i
            elastic = V_comps / np.maximum(C_rs_arr, 0.1)
            Q_comps = -(elastic / np.maximum(R_exp_arr_now, 0.1))
            Q_total = float(Q_comps.sum())

            V_comps = np.maximum(V_comps + Q_comps * 1000.0 * DT, 0.0)

            # Displayed Pao during expiration: PEEP + ETT Rohrer drop (small dip)
            P_ett_drop = _rohrer_resistance(Q_total, K1_ett, K2_ett)
            Pao        = peep 

            idx = offset + n_insp + n_pause + k
            time_arr[idx]      = t0 + t_insp + t_pause + (k + 1) * DT
            pressure_arr[idx]  = Pao
            flow_arr[idx]      = Q_total
            volume_arr[idx]    = float(V_comps.sum())
            p_branch_arr[idx]  = peep
            p_ett_arr[idx]     = P_ett_drop
            vol_per_comp[idx]  = V_comps.copy()
           
        

        
        t_cursor = t0 + t_insp + t_pause + n_exp * DT
    # print(q_total_list)
    # print(p_ett_drop_list)
    # ---- Derived metrics from the LAST cycle ----------------------------
    last_s = (n_cycles - 1) * n_per
    last_e = last_s + n_per
    last_p = pressure_arr[last_s:last_e]
    last_v = volume_arr[last_s:last_e]
    last_t = time_arr[last_s:last_e]

    ppeak = float(last_p.max())

    # Pplat = pressure at last sample of pause phase
    pause_end_idx = last_s + n_insp + n_pause - 1
    pplat = float(pressure_arr[pause_end_idx])

    driving_p = max(0.0, pplat - peep)
    mean_paw  = float(np.mean(last_p))

    # Delivered VT = end-inspiratory volume minus cycle-start volume
    vt_raw = float(last_v[n_insp - 1] - last_v[0])
    delivered_vt = _circuit_vt_correction(
        vt_raw, ppeak, peep, C_circ=circuit_c, compensated=circ_compensated
    )
    # ETT cuff leak (volume-balance only — no effect on cycling in VCV)
    delivered_vt = max(0.0, delivered_vt * (1.0 - cuff_leak_frac))

    minute_vent = (rr * delivered_vt) / 1000.0

    # Auto-PEEP from end-expiratory residual volume
    C_rs_end = _per_compartment_C_rs(V_comps)
    C_total_end = float(C_rs_end.sum())  # parallel C add
    auto_peep = max(0.0, float(V_comps.sum()) / max(C_total_end, 0.1))

    # Stress index — only meaningful for square flow
    si_computed: Optional[float] = None
    if pattern == "square":
        si_computed = _estimate_stress_index(
            last_t[:n_insp], last_p[:n_insp], peep
        )

    # ---- Validity filter ------------------------------------------------
    is_valid = True
    invalid_reason = ""

    if ppeak > ppeak_max:
        is_valid = False
        invalid_reason = (
            f"PPeak {ppeak:.1f} cmH2O exceeds barotrauma threshold "
            f"({ppeak_max} cmH2O)"
        )
    elif population != "neonate" and driving_p > DRIVING_P_MAX_CMHH2O:
        is_valid = False
        invalid_reason = (
            f"Driving pressure {driving_p:.1f} cmH2O exceeds ARDS "
            f"mortality threshold ({DRIVING_P_MAX_CMHH2O} cmH2O)"
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
    return {
        # Core waveforms
        "time":                time_arr,
        "pressure":            pressure_arr,
        "flow":                flow_arr,
        "volume":              volume_arr,
        # Auxiliary
        "pressure_branch":     p_branch_arr,
        "pressure_ett":        p_ett_arr,
        "volume_per_comp":     vol_per_comp,
        # Derived metrics
        "ppeak_cmH2O":         round(ppeak,        2),
        "pplat_cmH2O":         round(pplat,        2),
        "driving_p_cmH2O":     round(driving_p,    2),
        "mean_paw_cmH2O":      round(mean_paw,     2),
        "auto_peep_cmH2O":     round(auto_peep,    2),
        "delivered_vt_ml":     round(delivered_vt, 2),
        "minute_vent_l":       round(minute_vent,   3),
        "stress_index":        (round(si_computed, 3) if si_computed is not None
                                  else None),
        "n_compartments":      n_comps,
        "condition":           condition,
        # Validity
        "is_valid":            is_valid,
        "invalid_reason":      invalid_reason,
    }


# ---------------------------------------------------------------------------
# Section 9 — Public interface: generate_dataset
# ---------------------------------------------------------------------------

def _make_scenario_id(condition: str, params: dict) -> str:
    cond_short = condition.replace(" ", "")
    return (
        f"VCV_{cond_short}"
        f"_C{int(round(params['compliance_ml_per_cmH2O'] * (10 if params.get('population') == 'neonate' else 1))):03d}"
        f"_R{int(round(params['resistance_cmH2O_L_s'])):03d}"
        f"_VT{int(round(params['tidal_volume_ml'] / IBW_KG)):02d}"
        f"_RR{int(round(params['respiratory_rate'])):03d}"
        f"_PEEP{int(round(params['peep_cmH2O'])):02d}"
        f"_IE{int(round(params['ie_ratio'] * 100)):03d}"
        f"_{params['flow_pattern'].upper()[:3]}"
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
    Sweep the full VCV parameter grid for one condition + mechanics pair.
    The `condition_name` selects which COMPARTMENT_PROFILE to use.
    """
    scenarios: List[dict] = []

    keys   = ["tidal_volume_ml_per_kg", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "flow_pattern"]
    values = [PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        vt_per_kg, rr, peep, ie, pattern = combo
        vt_mL = vt_per_kg * IBW_KG

        params = {
            "respiratory_rate":        rr,
            "tidal_volume_ml":         vt_mL,
            "compliance_ml_per_cmH2O": compliance_ml_per_cmH2O,
            "resistance_cmH2O_L_s":    resistance_cmH2O_L_s,
            "ie_ratio":                ie,
            "peep_cmH2O":              peep,
            "flow_pattern":            pattern,
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
            "ppeak_cmH2O":     result["ppeak_cmH2O"],
            "pplat_cmH2O":     result["pplat_cmH2O"],
            "driving_p_cmH2O": result["driving_p_cmH2O"],
            "mean_paw_cmH2O":  result["mean_paw_cmH2O"],
            "auto_peep_cmH2O": result["auto_peep_cmH2O"],
            "delivered_vt_ml": result["delivered_vt_ml"],
            "minute_vent_l":   result["minute_vent_l"],
            "stress_index":    result["stress_index"],
            "n_compartments":  result["n_compartments"],
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
# Section 10 — Smoke test
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
    print("  VCV Generator — Multi-Compartment Smoke Test")
    print("=" * 65)

    # Common Normal-lung baseline used across tests
    base = {
        "respiratory_rate":         15,
        "tidal_volume_ml":          420,        # 6 mL/kg at 70 kg IBW
        "compliance_ml_per_cmH2O":   60.0,
        "resistance_cmH2O_L_s":       8.0,
        "ie_ratio":                   0.5,
        "peep_cmH2O":                 5.0,
        "condition":                  "Normal",
    }

    # ---- Test 1: single scenario, both flow patterns --------------------
    print("\n[1/4] Single scenario — Normal lung, both flow patterns")
    p_sq  = {**base, "flow_pattern": "square"}
    p_dec = {**base, "flow_pattern": "decelerating"}
    r_sq  = generate_breath_cycles(p_sq,  n_cycles=3)
    r_dec = generate_breath_cycles(p_dec, n_cycles=3)

    _check("square returns dict",            isinstance(r_sq, dict))
    _check("decelerating returns dict",      isinstance(r_dec, dict))
    _check("core waveforms non-empty",       len(r_sq["time"]) > 0 and len(r_dec["time"]) > 0)
    _check("Normal uses 1 compartment",      r_sq["n_compartments"] == 1)
    _check("volume_per_comp shape correct",  r_sq["volume_per_comp"].shape[1] == 1)
    _check("pressure decomposition present",
           "pressure_branch" in r_sq and "pressure_ett" in r_sq)
    _check("square Ppeak > decelerating Ppeak (classic VCV)",
           r_sq["ppeak_cmH2O"] > r_dec["ppeak_cmH2O"],
           f"sq={r_sq['ppeak_cmH2O']:.1f} dec={r_dec['ppeak_cmH2O']:.1f}")
    _check("both patterns deliver ~same VT (VCV guarantees VT)",
           abs(r_sq["delivered_vt_ml"] - r_dec["delivered_vt_ml"]) < 30,
           f"sq={r_sq['delivered_vt_ml']:.0f} dec={r_dec['delivered_vt_ml']:.0f}")
    _check("square stress index ~ 1.0 (linear 1-compartment)",
           r_sq["stress_index"] is not None and abs(r_sq["stress_index"] - 1.0) < 0.10,
           f"SI={r_sq['stress_index']}")
    _check("decelerating stress_index is None (only computed for square)",
           r_dec["stress_index"] is None)
    print(f"     sq:  Ppeak={r_sq['ppeak_cmH2O']:5.1f}  Pplat={r_sq['pplat_cmH2O']:5.1f}  "
          f"ΔP={r_sq['driving_p_cmH2O']:4.1f}  VT={r_sq['delivered_vt_ml']:5.0f}")
    print(f"     dec: Ppeak={r_dec['ppeak_cmH2O']:5.1f}  Pplat={r_dec['pplat_cmH2O']:5.1f}  "
          f"ΔP={r_dec['driving_p_cmH2O']:4.1f}  VT={r_dec['delivered_vt_ml']:5.0f}")

    # ---- Test 2: physiology direction checks (multi-compartment) -------
    print("\n[2/4] Physiology direction checks across conditions")

    # Severe ARDS at reduced VT vs Normal at full VT → ARDS still has higher ΔP
    p_normal_sq = {**base, "flow_pattern": "square"}
    p_ards = {**p_normal_sq, "condition": "Severe ARDS",
              "compliance_ml_per_cmH2O": 18.0, "resistance_cmH2O_L_s": 16.0,
              "tidal_volume_ml": 280}
    r_normal = generate_breath_cycles(p_normal_sq, n_cycles=3)
    r_ards   = generate_breath_cycles(p_ards,      n_cycles=3)
    _check("Severe ARDS ΔP > Normal ΔP (low C dominates)",
           r_ards["driving_p_cmH2O"] > r_normal["driving_p_cmH2O"],
           f"ARDS={r_ards['driving_p_cmH2O']:.1f} Normal={r_normal['driving_p_cmH2O']:.1f}")
    _check("Severe ARDS uses 2 compartments", r_ards["n_compartments"] == 2)

    # COPD: stress index deviates from 1.0 due to time-constant heterogeneity
    p_copd = {**p_normal_sq, "condition": "COPD",
              "compliance_ml_per_cmH2O": 100.0, "resistance_cmH2O_L_s": 22.0}
    r_copd = generate_breath_cycles(p_copd, n_cycles=3)
    _check("COPD uses 3 compartments", r_copd["n_compartments"] == 3)
    _check("COPD stress index deviates from 1.0 (heterogeneity-driven)",
           r_copd["stress_index"] is not None
              and abs(r_copd["stress_index"] - 1.0) > 0.05,
           f"SI={r_copd['stress_index']}")

    # COPD multi-cycle: progressive auto-PEEP from dynamic hyperinflation
    p_copd_fast = {**p_copd, "respiratory_rate": 22}
    r_copd_3  = generate_breath_cycles(p_copd_fast, n_cycles=3)
    r_copd_10 = generate_breath_cycles(p_copd_fast, n_cycles=10)
    _check("COPD auto-PEEP grows with cycle count (hyperinflation)",
           r_copd_10["auto_peep_cmH2O"] > r_copd_3["auto_peep_cmH2O"],
           f"3-cyc={r_copd_3['auto_peep_cmH2O']:.2f} "
           f"10-cyc={r_copd_10['auto_peep_cmH2O']:.2f}")

    # Bronchospasm now 2 compartments — high R produces high Ppeak
    p_broncho = {**p_normal_sq, "condition": "Bronchospasm",
                 "compliance_ml_per_cmH2O": 70.0, "resistance_cmH2O_L_s": 35.0}
    r_broncho = generate_breath_cycles(p_broncho, n_cycles=3)
    _check("Bronchospasm uses 2 compartments", r_broncho["n_compartments"] == 2)
    _check("Bronchospasm Ppeak >> Normal Ppeak (high R term dominates)",
           r_broncho["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"] + 5.0,
           f"broncho={r_broncho['ppeak_cmH2O']:.1f} normal={r_normal['ppeak_cmH2O']:.1f}")

    # Pneumonia uses 3 compartments
    p_pneu = {**p_normal_sq, "condition": "Pneumonia",
              "compliance_ml_per_cmH2O": 50.0, "resistance_cmH2O_L_s": 12.0}
    r_pneu = generate_breath_cycles(p_pneu, n_cycles=3)
    _check("Pneumonia uses 3 compartments", r_pneu["n_compartments"] == 3)

    print(f"     ARDS:     Ppeak={r_ards['ppeak_cmH2O']:5.1f}  "
          f"ΔP={r_ards['driving_p_cmH2O']:4.1f}  nC={r_ards['n_compartments']}")
    print(f"     COPD:     Ppeak={r_copd['ppeak_cmH2O']:5.1f}  "
          f"SI={r_copd['stress_index']:.3f}  "
          f"autoPEEP(10c)={r_copd_10['auto_peep_cmH2O']:.2f}  nC=3")
    print(f"     Broncho:  Ppeak={r_broncho['ppeak_cmH2O']:5.1f}  "
          f"ΔP={r_broncho['driving_p_cmH2O']:4.1f}  nC=2")

    # ---- Test 3: validity filter ---------------------------------------
    print("\n[3/4] Validity filter")

    # Invalid — Severe ARDS at large VT → driving pressure > 20
    p_bad_dp = {**p_normal_sq, "condition": "Severe ARDS",
                "compliance_ml_per_cmH2O": 18.0,
                "resistance_cmH2O_L_s":    16.0,
                "tidal_volume_ml":          600}
    r_bad_dp = generate_breath_cycles(p_bad_dp, n_cycles=2)
    _check("Severe ARDS at high VT flagged invalid",
           (not r_bad_dp["is_valid"]) and r_bad_dp["invalid_reason"] != "",
           f"{r_bad_dp['invalid_reason'][:60]}")

    # Invalid — VT below 3 mL/kg IBW
    p_low_vt = {**p_normal_sq, "tidal_volume_ml": 150}
    r_low_vt = generate_breath_cycles(p_low_vt, n_cycles=2)
    _check("VT below 3 mL/kg flagged invalid (low-VT bound)",
           (not r_low_vt["is_valid"]) and "VT" in r_low_vt["invalid_reason"],
           f"{r_low_vt['invalid_reason'][:60]}")

    # Invalid — VT above 12 mL/kg IBW
    p_hi_vt = {**p_normal_sq, "tidal_volume_ml": 900}
    r_hi_vt = generate_breath_cycles(p_hi_vt, n_cycles=2)
    _check("VT above 12 mL/kg flagged invalid (high-VT bound)",
           (not r_hi_vt["is_valid"]) and "VT" in r_hi_vt["invalid_reason"],
           f"{r_hi_vt['invalid_reason'][:60]}")

    # Valid — standard Normal-lung settings
    r_good = generate_breath_cycles(p_normal_sq, n_cycles=2)
    _check("standard Normal-lung scenario passes filter",
           r_good["is_valid"] and r_good["invalid_reason"] == "",
           f"Ppeak={r_good['ppeak_cmH2O']:.1f} ΔP={r_good['driving_p_cmH2O']:.1f}")

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

    # Grid product: 4 VT_per_kg × 7 RR × 6 PEEP × 3 IE × 2 patterns = 1008
    expected = 4 * 7 * 6 * 3 * 2

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
          f"Pplat={scenarios[0]['metrics'].get('pplat_cmH2O','—')}  "
          f"ΔP={scenarios[0]['metrics'].get('driving_p_cmH2O','—')}")
    
    # ---- Test 5: Neonatal population branch ------------------------------
    print("\n[5/5] Neonatal population branch — weight scaling, leak")
    p_neo = {
        "respiratory_rate":         50,
        "tidal_volume_ml":          15,
        "compliance_ml_per_cmH2O":  4.0,
        "resistance_cmH2O_L_s":     80,
        "ie_ratio":                 0.5,
        "peep_cmH2O":               5,
        "flow_pattern":             "square",
        "condition":                "Normal Neonate",
        "population":               "neonate",
        "weight_kg":                3.0,
    }
    r_neo = generate_breath_cycles(p_neo, n_cycles=3)
    _check("neonate scenario returns dict", isinstance(r_neo, dict))
    _check("neonate scenario is valid",     r_neo["is_valid"], r_neo.get("invalid_reason", ""))

    p_underweight_vt = {**p_neo, "weight_kg": 10.0}   # same 15 mL VT, floor scales to 40 mL
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
    print(f"  VCV generator smoke test: {n_pass}/{n_total} checks passed")
    if n_pass < n_total:
        print("  WARNING: some checks failed — review output above")
    print(f"{'=' * 65}\n")
    sys.exit(0 if n_pass == n_total else 1)