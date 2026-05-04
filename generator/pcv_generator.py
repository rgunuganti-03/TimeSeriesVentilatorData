"""
generator/pcv_generator.py
--------------------------
Pressure-Controlled Ventilation (PCV) waveform generator.

Control loop:
    The ventilator prescribes inspiratory pressure — a target pressure above
    PEEP applied at the airway opening during inspiration. Volume and flow are
    the dependent variables: they emerge from the interaction between the
    applied pressure and the patient's lung mechanics.

    This is the fundamental distinction from VCV:
        vcv_generator  — prescribes flow,     derives pressure
        pcv_generator  — prescribes pressure, derives volume and flow

Ventilation mode: Pressure-Controlled Continuous Mandatory Ventilation (PC-CMV)

Governing physics:
    Single-compartment RC lung model. Equation of motion rearranged as ODE:

        dV/dt = (P_vent(t) - V(t)/C - PEEP) / R  * 1000

    where V is in mL, P_vent is the ventilator pressure profile (cmH2O),
    C is compliance (mL/cmH2O), R is resistance (cmH2O/L/s).

    The *1000 converts L/s → mL/s to match the mL volume state variable.

Pressure profile — three phases per breath:
    1. Rise phase   (0 → t_rise):
           P rises linearly from PEEP to PIP.
           Rise time t_rise is a ventilator setting (0.0–0.4 s).
           t_rise = 0 produces a true square-wave step (instantaneous rise).
           Longer rise times produce a slower ramp — reduces peak flow,
           improves patient comfort, decreases work of breathing in
           spontaneously breathing patients.

    2. Plateau phase (t_rise → t_insp):
           P held constant at PIP = PEEP + insp_pressure.
           Volume accumulates exponentially toward steady state.

    3. Expiratory phase (t_insp → t_cycle):
           P drops to PEEP. Lung deflates passively via elastic recoil.
           Governed by the same ODE with P_vent = PEEP:
               dV/dt = -V(t) / (R * C) * 1000
           Analytical solution: V(t) = V_end_insp * exp(-t / tau)

Key distinction from ode_single.py:
    ode_single.py derives PIP from a target tidal volume.
    pcv_generator prescribes PIP directly as a clinical setting.
    Delivered tidal volume is therefore the DEPENDENT variable —
    it must be computed after solving and checked against safety limits.
    This correctly models clinical PCV: volume is not guaranteed.

Derived metrics returned per scenario:
    ppeak_cmH2O      : peak airway pressure (= PIP during plateau phase)
    delivered_vt_mL  : actual tidal volume delivered (integral of insp flow)
    driving_p_cmH2O  : insp_pressure (PIP - PEEP = net driving pressure)
    mean_paw_cmH2O   : mean airway pressure across full cycle
    auto_peep_cmH2O  : residual pressure above PEEP at end of expiration
    fill_fraction    : fraction of steady-state volume reached (0–1)
    minute_vent_L    : respiratory_rate * delivered_vt / 1000
    time_to_peak_flow_s : time from breath start to peak inspiratory flow

Validity filter:
    is_valid        : bool
    invalid_reason  : str

    Thresholds:
        PPeak > 50 cmH2O               → barotrauma risk
        Driving pressure > 35 cmH2O    → dangerously high insp pressure
        Delivered VT < 3 mL/kg IBW     → inadequate ventilation (210 mL)
        Delivered VT > 12 mL/kg IBW    → overdistension (840 mL)
        Fill fraction < 0.20           → lung barely fills — clinically
                                         meaningless scenario (high R, short
                                         t_insp relative to tau)

    Note: driving pressure threshold differs from VCV (35 vs 20 cmH2O).
    In PCV, insp_pressure is the set driving pressure — it is the direct
    ventilator control variable, not a derived metric. Clinical PCV ranges
    up to 35 cmH2O above PEEP in severe disease. The 20 cmH2O threshold
    applies to VCV driving pressure (Pplat - PEEP), which is an elastic
    load metric, not an applied pressure setting.

Interface contract:
    generate_breath_cycles(params, n_cycles) -> dict
        Same core keys as all other engines: time, pressure, flow, volume.
        Plus derived metrics and validity keys.

    generate_dataset(condition_name, compliance, resistance, n_cycles) -> list
        Sweeps the full PCV parameter grid for one condition + mechanics pair.

    PARAMETER_GRID : dict
        Full PCV parameter grid. Import to inspect or iterate externally.
"""

import itertools
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# PCV Parameter Grid
# ---------------------------------------------------------------------------
# Grounded in the parameter grid definition from the project brief.
#
# insp_pressure_cmH2O : pressure ABOVE PEEP applied during inspiration (PIP - PEEP)
# ie_ratio            : t_insp / t_exp expressed as insp fraction
#                       1.0 = 1:1, 0.5 = 1:2, 0.33 = 1:3
# rise_time_s         : seconds for pressure to ramp from PEEP to PIP
#                       0.0 = instantaneous square wave step

PARAMETER_GRID = {
    "insp_pressure_cmH2O": [5, 10, 15, 20, 25, 30, 35],  # cmH2O above PEEP
    "respiratory_rate":    [8, 12, 16, 20, 24, 28, 30],   # bpm
    "peep_cmH2O":          [0, 4, 8, 12, 16, 20],         # cmH2O
    "ie_ratio":            [1.0, 0.5, 0.33],              # 1:1, 1:2, 1:3
    "rise_time_s":         [0.0, 0.1, 0.2, 0.4],          # seconds
}

# IBW assumption — consistent with vcv_generator
IBW_KG = 70.0

# Safety thresholds
PPEAK_MAX_CMHH2O        = 50.0    # cmH2O — barotrauma risk
INSP_PRESSURE_MAX_CMHH2O = 35.0  # cmH2O — max driving pressure above PEEP
VT_MIN_ML               = IBW_KG * 3    # 210 mL — inadequate ventilation
VT_MAX_ML               = IBW_KG * 12   # 840 mL — overdistension
FILL_FRACTION_MIN        = 0.20   # below this the scenario is clinically void


# ---------------------------------------------------------------------------
# Public interface — waveform generation
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate PCV waveforms for n_cycles breath cycles.

    Parameters
    ----------
    params : dict
        respiratory_rate         : float — breaths per minute (8–30)
        insp_pressure_cmH2O      : float — inspiratory pressure above PEEP (5–35)
        compliance_mL_per_cmH2O  : float — lung compliance
        resistance_cmH2O_L_s     : float — airway resistance
        ie_ratio                 : float — insp fraction (0.33=1:3, 0.5=1:2, 1.0=1:1)
        peep_cmH2O               : float — PEEP
        rise_time_s              : float — pressure rise time in seconds (0.0–0.4)

    n_cycles : int
        Number of complete breath cycles.

    Returns
    -------
    dict
        Core keys  : "time", "pressure", "flow", "volume"
        Metrics    : "ppeak_cmH2O", "delivered_vt_mL", "driving_p_cmH2O",
                     "mean_paw_cmH2O", "auto_peep_cmH2O", "fill_fraction",
                     "minute_vent_L", "time_to_peak_flow_s"
        Validity   : "is_valid", "invalid_reason"
    """
    _validate_params(params)

    rr      = params["respiratory_rate"]
    p_insp  = params["insp_pressure_cmH2O"]   # driving pressure above PEEP
    C       = params["compliance_mL_per_cmH2O"]
    R       = params["resistance_cmH2O_L_s"]
    ie      = params["ie_ratio"]
    peep    = params["peep_cmH2O"]
    t_rise  = params["rise_time_s"]

    PIP     = peep + p_insp               # absolute peak inspiratory pressure

    # --- Timing -----------------------------------------------------------
    t_cycle = 60.0 / rr
    t_insp  = t_cycle * ie / (1.0 + ie)
    t_exp   = t_cycle - t_insp
    tau     = _rc_tau(R, C)

    # Guard: rise time must not exceed inspiratory time
    t_rise  = min(t_rise, t_insp * 0.5)

    # --- Fill fraction — key PCV metric -----------------------------------
    # Fraction of steady-state volume reached during the plateau phase.
    # Plateau duration = t_insp - t_rise (rise phase doesn't contribute fully)
    t_plateau     = t_insp - t_rise
    fill_fraction = 1.0 - np.exp(-t_plateau / tau) if tau > 0 else 1.0
    fill_fraction = float(np.clip(fill_fraction, 0.0, 1.0))

    # Expected delivered VT from steady-state solution
    # V_ss = p_insp * C  (volume if plateau ran to steady state)
    # Delivered = V_ss * fill_fraction
    expected_vt = p_insp * C * fill_fraction   # mL

    # --- Sample grid ------------------------------------------------------
    dt     = 0.01    # 100 Hz
    n_insp = max(2, int(round(t_insp / dt)))
    n_exp  = max(2, int(round(t_exp  / dt)))
    n_tot  = (n_insp + n_exp) * n_cycles

    time_arr     = np.zeros(n_tot)
    flow_arr     = np.zeros(n_tot)
    volume_arr   = np.zeros(n_tot)
    pressure_arr = np.zeros(n_tot)

    # --- Ventilator pressure profile --------------------------------------
    def vent_pressure(t_in_breath: float) -> float:
        """
        Pressure applied by ventilator at time t_in_breath seconds into
        the current breath cycle.

        Rise phase  (0 → t_rise)  : linear ramp PEEP → PIP
        Plateau     (t_rise → t_insp) : constant PIP
        Expiration  (t_insp → t_cycle): constant PEEP
        """
        if t_in_breath < 0:
            return peep
        if t_in_breath < t_rise:
            # Linear ramp
            return peep + p_insp * (t_in_breath / t_rise) if t_rise > 0 else PIP
        if t_in_breath < t_insp:
            return PIP
        return peep

    # --- ODE --------------------------------------------------------------
    def lung_ode(t, y):
        """
        State: y[0] = V (mL above FRC)
        dV/dt = (P_vent - V/C - PEEP) / R  * 1000  [mL/s]
        """
        V           = y[0]
        t_in_breath = t % t_cycle
        P_vent      = vent_pressure(t_in_breath)
        dVdt        = ((P_vent - V / C - peep) / R) * 1000.0
        return [dVdt]

    # --- Solve across all cycles ------------------------------------------
    t_end  = t_cycle * n_cycles
    t_eval = np.arange(0.0, t_end, dt)

    sol = solve_ivp(
        fun=lung_ode,
        t_span=(0.0, t_end),
        y0=[0.0],
        method="RK45",
        t_eval=t_eval,
        max_step=dt,
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    time_arr   = sol.t
    volume_arr = sol.y[0]                        # mL

    # --- Derive flow and pressure -----------------------------------------
    # Flow (L/s) — numerical derivative of volume
    flow_arr = np.gradient(volume_arr, time_arr) / 1000.0

    # Pressure — reconstruct ventilator profile at each time point
    pressure_arr = np.array([
        vent_pressure(t % t_cycle) for t in time_arr
    ])

    # --- Derived metrics --------------------------------------------------
    ppeak        = float(pressure_arr.max())
    delivered_vt = float(volume_arr.max())
    mean_paw     = float(np.mean(pressure_arr))
    auto_peep    = max(0.0, float(pressure_arr[-1]) - peep)
    minute_vent  = (rr * delivered_vt) / 1000.0

    # Time to peak flow — find first sample index where flow is maximum
    peak_flow_idx     = int(np.argmax(flow_arr))
    time_to_peak_flow = float(time_arr[peak_flow_idx] % t_cycle)

    # --- Validity filter --------------------------------------------------
    is_valid       = True
    invalid_reason = ""

    if ppeak > PPEAK_MAX_CMHH2O:
        is_valid = False
        invalid_reason = (
            f"PPeak {ppeak:.1f} cmH2O exceeds barotrauma threshold "
            f"({PPEAK_MAX_CMHH2O} cmH2O)"
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
            f"({FILL_FRACTION_MIN}) — lung barely fills at these "
            f"mechanics and inspiratory time"
        )
    elif delivered_vt < VT_MIN_ML:
        is_valid = False
        invalid_reason = (
            f"Delivered VT {delivered_vt:.0f} mL below minimum "
            f"({VT_MIN_ML:.0f} mL = 3 mL/kg IBW)"
        )
    elif delivered_vt > VT_MAX_ML:
        is_valid = False
        invalid_reason = (
            f"Delivered VT {delivered_vt:.0f} mL exceeds maximum "
            f"({VT_MAX_ML:.0f} mL = 12 mL/kg IBW)"
        )

    return {
        # Core waveform arrays
        "time":                 time_arr,
        "pressure":             pressure_arr,
        "flow":                 flow_arr,
        "volume":               volume_arr,
        # Derived metrics
        "ppeak_cmH2O":          round(ppeak,            2),
        "delivered_vt_mL":      round(delivered_vt,     2),
        "driving_p_cmH2O":      round(float(p_insp),    2),
        "mean_paw_cmH2O":       round(mean_paw,         2),
        "auto_peep_cmH2O":      round(auto_peep,        2),
        "fill_fraction":        round(fill_fraction,     4),
        "minute_vent_L":        round(minute_vent,       3),
        "time_to_peak_flow_s":  round(time_to_peak_flow, 4),
        # Validity
        "is_valid":             is_valid,
        "invalid_reason":       invalid_reason,
    }


# ---------------------------------------------------------------------------
# Public interface — dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    condition_name:           str,
    compliance_mL_per_cmH2O: float,
    resistance_cmH2O_L_s:    float,
    n_cycles:                 int = 10,
) -> list:
    """
    Sweep the full PCV parameter grid for one condition + mechanics pair.

    Parameters
    ----------
    condition_name           : str   — e.g. "Moderate ARDS"
    compliance_mL_per_cmH2O : float — single compliance value for this run
    resistance_cmH2O_L_s    : float — single resistance value for this run
    n_cycles                 : int   — breath cycles per scenario (min 10)

    Returns
    -------
    list of dicts, one per parameter combination. Each dict contains:
        "scenario_id"    : str
        "condition"      : str
        "params"         : dict — full parameter set
        "metrics"        : dict — derived clinical metrics
        "is_valid"       : bool
        "invalid_reason" : str
        "waveforms"      : dict — time, pressure, flow, volume arrays
        "generated_at"   : str  — ISO timestamp
    """
    scenarios = []

    keys   = ["insp_pressure_cmH2O", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "rise_time_s"]
    values = [PARAMETER_GRID[k] for k in keys]

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
            "delivered_vt_mL":     result["delivered_vt_mL"],
            "driving_p_cmH2O":     result["driving_p_cmH2O"],
            "mean_paw_cmH2O":      result["mean_paw_cmH2O"],
            "auto_peep_cmH2O":     result["auto_peep_cmH2O"],
            "fill_fraction":       result["fill_fraction"],
            "minute_vent_L":       result["minute_vent_L"],
            "time_to_peak_flow_s": result["time_to_peak_flow_s"],
        }

        scenarios.append({
            "scenario_id":    _make_scenario_id(condition_name, params),
            "condition":      condition_name,
            "params":         params,
            "metrics":        metrics,
            "is_valid":       result["is_valid"],
            "invalid_reason": result["invalid_reason"],
            "waveforms": {
                "time":     result["time"],
                "pressure": result["pressure"],
                "flow":     result["flow"],
                "volume":   result["volume"],
            },
            "generated_at": _timestamp(),
        })

    return scenarios


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rc_tau(R: float, C: float) -> float:
    """RC time constant in seconds. Floor at 50ms."""
    return max((R * C) / 1000.0, 0.05)


def _make_scenario_id(condition: str, params: dict) -> str:
    """
    Build a human-readable scenario ID encoding all key parameters.
    Format:
        PCV_<COND>_C<compliance>_R<resistance>_
        P<insp_pressure>_RR<rr>_PEEP<peep>_IE<ie>_RT<rise_time>
    Example:
        PCV_ModerateARDS_C025_R008_P015_RR016_PEEP10_IE050_RT010
    """
    cond_slug = condition.replace(" ", "").replace("_", "")
    C      = int(params["compliance_mL_per_cmH2O"])
    R      = int(params["resistance_cmH2O_L_s"])
    p      = int(params["insp_pressure_cmH2O"])
    rr     = int(params["respiratory_rate"])
    peep   = int(params["peep_cmH2O"])
    ie_str = f"IE{int(params['ie_ratio'] * 100):03d}"
    # Rise time encoded in centiseconds (0.1s → RT010, 0.4s → RT040)
    rt_str = f"RT{int(params['rise_time_s'] * 100):03d}"
    return (
        f"PCV_{cond_slug}_"
        f"C{C:03d}_R{R:03d}_"
        f"P{p:03d}_RR{rr:03d}_"
        f"PEEP{peep:02d}_{ie_str}_{rt_str}"
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_params(params: dict) -> None:
    """Raise ValueError for missing or out-of-range parameters."""
    required = [
        "respiratory_rate",
        "insp_pressure_cmH2O",
        "compliance_mL_per_cmH2O",
        "resistance_cmH2O_L_s",
        "ie_ratio",
        "peep_cmH2O",
        "rise_time_s",
    ]
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: '{key}'")

    if not (5    <= params["respiratory_rate"]         <= 35):
        raise ValueError("respiratory_rate must be 5–35 bpm")
    if not (1    <= params["insp_pressure_cmH2O"]      <= 50):
        raise ValueError("insp_pressure_cmH2O must be 1–50 cmH2O")
    if not (5    <= params["compliance_mL_per_cmH2O"]  <= 150):
        raise ValueError("compliance must be 5–150 mL/cmH2O")
    if not (0.5  <= params["resistance_cmH2O_L_s"]     <= 50):
        raise ValueError("resistance must be 0.5–50 cmH2O/L/s")
    if not (0.2  <= params["ie_ratio"]                 <= 1.0):
        raise ValueError("ie_ratio must be 0.2–1.0")
    if not (0    <= params["peep_cmH2O"]               <= 20):
        raise ValueError("peep_cmH2O must be 0–20 cmH2O")
    if not (0.0  <= params["rise_time_s"]              <= 0.4):
        raise ValueError("rise_time_s must be 0.0–0.4 s")


# ---------------------------------------------------------------------------
# Smoke test — run directly: python generator/pcv_generator.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  PCV Generator — Smoke Test")
    print("=" * 65)

    base = {
        "respiratory_rate":        15,
        "insp_pressure_cmH2O":     10,   # 10 cmH2O above PEEP → ~600 mL at C=60
        "compliance_mL_per_cmH2O": 60,
        "resistance_cmH2O_L_s":     2,
        "ie_ratio":                0.5,
        "peep_cmH2O":               5,
        "rise_time_s":             0.0,
    }

    # --- Test 1: all rise times, Normal lung -----------------------------
    print("\n[ Test 1 ] Rise time effect — Normal lung\n")
    for rt in [0.0, 0.1, 0.2, 0.4]:
        p = {**base, "rise_time_s": rt}
        r = generate_breath_cycles(p, n_cycles=5)
        print(f"  Rise={rt:.1f}s | "
              f"PPeak={r['ppeak_cmH2O']:5.1f} cmH2O | "
              f"VT={r['delivered_vt_mL']:5.0f} mL | "
              f"FF={r['fill_fraction']:.3f} | "
              f"t_peak_flow={r['time_to_peak_flow_s']:.3f}s | "
              f"Valid={r['is_valid']}")

    # --- Test 2: physiology direction checks -----------------------------
    print("\n[ Test 2 ] Physiological direction checks\n")

    # Lower compliance → lower delivered VT at same pressure
    r_hc = generate_breath_cycles({**base, "compliance_mL_per_cmH2O": 60})
    r_lc = generate_breath_cycles({**base, "compliance_mL_per_cmH2O": 20})
    assert r_lc["delivered_vt_mL"] < r_hc["delivered_vt_mL"], \
        "FAIL: lower compliance should reduce delivered VT"
    print(f"  Compliance check  PASS — C=60: {r_hc['delivered_vt_mL']:.0f} mL | "
          f"C=20: {r_lc['delivered_vt_mL']:.0f} mL")

    # Higher resistance → lower fill fraction → lower VT
    r_lr = generate_breath_cycles({**base, "resistance_cmH2O_L_s": 2})
    r_hr = generate_breath_cycles({**base, "resistance_cmH2O_L_s": 20})
    assert r_hr["delivered_vt_mL"] < r_lr["delivered_vt_mL"], \
        "FAIL: higher resistance should reduce delivered VT"
    print(f"  Resistance check  PASS — R=2: {r_lr['delivered_vt_mL']:.0f} mL | "
          f"R=20: {r_hr['delivered_vt_mL']:.0f} mL")

    # Higher insp pressure → higher VT
    r_lp = generate_breath_cycles({**base, "insp_pressure_cmH2O": 10})
    r_hp = generate_breath_cycles({**base, "insp_pressure_cmH2O": 25})
    assert r_hp["delivered_vt_mL"] > r_lp["delivered_vt_mL"], \
        "FAIL: higher insp pressure should increase VT"
    print(f"  Pressure check    PASS — P=10: {r_lp['delivered_vt_mL']:.0f} mL | "
          f"P=25: {r_hp['delivered_vt_mL']:.0f} mL")

    # Longer rise time → lower peak flow, higher time_to_peak_flow
    r_rt0 = generate_breath_cycles({**base, "rise_time_s": 0.0})
    r_rt4 = generate_breath_cycles({**base, "rise_time_s": 0.4})
    assert r_rt4["time_to_peak_flow_s"] >= r_rt0["time_to_peak_flow_s"], \
        "FAIL: longer rise time should delay peak flow"
    print(f"  Rise time check   PASS — RT=0.0s: t_peak={r_rt0['time_to_peak_flow_s']:.3f}s | "
          f"RT=0.4s: t_peak={r_rt4['time_to_peak_flow_s']:.3f}s")

    # Faster RR → shorter t_insp → lower fill fraction
    r_slow = generate_breath_cycles({**base, "respiratory_rate": 8})
    r_fast = generate_breath_cycles({**base, "respiratory_rate": 30})
    assert r_fast["fill_fraction"] < r_slow["fill_fraction"], \
        "FAIL: faster RR → shorter t_insp → lower fill fraction"
    print(f"  Fill fraction     PASS — RR=8: {r_slow['fill_fraction']:.3f} | "
          f"RR=30: {r_fast['fill_fraction']:.3f}")

    # --- Test 3: validity filter -----------------------------------------
    print("\n[ Test 3 ] Validity filter\n")

    # Invalid — max resistance + fast RR + short I:E + long rise time
    # R=50, C=60, RR=30, IE=0.33, RT=0.4 → fill_fraction ≈ 0.079 < 0.20
    bad_ff = {**base, "resistance_cmH2O_L_s": 50, "respiratory_rate": 30,
              "ie_ratio": 0.33, "rise_time_s": 0.4}
    r_ff = generate_breath_cycles(bad_ff)
    assert not r_ff["is_valid"], (
        f"FAIL: should be invalid (low fill fraction), "
        f"got ff={r_ff['fill_fraction']:.4f}, valid={r_ff['is_valid']}, "
        f"reason='{r_ff['invalid_reason']}'"
    )
    print(f"  Fill fraction     PASS — {r_ff['invalid_reason']}")

    # Invalid — high pressure + high PEEP → PPeak > 50
    bad_pk = {**base, "insp_pressure_cmH2O": 35, "peep_cmH2O": 20}
    r_pk = generate_breath_cycles(bad_pk)
    assert not r_pk["is_valid"], "FAIL: should be invalid (PPeak)"
    print(f"  PPeak filter      PASS — {r_pk['invalid_reason']}")

    # Valid — normal lung, standard settings
    r_good = generate_breath_cycles(base)
    assert r_good["is_valid"], "FAIL: should be valid"
    print(f"  Valid scenario    PASS — PPeak {r_good['ppeak_cmH2O']:.1f} cmH2O | "
          f"VT {r_good['delivered_vt_mL']:.0f} mL")

    # --- Test 4: dataset sweep -------------------------------------------
    print("\n[ Test 4 ] Dataset sweep — Normal, C=60, R=2\n")

    scenarios = generate_dataset(
        condition_name="Normal",
        compliance_mL_per_cmH2O=60,
        resistance_cmH2O_L_s=2,
        n_cycles=1,   # 1 cycle in smoke test — enough to verify structure
    )

    total   = len(scenarios)
    valid   = sum(1 for s in scenarios if s["is_valid"])
    invalid = total - valid

    print(f"  Total scenarios : {total}")
    print(f"  Valid           : {valid}")
    print(f"  Invalid         : {invalid} ({100*invalid/total:.0f}%)")
    print(f"  Example ID      : {scenarios[0]['scenario_id']}")

    # Confirm all IDs unique
    ids = [s["scenario_id"] for s in scenarios]
    assert len(ids) == len(set(ids)), "FAIL: scenario IDs are not unique"
    print(f"  ID uniqueness   : PASS ({len(set(ids))} unique IDs)")

    print(f"\n{'=' * 65}")
    print("  All smoke tests passed. PCV generator complete.")
    print("=" * 65)
