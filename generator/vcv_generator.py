"""
generator/vcv_generator.py
--------------------------
Volume-Controlled Ventilation (VCV) waveform generator.

Control loop:
    The ventilator prescribes flow — either a constant square profile or a
    linearly decelerating ramp. Pressure is the dependent variable: it goes
    wherever it needs to go to deliver the set flow against the patient's
    lung mechanics.

    This is the fundamental distinction from PCV and ode_single.py:
        ode_single.py  — prescribes pressure, derives volume/flow
        vcv_generator  — prescribes flow, derives pressure

Ventilation mode: Volume-Controlled Continuous Mandatory Ventilation (VC-CMV)

Governing physics:
    Equation of motion applied forward:
        P(t) = V(t)/C + R * Flow(t) + PEEP

    During inspiration, Flow(t) is prescribed (square or decelerating).
    Volume is the integral of flow. Pressure is computed directly from
    the equation of motion at every sample — no ODE solver required
    during inspiration because the input (flow) is fully known.

    During expiration, the ventilator opens to atmosphere (PEEP).
    The lung deflates passively. This IS governed by the ODE:
        dV/dt = (PEEP - V(t)/C - PEEP) / R = -V(t) / (R * C)
    Which has the analytical solution:
        V(t) = V_end_insp * exp(-t / tau)

Flow profiles:
    Square       : constant flow throughout inspiration
                   Flow = VT / t_insp
                   Produces a characteristic rising pressure ramp
                   (elastic pressure increases as volume accumulates)

    Decelerating : linearly decreasing flow from peak to near-zero
                   Flow(t) = Flow_peak * (1 - t/t_insp)
                   Flow_peak = 2 * VT / t_insp  (preserves integral = VT)
                   Produces a pressure curve that rises then plateaus —
                   more comfortable for spontaneously breathing patients

Derived metrics returned per scenario:
    ppeak_cmH2O     : peak airway pressure (resistive + elastic + PEEP)
    pplat_cmH2O     : plateau pressure proxy (elastic + PEEP, end-inspiration)
    driving_p_cmH2O : Pplat - PEEP (pure elastic load)
    mean_paw_cmH2O  : mean airway pressure across full cycle
    auto_peep_cmH2O : residual pressure at end of expiration above PEEP
    delivered_vt_mL : actual peak volume delivered (should match target)
    minute_vent_L   : respiratory_rate * delivered_vt / 1000

Validity filter:
    is_valid        : bool — False if any safety threshold is breached
    invalid_reason  : str  — human-readable reason if is_valid is False

    Thresholds (from brief and clinical literature):
        PPeak > 50 cmH2O          → barotrauma risk
        Driving pressure > 20     → ARDS mortality threshold
        Delivered VT < 3 mL/kg*   → inadequate ventilation
        Delivered VT > 12 mL/kg*  → overdistension
        (* assumes 70 kg IBW — 210 mL min, 840 mL max)

Interface contract:
    generate_breath_cycles(params, n_cycles) -> dict
        Returns waveform arrays + derived metrics + validity flag.
        Core keys (time, pressure, flow, volume) match all other engines.

    generate_dataset(condition_name, n_cycles) -> list[dict]
        Sweeps the full VCV parameter grid for one condition tier.
        Returns list of scenario dicts, each containing params +
        waveforms + metrics + validity.

    PARAMETER_GRID : dict
        The full VCV parameter grid definition.
        Import this to inspect or iterate the grid externally.
"""

import itertools
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.conditions import get_condition_meta, list_conditions


# ---------------------------------------------------------------------------
# VCV Parameter Grid
# ---------------------------------------------------------------------------
# All ranges and step sizes are grounded in the parameter grid definition
# established in the project brief. See ARCHITECTURE.md for rationale.
#
# tidal_volume_mL_per_kg : list — mL per kg IBW (assuming 70 kg → multiply by 70)
# ie_ratio               : list — t_insp / t_exp expressed as insp fraction
#                          0.5 = 1:2, 0.33 = 1:3, 1.0 = 1:1

PARAMETER_GRID = {
    "tidal_volume_mL_per_kg": [4, 6, 8, 10],          # mL/kg IBW
    "respiratory_rate":       [8, 12, 16, 20, 24, 28, 30],  # bpm
    "peep_cmH2O":             [0, 4, 8, 12, 16, 20],   # cmH2O
    "ie_ratio":               [1.0, 0.5, 0.33],        # 1:1, 1:2, 1:3
    "flow_pattern":           ["square", "decelerating"],
}

# IBW assumption for tidal volume conversion
IBW_KG = 70.0

# Safety thresholds for validity filter
PPEAK_MAX_CMHH2O      = 50.0   # cmH2O — barotrauma risk above this
DRIVING_P_MAX_CMHH2O  = 20.0   # cmH2O — ARDS mortality threshold
VT_MIN_ML             = IBW_KG * 3    # 210 mL — inadequate ventilation
VT_MAX_ML             = IBW_KG * 12   # 840 mL — overdistension


# ---------------------------------------------------------------------------
# Public interface — waveform generation
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate VCV waveforms for n_cycles breath cycles.

    Parameters
    ----------
    params : dict
        respiratory_rate        : float  — breaths per minute (8–30)
        tidal_volume_mL         : float  — target tidal volume in mL
        compliance_mL_per_cmH2O : float  — lung compliance
        resistance_cmH2O_L_s    : float  — airway resistance
        ie_ratio                : float  — insp fraction (0.33=1:3, 0.5=1:2, 1.0=1:1)
        peep_cmH2O              : float  — PEEP
        flow_pattern            : str    — "square" or "decelerating"

    n_cycles : int
        Number of complete breath cycles.

    Returns
    -------
    dict
        Core waveform keys: "time", "pressure", "flow", "volume"
        Derived metric keys: "ppeak_cmH2O", "pplat_cmH2O",
            "driving_p_cmH2O", "mean_paw_cmH2O", "auto_peep_cmH2O",
            "delivered_vt_mL", "minute_vent_L"
        Validity keys: "is_valid", "invalid_reason"
    """
    _validate_params(params)

    rr      = params["respiratory_rate"]
    vt      = params["tidal_volume_mL"]          # mL
    C       = params["compliance_mL_per_cmH2O"]  # mL/cmH2O
    R       = params["resistance_cmH2O_L_s"]     # cmH2O/L/s
    ie      = params["ie_ratio"]
    peep    = params["peep_cmH2O"]
    pattern = params.get("flow_pattern", "decelerating")

    # --- Timing -----------------------------------------------------------
    t_cycle = 60.0 / rr
    t_insp  = t_cycle * ie / (1.0 + ie)
    t_exp   = t_cycle - t_insp
    tau     = _rc_tau(R, C)

    # --- Sample grid ------------------------------------------------------
    dt     = 0.01    # 100 Hz
    n_insp = max(2, int(round(t_insp / dt)))
    n_exp  = max(2, int(round(t_exp  / dt)))
    n_tot  = (n_insp + n_exp) * n_cycles

    time_arr     = np.zeros(n_tot)
    flow_arr     = np.zeros(n_tot)
    volume_arr   = np.zeros(n_tot)
    pressure_arr = np.zeros(n_tot)

    # --- Build inspiratory flow profile -----------------------------------
    t_i = np.linspace(0, t_insp, n_insp, endpoint=False)

    if pattern == "square":
        # Constant flow — Flow = VT / t_insp  (L/s, VT in L)
        flow_insp = np.full(n_insp, (vt / 1000.0) / t_insp)   # L/s

    elif pattern == "decelerating":
        # Linear ramp from peak to zero.
        # Integral of Flow_peak*(1 - t/t_insp) from 0 to t_insp = Flow_peak*t_insp/2
        # Set equal to VT/1000 → Flow_peak = 2*VT/(1000*t_insp)
        flow_peak = 2.0 * (vt / 1000.0) / t_insp
        flow_insp = flow_peak * (1.0 - t_i / t_insp)          # L/s, linear decay

    else:
        raise ValueError(f"Unknown flow_pattern: '{pattern}'. Use 'square' or 'decelerating'.")

    # --- Inspiratory volume and pressure ----------------------------------
    # Volume: cumulative integral of flow (trapezoidal), L → mL
    vol_insp   = np.cumsum(flow_insp) * dt * 1000.0            # mL

    # Pressure from equation of motion — applied directly, no ODE needed
    # P(t) = V(t)/C + R*Flow(t) + PEEP
    press_insp = (vol_insp / C) + (R * flow_insp) + peep      # cmH2O

    V_end_insp = vol_insp[-1]   # mL at end of inspiration

    # --- Expiratory phase — analytical ODE solution -----------------------
    # Passive recoil: V(t) = V_end * exp(-t / tau)
    t_e      = np.linspace(0, t_exp, n_exp, endpoint=False)
    vol_exp  = V_end_insp * np.exp(-t_e / tau)                # mL

    # Flow: dV/dt → derivative of exponential decay, L → mL conversion
    flow_exp = -(V_end_insp / tau) * np.exp(-t_e / tau) / 1000.0  # L/s (negative)

    # Pressure: equation of motion during passive expiration
    press_exp = (vol_exp / C) + (R * flow_exp) + peep         # cmH2O

    # --- Assemble n_cycles ------------------------------------------------
    for cycle in range(n_cycles):
        t0     = cycle * t_cycle
        offset = cycle * (n_insp + n_exp)
        idx_i  = slice(offset,          offset + n_insp)
        idx_e  = slice(offset + n_insp, offset + n_insp + n_exp)

        time_arr[idx_i]     = t0 + t_i
        time_arr[idx_e]     = t0 + t_insp + t_e
        flow_arr[idx_i]     = flow_insp
        flow_arr[idx_e]     = flow_exp
        volume_arr[idx_i]   = vol_insp
        volume_arr[idx_e]   = vol_exp
        pressure_arr[idx_i] = press_insp
        pressure_arr[idx_e] = press_exp

    # --- Derived metrics --------------------------------------------------
    ppeak         = pressure_arr.max()
    # Plateau pressure: pressure at end of inspiration (flow → 0)
    # Use last 10% of inspiratory samples where flow has decelerated most
    pplat         = float(np.mean(press_insp[int(0.9 * n_insp):]))
    driving_p     = pplat - peep
    mean_paw      = float(np.mean(pressure_arr))
    delivered_vt  = float(volume_arr.max())
    minute_vent   = (rr * delivered_vt) / 1000.0   # L/min
    # Auto-PEEP: pressure above PEEP remaining at end of last expiration
    auto_peep     = max(0.0, float(pressure_arr[-1]) - peep)

    # --- Validity filter --------------------------------------------------
    is_valid      = True
    invalid_reason = ""

    if ppeak > PPEAK_MAX_CMHH2O:
        is_valid = False
        invalid_reason = (
            f"PPeak {ppeak:.1f} cmH2O exceeds barotrauma threshold "
            f"({PPEAK_MAX_CMHH2O} cmH2O)"
        )
    elif driving_p > DRIVING_P_MAX_CMHH2O:
        is_valid = False
        invalid_reason = (
            f"Driving pressure {driving_p:.1f} cmH2O exceeds ARDS "
            f"mortality threshold ({DRIVING_P_MAX_CMHH2O} cmH2O)"
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
        "time":             time_arr,
        "pressure":         pressure_arr,
        "flow":             flow_arr,
        "volume":           volume_arr,
        # Derived metrics
        "ppeak_cmH2O":      round(ppeak,        2),
        "pplat_cmH2O":      round(pplat,        2),
        "driving_p_cmH2O":  round(driving_p,    2),
        "mean_paw_cmH2O":   round(mean_paw,     2),
        "auto_peep_cmH2O":  round(auto_peep,    2),
        "delivered_vt_mL":  round(delivered_vt, 2),
        "minute_vent_L":    round(minute_vent,   3),
        # Validity
        "is_valid":         is_valid,
        "invalid_reason":   invalid_reason,
    }


# ---------------------------------------------------------------------------
# Public interface — dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    condition_name: str,
    compliance_mL_per_cmH2O: float,
    resistance_cmH2O_L_s:    float,
    n_cycles:                 int = 10,
) -> list:
    """
    Sweep the full VCV parameter grid for one condition + mechanics pair.

    Parameters
    ----------
    condition_name           : str   — e.g. "Moderate ARDS"
    compliance_mL_per_cmH2O : float — single compliance value for this run
    resistance_cmH2O_L_s    : float — single resistance value for this run
    n_cycles                 : int   — breath cycles per scenario (min 10)

    Returns
    -------
    list of dicts, one per parameter combination.
    Each dict contains:
        "scenario_id"   : str
        "condition"     : str
        "params"        : dict  — full parameter set used
        "metrics"       : dict  — derived clinical metrics
        "is_valid"      : bool
        "invalid_reason": str
        "waveforms"     : dict  — time, pressure, flow, volume arrays
        "generated_at"  : str   — ISO timestamp
    """
    scenarios = []

    # Build all combinations from the grid
    keys   = ["tidal_volume_mL_per_kg", "respiratory_rate",
               "peep_cmH2O", "ie_ratio", "flow_pattern"]
    values = [PARAMETER_GRID[k] for k in keys]

    for combo in itertools.product(*values):
        vt_per_kg, rr, peep, ie, pattern = combo

        # Convert mL/kg to absolute mL using IBW assumption
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

        try:
            result = generate_breath_cycles(params, n_cycles=n_cycles)
        except Exception as e:
            # Log but don't crash the sweep — mark as invalid
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
            "delivered_vt_mL": result["delivered_vt_mL"],
            "minute_vent_L":   result["minute_vent_L"],
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
    Build a human-readable scenario ID encoding key parameters.
    Format: VCV_<COND>_C<compliance>_R<resistance>_VT<vt_per_kg>_RR<rr>_PEEP<peep>_<PATTERN>
    Example: VCV_ModerateARDS_C025_R008_VT06_RR016_PEEP10_DEC
    """
    cond_slug = condition.replace(" ", "").replace("_", "")
    C     = int(params["compliance_mL_per_cmH2O"])
    R     = int(params["resistance_cmH2O_L_s"])
    vt_kg = int(params["tidal_volume_mL"] / IBW_KG)
    rr    = int(params["respiratory_rate"])
    peep  = int(params["peep_cmH2O"])
    # Encode I:E ratio as a string: 1.0 → IE100, 0.5 → IE050, 0.33 → IE033
    ie_str = f"IE{int(params['ie_ratio'] * 100):03d}"
    pat   = "SQR" if params["flow_pattern"] == "square" else "DEC"
    return (
        f"VCV_{cond_slug}_"
        f"C{C:03d}_R{R:03d}_"
        f"VT{vt_kg:02d}_RR{rr:03d}_"
        f"PEEP{peep:02d}_{ie_str}_{pat}"
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_params(params: dict) -> None:
    """Raise ValueError for missing or out-of-range parameters."""
    required = [
        "respiratory_rate",
        "tidal_volume_mL",
        "compliance_mL_per_cmH2O",
        "resistance_cmH2O_L_s",
        "ie_ratio",
        "peep_cmH2O",
    ]
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: '{key}'")

    if not (5   <= params["respiratory_rate"]        <= 35):
        raise ValueError("respiratory_rate must be 5–35 bpm")
    if not (100 <= params["tidal_volume_mL"]          <= 1000):
        raise ValueError("tidal_volume_mL must be 100–1000 mL")
    if not (5   <= params["compliance_mL_per_cmH2O"] <= 150):
        raise ValueError("compliance must be 5–150 mL/cmH2O")
    if not (0.5 <= params["resistance_cmH2O_L_s"]    <= 50):
        raise ValueError("resistance must be 0.5–50 cmH2O/L/s")
    if not (0.2 <= params["ie_ratio"]                <= 1.0):
        raise ValueError("ie_ratio must be 0.2–1.0")
    if not (0   <= params["peep_cmH2O"]              <= 20):
        raise ValueError("peep_cmH2O must be 0–20 cmH2O")


# ---------------------------------------------------------------------------
# Smoke test — run directly: python generator/vcv_generator.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 65)
    print("  VCV Generator — Smoke Test")
    print("=" * 65)

    # --- Test 1: single scenario, both flow patterns ---------------------
    print("\n[ Test 1 ] Single scenario — Normal lung, both flow patterns\n")

    base = {
        "respiratory_rate":        15,
        "tidal_volume_mL":         500,
        "compliance_mL_per_cmH2O": 60,
        "resistance_cmH2O_L_s":    2,
        "ie_ratio":                0.5,
        "peep_cmH2O":              5,
    }

    for pattern in ["square", "decelerating"]:
        p = {**base, "flow_pattern": pattern}
        r = generate_breath_cycles(p, n_cycles=5)

        print(f"  Pattern       : {pattern.upper()}")
        print(f"  PPeak         : {r['ppeak_cmH2O']:.1f} cmH2O")
        print(f"  Pplat         : {r['pplat_cmH2O']:.1f} cmH2O")
        print(f"  Driving P     : {r['driving_p_cmH2O']:.1f} cmH2O")
        print(f"  Mean Paw      : {r['mean_paw_cmH2O']:.1f} cmH2O")
        print(f"  Delivered VT  : {r['delivered_vt_mL']:.0f} mL")
        print(f"  Minute vent   : {r['minute_vent_L']:.2f} L/min")
        print(f"  Auto-PEEP     : {r['auto_peep_cmH2O']:.2f} cmH2O")
        print(f"  Valid         : {r['is_valid']}")
        print()

    # --- Test 2: physiology checks ----------------------------------------
    print("[ Test 2 ] Physiological direction checks\n")

    # Lower compliance → higher PPeak (elastic term rises)
    r_hc = generate_breath_cycles({**base, "compliance_mL_per_cmH2O": 60,
                                    "flow_pattern": "square"})
    r_lc = generate_breath_cycles({**base, "compliance_mL_per_cmH2O": 15,
                                    "flow_pattern": "square"})
    assert r_lc["ppeak_cmH2O"] > r_hc["ppeak_cmH2O"], \
        "FAIL: lower compliance should raise PPeak"
    print(f"  Compliance check  PASS — C=60: {r_hc['ppeak_cmH2O']:.1f} | "
          f"C=15: {r_lc['ppeak_cmH2O']:.1f} cmH2O")

    # Higher resistance → higher PPeak (resistive term rises)
    r_lr = generate_breath_cycles({**base, "resistance_cmH2O_L_s": 2,
                                    "flow_pattern": "square"})
    r_hr = generate_breath_cycles({**base, "resistance_cmH2O_L_s": 20,
                                    "flow_pattern": "square"})
    assert r_hr["ppeak_cmH2O"] > r_lr["ppeak_cmH2O"], \
        "FAIL: higher resistance should raise PPeak"
    print(f"  Resistance check  PASS — R=2: {r_lr['ppeak_cmH2O']:.1f} | "
          f"R=20: {r_hr['ppeak_cmH2O']:.1f} cmH2O")

    # Square pattern → higher PPeak than decelerating (higher mean flow)
    r_sq  = generate_breath_cycles({**base, "flow_pattern": "square"})
    r_dec = generate_breath_cycles({**base, "flow_pattern": "decelerating"})
    assert r_sq["ppeak_cmH2O"] > r_dec["ppeak_cmH2O"], \
        "FAIL: square pattern should produce higher PPeak than decelerating"
    print(f"  Flow pattern check PASS — Square: {r_sq['ppeak_cmH2O']:.1f} | "
          f"Decelerating: {r_dec['ppeak_cmH2O']:.1f} cmH2O")

    # Driving pressure = Pplat - PEEP (elastic only — independent of R)
    r_chk = generate_breath_cycles({**base, "flow_pattern": "decelerating"})
    expected_dp = r_chk["pplat_cmH2O"] - base["peep_cmH2O"]
    assert abs(r_chk["driving_p_cmH2O"] - expected_dp) < 0.5, \
        "FAIL: driving pressure must equal Pplat - PEEP"
    print(f"  Driving P check   PASS — {r_chk['driving_p_cmH2O']:.1f} cmH2O "
          f"(Pplat {r_chk['pplat_cmH2O']:.1f} - PEEP {base['peep_cmH2O']})")

    # --- Test 3: validity filter ------------------------------------------
    print("\n[ Test 3 ] Validity filter\n")

    # Should be invalid — very low compliance, high VT → driving P > 20
    bad = {**base, "compliance_mL_per_cmH2O": 10, "tidal_volume_mL": 500,
           "flow_pattern": "square"}
    r_bad = generate_breath_cycles(bad)
    assert not r_bad["is_valid"], "FAIL: should be invalid"
    print(f"  Invalid scenario  PASS — {r_bad['invalid_reason']}")

    # Should be valid — normal lung, standard settings
    r_good = generate_breath_cycles({**base, "flow_pattern": "square"})
    assert r_good["is_valid"], "FAIL: should be valid"
    print(f"  Valid scenario    PASS — PPeak {r_good['ppeak_cmH2O']:.1f} cmH2O")

    # --- Test 4: dataset sweep (small slice) ------------------------------
    print("\n[ Test 4 ] Dataset sweep — Normal, C=60, R=2\n")

    scenarios = generate_dataset(
        condition_name="Normal",
        compliance_mL_per_cmH2O=60,
        resistance_cmH2O_L_s=2,
        n_cycles=5,
    )

    total   = len(scenarios)
    valid   = sum(1 for s in scenarios if s["is_valid"])
    invalid = total - valid

    print(f"  Total scenarios : {total}")
    print(f"  Valid           : {valid}")
    print(f"  Invalid         : {invalid} ({100*invalid/total:.0f}%)")
    print(f"  Example ID      : {scenarios[0]['scenario_id']}")
    print(f"  Example metrics : PPeak={scenarios[0]['metrics'].get('ppeak_cmH2O','—')} | "
          f"Pplat={scenarios[0]['metrics'].get('pplat_cmH2O','—')} | "
          f"DrivingP={scenarios[0]['metrics'].get('driving_p_cmH2O','—')}")

    print(f"\n{'=' * 65}")
    print("  All tests passed. VCV generator smoke test complete.")
    print("=" * 65)
