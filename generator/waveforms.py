"""
generator/waveforms.py
----------------------
Rule-based synthetic ventilator waveform generator.

Generates Pressure, Flow, and Volume time-series for a given number of
breath cycles based on physiological parameters.

Interface contract (preserved across Phase 1 → Phase 2 ODE upgrade):
    generate_breath_cycles(params: dict, n_cycles: int) -> dict
    Returns: {
        "time":     np.ndarray  (seconds)
        "pressure": np.ndarray  (cmH2O)
        "flow":     np.ndarray  (L/s)
        "volume":   np.ndarray  (mL)
    }

Physiological basis:
    Equation of motion: P(t) = V(t)/C + R*Flow(t) + PEEP
    where C = compliance (mL/cmH2O), R = resistance (cmH2O/L/s)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate synthetic ventilator waveforms for n_cycles breath cycles.

    Parameters
    ----------
    params : dict
        Keys (all required):
            respiratory_rate        : float  — breaths per minute (bpm)
            tidal_volume_mL         : float  — target tidal volume (mL)
            compliance_mL_per_cmH2O : float  — lung compliance
            resistance_cmH2O_L_s    : float  — airway resistance
            ie_ratio                : float  — inspiration:expiration ratio (e.g. 0.5 = 1:2)
            peep_cmH2O              : float  — positive end-expiratory pressure

    n_cycles : int
        Number of complete breath cycles to generate.

    Returns
    -------
    dict with keys: "time", "pressure", "flow", "volume"
        Each value is a 1-D numpy array of the same length.
    """
    _validate_params(params)

    rr    = params["respiratory_rate"]
    vt    = params["tidal_volume_mL"]
    C     = params["compliance_mL_per_cmH2O"]
    R     = params["resistance_cmH2O_L_s"]
    ie    = params["ie_ratio"]          # t_insp / t_exp  (typically 0.5 = 1:2)
    peep  = params["peep_cmH2O"]

    # Timing
    t_cycle  = 60.0 / rr               # total cycle duration (s)
    t_insp   = t_cycle * ie / (1 + ie) # inspiratory phase duration (s)
    t_exp    = t_cycle - t_insp        # expiratory phase duration (s)

    # Sample rate: 100 Hz (0.01 s resolution — standard for ventilator data)
    dt      = 0.01
    n_insp  = max(2, int(round(t_insp / dt)))
    n_exp   = max(2, int(round(t_exp  / dt)))
    n_total = (n_insp + n_exp) * n_cycles

    time_arr     = np.zeros(n_total)
    flow_arr     = np.zeros(n_total)
    volume_arr   = np.zeros(n_total)
    pressure_arr = np.zeros(n_total)

    for cycle in range(n_cycles):
        offset      = cycle * (n_insp + n_exp)
        t_cycle_start = cycle * t_cycle

        # --- Inspiratory phase ------------------------------------------
        t_i  = np.linspace(0, t_insp, n_insp, endpoint=False)
        # Decelerating flow profile: starts at peak, decays exponentially
        # This is the most common volume-controlled waveform shape
        tau_insp    = t_insp / 3.0           # time constant for decay
        flow_peak   = _calc_peak_flow(vt, t_insp, tau_insp)
        flow_insp   = flow_peak * np.exp(-t_i / tau_insp)   # L/s

        # Volume: integral of flow (trapezoidal) — converted to mL
        vol_insp    = np.cumsum(flow_insp) * dt * 1000.0     # mL

        # Pressure via equation of motion: P = V/C + R*Flow + PEEP
        press_insp  = (vol_insp / C) + (R * flow_insp) + peep

        # --- Expiratory phase -------------------------------------------
        t_e         = np.linspace(0, t_exp, n_exp, endpoint=False)
        # Passive expiration: exponential decay driven by lung recoil
        tau_exp     = R * C / 1000.0         # RC time constant (s), C in mL → convert
        tau_exp     = max(tau_exp, 0.3)      # floor to prevent near-zero tau

        vol_end     = vol_insp[-1]           # volume at end of inspiration (mL)
        vol_exp     = vol_end * np.exp(-t_e / tau_exp)   # mL, decays toward 0

        # Flow is negative during expiration (gas leaving lungs)
        # Flow = dV/dt → derivative of volume decay
        flow_exp    = -(vol_end / tau_exp) * np.exp(-t_e / tau_exp) / 1000.0  # L/s

        # Pressure during expiration
        press_exp   = (vol_exp / C) + (R * flow_exp) + peep

        # --- Assemble into output arrays --------------------------------
        idx_i = slice(offset, offset + n_insp)
        idx_e = slice(offset + n_insp, offset + n_insp + n_exp)

        time_arr[idx_i]     = t_cycle_start + t_i
        time_arr[idx_e]     = t_cycle_start + t_insp + t_e

        flow_arr[idx_i]     = flow_insp
        flow_arr[idx_e]     = flow_exp

        volume_arr[idx_i]   = vol_insp
        volume_arr[idx_e]   = vol_exp

        pressure_arr[idx_i] = press_insp
        pressure_arr[idx_e] = press_exp

    return {
        "time":     time_arr,
        "pressure": pressure_arr,
        "flow":     flow_arr,
        "volume":   volume_arr,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_peak_flow(vt_mL: float, t_insp: float, tau: float) -> float:
    """
    Solve for the peak flow (L/s) such that the integral of the decelerating
    flow profile equals the target tidal volume.

    Integral of flow_peak * exp(-t/tau) from 0 to t_insp  =  vt_mL / 1000
    => flow_peak = (vt / 1000) / (tau * (1 - exp(-t_insp/tau)))
    """
    integral_factor = tau * (1 - np.exp(-t_insp / tau))
    return (vt_mL / 1000.0) / integral_factor


def _validate_params(params: dict) -> None:
    """Raise ValueError if required keys are missing or values are out of range."""
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

    if not (5 <= params["respiratory_rate"] <= 40):
        raise ValueError("respiratory_rate must be between 5 and 40 bpm")
    if not (100 <= params["tidal_volume_mL"] <= 1000):
        raise ValueError("tidal_volume_mL must be between 100 and 1000 mL")
    if not (5 <= params["compliance_mL_per_cmH2O"] <= 150):
        raise ValueError("compliance must be between 5 and 150 mL/cmH2O")
    if not (0.5 <= params["resistance_cmH2O_L_s"] <= 50):
        raise ValueError("resistance must be between 0.5 and 50 cmH2O/L/s")
    if not (0.2 <= params["ie_ratio"] <= 1.0):
        raise ValueError("ie_ratio must be between 0.2 and 1.0")
    if not (0 <= params["peep_cmH2O"] <= 20):
        raise ValueError("peep_cmH2O must be between 0 and 20 cmH2O")


# ---------------------------------------------------------------------------
# Quick smoke test — run directly: python generator/waveforms.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd

    test_params = {
        "respiratory_rate":        15,
        "tidal_volume_mL":        500,
        "compliance_mL_per_cmH2O": 60,
        "resistance_cmH2O_L_s":     2,
        "ie_ratio":               0.5,
        "peep_cmH2O":               5,
    }

    result = generate_breath_cycles(test_params, n_cycles=3)

    df = pd.DataFrame(result)
    print(df.head(20).to_string(index=False))
    print(f"\nTotal samples : {len(df)}")
    print(f"Duration      : {df['time'].max():.2f}s")
    print(f"Peak pressure : {df['pressure'].max():.2f} cmH2O")
    print(f"Peak flow     : {df['flow'].max():.3f} L/s")
    print(f"Peak volume   : {df['volume'].max():.1f} mL")
    print("\nSmoke test passed.")