"""
generator/ode_solver.py
-----------------------
Phase 2: ODE-based single-compartment lung mechanics model.

Drop-in replacement for generator/waveforms.py using scipy.integrate.solve_ivp
instead of rule-based analytical approximations.

Model
-----
Single-compartment RC circuit — the standard lung mechanics model:

    Lung  = elastic compartment with compliance C (mL/cmH2O)
    Airway = resistive element with resistance R (cmH2O/L/s)

Ventilation mode: Pressure-Controlled Continuous Mandatory Ventilation (PC-CMV)
    - Inspiration: ventilator applies constant Peak Inspiratory Pressure (PIP)
    - Expiration:  ventilator opens to PEEP, lung deflates passively via recoil

Equation of motion (standard single-compartment):
    P_vent(t) = V(t)/C + R * dV/dt + PEEP

Rearranged as ODE (state: V in litres):
    dV/dt = (P_vent(t) - V(t)/C - PEEP) / R

Inspiration:  P_vent = PIP  → dV/dt = (delta_P - V/C) / R
Expiration:   P_vent = PEEP → dV/dt = -V / (R*C)

PIP is derived from the target tidal volume using the analytical solution
for a step-pressure input starting from rest (V=0):

    V_T = delta_P * C * (1 - exp(-t_insp / tau))
    => delta_P = V_T / [C * (1 - exp(-t_insp / tau))]

Multi-cycle simulation carries V at end of expiration forward as the
initial condition for the next cycle, allowing auto-PEEP effects to
develop naturally in high-resistance conditions (COPD, Bronchospasm).

Output waveforms
----------------
    Pressure : alveolar pressure = V(t)/C + PEEP  (cmH2O)
               Shows elastic loading during inspiration and recoil during
               expiration. Distinguishable from Phase 1 (PC vs VC mode).
    Flow     : dV/dt from ODE RHS  (L/s, negative during expiration)
    Volume   : ODE solution, converted to mL

Interface contract (identical to generator/waveforms.py):
    generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict
    Returns: {
        "time":     np.ndarray  (seconds)
        "pressure": np.ndarray  (cmH2O)
        "flow":     np.ndarray  (L/s)
        "volume":   np.ndarray  (mL)
    }
"""

import numpy as np
from scipy.integrate import solve_ivp

from generator.waveforms import _validate_params


# ---------------------------------------------------------------------------
# Public interface — identical signature to generator/waveforms.py
# ---------------------------------------------------------------------------

def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    """
    Generate ODE-based ventilator waveforms for n_cycles breath cycles.

    Parameters
    ----------
    params : dict
        Same keys as generator/waveforms.py::generate_breath_cycles().
        respiratory_rate, tidal_volume_mL, compliance_mL_per_cmH2O,
        resistance_cmH2O_L_s, ie_ratio, peep_cmH2O

    n_cycles : int
        Number of complete breath cycles to simulate.

    Returns
    -------
    dict with keys "time", "pressure", "flow", "volume"
        Each value is a 1-D numpy array of the same length (100 Hz).

    Notes
    -----
    Auto-PEEP effect: in high-resistance conditions the lung may not fully
    deflate before the next inspiration. The residual volume is propagated
    forward, so progressive air-trapping is visible in the volume waveform.
    """
    _validate_params(params)

    rr   = params["respiratory_rate"]
    V_T  = params["tidal_volume_mL"] / 1000.0          # L
    C    = params["compliance_mL_per_cmH2O"] / 1000.0  # L/cmH2O
    R    = params["resistance_cmH2O_L_s"]              # cmH2O·s/L
    ie   = params["ie_ratio"]
    peep = params["peep_cmH2O"]

    # ---- Timing -----------------------------------------------------------
    t_cycle = 60.0 / rr
    t_insp  = t_cycle * ie / (1.0 + ie)
    t_exp   = t_cycle - t_insp
    tau     = R * C                                    # RC time constant (s)

    # ---- Driving pressure (PIP) from target tidal volume ------------------
    # Analytical solution for step-pressure input starting from V=0:
    #   V_T = delta_P * C * (1 - exp(-t_insp / tau))
    insp_fill_fraction = 1.0 - np.exp(-t_insp / tau)
    insp_fill_fraction = max(insp_fill_fraction, 1e-6)  # guard zero-division
    delta_P = V_T / (C * insp_fill_fraction)           # cmH2O above PEEP
    PIP     = peep + delta_P                            # absolute PIP (cmH2O)

    # ---- Sampling grid ----------------------------------------------------
    dt      = 0.01                                     # 100 Hz
    n_insp  = max(2, int(round(t_insp / dt)))
    n_exp   = max(2, int(round(t_exp  / dt)))
    n_cycle = n_insp + n_exp
    n_total = n_cycle * n_cycles

    time_arr     = np.zeros(n_total)
    flow_arr     = np.zeros(n_total)
    volume_arr   = np.zeros(n_total)
    pressure_arr = np.zeros(n_total)

    # ---- ODE right-hand side ----------------------------------------------
    def _lung_ode(t, y, P_drive):
        """Single-compartment lung ODE.  y[0] = V (L)."""
        return [(P_drive - y[0] / C - peep) / R]

    # ---- Multi-cycle simulation -------------------------------------------
    V_start = 0.0  # Volume (L) above FRC at start of each inspiration

    for cycle in range(n_cycles):
        t0      = cycle * t_cycle
        offset  = cycle * n_cycle

        t_i_eval = np.linspace(0.0, t_insp, n_insp, endpoint=False)
        t_e_eval = np.linspace(0.0, t_exp,  n_exp,  endpoint=False)

        # -- Inspiration: P_vent = PIP --------------------------------------
        sol_insp = solve_ivp(
            _lung_ode,
            t_span=(0.0, t_insp),
            y0=[V_start],
            args=(PIP,),
            t_eval=t_i_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            dense_output=False,
        )
        V_i = sol_insp.y[0]                            # L

        # Flow from ODE RHS — exact, no numerical differentiation noise
        F_i = (PIP - V_i / C - peep) / R              # L/s  (positive)

        # Alveolar pressure = elastic + PEEP  (excludes resistive component)
        P_i = V_i / C + peep                           # cmH2O

        V_end_insp = V_i[-1]

        # -- Expiration: P_vent = PEEP (passive recoil) ---------------------
        sol_exp = solve_ivp(
            _lung_ode,
            t_span=(0.0, t_exp),
            y0=[V_end_insp],
            args=(peep,),
            t_eval=t_e_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            dense_output=False,
        )
        V_e = sol_exp.y[0]                             # L

        F_e = -V_e / (R * C)                          # L/s  (negative, exact)
        P_e = V_e / C + peep                           # cmH2O

        # Carry residual volume into next cycle (auto-PEEP)
        V_start = max(V_e[-1], 0.0)

        # -- Assemble into output arrays ------------------------------------
        idx_i = slice(offset, offset + n_insp)
        idx_e = slice(offset + n_insp, offset + n_insp + n_exp)

        time_arr[idx_i]     = t0 + t_i_eval
        time_arr[idx_e]     = t0 + t_insp + t_e_eval

        flow_arr[idx_i]     = F_i
        flow_arr[idx_e]     = F_e

        volume_arr[idx_i]   = V_i * 1000.0            # L → mL
        volume_arr[idx_e]   = V_e * 1000.0

        pressure_arr[idx_i] = P_i
        pressure_arr[idx_e] = P_e

    return {
        "time":     time_arr,
        "pressure": pressure_arr,
        "flow":     flow_arr,
        "volume":   volume_arr,
    }


# ---------------------------------------------------------------------------
# Quick smoke test — run directly: python generator/ode_solver.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from generator.conditions import list_conditions, get_condition

    print("Phase 2 — ODE Single-Compartment Lung Model\n")

    for name in list_conditions():
        params = get_condition(name)
        result = generate_breath_cycles(params, n_cycles=5)

        peak_p = result["pressure"].max()
        peak_f = result["flow"].max()
        peak_v = result["volume"].max()
        min_f  = result["flow"].min()
        end_v  = result["volume"][-1]

        print(f"{'─' * 55}")
        print(f"  {name}")
        print(f"  Peak pressure : {peak_p:.1f} cmH2O")
        print(f"  Peak flow     : {peak_f:.3f} L/s  |  Min flow: {min_f:.3f} L/s")
        print(f"  Peak volume   : {peak_v:.1f} mL   |  End vol: {end_v:.2f} mL")
        if end_v > 5.0:
            print(f"  *** Auto-PEEP detected — residual volume {end_v:.1f} mL")
        print()

    print("Smoke test passed.")
