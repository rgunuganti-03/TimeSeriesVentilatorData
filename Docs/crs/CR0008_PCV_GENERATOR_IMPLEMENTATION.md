# CR0009 — PCV Generator Implementation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Once the PCV control loop was documented and the parameter grid was defined, a working generator needed to be implemented that correctly models the PCV control loop mechanics. The existing generators in the project were not suitable for PCV dataset generation in their current form. The rule-based generator and the VCV generator both prescribed flow as the independent variable. The ode_single.py engine modeled pressure-controlled ventilation but derived its driving pressure from a target tidal volume rather than accepting inspiratory pressure as a direct clinical setting — the opposite of how PCV works in practice. None of these generators exposed rise time as a parameter, implemented the three-phase pressure profile, computed the fill fraction metric, or provided a dataset sweep function compatible with the standard export schema.

A dedicated PCV generator was needed that prescribes pressure directly as the clinical control variable, derives volume and flow through ODE integration, exposes rise time as a first-class parameter, computes all PCV-specific derived metrics including fill fraction and time to peak flow, applies the correct validity thresholds for a pressure-controlled mode, and sweeps the full parameter grid via a generate_dataset() function that produces structured scenario dicts ready for export.

---

## Current State

The PCV generator has been fully implemented, tested, and the dataset generation batch run has been launched across all seven condition tiers.

generator/pcv_generator.py was built implementing Pressure-Controlled Continuous Mandatory Ventilation. The generator prescribes inspiratory pressure as the independent variable — the clinician sets a pressure target and delivered tidal volume is the dependent variable that results from the interaction between that pressure and the patient's lung mechanics. This is the correct clinical model of PCV: volume is not guaranteed and must be checked after solving rather than set before.

The pressure profile is implemented as a three-phase waveform. During the rise phase (0 to t_rise seconds), pressure ramps linearly from PEEP to PIP. During the plateau phase (t_rise to t_insp), pressure is held constant at PIP. During the expiratory phase, pressure drops to PEEP and the lung deflates passively. Rise time is capped internally at 50% of inspiratory time to prevent the pathological edge case where no plateau exists. At rise time zero the profile is a true square wave step. Longer rise times delay and reduce peak inspiratory flow, directly increasing the time_to_peak_flow metric.

The ODE governing the lung response is solved using scipy.integrate.solve_ivp with RK45 adaptive step size control across the full breath cycle: dV/dt = (P_vent(t) - V(t)/C - PEEP) / R * 1000, where the multiplication by 1000 converts L/s to mL/s to match the mL volume state variable. The ODE solver is called for every scenario including the rise phase, because the linearly ramping pressure during rise time cannot be handled by the analytical solutions used in the expiratory phase.

Eight derived metrics are computed per scenario: PPeak (equal to PIP since the plateau always reaches the set pressure), delivered tidal volume (the actual dependent variable), driving pressure (the set inspiratory pressure above PEEP, passed through directly from params rather than derived), mean airway pressure, auto-PEEP, fill fraction (1 minus exp(-t_plateau / tau), where t_plateau is inspiratory time minus rise time and tau is R times C divided by 1000), minute ventilation, and time to peak inspiratory flow (the time elapsed from breath start to the moment of maximum flow, quantifying the rise time effect directly).

The validity filter applies five thresholds in priority order: PPeak above 50 cmH2O (barotrauma risk), inspiratory pressure above 35 cmH2O (maximum safe driving pressure above PEEP), fill fraction below 0.20 (lung barely fills — the threshold was corrected from an initial value of 0.10 after calculation confirmed that breaching 0.10 would require resistance above 78.5 cmH2O/L/s, exceeding the validation ceiling of 50), delivered tidal volume below 210 mL (3 mL/kg at 70 kg IBW), and delivered tidal volume above 840 mL (12 mL/kg IBW).

Three bugs were identified and fixed during smoke testing and unit testing. First, the fill fraction threshold of 0.10 was unreachable within the allowed resistance range — corrected to 0.20 after calculating the required resistance to breach the threshold. Second, the base smoke test used inspiratory pressure of 15 cmH2O on a C=60 lung, producing 900 mL which correctly tripped the VT_MAX filter — corrected to 10 cmH2O producing 600 mL. Third, the dataset sweep at n_cycles=5 in the smoke test projected to 16 minutes — reduced to n_cycles=1 for all smoke tests and unit tests, with the rationale documented that auto-PEEP is invisible at one cycle but structural correctness is fully verifiable, and that the performance tradeoff is acceptable for testing while the actual dataset generation uses n_cycles=10.

80 unit tests were written and all pass. The full dataset generation batch run was launched via nohup python -u generate_pcv_dataset.py > pcv_generation.log 2>&1 and is currently executing as PID 59770, projecting 8–10 hours of total runtime due to the ODE solver overhead of approximately 0.27 seconds per scenario.

---

## Proposed Change

Implement a dedicated PCV generator module that correctly models the PCV control loop, prescribes inspiratory pressure as the direct clinical control variable, implements the three-phase pressure profile with configurable rise time, solves the governing ODE using scipy.integrate.solve_ivp, computes all eight PCV-specific derived metrics, applies the five-threshold validity filter with thresholds appropriate for a pressure-controlled mode, and provides a generate_dataset() function that sweeps the full parameter grid for any given condition and mechanics point. The generator must satisfy the shared interface contract and extend it with PCV-specific derived metrics and validity keys required for dataset generation.

---

## Acceptance Criteria

- generate_breath_cycles() returns a dict containing the four core waveform arrays (time, pressure, flow, volume) as NumPy arrays of equal length, all eight derived metric keys as numeric values, and both validity keys (is_valid as bool, invalid_reason as str)
- The pressure array equals PIP within 0.5 cmH2O throughout the plateau phase — the ventilator holds pressure constant at PIP during inspiration
- PPeak equals PIP (peep plus insp_pressure) within 0.5 cmH2O — the plateau always reaches the set pressure regardless of rise time
- Rise time zero produces an earlier time to peak flow than rise time 0.4 seconds
- Rise time zero produces a higher peak inspiratory flow than rise time 0.4 seconds
- Rise time does not change PPeak — the plateau always reaches PIP regardless of how slowly the ramp approached it
- Lower compliance reduces delivered tidal volume at the same inspiratory pressure
- Higher resistance reduces fill fraction and therefore delivered tidal volume at the same inspiratory pressure
- Higher inspiratory pressure increases delivered tidal volume
- Fill fraction is independent of inspiratory pressure magnitude — it depends only on tau and t_insp
- Faster respiratory rate reduces fill fraction due to shorter inspiratory time
- The fill fraction threshold constant equals 0.20
- The validity filter correctly identifies and labels scenarios breaching any of the five thresholds with a non-empty reason string mentioning the specific threshold
- generate_dataset() returns exactly 3,528 scenarios per mechanics point with all scenario IDs unique, every required key present in every scenario dict, and both flow patterns and all rise time values represented
- The Normal lung dataset has greater than 25% valid scenarios and the high-resistance Bronchospasm dataset has greater than 30% invalid scenarios
- All 80 unit tests in tests/test_pcv_generator.py pass

---

## Files Likely to Be Touched

- **Created:** `generator/pcv_generator.py` — the PCV generator module containing generate_breath_cycles(), generate_dataset(), PARAMETER_GRID, the three-phase pressure profile, the ODE definition, the fill fraction calculation, the eight derived metrics, the five-threshold validity filter, the scenario ID encoder, and parameter validation
- **Created:** `tests/test_pcv_generator.py` — 80 unit tests across five classes covering interface contract, physiological plausibility, PCV waveform shape including rise time effects and fill fraction physics, validity filter logic, and dataset generation structure
- **Created:** `generate_pcv_dataset.py` — the batch script that sweeps the full mechanics grid across all seven condition tiers, writes per-scenario waveform CSVs, produces the manifest and generation log, and prints an ETA after each completed tier for overnight monitoring
- **Populating:** `data/exports/pcv/` — waveform CSV files, pcv_manifest.csv (one row per scenario), and pcv_generation_log.json (run provenance including parameter grid, per-tier counts, and total runtime), currently being written by the running batch process as PID 59770

---

## Status

**Complete**

generator/pcv_generator.py is implemented and all 80 unit tests in tests/test_pcv_generator.py pass. The full PCV dataset generation batch run is currently executing as PID 59770 and writing output to data/exports/pcv/. The status field will be updated with the final valid scenario count and total runtime once the batch run completes.
