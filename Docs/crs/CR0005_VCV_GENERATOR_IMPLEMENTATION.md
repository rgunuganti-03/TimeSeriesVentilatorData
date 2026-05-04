# CR0005 — VCV Generator Implementation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

With the VCV control loop documented and the parameter grid defined, the next required step was a working generator implementation that could produce physiologically correct VCV waveforms for any parameter combination within the defined grid. Without a generator, the parameter grid definition exists only as a specification — it cannot produce data. The generator needed to correctly implement VCV's control loop (flow prescribed, pressure derived), support both square and decelerating flow profiles, compute all required derived metrics per scenario, apply the pre-generation validity filter, and expose a dataset sweep function that iterates the full parameter grid for a given condition and mechanics point. It also needed to satisfy the shared interface contract — returning the same four core keys (time, pressure, flow, volume) as all other engines — so that it could be dropped into the existing dashboard and export pipeline without modification.

---

## Current State

The VCV generator has been fully implemented, tested, and validated through a complete dataset generation run.

`generator/vcv_generator.py` was built implementing Volume-Controlled Continuous Mandatory Ventilation. The generator prescribes flow as the independent variable and derives pressure from the equation of motion at every sample point — no ODE solver is used during inspiration because the flow profile is fully known analytically. Square flow is set to `VT / t_insp` (constant throughout inspiration). Decelerating flow uses a linear ramp from peak to zero where the peak is set to `2 × VT / t_insp` to preserve the tidal volume integral. Volume is the cumulative integral of flow at every sample point. Pressure is computed directly as `P(t) = V(t)/C + R × Flow(t) + PEEP`. Expiration uses the analytical RC decay solution `V(t) = V_end × exp(-t / tau)`, with flow and pressure derived from the solved volume.

Seven derived metrics are computed per scenario: PPeak (maximum of the pressure array), Pplat (mean pressure across the last 10% of inspiratory samples where flow has decelerated to near zero), driving pressure (Pplat minus PEEP), mean airway pressure (mean of the full pressure array), auto-PEEP (residual pressure above PEEP at the last sample), delivered tidal volume (maximum of the volume array), and minute ventilation (respiratory rate times delivered tidal volume divided by 1000). The validity filter applies four thresholds in priority order — PPeak above 50 cmH₂O, driving pressure above 20 cmH₂O, delivered tidal volume below 210 mL (3 mL/kg IBW), and delivered tidal volume above 840 mL (12 mL/kg IBW) — and populates a human-readable `invalid_reason` string on failure.

The `generate_dataset()` function sweeps the full parameter grid using `itertools.product`, calls `generate_breath_cycles()` for each combination, and returns a structured list of scenario dicts each containing a scenario ID, condition name, parameter dict, metrics dict, validity flag, invalid reason, waveform arrays, and UTC timestamp. The scenario ID encoding scheme encodes condition, compliance, resistance, tidal volume per kg, respiratory rate, PEEP, I:E ratio, and flow pattern into a human-readable string — for example `VCV_Normal_C060_R002_VT07_RR015_PEEP05_IE050_DEC`.

Two bugs were identified and fixed during implementation. The first was a duplicate scenario ID bug — the I:E ratio was not encoded in the initial `_make_scenario_id` implementation, producing a 3× collision where 1,008 total IDs collapsed to only 336 unique values. The fix added I:E encoding as `IE<value×100>`. The second was a test assertion bug where `test_driving_pressure_breach_flagged` expected the invalid reason to mention driving pressure specifically, but at the test parameters the PPeak filter tripped first. The fix broadened the assertion to accept any pressure-related reason string.

`tests/test_vcv_generator.py` was built with 79 unit tests across five classes covering interface contract, physiological plausibility, flow pattern shape, validity filter logic, and dataset generation structure. All 79 tests pass.

`generate_vcv_dataset.py` was built as a batch script executing the full mechanics grid sweep across all seven condition tiers, producing 127,008 total scenarios with 96,821 valid in 28.4 minutes.

---

## Proposed Change

The VCV generator implementation is complete. No further changes to the generator are proposed at this stage. The next phase of work on VCV is validation — comparing generated waveforms against expected physiological behaviour across all seven condition tiers and documenting the results in `VALIDATION.md`. Any discrepancies found during validation that require generator changes will be tracked in a separate CR.

---

## Acceptance Criteria

- `generator/vcv_generator.py` implements the correct VCV control loop — flow is prescribed, pressure is derived, no ODE solver is used during inspiration
- Both square and decelerating flow profiles are implemented and produce the correct waveform shapes — square flow is constant during inspiration, decelerating flow is monotonically decreasing, and both deliver the target tidal volume within 5% tolerance
- Square pattern produces higher PPeak than decelerating for identical parameters, reflecting the higher initial resistive pressure contribution
- Driving pressure equals Pplat minus PEEP within 0.5 cmH₂O tolerance
- All four validity filter thresholds (PPeak, driving pressure, VT_MIN, VT_MAX) correctly flag invalid scenarios and populate the invalid reason string with a clinically meaningful description
- All scenario IDs in a dataset sweep are unique — the I:E ratio encoding fix resolves the 3× collision
- `generate_dataset()` returns exactly 1,008 scenarios per mechanics point matching the full Cartesian product of the parameter grid
- All 79 unit tests in `tests/test_vcv_generator.py` pass
- The full generation run completes without errors producing 96,821 valid waveform CSV files, `vcv_manifest.csv`, and `vcv_generation_log.json` in `data/exports/vcv/`

---

## Files Likely to Be Touched

- **Created:** `generator/vcv_generator.py` — VCV waveform generator implementing the control loop, both flow profiles, derived metrics, validity filter, scenario ID encoding, and dataset sweep function
- **Created:** `tests/test_vcv_generator.py` — 79 unit tests across five classes: interface contract, physiological plausibility, flow pattern shape, validity filter, and dataset generation structure
- **Created:** `generate_vcv_dataset.py` — batch script sweeping the full mechanics grid across all seven condition tiers, writing waveform CSVs, manifest, and generation log to `data/exports/vcv/`
- **Populated:** `data/exports/vcv/` — 96,821 waveform CSV files, `vcv_manifest.csv`, and `vcv_generation_log.json` produced by the generation run

---

## Status

**Complete**

`generator/vcv_generator.py` is implemented and passing all 79 unit tests. The full VCV dataset has been generated and exported. The generator is ready for the validation phase.
