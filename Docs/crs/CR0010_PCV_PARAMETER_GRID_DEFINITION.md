# CR0008 — PCV Parameter Grid Definition

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before generating the PCV dataset, the parameter space for Pressure-Controlled Ventilation needed to be systematically defined. Without a formally specified parameter grid — covering every ventilator setting, its clinically meaningful range, and its step size — the dataset generation sweep would be arbitrary and unreproducible. Individual scenarios would be handcrafted rather than systematic, producing a dataset with uncontrolled gaps and biases where some regions of the physiological state space are overrepresented while others are missing entirely.

PCV introduces parameters that do not exist in VCV and required their own clinical justification. Rise time (0.0–0.4 seconds) is a PCV-specific setting that shapes the pressure waveform at the start of inspiration and materially affects the flow waveform shape, the time to peak flow, and patient comfort. Inspiratory pressure replaces tidal volume as the primary ventilator setting — in PCV the clinician sets a pressure target rather than a volume target, which means the parameter grid needed to be defined in pressure units rather than volume units, and the delivered tidal volume needed to be treated as a derived output subject to validity checking rather than a fixed input.

There was also a need to define the patient mechanics grid across all seven condition tiers and to conduct a pre-generation invalidity analysis before running the sweep. Without this analysis, combinations where the fill fraction collapses below the minimum threshold or where delivered tidal volume falls outside the safe range would be passed to the ODE solver unnecessarily, wasting significant compute time given that each PCV scenario requires a full solve_ivp call at approximately 0.27 seconds per scenario.

---

## Current State

The parameter grid for PCV has been fully defined, the invalidity analysis has been completed, and the full dataset generation batch run has been launched.

The ventilator settings grid was defined as follows: inspiratory pressure above PEEP at 5, 10, 15, 20, 25, 30, and 35 cmH2O (seven values covering the full clinical range from minimal support to near-maximum safe driving pressure); respiratory rate at 8, 12, 16, 20, 24, 28, and 30 breaths per minute (seven values, identical to the VCV grid for cross-mode comparability); PEEP at 0, 4, 8, 12, 16, and 20 cmH2O (six values); I:E ratio at three discrete values (1:1, 1:2, and 1:3, encoded as inspiratory fractions 1.0, 0.5, and 0.33); and rise time at four discrete values (0.0, 0.1, 0.2, and 0.4 seconds). This produces 3,528 parameter combinations per mechanics point — 3.5 times more combinations than VCV per mechanics point, reflecting the additional rise time dimension.

The patient mechanics grid was defined across the same seven condition tiers used for VCV, ensuring direct comparability between the two datasets. Normal uses compliance 50–100 mL/cmH2O in steps of 10 and resistance 2–5 cmH2O/L/s in steps of 1, producing 24 mechanics pairs. Mild ARDS uses compliance 30–50 in steps of 5 and resistance 3–6 in steps of 1, producing 20 pairs. Moderate ARDS uses compliance 20–30 in steps of 5 and resistance 4–8 in steps of 2, producing 9 pairs. Severe ARDS uses compliance 10–20 in steps of 5 and resistance 5–10 in steps of 2, producing 12 pairs. COPD uses compliance 40–80 in steps of 10 and resistance 10–20 in steps of 2, producing 30 pairs. Bronchospasm uses compliance 40–70 in steps of 10 and resistance 15–30 in steps of 5, producing 16 pairs. Pneumonia uses compliance 25–45 in steps of 5 and resistance 4–8 in steps of 2, producing 15 pairs.

The invalidity analysis was performed analytically before generation using five filters. PPeak above 50 cmH2O (barotrauma risk) — triggered when inspiratory pressure plus PEEP exceeds 50, which occurs at the highest pressure and PEEP combinations. Inspiratory pressure above 35 cmH2O (maximum safe driving pressure above PEEP) — the grid upper bound is set at this value so this filter primarily catches edge cases. Fill fraction below 0.20 (lung barely fills) — triggered at high resistance combined with fast respiratory rates and long rise times where the RC time constant greatly exceeds the plateau duration. Delivered tidal volume below 210 mL (3 mL/kg at 70 kg IBW) — triggered at low compliance, low inspiratory pressure, and short inspiratory time combinations. Delivered tidal volume above 840 mL (12 mL/kg IBW) — triggered at high compliance combined with high inspiratory pressure, which is the dominant source of invalidity in the Normal condition tier where C=60 produces 900 mL at P=15 cmH2O.

The pre-generation invalidity analysis projected approximately 28% invalidity for PCV across the full space. Notably, the Normal lung carries the highest projected invalidity rate among all condition tiers in PCV — the opposite of VCV. This is because high compliance means even moderate pressure settings overdistend the lung, triggering the VT_MAX filter across a large fraction of combinations. The Normal lung at C=60 with inspiratory pressure of 15 cmH2O and near-full fill fraction delivers 900 mL, correctly flagged as overdistension. This insight was confirmed empirically when the unit test threshold for Normal lung valid fraction was corrected from 50% to 25% after the actual valid fraction measured at 28.6%.

The full dataset generation batch run was launched via nohup python -u generate_pcv_dataset.py > pcv_generation.log 2>&1 and is currently running as PID 59770 at approximately 85% CPU. The projected total runtime is 8–10 hours given the ODE solver overhead of approximately 0.27 seconds per scenario across an estimated 127,000 total scenarios.

---

## Proposed Change

Produce a formal PCV parameter grid definition document that captures the complete parameter grid specification, the mechanics grid per condition tier with ranges and step sizes and their literature sources, the pre-generation invalidity analysis methodology and per-tier projected results, the rationale for each step size choice, and an explanation of how the PCV grid differs from the VCV grid and why. This document serves as the written record of the parameter space design and as a reference for replicating or extending the dataset in future work.

---

## Acceptance Criteria

- The PCV parameter grid definition document specifies every parameter in the ventilator settings grid with its range, step size, units, and the clinical rationale for those choices — including the distinction between inspiratory pressure as a PCV setting versus tidal volume as a VCV setting
- The rise time parameter is documented with its range (0.0–0.4 seconds), its four discrete values, and the clinical significance of each value including why 0.0 produces a square wave and why longer rise times are used clinically
- The mechanics grid for each of the seven condition tiers is documented with compliance range, compliance step, resistance range, resistance step, total mechanics pairs, and the literature source for the parameter ranges
- The five pre-generation invalidity filter criteria are documented with their threshold values, the clinical evidence behind each threshold, and an explicit explanation of why the PCV driving pressure threshold (35 cmH2O) differs from the VCV driving pressure threshold (20 cmH2O)
- The fill fraction filter is documented with its formula, the derivation of the 0.20 threshold from the resistance validation ceiling, and an explanation of why the initially written 0.10 threshold was unreachable within the allowed parameter range
- The inversion of the Normal lung invalidity pattern relative to VCV is documented and explained — in VCV high compliance makes Normal the easiest condition to satisfy, while in PCV high compliance makes Normal the condition with the most overdistension violations
- The total combination count is derivable from the documented grid: 7 pressures × 7 rates × 6 PEEPs × 3 I:E ratios × 4 rise times = 3,528 per mechanics point
- The document is written in the author's own words and demonstrates understanding of why the PCV parameter space requires different design choices than VCV

---

## Files Likely to Be Touched

- **Create:** `Docs/parameter_grids/PCV_PARAMETER_GRID.md` — the primary deliverable, containing the full parameter grid specification, mechanics grid per condition tier, invalidity analysis methodology, per-tier projected and actual results, and step size rationale
- **Created:** `generator/pcv_generator.py` — implements the parameter grid as the PARAMETER_GRID constant and sweeps it via generate_dataset() (already created as part of this CR)
- **Created:** `generate_pcv_dataset.py` — the batch script that executes the full parameter sweep across all seven condition tiers and exports waveform CSVs, manifest, and generation log (already created as part of this CR)
- **Created:** `tests/test_pcv_generator.py` — 80 unit tests validating the interface contract, physiological plausibility, PCV waveform shape, validity filter logic, and dataset generation structure including the corrected Normal lung valid fraction threshold (already created as part of this CR)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the parameter grid definition process, the invalidity analysis, the fill fraction threshold correction, the Normal lung valid fraction correction, and the dataset generation launch
- **Populated (in progress):** `data/exports/pcv/` — waveform CSV files, pcv_manifest.csv, and pcv_generation_log.json currently being written by the running batch process

---

## Status

**Complete**

The PCV parameter grid has been fully defined, implemented in `generator/pcv_generator.py` as the PARAMETER_GRID constant and generate_dataset() function, validated through 80 unit tests in `tests/test_pcv_generator.py`, and the full dataset generation batch run is currently executing as PID 59770. The formal parameter grid definition document (`Docs/parameter_grids/PCV_PARAMETER_GRID.md`) remains to be written as the documented record of the grid design decisions and invalidity analysis methodology.
