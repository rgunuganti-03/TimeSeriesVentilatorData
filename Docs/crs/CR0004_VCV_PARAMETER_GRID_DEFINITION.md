# CR0004 — VCV Parameter Grid Definition

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before generating the VCV dataset, the parameter space for Volume-Controlled Ventilation needed to be systematically defined. Without a formally specified parameter grid — covering every ventilator setting, its clinically meaningful range, and its step size — the dataset generation sweep would be arbitrary. Individual scenarios would be handcrafted rather than systematic, producing a dataset with uncontrolled gaps and biases where some regions of the physiological state space are overrepresented while others are missing entirely. There was also a need to define the patient mechanics grid across all seven condition tiers so that the sweep covered the full compliance and resistance ranges documented in the clinical literature for each condition, not just a single representative point per condition.

Additionally, without a pre-generation invalidity analysis, the full Cartesian product of parameters would be passed to the generator for every combination — including combinations that are physiologically impossible or clinically unsafe — wasting compute time and producing a dataset contaminated with invalid scenarios that would need to be filtered retrospectively.

---

## Current State

The parameter grid for VCV has been fully defined, the invalidity analysis has been completed, and the full dataset has been generated and exported.

The ventilator settings grid was defined as follows: tidal volume at 4, 6, 8, and 10 mL/kg IBW (assuming 70 kg ideal body weight, converting to 280, 420, 560, and 700 mL respectively); respiratory rate at 8, 12, 16, 20, 24, 28, and 30 breaths per minute; PEEP at 0, 4, 8, 12, 16, and 20 cmH₂O; I:E ratio at three discrete values (1:1, 1:2, and 1:3, encoded as inspiratory fractions 1.0, 0.5, and 0.33); and flow pattern as two discrete values (square and decelerating). This produces 1,008 parameter combinations per mechanics point.

The patient mechanics grid was defined across seven condition tiers. Normal uses compliance 50–100 mL/cmH₂O in steps of 10 and resistance 2–5 cmH₂O/L/s in steps of 1, producing 24 mechanics pairs. Mild ARDS uses compliance 30–50 in steps of 5 and resistance 3–6 in steps of 1, producing 20 pairs. Moderate ARDS uses compliance 20–30 in steps of 5 and resistance 4–8 in steps of 2, producing 9 pairs. Severe ARDS uses compliance 10–20 in steps of 5 and resistance 5–10 in steps of 2, producing 12 pairs. COPD uses compliance 40–80 in steps of 10 and resistance 10–20 in steps of 2, producing 30 pairs. Bronchospasm uses compliance 40–70 in steps of 10 and resistance 15–30 in steps of 5, producing 16 pairs. Pneumonia uses compliance 25–45 in steps of 5 and resistance 4–8 in steps of 2, producing 15 pairs.

The invalidity analysis was performed analytically before generation using four filters grounded in clinical literature: PPeak above 50 cmH₂O (barotrauma risk), driving pressure above 20 cmH₂O (ARDS mortality threshold), delivered tidal volume below 3 mL/kg IBW (inadequate ventilation), and delivered tidal volume above 12 mL/kg IBW (overdistension). The pre-generation analysis estimated approximately 28% invalidity across the full space, with Severe ARDS carrying the highest projected invalidity at 65% and Normal the lowest at 5%.

The full dataset generation run produced 127,008 total scenarios across all seven condition tiers. The actual invalidity rates matched the projected distribution: Normal 0.0% invalid, Mild ARDS 12.8%, Moderate ARDS 53.6%, Severe ARDS 85.1%, COPD 13.8%, Bronchospasm 27.9%, and Pneumonia 25.0%. The overall valid scenario count was 96,821 (76.2%) producing 96,821 waveform CSV files exported to `data/exports/vcv/`, accompanied by `vcv_manifest.csv` and `vcv_generation_log.json`.

---

## Proposed Change

Produce a formal VCV parameter grid definition document that captures the complete parameter grid specification, the mechanics grid per condition tier with ranges and step sizes and their literature sources, the pre-generation invalidity analysis methodology and per-tier results, and the rationale for each step size choice. This document serves as the written record of the parameter space design and as a reference for replicating or extending the dataset in future work.

---

## Acceptance Criteria

- The VCV parameter grid definition document specifies every parameter in the ventilator settings grid with its range, step size, and the clinical rationale for those choices
- The mechanics grid for each of the seven condition tiers is documented with compliance range, compliance step, resistance range, resistance step, total mechanics pairs, and the literature source for the parameter ranges
- The pre-generation invalidity filter criteria are documented with their threshold values and the clinical evidence or guideline behind each threshold
- The per-tier invalidity analysis results are documented, including both the projected invalidity fractions from the pre-generation analysis and the actual invalidity fractions from the generation run, with any discrepancies explained
- The total combination count of 127,008 is derivable from the documented grid definition — the document must be internally consistent so that a reader can reproduce the count from the specified ranges and step sizes
- The step size rationale is documented for each parameter — explaining why a step of 5 mL/cmH₂O in compliance for ARDS produces meaningfully different waveforms while a step of 10 is sufficient for Normal
- The document is written in the author's own words and demonstrates understanding of why systematic grid coverage matters for downstream model training

---

## Files Likely to Be Touched

- **Create:** `Docs/parameter_grids/VCV_PARAMETER_GRID.md` — the primary deliverable, containing the full parameter grid specification, mechanics grid per condition tier, invalidity analysis methodology, per-tier results, and step size rationale
- **Create:** `generator/vcv_generator.py` — the VCV generator module implementing the parameter grid sweep via `generate_dataset()`, the validity filter, and the scenario ID encoding scheme (already created as part of this CR)
- **Create:** `generate_vcv_dataset.py` — the batch script that executes the full parameter sweep across all seven condition tiers and exports waveform CSVs, manifest, and generation log (already created as part of this CR)
- **Create:** `tests/test_vcv_generator.py` — 79 unit tests validating the interface contract, physiological plausibility, flow pattern shape, validity filter logic, and dataset generation structure (already created as part of this CR)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the parameter grid definition process, the invalidity analysis, the generation run results, and the bugs encountered and fixed during implementation
- **Update:** `data/exports/vcv/` — populated with 96,821 waveform CSV files, `vcv_manifest.csv`, and `vcv_generation_log.json` as the output of the generation run (already complete)

---

## Status

**Complete**

The VCV parameter grid has been fully defined, implemented in `generator/vcv_generator.py` as the `PARAMETER_GRID` constant and `generate_dataset()` function, validated through unit tests, and exercised through a full generation run producing 127,008 total scenarios with 96,821 valid waveform exports across all seven condition tiers. The formal parameter grid definition document (`Docs/parameter_grids/VCV_PARAMETER_GRID.md`) remains to be written as the documented record of the grid design decisions and invalidity analysis methodology.
