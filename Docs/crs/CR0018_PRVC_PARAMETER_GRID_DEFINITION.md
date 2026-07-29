# CR0018: PRVC Parameter Grid Definition

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-07-28
**Related:** CR0015 (PRVC Control Loop Documentation), CR0016 (PRVC Generator Implementation), CR0017 (PRVC Dataset Generation)

---

## Problem

CR0015 defined PRVC's control loop but explicitly left five design decisions open: the test breath bootstrap formula, the convergence tolerance value, the pressure ceiling sweep values, the rise time treatment, and whether PRVC should inherit PSV's full physiological refinement set. None of these could be resolved from the project brief alone, since PRVC's parameter table in the brief omits several PRVC-specific settings entirely and gives no guidance on the outer-loop algorithm's tuning. The parameter grid — both the full sweep and a thinned version suitable for practical dataset generation — needed to be defined and grounded before the generator (CR0016) or dataset run (CR0017) could proceed.

---

## Current State

An initial literature-grounded research pass resolved the five open decisions from CR0015 using manufacturer documentation (Getinge Servo, Dräger AutoFlow, Hamilton APV) and clinical literature (ARDSnet ARMA trial, Amato et al. 2015 driving pressure). Two errors in that initial pass were subsequently caught and corrected before implementation:

First, the settings-grid combination count was originally miscalculated as 2,880 per mechanics point; the correct product of VT(4) × RR(7) × PEEP(6) × I:E(3) × adaptation_step(3) × ceiling(5) × tolerance(2) is 15,120. Once `adaptation_step_cmH2O` and `vt_tolerance_frac` were fixed to single uniform values rather than swept (see below), the correct collapsed total is 2,520 per mechanics point, not 480 as originally stated.

Second, the "126 total mechanics points" figure was computed from the brief's original, uncorrected compliance/resistance ranges rather than the actual `CONDITION_TIERS` shared across `vcv_generator.py`, `pcv_generator.py`, and `psv_generator.py` — which use corrected, more physiologically realistic resistance floors (e.g., Normal R 8–12 rather than the brief's R 2–5). Recomputing against the actual condition tiers gives 137 total mechanics points (25 + 12 + 12 + 15 + 25 + 24 + 24 across Normal, Mild ARDS, Moderate ARDS, Severe ARDS, COPD, Bronchospasm, and Pneumonia respectively), not 126.

A separate decision point arose on whether `adaptation_step_cmH2O` and `vt_tolerance_frac` should vary by condition or be held uniform. Analysis showed the choice is combinatorially free either way — fixing these two parameters removes them from the multiplication regardless of whether the fixed value is shared across all conditions or set independently per condition, so the decision reduces to a question of dataset realism rather than dataset size. The decision was made to hold both **uniform across all conditions**, on the grounds that a real deployed PRVC controller does not know a patient's diagnosis in advance — the entire premise of the adaptive algorithm is discovering the patient's mechanics through feedback, not being handed them. Condition-specific tuning of these two parameters would implicitly assume the ventilator has information no real device has.

---

## Proposed Change

Document and implement the full PRVC parameter grid:

**Swept dimensions (settings grid, 2,520 combinations per mechanics point):**
- Tidal volume target: 4, 6, 8, 10 mL/kg IBW
- Respiratory rate: 8, 12, 16, 20, 24, 28, 30 bpm
- PEEP: 0, 4, 8, 12, 16, 20 cmH2O
- I:E ratio: 1:1, 1:2, 1:3
- Pressure ceiling: 15, 20, 25, 30, 35 cmH2O above PEEP — grounded in Amato et al. 2015 (driving pressure mortality association) and the ARDSnet plateau ≤30 cmH2O ceiling

**Fixed uniform constants (not swept):**
- `adaptation_step_cmH2O` = 2.0 — within the ≤3 cmH2O per-breath cap documented across Servo and Dräger AutoFlow
- `vt_tolerance_frac` = 0.10 — no manufacturer publishes a numeric convergence tolerance; set by engineering analogy to bench-measured no-leak accuracy
- Rise time = 0.10s — the brief's parameter table omits rise time for PRVC entirely; fixed rather than swept since sweeping it would confound the pressure-staircase convergence trajectory that is the dataset's primary product

**Mechanics grid:** adopts the same corrected, condition-specific compliance and resistance ranges already implemented across the other three generators (137 total mechanics points), rather than the brief's uncorrected baseline ranges.

**Thinned grid for production generation (576 combinations per mechanics point, 77.1% reduction):** VT thinned to 3 values, RR and PEEP thinned to 4 values each (matching PCV's exact thinning), I:E fully retained (all 3), and pressure ceiling thinned least aggressively of the five dimensions (4 of 5 values retained) given its outsized role in determining whether ceiling-limited non-convergence appears at all.

**Physiological refinements:** of the twelve refinements already implemented across VCV/PCV/PSV, multi-compartment lung mechanics, non-linear compliance, flow-dependent (Rohrer) resistance, volume-dependent expiratory resistance, and PEEP-recruited compliance were classified essential for PRVC; ETT complications, chest wall compliance, and circuit compliance correction were classified optional; breath-to-breath Pmus variability, patient-ventilator dyssynchrony, and SBT temporal sequencing were classified not applicable to the purely mandatory mode as scoped. One new refinement with no VCV/PCV/PSV analogue — multi-breath moving-average damping of the outer loop's error signal — was identified as necessary specifically because PRVC is the first mode with an inter-breath feedback controller at all.

---

## Acceptance Criteria

- The full settings grid is documented with the correct combination count shown as an explicit product (2,520 per mechanics point), and the prior 2,880/480 miscalculation is corrected rather than silently replaced
- The decision to hold `adaptation_step_cmH2O` and `vt_tolerance_frac` uniform across conditions is documented with the device-realism rationale, and the finding that this choice is combinatorially free (rather than a size/realism tradeoff) is explicitly stated
- The mechanics grid is documented using the actual `CONDITION_TIERS` values shared across `vcv_generator.py`, `pcv_generator.py`, and `psv_generator.py`, with the corrected 137-mechanics-point total shown per tier, not the brief's original uncorrected ranges
- `pressure_ceiling_cmH2O`'s sweep values are grounded in cited literature (Amato et al. 2015; ARDSnet ARMA trial), distinguished clearly from values chosen by engineering analogy where no literature source exists
- The thinned grid (576 combinations per mechanics point) is documented with per-dimension rationale for what was kept and dropped, matching the depth of the PCV thinned-grid documentation
- The essential/optional/not-applicable classification of all twelve existing physiological refinements is documented for PRVC specifically, with reasoning for each
- The one new refinement (outer-loop moving-average damping) with no VCV/PCV/PSV precedent is documented and its necessity explained in terms of PRVC's unique inter-breath feedback structure
- The actual completed generation run results (CR0017) are recorded per condition tier — total scenarios, valid percentage, converged percentage, and ceiling-limited percentage — matching the real production output rather than only projected estimates
- The document is written in the author's own words and demonstrates understanding of why PRVC's parameter space requires design choices with no VCV/PCV/PSV precedent, rather than a reproduction of manufacturer documentation

---

## Files Likely to Be Touched

- **Create:** `Docs/parameter_grids/PRVC_PARAMETER_GRID.md` — the primary deliverable, containing the full and thinned grid specifications, the corrected combinatorics, the mechanics grid per condition tier, the literature-grounded rationale for `pressure_ceiling_cmH2O`, the refinement classification, and the actual per-tier generation results from CR0017 (remains to be written as a standalone document; the underlying analysis is complete and referenced throughout this CR and in `generator/prvc_generator.py`'s module docstring)
- **Created:** `generator/prvc_generator.py` — implements the full parameter grid as the `PARAMETER_GRID` constant and the mechanics grid as `CONDITION_TIERS` (already created as part of CR0016)
- **Created:** `generate_prvc_dataset_thinned.py` — implements the thinned grid as `THINNED_PARAMETER_GRID` and executes the full sweep across all seven condition tiers (already created as part of CR0017)
- **Created:** `tests/test_prvc_generator.py` — includes tests validating the outer-loop control constants and the compartment/recruitment-slope data underlying the mechanics grid (already created as part of CR0016)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the combinatorics corrections, the uniform-vs-per-condition decision, and the literature sources used

---

## Status

**Complete**

The PRVC parameter grid has been fully defined and implemented in `generator/prvc_generator.py` (`PARAMETER_GRID`, `CONDITION_TIERS`) and `generate_prvc_dataset_thinned.py` (`THINNED_PARAMETER_GRID`), validated through the unit tests in `tests/test_prvc_generator.py`, and exercised through the completed dataset generation run documented in CR0017 (78,912 scenarios, 96.9% valid). The formal parameter grid definition document (`Docs/parameter_grids/PRVC_PARAMETER_GRID.md`) remains to be written as the standalone documented record of the grid design decisions, literature sources, and refinement classification — consistent with the same gap noted in CR0010 for PCV's equivalent document.
