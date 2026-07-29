# CR0022 — SIMV Parameter Grid Definition

**Author:** Riya Gunuganti
**Date:** 2026-07-29
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

With the SIMV control loop defined (CR0019) and three open modeling decisions resolved, the full parameter grid needed to be defined before generator implementation could begin. This was a larger design problem than VCV's or PCV's own parameter grids: SIMV stacks three axes of complexity that no sibling engine combines — a mandatory-mode choice (VC or PC) that changes which sub-physics and settings apply, the full spontaneous-breath / patient-effort parameter set PSV needs, and SIMV's own mode-defining synchronization-window parameter, `f_window`, which has no existing implementation anywhere in the project to inherit a value from. The brief's headline parameter table also could not be trusted as-is for two dimensions (flow-cycle threshold and the mandatory-rate range) without literature grounding, since a same-magnitude reuse of PSV's original values risked baking in defaults the clinical literature does not actually support for SIMV's use case.

---

## Current State

The SIMV parameter grid has been fully defined, implemented in `generator/simv_generator.py` as the `PARAMETER_GRID` constant, and grounded by a dedicated literature-review pass.

**Literature grounding.** Before finalizing numeric ranges, external literature was reviewed on four specific questions: how ICU ventilator platforms actually define the synchronization window (Servo, Dräger, Puritan Bennett 840/980, CareFusion AVEA all differ — Servo frames it as "first 90% of breath cycle time," Dräger as "~20% of expiratory time, 0–80% selectable," with no single agreed value across vendors); whether flow-cycle threshold should vary by condition (Tassaux et al. 2005 supports a high ~0.65–0.70 threshold for obstructive disease; Tokioka et al. 2001 found very low thresholds in ARDS/ALI patients prolong inspiration and increase expiratory work, indicating the ARDS default should sit in the ~0.25–0.40 range rather than the ~0.10 originally carried over from PSV); how asynchrony is characterized and thresholded in the ICU literature (Thille et al. 2006 establishes asynchrony index > 10% as the threshold associated with prolonged ventilation and worse outcomes); and how SIMV compares to other weaning strategies (Brochard 1994 and Esteban 1995, the two pivotal weaning RCTs, both found SIMV among the least effective modes, appropriately hedged given the two trials disagree on which alternative is best and predate routine SIMV+PS practice).

**Parameter grid.** The full grid (`PARAMETER_GRID` in `generator/simv_generator.py`) covers fifteen dimensions split into shared, VC-specific, and PC-specific groups: `mandatory_mode` (VC, PC); `tidal_volume_ml_per_kg` (4, 6, 8, 10 — VC only) and `flow_pattern` (square, decelerating — VC only); `insp_pressure_cmH2O` (10–35 across six values — PC only); `respiratory_rate` (4–12 bpm across five values, deliberately scoped to the clinically relevant SIMV weaning range rather than VCV/PCV's full 8–30 bpm CMV range); `peep_cmH2O` (0–20); `ie_ratio` (1.0, 0.5, 0.33 — mandatory breaths only); `rise_time_s` (0.0–0.4 — PC-mandatory and spontaneous breaths); `f_window` (0.15–0.30, the literature-grounded tunable range); `pressure_support_cmH2O` (5–20 — spontaneous breaths); `flow_cycle_threshold` (0.25, 0.40, 0.65 — the literature-refined set, replacing PSV's original ~0.10 low anchor for the restrictive default); `trigger_threshold_cmH2O` (0.5–3.0); and the patient-effort dimensions `pmus_peak_cmH2O`, `effort_rate_per_min`, `effort_duration_s`, and `pmus_cv`, matching PSV's own patient-effort model exactly since spontaneous breaths reuse that physics wholesale.

**Physiological refinements.** The refinements carried into SIMV's physics are identical to those already validated in VCV, PCV, and PSV, applied per breath-type regime rather than reimplemented: multi-compartment lung mechanics with the same `COMPARTMENT_PROFILES` structure (Normal = 1, ARDS tiers = 2, COPD = 3, Bronchospasm = 2, Pneumonia = 3); flow-dependent (Rohrer) resistance; volume-dependent expiratory resistance; non-linear compliance via stress index; PEEP-recruited compliance with the same condition-specific `RECRUITMENT_SLOPES` (zero for COPD and Bronchospasm); chest wall compliance in series; circuit compliance correction on mandatory-breath delivered volume; and endotracheal tube complications (cuff leak, partial obstruction). One refinement is genuinely new to this engine rather than reused: continuous compartment-volume and auto-PEEP state across breath-type transitions within a single scenario, which none of the four sibling generators need since each is a single-regime simulator. Spontaneous-breath dyssynchrony labeling reuses a five-category subset of PSV's classifier (ineffective trigger, double trigger, delayed cycling, premature cycling, flow starvation); mandatory breaths carry a fixed `"controlled"` label, since they are ventilator-paced by construction regardless of trigger source.

A `FCT_CONDITION_DEFAULTS` guidance dictionary was added alongside the grid, mapping each condition to its literature-informed flow-cycle-threshold default (0.25 for Normal/Pneumonia scaling up to 0.40 for Severe ARDS, 0.65 for COPD/Bronchospasm) for use in future UI presets, without hard-coding the mapping into the generator itself — `flow_cycle_threshold` remains a plain sweepable parameter, matching PSV's own convention.

---

## Proposed Change

Produce a formal SIMV parameter grid definition document that captures the complete grid specification with ranges and rationale for every dimension, the literature-grounding findings that informed the flow-cycle-threshold and synchronization-window ranges specifically, the physiological refinements carried into the mode and the one refinement genuinely new to it, and the reasoning behind scoping the mandatory-rate range more narrowly than VCV/PCV's CMV range. This document serves as the written record of the parameter-space design and as the reference for the thinning decisions made in CR0021.

---

## Acceptance Criteria

- The document specifies every one of the fifteen `PARAMETER_GRID` dimensions with its range, which mandatory sub-mode(s) it applies to, and the rationale for the range chosen
- The literature-grounding findings are documented with citations for the two ranges they directly informed: `flow_cycle_threshold`'s restrictive-condition default (revised from ~0.10 to ~0.25–0.40 per Tokioka et al. 2001) and obstructive-condition default (~0.65, supported by Tassaux et al. 2005), and `f_window`'s tunable range (0.15–0.30, reflecting the absence of a single cross-vendor value)
- The asynchrony-index threshold (> 10%, Thille et al. 2006) used for dyssynchrony-regime labeling is documented with its source
- The decision to scope `respiratory_rate` to 4–12 bpm rather than VCV/PCV's 8–30 bpm range is explained in terms of SIMV's clinical use as a weaning mode
- Every physiological refinement reused from VCV/PCV/PSV is listed, and the one refinement genuinely new to SIMV — continuous compartment/auto-PEEP state across breath-type transitions — is called out explicitly as the exception
- The `FCT_CONDITION_DEFAULTS` guidance dictionary is documented as UI-preset guidance rather than an enforced constraint, consistent with `flow_cycle_threshold` remaining a plain sweepable parameter
- The document is written in the author's own words and demonstrates understanding of why SIMV's parameter space needed literature grounding in places PSV's original grid did not

---

## Files Likely to Be Touched

- **Create:** `Docs/parameter_grids/SIMV_PARAMETER_GRID.md` — the primary deliverable, containing the full parameter grid specification, literature-grounding summary and citations, physiological refinements list, and the mandatory-rate scoping rationale
- **Created:** `generator/simv_generator.py` — implements the parameter grid as the `PARAMETER_GRID` constant, the `FCT_CONDITION_DEFAULTS` guidance dictionary, and the physiological refinements described above (already created as part of this CR; see CR0020 for implementation-correctness detail)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the literature-grounding review, the flow-cycle-threshold revision away from PSV's original ARDS default, and the mandatory-rate scoping decision

---

## Status

**Complete**

The SIMV parameter grid has been fully defined and implemented in `generator/simv_generator.py` as the `PARAMETER_GRID` constant, grounded by a dedicated literature-review pass covering synchronization-window conventions, flow-cycle-threshold evidence, asynchrony-index thresholds, and SIMV-versus-alternative-weaning-mode outcomes. The formal parameter grid definition document (`Docs/parameter_grids/SIMV_PARAMETER_GRID.md`) remains to be written as the documented record of the grid design decisions and literature-grounding methodology.
