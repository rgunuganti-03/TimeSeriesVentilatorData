# CR0024: Mode-Stratified Parameter Architecture (Paralyzed vs. Spontaneous)

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0025 (Literature-Grounded Parameter Audit)

---

## Problem

Condition presets in `conditions.py` were designed as a single parameter set per condition, shared across all five ventilation modes. This conflated two physiologically distinct patient states that the project's own modes actually represent: VCV, PCV, and PRVC model a paralyzed or deeply sedated patient with no spontaneous respiratory effort, while PSV and SIMV model a spontaneously breathing or recovering patient actively triggering and shaping their own breaths. A single preset cannot correctly serve both without contradiction.

This surfaced concretely in the Bronchospasm preset: the same effort-related parameters and PEEP value were applied regardless of which mode was simulating the condition, producing outputs that made physiological sense for one patient state and not the other — most visibly, spontaneous-mode PEEP was not distinguishable in the model from paralyzed-mode PEEP, despite the two serving fundamentally different physiological purposes.

---

## Current State

Mode family now gates which parameter blocks are active:

- **Paralyzed/deeply sedated modes (VCV, PCV, PRVC):** the patient-effort block (`pmus_peak_cmH2O`, `effort_rate_per_min`, `effort_duration_s`, `pmus_cv`, `trigger_threshold_cmH2O`) is zeroed or inactive. PEEP's only physiological role in this state is raising end-expiratory lung volume — in obstructive disease this risks worsening gas trapping, so paralyzed-mode PEEP is kept low/conservative.
- **Spontaneous/recovering modes (PSV, SIMV):** the patient-effort block is active and drives triggering, cycling, and dyssynchrony behavior. PEEP can additionally counterbalance auto-PEEP in this state, but only where dynamic airway collapse creates a genuine Starling-resistor flow-limitation mechanism.

This asymmetry is condition-specific, not just mode-specific: COPD exhibits the Starling-resistor mechanism (dynamic airway collapse during expiration), so higher spontaneous-mode PEEP is physiologically justified there. Bronchospasm does not exhibit this mechanism (its obstruction is smooth-muscle-mediated bronchoconstriction, not dynamic collapse), so aggressive PEEP counterbalancing is not justified for Bronchospasm even in spontaneous modes. This is why COPD and Bronchospasm now carry different PEEP escalation between paralyzed and spontaneous presets, rather than a uniform delta.

Finalized PEEP values, derived from the simulator's own auto-PEEP output (using a 75–80%-of-auto-PEEP rule rather than borrowed external clinical averages, to stay internally consistent with the model's own physics):

| Condition | Paralyzed (VCV/PCV/PRVC) | PSV | SIMV |
|---|---|---|---|
| COPD | 5 | 7 | 7 |
| Bronchospasm | 0 | 3 | 1 |

---

## Proposed Change

Formalize mode-family gating as a first-class structural rule in `conditions.py` rather than an implicit convention:

- Document, per condition, which parameter fields are mode-family-gated (effort block, PEEP) versus mode-family-invariant (compliance, resistance, tidal volume target, etc.)
- Encode the PEEP table above as the authoritative source for COPD and Bronchospasm across all five generators
- Apply the same paralyzed-vs-spontaneous PEEP reasoning to any future condition added to the preset library, rather than re-deriving it ad hoc per condition

---

## Acceptance Criteria

- Bronchospasm no longer produces contradictory outputs between paralyzed and spontaneous modes — effort-block parameters are inactive in VCV/PCV/PRVC runs and active in PSV/SIMV runs for the same condition
- COPD and Bronchospasm PEEP values match the table above exactly across all five generators
- The physiological rationale (Starling-resistor mechanism present in COPD, absent in Bronchospasm) is documented alongside the parameter values, not just encoded silently as a number difference
- No other condition preset shows the same paralyzed/spontaneous contradiction Bronchospasm did — spot-checked across the remaining eight condition tiers

---

## Files Likely to Be Touched

- **Update:** `generator/conditions.py` — mode-family gating logic, finalized COPD/Bronchospasm PEEP values per mode
- **Update:** `EXPERIMENT_LOG.md` — record the Bronchospasm inconsistency, its root cause, and the resolution
- **Update:** `ARCHITECTURE.md` — document the paralyzed-vs-spontaneous mode-family split as a standing design rule

---

## Status

**Complete**

The mode-stratified architecture is implemented, the Bronchospasm inconsistency is resolved, and COPD/Bronchospasm PEEP values are finalized across all five generators.
