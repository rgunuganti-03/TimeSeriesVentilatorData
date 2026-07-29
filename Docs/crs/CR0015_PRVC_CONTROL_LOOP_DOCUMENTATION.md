# CR0015: PRVC Control Loop Documentation

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-07-28
**Related:** CR0007 (PCV Control Loop Documentation), CR0016, CR0017, CR0018

---

## Problem

PRVC (Pressure-Regulated Volume Control) is the fourth ventilation mode in the project's mode-by-mode sequence, following VCV, PCV, and PSV. Per the established workflow (define control loop → define parameter grid → implement and test the generator → create a thinned dataset script → integrate into the dashboard), no implementation work should begin until the control loop's mechanistic behavior is written down and understood. Unlike VCV, PCV, and PSV — where a single breath is a complete, self-contained physiological unit — PRVC introduces a structural problem none of the prior modes have: the ventilator does not know, on the first breath, what pressure will produce the target volume in this specific patient's lungs. Before any code is written, this cold-start problem and the breath-to-breath feedback mechanism that resolves it need to be defined precisely enough to serve as the basis for the generator's physics.

---

## Current State

VCV, PCV, and PSV are complete, implemented, and validated. Each of those modes' control loops is fully self-contained per breath: VCV prescribes flow (volume follows), PCV prescribes pressure (volume follows), and PSV prescribes pressure support triggered by patient effort (volume follows). None of the three requires information from a prior breath to determine the current breath's ventilator-side control variable. PRVC has no existing control loop documentation, no generator, and no tests. The brief (`BRIEF_RIYA_MODE_BY_MODE_DATASET.md`) describes PRVC only at a high level — first breath is a test breath, subsequent breaths adjust pressure to hit the volume target, flow looks like PCV, volume is regulated like VCV — without specifying the mathematical form of the breath-to-breath adjustment, the test breath's bootstrap method, the convergence criterion, or the failure mode when the algorithm cannot reach the target.

---

## Proposed Change

Produce a formal PRVC control loop document that captures the mode's defining structural property — two nested control loops, not one — and defines both precisely enough to implement against:

- **Inner loop (intra-breath):** identical in structure to PCV — a constant working pressure `P_work(n)` is applied for breath `n`'s inspiration (rise, plateau, passive expiratory decay), with volume and flow as dependent variables produced by the standard multi-compartment RC equation of motion.
- **Outer loop (inter-breath):** a discrete feedback controller with no VCV/PCV/PSV analogue. After each breath, delivered tidal volume is measured and compared against the target; if outside tolerance, the working pressure for the next breath is stepped by a fixed increment, clipped between a floor and a safety ceiling.
- **Test breath bootstrap:** breath 1 is a volume-controlled maneuver breath (not pressure-controlled) whose measured plateau pressure seeds the working pressure for breath 2, following documented AutoFlow/Servo behavior rather than a blind population-average compliance guess.
- **Two terminal states:** converged (delivered volume settles within tolerance) and ceiling-limited non-convergence (the pressure required to hit the target exceeds the safety ceiling — a real, clinically meaningful alarm condition, not a simulation defect).
- **Dataset generation implication:** because the working pressure changes breath-to-breath, a valid PRVC scenario cannot be a single-breath snapshot — it must be a multi-breath sequence capturing the full pressure trajectory and a convergence label.

The document also positions PRVC relative to VCV and PCV (comparison table: control variable, dependent variable, inter-breath behavior, volume guarantee, single-breath generatability) and flags open design decisions — test breath formula, convergence tolerance, pressure ceiling values, rise time treatment, refinement inheritance from PSV — explicitly rather than silently resolving them, consistent with the project's practice of surfacing grey zones rather than picking values without documented rationale.

---

## Acceptance Criteria

- The document accurately describes PRVC as having two nested control loops operating at different timescales, and explains why this is structurally distinct from VCV, PCV, and PSV — none of which require information from a prior breath
- The inner loop's equation of motion is stated correctly, using `P_work(n)` notation to make explicit that the applied pressure is a per-breath variable rather than a clinician-set constant
- The outer loop's breath-to-breath algorithm is documented precisely: measured delivered volume, error computation, the fixed-increment step rule matching `ARCHITECTURE.md`'s existing specification ("typically 1–3 cmH2O"), and the floor/ceiling clipping
- The test breath bootstrap is documented with an explicit, implementable formula, and the choice between a fixed-pressure start, a blind assumed-compliance start, and a measured-plateau start is made with stated rationale rather than left ambiguous
- Both terminal states — converged and ceiling-limited non-convergence — are defined precisely, and ceiling-limited non-convergence is explicitly identified as a labeled, clinically real outcome rather than an error condition
- The document explains why PRVC scenarios require multi-breath sequences and cannot be generated as single-breath snapshots, unlike every prior mode
- A comparison table against VCV and PCV is included, covering control variable, dependent variable, inter-breath behavior, volume guarantee, and single-breath generatability
- Open design decisions requiring confirmation are explicitly flagged as such, not silently resolved
- The document is written in the author's own words and demonstrates understanding of the physiology, not a reproduction of textbook or manufacturer language

---

## Files Likely to Be Touched

- **Created:** `Docs/control_loops/PRVC_CONTROL_LOOP.md` — the primary deliverable, containing the dual-loop control description, equation of motion, test breath bootstrap, convergence/ceiling-limited terminal states, VCV/PCV comparison table, dataset generation implications, and explicitly flagged open design decisions (already created as part of this CR)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the control loop definition process and the open design decisions flagged for mentor review
- **Update:** `ARCHITECTURE.md` — confirm PRVC is documented as the fourth mode in the generator layer description, alongside VCV, PCV, and PSV

---

## Status

**Complete**

The PRVC control loop has been fully defined and written up in `Docs/control_loops/PRVC_CONTROL_LOOP.md`, covering the dual-loop structure, the intra-breath equation of motion, the outer-loop adaptive algorithm, the test breath bootstrap, both terminal states, and the comparison to VCV/PCV. Five open design decisions were flagged explicitly for confirmation before the parameter grid was defined (CR0018): the test breath formula, the convergence tolerance value, the pressure ceiling sweep values, the rise time treatment, and whether PRVC should inherit PSV's full physiological refinement set. All five were subsequently resolved during CR0018 and implemented in CR0016.
