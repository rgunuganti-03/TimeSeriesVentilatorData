# CR0019 — SIMV Control Loop Documentation

**Author:** Riya Gunuganti
**Date:** 2026-07-29
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before implementing the SIMV generator, the control loop logic for Synchronized Intermittent Mandatory Ventilation needed to be formally defined and documented. SIMV is structurally different from every mode built so far: VCV, PCV, PSV, and PRVC each have exactly one control regime active for the entire waveform, so "what does the ventilator control?" has one answer for the whole scenario. SIMV does not have a single answer — it switches between a mandatory breath (VC or PC sub-mode, reusing VCV/PCV physics) and a spontaneous breath (reusing PSV physics), on the same patient, within one continuous waveform. Without a written specification of the scheduling logic that decides which regime is active at a given moment, there was a real risk of building a generator that either mis-sequences the two breath types, fails to prevent breath stacking, or loses lung-state continuity across the transition between mandatory and spontaneous physics — none of which the other four generators had to solve, since none of them hand off live compartment state mid-scenario between two different physics regimes.

---

## Current State

The control loop logic for SIMV has been fully defined and documented in `Docs/control_loops/SIMV_CONTROL_LOOP.md` (427 lines).

SIMV was specified as introducing no new lung physics — every equation of motion it needs already exists in `vcv_generator.py`, `pcv_generator.py`, and `psv_generator.py`. What it adds is entirely new scheduling logic: the synchronization window. The mandatory-rate macro-cycle length is `T_mand = 60 / respiratory_rate`, and a window of width `W = f_window × T_mand` opens at `T_mand − W` and closes at `T_mand`. Patient effort attempts occurring before the window opens trigger a spontaneous (PSV-style) breath; attempts occurring inside the window trigger a synchronized mandatory breath immediately; if the window closes with no successful trigger, a time-triggered mandatory breath is delivered at the scheduled time to guarantee the set rate. Once a mandatory breath is triggered — synchronized or time-triggered — patient effort plays no further role in its delivery; the breath proceeds exactly as a plain VCV or PCV breath would, matching how neither of those generators has a patient-effort term in its own equation of motion.

Breath-stacking prevention — the requirement `ARCHITECTURE.md` explicitly flags for this mode — was specified as following from two properties of the design rather than a single explicit check: exactly one mandatory breath is delivered per macro-cycle by construction, and the sequential single-active-breath execution model (no new breath of either type starts while one is already in progress, matching the event-driven loop structure `psv_generator.py` already uses) means a spontaneous breath in progress when the window opens is simply allowed to finish before the window check applies to the next candidate breath.

Continuity of compartment volume and auto-PEEP state across breath-type transitions was identified as the one genuinely new architectural requirement this mode introduces — the other four generators are each self-contained, single-regime simulators that never need to hand live lung state from one physics regime to another mid-scenario. The document also flags, as a Step 3 implementation question rather than resolving it here, that all four existing generators independently note the same deferred `generator/lung_physics.py` refactor in their own docstrings, and that SIMV is the first mode where that refactor's value becomes concrete rather than hypothetical.

While drafting the document, a compartment-count cross-reference between the module docstrings and the live `COMPARTMENT_PROFILES` dictionaries in `vcv_generator.py` and `pcv_generator.py` surfaced a stale comment claiming Bronchospasm uses 1 compartment; the actual dictionaries and smoke-test assertions in both files confirm 2, matching PSV. This was noted as a one-line docstring fix, not a physiology issue, and does not block SIMV implementation since SIMV reads the correct 2-compartment structure directly.

Three open decisions were flagged in the document and confirmed in a follow-up review before parameter-grid work began:

1. **Trigger mechanism** — SIMV's mandatory-breath trigger reuses PSV's existing pressure-based `_check_trigger` (comparing patient effort against a cmH₂O threshold net of auto-PEEP) rather than the brief's separate flow-based "trigger sensitivity" units, since PSV's actual shipped implementation is pressure-based and reusing it avoids introducing a second, differently-typed trigger mechanic that would need its own validation.
2. **Synchronization window width** — formalized as `W = f_window × T_mand`, with `f_window` swept as a tunable parameter rather than fixed to one literature value, since no single ventilator platform defines this the same way (see CR0022).
3. **I:E ratio and rise time** — the brief's ventilator-settings table did not list SIMV as a mode either parameter applies to, which was judged a table gap rather than an intentional exclusion; both are inherited unchanged for the corresponding sub-breath type (I:E for mandatory-breath timing, rise time for PC-mandatory and spontaneous breaths).

---

## Proposed Change

Produce a formal SIMV control loop document that captures the mechanistic description of the mode's hybrid control regime, the synchronization window state machine that governs breath-type selection, the equation of motion for each of the three breath-type regimes, the breath-stacking prevention argument, the compartment/auto-PEEP continuity requirement, and the architecture-fork question this mode raises for Step 3. This document serves as the written record of the domain understanding behind the implementation and as the reference against which the generated waveforms in `generator/simv_generator.py` can be validated.

---

## Acceptance Criteria

- The document explains why SIMV has no single independent/dependent variable pair, and states the correct pair for each of the three breath-type regimes (mandatory VC, mandatory PC, spontaneous)
- The synchronization window is defined precisely: `T_mand`, `W`, `window_open`, and the three possible outcomes of a patient-effort attempt (spontaneous, synchronized mandatory, time-triggered mandatory) depending on when it occurs
- The breath-stacking prevention argument is stated explicitly and follows from the design rather than an ad hoc check
- The requirement to carry compartment volume and auto-PEEP state continuously across breath-type transitions is documented as the one architectural property unique to this engine among the five
- The compartment-count discrepancy found while cross-referencing `vcv_generator.py` / `pcv_generator.py` docstrings against their live `COMPARTMENT_PROFILES` dictionaries is documented, with the correct value (Bronchospasm = 2) stated and the source of the stale comment identified
- The three decisions left open in the document (trigger mechanism, window-width formula, I:E ratio / rise-time applicability) are documented as open questions along with the reasoning that resolved each one
- The document flags, without resolving, the Step 3 implementation-strategy fork between refactoring shared physics into `generator/lung_physics.py` and inlining a fifth self-contained physics copy
- The document is written in the author's own words and demonstrates understanding of why SIMV's scheduling logic — not new lung physics — is the genuinely novel part of this mode

---

## Files Likely to Be Touched

- **Create:** `Docs/control_loops/SIMV_CONTROL_LOOP.md` — the primary deliverable, containing the full hybrid control-loop description, synchronization window state machine, per-regime equations of motion, breath-stacking prevention argument, compartment-continuity requirement, and the Step 3 architecture-fork note (already created as part of this CR)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the control loop definition process, the compartment-count docstring discrepancy found during cross-referencing, and the three open decisions and how each was resolved
- **Update:** `ARCHITECTURE.md` — move SIMV from the "future" scaling-path entry to a documented, in-progress component alongside VCV, PCV, PSV, and PRVC

---

## Status

**Complete**

The SIMV control loop logic has been fully defined and documented in `Docs/control_loops/SIMV_CONTROL_LOOP.md`. All three decisions flagged as open have been confirmed and carried forward into the parameter grid definition (CR0022) and generator implementation (CR0020). The document is ready to serve as the reference specification against which `generator/simv_generator.py`'s waveforms were validated during implementation.
