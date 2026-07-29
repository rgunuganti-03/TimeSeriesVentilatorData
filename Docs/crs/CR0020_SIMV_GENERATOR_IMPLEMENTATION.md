# CR0020 — SIMV Generator Implementation

**Author:** Riya Gunuganti
**Date:** 2026-07-29
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

With the control loop defined (CR0019) and the parameter grid specified (CR0022), `generator/simv_generator.py` needed to be implemented and validated. This carried more implementation risk than any prior generator: SIMV is the first engine that must hand live compartment volume and auto-PEEP state between two different physics regimes — mandatory (VC/PC) and spontaneous (PSV) — within a single running scenario, rather than simulating one regime start to finish. A design that got the breath-type scheduling right but the state handoff wrong would produce waveforms that looked plausible breath-to-breath while silently corrupting auto-PEEP and compartment continuity across the transitions, which is exactly the kind of error that would not necessarily fail a smoke test built around a single breath type at a time.

---

## Current State

`generator/simv_generator.py` (1,413 lines) is implemented and passing all unit tests.

**Architecture.** The generator uses a single event-driven time cursor, matching `psv_generator.py`'s structure rather than VCV/PCV/PRVC's fixed-cycle-count loops, since SIMV's breath sequence is not known in advance. Each breath type's physics is implemented as an inspiration-only function — `_run_mandatory_vc_inspiration`, `_run_mandatory_pc_inspiration`, `_run_spontaneous_inspiration` — and expiration is handled by one generic passive-decay function, `_advance_passive`, called identically regardless of which breath type preceded it. This is what makes compartment and auto-PEEP state carry forward seamlessly across breath-type transitions: the same `V_comps` array is threaded through the entire simulation, and expiration physics does not need to know or care whether the breath that just ended was mandatory or spontaneous. The synchronization-window state machine from CR0019 is implemented as the main loop's breath-type selector, classifying each patient-effort attempt as spontaneous, synchronized-mandatory, or (on window timeout) time-triggered-mandatory. Per the architecture question CR0019 raised but did not resolve, the generator inlines its own physics helpers rather than triggering the deferred `generator/lung_physics.py` refactor, matching the existing project pattern in the other four generators.

**Bugs found and fixed during initial implementation.** Three bugs surfaced while getting the generator's own smoke test to pass:

1. **Control-loop inversion in PC mode.** The ETT Rohrer pressure drop was being added on top of the servo-clamped ventilator pressure for both PC-mandatory and spontaneous breaths, inflating driving pressure roughly 2.5× (39 cmH₂O measured against a set 15 cmH₂O). Pressure-targeted breaths are clamped to `P_vent` directly; the Rohrer-drop reconstruction is only correct for flow-prescribed VC breaths, where pressure genuinely is the dependent variable. This is the same class of bug the project's own history flags as previously caught in `psv_generator.py`.
2. **Scenario ID collisions from incomplete parameter encoding.** `_make_scenario_id` omitted `effort_duration_s` and `pmus_cv`, so scenarios differing only in those two dimensions collided on an identical ID (8 unique IDs out of 12 generated in a smoke-test dataset slice). Matches the scenario-ID completeness bug class already on record for PSV and PRVC.
3. **Mode-ordering bug in `generate_dataset()`.** `mandatory_mode` was the outermost loop in the dataset sweep, so a capped or thinned slice could exhaust its scenario budget entirely on VC scenarios before ever reaching PC. Fixed by nesting mode inside the shared-parameter combination loop.

All three were caught and fixed before the generator's own smoke test (17/17 checks) and an extended ad-hoc validation pass — all seven condition tiers in both mandatory sub-modes, edge cases (extreme mandatory rates, tachypneic effort, wide/narrow windows), endotracheal tube complications, and a 60-scenario dataset slice — were confirmed clean.

**Test suite.** `tests/test_simv_generator.py` (1,000 lines, 154 tests across thirteen classes: `TestInterfaceContract`, `TestThresholdConstants`, `TestPhysiologicalPlausibility`, `TestSynchronizationWindow`, `TestMandatoryBreathPhysics`, `TestSpontaneousBreathPhysics`, `TestDyssynchrony`, `TestMultiCompartmentMechanics`, `TestETTComplications`, `TestPhysiologicalDirections`, `TestValidityFilter`, `TestDatasetGeneration`, `TestParameterGrid`) was built modeled on the VCV/PCV/PSV/PRVC test files, with `TestSynchronizationWindow` covering the mode-defining mechanism no sibling test file needs: exactly one mandatory breath per macro-cycle regardless of effort rate, correct synchronized/time-triggered classification, and mandatory-breath start intervals never exceeding `T_mand`.

**Two further bugs found via the new test suite**, neither caught by the generator's own smoke test since both only manifest across a full scenario's end state or a large-tidal-volume edge case rather than a single representative breath:

4. **Missing final expiration.** The main loop exits immediately after delivering the `n_cycles`-th mandatory breath's inspiration, without ever running that last breath's expiration. `auto_peep_cmH2O` was therefore being computed from a full end-inspiratory volume rather than a genuine end-expiratory one — Normal was reporting roughly 10 cmH₂O of "auto-PEEP" that was really just the last breath's unexhaled tidal volume. Fixed by running one natural mandatory expiratory duration of passive decay after the loop exits, before final metrics are assembled.
5. **Unphysical negative airway pressure during passive exhalation.** After large spontaneous breaths on compliant lungs — legitimate patient-driven volumes exceeding what pressure support alone would suggest, consistent with expected PSV over-assistance behavior — the ETT Rohrer-drop reconstruction on the resulting large expiratory flow produced airway pressure dips to −20 to −29 cmH₂O. Real ventilators regulate the expiratory limb to PEEP, so the airway opening cannot swing that far below set PEEP. Fixed with a physiological floor (PEEP − 5 cmH₂O) on the passive-expiration pressure calculation.

Both fixes were verified against the full 154-test suite, the generator's own 17-check smoke test, and a re-run of the extended ad-hoc validation pass, with no regressions in any of the three.

---

## Proposed Change

The SIMV generator implementation is complete. No further changes to the generator are proposed at this stage beyond what CR0021 (dataset generation) surfaces. `python -m pyflakes` reports no warnings on either `generator/simv_generator.py` or `tests/test_simv_generator.py`.

---

## Acceptance Criteria

- `generator/simv_generator.py` implements the correct hybrid control loop: mandatory breaths reuse VCV/PCV inspiration physics exactly, spontaneous breaths reuse PSV inspiration physics exactly, and expiration is handled by one generic function regardless of which breath type preceded it
- The synchronization window correctly classifies every mandatory breath as `synchronized` or `time_triggered`, and exactly one mandatory breath is delivered per macro-cycle regardless of patient effort rate (no breath stacking)
- Pressure-targeted breaths (PC-mandatory, spontaneous) report airway pressure as the servo-clamped `P_vent`, not `P_vent` plus a reconstructed ETT drop
- `auto_peep_cmH2O` reflects genuine end-expiratory state for every scenario, including the last mandatory breath simulated
- Passive-expiration airway pressure never drops more than 5 cmH₂O below set PEEP
- All scenario IDs in a dataset sweep are unique across every swept parameter, including `effort_duration_s` and `pmus_cv`
- `generate_dataset()` samples both mandatory sub-modes even under a capped or thinned scenario budget
- All 154 unit tests in `tests/test_simv_generator.py` pass, and the generator's own 17-check smoke test (`python generator/simv_generator.py`) passes
- `python -m pyflakes` reports no warnings on either file

---

## Files Likely to Be Touched

- **Created:** `generator/simv_generator.py` — SIMV waveform generator implementing the hybrid control loop, the synchronization-window state machine, per-breath-type inspiration physics, the generic passive-expiration function, derived metrics, the validity filter, scenario ID encoding, and the dataset sweep function
- **Created:** `tests/test_simv_generator.py` — 154 unit tests across thirteen classes covering interface contract, threshold constants, physiological plausibility, the synchronization window, mandatory- and spontaneous-breath physics, dyssynchrony labeling, multi-compartment mechanics, ETT complications, physiological direction checks, the validity filter, dataset generation, and parameter grid completeness
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the five bugs found and fixed during implementation and testing, and the extended validation pass results

---

## Status

**Complete**

`generator/simv_generator.py` is implemented and passing all 154 unit tests plus its own 17-check smoke test. Five bugs were found and fixed during implementation and test-suite development, all verified against the full test suite with no regressions. The generator is ready for the dataset generation phase (CR0021).
