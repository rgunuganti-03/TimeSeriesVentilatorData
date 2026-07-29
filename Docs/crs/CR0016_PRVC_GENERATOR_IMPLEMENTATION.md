# CR0016: PRVC Generator Implementation

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-07-28
**Related:** CR0015 (PRVC Control Loop Documentation), CR0017, CR0018

---

## Problem

With the control loop defined (CR0015) and the parameter grid defined (CR0018), PRVC required a dedicated generator module implementing the dual-loop control logic: multi-compartment intra-breath physics matching PCV's architecture, a volume-controlled test breath bootstrap, and a breath-to-breath outer loop that adapts working pressure toward a volume target. No prior generator in this project implements a control variable that depends on the outcome of a previous breath, so this required new mechanics beyond what could be copied from `vcv_generator.py`, `pcv_generator.py`, or `psv_generator.py`.

---

## Current State

Three bugs were identified and fixed during smoke testing and unit testing.

First, `pressure_trajectory[0]` was recorded after `P_work` had already been reassigned for breath 2, so breath 1's slot in the trajectory array silently held breath 2's seeded value instead of the test breath's own measured plateau pressure. Fixed by separating "the pressure this breath actually used" from "the pressure being computed for the next breath," and by excluding the volume-controlled test breath from convergence/stability tracking, since it trivially hits its own target by construction and isn't yet under adaptive control.

Second, two of the initial smoke test's own assertions were incorrect, not the generator: the test assumed the pressure staircase climbs monotonically starting from breath 1, when the documented AutoFlow behavior is that breath 2 deliberately undershoots the measured test-breath plateau (seeded at 75% of the measured driving pressure) before climbing from there. Corrected the assertions to check for the actual documented pattern — a high measured plateau on breath 1, a deliberate drop on breath 2, and either a genuine multi-step climb to convergence or a flat line pinned at the ceiling — rather than a naive monotonic increase.

Third, `generate_dataset()` initially did not match the codebase-wide convention used by `vcv_generator.py`, `pcv_generator.py`, and `psv_generator.py` — it returned the entire `generate_breath_cycles()` output under a single `"result"` key instead of the established `scenario_id`/`condition`/`params`/`metrics`/`waveforms`/`is_valid`/`invalid_reason`/`generated_at` structure, with empty waveforms on invalid scenarios. Corrected before any tests were written against the inconsistent structure, avoiding the need to retrofit the interface later.

During test file authoring, two further bugs were found in code that had already passed its own smoke test: `_make_scenario_id()` did not encode `ie_ratio`, so any two scenarios differing only in I:E ratio collided on the same scenario ID — a data-integrity bug that would have silently overwritten roughly one in three scenarios in any real dataset run. Fixed by adding an `IE{nnn}` tag to the ID. Separately, the decelerating-flow shape test was checking peak-flow position across the whole concatenated multi-breath waveform array, which incorrectly included breath 1's flow-prescribed (flat, non-decelerating) test breath in a check meant for pressure-controlled breaths — fixed by isolating a single PC-controlled breath via the fallback (no-test-breath) code path rather than changing the generator.

75 unit tests were written across nine classes and all pass. A subsequent 105-scenario stress sweep across all seven condition tiers, fifteen seeds each, with varying respiratory rate and PEEP, produced zero crashes and zero NaN/Inf values in any output array.

---

## Proposed Change

Implement `generator/prvc_generator.py` with:

- Multi-compartment lung mechanics reusing the exact compartment profiles, recruitment slopes, and condition-tier resistance floors already established in `psv_generator.py` (Normal=1, ARDS tiers=2, COPD=3, Pneumonia=3, Bronchospasm=2 compartments)
- Explicit Euler integration at 100 Hz per compartment for both the flow-prescribed test breath (via an algebraic branch-point pressure solve at each timestep, given the shared airway pressure across parallel compartments) and the pressure-prescribed breaths that follow — no `scipy.integrate` dependency
- A volume-controlled test breath on breath 1, computing plateau pressure via the parallel-compliance equilibrium relationship (`P_plat = PEEP + V_total / sum(C_i)`), documented as a simplification of full pendelluft redistribution physics
- An outer loop that seeds breath 2 at 75% of the measured test-breath driving pressure (matching documented AutoFlow behavior), then steps by a fixed `adaptation_step_cmH2O` toward the volume target for all subsequent breaths, clipped to `[PEEP + 5, PEEP + pressure_ceiling_cmH2O]`
- A two-breath moving average of delivered volume as the outer loop's error signal (not single-breath error), matching documented anti-hunting behavior on real dual-control ventilators — the one physiological refinement with no VCV/PCV/PSV analogue
- Auto-PEEP carry-forward between breaths, reusing the same mechanism already validated in the other three generators
- A validity filter that hard-invalidates only barotrauma (Ppeak > 50 cmH2O) and out-of-range delivered volume on a *converged* breath, while explicitly retaining ceiling-limited non-convergence as a valid, labeled scenario per CR0015's terminal-state definition
- A `generate_dataset()` function matching the codebase-wide scenario dict convention exactly

---

## Acceptance Criteria

- `generate_breath_cycles()` returns a dict containing the four core waveform arrays (`time`, `pressure`, `flow`, `volume`) as NumPy arrays of equal length, the two per-breath trajectory arrays (`pressure_trajectory`, `delivered_vt_trajectory`) of length `n_cycles`, all numeric metric keys, the `converged`/`ceiling_limited` booleans, `breaths_to_converge` as an int or `None`, and both validity keys
- Breath 1 delivers approximately the target tidal volume by construction (flow-prescribed), and its recorded plateau pressure is diagnostic of the scenario's true compliance — a stiffer lung produces a measurably higher test-breath plateau than a more compliant one, all else equal
- Breath 2's working pressure is measurably below breath 1's plateau pressure whenever the ceiling doesn't force an even lower clip, consistent with the documented AutoFlow undershoot rule
- The working pressure never exceeds `PEEP + pressure_ceiling_cmH2O` or drops below `PEEP + 5` from breath 2 onward
- A tight-ceiling Severe ARDS scenario produces `converged=False`, `ceiling_limited=True`, `breaths_to_converge=None`, and remains `is_valid=True`
- A representative Moderate ARDS scenario shows a genuine multi-step pressure climb (not a single correction) between the AutoFlow-seeded breath 2 and eventual convergence, and delivered volume error decreases monotonically toward convergence
- COPD and Bronchospasm scenarios develop measurable auto-PEEP (> 0.3 cmH2O) over a 20+ cycle sequence; Normal develops negligible auto-PEEP (< 1.0 cmH2O) over the same length
- Scenario IDs are unique across the full parameter grid, including combinations that differ only in I:E ratio
- `generate_dataset()` returns scenario dicts matching the `scenario_id`/`condition`/`params`/`metrics`/`waveforms`/`is_valid`/`invalid_reason`/`generated_at` structure used by `vcv_generator.py`, `pcv_generator.py`, and `psv_generator.py`, with empty `waveforms` on invalid or errored scenarios
- All 75 unit tests in `tests/test_prvc_generator.py` pass, across nine test classes: interface contract, physiological plausibility, test breath bootstrap, outer loop control, PRVC waveform shape, convergence and ceiling-limited terminal states, multi-compartment mechanics, validity filter, and dataset generation

---

## Files Likely to Be Touched

- **Created:** `generator/prvc_generator.py` — the PRVC waveform generator implementing the dual-loop control architecture, the VC test breath bootstrap, the outer-loop adaptive algorithm with moving-average damping, multi-compartment intra-breath physics, and the standard `generate_dataset()` interface
- **Created:** `tests/test_prvc_generator.py` — 75 unit tests across nine classes, including tests with no VCV/PCV/PSV precedent (test breath bootstrap, outer loop control, convergence vs. ceiling-limited terminal states)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the three implementation bugs, the two test-writing-stage bugs (scenario ID collision, decelerating-flow test flaw), and the stress test results

---

## Status

**Complete**

`generator/prvc_generator.py` is implemented and passing all 75 unit tests in `tests/test_prvc_generator.py`. A 105-scenario stress sweep across all seven condition tiers produced zero crashes and zero invalid numeric output. The generator is ready for dataset generation (CR0017) and dashboard integration.
