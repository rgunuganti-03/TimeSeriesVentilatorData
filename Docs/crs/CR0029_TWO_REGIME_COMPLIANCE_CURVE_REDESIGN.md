# CR0029: Two-Regime Compliance Curve Redesign

**Status:** Complete (with a flagged follow-up)
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0028 (V_ref Anchoring Bug Fix)

---

## Problem

`_compliance_nonlinear`'s `stress_index < 1.0` branch used an unbounded power-law formula. This produced a confirmed 782 mL/breath runaway in `simv_generator.py` — a clearly non-physiological failure mode, since no real breath under the tested parameters should deliver anywhere near that volume. The formula needed to be bounded without losing the recruiting behavior it was originally designed to model.

---

## Current State

The `stress_index < 1.0` branch was redesigned from the unbounded power-law into a two-regime formula:

- **Below `V_turnover`:** the original recruiting formula, unchanged
- **Above `V_turnover`:** an independent declining power-law, bounding delivered volume rather than letting it run away

`V_turnover_ratio = 1.4` and `stress_index_decline = 15.0` are shared constants across `vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`, `simv_generator.py`, and `prvc_generator.py`. These are duplicated per file — this project is not extracting a shared physics module, so each generator carries its own independent copy of the formula and constants.

**Validation scope, by generator** — this is a deliberate part of the record, not an oversight, since not every generator received the same depth of testing:

- **`simv_generator.py`:** independently stress-tested end-to-end — this is the original runaway reproduction case, and the fix is confirmed to resolve it
- **`pcv_generator.py`:** independently stress-tested end-to-end — a prior bell-curve design candidate was found to crush compliance on the recruiting side for ordinary partial-fill breaths, and the two-regime formula was confirmed not to have this problem
- **`vcv_generator.py`:** structurally near-immune to ever needing the declining regime. VCV guarantees exact delivery of `vt_target`, which now equals `V_ref` (per CR0028's fix), so `V/V_ref` stays ~1.0 for any real breath — the declining regime is effectively unreachable in normal operation
- **`prvc_generator.py`:** structurally near-immune for a different reason — its outer adaptive loop converges delivered VT toward `vt_target_ml`, and showed zero Regime 2 activation across every fixture tested, including a deliberately aggressive one
- **`psv_generator.py`:** the one file with confirmed real exposure to the declining regime under ordinary (not edge-case) use. An aggressive stress fixture (PS = 15 cmH₂O, `pmus_peak` = 15 cmH₂O) crossed into the declining regime repeatedly, and the mechanism worked correctly — delivered volume was bounded below what the fully unbounded original formula would have produced. However, the specific `1.4`/`15.0` tuning was borrowed directly from `simv_generator.py`'s calibration, not independently re-derived for PSV's event-driven, flow-cycled dynamics

---

## Proposed Change

Ship the two-regime formula as-is across all five generators, with the validation-scope caveat above documented rather than glossed over. Explicitly do not re-derive PSV's own calibration as part of this CR — see the urgency assessment below.

---

## Urgency / Follow-up Assessment

Re-deriving PSV's own `V_turnover_ratio`/`stress_index_decline` calibration is assessed as **low urgency, low physiological-accuracy payoff**. Neither constant is clinically sourced in the first place — both were tuned against this project's own bug-reproduction case (the SIMV runaway), not against real pressure-volume curve data. Re-deriving PSV's calibration independently would only buy internal consistency with SIMV's rigor, not genuine clinical accuracy. This is worth revisiting only as part of a dedicated future effort to source real overdistension turnover data for all five generators at once — not in isolation for PSV.

---

## Acceptance Criteria

- No unbounded runaway is reproducible in `simv_generator.py` — a regression test locks in the original 782 mL/breath failure case and confirms the corrected, bounded output
- `pcv_generator.py`'s recruiting-side compliance is no longer crushed for ordinary partial-fill breaths — regression test comparing against the prior bell-curve candidate's behavior
- `psv_generator.py`'s declining-regime activation under the aggressive stress fixture (PS = 15, `pmus_peak` = 15) produces bounded delivered volume, confirmed by regression test
- The validation-scope breakdown above (tested end-to-end vs. structurally immune vs. borrowed calibration) is recorded in `EXPERIMENT_LOG.md`, not just implied by which generators have regression tests
- This CR does not assume or depend on a shared `lung_physics.py` module — the fix is applied as five separate, duplicated implementations by design

---

## Files Likely to Be Touched

- **Update:** `generator/vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`, `simv_generator.py`, `prvc_generator.py` — `_compliance_nonlinear`'s `stress_index < 1.0` branch, two-regime formula with `V_turnover_ratio = 1.4` and `stress_index_decline = 15.0`
- **Update:** `tests/test_simv_generator.py` — regression test for the original 782 mL runaway case
- **Update:** `tests/test_pcv_generator.py` — regression test for recruiting-side compliance on partial-fill breaths
- **Update:** `tests/test_psv_generator.py` — regression test for the aggressive stress fixture's bounded output
- **Update:** `EXPERIMENT_LOG.md` — full validation-scope record, and the note that PSV's calibration re-derivation is deferred

---

## Status

**Complete**, with PSV-specific calibration re-derivation explicitly flagged as deferred future work (not scheduled).
