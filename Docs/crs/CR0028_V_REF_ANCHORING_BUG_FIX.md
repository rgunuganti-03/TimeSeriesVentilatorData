# CR0028: Reference-Volume (V_ref) Anchoring Bug Fix

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0029 (Two-Regime Compliance Curve Redesign)

---

## Problem

A compartment's reference volume (`V_ref`) — the value against which delivered volume is compared to determine compliance behavior — must be anchored to a breath's actual achievable or guaranteed full volume. Several generators instead used a silently reduced "mid-fill" convention for `V_ref`, unrelated to the breath's real target volume. This is a correctness bug independent of any compliance-curve behavior: an incorrectly low `V_ref` misrepresents how "full" a given delivered volume actually is, which distorts any downstream calculation that depends on `V/V_ref` — most consequentially, it can push a flow-prescribed breath into the wrong compliance regime by construction (see CR0029), but the anchoring error itself is a bug regardless of which compliance formula is in use.

This bug was found while building CR0029, but is documented and fixed separately since it's a distinct defect with its own acceptance criteria.

---

## Current State

Confirmed bugs, by generator:

- **`vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`:** a silent `* 0.5` reduction at four separate call sites, halving `V_ref` relative to the breath's actual guaranteed full volume
- **`prvc_generator.py`:** two independent instances —
  - `_run_vc_test_breath` used a `* 0.6` reduction
  - `_run_pc_breath` used a flat/weight-scaled 50 mL offset, fully decoupled from `vt_target_ml` — confirmed as a real bug via a converged breath reaching ~405 mL of delivered volume while still being compared against a fixed 50 mL reference

---

## Proposed Change

Anchor `V_ref` to the breath's actual achievable/guaranteed full volume in all four affected generators, removing the halving and offset shortcuts:

- `vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`: remove the `* 0.5` reduction at all four call sites; `V_ref` reflects the breath's real target/guaranteed volume directly
- `prvc_generator.py`: replace `_run_vc_test_breath`'s `* 0.6` reduction with a direct anchor to the test breath's real target volume; replace `_run_pc_breath`'s flat 50 mL offset with `vt_target_ml` (or the equivalent guaranteed full-fill volume for that breath)

---

## Acceptance Criteria

- `V_ref` in all four generators matches the breath's real guaranteed/target volume — no residual `* 0.5`, `* 0.6`, or flat-offset shortcuts remain at any of the identified call sites
- The PRVC regression case (a converged breath reaching ~405 mL against what was previously a 50 mL reference) is captured as a named regression test with the correct expected `V_ref`
- No unintended change in delivered-volume output for scenarios that were previously working correctly — this is a reference-value correction, not a change to the physics governing delivered volume itself
- Cross-checked against CR0029: with `V_ref` correctly anchored, ordinary partial-fill breaths in vcv, pcv, and psv no longer risk being pushed into the declining compliance regime purely due to a mis-anchored reference

---

## Files Likely to Be Touched

- **Update:** `generator/vcv_generator.py`, `pcv_generator.py`, `psv_generator.py` — remove `* 0.5` at all four call sites
- **Update:** `generator/prvc_generator.py` — fix `_run_vc_test_breath`'s `* 0.6` reduction and `_run_pc_breath`'s flat 50 mL offset
- **Update:** `tests/test_vcv_generator.py`, `test_pcv_generator.py`, `test_psv_generator.py`, `test_prvc_generator.py` — regression tests confirming `V_ref` anchoring, including the ~405 mL PRVC case
- **Update:** `EXPERIMENT_LOG.md` — record the bug, its discovery during CR0029's development, and the fix across all four generators

---

## Status

**Complete**
