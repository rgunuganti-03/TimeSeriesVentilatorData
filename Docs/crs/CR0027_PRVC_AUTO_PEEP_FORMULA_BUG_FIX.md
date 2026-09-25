# CR0027: PRVC Auto-PEEP Formula Bug Fix

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0016 (PRVC Generator Implementation), CR0026 (PSV Auto-PEEP Measurement Methodology)

---

## Problem

PRVC's parallel-compartment auto-PEEP calculation used `np.mean` to combine per-compartment auto-PEEP contributions, which does not match the convention used in VCV and PCV's parallel-compartment formulas (`.sum()`). This produced auto-PEEP values that understated true auto-PEEP in multi-compartment PRVC scenarios, since averaging across compartments dilutes the contribution of any single compartment developing significant gas trapping, where summing correctly reflects the combined physiological effect.

---

## Current State

The bug was identified during cross-engine review of auto-PEEP convergence patterns (see CR0026) — PRVC's auto-PEEP values were inconsistent with the VCV/PCV convention for otherwise comparable multi-compartment scenarios.

---

## Proposed Change

Replace `np.mean` with `.sum()` in the parallel-compartment auto-PEEP formula in `prvc_generator.py`, matching the convention already used in `vcv_generator.py` and `pcv_generator.py`.

---

## Acceptance Criteria

- PRVC auto-PEEP values for multi-compartment scenarios match the `.sum()`-based convention used in VCV and PCV
- A regression test exists comparing PRVC auto-PEEP output before and after the fix for a representative multi-compartment scenario (e.g. COPD or Bronchospasm), confirming the corrected value is higher than the prior `np.mean`-based value
- No regression in single-compartment PRVC scenarios, where `.sum()` and `.mean()` are equivalent by construction

---

## Files Likely to Be Touched

- **Update:** `generator/prvc_generator.py` — `np.mean` → `.sum()` in the parallel-compartment auto-PEEP formula
- **Update:** `tests/test_prvc_generator.py` — regression test for the corrected formula
- **Update:** `EXPERIMENT_LOG.md` — record the bug, root cause, and fix

---

## Status

**Complete**
