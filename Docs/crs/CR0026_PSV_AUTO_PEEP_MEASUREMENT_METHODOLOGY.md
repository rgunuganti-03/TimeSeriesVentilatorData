# CR0026: PSV Auto-PEEP Measurement Methodology (Multi-Seed Averaging)

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0024 (Mode-Stratified Parameter Architecture)

---

## Problem

Auto-PEEP convergence was being evaluated the same way across all five generators: run enough cycles and check whether the value settles. That works for PCV and VCV, which converge cleanly by cycle 5–6, and for PRVC and SIMV, which have well-defined instrumentation points (a formula fix in PRVC's case, see CR0027; the macro-cycle-end checkpoint for SIMV). PSV does not behave this way — patient-driven trigger timing with no mandatory-breath anchor means a single seed does not converge to one stable auto-PEEP value, it produces a genuine breath-to-breath distribution. Evaluating PSV auto-PEEP with a single-seed point comparison, as the other modes allow, was producing inconsistent and non-reproducible readings.

---

## Current State

PSV auto-PEEP is confirmed as a genuine breath-to-breath distribution rather than a converging point value, a direct consequence of patient-driven timing with no mandatory-breath anchor to lock cycles to. Multi-seed averaging (30 seeds) was adopted as the measurement methodology and produces stable, reproducible mean auto-PEEP values suitable for both parameter tuning (e.g. the PEEP table in CR0024) and dataset labeling.

---

## Proposed Change

Formalize 30-seed mean auto-PEEP as the standard measurement convention specifically for PSV, distinct from the single-seed or cycle-N-checkpoint conventions used by the other four modes:

- PSV auto-PEEP tests and any downstream derivation (e.g. PEEP presets, dataset labeling) use a 30-seed mean rather than a single-seed value
- This convention is documented as PSV-specific, so future work doesn't assume it applies uniformly to the other four generators or forget to apply it to PSV

---

## Acceptance Criteria

- `tests/test_psv_generator.py` auto-PEEP assertions use multi-seed (30-seed) mean comparison, not single-seed point comparison
- The convergence-pattern differences across all five modes are documented in one place (PCV/VCV: cycle 5–6; PRVC: post-bug-fix formula, see CR0027; SIMV: macro-cycle-end checkpoint; PSV: 30-seed mean)
- Any parameter derived from PSV's auto-PEEP output (e.g. the PSV PEEP values in CR0024) is traceable to the 30-seed methodology, not a single stochastic run

---

## Files Likely to Be Touched

- **Update:** `tests/test_psv_generator.py` — multi-seed mean auto-PEEP assertions
- **Update:** `EXPERIMENT_LOG.md` — document the auto-PEEP convergence pattern differences across all five modes and why PSV needs a different methodology

---

## Status

**Complete**
