# CR0025: Literature-Grounded Parameter Audit Across `conditions.py`

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-09-24
**Related:** CR0024 (Mode-Stratified Parameter Architecture)

---

## Problem

`conditions.py` had grown to roughly 150 parameter values across nine condition tiers without a systematic check against primary literature. Some values had been set by early-stage estimation, inherited from earlier phases of the project, or set for physiologically plausible but not well-documented reasons. A comprehensive audit was needed to confirm which values were literature-grounded, which were reasonable interpolations, and which were outright wrong.

---

## Current State

A full literature-grounded audit was completed across all ~150 parameter values in `conditions.py`. The audit surfaced several concrete errors and produced targeted fixes:

- **COPD compliance:** 100 → 60 mL/cmH₂O (Arnal 2018)
- **Severe ARDS compliance:** 18 → 25 mL/cmH₂O — the prior value made plateau pressure exceed the pressure ceiling under achievable parameter combinations, an internally impossible state
- **Severe ARDS resistance:** 16 → 14 cmH₂O/L/s
- **Driving-pressure mortality threshold:** 20 → 15 cmH₂O (Amato 2015)
- **ARDSNet plateau-pressure check:** added as an independent ≤30 cmH₂O filter, applied separately from the existing 50 cmH₂O barotrauma filter — the two thresholds serve different clinical purposes and should not be conflated into one check
- **ETT Rohrer K2 (7.5 mm tube):** revised to ~6.0
- **ARDS recruitment slopes:** corrected to monotonically increasing across mild → moderate → severe (Caironi 2015) — the prior ordering was non-monotonic, a physiological error, since recruitment potential should scale with disease severity in ARDS specifically
- **Bronchospasm `effort_duration_s`:** revised from 0.85 to 0.55

A related citation-integrity issue was identified during the audit: the Tokioka 2001 citation used elsewhere in the project (CR0022, SIMV flow-cycle threshold) was found to be factually inverted relative to the paper's actual findings. This CR does not correct that citation directly (it's SIMV-specific and belongs with CR0022), but it's noted here because the audit process — checking secondary summaries against actual paper findings rather than trusting inherited citations — is what surfaced it, and the same discipline should apply going forward to any parameter carrying a literature citation.

---

## Proposed Change

Adopt literature-first parameter validation as a standing practice for `conditions.py`, not a one-time cleanup:

- Every parameter value grounded in primary literature carries an inline citation
- Every interpolated or estimated value is explicitly flagged with an `ASSUMPTION` tag rather than presented as sourced
- Citations are verified against actual paper findings at the point of use, not trusted from secondary summaries or carried over from an earlier draft without re-checking

---

## Acceptance Criteria

- All six parameter changes listed above are applied in `generator/conditions.py` and reflected in any dependent generator constants
- Severe ARDS no longer produces the plateau-exceeds-ceiling impossibility under its corrected compliance value
- ARDS recruitment slopes are monotonically increasing (mild < moderate < severe) and this is covered by a regression test
- The ARDSNet Pplat ≤30 cmH₂O check is a distinct, independently testable filter from the 50 cmH₂O barotrauma filter
- Every literature-cited parameter in `conditions.py` has a verifiable citation; every unsourced value is flagged `ASSUMPTION`

---

## Files Likely to Be Touched

- **Update:** `generator/conditions.py` — all six parameter value changes, `ASSUMPTION` flagging pass
- **Update:** validity-filter logic wherever the Pplat and barotrauma checks are implemented (shared pattern across vcv/pcv/psv/prvc/simv generators) — add the independent ARDSNet Pplat ≤30 cmH₂O check
- **Update:** `tests/test_*_generator.py` (all five) — regression tests for monotonic ARDS recruitment slopes and the Severe ARDS compliance fix
- **Update:** `EXPERIMENT_LOG.md` — record the audit process, the six changes, and the Tokioka citation-integrity finding

---

## Status

**Complete**

All six parameter changes are applied. The Tokioka 2001 citation correction is tracked separately against CR0022.
