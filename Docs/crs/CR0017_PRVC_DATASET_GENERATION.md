# CR0017: PRVC Dataset Generation

**Status:** Complete
**Source:** Riya Gunuganti, drafted with Claude
**Date:** 2026-07-28
**Related:** CR0015 (PRVC Control Loop Documentation), CR0016 (PRVC Generator Implementation), CR0018 (PRVC Parameter Grid Definition)

---

## Problem

With the generator implemented and passing all unit tests (CR0016), PRVC required a batch script to actually produce the labeled dataset across all seven condition tiers, following the thinned-grid, no-HDF5, CSV-manifest-plus-JSON-log pattern already established for VCV, PCV, and PSV. Unlike those three modes, a PRVC dataset entry is a multi-breath sequence rather than a single-breath snapshot (per CR0015), so the manifest needed to carry per-scenario convergence and ceiling-limited labels, not just the usual scalar clinical metrics.

---

## Current State

The full thinned dataset generation run completed successfully across all seven condition tiers with no crashes.

**Per-tier results:**

| Condition | Total | Valid | Valid % | Converged % | Ceiling-limited % | Time (s) |
|---|---|---|---|---|---|---|
| Normal | 14,400 | 14,396 | 100.0% | 67.9% | 1.1% | 1,599.7 |
| Mild ARDS | 6,912 | 6,860 | 99.2% | 72.6% | 7.6% | 993.7 |
| Moderate ARDS | 6,912 | 6,696 | 96.9% | 75.2% | 15.5% | 984.8 |
| Severe ARDS | 8,640 | 7,008 | 81.1% | 65.9% | 33.6% | 1,233.2 |
| COPD | 14,400 | 14,376 | 99.8% | 70.9% | 0.4% | 5,244.5 |
| Bronchospasm | 13,824 | 13,304 | 96.2% | 80.6% | 5.7% | 4,075.1 |
| Pneumonia | 13,824 | 13,808 | 99.9% | 69.4% | 4.8% | 2,373.4 |

**Overall:** 78,912 total scenarios, 76,448 valid (96.9%), 2,464 invalid (3.1%), 56,635 converged (71.8%), 6,161 ceiling-limited (7.8%). Total runtime 16,507.4s (275.1 min), with COPD and Bronchospasm — the two tiers using 25 cycles per scenario instead of 12 — together accounting for roughly 57% of total runtime despite being only 36% of total scenarios, confirming the projected 25-cycle cost dominance from CR0018.

The invalidity distribution tracks condition severity as expected: Normal is effectively 0% invalid, climbing to Severe ARDS's 18.9%, which is the barotrauma filter behaving correctly — a working pressure up to 55 cmH2O (PEEP 20 + ceiling 35) exceeds the 50 cmH2O threshold, and Severe ARDS is the tier most often forced to push toward that combination to have any chance at the volume target.

Subsequent review identified that `converged_pct + ceiling_limited_pct` does not sum to 100% in any tier — the remainder (20.4% overall, as high as 31.0% in Normal) represents scenarios that were still adjusting, neither converged nor pinned at the ceiling, when `n_cycles` ran out. This "unresolved" outcome is not currently a distinct manifest column and is largest in the high-compliance tiers (Normal, COPD, Pneumonia), consistent with the `C_threshold = (tolerance × VT_target) / step` relationship from CR0018 — high compliance means the fixed 2 cmH2O adaptation step moves enough volume to risk overshooting the tolerance window on every correction, making it comparatively hard to land two consecutive in-tolerance breaths regardless of `n_cycles`. This is flagged as a follow-up item, not resolved in this CR (see Status).

---

## Proposed Change

Implement `generate_prvc_dataset_thinned.py` as a self-contained batch script following the same structure as `generate_vcv_dataset_thinned.py`, `generate_pcv_dataset_thinned.py`, and `generate_psv_dataset_thinned.py`:

- A thinned parameter grid (576 combinations per mechanics point, a 77.1% reduction from the full 2,520 defined in CR0018), sweeping tidal volume target, respiratory rate, PEEP, I:E ratio, and pressure ceiling, with `adaptation_step_cmH2O` and `vt_tolerance_frac` held fixed per CR0018's uniform-constant decision rather than swept
- The same seven-tier `CONDITION_TIERS` structure used across the codebase, with COPD and Bronchospasm using 25 cycles per scenario (for auto-PEEP and pressure-staircase steady state) against 12 for the remaining five tiers
- A local `_generate_thinned_dataset()` helper producing scenario dicts with metrics only, no waveform arrays — full waveforms including `pressure_trajectory` and `delivered_vt_trajectory` are regeneratable on demand via `generate_breath_cycles(params, seed=seed)`, since the generator has no stochastic elements and identical params always reproduce identical output regardless of seed
- A manifest CSV (`prvc_manifest_thinned.csv`) with one row per scenario carrying all ventilator-side parameters plus PRVC-specific columns with no VCV/PCV/PSV analogue: `pressure_ceiling_cmH2O`, `adaptation_step_cmH2O`, `vt_tolerance_frac`, `test_breath_plateau_cmH2O`, `breaths_to_converge`, `converged`, `ceiling_limited`
- A generation log JSON (`prvc_generation_log.json`) with per-tier and grand totals for scenario count, valid/invalid, converged, and ceiling-limited counts, plus timing

---

## Acceptance Criteria

- The generation run completes without errors across all seven condition tiers and all mechanics pairs
- `prvc_manifest_thinned.csv` contains one row per scenario, with populated metric fields for valid scenarios and blank metric fields for invalid ones
- All scenario IDs in the manifest are unique, including across combinations that differ only in I:E ratio (confirming the CR0016 scenario-ID fix holds under the full production sweep, not just the unit test sample)
- The manifest contains PRVC-specific columns absent from the vcv/pcv/psv manifests: `pressure_ceiling_cmH2O`, `adaptation_step_cmH2O`, `vt_tolerance_frac`, `test_breath_plateau_cmH2O`, `breaths_to_converge`, `converged`, `ceiling_limited`
- Ceiling-limited non-convergence is preserved as `is_valid=True` in the manifest — it is not filtered out or conflated with genuine invalidity, matching CR0015's terminal-state definition
- The per-tier invalidity distribution reflects condition severity, with Severe ARDS carrying the highest invalidity rate (18.9%) driven by barotrauma at the PEEP-ceiling combinations that exceed 50 cmH2O working pressure
- The per-tier ceiling-limited distribution reflects the compliance/severity gradient documented in CR0018 — Severe ARDS highest (33.6%), Normal near-zero (1.1%)
- `prvc_generation_log.json` contains the thinned parameter grid definition, combinations-per-point count, per-tier counts and elapsed times, and grand totals
- Total runtime is dominated by the two 25-cycle tiers (COPD, Bronchospasm), confirming the cycle-count-driven cost projection from CR0018 rather than an unexpected performance regression

---

## Files Likely to Be Touched

- **Created:** `generate_prvc_dataset_thinned.py` — the production batch script sweeping the thinned grid across all seven condition tiers and writing the manifest and generation log to `data/exports/prvc/`
- **Populated:** `data/exports/prvc/` — `prvc_manifest_thinned.csv` (78,912 rows) and `prvc_generation_log.json`, produced by the completed generation run
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the completed run's per-tier results, the runtime breakdown, and the unresolved-outcome finding

---

## Status

**Complete, with one follow-up item identified and not yet resolved.**

The full PRVC thinned dataset has been generated: 78,912 scenarios across seven condition tiers, 96.9% valid, 71.8% converged, 7.8% ceiling-limited, in 275.1 minutes. The manifest and generation log are written to `data/exports/prvc/`.

**Follow-up (not addressed in this CR):** roughly 20.4% of scenarios overall (up to 31.0% in Normal) are neither converged nor ceiling-limited when `n_cycles` runs out — an "unresolved" outcome with no current manifest label. Whether this reflects genuine breath-to-breath pressure oscillation (a real consequence of the uniform `adaptation_step_cmH2O`/`vt_tolerance_frac` choice interacting with high-compliance tiers) or simply an insufficient `n_cycles` budget has not yet been distinguished empirically, and should be investigated — by inspecting `pressure_trajectory` on a sample of unresolved Normal scenarios — before deciding whether the fix is a labeling addition, a `vt_averaging_window` increase, or an `n_cycles` increase for the affected tiers.
