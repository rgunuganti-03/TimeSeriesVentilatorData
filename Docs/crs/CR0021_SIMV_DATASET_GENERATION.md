# CR0021 — SIMV Dataset Generation

**Author:** Riya Gunuganti
**Date:** 2026-07-29
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

With the SIMV generator implemented and validated (CR0020), a thinned dataset generation script was needed to actually produce the SIMV dataset, matching the thinning approach already established for VCV, PCV, PSV, and PRVC rather than sweeping the full `PARAMETER_GRID` combinatorially. SIMV's thinning problem was harder than any sibling engine's: it carries three multiplicative axes of complexity — the mandatory-mode split (VC vs. PC), the full spontaneous-breath / patient-effort dimension set PSV needs, and SIMV's own synchronization-window parameter — stacked on top of each other, where the sibling scripts each only had one or two axes to thin. A naive port of PSV's or PRVC's thinning ratios would either explode the combinatorial count well past a practical overnight run or thin away exactly the dimension (`f_window`) that makes SIMV's data distinctive.

---

## Current State

`generate_simv_dataset_thinned.py` (594 lines) is implemented and validated at reduced scale; the full production run has not yet been launched.

**Thinning approach.** The script is self-contained, matching the `generate_vcv_dataset_thinned.py` / `generate_pcv_dataset_thinned.py` / `generate_psv_dataset_thinned.py` / `generate_prvc_dataset_thinned.py` pattern: the thinned grid is defined directly in the file rather than imported from the generator's own `PARAMETER_GRID`. Because SIMV's mandatory-mode axis changes which settings apply, the grid is split into `THINNED_SHARED_GRID` (respiratory rate, PEEP, I:E ratio, rise time, synchronization window, pressure support, flow-cycle threshold, trigger threshold, and the patient-effort dimensions) plus `THINNED_VC_GRID` and `THINNED_PC_GRID` for the mode-specific settings — a bifurcation none of the four sibling thinned scripts need. Wherever a sibling script already established a value or rationale for a dimension SIMV shares (tidal volume, I:E ratio, flow pattern from VCV/PRVC; pressure support and flow-cycle threshold from PSV; rise time from PSV), the same values and reasoning were reused directly rather than re-derived. `f_window`, SIMV's own signature parameter, was thinned least aggressively of any dimension — three of its four literature-grounded values were kept, the same treatment `prvc_generator`'s thinned script gives `pressure_ceiling`, its own signature parameter — while dimensions with the weakest "must keep multiple values" case even in PSV's own thinned script (trigger threshold, effort rate, effort duration, Pmus CV) were cut to single representative values to offset the extra combinatorial load the mandatory-mode axis adds.

**Runtime tuning.** A first draft of the thinned grid produced 5,832 combinations per mechanics point. A directly timed 60-scenario sample (mixed VC/PC, mixed mandatory rate) measured roughly 82 ms/scenario, projecting a full production run of approximately 17 hours — well past the 8–10 hour overnight precedent VCV's and PCV's own full runs established. PEEP was identified as the dimension with the weakest case for keeping multiple values (its documented effect in every sibling script's own thinning rationale is a vertical baseline shift, not a waveform-shape change) and was cut from a two-value bookend to a single mid-clinical value, bringing the grid to 2,916 combinations per mechanics point — 324 shared × 6 (VC: 3 tidal volumes × 2 flow patterns) plus 324 × 3 (PC: 3 inspiratory pressures). Against the measured per-scenario cost and the roughly 126 total mechanics pairs across all seven condition tiers, this projects to approximately 367,000 total scenarios and 8–9 hours of runtime, back in line with the sibling precedent.

**Mandatory-cycle count.** SIMV's mandatory cycle time (`T_mand = 60/respiratory_rate`, 5–15 s across the 4–12 bpm mandatory-rate range used here) is substantially longer than the sibling engines' typical 2–7.5 s mandatory cycle at their own thinned rate ranges, and each cycle additionally simulates however many spontaneous breaths the effort-rate schedule interleaves. `n_cycles` was set lower than PSV/PRVC's 12/25 convention — 6 mandatory cycles for most conditions, 10 for COPD and Bronchospasm — to keep total simulated time, and therefore per-scenario runtime, comparable to the sibling scripts rather than several times longer.

**Bug found during pipeline validation.** Because a full production run takes hours, the script was validated end-to-end at reduced scale first: a patched-in tiny grid and mechanics slice exercising the full sweep-to-manifest-to-log pipeline. This surfaced a bug in `generator/simv_generator.py` itself rather than in the new script — `_make_scenario_id` never encoded `compliance_ml_per_cmH2O` or `resistance_cmH2O_L_s` at all, so every mechanics pair swept within one condition tier collided on an identical scenario ID. This was invisible to the generator's own smoke test and unit tests, which always call the generator with one fixed mechanics pair at a time, but immediately visible once multiple mechanics pairs were swept within a tier — exactly what this script's design does. This is the same scenario-ID-completeness bug class already on record for PSV and PRVC, now caught for a third time by the first workflow that actually exercised the failure condition. Fixed directly in `generator/simv_generator.py`; the full 154-test suite and the generator's own smoke test were re-run afterward with no regressions.

The reduced-scale pipeline run, after the fix, produced correct output on every check: unique scenario IDs, both mandatory modes and both tested conditions represented, manifest row count matching the JSON log's `grand_total`, and valid-scenario counts agreeing between the manifest and the log.

---

## Proposed Change

Launch the full SIMV thinned-dataset production run via `nohup python -u generate_simv_dataset_thinned.py > simv_thinned.log 2>&1 &`, matching the detached overnight-run pattern PCV's generation used, and monitor it to completion. Once complete, verify the manifest and generation log against the acceptance criteria below before treating the SIMV dataset as ready for downstream use.

---

## Acceptance Criteria

- The generation run completes without errors across all seven condition tiers and all mechanics pairs
- `simv_manifest_thinned.csv` contains one row per scenario, with metrics populated for valid scenarios and empty metric fields for invalid ones
- All scenario IDs in `simv_manifest_thinned.csv` are unique
- Both mandatory sub-modes (VC and PC) are represented within every condition tier, not just the dataset as a whole
- `simv_generation_log.json` is present and contains the thinned grid definitions, combinations-per-point breakdown, per-tier counts and elapsed times, and total runtime
- Total runtime lands within the 8–10 hour range projected from the timed 60-scenario sample; a large deviation would indicate the timing estimate or the grid size was wrong
- COPD and Bronchospasm show measurably higher mean auto-PEEP and higher mean ineffective-trigger fraction than Normal, consistent with the multi-compartment and recruitment-slope behavior already validated in `tests/test_simv_generator.py`

---

## Files Likely to Be Touched

- **Created:** `generate_simv_dataset_thinned.py` — the thinned dataset generation script defining `THINNED_SHARED_GRID`, `THINNED_VC_GRID`, `THINNED_PC_GRID`, and the seven condition tier mechanics grids, sweeping both mandatory sub-modes per mechanics pair, and writing the manifest and generation log
- **Updated:** `generator/simv_generator.py` — `_make_scenario_id` fixed to encode compliance and resistance (bug found during this CR's pipeline validation; see also CR0020)
- **Pending:** `data/exports/simv/` — `simv_manifest_thinned.csv` and `simv_generation_log.json`, to be populated once the full production run (not yet launched) completes
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the runtime-tuning decision (PEEP thinned to a single value), the scenario-ID bug found during pipeline validation, and the production run launch once it happens

---

## Status

**Complete**

`generate_simv_dataset_thinned.py` is implemented and validated end-to-end at reduced scale, with the scenario-ID bug it surfaced fixed and verified against the full test suite. The thinned grid, mechanics tiers, and runtime have been sized against a directly measured per-scenario timing sample to land within the project's established 8–10 hour overnight-run precedent. The full production run has not yet been launched; that is the remaining action before `data/exports/simv/` is populated.
