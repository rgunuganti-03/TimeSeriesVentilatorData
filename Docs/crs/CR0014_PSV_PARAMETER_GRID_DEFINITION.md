# CR0014 — PSV Parameter Grid Definition

**Author:** Riya Gunuganti
**Date:** 2026-05-20
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

PSV requires defining two distinct parameter grids rather than one, because the parameter space is fundamentally two-dimensional in a way that VCV and PCV are not. In VCV and PCV, the ventilator controls one variable (flow or pressure) and the patient is modeled as a fixed passive load defined by compliance and resistance. The only parameters to grid are the ventilator settings and the mechanics. In PSV, the patient is an active co-driver of gas delivery, and the patient's effort amplitude (Pmus_peak), effort rate, effort duration, and effort variability are all clinically significant independent variables that interact with the ventilator settings to produce the waveform. Failing to systematically vary patient effort parameters would produce a dataset where all PSV waveforms are generated with a single representative patient effort — missing the large portion of the clinical state space that is driven by differences in patient drive, not ventilator settings.

There was also a new class of parameter with no VCV or PCV analog: the expiratory trigger sensitivity (flow cycle threshold). This parameter determines when the ventilator terminates inspiration and has dramatically different optimal values depending on the patient's respiratory mechanics. In ARDS with short time constants, a standard 25% ETS causes premature cycling. In COPD with long time constants, the same 25% ETS causes delayed cycling. The parameter grid must span the range from low ETS (0.10) to high ETS (0.40 and beyond) to capture these asynchrony patterns, and the condition-specific presets must use calibrated ETS values rather than a single global default.

Additionally, without formal documentation of what tidal volumes the parameter combinations are expected to produce before generation, it would be impossible to identify calibration errors — scenarios where the combined driving pressure (PS + Pmus) and compliance are inconsistent with lung-protective ventilation targets for that condition tier.

---

## Current State

The PSV parameter grid has been fully defined, implemented, and the thinned dataset generation has been completed. The following work has been completed.

**The full PARAMETER_GRID.** The complete clinical parameter space implemented in `generator/psv_generator.py` as the PARAMETER_GRID constant covers the following dimensions.

Ventilator settings: pressure_support_cmH2O at [5, 8, 12, 16, 20]; peep_cmH2O at [0, 4, 8, 12, 16]; rise_time_s at [0.0, 0.1, 0.2, 0.4]; flow_cycle_threshold at [0.10, 0.25, 0.40]; trigger_threshold_cmH2O at [0.5, 1.5, 3.0].

Patient parameters: pmus_peak_cmH2O at [5, 8, 12, 16, 20]; effort_rate_per_min at [12, 16, 20, 25, 30]; effort_duration_s at [0.5, 0.7, 0.9, 1.1]; pmus_cv at [0.15, 0.25, 0.35].

The full Cartesian product of ventilator settings produces 5 × 5 × 4 × 3 × 3 = 900 combinations per mechanics point. The full Cartesian product of patient parameters produces 5 × 5 × 4 × 3 = 300 combinations. The combined space is 270,000 parameter combinations per mechanics point — far too large to sweep exhaustively, which is why the thinned grid was used for dataset generation.

**The thinned DATASET_GRID.** The DATASET_GRID constant in `generator/psv_generator.py` defines the reduced parameter space used for systematic dataset generation: pressure_support_cmH2O at [5, 10, 15, 20]; peep_cmH2O at [0, 5, 10, 15]; flow_cycle_threshold at [0.10, 0.25, 0.40]; trigger_threshold_cmH2O at [0.5, 2.0]; rise_time_s at [0.0, 0.2]; pmus_peak_cmH2O at [5, 10, 15, 20]; effort_rate_per_min at [12, 20, 30]; effort_duration_s at [0.6, 0.9]; pmus_cv at [0.15, 0.25]. This produces 4 × 4 × 3 × 2 × 2 = 192 ventilator combinations and 4 × 3 × 2 × 2 = 48 patient combinations per mechanics point, for 9,216 total parameter combinations per mechanics point before the validity filter.

**Step size rationale.** The flow_cycle_threshold step sizes were chosen to span the three clinically meaningful regimes: 0.10 produces delayed cycling behavior (inspiration extends past the patient's neural Ti), 0.25 is the standard factory default shared by Hamilton, Dräger, PB840, and Servo-u, and 0.40 produces premature cycling behavior. These three values are sufficient to teach the full ETS effect without generating redundant intermediate values.

Pressure support step sizes of 5 cmH₂O (in the full grid) and 5 cmH₂O (in the thinned grid) were chosen so that adjacent PS levels produce visibly different tidal volumes. The effect of PS on Vt scales with compliance: at C = 70 mL/cmH₂O (Normal), a 5 cmH₂O change in PS changes Vt by approximately 5 × 0.53 × 70 = 186 mL — a difference visible on any scale. At C = 18 mL/cmH₂O (Severe ARDS), the same step changes Vt by approximately 5 × 0.55 × 25 = 69 mL — still a meaningful difference.

The pmus_peak step of 5 cmH₂O (full grid) and 5 cmH₂O (thinned grid) was chosen on the same basis. Because PS and Pmus_peak add together in the driving term, their combined effect on Vt is symmetric: a 5 cmH₂O increase in Pmus_peak has the same volume effect as a 5 cmH₂O increase in PS at the same mechanics. The step sizes were matched deliberately to make this symmetry visible in the dataset.

**Condition-specific ETS presets.** The flow_cycle_threshold in the DATASET_GRID covers the full 0.10–0.40 range, but the condition presets in conditions.py use calibrated ETS values that sit outside or at the edges of the dataset grid for two conditions: COPD uses ETS = 0.55 and bronchospasm uses ETS = 0.65. These values exceed the DATASET_GRID maximum of 0.40 because the time constants of those conditions (τ ≈ 2.2 s for COPD, τ ≈ 2.45 s for bronchospasm) are so long that even 0.40 produces delayed cycling. The condition presets are used for the educational dashboard display, while the dataset grid covers the systematic parameter space for AiRA training.

**The mechanics grid per condition tier.** The patient mechanics grid follows the same structure as VCV and PCV: compliance and resistance are varied across a tier-specific range with step sizes chosen so adjacent points produce visibly different fill fractions and auto-PEEP values.

Normal uses compliance 50–100 mL/cmH₂O in steps of 10 and resistance 2–15 cmH₂O/L/s in steps of 3, producing 18 mechanics pairs. Mild ARDS uses compliance 35–50 in steps of 5 and resistance 5–15 in steps of 5, producing 12 pairs. Moderate ARDS uses compliance 25–40 in steps of 5 and resistance 8–18 in steps of 5, producing 9 pairs. Severe ARDS uses compliance 15–25 in steps of 5 and resistance 10–20 in steps of 5, producing 6 pairs. COPD uses compliance 60–120 in steps of 20 and resistance 15–35 in steps of 10, producing 12 pairs. Bronchospasm uses compliance 50–90 in steps of 20 and resistance 25–50 in steps of 10, producing 9 pairs. Pneumonia uses compliance 35–55 in steps of 10 and resistance 6–16 in steps of 5, producing 9 pairs.

**The validity filter.** Five thresholds define invalid scenarios. PPEAK_MAX_CMHH2O: peak airway pressure exceeds 50 cmH₂O — note that in PSV the peak pressure equals PEEP + PS, so this filter catches combinations with very high PEEP plus high PS. PS_MAX_CMHH2O: pressure support above the clinical maximum; prevents scenarios where the PS level alone constitutes excessive driving pressure. VT_MIN_ML: delivered tidal volume below 3 mL/kg IBW (210 mL for 70 kg IBW) — catches very-low-compliance plus very-low-PS scenarios where the combined driving is insufficient. VT_MAX_ML: delivered tidal volume above 12 mL/kg IBW (840 mL) — catches high-compliance plus high combined driving (PS + Pmus) scenarios that produce dangerous overdistension. FILL_FRACTION_MIN: fill fraction below the minimum threshold — catches scenarios where the lung barely fills before the flow-cycle criterion fires, producing waveforms too short to be clinically interpretable.

**Tidal volume calibration.** The relationship Vt ≈ fill_fraction × (PS + Pmus_mean) × C_lung_rec was used to calibrate the condition presets so that each tier produces tidal volumes in the physiologically appropriate range for its compliance and clinical scenario. The fill fraction for each condition was read from the simulator output, then the required combined driving (PS + Pmus) was back-calculated as Vt_target / (fill_fraction × C_lung_rec). This analysis revealed that the Moderate ARDS and Pneumonia presets initially had over-compensating PS and Pmus settings that delivered volumes inconsistent with their compliance tier — Pneumonia at 798 mL despite lower compliance than Normal, and Moderate ARDS at 577 mL at higher peak pressure than Mild ARDS. The root cause was that PS and Pmus were both set high simultaneously, and because they add linearly in the driving term, the combined pressure exceeded what the compliance could absorb into a lung-protective volume. Presets were recalibrated by holding PS approximately constant across tiers and tuning Pmus to achieve the desired Vt given each tier's measured fill fraction and effective compliance.

---

## Proposed Change

Produce a formal PSV parameter grid definition document that captures the complete PARAMETER_GRID and DATASET_GRID specifications, the mechanics grid per condition tier, the condition-specific ETS calibration rationale, the validity filter thresholds, the tidal volume calibration methodology using the fill fraction formula, and the step size rationale for each parameter. This document serves as the written record of the parameter space design and as a reference for replicating or extending the PSV dataset.

---

## Acceptance Criteria

- The PSV parameter grid definition document specifies every parameter in both PARAMETER_GRID and DATASET_GRID with its range, discrete values, units, and clinical rationale
- The flow_cycle_threshold parameter is documented as PSV-specific (absent from VCV and PCV grids) with an explanation of what it controls physiologically and why its optimal value is condition-dependent
- The two-dimensional structure of the PSV parameter space (ventilator settings and patient effort parameters) is documented and the clinical rationale for varying both dimensions is explained
- The mechanics grid for each of the seven condition tiers is documented with compliance range, compliance step, resistance range, resistance step, total mechanics pairs, and the literature source for the parameter ranges
- The tidal volume calibration methodology is documented with the fill fraction formula (Vt = fill_fraction × (PS + Pmus_mean) × C_lung_rec), the derivation of the required combined driving pressure for each tier, and the preset changes made to bring delivered volumes into the lung-protective range
- The condition-specific ETS presets (particularly COPD at 0.55 and bronchospasm at 0.65, which exceed the DATASET_GRID maximum) are documented with the time-constant argument that justifies values above the standard 25% default
- The five validity filter thresholds are documented with their values, the clinical evidence behind each, and an explanation of how the PPEAK_MAX filter behaves differently in PSV (catching PEEP + PS combinations) than in VCV (catching PEEP + driving pressure combinations)
- The total combination counts are derivable from the documented grid: 900 ventilator × 300 patient = 270,000 per mechanics point for PARAMETER_GRID, and 192 ventilator × 48 patient = 9,216 per mechanics point for DATASET_GRID
- The document is written in the author's own words and demonstrates understanding of why the PSV parameter space requires a two-dimensional grid design that VCV and PCV did not

---

## Files Likely to Be Touched

- **Create:** `Docs/parameter_grids/PSV_PARAMETER_GRID.md` — the primary deliverable, containing the full PARAMETER_GRID and DATASET_GRID specifications, the two-dimensional parameter structure, the mechanics grid per condition tier, the ETS calibration rationale, the validity filter documentation, the tidal volume calibration methodology, and the step size rationale
- **Created:** `generator/psv_generator.py` — implements the parameter grid as the PARAMETER_GRID and DATASET_GRID constants and sweeps it via generate_dataset() (already created as part of this work)
- **Created:** `generator/conditions.py` — implements the condition-specific presets including calibrated ETS values, PS, Pmus, and other per-condition parameters (already created as part of this work)
- **Created:** `generate_psv_dataset_thinned.py` — the batch script that executes the full thinned parameter sweep across all seven condition tiers (already created as part of this work)
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the parameter grid definition process, the tidal volume calibration analysis, the fill fraction formula derivation, the preset recalibration results per tier, and the decision to use condition-specific ETS presets outside the DATASET_GRID range for the educational dashboard

---

## Status

**Complete**

The PSV parameter grid has been fully defined, implemented in `generator/psv_generator.py` as the PARAMETER_GRID and DATASET_GRID constants and `generate_dataset()` function, condition presets calibrated in `generator/conditions.py`, and the full thinned dataset generation run has been completed. The formal parameter grid definition document (`Docs/parameter_grids/PSV_PARAMETER_GRID.md`) remains to be written as the documented record of the grid design decisions and tidal volume calibration methodology.
