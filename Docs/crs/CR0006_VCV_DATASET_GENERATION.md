# CR0006 — VCV Dataset Generation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Once the VCV generator was implemented and validated, the full VCV dataset needed to be generated systematically across all seven condition tiers and all mechanics pairs within each tier. Running generate_dataset() interactively for a single mechanics point was sufficient for unit testing, but not for producing a dataset with comprehensive physiological coverage. A batch script was needed that could iterate over every compliance-resistance pair within each condition tier's mechanics grid, call the generator for each pair, write valid scenario waveforms to disk, collect all scenario metadata into a single queryable manifest, and produce a provenance log documenting exactly what was generated and when.

Without a structured batch generation run, the dataset would have no systematic coverage guarantees. A researcher loading the data downstream would have no way to know which regions of the parameter space were sampled, whether the coverage was intentional or accidental, or whether the data could be reproduced. The manifest and generation log were therefore as important as the waveform files themselves — they are what make the dataset auditable and reproducible.

---

## Current State

The VCV dataset has been fully generated and exported. The batch script generate_vcv_dataset.py swept the complete mechanics grid across all seven condition tiers, calling generate_dataset() for every compliance-resistance pair and writing outputs to data/exports/vcv/.

The mechanics grid covered 126 unique compliance-resistance pairs across the seven tiers: 24 pairs for Normal, 20 for Mild ARDS, 9 for Moderate ARDS, 12 for Severe ARDS, 30 for COPD, 16 for Bronchospasm, and 15 for Pneumonia. Each pair was swept against the full 1,008-combination VCV parameter grid (4 tidal volumes × 7 respiratory rates × 6 PEEPs × 3 I:E ratios × 2 flow patterns), producing 127,008 total scenarios across the full dataset.

The generation run completed in 28.4 minutes, reflecting the speed advantage of VCV's analytical integration approach — no ODE solver is required during inspiration, making each scenario approximately 20 times faster to compute than a comparable PCV scenario. Each scenario was run at n_cycles=10 to ensure auto-PEEP accumulation was visible in high-resistance conditions and waveform stability was reached before the metrics were computed.

The per-tier valid and invalid counts confirmed the expected physiological pattern. Normal produced 24,192 total scenarios with 24,186 valid (100.0%) and only 6 invalid — edge cases at the extreme ends of the grid where high tidal volume combined with high PEEP pushed peak pressure toward the 50 cmH2O ceiling. Mild ARDS produced 20,160 total with 17,580 valid (87.2%). Moderate ARDS produced 9,072 total with 4,212 valid (46.4%), marking the inflection point where driving pressure breaches become the dominant source of invalidity. Severe ARDS produced 12,096 total with only 1,800 valid (14.9%) — the lowest valid rate in the dataset, correctly reflecting the narrow window of safe VCV settings at compliance values of 10–20 mL/cmH2O. COPD produced 30,240 total with 26,082 valid (86.2%), with invalidity concentrated in high-resistance, fast-rate, large tidal volume combinations where the resistive pressure term pushes PPeak above 50 cmH2O. Bronchospasm produced 16,128 total with 11,627 valid (72.1%). Pneumonia produced 15,120 total with 11,334 valid (75.0%). The overall dataset contains 96,821 valid scenarios (76.2%) and 30,187 invalid (23.8%).

Each valid scenario produced a waveform CSV file containing four columns — time_s, pressure_cmH2O, flow_Ls, volume_mL — at 100 Hz sampling rate across 10 complete breath cycles. The vcv_manifest.csv contains 127,008 rows, one per scenario, with columns for scenario ID, condition, generated_at timestamp, is_valid flag, invalid_reason string, waveform file path, compliance, resistance, respiratory rate, tidal volume, tidal volume in mL/kg, I:E ratio, PEEP, flow pattern, PPeak, Pplat, driving pressure, mean airway pressure, auto-PEEP, delivered tidal volume, and minute ventilation. The vcv_generation_log.json captures the full run provenance including the parameter grid definition, n_cycles setting, IBW assumption, per-tier counts and timing, and total runtime.

---

## Proposed Change

Execute the full VCV dataset generation sweep across all seven condition tiers using a structured batch script that writes waveform CSVs for every valid scenario, collects all scenario metadata into a single manifest CSV, and produces a JSON provenance log documenting the complete run configuration. The batch script must handle the full mechanics grid per tier, apply the pre-generation validity filter before writing any files, and produce a final summary to the terminal showing total counts, valid fraction, and total runtime.

---

## Acceptance Criteria

- The generation run completes without errors across all seven condition tiers and all mechanics pairs
- vcv_manifest.csv contains exactly 127,008 rows, one per scenario, with all required columns populated for valid scenarios and empty metric fields for invalid scenarios
- The number of waveform CSV files in data/exports/vcv/ equals the valid scenario count of 96,821
- Every waveform CSV contains exactly four columns (time_s, pressure_cmH2O, flow_Ls, volume_mL) at 100 Hz sampling rate across 10 complete breath cycles
- All scenario IDs in vcv_manifest.csv are unique — no two scenarios share the same ID
- The per-tier valid fractions match the expected physiological pattern: Normal near 100%, Mild ARDS and COPD above 85%, Bronchospasm and Pneumonia above 70%, Moderate ARDS near 46%, and Severe ARDS below 20%
- The Severe ARDS valid rate is the lowest of all seven tiers, confirming that the driving pressure filter correctly constrains the parameter space at very low compliance values
- The Normal tier valid rate is the highest of all seven tiers, confirming that healthy lung mechanics satisfy safety thresholds across nearly the entire parameter grid in VCV
- vcv_generation_log.json is present and contains the parameter grid definition, n_cycles value, IBW assumption, per-tier counts, per-tier elapsed times, and total runtime
- The total runtime is consistent with VCV's analytical integration approach — the absence of an ODE solver per scenario should produce a substantially faster generation run than the equivalent PCV sweep

---

## Files Likely to Be Touched

- **Created:** `generate_vcv_dataset.py` — the batch script defining the seven condition tier mechanics grids, iterating over all compliance-resistance pairs, calling generate_dataset() per pair, writing waveform CSVs for valid scenarios, assembling the manifest, and writing the generation log
- **Populated:** `data/exports/vcv/` — 96,821 waveform CSV files named by scenario ID, vcv_manifest.csv containing 127,008 rows with all scenario metadata, and vcv_generation_log.json containing full run provenance
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the generation run results, the per-tier valid and invalid counts, the total runtime, and confirmation that the invalidity distribution matches the expected physiological pattern

---

## Status

**Complete**

The full VCV dataset has been generated and exported. generate_vcv_dataset.py completed in 28.4 minutes producing 127,008 total scenarios with 96,821 valid waveform CSV files (76.2%) across all seven condition tiers. vcv_manifest.csv and vcv_generation_log.json are present in data/exports/vcv/. The per-tier invalidity distribution matches the expected physiological pattern with Severe ARDS at the lowest valid rate (14.9%) and Normal at the highest (100.0%).
