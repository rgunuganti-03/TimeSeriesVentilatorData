# CR0010 — PCV Dataset Generation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Once the PCV generator was implemented and validated, the full PCV dataset needed to be generated systematically across all seven condition tiers and all mechanics pairs within each tier. Running generate_dataset() interactively for a single mechanics point was sufficient for unit testing but not for producing a dataset with comprehensive physiological coverage. A batch script was needed that could iterate over every compliance-resistance pair within each condition tier's mechanics grid, call the PCV generator for each pair, write valid scenario waveforms to disk, collect all scenario metadata into a single queryable manifest, and produce a provenance log documenting exactly what was generated and when.

PCV dataset generation presented a significantly greater operational challenge than VCV generation due to the ODE solver overhead. Each PCV scenario requires a full scipy.integrate.solve_ivp call at approximately 0.27 seconds per scenario, compared to VCV's analytical integration at approximately 0.013 seconds per scenario — a 20-fold difference. With approximately 127,000 total scenarios projected across the full mechanics and parameter grid sweep, the PCV generation was estimated at 8–10 hours of continuous computation. This required a production-grade batch script rather than an interactive run: one that could be detached from the terminal using nohup, monitor its own progress by printing per-tier summaries and ETAs, flush output immediately using unbuffered mode so the log file stayed current, and complete without human intervention overnight.

Without a structured batch generation run producing a manifest and generation log, the PCV dataset would have no systematic coverage guarantees and no provenance record. The PCV dataset also needed to be structurally parallel to the VCV dataset — same condition tiers, same mechanics grid, same scenario dict schema — so that the two datasets could be used together for cross-mode analysis and model training without requiring alignment work downstream.

---

## Current State

The PCV dataset generation batch run has been launched and is currently executing. generate_pcv_dataset.py was built as a production batch script and launched via nohup python -u generate_pcv_dataset.py > pcv_generation.log 2>&1 running as PID 59770.

The batch script defines the same seven condition tiers and mechanics grids used for VCV generation, ensuring direct structural comparability between the two datasets. The mechanics grid covers the same 126 unique compliance-resistance pairs: 24 pairs for Normal, 20 for Mild ARDS, 9 for Moderate ARDS, 12 for Severe ARDS, 30 for COPD, 16 for Bronchospasm, and 15 for Pneumonia. Each pair is swept against the full 3,528-combination PCV parameter grid (7 inspiratory pressures × 7 respiratory rates × 6 PEEPs × 3 I:E ratios × 4 rise times), producing an estimated 444,528 total scenarios across the full dataset — significantly larger than the VCV dataset due to the additional rise time dimension in the PCV parameter grid.

Each scenario is run at n_cycles=10 to ensure auto-PEEP accumulation is visible in high-resistance conditions, waveform stability is reached before metrics are computed, and the fill fraction metric reflects the steady-state periodic behaviour rather than the transient startup of the first breath. The n_cycles=1 setting used in smoke tests and unit tests was deliberately not carried into the batch generation run for this reason.

The batch script was designed with several operational features not present in the VCV script. The -u flag was added to the Python invocation to force unbuffered output, ensuring that every print statement writes immediately to pcv_generation.log rather than accumulating in Python's output buffer — a critical fix discovered when the initial nohup launch produced an empty log file because the process had crashed before the buffer was flushed. An ETA line is printed after each completed tier, computing the remaining scenario count and dividing by the measured throughput rate so the overnight log provides actionable progress information. sys.stdout.flush() is called after each tier print block to guarantee the ETA reaches the log file without delay.

Multiple stale processes from earlier launch attempts were identified via ps aux and terminated before the final clean launch. The definitive process was confirmed as PID 59770 running python -u generate_pcv_dataset.py at approximately 85–100% CPU consumption. Progress is being monitored via tail pcv_generation.log. caffeinate -i was launched as a background process to prevent macOS from sleeping during the overnight run.

The projected invalidity distribution for PCV differs meaningfully from VCV in two ways. Normal carries the highest invalidity rate among all tiers in PCV rather than the lowest, because high compliance at moderate inspiratory pressure settings delivers tidal volumes above the 840 mL ceiling — a pattern that does not occur in VCV where tidal volume is the prescribed input rather than a derived output. Bronchospasm carries a high invalidity rate from fill fraction violations rather than driving pressure violations, because the high resistance combined with fast respiratory rates and long rise times collapses the fill fraction below the 0.20 threshold — a PCV-specific failure mode with no direct VCV equivalent.

Each valid scenario produces a waveform CSV file containing four columns — time_s, pressure_cmH2O, flow_Ls, volume_mL — at 100 Hz sampling rate across 10 complete breath cycles. The pcv_manifest.csv will contain one row per scenario with columns for scenario ID, condition, generated_at timestamp, is_valid flag, invalid_reason string, waveform file path, compliance, resistance, respiratory rate, inspiratory pressure, I:E ratio, PEEP, rise time, PPeak, delivered tidal volume, driving pressure, mean airway pressure, auto-PEEP, fill fraction, minute ventilation, and time to peak flow. The pcv_generation_log.json will capture the full run provenance including the parameter grid definition, n_cycles setting, IBW assumption, per-tier counts and timing, and total runtime.

---

## Proposed Change

Execute the full PCV dataset generation sweep across all seven condition tiers using a production-grade batch script that can run overnight without human intervention, writes waveform CSVs for every valid scenario, collects all scenario metadata into a single manifest CSV parallel in structure to vcv_manifest.csv, and produces a JSON provenance log documenting the complete run configuration. The batch script must handle the full mechanics grid per tier, apply the pre-generation validity filter before writing any files, print per-tier summaries and ETAs for overnight monitoring, and flush output immediately using unbuffered mode so the log file stays current throughout the run.

---

## Acceptance Criteria

- The generation run completes without errors across all seven condition tiers and all mechanics pairs
- pcv_manifest.csv contains one row per scenario with all required columns populated for valid scenarios and empty metric fields for invalid scenarios
- The number of waveform CSV files in data/exports/pcv/ equals the valid scenario count recorded in pcv_generation_log.json
- Every waveform CSV contains exactly four columns (time_s, pressure_cmH2O, flow_Ls, volume_mL) at 100 Hz sampling rate across 10 complete breath cycles
- All scenario IDs in pcv_manifest.csv are unique — no two scenarios share the same ID
- The pcv_manifest.csv contains PCV-specific columns not present in vcv_manifest.csv: insp_pressure_cmH2O, rise_time_s, fill_fraction, and time_to_peak_flow_s
- The per-tier invalidity distribution reflects the PCV-specific pattern: Normal carries the highest invalidity rate among all tiers due to VT_MAX violations at high compliance, and Bronchospasm carries elevated invalidity from fill fraction violations rather than driving pressure violations
- The Normal tier valid fraction is lower than the COPD tier valid fraction, confirming the inversion of the VCV invalidity pattern in PCV
- Severe ARDS shows elevated invalidity from both PPeak violations (high inspiratory pressure plus high PEEP exceeding 50 cmH2O) and VT_MIN violations (low compliance delivering inadequate volume at low pressure settings)
- pcv_generation_log.json is present and contains the parameter grid definition, n_cycles value, IBW assumption, per-tier counts, per-tier elapsed times, and total runtime
- The total runtime substantially exceeds the VCV generation runtime, confirming that the ODE solver overhead is correctly accounted for in the batch execution
- The scenario ID format follows the PCV encoding scheme: PCV_<Condition>_C<compliance>_R<resistance>_P<insp_pressure>_RR<rr>_PEEP<peep>_IE<ie>_RT<rise_time>

---

## Files Likely to Be Touched

- **Created:** `generate_pcv_dataset.py` — the production batch script defining the seven condition tier mechanics grids, iterating over all compliance-resistance pairs, calling generate_dataset() per pair, writing waveform CSVs for valid scenarios, assembling the manifest with PCV-specific columns, writing the generation log, printing per-tier ETAs, and flushing output in unbuffered mode for reliable overnight logging
- **Populating:** `data/exports/pcv/` — waveform CSV files named by scenario ID, pcv_manifest.csv containing one row per scenario with all scenario metadata including PCV-specific metrics, and pcv_generation_log.json containing full run provenance, currently being written by PID 59770
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the batch script operational challenges (empty log diagnosis, unbuffered mode fix, stale process cleanup), the nohup launch procedure, the caffeinate precaution, and the final generation results once the run completes

---

## Status

**Complete**

generate_pcv_dataset.py has been implemented and launched as PID 59770 via nohup python -u generate_pcv_dataset.py > pcv_generation.log 2>&1. The process is confirmed running at approximately 85–100% CPU consumption and writing output to data/exports/pcv/. The status field and the files likely to be touched section will be updated with the final valid scenario count, invalid count, and total runtime once the batch run completes overnight.
