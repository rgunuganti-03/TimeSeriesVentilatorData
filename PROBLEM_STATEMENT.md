# Problem Statement — Ventilator Waveform Simulator
**Project:** Time Series Ventilator Data — Aiden Medical Internship
**Version:** 0.4 (VCV + PCV + PSV + PRVC + SIMV Implemented; Neonatal/Pediatric Extension In Progress)
**Date:** September 2026

---

## Background

Mechanical ventilators generate continuous physiological signals during patient breathing cycles. Understanding and modeling these signals is foundational to building intelligent respiratory care systems. Currently, access to real clinical ventilator data is limited by privacy constraints, device access, and the rarity of specific pathological conditions in controlled settings.

This project addresses that gap by building a **synthetic data platform** that simulates ventilator waveforms across a range of respiratory conditions — enabling experimentation, model development, and education without requiring real patient data.

---

## Problem

There is no lightweight, accessible tool for generating and visualizing synthetic ventilator physiological time-series data that:

- Supports multiple respiratory conditions (Normal, ARDS, COPD, etc.)
- Supports multiple ventilation modes with mode-accurate control loop physics
- Allows parameter-level control (compliance, resistance, respiratory rate, tidal volume)
- Produces structured, exportable data in standard formats
- Can scale from simple rule-based simulation toward full lung mechanics modeling

---

## Goal

Build a modular, interactive ventilator waveform simulator that:

1. Generates synthetic time-series data for the three primary ventilator signals:
   - **Pressure vs Time**
   - **Flow vs Time**
   - **Volume vs Time**

2. Supports selectable respiratory condition presets including:
   - Normal healthy lung
   - Mild ARDS (P/F 200–300 — moderately stiff lungs)
   - Moderate ARDS (P/F 100–200 — baby lung concept)
   - Severe ARDS (P/F < 100 — critically reduced compliance)
   - COPD (high resistance — obstructed airways)
   - Bronchospasm (very high resistance — acute bronchoconstriction)
   - Pneumonia (moderate compliance reduction — alveolar consolidation)
   - Normal Neonate and RDS (Respiratory Distress Syndrome) — neonatal/pediatric extension, in progress

3. Provides an interactive UI with adjustable parameters via sliders

4. Exports generated data as structured CSV files for downstream modeling

5. Supports all five ventilation modes — VCV and PCV (mandatory), PSV (spontaneous), SIMV (hybrid mandatory/spontaneous), and PRVC (dual-control adaptive) — behind the same interface contract, so the UI and data layers require no changes as new modes are added

---

## Scope

### Implemented — VCV (Volume-Controlled Ventilation)
- Analytical waveform generation with inspiratory pause phase (`generator/vcv_generator.py`)
- Square and decelerating flow patterns
- Ppeak, Pplat, driving pressure, stress index, and auto-PEEP computation
- Inter-cycle residual volume carry-forward for dynamic hyperinflation modeling
- Seven condition presets: Normal, Mild ARDS, Moderate ARDS, Severe ARDS,
  COPD, Bronchospasm, Pneumonia
- Interactive visualization dashboard with VCV-specific controls
- Full parameter grid dataset generation with validity filter
- CSV and JSON export of scenarios and time-series data

### Implemented — PCV (Pressure-Controlled Ventilation)
- ODE-based waveform generation using `scipy.integrate.solve_ivp`
  (`generator/pcv_generator.py`)
- Three-phase pressure profile: rise ramp, plateau, expiration
- Configurable rise time (0.0–0.4 s)
- Fill fraction, delivered tidal volume, and auto-PEEP computation
- Auto-PEEP emerges naturally in high-resistance conditions from the ODE
- Same seven condition presets as VCV
- Interactive visualization dashboard with PCV-specific controls
- Full parameter grid dataset generation with validity filter

### Implemented — PSV (Pressure Support Ventilation)
- Event-driven breath simulation — advances by detecting patient effort onsets, checking trigger success, running the inspiratory ODE until   the flow-cycle criterion is met, then the expiratory ODE until the next effort onset (`generator/psv_generator.py`)
- Patient effort (Pmus) term added to the equation of motion — tidal volume and breath timing are both patient-dependent; breath-to-breath    variability is a feature, not an error
- Dyssynchrony modeling: ineffective triggering, trigger delay, and flow-cycle threshold variability
- ETT complications (obstruction, cuff leak) modeled as overlays
- Same seven condition presets as VCV/PCV
- Interactive visualization dashboard with PSV-specific controls
- Full parameter grid dataset generation with validity filter

### Implemented — SIMV (Synchronized Intermittent Mandatory Ventilation)
- Event-driven single time cursor threading mandatory (VC/PC) and spontaneous (PSV-style) breaths through continuous compartment and auto-    PEEP state (`generator/simv_generator.py`)
- Synchronization-window state machine classifying each patient effort as spontaneous, synchronized-mandatory, or time-triggered-mandatory
- Selectable mandatory sub-mode per scenario: VC (tidal-volume-targeted) or PC (pressure-targeted)
- Same seven condition presets as VCV/PCV/PSV
- Interactive visualization dashboard with SIMV-specific controls, including mandatory breath type selection
- Full parameter grid dataset generation with validity filter

### Implemented — PRVC (Pressure-Regulated Volume Control)
- Dual-loop breath-to-breath adaptive control (`generator/prvc_generator.py`): an inner loop identical in structure to PCV, and an outer      loop that adjusts the working pressure breath-by-breath toward a tidal volume target
- Volume-controlled test breath (breath 1) bootstraps the working pressure for breath 2, matching documented Servo/Dräger AutoFlow behavior
- Convergence and ceiling-limited terminal states tracked and retained as valid, labeled outcomes rather than hard-invalidated
- Same seven condition presets as VCV/PCV/PSV/SIMV
- Interactive visualization dashboard with PRVC-specific controls
- Full parameter grid dataset generation with validity filter

### In Progress — Neonatal/Pediatric Extension (CR0023)
- Extends the platform beyond adult physiology to neonatal/pediatric scenarios
- Normal Neonate and RDS (Respiratory Distress Syndrome) implemented across all five generators and the dashboard
- Meconium Aspiration Syndrome (MAS) scoped but deferred, pending genuine two-compartment modeling
- See `ARCHITECTURE.md` → 1a for full detail

### Out of Scope (current)
- Real patient data ingestion
- Cloud deployment
- Multi-user access
- Clinical validation

---

## Success Criteria

- A user can select a respiratory condition and ventilation mode, adjust
  parameters, and immediately see updated waveforms
- Generated data is physiologically plausible (correct shape, direction,
  relative scale, and mode-specific waveform morphology)
- Output CSV can be loaded into a Python notebook for further analysis
- Codebase is modular enough that a new ventilation mode generator can be added
  without rewriting the UI or data layer


---

## Users

- **Primary:** Intern / developer building and iterating on the platform
- **Secondary:** Mentor / technical reviewer validating physiological plausibility
- **Future:** Researchers and engineers at Aiden Medical building models on top of the dataset

---

## Key Physiological Concepts

The simulator is grounded in the **Equation of Motion for the Respiratory System:**

```
P(t) = (V(t) / C) + (R × Flow(t)) + PEEP
```

Where:
- `P(t)` — Airway pressure at time t (cmH₂O)
- `V(t)` — Volume at time t (mL)
- `C` — Lung compliance (mL/cmH₂O) — reduced in ARDS
- `R` — Airway resistance (cmH₂O/L/s) — elevated in COPD
- `PEEP` — Positive End-Expiratory Pressure (cmH₂O)

---

## Open Questions (for discussion with mentor)

1. Should the simulator model passive (fully ventilated) patients only, or also spontaneously breathing patients?
2. What patient effort profile (Pmus waveform shape, amplitude range) should be used for PSV and SIMV spontaneous breaths?
3. What is the target dataset size per mode — full grid sweep or a curated clinically representative subset?
