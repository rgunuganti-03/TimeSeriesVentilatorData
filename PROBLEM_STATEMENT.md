# Problem Statement — Ventilator Waveform Simulator
**Project:** Time Series Ventilator Data — Aiden Medical Internship
**Version:** 0.3 (VCV + PCV Implemented)
**Date:** May 2026

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

3. Provides an interactive UI with adjustable parameters via sliders

4. Exports generated data as structured CSV files for downstream modeling

5. Is architected to scale from two implemented mandatory ventilation modes (VCV, PCV) toward spontaneous and hybrid modes (PSV, SIMV, PRVC)

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

### Next Steps — Additional Ventilation Modes

The following modes are the planned next steps for the simulator. Each
introduces a new dimension of complexity — patient effort, breath-to-breath
adaptation, or hybrid mandatory/spontaneous sequencing — that builds directly
on the VCV and PCV physics already implemented.

**PSV (Pressure Support Ventilation)**
Patient-triggered, pressure-limited, flow-cycled ventilation. The patient
initiates every breath; the ventilator delivers a set pressure support above
PEEP and cycles off when inspiratory flow decays to a threshold fraction of
peak flow. Modeling PSV requires adding a patient effort term (Pmus) to the
equation of motion, making tidal volume and breath timing both patient-
dependent. Breath-to-breath variability is a feature, not an error.

**SIMV (Synchronized Intermittent Mandatory Ventilation)**
A hybrid mode that delivers a set number of mandatory breaths per minute
(either VC or PC) synchronized to the patient's effort, while allowing
spontaneous pressure-supported breaths between mandatory cycles. Generating
SIMV waveforms requires producing two distinct breath types — mandatory and
spontaneous — within the same time series, with correct synchronization windows
and phase-appropriate waveform shapes for each.

**PRVC (Pressure-Regulated Volume Control)**
A dual-control mode that targets a set tidal volume but adjusts the applied
inspiratory pressure breath-by-breath to achieve it. The control algorithm
measures delivered Vt on each breath and increases or decreases the next
breath's pressure by a fixed increment (typically 1–3 cmH₂O) to converge on
the target. Generating PRVC requires multi-breath sequences where the pressure
waveform changes across cycles — it cannot be produced from single-breath
snapshots.

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
