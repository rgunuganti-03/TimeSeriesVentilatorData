# Problem Statement — Ventilator Waveform Simulator
**Project:** Time Series Ventilator Data — Aiden Medical Internship
**Version:** 0.2 (Phase 1 + Phase 2 Implemented)
**Date:** March 2026

---

## Background

Mechanical ventilators generate continuous physiological signals during patient breathing cycles. Understanding and modeling these signals is foundational to building intelligent respiratory care systems. Currently, access to real clinical ventilator data is limited by privacy constraints, device access, and the rarity of specific pathological conditions in controlled settings.

This project addresses that gap by building a **synthetic data platform** that simulates ventilator waveforms across a range of respiratory conditions — enabling experimentation, model development, and education without requiring real patient data.

---

## Problem

There is no lightweight, accessible tool for generating and visualizing synthetic ventilator physiological time-series data that:

- Supports multiple respiratory conditions (Normal, ARDS, COPD, etc.)
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
   - ARDS (low compliance — stiff lungs)
   - COPD (high resistance — obstructed airways)
   - Bronchospasm (very high resistance — acute bronchoconstriction)
   - Pneumonia (moderate compliance reduction — alveolar consolidation)

3. Provides an interactive UI with adjustable parameters via sliders

4. Exports generated data as structured CSV files for downstream modeling

5. Is architected to scale from rule-based signal generation toward ODE-based lung mechanics models

---

## Scope

### Implemented (Phase 1 — Rule-Based)
- Rule-based synthetic waveform generation (`generator/waveforms.py`)
- Five condition presets: Normal, ARDS, COPD, Bronchospasm, Pneumonia
- Interactive visualization dashboard
- CSV export of time-series data
- JSON storage of scenario configuration
- Locally runnable Python application

### Implemented (Phase 2 — ODE Single-Compartment)
- ODE lung mechanics model using `scipy.integrate.solve_ivp` (`generator/ode_solver.py`)
- PC-CMV ventilation mode with emergent auto-PEEP in high-resistance conditions
- Both models share the same interface and UI — switchable via sidebar radio button

### Out of Scope (current)
- Real patient data ingestion
- Cloud deployment
- Multi-user access
- Clinical validation

---

## Success Criteria

- A user can select a respiratory condition, adjust parameters, and immediately see updated waveforms
- Generated data is physiologically plausible (correct shape, direction, and relative scale)
- Output CSV can be loaded into a Python notebook for further analysis
- Codebase is modular enough that the signal generator can be swapped without rewriting the UI

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
2. What respiratory rate and I:E ratio ranges should be supported?
3. How many breath cycles per export is useful for modeling purposes?
4. What is the eventual target fidelity — rule-based, ODE, or learned generative model?
