# Architecture — Ventilator Waveform Simulator
**Project:** Time Series Ventilator Data — Aiden Medical Internship
**Version:** 0.2 (Phase 1 + Phase 2 Implemented)
**Date:** March 2026

---

## Overview

The simulator is a locally-runnable Python application structured in three independent layers:
**Data Generation → Data Layer → Visualization UI**

Each layer is decoupled so that any one can be upgraded independently as the project scales.

---

## Folder Structure

```
time-series-ventilator-data/
│
├── README.md                  # Setup and run instructions
├── PROBLEM_STATEMENT.md       # Problem framing document
├── ARCHITECTURE.md            # This file
│
├── app.py                     # App entry point — launches Streamlit UI
│
├── generator/
│   ├── __init__.py
│   ├── waveforms.py           # Phase 1: rule-based waveform generation
│   ├── ode_solver.py          # Phase 2: scipy ODE single-compartment model
│   └── conditions.py          # Condition presets (Normal, ARDS, COPD, Bronchospasm, Pneumonia)
│
├── data/
│   ├── exports/               # CSV exports of generated time-series
│   └── scenarios/             # JSON files storing scenario configurations
│
├── ui/
│   ├── __init__.py
│   └── dashboard.py           # Streamlit dashboard — sliders, plots, export
│
├── tests/
│   ├── test_waveforms.py      # Phase 1 unit tests (48 tests)
│   └── test_ode_solver.py     # Phase 2 unit tests (32 tests)
│
└── requirements.txt           # Python dependencies
```

---

## Layer Descriptions

### 1. Generator Layer (`generator/`)

Responsible for all signal computation. Takes physiological parameters as input and returns NumPy arrays for pressure, flow, and volume over time.

**Phase 1 — Rule-Based:**
- Waveforms are constructed using mathematical approximations of physiological shapes
- Sinusoidal and piecewise functions model the breath cycle
- Parameters: respiratory rate, tidal volume, compliance, resistance, I:E ratio, PEEP

**Phase 2 — ODE-Based (implemented):**
- `ode_solver.py` solves the single-compartment RC lung ODE using `scipy.integrate.solve_ivp`
- Models PC-CMV ventilation; auto-PEEP emerges naturally in high-resistance conditions
- All other layers (UI, data) are unchanged — same interface contract as Phase 1

**Interface contract (preserved across phases):**
```python
def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    # Returns: { "time": np.array, "pressure": np.array,
    #            "flow": np.array, "volume": np.array }
```

---

### 2. Data Layer (`data/`)

Handles structured storage and export.

**CSV format (time-series export):**
```
time_s, pressure_cmH2O, flow_Ls, volume_mL
0.00,   5.0,            0.42,    0.0
0.01,   5.8,            0.41,    4.2
...
```

**JSON format (scenario config):**
```json
{
  "condition": "ARDS",
  "respiratory_rate": 18,
  "tidal_volume_mL": 400,
  "compliance_mL_per_cmH2O": 20,
  "resistance_cmH2O_per_L_s": 10,
  "ie_ratio": 0.5,
  "peep_cmH2O": 8,
  "generated_at": "2026-03-19T12:00:00"
}
```

---

### 3. UI Layer (`ui/`)

Streamlit dashboard providing:
- Condition selector (dropdown)
- Parameter sliders (respiratory rate, tidal volume, compliance, resistance, PEEP)
- Live waveform plots (Plotly — 3 subplots: Pressure, Flow, Volume)
- Export button → writes CSV + JSON to `data/exports/` and `data/scenarios/`

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Signal generation | NumPy | Waveform math |
| Data handling | Pandas | DataFrame + CSV export |
| Visualization | Plotly | Interactive waveform charts |
| UI framework | Streamlit | Browser-based dashboard |
| Config storage | JSON (stdlib) | Scenario parameters |
| Language | Python 3.10+ | Primary language |
| Dependency mgmt | pip + requirements.txt | Package management |

---

## Condition Presets

| Condition | Compliance (mL/cmH₂O) | Resistance (cmH₂O/L/s) | Notes |
|---|---|---|---|
| Normal | 60 | 2 | Healthy adult baseline |
| ARDS | 18 | 4 | Severely stiff lungs, lung-protective TV |
| COPD | 55 | 18 | High resistance, prolonged expiration, auto-PEEP risk |
| Bronchospasm | 50 | 30 | Very high resistance, acute bronchoconstriction |
| Pneumonia | 35 | 6 | Moderate compliance drop, alveolar consolidation |

These values are adjustable via UI sliders — presets set the starting point.

---

## Scaling Path

```
Phase 1 (done)        Phase 2 (done)           Phase 3 (future)
─────────────         ──────────────────────   ──────────────────────────
Rule-based        →   ODE lung mechanics    →  Learned generative model
waveforms.py          ode_solver.py             ml_generator.py

Same interface contract throughout — only generator/ changes
UI and data layers are untouched
```

---

## Setup Instructions (to be expanded in README.md)

```bash
# 1. Clone or open the project folder in VS Code

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## Dependencies (`requirements.txt`)

```
numpy>=1.24
pandas>=2.0
plotly>=5.0
streamlit>=1.30
scipy>=1.10          # Used by Phase 2 ODE solver (ode_solver.py)
```
