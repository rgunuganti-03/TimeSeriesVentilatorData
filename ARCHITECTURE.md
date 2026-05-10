# Architecture — Ventilator Waveform Simulator
**Project:** Time Series Ventilator Data — Aiden Medical Internship
**Version:** 0.2 (Phase 1 + Phase 2 Implemented)
**Date:** March 2026

---

## Overview

The simulator is a locally-runnable Python application structured in three independent layers:
**Data Generation → Data Layer → Visualization UI**

Each layer is decoupled so that any one can be upgraded independently as the project scales. New ventilation mode generators can be added to `generator/` without touching the UI or data layers.

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
│   ├── vcv_generator.py          # VCV: volume-controlled waveform generator
│   ├── pcv_generator.py          # PCV: pressure-controlled waveform generator
│   └── conditions.py             # Condition presets (Normal, ARDS, COPD, ...)
│
├── data/
│   ├── exports/               # CSV exports of generated time-series
|   |   ├── vcv/                  # VCV scenario CSVs + manifest
│   │   └── pcv/                  # PCV scenario CSVs + manifest
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

Responsible for all signal computation. Takes physiological parameters as
input and returns NumPy arrays for pressure, flow, and volume over time.
Each generator implements a distinct ventilation mode control loop.

**VCV — Volume-Controlled Ventilation (implemented):**
- `vcv_generator.py` — the ventilator prescribes flow; pressure is the
  dependent variable computed from the equation of motion
- Two inspiratory flow patterns: square (constant) and decelerating (linear
  ramp to zero)
- Analytical inspiratory phase; exponential analytical expiratory ODE solution
- Inspiratory pause (0.3 s) produces the Ppeak-to-Pplat step for resistance
  and driving pressure computation
- Inter-cycle residual volume carry-forward models progressive air trapping
  in COPD and Bronchospasm
- Derived metrics: Ppeak, Pplat, driving pressure, mean Paw, auto-PEEP,
  delivered Vt, minute ventilation

**PCV — Pressure-Controlled Ventilation (implemented):**
- `pcv_generator.py` — the ventilator prescribes inspiratory pressure; flow
  and volume are the dependent variables
- Three-phase pressure profile: linear rise ramp (0.0–0.4 s), plateau at PIP,
  drop to PEEP at expiration
- Full ODE solution using `scipy.integrate.solve_ivp` (RK45, 100 Hz output)
  across the complete multi-cycle time span
- Fill fraction computed analytically from the RC time constant and inspiratory
  time; delivered Vt = insp_pressure × C × fill_fraction
- Derived metrics: Ppeak, delivered Vt, driving pressure, mean Paw, auto-PEEP,
  fill fraction, minute ventilation, time to peak flow

**PSV — Pressure Support Ventilation (not yet implemented):**
- Patient-triggered, pressure-limited, flow-cycled
- Requires adding patient effort (Pmus) to the equation of motion:
  `P_vent + Pmus(t) = V(t)/C + R × Flow(t) + PEEP`
- Tidal volume and breath timing are both patient-dependent; breath-to-breath
  variability must be modeled
- Cycling criterion: inspiration ends when flow decays to a set fraction
  (typically 25%) of peak inspiratory flow

**SIMV — Synchronized Intermittent Mandatory Ventilation (not yet implemented):**
- Hybrid mode: a set number of mandatory VC or PC breaths per minute,
  synchronized to detected patient effort, with spontaneous PSV breaths
  permitted between mandatory cycles
- Requires generating two distinct breath types in the same time series,
  with mode-appropriate waveform shapes for each
- Synchronization window logic must be implemented to prevent breath stacking

**PRVC — Pressure-Regulated Volume Control (not yet implemented):**
- Dual-control mode: targets a set tidal volume but delivers each breath as
  a pressure-controlled breath, adjusting PIP breath-by-breath to converge
  on the Vt target
- Control algorithm: measures delivered Vt at end of each breath; increases
  or decreases next breath's PIP by a fixed increment (typically 1–3 cmH₂O)
  until Vt target is met
- Cannot be generated from single-breath snapshots — requires multi-breath
  sequences where pressure varies across cycles

**Interface contract (shared across all generators):**
```python
def generate_breath_cycles(params: dict, n_cycles: int = 5) -> dict:
    # Returns: {
    #   "time":     np.ndarray,   # seconds, 100 Hz
    #   "pressure": np.ndarray,   # cmH2O
    #   "flow":     np.ndarray,   # L/s
    #   "volume":   np.ndarray,   # mL
    #   ... plus mode-specific derived metrics and validity keys
    # }
```

All generators must conform to this contract so that the UI and data layers
require no modification when a new mode is added.

---

### 2. Data Layer (`data/`)

Handles structured storage and export. Each ventilation mode writes to its own subdirectory under `data/exports/`.

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
  "scenario_id": "VCV_Normal_C070_R010_VT006_RR015_PEEP05_IE050_square",
  "condition": "Normal",
  "engine": "vcv",
  "params": {
    "respiratory_rate": 15,
    "tidal_volume_mL": 420,
    "compliance_mL_per_cmH2O": 70,
    "resistance_cmH2O_L_s": 10,
    "ie_ratio": 0.5,
    "peep_cmH2O": 5,
    "flow_pattern": "square"
  },
  "metrics": { "ppeak_cmH2O": 17.1, "pplat_cmH2O": 13.3, "driving_p_cmH2O": 8.3 },
  "is_valid": true,
  "generated_at": "2026-05-08T22:00:00+00:00"
}
```

---

### 3. UI Layer (`ui/`)

Streamlit dashboard providing:
- Engine selector (VCV, PCV — expandable to PSV, SIMV, PRVC as implemented)
- Condition selector dropdown (seven presets)
- Parameter sliders (shared: RR, compliance, resistance, I:E, PEEP;
  VCV-specific: tidal volume, flow pattern;
  PCV-specific: inspiratory pressure, rise time)
- Live waveform plots (Plotly — three subplots: Pressure, Flow, Volume)
- Metric strip (Ppeak, Pplat, driving pressure, tidal volume, mean Paw,
  peak flow ↑, peak flow ↓, minute ventilation, auto-PEEP;
  PCV adds fill fraction)
- Export buttons → CSV time-series and JSON scenario config

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Signal generation | NumPy | Waveform math and array operations |
| ODE solving | SciPy (`solve_ivp`) | PCV ODE integration (RK45) |
| Data handling | Pandas | DataFrame + CSV export |
| Visualization | Plotly | Interactive waveform charts |
| UI framework | Streamlit | Browser-based dashboard |
| Config storage | JSON (stdlib) | Scenario parameters and metadata |
| Language | Python 3.10+ | Primary language |
| Dependency mgmt | pip + requirements.txt | Package management |

---

## Condition Presets

Seven conditions are defined in `generator/conditions.py`. Each preset
sets physiologically grounded defaults for all mechanical parameters and
is shared across all generator engines.

| Condition | Compliance (mL/cmH₂O) | Resistance (cmH₂O/L/s) | PEEP (cmH₂O) | Clinical notes |
|---|---|---|---|---|
| Normal | 70 | 10 | 5 | Healthy adult with ETT |
| Mild ARDS | 45 | 12 | 8 | P/F 200–300 |
| Moderate ARDS | 30 | 14 | 12 | P/F 100–200, baby lung |
| Severe ARDS | 18 | 16 | 16 | P/F < 100 |
| COPD | 100 | 22 | 5 | High compliance, severe obstruction |
| Bronchospasm | 70 | 35 | 3 | Near-normal compliance, very high R |
| Pneumonia | 50 | 12 | 8 | Consolidation, secretions |

Resistance values reflect total system resistance including the endotracheal
tube (ETT) contribution of approximately 5–7 cmH₂O/L/s for a 7.5 mm ID tube.
All values are adjustable via UI sliders — presets set the starting point only.

---

## Scaling Path

─────────────────────────────────────────────────────────────────┐
│  VCV — done                                                     │
│  vcv_generator.py                                               │
│  Mandatory, volume-targeted. Analytical inspiration + expiry.   │
│  Square and decelerating flow. Inspiratory pause for Pplat.     │
└──────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  PCV — done                                                     │
│  pcv_generator.py                                               │
│  Mandatory, pressure-targeted. Full ODE (solve_ivp, RK45).      │
│  Rise ramp, plateau, passive expiry. Fill fraction computed.    │
└──────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  PSV — next                                                     │
│  psv_generator.py                                               │
│  Spontaneous, patient-triggered, pressure-supported, flow-      │
│  cycled. Adds Pmus(t) patient effort term to equation of        │
│  motion. Tidal volume and breath timing are patient-dependent.  │
└──────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  SIMV — future                                                  │
│  simv_generator.py                                              │
│  Hybrid: mandatory VC/PC breaths synchronized to patient        │
│  effort, with spontaneous PSV breaths between mandatory         │
│  cycles. Two distinct waveform types in one time series.        │
└──────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  PRVC — future                                                  │
│  prvc_generator.py                                              │
│  Dual-control: pressure delivery with breath-by-breath          │
│  Vt-targeting. PIP adjusts ±1–3 cmH₂O per cycle until Vt       │
│  target is met. Multi-breath sequences only.                    │
└──────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│  Learned generative model — long-term future                    │
│  ml_generator.py                                                │
│  Data-driven model trained on the synthetic dataset produced    │
│  above. Generalizes across mode, condition, and parameter       │
│  combinations without explicit physics equations.               │
└─────────────────────────────────────────────────────────────────┘

Same interface contract throughout — only `generator/` changes.
The UI and data layers require no modification when a new engine is added.

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
scipy>=1.10          # Required by pcv_generator.py (solve_ivp)
```
