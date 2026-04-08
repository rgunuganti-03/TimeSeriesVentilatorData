# Ventilator Waveform Simulator
**Aiden Medical — Time Series Ventilator Data**
Phase 1 (Rule-Based) + Phase 2 (ODE Single-Compartment Lung Mechanics)

---

## What this project does

This is an interactive synthetic ventilator waveform simulator. It generates
physiologically plausible time-series data for the three primary ventilator
signals — **Pressure**, **Flow**, and **Volume** — across five respiratory
conditions: Normal, ARDS, COPD, Bronchospasm, and Pneumonia.

Two generator models are available:

| Model | Description |
|---|---|
| **Phase 1 — Rule-Based** | Analytical waveforms using piecewise/exponential functions |
| **Phase 2 — ODE Single-Compartment** | Solves the single-compartment RC lung ODE using `scipy.integrate.solve_ivp` (PC-CMV mode). Auto-PEEP emerges naturally in high-resistance conditions. |

Both models use the same interface and UI. Switch between them using the
**Generator Model** radio button in the sidebar.

---

## Requirements

- **Python 3.10 or later** (tested on 3.12)
- The packages listed in `requirements.txt`

```
numpy>=1.24
pandas>=2.0
plotly>=5.0
streamlit>=1.30
scipy>=1.10
```

---

## Option A — Running with Claude Code

If you have [Claude Code](https://claude.ai/code) installed, open a terminal
in this project folder and start a session:

```bash
claude
```

Then ask Claude to run the app:

```
! streamlit run app.py
```

The `!` prefix runs the command in your current shell session and streams
output directly into the conversation. Claude Code will also open the app
in your browser automatically.

To run the tests from inside a Claude Code session:

```
! python -m pytest tests/ -v
```

---

## Option B — Running without Claude Code (terminal + Python)

### 1. Clone or download the project

```bash
git clone https://github.com/rgunuganti-03/TimeSeriesVentilatorData.git
cd TimeSeriesVentilatorData
```

### 2. Create and activate a virtual environment

```bash
# Create the environment
python3 -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the app

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`). Open
it in any browser. The dashboard loads immediately — no login or account
required.

### 5. Use the simulator

1. **Generator Model** — select Phase 1 or Phase 2 at the top of the sidebar.
2. **Respiratory Condition** — pick a preset (Normal, ARDS, COPD, Bronchospasm, Pneumonia).
3. **Adjust parameters** — use the sliders to override any preset value.
4. **Export** — click "Download CSV" for the time-series data or "Download JSON"
   for the scenario configuration.

---

## Running the tests

```bash
# Activate your virtual environment first (see Step 2 above), then:

# Run all tests
python -m pytest tests/ -v

# Run only Phase 2 ODE tests
python -m pytest tests/test_ode_solver.py -v
```

The test suite covers interface contract, physiological plausibility, all five
condition presets, and auto-PEEP detection (32 tests for Phase 2).

---

## Running the generators directly (smoke tests)

Each generator module can be run as a standalone script:

```bash
# Phase 1 rule-based generator
python generator/waveforms.py

# Phase 2 ODE generator
python generator/ode_solver.py
```

Both print a summary table of peak pressure, flow, and volume for each
condition, plus an auto-PEEP warning if residual volume is detected.

---

## Project structure

```
TimeSeriesVentilatorData/
│
├── app.py                     # Entry point — streamlit run app.py
├── requirements.txt
├── PROBLEM_STATEMENT.md       # Project background and scope
├── ARCHITECTURE.md            # Layer design and scaling path
│
├── generator/
│   ├── waveforms.py           # Phase 1: rule-based waveform generation
│   ├── ode_solver.py          # Phase 2: scipy ODE single-compartment model
│   └── conditions.py          # Condition presets (Normal, ARDS, COPD, ...)
│
├── ui/
│   └── dashboard.py           # Streamlit dashboard (shared by both phases)
│
├── data/
│   ├── exports/               # CSV exports written here
│   └── scenarios/             # JSON scenario configs written here
│
└── tests/
    ├── test_waveforms.py      # Phase 1 tests
    └── test_ode_solver.py     # Phase 2 tests (32 tests)
```

---

## Phase 1 — Rule-based model details

Phase 1 models **Volume-Controlled Continuous Mandatory Ventilation (VC-CMV)**
using analytical mathematical functions rather than a differential equation
solver.

```
Equation of motion (applied directly):
    P(t) = V(t)/C + R × Flow(t) + PEEP

where C = compliance (mL/cmH2O), R = resistance (cmH2O/L/s)
```

**Inspiration** — a decelerating (exponential decay) flow profile is
prescribed, matching the most common clinical VC-CMV waveform shape:

```
Flow(t) = Flow_peak × exp(−t / τ_insp)       τ_insp = t_insp / 3

Volume(t) = cumulative integral of Flow × dt  (mL)

Pressure(t) = V(t)/C + R × Flow(t) + PEEP    (equation of motion)
```

`Flow_peak` is solved analytically so the integral of the flow curve
exactly equals the target tidal volume:

```
Flow_peak = (V_T / 1000) / [τ_insp × (1 − exp(−t_insp / τ_insp))]
```

**Expiration** — passive recoil modeled as exponential volume decay driven
by the RC time constant of the lung:

```
τ_exp = R × C / 1000    (seconds)

Volume(t) = V_end × exp(−t / τ_exp)

Flow(t) = −(V_end / τ_exp) × exp(−t / τ_exp) / 1000   (L/s, negative)

Pressure(t) = V(t)/C + R × Flow(t) + PEEP
```

Because pressure is computed from the equation of motion, its shape reflects
the applied flow waveform — rising to a peak at the start of inspiration
(resistive + elastic load) then decaying as flow decreases. This is the
characteristic triangular/ramp pressure shape of VC-CMV ventilation.

Each breath cycle is computed independently. Volume resets to zero at the
start of every inspiration, so there is no inter-cycle interaction or
auto-PEEP modelling (see Phase 2 for that).

---

## Phase 2 — ODE model details

Phase 2 models the lung as a **single-compartment RC circuit** under
Pressure-Controlled Continuous Mandatory Ventilation (PC-CMV):

```
Equation of motion:
    P_vent(t) = V(t)/C + R × dV/dt + PEEP

Rearranged as ODE (state: V in litres):
    dV/dt = (P_vent(t) − V(t)/C − PEEP) / R

Inspiration:  P_vent = PIP  (constant driving pressure)
Expiration:   P_vent = PEEP (passive recoil)
```

PIP is derived automatically from the target tidal volume. In high-resistance
conditions (COPD, Bronchospasm), the lung does not fully deflate before the
next breath — residual volume accumulates across cycles, which is visible in
the Volume waveform as progressive air-trapping (auto-PEEP).

The output **Pressure** signal shows **alveolar pressure** (`V/C + PEEP`),
which rises as the lung fills during inspiration and decays during expiration.
This is distinct from the Phase 1 pressure shape and reflects the PC-CMV mode.
