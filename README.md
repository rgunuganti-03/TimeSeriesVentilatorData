# Ventilator Waveform Simulator
**Aiden Medical — Time Series Ventilator Data**
Volume-Controlled Ventilation (VCV) + Pressure-Controlled Ventilation (PCV)

---

## What this project does

This is an interactive synthetic ventilator waveform simulator. It generates
physiologically plausible time-series data for the three primary ventilator
signals — **Pressure**, **Flow**, and **Volume** — across seven respiratory
conditions: Normal, Mild ARDS, Moderate ARDS, Severe ARDS, COPD,
Bronchospasm, and Pneumonia.

Two generator models are available, each implementing a distinct clinical
ventilation mode:

| Generator | Mode | Control variable | Dependent variable |
|---|---|---|---|
| **VCV** | Volume-Controlled Ventilation | Flow (and therefore volume) | Pressure |
| **PCV** | Pressure-Controlled Ventilation | Airway pressure | Flow and volume |

Both generators share the same interface contract and dashboard. Switch
between them using the **Simulation Engine** selector in the sidebar.

---

## Requirements

- **Python 3.10 or later** (tested on 3.12)
- The packages listed in `requirements.txt`
numpy>=1.24
pandas>=2.0
plotly>=5.0
streamlit>=1.30
scipy>=1.10

---

## Option A — Running with Claude Code

If you have [Claude Code](https://claude.ai/code) installed, open a terminal
in this project folder and start a session:

```bash
claude
```

Then ask Claude to run the app:
! streamlit run app.py

The `!` prefix runs the command in your current shell session and streams
output directly into the conversation. Claude Code will open the app in
your browser automatically.

To run the tests from inside a Claude Code session:
! python -m pytest tests/ -v

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

1. **Simulation Engine** — select VCV or PCV in the sidebar.
2. **Respiratory Condition** — pick a preset (Normal, Mild ARDS, Moderate ARDS,
   Severe ARDS, COPD, Bronchospasm, Pneumonia).
3. **Adjust parameters** — use the sliders to override any preset value.
4. **Export** — click "Download CSV" for the time-series data or "Download JSON"
   for the full scenario configuration.

---

## Running the tests

```bash
# Activate your virtual environment first (see Step 2 above), then:

# Run all tests
python -m pytest tests/ -v

# Run only VCV generator tests
python -m pytest tests/test_vcv_generator.py -v

# Run only PCV generator tests
python -m pytest tests/test_pcv_generator.py -v
```

Each test suite covers five areas: interface contract (return types, key
presence, array shapes, parameter validation), physiological plausibility
(time monotonicity, pressure never below PEEP, volume non-negative),
waveform morphology specific to each mode, validity filter threshold logic,
and dataset generation structure and scenario ID uniqueness.

---

## Running the generators directly (smoke tests)

Each generator module can be run as a standalone script to verify that
waveforms, metrics, and the validity filter are all working correctly:

```bash
# VCV generator smoke test
python generator/vcv_generator.py

# PCV generator smoke test
python generator/pcv_generator.py
```

Both scripts run four self-contained tests: single-scenario waveform
generation, physiological direction checks (e.g. lower compliance raises
Ppeak), validity filter verification, and a small dataset sweep. A
pass/fail summary is printed for each test along with key metrics.

---

## Project structure
TimeSeriesVentilatorData/
│
├── app.py                        # Entry point — streamlit run app.py
├── requirements.txt
├── PROBLEM_STATEMENT.md          # Project background and scope
├── ARCHITECTURE.md               # Layer design and scaling path
├── EXPERIMENT_LOG.md             # Development decisions and fixes
│
├── generator/
│   ├── vcv_generator.py          # VCV waveform generator
│   ├── pcv_generator.py          # PCV waveform generator
│   └── conditions.py             # Condition presets (Normal, ARDS, COPD, ...)
│
├── ui/
│   └── dashboard.py              # Streamlit dashboard (shared by both engines)
│
├── data/
│   ├── exports/
│   │   ├── vcv/                  # VCV scenario CSVs + manifest
│   │   └── pcv/                  # PCV scenario CSVs + manifest
│   └── scenarios/                # JSON scenario configs
│
└── tests/
├── test_vcv_generator.py     # VCV generator tests
└── test_pcv_generator.py     # PCV generator tests

---

## VCV Generator — Volume-Controlled Ventilation

### Control loop

In VCV the ventilator prescribes a fixed **tidal volume** and **inspiratory
flow pattern**. Pressure is the dependent variable — it rises and falls in
direct response to the delivered volume and the patient's lung mechanics.
Because volume delivery is guaranteed, changes in compliance or resistance are
immediately visible in the pressure waveform, making VCV the preferred mode
for monitoring respiratory mechanics at the bedside.

### Equation of motion

The pressure required to deliver volume V at flow Q̇ is:
Paw(t) = V(t)/C  +  Q̇(t) × R  +  PEEP
───────    ──────────
elastic    resistive

where **C** is compliance (mL/cmH₂O), **R** is total airway resistance
(cmH₂O/L/s), and **PEEP** is the end-expiratory pressure baseline.
The elastic term grows linearly as volume accumulates during inspiration.
The resistive term is proportional to instantaneous flow and vanishes when
flow reaches zero, enabling the Ppeak-to-Pplat step.

### Two inspiratory flow patterns

**Square (constant) flow**

Flow is held at `Q̇ = Vt / t_insp` throughout inspiration. Volume rises as
a straight linear ramp. Pressure also rises as a straight ramp, since both
the elastic term (growing linearly) and the resistive term (constant) are
steady throughout. The slope of the ramp equals `Q̇ / C`.

The Ppeak − Pplat gap equals `Q̇ × R` exactly and directly encodes total
airway resistance. The **stress index** — the curvature of the pressure ramp
during constant-flow inspiration — is the primary tool for titrating PEEP in
ARDS: a straight ramp (SI ≈ 1.0) indicates safe ventilation; concave-up
(SI < 0.95) signals cyclic recruitment at insufficient PEEP; concave-down
(SI > 1.05) signals progressive overdistension.

**Decelerating flow**

Flow starts at a high peak and decreases linearly to zero at end-inspiration.
Peak flow = `2 × Vt / t_insp` — twice the square-wave value — ensuring the
triangular profile integrates to the same tidal volume. Volume rises steeply
at first and flattens as flow tapers. Ppeak is lower than in square flow
because the resistive contribution near zero by end-inspiration is minimal.
Pplat is identical between both patterns for the same Vt and mechanics, since
it depends only on the elastic term.

### Inspiratory pause

A 0.3 s inspiratory hold follows inspiration in both flow patterns. During
the pause, flow is zero, volume is constant, and airway pressure drops from
Ppeak to Pplat as the resistive component vanishes. This produces the classic
Ppeak-to-Pplat step and enables direct measurement of driving pressure:
ΔP = Pplat − PEEP = Vt / C

Driving pressure is the metric most strongly associated with ARDS mortality
(Amato et al., NEJM 2015) and should be kept below 15 cmH₂O in ARDS.

### Expiratory phase

Expiration is entirely passive. The lung deflates along the analytical
solution to the RC decay ODE:
V(t) = V_end_insp × exp(−t / τ)      τ = R × C / 1000  (seconds)

Peak expiratory flow reflects elastic recoil. In high-resistance conditions
(COPD, Bronchospasm), τ is long, expiratory flow never reaches zero before
the next breath, and end-expiratory volume accumulates across cycles — the
graphical signature of dynamic hyperinflation and auto-PEEP. Residual volume
carries forward between cycles so multi-cycle simulations realistically model
progressive air trapping.

### Waveform characteristics by phase

| Phase | Pressure | Flow | Volume |
|---|---|---|---|
| Inspiration (square) | Linear ramp from PEEP | Flat rectangular pulse | Straight linear rise |
| Inspiration (decelerating) | Curved rise, flattening | Descending ramp to zero | Curved rise, flattening |
| Pause | Drop: Ppeak → Pplat, flat | Zero | Flat hold at Vt |
| Expiration | Decay: Pplat → PEEP | Negative spike, mono-exp decay | Exponential decay to baseline |

### Validity thresholds

| Metric | Threshold | Clinical rationale |
|---|---|---|
| Ppeak | > 50 cmH₂O | Barotrauma risk |
| Driving pressure (Pplat − PEEP) | > 20 cmH₂O | ARDS mortality threshold |
| Delivered Vt | < 210 mL (3 mL/kg IBW) | Inadequate ventilation |
| Delivered Vt | > 840 mL (12 mL/kg IBW) | Overdistension |

### Key parameters

| Parameter | Valid range | Grid values |
|---|---|---|
| Respiratory rate | 8–30 bpm | 8, 12, 16, 20, 24, 28, 30 |
| Tidal volume | 100–1000 mL | 4, 6, 8, 10 mL/kg IBW |
| Compliance | 5–150 mL/cmH₂O | — |
| Resistance | 0.5–50 cmH₂O/L/s | — |
| I:E ratio | 0.2–1.0 | 1:1, 1:2, 1:3 |
| PEEP | 0–20 cmH₂O | 0, 4, 8, 12, 16, 20 |
| Flow pattern | — | square, decelerating |

---

## PCV Generator — Pressure-Controlled Ventilation

### Control loop

In PCV the ventilator prescribes a fixed **inspiratory pressure** above PEEP
and maintains it for the entire inspiratory time. Flow and volume are the
dependent variables — they emerge from the interaction between the applied
pressure and the patient's lung mechanics. Tidal volume is not guaranteed:
it depends on compliance, resistance, inspiratory time, and rise time, and
must be monitored continuously. This is the fundamental clinical distinction
from VCV.

### Equation of motion

The same equation of motion applies, but the ventilator prescribes the
left-hand side:
P_vent(t) = V(t)/C  +  Q̇(t) × R  +  PEEP

Rearranged as the ODE solved at each time step:
dV/dt = [ P_vent(t) − V(t)/C − PEEP ] / R  ×  1000    (mL/s)

At the start of inspiration, V ≈ 0 and the driving gradient is at its
maximum (`P_insp / R`), producing the characteristic peak inspiratory flow.
As the lung fills, V(t)/C rises toward P_insp, the gradient narrows, and
inspiratory flow decays exponentially toward zero.

The ODE is solved numerically across the full multi-cycle time span using
`scipy.integrate.solve_ivp` (RK45 method, 100 Hz output via `t_eval`,
`max_step = dt` to prevent the adaptive solver from skipping over the
pressure step discontinuity at the start and end of inspiration).

### Three phases per breath

**Phase 1 — Rise (0 → t_rise)**

Pressure ramps linearly from PEEP to PIP = PEEP + insp_pressure. Rise time
is settable from 0.0 to 0.4 s. At `rise_time = 0.0` the step is
instantaneous, producing maximum peak inspiratory flow and the fastest lung
filling. Longer rise times reduce peak flow and smooth the pressure profile —
used to improve comfort in partially spontaneous patients and reduce the risk
of flow-triggered dyssynchrony. During this phase, inspiratory flow is at its
highest and volume climbs rapidly.

**Phase 2 — Plateau (t_rise → t_insp)**

Pressure is held constant at PIP. The lung fills exponentially toward the
steady-state volume `V_ss = insp_pressure × C`. Fill fraction quantifies
how close the lung gets:
τ             = R × C / 1000                       (seconds)
t_plateau     = t_insp − t_rise
fill_fraction = 1 − exp(−t_plateau / τ)
Vt_delivered  = insp_pressure × C × fill_fraction

At fill_fraction = 1.0 the lung has fully equilibrated and inspiratory flow
is zero before t_insp ends — a true pressure plateau is achieved. At
fill_fraction < 1.0, flow is still positive when the valve cycles, meaning
the lung was still filling at end-Ti. This flow truncation is the most
important diagnostic feature in PCV: it indicates that delivered Vt is less
than the pressure setting implies, and it worsens as resistance rises or
inspiratory time shortens.

**Phase 3 — Expiration (t_insp → t_cycle)**

Pressure drops to PEEP and the lung deflates passively:
V(t) = V_end_insp × exp(−t / τ)

The expiratory flow waveform shows a negative spike at valve opening that
decays mono-exponentially toward zero. In obstructive conditions, the decay
is slow and expiratory flow may not reach zero before the next inspiration
begins — the flow waveform signature of auto-PEEP in PCV.

### Waveform characteristics by phase

| Phase | Pressure | Flow | Volume |
|---|---|---|---|
| Rise | Linear ramp: PEEP → PIP | Peak then rapid decay begins | Steep initial rise |
| Plateau | Flat rectangle at PIP | Decelerating ramp → 0 (or truncated) | Exponential rise, plateaus if fully filled |
| Expiration | Step: PIP → PEEP | Negative spike, mono-exp decay | Exponential decay to baseline |

### Fill fraction and clinical implications

| Fill fraction | Interpretation |
|---|---|
| 0.95 – 1.00 | Lung fully equilibrated; Vt predictable as P_insp × C |
| 0.80 – 0.95 | Partial filling; Vt moderately reduced; normal for higher resistance |
| 0.50 – 0.80 | Significant truncation; Vt substantially below theoretical maximum |
| 0.20 – 0.50 | Severe truncation; obstructive disease; Vt highly unpredictable |
| < 0.20 | Invalid scenario — lung barely fills; no clinical meaning |

### Validity thresholds

| Metric | Threshold | Clinical rationale |
|---|---|---|
| Ppeak | > 50 cmH₂O | Barotrauma risk |
| Inspiratory pressure | > 35 cmH₂O | Maximum clinical driving pressure |
| Delivered Vt | < 210 mL (3 mL/kg IBW) | Inadequate ventilation |
| Delivered Vt | > 840 mL (12 mL/kg IBW) | Overdistension |
| Fill fraction | < 0.20 | Lung barely fills — clinically void scenario |

### Key parameters

| Parameter | Valid range | Grid values |
|---|---|---|
| Respiratory rate | 8–30 bpm | 8, 12, 16, 20, 24, 28, 30 |
| Inspiratory pressure | 1–50 cmH₂O above PEEP | 5, 10, 15, 20, 25, 30, 35 |
| Rise time | 0.0–0.4 s | 0.0, 0.1, 0.2, 0.4 |
| Compliance | 5–150 mL/cmH₂O | — |
| Resistance | 0.5–50 cmH₂O/L/s | — |
| I:E ratio | 0.2–1.0 | 1:1, 1:2, 1:3 |
| PEEP | 0–20 cmH₂O | 0, 4, 8, 12, 16, 20 |

---

## Condition presets

Seven conditions are defined in `generator/conditions.py`. Each preset
sets physiologically grounded defaults for all mechanical parameters. Slider
adjustments in the dashboard override any preset within a session; selecting
a new condition resets all sliders to that condition's defaults.

| Condition | Compliance (mL/cmH₂O) | Resistance (cmH₂O/L/s) | PEEP (cmH₂O) | Clinical feature |
|---|---|---|---|---|
| Normal | 70 | 10 | 5 | Healthy adult with ETT |
| Mild ARDS | 45 | 12 | 8 | P/F 200–300, moderately stiff |
| Moderate ARDS | 30 | 14 | 12 | P/F 100–200, baby lung concept |
| Severe ARDS | 18 | 16 | 16 | P/F < 100, critically reduced compliance |
| COPD | 100 | 22 | 5 | High compliance, severe expiratory obstruction |
| Bronchospasm | 70 | 35 | 3 | Near-normal compliance, very high resistance |
| Pneumonia | 50 | 12 | 8 | Consolidation, secretions, moderate reduction |

Resistance values reflect **total system resistance** including the
endotracheal tube (ETT) contribution of approximately 5–7 cmH₂O/L/s for a
standard 7.5 mm internal diameter tube. A resistance of 2 cmH₂O/L/s —
below the ETT contribution alone — is physiologically unrealistic for any
mechanically ventilated patient and is not used in any preset.

