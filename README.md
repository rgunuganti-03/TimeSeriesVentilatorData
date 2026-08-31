# TimeSeriesVentilatorData

**Aiden Medical — Synthetic Mechanical Ventilator Waveform Platform**

A Python simulation platform that generates physiologically accurate mechanical ventilator waveforms — pressure, flow, and volume over time — across multiple ventilation modes and respiratory conditions. It serves two purposes:

1. **An interactive clinical education dashboard** (Streamlit) for exploring how ventilation mode, respiratory condition, and parameter settings shape the resulting waveforms.
2. **A labeled synthetic dataset generator** feeding **AiRA**, an AI respiratory care assistant, with physiologically-grounded time-series data — enumerating the parameter space from published literature rather than depending on scarce real ICU waveform data.

---

## What's implemented

**All five adult ventilation modes are complete**, each with a documented control loop, a literature-grounded parameter grid, a tested generator, and a generated dataset:

| Mode | Description |
|---|---|
| **VCV** | Volume-Controlled Ventilation — ventilator prescribes flow; pressure is the dependent variable |
| **PCV** | Pressure-Controlled Ventilation — ventilator prescribes pressure; flow and volume are dependent |
| **PSV** | Pressure Support Ventilation — patient-triggered, pressure-limited, flow-cycled, with modeled dyssynchrony and ETT complications |
| **PRVC** | Pressure-Regulated Volume Control — dual-loop mode that adapts pressure breath-to-breath toward a volume target |
| **SIMV** | Synchronized Intermittent Mandatory Ventilation — hybrid mandatory + spontaneous breathing with synchronization-window logic |

**Seven adult respiratory conditions:** Normal, Mild ARDS, Moderate ARDS, Severe ARDS, COPD, Bronchospasm, Pneumonia.

**In progress — neonatal/pediatric extension (CR0023):** three new conditions (Normal Neonate, RDS, Meconium Aspiration Syndrome) are being added on top of the adult platform. Neonatal physiology is modeled as genuinely distinct from adult physiology, not a rescaled version of it — much lower absolute compliance, ETT-dominated resistance, faster rates, and ETT leak (VTi > VTe) as a defining feature. Population-gating is implemented across all five generators and the dashboard sidebar; RDS is implemented as a single compartment, MAS is deferred pending two-compartment modeling, and neonatal dataset generation has not yet been run.

See `ARCHITECTURE.md` for the full technical breakdown, folder structure, and current status by mode.

---

## Setup

```bash
# 1. Clone or open the project folder

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Requires Python 3.12. Developed and tested on macOS.

**Dependencies:** `numpy`, `pandas`, `plotly`, `streamlit`, `scipy` (used by the ODE-solving generators via `solve_ivp`).

---

## Running the dashboard

```bash
streamlit run app.py
```

Pick a ventilation mode and condition from the sidebar; sliders adjust the underlying mechanical parameters (compliance, resistance, PEEP, rate, and mode-specific settings like tidal volume, inspiratory pressure, or SIMV's mandatory-breath sub-type). Selecting a new condition resets sliders to that condition's physiologically grounded defaults; manual adjustments persist within a session until the condition changes again. Waveforms, derived metrics, and CSV/JSON export are generated live.

---

## Generating datasets

Each mode has its own batch generation script that sweeps a thinned parameter grid across all seven condition tiers and writes a manifest CSV plus a generation log JSON to `data/exports/<mode>/`:

```bash
python generate_vcv_dataset_thinned.py
python generate_pcv_dataset_thinned.py
python generate_psv_dataset_thinned.py
python generate_prvc_dataset_thinned.py
python generate_simv_dataset_thinned.py
```

Longer runs (SIMV, PRVC) are typically launched detached and monitored to completion:

```bash
nohup python -u generate_simv_dataset_thinned.py > simv_thinned.log 2>&1 &
```

Datasets are deterministic given the same parameters — full waveform arrays are not duplicated to disk for every scenario; they're regeneratable on demand via `generate_breath_cycles(params, seed=seed)`.

---

## Running tests

```bash
pytest tests/
```

Each mode has a dedicated test file (`tests/test_<mode>_generator.py`) covering interface contract, physiological plausibility, waveform shape, condition presets, and mode-specific behavior.

---

## Project structure

```
time-series-ventilator-data/
├── app.py                          # Entry point — launches the Streamlit dashboard
├── generator/                      # All waveform generation logic
│   ├── conditions.py                # Condition presets
│   ├── vcv_generator.py / pcv_generator.py / psv_generator.py /
│   │   prvc_generator.py / simv_generator.py
│   └── dataset_io_helpers.py        # Shared dataset export helpers
├── generate_<mode>_dataset_thinned.py   # One batch generation script per mode
├── data/exports/<mode>/             # Generated manifests + generation logs
├── ui/dashboard.py                  # Streamlit dashboard
├── tests/                           # One test file per mode
├── Docs/
│   └── crs/                         # Numbered change request (CR) documents
└── EXPERIMENT_LOG.md                # Chronological build log
```

Full layer-by-layer detail lives in `ARCHITECTURE.md`.

---

## Documentation map

- **`ARCHITECTURE.md`** — technical architecture, interface contracts, per-mode status, open items
- **`PROBLEM_STATEMENT.md`** — problem framing
- **`EXPERIMENT_LOG.md`** — chronological record of what was built, what broke, and why
- **`Docs/crs/`** — sequential CR documents (CR0001+), each with problem, current state, proposed change, acceptance criteria, and status

---

## Workflow

New work is broken into small, numbered CRs before implementation. Each ventilation mode follows the same lifecycle: **control loop documentation → parameter grid definition (literature-grounded) → generator implementation and testing → thinned dataset generation → dashboard integration**, closing with a formal CR. See `Docs/crs/CR0001_PROJECT_STRUCTURE_REVIEW.md` for the origin of this workflow.

---

## Status / roadmap

- ✅ All five adult ventilation modes complete, tested, and dataset-generated
- 🔶 Neonatal/pediatric extension (CR0023) in progress — Normal Neonate and RDS implemented, MAS deferred, neonatal dataset generation not yet run
- ⏳ Open PRVC refinements (COPD compliance correction, pressure ceiling preset alignment, ARDSnet plateau check)
- ⏳ `VALIDATION.md` not yet produced
- ⏳ Additional regression test categories (scenario-ID completeness, cross-generator constant consistency) identified but not yet implemented

See `ARCHITECTURE.md` → **Known Open Items** for the full list.

