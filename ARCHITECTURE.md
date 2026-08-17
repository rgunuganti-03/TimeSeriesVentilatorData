# Architecture — Ventilator Waveform Simulator

**Project:** TimeSeriesVentilatorData — Aiden Medical Internship
**Version:** 0.6 (Five adult modes complete · Neonatal/pediatric extension in progress — CR0023)
**Date:** August 2026

---

## Overview

The simulator is a locally-runnable Python application with two purposes:

1. **A Streamlit-based interactive clinical education dashboard** — lets a user pick a ventilation mode and respiratory condition, tune parameters, and see the resulting pressure/flow/volume waveforms and derived metrics in real time.
2. **A labeled synthetic dataset pipeline feeding AiRA** — an AI respiratory care assistant being built elsewhere. Each generator can be swept across a full parameter grid to produce large, labeled, physiologically-grounded synthetic waveform datasets, rather than relying on scarce real ICU waveform data.

The application is structured in three independent layers:

**Data Generation → Data Layer → Visualization UI**

Each layer is decoupled so it can be upgraded independently. All five generators conform to the same interface contract, so the UI and data layers require no changes when a mode is added — this held true across all five mode implementations.

---

## Folder Structure

```
time-series-ventilator-data/
│
├── README.md                       # Setup and run instructions
├── PROBLEM_STATEMENT.md            # Problem framing document
├── ARCHITECTURE.md                 # This file
├── EXPERIMENT_LOG.md                # Chronological build log — decisions, fixes, why
│
├── app.py                          # App entry point — launches Streamlit UI
│
├── generator/
│   ├── __init__.py
│   ├── conditions.py                # Condition presets (7 adult tiers + neonatal, in progress)
│   ├── vcv_generator.py             # VCV — volume-controlled, mandatory
│   ├── pcv_generator.py             # PCV — pressure-controlled, mandatory
│   ├── psv_generator.py             # PSV — pressure support, patient-triggered
│   ├── prvc_generator.py            # PRVC — dual-loop pressure-regulated volume control
│   ├── simv_generator.py            # SIMV — hybrid mandatory + spontaneous
│   └── dataset_io_helpers.py        # Shared manifest/export helpers used by all dataset scripts
│
├── generate_vcv_dataset_thinned.py  # Full-grid thinned dataset generation, one per mode
├── generate_pcv_dataset_thinned.py
├── generate_psv_dataset_thinned.py
├── generate_prvc_dataset_thinned.py
├── generate_simv_dataset_thinned.py
│
├── data/
│   ├── exports/
│   │   ├── vcv/                     # Manifest CSV + generation log JSON
│   │   ├── pcv/
│   │   ├── psv/
│   │   ├── prvc/                    # 78,912 scenarios, 96.9% valid (CR0017)
│   │   └── simv/
│   └── scenarios/                   # Individual JSON scenario configs
│
├── ui/
│   ├── __init__.py
│   └── dashboard.py                 # Streamlit dashboard — sidebar, plots, metrics, export
│
├── tests/
│   ├── test_vcv_generator.py
│   ├── test_pcv_generator.py
│   ├── test_psv_generator.py
│   ├── test_prvc_generator.py
│   └── test_simv_generator.py       # 154 tests across the suite as of SIMV completion
│
├── Docs/
│   ├── control_loops/               # Per-mode control loop specs (written before implementation)
│   │   ├── PRVC_CONTROL_LOOP.md
│   │   └── SIMV_CONTROL_LOOP.md
│   └── crs/                         # Numbered, sequential change request documents
│       ├── CR0001_PROJECT_STRUCTURE_REVIEW.md
│       ├── CR0002_DOCUMENTATION_REVIEW.md
│       ├── ...
│       └── CR0023_...                # Neonatal/pediatric extension (in progress)
│
└── requirements.txt                  # Python dependencies
```

---

## Layer Descriptions

### 1. Generator Layer (`generator/`)

Responsible for all signal computation. Takes physiological parameters as input and returns NumPy arrays for pressure, flow, and volume over time, plus mode-specific derived metrics. Each generator implements a distinct ventilation mode control loop, following the shared equation-of-motion foundation:

```
P(t) = V(t)/C + R × Flow(t) + PEEP
```

**Shared interface contract (all five generators):**
```python
def generate_breath_cycles(params: dict, n_cycles: int = 5, seed: int = None) -> dict:
    # Returns: {
    #   "time":     np.ndarray,   # seconds, 100 Hz
    #   "pressure": np.ndarray,   # cmH2O
    #   "flow":     np.ndarray,   # L/s
    #   "volume":   np.ndarray,   # mL
    #   ... plus mode-specific derived metrics and an is_valid flag
    # }
```

All five modes are implemented and all have completed control-loop documentation, a literature-grounded parameter grid, generator implementation with unit tests, and a thinned dataset generation run. A future refactor into a shared `generator/lung_physics.py` module has been noted (would let all five generators call the same underlying physics functions instead of maintaining parallel copies that can drift) but has not been undertaken — it is explicitly flagged as an open architecture question rather than resolved.

**VCV — Volume-Controlled Ventilation (complete):**
- `vcv_generator.py` — the ventilator prescribes flow; pressure is the dependent variable computed from the equation of motion
- Two inspiratory flow patterns: square (constant) and decelerating (linear ramp to zero)
- Inspiratory pause produces the Ppeak-to-Pplat step for resistance and driving pressure computation
- Inter-cycle residual volume carry-forward models progressive air trapping in COPD and Bronchospasm
- Derived metrics: Ppeak, Pplat, driving pressure, mean Paw, auto-PEEP, delivered Vt, minute ventilation

**PCV — Pressure-Controlled Ventilation (complete):**
- `pcv_generator.py` — the ventilator prescribes inspiratory pressure; flow and volume are the dependent variables
- Three-phase pressure profile: linear rise ramp, plateau at PIP, drop to PEEP at expiration
- Fill fraction computed analytically from the RC time constant and inspiratory time
- Derived metrics: Ppeak, delivered Vt, driving pressure, mean Paw, auto-PEEP, fill fraction, minute ventilation

**PSV — Pressure Support Ventilation (complete):**
- `psv_generator.py` — patient-triggered, pressure-limited, flow-cycled
- Adds a patient effort term Pmus(t) to the equation of motion
- Twelve physiological modeling features including seven dyssynchrony subtypes, ETT complications (cuff leak, partial obstruction), and a spontaneous breathing trial (SBT) temporal sequence with RSBI tracking
- Cycling criterion: inspiration ends when flow decays to a set fraction of peak inspiratory flow (condition-dependent threshold; see SIMV literature-grounding notes below)
- Servo-controlled pressure display, with an internal pressure decomposition model for diagnostics

**PRVC — Pressure-Regulated Volume Control (complete):**
- `prvc_generator.py` — dual-control mode with two nested control loops at different timescales: an inner loop (equation of motion for a single breath, given the current working pressure `P_work(n)`) and an outer loop (breath-to-breath adaptive algorithm that adjusts `P_work` toward a target tidal volume)
- Outer loop: measures delivered Vt at the end of each breath, adjusts next breath's pressure by a fixed increment (`adaptation_step_cmH2O`) until Vt is within tolerance
- Two defined terminal states: **converged** (Vt within tolerance) and **ceiling-limited** (pressure hits `pressure_ceiling_cmH2O` before convergence) — the latter is a labeled, clinically real outcome, not an error
- Cannot be generated from single-breath snapshots — requires multi-breath sequences
- A third, unresolved outcome (neither converged nor ceiling-limited when `n_cycles` runs out, ~20% of scenarios overall) is a known open item — see Open Items below

**SIMV — Synchronized Intermittent Mandatory Ventilation (complete):**
- `simv_generator.py` — hybrid mode: mandatory breaths (VC or PC sub-mode) synchronized to patient effort within a synchronization window, with spontaneous PSV-style breaths permitted between mandatory cycles
- Synchronization window formalized as `W = f_window × T_mand`; three possible outcomes per patient-effort attempt: spontaneous, synchronized mandatory, or time-triggered mandatory
- The only engine among the five that must carry compartment volume and auto-PEEP state continuously across breath-type transitions (mandatory ↔ spontaneous)
- Explicit breath-stacking prevention built into the scheduling logic
- Resolved bugs during implementation: missing refractory gap after mandatory breaths, `delivered_vt_ml` reporting absolute rather than delta volume, deterministic attempt timing causing phase-lock between clocks, a missing final expiration causing inflated auto-PEEP readings, and an unphysical negative-pressure floor during passive exhalation — all fixed and covered by regression tests (CR0019–CR0022)
- A literature accuracy issue was also identified and corrected: an earlier citation of Tokioka et al. 2001 had inverted that paper's actual findings on flow-cycle thresholds

---

### 1a. Neonatal / Pediatric Extension (in progress — CR0023)

The most recent phase of work extends the platform beyond adult physiology to neonatal/pediatric scenarios, adding three new conditions: **Normal Neonate**, **RDS** (Respiratory Distress Syndrome), and **Meconium Aspiration Syndrome (MAS)**.

Neonatal physiology is treated as genuinely distinct from adult physiology, not a rescaled version of it — absolute compliance is roughly 10–20× lower, resistance is dominated by the narrow endotracheal tube (50–150 cmH₂O/L/s vs. 2–10 in adults), respiratory rates run 30–60+ bpm, time constants are much shorter, and uncuffed tubes introduce ETT leak (inspired volume exceeding expired volume) as a defining, model-able feature.

**Architectural decisions made so far:**
- Every one of the seven existing adult conditions was explicitly backfilled with `"population": "adult"` in `conditions.py`, so no condition is ambiguous about which population it belongs to
- Each of the five generator files now has a population-gated constants block (`if population == "neonate": ...`) and a shared `_neonate_or_adult()` helper, resolving an early bug class where hardcoded adult safety constants (`VT_MIN_ML`, `VT_MAX_ML`, `PPEAK_MAX_CMHH2O`, etc.) silently rejected valid neonatal scenarios
- `render_sidebar()` in `dashboard.py` branches on an `is_neonatal` flag so slider ranges (e.g. compliance 0.1–8.0 mL/cmH₂O vs. 5–150) don't silently clip neonatal parameter values
- `params["population"]` and `params["weight_kg"]` are set once, after the `if/elif engine_key ==` chain closes, rather than duplicated inside each branch
- The Amato 2015 driving-pressure check is intentionally omitted for neonates — no sourced neonatal equivalent exists
- Two adult-specific constants (`NEONATE_ETT_K1/K2`, `VT_MAX_ML_PER_KG_NEONATE` variants beyond the sourced VT floor/ceiling) were deliberately **not** added because no primary source was found — flagged rather than guessed
- ETT leak reuses the existing fixed-fraction `cuff_leak` machinery, default-on for neonatal scenarios
- RDS currently ships as a single compartment; MAS is deferred entirely — it requires genuine two-compartment modeling (an obstructive/air-trapping compartment plus an atelectatic/surfactant-inactivated compartment), and is treated as a distinct pathophysiology rather than a rescaled COPD preset
- Scenario IDs use population-gated ×10 compliance precision for neonatal scenarios, to avoid sub-1-unit rounding collisions across the much finer neonatal compliance range (particularly in RDS sweeps)
- Neonatal dataset generation will use parallel, population-specific scripts rather than sharing the adult thinned grid

**Status:** Population-gating constants and the `_neonate_or_adult()` helper are implemented across all five generator files, and the dashboard sidebar already branches correctly on `is_neonatal`. Not yet complete: MAS's two-compartment model, the five neonatal dataset generation scripts, a shared `dataset_io_helpers.py` update for neonatal-specific manifest columns, formal CR0023 write-up, and `VALIDATION.md`.

---

### 2. Data Layer (`data/`)

Handles structured storage and export. Each ventilation mode writes to its own subdirectory under `data/exports/`, via a shared `generator/dataset_io_helpers.py` module used by all five `generate_<mode>_dataset_thinned.py` scripts.

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
  "population": "adult",
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
  "generated_at": "2026-08-06T00:00:00+00:00"
}
```

Datasets are generated as **thinned** parameter grids rather than the full combinatorial space — each mode's CR documents the reduction (e.g. PRVC's thinned grid is a 77.1% reduction from the full 2,520-point grid, producing 78,912 scenarios once condition tiers and mechanics pairs are swept in). Full waveform arrays are not stored in the manifest for most scenarios; since generation is deterministic given the same parameters, waveforms are regeneratable on demand via `generate_breath_cycles(params, seed=seed)` rather than duplicated to disk.

Manifests carry mode-specific columns beyond the shared schema — e.g. PRVC's manifest includes `pressure_ceiling_cmH2O`, `breaths_to_converge`, `converged`, and `ceiling_limited`, none of which have a VCV/PCV/PSV analogue.

---

### 3. UI Layer (`ui/dashboard.py`)

Streamlit dashboard built around pure rendering functions orchestrated by a single `render()` entry point, called by `app.py`. Render order: page config → CSS injection → sidebar (returns user selections) → header → engine execution → metrics → waveform plot → export.

- `render_sidebar()` returns `(params, condition_name, engine_name, n_cycles)`. Slider `key=` arguments include condition and engine names so Streamlit reinitializes from the condition preset whenever either changes, while manual adjustments persist within a session otherwise. Branches on `is_neonatal` for slider ranges (see 1a above).
- `render_header()` displays condition and engine badges.
- `render_metrics()` renders the metric strip via custom HTML cards (switched from `st.columns()` + `.metric()` early on to avoid value truncation).
- `render_waveform_plot()` builds a three-row `plotly` subplot (`make_subplots(rows=3, cols=1, shared_xaxes=True)`) — the shared x-axis is the deliberate clinical design choice, since it lets phase relationships between pressure, flow, and volume be read directly.
- `render_export()` builds both CSV and JSON exports in memory and serves them via `st.download_button` — nothing is written to disk from the dashboard itself.
- PSV and SIMV both carry a seed in `st.session_state` (default 42) so their stochastic elements are reproducible within a session and can be advanced via a "Regenerate" action.

Same interface contract throughout — the UI and data layers require no modification when a new engine is added; this has now been validated five times.

---

## Testing (`tests/`)

Each mode has a dedicated test file (`test_<mode>_generator.py`) covering interface contract, physiological plausibility, waveform shape, all condition presets, and mode-specific behavior (e.g. PSV's `TestDyssynchrony` and `TestETTComplications`, PRVC's convergence/ceiling-limited terminal-state tests, SIMV's compartment-continuity and breath-stacking tests). The full suite stood at 154 tests as of SIMV's completion, before the neonatal extension's test additions.

**Test categories identified but not yet implemented:**
- Scenario-ID completeness regression tests (the scenario-ID collision bug class has now been caught independently in PSV, PRVC, and SIMV — each time by the first workflow that actually swept multiple mechanics pairs within a tier, never by a generator's own smoke test)
- `generate_dataset()` vs. `generate_breath_cycles()` cross-checks
- `tests/test_cross_generator_consistency.py`, asserting shared constants stay identical across the five generator files

---

## Change Request (CR) Workflow

Work is broken into small, numbered, sequentially-tracked CR documents under `Docs/crs/`, each with Problem, Current State, Proposed Change, Acceptance Criteria, Files Likely to Be Touched, and Status sections. Every ventilation mode follows the same five-step lifecycle before being considered complete: **control loop documentation → parameter grid definition (literature-grounded) → generator implementation and testing → thinned dataset generation → dashboard integration**, closing with a formal CR.

---

## Current Status By Mode

| Mode | Control Loop Doc | Parameter Grid | Generator + Tests | Dataset Generated | Dashboard |
|---|---|---|---|---|---|
| VCV | ✅ | ✅ | ✅ | ✅ | ✅ |
| PCV | ✅ | ✅ | ✅ | ✅ | ✅ |
| PSV | ✅ | ✅ | ✅ | ✅ | ✅ |
| PRVC | ✅ | ✅ | ✅ | ✅ (78,912 scenarios) | ✅ |
| SIMV | ✅ | ✅ | ✅ | ✅ | ✅ |
| Neonatal/Pediatric (CR0023) | 🔶 partial | 🔶 partial | 🔶 in progress | ❌ not started | 🔶 sliders gated |

---

## Known Open Items

- **MAS (Meconium Aspiration Syndrome):** deferred — requires a genuine two-compartment model (obstructive/air-trapping + atelectatic/surfactant-inactivated), distinct from a single-compartment preset
- **PRVC — COPD compliance:** currently set at 100 mL/cmH₂O; correction to 65–80 mL/cmH₂O identified but not applied
- **PRVC — `pressure_ceiling_cmH2O` preset misalignment** for Mild ARDS and Pneumonia, not yet resolved
- **PRVC — missing 30 cmH₂O ARDSnet plateau pressure check**, separate from the existing 50 cmH₂O barotrauma filter
- **PRVC — unresolved terminal state:** ~20% of scenarios (up to 31% in Normal) are neither converged nor ceiling-limited when `n_cycles` runs out; not yet distinguished empirically between genuine oscillation and an insufficient cycle budget
- **K1/K2 Rohrer resistance recalibration:** deferred pending access to the primary Flevari et al. (2011) source, due to flow-unit ambiguity (L/s vs. L/min) in secondary sources
- **`VALIDATION.md`** has not yet been produced — no formal document yet defines what "physiologically plausible" means for this project across all modes
- **Shared `generator/lung_physics.py` refactor:** flagged as an open architecture question (would deduplicate physics logic currently copied across all five generator files) but not undertaken

---

## Key Design Principles

- **Physiological correctness over rescaling.** RDS is a compliance-collapse disease with near-normal resistance — not a rescaled Severe ARDS. MAS is genuinely heterogeneous and two-compartment — not a rescaled COPD preset. Condition identity is grounded in distinct pathophysiology, not parameter scaling.
- **Neonatal physiology is not scaled-down adult physiology** — different absolute magnitudes, different dominant mechanisms (ETT resistance vs. airway resistance), and an entirely new phenomenon (leak) with no adult analogue.
- **Hardcoded safety/validity constants must be population-gated**, not just parameter-gated — an adult-only constant will silently reject valid neonatal input rather than erroring loudly.
- **Literature citations require primary-source verification.** Two separate incidents (the Tokioka et al. 2001 inversion, the K1/K2 unit ambiguity) demonstrate the risk of relying on secondary sources.
- **`t_cursor` running-sum time tracking** replaces the nominal-clock formula (`t0 = cycle * t_cycle`) to prevent time-monotonicity failures from independently-rounded sample counts.
- **HDF5 was abandoned** as a redundancy-detection strategy — derived metrics proved too algebraically correlated in metric space. Redundancy is instead addressed at the parameter-grid level, via the thinned generation scripts.
- **Condition switching alone does not reduce multi-compartment compliance capacity** — normalization in `C_comps_base` preserves the sum across compartments, so explicit mechanics parameters must be supplied rather than relying on condition switching to imply a mechanics change.

---

## Dependencies (`requirements.txt`)

```
numpy>=1.24
pandas>=2.0
plotly>=5.0
streamlit>=1.30
scipy>=1.10          # Required by pcv_generator.py and others (solve_ivp)
```

Python 3.12, tested on macOS.

---

## Setup Instructions

See `README.md` for full setup and run instructions.
