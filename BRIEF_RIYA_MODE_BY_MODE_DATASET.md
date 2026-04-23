# Brief — Mode-by-Mode Ventilator Waveform Dataset Generation

**From:** Prashanth (Mentor) + Claude Opus (Trinity)
**To:** Riya + Claude Code (her agent)
**Date:** 2026-04-10
**Purpose:** Define the work for Angle 1 — systematic synthetic waveform generation, mode by mode
**Expected output:** CRs that Riya creates, implements, and documents over the coming weeks

---

## The Big Picture

AiRA is building an AI respiratory care assistant for ICU nurses. One of the
layers we need is comprehensive synthetic ventilator waveform data — not a
handful of example waveforms, but a systematically generated dataset that covers
the full physiological state space for each ventilator mode.

Your prototype simulator already generates pressure, flow, and volume waveforms
for several conditions. This brief takes that work in a specific direction:
**mode-by-mode enumeration of the parameter space, grounded in published
physiological ranges, producing labeled synthetic data that can be used for
model training, validation, and eventually a digital twin.**

---

## The Core Insight Driving This Work

Most researchers treat ventilator waveform data as scarce because they think in
terms of real patient data — IRB approvals, MIMIC datasets, sparse sensor
coverage. But the governing physics of a ventilator-patient system is known and
closed-form.

A single-compartment lung model is a differential equation with a small number
of parameters. Each parameter has clinically documented ranges. The literature
tells us what compliance looks like in ARDS, what resistance looks like in COPD,
what happens when you combine the two.

Instead of hoping for enough real patient data to cover the space, we enumerate
the space mathematically:

- Take parameter ranges from published literature
- Discretise at clinically meaningful intervals
- Generate waveforms for every plausible combination
- Label each one with the physiological interpretation

Real patient data (MIMIC waveforms, if available) becomes the validation set,
not the training set. If real waveforms fall within regions our synthetic data
covers, the synthetic space is calibrated. If they don't, our parameter ranges
need expanding.

**This flips the standard approach: physics + literature → comprehensive
synthetic generation → validate against real data.**

---

## Why Mode-by-Mode Matters

Ventilator modes are not just parameter variations. They have fundamentally
different control logic, which determines what the waveforms look like.

You cannot generate realistic synthetic data by tweaking parameters on a single
generator. You need to understand the control loop per mode.

Here is why:

### VCV — Volume-Controlled Ventilation

The ventilator controls: flow (rate and pattern — square or decelerating)
What results: pressure is the dependent variable — it goes wherever it needs to
to deliver the set flow

Key waveform characteristics:
- Flow waveform is fixed (the input you set)
- Pressure waveform reveals patient mechanics (the output you read)
- PPeak = resistive pressure + elastic pressure + PEEP
- Pplat (during inspiratory pause) = elastic pressure + PEEP
- The difference (PPeak - Pplat) tells you about airway resistance
- Pplat itself tells you about compliance

Clinical relevance: PPeak rising with stable Pplat = increasing resistance
(bronchospasm, secretions). Both rising together = decreasing compliance (ARDS
progression, pneumothorax).

### PCV — Pressure-Controlled Ventilation

The ventilator controls: inspiratory pressure (set pressure above PEEP)
What results: flow and volume are dependent — they depend on patient mechanics

Key waveform characteristics:
- Pressure waveform is fixed (square wave to set pressure)
- Flow waveform is exponentially decelerating (high initial flow that tapers)
- Volume delivered depends on compliance, resistance, and inspiratory time
- If compliance drops, the same pressure delivers less volume
- If resistance increases, flow decelerates more slowly

Clinical relevance: volume is not guaranteed. If patient mechanics worsen, tidal
volume drops — the ventilator does not compensate. This is why PCV requires
closer monitoring of delivered volumes.

### PSV — Pressure Support Ventilation

The ventilator controls: pressure support level above PEEP
What results: patient triggers the breath, ventilator assists with set pressure

Key waveform characteristics:
- Breath timing is patient-driven (variable respiratory rate)
- Flow termination is based on flow decay (typically when flow drops to 25% of
  peak)
- No mandatory breaths — if patient stops triggering, no ventilation occurs
- Waveforms show breath-to-breath variability (unlike VCV/PCV)

Clinical relevance: this is used for weaning. The waveform variability is a
feature, not noise — it reflects patient effort and respiratory drive.

### PRVC — Pressure-Regulated Volume Control

The ventilator controls: adapts pressure breath-to-breath to achieve a target
tidal volume
What results: pressure changes over successive breaths based on measured volume

Key waveform characteristics:
- First breath is a test breath (low pressure)
- Subsequent breaths adjust pressure up or down to hit volume target
- Flow pattern looks like PCV (decelerating)
- But volume is regulated like VCV (consistent)
- Pressure waveform shows adaptive stepping between breaths

Clinical relevance: combines PCV's flow pattern advantages with VCV's volume
guarantee. The adaptive behavior means the waveform changes over time — a
dataset must capture multi-breath sequences, not just single breaths.

### SIMV — Synchronized Intermittent Mandatory Ventilation

The ventilator controls: a set number of mandatory breaths (VCV or PCV style)
synchronized to patient effort
What results: mix of mandatory and spontaneous breaths in the same waveform

Key waveform characteristics:
- Mandatory breaths look like VCV or PCV (depending on mode setting)
- Between mandatory breaths, patient can take spontaneous breaths (with or
  without pressure support)
- The waveform alternates between two distinct breath types
- Synchronization means mandatory breaths are triggered by patient effort when
  possible

Clinical relevance: commonly used for weaning. The ratio of mandatory to
spontaneous breaths is gradually reduced.

---

## The Parameter Space

For each mode, the following parameters define the waveform:

### Patient parameters (independent of mode)

| Parameter | Symbol | Normal range | Units | Clinical meaning |
|-----------|--------|-------------|-------|-----------------|
| Compliance | C | 50–100 | mL/cmH2O | Lung + chest wall stretchability |
| Resistance | Raw | 2–5 | cmH2O/L/s | Airway resistance to flow |
| Patient effort | Pmus | 0 to -15 | cmH2O | Muscular pressure (negative = inspiratory) |

### Disease-modified parameter ranges

| Condition | Compliance | Resistance | Notes |
|-----------|-----------|------------|-------|
| Normal | 50–100 | 2–5 | Baseline |
| Mild ARDS | 30–50 | 3–6 | Berlin criteria: P/F 200–300 |
| Moderate ARDS | 20–30 | 4–8 | Berlin criteria: P/F 100–200 |
| Severe ARDS | 10–20 | 5–10 | Berlin criteria: P/F < 100 |
| COPD | 40–80 | 10–20 | Elevated resistance, air trapping |
| Bronchospasm | 40–70 | 15–30 | Acute resistance spike |
| Pneumonia | 25–45 | 4–8 | Reduced compliance, mild resistance increase |
| Pulmonary fibrosis | 15–30 | 3–5 | Very low compliance, normal resistance |
| Obesity | 30–50 | 3–6 | Reduced chest wall compliance |
| Neuromuscular | 50–100 | 2–5 | Normal mechanics, absent patient effort |

### Ventilator settings (mode-dependent)

| Setting | Typical range | Applies to modes |
|---------|--------------|-----------------|
| Tidal volume (VT) | 4–10 mL/kg IBW | VCV, PRVC, SIMV-VC |
| Respiratory rate | 8–35 breaths/min | VCV, PCV, PRVC, SIMV |
| PEEP | 0–24 cmH2O | All modes |
| FiO2 | 0.21–1.0 | All modes (metadata, not waveform) |
| Inspiratory pressure | 5–35 cmH2O above PEEP | PCV, PSV, SIMV-PC |
| Pressure support | 5–20 cmH2O | PSV, SIMV (spontaneous breaths) |
| I:E ratio | 1:1 to 1:4 | VCV, PCV, PRVC |
| Flow pattern | Square / Decelerating | VCV |
| Rise time | 0–0.4 s | PCV, PSV |
| Flow trigger sensitivity | 1–5 L/min | PSV, SIMV |

---

## How to Approach Each Mode

For each ventilator mode, the work follows this sequence:

### Step 1 — Understand the control loop

Before writing any code, document how the mode works mechanistically:
- What does the ventilator control? (the independent variable)
- What is the dependent variable? (what results from patient + ventilator
  interaction)
- What does the equation of motion look like for this mode?
- What clinical information does each waveform (P, V, F) reveal?

Write this up as a short document per mode. This is your domain understanding
checkpoint — if the control loop description is wrong, the generated waveforms
will be wrong.

### Step 2 — Define the parameter grid

For the mode, define:
- Which patient parameters vary (compliance, resistance, effort)
- Which ventilator settings vary (VT, RR, PEEP, etc.)
- What are the clinically meaningful ranges for each
- What step size produces meaningfully different waveforms (e.g., compliance in
  steps of 5 mL/cmH2O, not 0.1)

The goal is not to generate millions of combinations. The goal is to cover the
clinically relevant state space at a resolution where adjacent points produce
visibly different waveforms.

### Step 3 — Implement the generator for this mode

Using your existing simulator architecture (rule-based or ODE-based), implement
the waveform generation for this specific mode's control logic.

Key requirements:
- The generator must produce pressure, flow, and volume as time series
- Multiple breath cycles (minimum 10, ideally 30+)
- Sampling rate: 50–100 Hz (clinical monitors typically sample at 50–100 Hz)
- Each run produces a single scenario (one parameter combination)

### Step 4 — Generate the dataset

Sweep the parameter grid systematically. For each combination:
- Generate the waveforms
- Compute derived metrics (PPeak, Pplat, mean airway pressure, auto-PEEP if
  applicable, VT delivered for PCV)
- Label with condition and expected clinical interpretation
- Export in the standard schema (defined below)

### Step 5 — Validate against known physiology

For each mode, check:
- Do normal parameter waveforms look like textbook examples?
- Do ARDS parameters produce the expected low-compliance, high-pressure pattern?
- Do COPD parameters show air trapping and incomplete expiration?
- Are the derived metrics (PPeak, Pplat) in expected ranges?

Document any discrepancies and what was adjusted.

---

## Export Schema

Every generated scenario must export with this structure. The format is
deliberately simple — JSON metadata plus CSV time series. This is
stack-independent and can be consumed by any downstream system.

### Metadata (JSON)

```json
{
  "scenario_id": "VCV_C030_R008_VT006_RR016_PEEP10_SQR",
  "mode": "VCV",
  "condition": "moderate_ards",
  "generator": "ode_single_compartment_v1",
  "timestamp": "2026-04-12T14:30:00Z",

  "patient_params": {
    "compliance_ml_cmh2o": 30,
    "resistance_cmh2o_l_s": 8,
    "patient_effort_cmh2o": 0
  },

  "ventilator_settings": {
    "tidal_volume_ml": 420,
    "respiratory_rate": 16,
    "peep_cmh2o": 10,
    "flow_pattern": "square",
    "ie_ratio": "1:2",
    "fio2": 0.6
  },

  "derived_metrics": {
    "ppeak_cmh2o": 38.2,
    "pplat_cmh2o": 24.0,
    "mean_airway_pressure_cmh2o": 16.8,
    "driving_pressure_cmh2o": 14.0,
    "auto_peep_cmh2o": 0.0,
    "delivered_vt_ml": 420,
    "minute_ventilation_l_min": 6.72
  },

  "labels": [
    "elevated_peak_pressure",
    "high_driving_pressure",
    "low_compliance_signature"
  ],

  "expected_clinical_interpretation": "Moderate ARDS pattern in VCV. Elevated driving pressure (14 cmH2O) approaches the 15 cmH2O threshold associated with increased mortality. Consider reducing VT to 5 mL/kg if driving pressure exceeds 15.",

  "waveform_file": "VCV_C030_R008_VT006_RR016_PEEP10_SQR.csv",

  "source_assumptions": {
    "compliance_range_source": "ARDSNet protocol, Berlin definition",
    "resistance_range_source": "Estimated for moderate ARDS with mild airway edema",
    "assumption_level": "literature_informed"
  }
}
```

### Time series (CSV)

```
time_s,pressure_cmh2o,flow_l_s,volume_ml,breath_number,phase
0.000,10.0,0.000,0.0,1,expiratory_pause
0.020,10.5,0.500,10.0,1,inspiration
0.040,11.2,0.500,20.0,1,inspiration
...
```

Column definitions:
- `time_s`: elapsed time in seconds from start of recording
- `pressure_cmh2o`: airway pressure
- `flow_l_s`: airway flow (positive = inspiratory, negative = expiratory)
- `volume_ml`: instantaneous volume above FRC
- `breath_number`: integer breath counter (1-indexed)
- `phase`: one of `inspiration`, `inspiratory_pause`, `expiration`,
  `expiratory_pause`

---

## Suggested Mode Sequence

Start with the simplest (most deterministic) and build toward complex:

### Week 1–2: VCV (Volume-Controlled Ventilation)

Why first: flow is controlled, so waveform shape is most predictable. Easiest to
validate against textbook examples. Largest existing literature base.

Deliverables:
- CR: VCV control loop documentation
- CR: VCV parameter grid definition
- CR: VCV generator implementation (if modifying existing)
- CR: VCV dataset generation (full parameter sweep)
- CR: VCV validation against expected waveform shapes

### Week 3–4: PCV (Pressure-Controlled Ventilation)

Why second: pressure is controlled, so this introduces the dependent-volume
concept. The exponential flow decay is a different mathematical shape.

### Week 5–6: PSV (Pressure Support Ventilation)

Why third: introduces patient-triggered variability. Breath-to-breath variation
is now a feature, not an error. Requires modeling patient effort.

### Week 7–8: PRVC (Pressure-Regulated Volume Control)

Why fourth: requires multi-breath sequences (the pressure adapts over time).
Cannot be generated as single-breath snapshots. This is the most complex control
loop.

### Week 9–10: SIMV

Why last: combines mandatory and spontaneous breaths. Requires generating two
different breath types in the same waveform. Builds on all previous modes.

---

## What "Done" Looks Like per Mode

For each ventilator mode, the completed work includes:

1. **Control loop document** — 1–2 page explanation of how the mode works
   mechanistically, what each waveform reveals clinically, and what the equation
   of motion looks like. Written in your own words, not copied.

2. **Parameter grid definition** — table of all parameters, their ranges, step
   sizes, and sources. Total number of combinations calculated.

3. **Generator implementation** — working code that produces P, V, F waveforms
   for any parameter combination in the grid.

4. **Generated dataset** — all scenarios generated, exported in standard schema.

5. **Validation document** — comparison of generated waveforms against expected
   physiological behavior. Screenshots or plots of representative examples.
   Notes on any adjustments made.

6. **Experiment log entries** — what was tried, what looked wrong, what was
   changed and why.

---

## Architecture Guidance

### Keep your current stack

Python, NumPy, SciPy, Pandas, Plotly — this is the right stack for
mathematical waveform generation. Do not introduce new frameworks unless
there is a specific technical reason.

### Data is data

The export schema above (JSON + CSV) is deliberately stack-independent.
Whatever downstream system consumes this data — whether it is TimesFM, a
Gemma fine-tuning pipeline, a clinical dashboard, or something that does not
exist yet — can read JSON and CSV. Do not optimise for a specific downstream
consumer. Optimise for clean, labeled, well-structured output.

### The choices that matter now

The architectural decisions that matter at this stage are not about which
framework or language to use. They are:

- **Schema decisions**: what metadata accompanies each waveform, how parameters
  are encoded, what labels exist. These are hard to retrofit later.
- **Naming conventions**: scenario IDs, file naming, label vocabulary. Establish
  these now and be consistent.
- **Units**: always explicit in column names and metadata. Never ambiguous.
- **Generator versioning**: tag which version of the generator produced each
  dataset. When you improve the ODE model, you need to know which data came
  from which version.

### What not to build yet

- Do not build a web application or API around this
- Do not build a database for the generated data (files are fine)
- Do not build a training pipeline (that is a separate project)
- Do not build a real-time visualization beyond what you need for validation

---

## How to Work

### Use change requests

Break work into small CRs. Each CR should define:
- Problem or goal
- Current state
- Proposed change
- Acceptance criteria
- Files to create or modify

### Document the learning path

The experiment log is as valuable as the code. When a waveform looks wrong,
document what looked wrong, what you investigated, and what you changed.
This is how you build domain expertise, and it is how we review your
understanding.

### Use your Claude Code account

Your agent can help with:
- Literature research on parameter ranges per condition
- Mathematical implementation of control loop equations
- Code review and testing
- Generating the parameter sweep code

Your agent cannot replace:
- Your understanding of why a waveform looks right or wrong
- Your judgment about whether parameter ranges are physiologically plausible
- Your documentation of what you learned

### Ask questions

If a control loop does not make sense, or a generated waveform looks wrong
and you cannot figure out why, ask. The goal is not to struggle silently.
The goal is to build understanding efficiently.

---

## References to Start With

These are starting points for understanding mode-specific waveform
characteristics:

- Pilbeam's Mechanical Ventilation — chapter on waveform analysis
- Cairo's Pilbeam — ventilator modes chapter
- Chatburn's classification of ventilator modes (2007) — the control variable
  framework
- ARDSNet protocol — for ARDS-specific VT and pressure targets
- Hess & Kacmarek, Essentials of Mechanical Ventilation — waveform
  interpretation
- Any ICU ventilator manual (Draeger, Hamilton, Maquet) — they all include
  waveform diagrams per mode

For parameter ranges by condition, the AiRA Literature Crawler has extracted
thresholds from PubMed papers. Ask your mentor for access to those extractions
if you need specific numbers.

---

## Summary

The work is: take each ventilator mode, understand its control logic deeply,
define the parameter space from literature, generate waveforms for the full
space, label everything rigorously, and validate against known physiology.

The output is: a comprehensive, labeled, export-ready synthetic dataset — one
mode at a time, starting with VCV.

The principle is: the physics is known, the parameter ranges are published, the
constraint is not data scarcity but systematic generation. We enumerate the
space rather than sampling from it.

Start with VCV. Create your first CR. Go.
