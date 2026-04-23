# Brainstorm - Riya Synthetic Data in the Larger AiRA Direction

**Author:** Codex
**Date:** 2026-04-09
**Status:** Brainstorm Memo

---

## Purpose

This note captures:

- the earlier cross-project assessment of `aira-agent` and `Literature Crawler`
- the likely system relationship between those projects
- the most promising ways Riya's synthetic ventilator time-series project can
  feed into that larger direction

The goal is not to lock architecture yet. The goal is to preserve the current
thinking while deeper brainstorming continues with Claude Opus.

---

## Short Read on the Two Other Projects

### 1. `Literature Crawler`

Current read:

- This is best understood as an evidence-ingestion and domain-expansion system
- Its strongest asset is the domain structure and search framing
- The key file is `domain_expansion_map.json`
- The pipeline split is strong: search, metadata, download, extraction

Strategic role:

- source of truth for domain ontology
- source of literature grounding
- source of paper-level evidence lineage
- source of clinically relevant thresholds and protocol concepts

Main weakness:

- still behaves like a research workspace rather than a clean product repo
- large downloaded artifacts are mixed in with code
- extraction is still heuristic and early-stage

---

### 2. `aira-agent`

Current read:

- This is best understood as the orchestration, curation, review, and future
  training layer
- It has stronger product and system framing
- It appears to be converging on review workflow, research workflow, and model
  training as connected tracks

Strategic role:

- human-in-the-loop review
- curation workflow
- downstream knowledge operations
- future training and evaluation workflows

Main weakness:

- operational rigor still needs work
- there is an immediate security issue around hardcoded Supabase secrets that
  should be treated urgently

---

## Best Combined View

The strongest directional read so far is:

- `Literature Crawler` should own evidence ingestion and ontology growth
- `aira-agent` should own curation, review, orchestration, and downstream use

That means the important integration point is not "more agents."

It is:

- shared ontology
- shared knowledge schema
- shared provenance and evidence model

The likely convergence layer should include:

- domain
- concept
- evidence tier
- source type
- citation and provenance
- validation state

---

## Where Riya's Work Fits

Riya's current project should not be treated as a disconnected sandbox app.

The better framing is:

**Riya's project is an upstream synthetic physiology and scenario-generation
layer that can strengthen the larger AiRA system.**

Its real value is not just waveform display.

Its value is:

- generating structured synthetic respiratory cases
- creating controllable physiological scenarios
- producing labeled time-series data
- acting as a bridge between literature concepts and simulated outputs
- serving as a validation harness for respiratory knowledge

---

## Main Angles for Using Riya's Work

### Angle 1 - Synthetic Scenario Generator for AiRA

Riya's simulator can become a controlled generator for respiratory scenarios:

- Normal
- ARDS
- COPD
- Bronchospasm
- Pneumonia
- later expanded conditions

This would allow the project to produce:

- parameterized time-series data
- known ground-truth labels
- scenario metadata
- reusable synthetic cases for downstream workflows

Potential use for AiRA:

- review fixtures
- test cases for reasoning pipelines
- structured synthetic examples for training
- regression data for future model validation

Key shift required:

The project should export not just waveforms, but:

- scenario definition
- labels
- expected physiological interpretation

---

### Angle 2 - Bridge Between Literature and Simulation

This is likely the highest-value direction.

Workflow idea:

1. `Literature Crawler` extracts domain concepts, thresholds, or parameter ranges
2. Those are normalized into a structured knowledge layer
3. Riya's simulator consumes that structure as constraints, presets, or scenario
   assumptions
4. The simulator generates evidence-informed synthetic cases

This turns the simulator from:

- "prototype waveform demo"

into:

- "simulation layer grounded by literature-derived assumptions"

This is a much stronger strategic position.

---

### Angle 3 - Validation Harness for Knowledge

Riya's work can also test whether extracted knowledge is operationally useful.

Example:

- literature-backed COPD knowledge suggests elevated resistance and air trapping
- the simulator should generate waveforms that visibly reflect those properties
- if the resulting simulated outputs do not align, either:
  - the simulator assumptions need revision, or
  - the extracted knowledge is too vague to operationalize

This makes the simulator useful as a validation bench for knowledge quality.

---

### Angle 4 - Educational / Review Asset

Riya's project can generate explainable, labeled examples for internal review.

Each synthetic case could include:

- condition
- parameter set
- waveform output
- expected physiological features
- short explanation of why the waveform looks that way

This could support:

- clinician review
- internal knowledge discussions
- review queues inside `aira-agent`

---

## Primary Direction Worth Emphasizing

The strongest version of the idea is:

**Use Riya's project to fortify the synthetic time-series layer so it becomes
expansive, reusable, and evidence-aware.**

That means:

- broader scenario coverage
- stronger labels
- clearer metadata
- more structured exports
- eventual grounding in literature-informed assumptions

This direction keeps the project useful without forcing it to become an
independent product too early.

---

## What Not to Do

Avoid pushing the simulator into a large standalone roadmap too soon.

The project is most valuable if it stays bounded and becomes excellent at one
clear role:

- synthetic respiratory scenario generation
- time-series data generation
- physiology-aligned test fixtures
- evidence-to-simulation mapping

If it expands too broadly before that foundation is solid, it will lose value.

---

## Practical Next-Step Framing for Riya

A strong framing for the next phase would be:

> Move the simulator from a prototype app toward a structured scenario engine
> that generates reusable, labeled, evidence-aware respiratory time-series data.

That gives the work a clear mission while still leaving room for exploration.

---

## Concrete Integration Ideas

### 1. Standardize scenario export

Define a machine-readable output format for each generated case, including:

- condition
- generator type
- parameter values
- time-series arrays or file references
- labels
- expected features
- notes

---

### 2. Add expected feature labels

Each preset or generated scenario should eventually include explicit tags such
as:

- `high_peak_pressure`
- `air_trapping`
- `reduced_compliance_signature`
- `prolonged_expiratory_flow_decay`
- `low_tidal_volume_strategy`

This makes the data much more useful downstream.

---

### 3. Add evidence hooks

Even before full integration, the simulator can be prepared for literature
grounding by adding placeholders such as:

- `source_basis`
- `assumption_level`
- `reference_notes`

These can later be linked to crawler outputs.

---

### 4. Create batch generation mode

The project should eventually support generating many labeled synthetic cases,
not only interactive single-case exploration.

That would support:

- dataset creation
- QA and regression testing
- future model training experiments

---

### 5. Define downstream consumption

Before overbuilding, define how another system would consume the output.

Possible downstream targets:

- `aira-agent` review queues
- future model-training pipelines
- clinician-facing internal review tools
- validation datasets for knowledge-driven logic

---

## Highest-Level Summary

The broader architecture currently looks like this:

- `Literature Crawler` -> evidence and domain knowledge
- Riya synthetic data project -> simulation and scenario generation
- `aira-agent` -> review, curation, orchestration, and downstream operations

Riya's work is therefore valuable not as an isolated app, but as a bridge
between respiratory knowledge and operational AI workflows.

That is the most promising lens to keep in mind while brainstorming further.

---

## Riya's Architectural Choices So Far

This section summarizes the architecture as it exists today, without making a
directional recommendation yet.

### 1. Implementation Stack

Current choices:

- Python-based implementation
- Streamlit for the local interactive UI
- Plotly for waveform visualization
- NumPy for signal generation
- Pandas for tabular export
- SciPy for the ODE-based model

Interpretation:

She has chosen a lightweight scientific Python stack optimized for rapid local
 prototyping rather than deployment, backend services, or production pipelines.

---

### 2. Application Shape

Current choices:

- local runnable application
- single entry point via `app.py`
- browser-based interactive dashboard through Streamlit
- parameter tuning through sliders and condition presets

Interpretation:

The project is currently shaped as an exploratory simulation tool, not as a
library, API service, dataset generator, or integrated platform component.

---

### 3. Generation Approach

Current choices:

- Phase 1: rule-based waveform generation
- Phase 2: ODE-based single-compartment lung model

What this means:

- one path generates waveforms analytically using physiological approximations
- the second path simulates a simplified lung mechanics model using differential
  equations

Interpretation:

She has already made the important architectural decision to support more than
one generation strategy behind a common interface.

That is a good choice because it allows:

- simple fast generation
- more mechanistic generation
- comparison between models
- future extension without rewriting the UI

---

### 4. Data Generation Style

Current choices:

- synthetic time-series generation
- primary signals are pressure, flow, and volume
- generation occurs from respiratory parameter inputs
- condition presets act as starting templates
- multiple breath cycles can be generated

Current generation inputs include:

- respiratory rate
- tidal volume
- compliance
- resistance
- I:E ratio
- PEEP

Interpretation:

The system is currently parameter-driven rather than data-driven.

That means:

- no learned model is used
- no real patient data is being fitted
- no retrieval-based grounding is used
- all outputs are determined by physiological assumptions encoded in code

---

### 5. Condition Modeling Choice

Current choices:

- Normal
- ARDS
- COPD
- Bronchospasm
- Pneumonia

Interpretation:

She is modeling disease variation primarily through preset parameter shifts,
rather than through richer condition-specific submodels.

This is a reasonable early architectural choice because it keeps the system
simple and inspectable.

---

### 6. Data Output Choice

Current choices:

- CSV export for time-series
- JSON export for scenario/configuration
- local file-oriented workflow

Interpretation:

She has chosen exportable structured artifacts, which is good, but the current
output design is still oriented toward local inspection rather than downstream
programmatic consumption at scale.

---

### 7. Modularity Choice

Current choices:

- `generator/` contains simulation logic
- `ui/` contains dashboard logic
- `tests/` is intended to contain verification
- `data/` is intended to contain outputs

Interpretation:

She has already adopted a modular folder structure with a clean conceptual split
between generation and presentation. This is one of the stronger architectural
choices in the project.

---

### 8. Testing Choice

Current choices:

- there is explicit intent to test the project
- ODE tests are present
- rule-based generator tests are not yet filled in

Interpretation:

Testing is part of the architecture story, but not yet implemented evenly across
the project.

---

### 9. Documentation Choice

Current choices:

- separate `README.md`
- separate `PROBLEM_STATEMENT.md`
- separate `ARCHITECTURE.md`

Interpretation:

She chose to frame the work explicitly, which is valuable. The current gap is
not absence of structure. The gap is that the docs are ahead of the evidence
trail and implementation discipline.

---

### 10. Current Architectural Identity

If described in one line, the project is currently:

**a Python-based local ventilator waveform simulation prototype with modular
generation paths, interactive exploration, and exportable synthetic time-series
outputs**

It is not yet:

- an evidence-grounded simulator
- a large-scale synthetic dataset generator
- an API service
- a model-training pipeline
- a production clinical tool

That distinction is important for deciding what to steer next.
