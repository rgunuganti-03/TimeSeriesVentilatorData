# CR0003 — VCV Control Loop Documentation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before implementing the VCV generator and generating the VCV dataset, the control loop logic for Volume-Controlled Ventilation needed to be formally defined and documented. Without a clear written specification of how VCV works mechanistically — what the ventilator controls, what the dependent variable is, how the equation of motion applies, and what each waveform reveals clinically — the generator implementation would have no ground truth to be validated against. There was also a risk of conflating VCV's control logic with PCV's, which would produce a generator that prescribes the wrong variable and generates physiologically incorrect waveforms.

---

## Current State

The control loop logic for VCV has been fully defined and implemented. The following work has been completed:

The VCV control loop was specified as Volume-Controlled Continuous Mandatory Ventilation (VC-CMV), where the ventilator prescribes flow as the independent variable and pressure is the dependent variable — it rises to whatever value is required to deliver the set flow against the patient's lung mechanics. Two flow profiles were defined and implemented: a square profile (constant flow throughout inspiration, producing a characteristic linearly rising pressure waveform) and a decelerating profile (linear ramp from peak to zero, with peak set to twice the mean to preserve the tidal volume integral, producing a pressure curve that rises then partially plateaus as the resistive term decreases).

The equation of motion was applied in the forward direction throughout: `P(t) = V(t)/C + R × Flow(t) + PEEP`. Because flow is fully prescribed at every sample point, no ODE solver is needed during inspiration — volume is the cumulative integral of flow and pressure is computed directly. Expiration uses the analytical solution to the passive RC decay ODE: `V(t) = V_end × exp(-t / tau)`.

Derived metrics were defined for each scenario: PPeak, Pplat (approximated as the mean pressure across the last 10% of inspiratory samples where flow has nearly reached zero), driving pressure (Pplat minus PEEP), mean airway pressure, auto-PEEP, delivered tidal volume, and minute ventilation. A validity filter was implemented with four thresholds grounded in clinical literature: PPeak above 50 cmH₂O (barotrauma risk), driving pressure above 20 cmH₂O (ARDS mortality threshold from the ARDSNet literature), delivered tidal volume below 3 mL/kg IBW (inadequate ventilation), and delivered tidal volume above 12 mL/kg IBW (overdistension).

The parameter grid was defined covering tidal volume (4–10 mL/kg IBW), respiratory rate (8–30 bpm), PEEP (0–20 cmH₂O), I:E ratio (1:1, 1:2, 1:3), and flow pattern (square and decelerating) across seven condition tiers (Normal, Mild ARDS, Moderate ARDS, Severe ARDS, COPD, Bronchospasm, Pneumonia). The full dataset was generated producing 127,008 total scenarios with 96,821 valid (76.2%) and 30,187 invalid (23.8%), exported to `data/exports/vcv/` with a manifest CSV and generation log JSON.

---

## Proposed Change

Produce a formal VCV control loop document that captures the mechanistic description of how VCV works, the equation of motion as applied in this mode, the clinical interpretation of each waveform signal, the parameter grid definition with ranges and step sizes, the validity filter logic and threshold rationale, and the key physiological direction claims that the unit tests verify. This document serves as the written record of the domain understanding behind the implementation and as the reference against which the generated waveforms can be clinically validated.

---

## Acceptance Criteria

- The VCV control loop document accurately describes the VCV control variable (flow) and dependent variable (pressure) and explains why this is distinct from PCV
- The equation of motion is stated correctly and the direction of causality — flow prescribed, pressure derived — is clearly explained
- Both flow profiles (square and decelerating) are described with the mathematical relationship between peak flow and tidal volume stated for each
- The RC time constant and its role in expiratory flow decay are explained
- The clinical interpretation of PPeak, Pplat, and driving pressure is documented, including the threshold values used in the validity filter and their literature sources
- The parameter grid is presented in full with ranges, step sizes, and the rationale for each step size
- The invalidity analysis results are documented per condition tier, matching the actual generation output (Normal 100%, Mild ARDS 87.2%, Moderate ARDS 46.4%, Severe ARDS 14.9%, COPD 86.2%, Bronchospasm 72.1%, Pneumonia 75.0%)
- The document is written in the author's own words and demonstrates understanding of the physiology, not a reproduction of textbook language

---

## Files Likely to Be Touched

- **Create:** `Docs/control_loops/VCV_CONTROL_LOOP.md` — the primary deliverable, containing the full control loop description, equation of motion, waveform interpretation, parameter grid, validity filter rationale, and per-tier invalidity analysis
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the VCV control loop definition, implementation decisions, bugs encountered and fixed, and the dataset generation results
- **Update:** `ARCHITECTURE.md` — confirm the VCV generator is documented as a completed component in the generator layer description

---

## Status

**Complete**

The VCV control loop logic has been fully defined, implemented in `generator/vcv_generator.py`, validated through 79 unit tests in `tests/test_vcv_generator.py`, and exercised through a full dataset generation run producing 96,821 valid scenarios across seven condition tiers. The formal control loop document (`Docs/control_loops/VCV_CONTROL_LOOP.md`) remains to be written as the written evidence of domain understanding behind the implementation.
