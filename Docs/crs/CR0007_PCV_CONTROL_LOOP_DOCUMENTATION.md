# CR0003 — PCV Control Loop Documentation

**Author:** Riya Gunuganti
**Date:** 2026-04-28
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before implementing the PCV generator and generating the PCV dataset, the control loop logic for Pressure-Controlled Ventilation needed to be formally defined and documented. Without a clear written specification of how PCV works mechanistically — what the ventilator controls, what the dependent variable is, how the equation of motion applies, what the fill fraction represents, and what each waveform reveals clinically — the generator implementation would have no ground truth to be validated against. There was also a risk of conflating PCV's control logic with VCV's, which would produce a generator that prescribes the wrong variable and generates physiologically incorrect waveforms. In PCV, pressure is prescribed and volume is the dependent variable — the opposite of VCV — and this fundamental distinction needed to be documented before any code was written.

---

## Current State

The control loop logic for PCV has been fully defined and implemented. The following work has been completed.

The PCV control loop was specified as Pressure-Controlled Continuous Mandatory Ventilation (PC-CMV), where the ventilator prescribes inspiratory pressure as the independent variable and volume and flow are the dependent variables — they emerge from the interaction between the applied pressure and the patient's lung mechanics. This correctly models clinical PCV where delivered tidal volume is not guaranteed: if patient mechanics worsen, the same pressure setting delivers less volume and the ventilator does not compensate.

The pressure profile was defined as a three-phase waveform per breath cycle. During the rise phase, pressure ramps linearly from PEEP to PIP over a clinician-set rise time (0.0–0.4 seconds). During the plateau phase, pressure is held constant at PIP until the end of inspiration. During the expiratory phase, pressure drops back to PEEP and the lung deflates passively via elastic recoil. Rise time of zero produces a true square wave step — the standard textbook PCV waveform. Longer rise times delay and reduce peak inspiratory flow, which decreases patient-ventilator dyssynchrony and reduces work of breathing in spontaneously triggering patients.

The governing ODE was specified as the single-compartment RC lung model equation of motion rearranged for dV/dt:

    dV/dt = (P_vent(t) - V(t)/C - PEEP) / R * 1000

where V is lung volume in mL above FRC, P_vent is the ventilator pressure profile in cmH2O, C is compliance in mL/cmH2O, R is resistance in cmH2O/L/s, and the multiplication by 1000 converts L/s to mL/s to match the mL volume state variable. The ODE is solved using scipy.integrate.solve_ivp with RK45 adaptive step size control throughout the full breath cycle.

The fill fraction metric was defined and implemented as a key clinical output unique to PCV. Fill fraction represents the fraction of the lung's steady-state volume reached during the plateau phase: 1 minus exp(-t_plateau / tau), where t_plateau is inspiratory time minus rise time and tau is the RC time constant R times C divided by 1000. A fill fraction of 1.0 means the lung fully reached steady state. A fill fraction of 0.2 means the lung only reached 20% of what the applied pressure could have delivered — the breath ended too early relative to the time constant. This is the primary mechanism of volume loss in high-resistance patients on PCV and the main source of invalid scenarios in Bronchospasm and COPD condition tiers.

The validity filter threshold for driving pressure was set at 35 cmH2O above PEEP — deliberately different from the VCV threshold of 20 cmH2O. In VCV, driving pressure is the derived elastic metric Pplat minus PEEP, for which a mortality-linked threshold of 20 cmH2O exists in the ARDS literature. In PCV, inspiratory pressure is the direct ventilator control variable and clinical PCV routinely uses pressures up to 35 cmH2O above PEEP in severe disease. Applying the VCV threshold to the PCV pressure setting would incorrectly invalidate a large fraction of clinically standard scenarios. A fill fraction threshold of 0.20 was also implemented — scenarios where the lung reaches less than 20% of steady-state volume are flagged as clinically void. The threshold was set at 0.20 after calculation confirmed that the initially written threshold of 0.10 was unreachable within the allowed resistance range of 50 cmH2O/L/s.

Eight derived metrics are returned per scenario: PPeak (equal to PIP in PCV because the plateau always reaches the set pressure), delivered tidal volume (the actual dependent variable, not guaranteed), driving pressure (the set inspiratory pressure above PEEP), mean airway pressure, auto-PEEP, fill fraction, minute ventilation, and time to peak inspiratory flow (unique to PCV — quantifies the rise time effect directly).

---

## Proposed Change

Produce a formal PCV control loop document that captures the mechanistic description of how PCV works, the equation of motion as applied in this mode, the three-phase pressure profile with rise time, the fill fraction physics and its clinical significance, the clinical interpretation of each waveform signal and how it differs from VCV, the validity filter logic and threshold rationale, and the key physiological direction claims that the unit tests verify. This document serves as the written record of the domain understanding behind the implementation and as the reference against which the generated waveforms can be clinically validated.

---

## Acceptance Criteria

- The PCV control loop document accurately describes the PCV control variable (pressure) and dependent variables (volume and flow) and clearly explains why this is the opposite of VCV
- The three-phase pressure profile is described correctly — rise phase, plateau phase, and expiratory phase — with the mathematical definition of each phase and the clinical rationale for rise time as a parameter
- The ODE governing the lung response to the prescribed pressure is stated correctly with units made explicit
- The fill fraction formula is documented with its derivation from the RC time constant, its clinical interpretation, and the reason the 0.20 threshold was chosen over the initially written 0.10
- The difference between the PCV driving pressure threshold (35 cmH2O) and the VCV driving pressure threshold (20 cmH2O) is explained with the clinical rationale for each
- The time to peak flow metric is documented as a PCV-specific output and its relationship to rise time is explained
- The delivered tidal volume is correctly identified as a dependent variable in PCV — not a guaranteed output — and the clinical implication of this is explained
- The document is written in the author's own words and demonstrates understanding of the physiology, not a reproduction of textbook language

---

## Files Likely to Be Touched

- **Create:** `Docs/control_loops/PCV_CONTROL_LOOP.md` — the primary deliverable, containing the full control loop description, three-phase pressure profile, ODE derivation, fill fraction physics, waveform interpretation, validity filter rationale with threshold justification, and comparison to VCV
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the PCV control loop definition, implementation decisions, the fill fraction threshold correction, the base parameter bug fix, the n_cycles reduction for smoke tests and unit tests, and the dataset generation launch
- **Update:** `ARCHITECTURE.md` — confirm the PCV generator is documented as a completed component in the generator layer description alongside the VCV generator

---

## Status

**Complete**

The PCV control loop logic has been fully defined, implemented in `generator/pcv_generator.py`, validated through 80 unit tests in `tests/test_pcv_generator.py`, and the dataset generation batch run has been launched via `nohup python -u generate_pcv_dataset.py > pcv_generation.log 2>&1 &` and is currently running as PID 59770. The formal control loop document (`Docs/control_loops/PCV_CONTROL_LOOP.md`) remains to be written as the written evidence of domain understanding behind the implementation.
