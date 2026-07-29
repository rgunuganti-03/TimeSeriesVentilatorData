# CR0011 — PSV Control Loop Documentation

**Author:** Riya Gunuganti
**Date:** 2026-05-20
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

Before implementing the PSV generator and generating the PSV dataset, the control loop logic for Pressure Support Ventilation needed to be formally defined and documented. Without a clear written specification of how PSV works mechanistically — what the ventilator controls, what the patient controls, how the equation of motion applies, what the flow-cycle criterion means, and what each waveform reveals clinically — the generator implementation would have no ground truth to be validated against.

PSV introduces a fundamental complexity absent from VCV and PCV: the patient is an active participant in every breath. In VCV, the patient is treated as a passive load on a prescribed flow. In PCV, the patient is treated as a passive load on a prescribed pressure. In PSV, the patient's inspiratory effort co-drives gas into the lung alongside the ventilator's pressure support, determines when breaths begin, and largely determines how long they last. Failing to model this interaction correctly would produce either a mode that behaves like PCV (patient effort ignored) or one where the displayed airway pressure is physiologically incorrect (showing alveolar pressure rather than airway-opening pressure).

There was also a risk of conflating the displayed airway pressure with the internal mechanical pressure. In a servo-controlled mode, the ventilator clamps airway-opening pressure to the set target regardless of what the patient's respiratory mechanics are doing internally. This distinction — that the displayed Pao is the servo target and not the sum of resistive, elastic, and PEEP components — is the most important conceptual distinction between PSV and the previous two modes and needed to be formally documented before implementation.

---

## Current State

The control loop logic for PSV has been fully defined and implemented. The following work has been completed.

**Modal classification.** PSV was specified as a spontaneous, patient-triggered, pressure-targeted, flow-cycled mode. Every breath requires a patient effort to initiate. The ventilator applies a clinician-set inspiratory pressure above PEEP (the pressure support level, PS) and holds it until inspiration is terminated by a flow-based criterion. The patient retains control of respiratory rate, inspiratory time, and tidal volume — the three variables that are fixed by the clinician in VCV and PCV.

**The trigger mechanism.** Inspiration begins when the patient's inspiratory effort (modeled as Pmus at effort onset, approximately 50% of peak Pmus at the trigger moment) overcomes the sum of the trigger threshold and the prevailing auto-PEEP. The trigger check is formalized as: effective_drive = Pmus_at_onset − auto_PEEP, and the breath is delivered if effective_drive > trigger_threshold. This correctly models the clinical phenomenon of ineffective triggering in COPD: auto-PEEP raises the effective threshold the patient must overcome, and if Pmus is insufficient, the effort produces a small flow perturbation on the waveform but no breath is delivered.

**The pressure control logic.** Once triggered, the ventilator's servo loop drives airway pressure to PEEP + PS over a clinician-set rise time, then holds it constant at PEEP + PS until the flow-cycle criterion fires. The displayed airway pressure is this servo-controlled target. This is the most important distinction from the internal mechanical decomposition: the ventilator output at the airway opening is PEEP + PS (after the rise ramp), not the sum of resistive, elastic, and total PEEP pressure components. The patient's inspiratory effort (Pmus) is a pleural-side pressure that adds to the effective driving force governing gas flow through the equation of motion, but it does not appear at the airway opening and must not be added to the displayed pressure.

**The equation of motion.** For each lung compartment, the inspiratory ODE is:

    drive_i = P_vent + Pmus(t) - V_i(t) / C_rs_i - PEEP_e
    dV_i/dt = drive_i / R_i × 1000

where P_vent is the ventilator's servo-controlled target (PEEP_e + PS after the rise ramp), Pmus(t) is the patient's time-varying inspiratory effort (a half-sine envelope scaled by the sampled peak effort), V_i is compartment volume in mL, C_rs_i is the respiratory system compliance of compartment i (lung plus chest wall in series), R_i is the compartment airway resistance, and the multiplication by 1000 converts L/s to mL/s. Passive expiration uses the deflation ODE: dV_i/dt = −(V_i / C_rs_i) / R_exp_i × 1000, driven by elastic recoil alone.

**The flow-cycle criterion.** Inspiration terminates when total inspiratory flow decays to a fraction of its peak value: Q(t) ≤ ETS × Q_peak, where ETS is the expiratory trigger sensitivity (also called flow cycle threshold, or FCT in the code). The default ETS of 0.25 means inspiration ends when flow falls to 25% of peak, matching the engineering default on Hamilton, Dräger, Servo-u, and PB840 ventilators. This criterion is what makes PSV flow-cycled rather than time-cycled: the ventilator does not have a fixed inspiratory time and the patient's respiratory mechanics directly determine how long inspiration lasts. A redundant maximum inspiratory time (MAX_INSP_TIME_S) prevents indefinite inspiration in degenerate cases.

**The fill fraction metric.** Fill fraction in PSV is defined as the ratio of actual delivered tidal volume to the theoretical equilibrium volume the lung would reach if inspiration ran to zero flow: fill_fraction = Vt_delivered / ((PS + Pmus_mean) × C_lung_rec). This differs from the PCV definition (which used 1 − exp(−t_plateau / τ)) because PSV does not have a fixed inspiratory time against which to compute the exponential. Instead, the equilibrium volume serves as the denominator. A fill fraction near 1.0 means the breath ran nearly to equilibrium before cycling — consistent with a long time constant relative to the actual inspiratory time. A fill fraction near 0.5 means roughly half the equilibrium volume was delivered. For ARDS with low compliance and short time constants, fill fractions around 0.5–0.6 are typical at default ETS; for COPD with high resistance and long time constants, fill fractions around 0.3–0.4 are typical even at elevated ETS settings.

**Auto-PEEP and intrinsic PEEP.** At the end of each passive expiration, residual lung volume above the FRC baseline produces an intrinsic PEEP: auto_PEEP = V_end_exp / C_rs_eff. This is computed at the effort onset of each breath and used in the trigger check and in the total PEEP metric. Auto-PEEP accumulates when the expiratory time is insufficient for complete lung emptying, which occurs when the time constant (τ = R × C) is long relative to the available expiratory time. COPD and bronchospasm have long time constants and therefore accumulate auto-PEEP. ARDS has short time constants and accumulates minimal auto-PEEP even at high respiratory rates.

**The pressure–volume–flow relationship.** Because Pao is servo-controlled to a constant plateau, the pressure waveform is rectangular. All breath-to-breath variability in the delivered physiology expresses itself in volume and flow, not in pressure — this is the opposite of VCV, where volume and flow are fixed and pressure is the variable trace. The trigger notch (a brief dip below PEEP just before the inspiratory valve opens) is the only visible pressure variability across breaths under synchronous conditions, and its depth varies with each breath's sampled Pmus amplitude.

**Condition-specific ETS calibration.** The expiratory trigger sensitivity is not a single global value in this simulation — it is a condition-specific preset calibrated to the time constant of each condition tier. ARDS conditions use low ETS (0.10–0.20) to prevent premature cycling caused by the very short time constants of low-compliance lungs. COPD and bronchospasm use high ETS (0.55–0.65) to prevent delayed cycling caused by the very long time constants of high-resistance lungs. This calibration is the single most impactful design choice distinguishing the PSV waveforms across condition tiers.

---

## Proposed Change

Produce a formal PSV control loop document that captures the mechanistic description of how PSV works, the modal classification as spontaneous-triggered-pressure-targeted-flow-cycled, the trigger mechanism and auto-PEEP interaction, the equation of motion with Pmus explicitly included on the driving side, the distinction between displayed airway pressure (servo target) and internal mechanical pressure (physiological decomposition), the flow-cycle criterion and its relationship to ETS, the fill fraction definition and its difference from the PCV definition, the auto-PEEP accumulation mechanism, and the condition-specific ETS calibration rationale. This document serves as the written record of the domain understanding behind the implementation and as the reference against which generated PSV waveforms can be clinically validated.

---

## Acceptance Criteria

- The PSV control loop document accurately classifies the mode as spontaneous, patient-triggered, pressure-targeted, and flow-cycled, and clearly explains what each of those four terms means in terms of who controls what
- The trigger mechanism is documented with the effective_drive formula (Pmus_at_onset − auto_PEEP > trigger_threshold) and the clinical explanation of why auto-PEEP raises the trigger burden in COPD
- The equation of motion is stated correctly with both P_vent and Pmus(t) on the driving side and with units made explicit for each term
- The distinction between displayed Pao (servo target = PEEP + PS) and internal physiological pressure (pres + pel + tpeep) is explicitly documented — this is the most important conceptual distinction from VCV and PCV
- The flow-cycle criterion is documented with the ETS formula, the default value of 0.25, and the clinical effect of raising versus lowering ETS on inspiratory time and tidal volume
- The fill fraction metric is documented with its PSV-specific definition (Vt / ((PS + Pmus_mean) × C_lung_rec)) and an explanation of why this differs from the PCV definition
- The auto-PEEP mechanism is documented with the formula (V_end_exp / C_rs_eff) and the explanation of why COPD accumulates auto-PEEP while ARDS does not
- The condition-specific ETS calibration is documented with the rationale for each tier's ETS value grounded in the time-constant argument
- The document is written in the author's own words and demonstrates understanding of the physiology, not a reproduction of textbook language

---

## Files Likely to Be Touched

- **Create:** `Docs/control_loops/PSV_CONTROL_LOOP.md` — the primary deliverable, containing the full control loop description, modal classification, trigger mechanism, equation of motion with Pmus, pressure display distinction, flow-cycle criterion, fill fraction definition, auto-PEEP mechanism, and condition-specific ETS rationale
- **Update:** `EXPERIMENT_LOG.md` — add an entry documenting the PSV control loop definition, the pressure display bug discovery and fix, the condition-specific ETS calibration decisions, the preset recalibration using the fill-fraction formula, and the key finding that pao_now must equal P_vent rather than the physiological decomposition sum
- **Update:** `ARCHITECTURE.md` — confirm the PSV generator is documented as a completed component in the generator layer description alongside VCV and PCV

---

## Status

**Complete**

The PSV control loop logic has been fully defined, implemented in `generator/psv_generator.py`, validated through unit tests in `tests/test_psv_generator.py`, and the thinned dataset generation script `generate_psv_dataset_thinned.py` has been executed. The formal control loop document (`Docs/control_loops/PSV_CONTROL_LOOP.md`) remains to be written as the written evidence of domain understanding behind the implementation.
