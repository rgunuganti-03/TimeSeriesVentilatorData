# CR0012 — PSV Generator Implementation

**Author:** Riya Gunuganti
**Date:** 2026-05-20
**Status:** Complete
**Priority:** High
**Project:** Time Series Ventilator Data

---

## Problem

PSV is architecturally the most complex mode implemented to date. VCV uses a direct analytical formula for inspiration and a single-compartment RC decay for expiration. PCV uses a scipy ODE solver on a single-compartment lung. PSV requires an event-driven, multi-compartment, Euler-stepped ODE that models the full breath cycle as a sequence of physiological events — trigger, rise, plateau, flow-cycle, passive expiration — rather than a fixed timing grid. It also requires explicit modeling of the patient as a co-contributor to gas delivery via the Pmus term, which has no analog in VCV or PCV. The generator must correctly produce seven distinct dyssynchrony subtypes, model two ETT complication types, implement a spontaneous breathing trial temporal sequence, and maintain a pressure display that reflects servo-controlled airway pressure rather than a physiological reconstruction.

Without a formally implemented PSV generator satisfying these requirements, the PSV condition tiers in the TimeSeriesVentilatorData platform would be either absent or populated with PCV-style fixed-timing waveforms that are physiologically incorrect for a spontaneous mode.

---

## Current State

The PSV generator has been fully implemented and validated. The following work has been completed.

**Architecture.** `generator/psv_generator.py` implements an event-driven main loop that processes one breath cycle at a time. Each cycle consists of: a passive expiration phase (preceding the next effort onset), a trigger check, an optional trigger notch segment, an inspiratory ODE loop, a breath-level metric computation, and a dyssynchrony classification. Unlike VCV and PCV, which operate on a fixed time grid of n_cycles × t_cycle samples, the PSV loop tracks wall-clock time via `t_current` and advances by the actual duration of each physiological phase. This means the total simulation time is not fixed in advance — it depends on the patient's neural rate, the number of triggered versus ineffective efforts, and the duration of each inspiratory and expiratory phase.

**Twelve modeling features.** The generator implements the following physiological mechanisms: (1) multi-compartment parallel lung model with condition-specific compartment profiles — COPD and Pneumonia use three compartments, ARDS uses two, all others use one; (2) per-compartment nonlinear compliance that stiffens as volume approaches a reference tidal volume, modeling stress-stiffening at high lung volumes; (3) Rohrer flow-dependent resistance in the Rohrer form K1×Q + K2×Q²  capturing turbulent pressure drop in the ETT and large airways; (4) per-compartment dynamic expiratory resistance that increases as lung volume decreases, modeling dynamic airway collapse and equal pressure point phenomena in obstructive disease; (5) inspiratory resistance with volume-dependent tethering, modeling the reduced airway resistance at higher lung volumes from surrounding parenchymal traction; (6) chest wall compliance in series with lung compliance via the standard 1/C_rs = 1/C_lung + 1/C_chest relationship; (7) PEEP-recruited compliance that increases with applied PEEP as previously collapsed alveoli open, with a condition-specific recruitment slope; (8) a log-normal breath effort distribution that samples a per-breath Pmus_peak and effort duration from distributions parameterized by the condition's pmus_mean and pmus_cv, producing realistic breath-to-breath variability; (9) a half-sine Pmus waveform within each breath that peaks at mid-inspiration and returns toward zero; (10) auto-PEEP computed from residual lung volume at each effort onset; (11) ETT complications (cuff leak and partial obstruction) modeled via Rohrer coefficient modification and volume loss fraction; and (12) a circuit compliance correction applied post-hoc to the delivered tidal volume.

**Pressure display.** The most important design decision in the implementation is that the displayed airway pressure (P_list) stores the ventilator's servo-controlled target (P_vent) during inspiration, not the sum of resistive, elastic, and total-PEEP components. During the rise phase, P_vent ramps linearly from PEEP_e to PEEP_e + PS over the set rise time. During the plateau, P_vent holds constant at PEEP_e + PS. During passive expiration, P_list stores PEEP_e. The physiological component arrays (Pres_list, Pel_list, Tpeep_list) are computed and stored separately for diagnostic metrics and the Pres/Pel ratio — they represent internal mechanical forces and do not sum to the displayed pressure. This distinction is fundamental: in a servo-controlled mode the airway opening is held at the target regardless of what the patient's mechanics are doing internally.

**Trigger notch.** Before the inspiratory ODE begins, a 40 ms trigger notch segment is appended to P_list, Q_list, and V_list. The notch depth scales with the sampled Pmus_peak for that breath (notch_depth = min(pmus_peak × 0.18, trigger_threshold)) and recovers linearly back to PEEP_e over the notch window. This adds physiologically meaningful breath-to-breath pressure variability — deeper notches for high-effort breaths, shallower for low-effort breaths — while the plateau remains constant at the servo target.

**Seven dyssynchrony subtypes.** Every breath is classified into one of seven categories by `_classify_dyssynchrony()`. Synchronous: normal triggering and cycling. Ineffective trigger: Pmus fails to overcome auto-PEEP plus trigger threshold; the effort produces a small flow perturbation but no breath is delivered. Double trigger: two successive inspiratory efforts within a single expiratory window, producing a second breath with small volume. Reverse trigger: patient effort detected during passive expiration in the previous label's window, producing a mechanically entrained breath. Delayed cycling: inspiratory time substantially exceeds effort duration, meaning the ventilator continues delivering after the patient has begun exhaling — characteristic of obstructive disease at standard ETS. Premature cycling: inspiratory time substantially undershoots effort duration, meaning the ventilator cycles off while the patient is still inspiring — characteristic of restrictive disease at standard ETS. Flow starvation: peak inspiratory flow is substantially below the patient's flow demand, indicated by a very low Q_at_trigger and high Pmus, producing a concave scooping of the pressure plateau.

**ETT complications.** `_get_ett_params()` modifies the Rohrer coefficients for partial obstruction (K1_eff = K1_base × obstruction_multiplier, K2_eff = K2_base × obstruction_multiplier) and returns a non-zero leak fraction for cuff leak. The leak fraction is applied to reduce delivered tidal volume to patient tidal volume via `_circuit_vt_correction()`. Partial obstruction increases the internal resistive pressure component (pres_peak_cmH2O) without changing the displayed airway pressure, which remains servo-controlled to PEEP + PS.

**SBT temporal sequence.** `generate_sbt_sequence()` implements a multi-phase spontaneous breathing trial scenario. A baseline phase runs at the patient's clinical support level. A trial phase reduces PS to the set trial level and runs a series of assessment windows. Each window computes an RRSBI (rapid shallow breathing index) and checks failure criteria. The function returns a structured dict with scenario_type, outcome (pass or fail), time_to_failure_min if applicable, per-window metrics, and the full RRSBI trajectory.

**Bugs identified and fixed.** Seven bugs were found and corrected during implementation, smoke testing, and unit testing.

The first and most significant was the pressure display bug in the inspiratory ODE loop. The original implementation computed pao_now as the sum of Rohrer resistive pressure, elastic pressure, and total PEEP — which correctly describes alveolar pressure but incorrectly represents displayed airway pressure in a servo-controlled mode. The displayed Pao was producing a spike-then-decay waveform instead of the correct rectangular plateau. Fixed by setting pao_now = P_vent inside the inspiratory loop.

The second was the same bug in the expiratory loop. During passive expiration, P_list.append() was called with pres + pel + tpeep, which decays exponentially from the end-inspiratory alveolar pressure down toward PEEP. This produced an exponential tail on the expiratory pressure trace rather than the correct flat PEEP baseline. Fixed by appending peep_e instead.

The third was the same bug in the trailing-breath block — the final expiration segment appended at the end of generate_breath_cycles() after the main event loop. This block had never been updated when the in-loop fix was applied and was still using the alveolar-pressure reconstruction, producing a spike-and-decay artifact on the last breath of every trace. Fixed by appending peep_e in the trailing block.

The fourth through seventh were test failures exposed by the pressure display fix. TestPressureDecomposition.test_pao_equals_sum_of_components asserted the old invariant (pressure = pres + pel + tpeep) which is intentionally false after the servo fix — updated to assert servo target bounds and array well-formedness instead. TestPressureDecomposition.test_pressure_decomposition_copd had the same issue plus an overly tight lower bound that did not account for the transient negative Rohrer pressure during flow reversal at the end of delayed-cycling COPD breaths — updated with a looser lower bound and a comment explaining the physical mechanism. TestETTComplications.test_partial_obstruction_elevates_ppeak asserted that partial obstruction increases ppeak_cmH2O, which is not true in a servo-controlled mode (the displayed Ppeak = PEEP + PS always) — updated to assert that pres_peak_cmH2O increases instead, with an updated docstring explaining the correct PSV physics.

---

## Proposed Change

Implement a dedicated PSV generator module that correctly models the PSV control loop, implements patient-triggered event-driven breath cycling, models the patient's inspiratory effort as a co-driver of gas delivery, servo-controls displayed airway pressure to PEEP + PS, implements the flow-cycle criterion for inspiratory termination, produces all seven dyssynchrony subtypes, implements ETT complications, provides the SBT temporal sequence, and satisfies the shared interface contract extended with PSV-specific metrics. The generator must correctly distinguish displayed airway pressure from internal physiological pressure at every timestep.

---

## Acceptance Criteria

- `generate_breath_cycles()` returns a dict containing the four core waveform arrays (time, pressure, flow, volume), the three pressure decomposition arrays (pressure_resistive, pressure_elastic, pressure_total_peep), all fifteen scalar metric keys, both validity keys, and the breath_dyssynchrony_labels list of length n_cycles
- The pressure array during the inspiratory plateau equals PEEP + PS within 0.5 cmH₂O — the ventilator holds servo-controlled pressure constant at the set target
- The pressure array during passive expiration equals PEEP_e within 0.5 cmH₂O — the expiratory baseline is the set extrinsic PEEP, not the alveolar pressure
- The pressure_resistive, pressure_elastic, and pressure_total_peep arrays do not sum to the pressure array — these are physiological components, not the displayed servo output
- Lower compliance reduces delivered tidal volume at the same pressure support level, confirming that Vt is a dependent variable in PSV
- Higher ETS (flow cycle threshold) reduces inspiratory time and tidal volume at the same compliance and resistance, confirming the flow-cycle mechanism
- COPD parameters produce auto-PEEP greater than 0.5 cmH₂O after sufficient cycles, and at least some breaths are classified as ineffective_trigger
- Cuff leak reduces patient_vt_ml below delivered_vt_ml, with the gap proportional to the leak fraction
- Partial obstruction increases pres_peak_cmH2O relative to the unobstructed case — the displayed ppeak_cmH2O does not change because it is servo-controlled
- The RRSBI trajectory in generate_sbt_sequence() produces at least two distinct window measurements, and the outcome is one of pass or fail
- All unit tests in tests/test_psv_generator.py pass, including the updated TestPressureDecomposition and TestETTComplications classes that reflect the servo pressure model
- The smoke test confirms Pao = servo target (PEEP + PS) during inspiration and the internal decomposition arrays are finite and bounded

---

## Files Likely to Be Touched

- **Created:** `generator/psv_generator.py` — PSV waveform generator implementing the event-driven control loop, twelve physiological modeling features, servo-controlled pressure display, trigger notch, seven dyssynchrony subtypes, ETT complications, SBT temporal sequence, fifteen scalar metrics, validity filter, and generate_dataset() sweep function
- **Created:** `tests/test_psv_generator.py` — unit tests across thirteen classes: TestInterfaceContract, TestPhysiologicalPlausibility, TestPSVWaveformShape, TestDyssynchrony, TestETTComplications, TestSBTTemporalSequence, TestPressureDecomposition, TestMultiCompartmentMechanics, TestPhysiologicalDirections, TestValidityFilter, TestDatasetGeneration, TestParameterGrid, and TestTriggerMechanism
- **Created:** `generator/conditions.py` — condition preset dictionaries for all seven tiers, each specifying compliance, resistance, PEEP, pressure support, Pmus parameters, effort rate, effort duration, flow cycle threshold, trigger threshold, stress index, compartment profile, and PEEP recruitment slope
- **Created:** `generate_psv_dataset_thinned.py` — batch script sweeping the thinned parameter grid across all seven condition tiers, writing waveform data, manifest, and generation log

---

## Status

**Complete**

`generator/psv_generator.py` is implemented and all unit tests pass. The thinned dataset has been generated and exported. The generator is ready for the validation phase.
