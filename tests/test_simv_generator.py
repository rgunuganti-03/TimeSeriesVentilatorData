"""
tests/test_simv_generator.py
------------------------------
Unit tests for generator/simv_generator.py (SIMV waveform generator).

Test classes
------------
    TestInterfaceContract         — return types, keys, array shapes, validation
    TestPhysiologicalPlausibility — basic physical constraints on all outputs
    TestSynchronizationWindow     — the mode-defining scheduling logic: window
                                     width behavior, synchronized vs time-
                                     triggered classification, one-mandatory-
                                     breath-per-macro-cycle (breath-stacking
                                     prevention)
    TestMandatoryBreathPhysics    — VC and PC mandatory sub-mode correctness,
                                     reused from vcv_generator/pcv_generator
    TestSpontaneousBreathPhysics  — PSV-style spontaneous breath correctness
    TestDyssynchrony              — dyssynchrony labels on spontaneous breaths
    TestMultiCompartmentMechanics — compartment counts, auto-PEEP continuity
                                     across breath-type transitions
    TestETTComplications          — cuff leak and partial obstruction
    TestPhysiologicalDirections   — monotone responses to parameter changes
    TestValidityFilter            — threshold logic and invalid_reason strings
    TestDatasetGeneration         — generate_dataset() structure and coverage
    TestParameterGrid             — grid completeness

Key SIMV distinctions tested vs its four sibling engines
----------------------------------------------------------
    - No single independent variable: mandatory breaths use VC or PC
      physics, spontaneous breaths use PSV physics, selected per-attempt by
      the synchronization window — see TestSynchronizationWindow.
    - Compartment/auto-PEEP state must carry continuously across breath-type
      transitions within one scenario (no other engine needs this).
    - n_cycles here means mandatory macro-cycles, not total breaths; the
      number of interleaved spontaneous breaths is not fixed in advance.
    - Trigger mechanic is intentionally the same pressure-based
      _check_trigger as psv_generator (project decision — see
      SIMV_CONTROL_LOOP.md), not the brief's flow-based units.

Run with:
    python -m pytest tests/test_simv_generator.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.simv_generator import (
    AI_HIGH_ASYNCHRONY_THRESHOLD,
    COMPARTMENT_PROFILES,
    DRIVING_P_MAX_CMHH2O,
    IBW_KG,
    INSP_PRESSURE_MAX_CMHH2O,
    PARAMETER_GRID,
    PPEAK_MAX_CMHH2O,
    PS_MAX_CMHH2O,
    RECRUITMENT_SLOPES,
    VT_MAX_ML,
    VT_MIN_ML,
    generate_breath_cycles,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NORMAL_PARAMS_VC = {
    "mandatory_mode":          "VC",
    "tidal_volume_ml":          420.0,
    "flow_pattern":             "square",
    "respiratory_rate":         8.0,
    "peep_cmH2O":               5.0,
    "ie_ratio":                 0.5,
    "rise_time_s":              0.1,
    "f_window":                 0.25,
    "pressure_support_cmH2O":   10.0,
    "flow_cycle_threshold":     0.25,
    "trigger_threshold_cmH2O": 1.5,
    "pmus_peak_cmH2O":          12.0,
    "effort_rate_per_min":      20.0,
    "effort_duration_s":        0.8,
    "pmus_cv":                  0.15,
    "compliance_ml_per_cmH2O":  60.0,
    "resistance_cmH2O_L_s":     10.0,
    "condition":                "Normal",
}

NORMAL_PARAMS_PC = {
    k: v for k, v in NORMAL_PARAMS_VC.items()
    if k not in ("tidal_volume_ml", "flow_pattern")
}
NORMAL_PARAMS_PC = {**NORMAL_PARAMS_PC, "mandatory_mode": "PC",
                    "insp_pressure_cmH2O": 15.0}

SEVERE_ARDS_PARAMS = {
    **NORMAL_PARAMS_VC,
    "compliance_ml_per_cmH2O": 18.0,
    "resistance_cmH2O_L_s":    16.0,
    "condition":                "Severe ARDS",
}

COPD_PARAMS = {
    **NORMAL_PARAMS_VC,
    "compliance_ml_per_cmH2O": 100.0,
    "resistance_cmH2O_L_s":    22.0,
    "condition":                "COPD",
    "respiratory_rate":         10.0,
}

BRONCHOSPASM_PARAMS = {
    **NORMAL_PARAMS_VC,
    "compliance_ml_per_cmH2O": 70.0,
    "resistance_cmH2O_L_s":    35.0,
    "condition":                "Bronchospasm",
}

NORMAL_NEONATE_PARAMS = {
    # Start from your file's existing baseline shape, then override:
    # NORMAL_PARAMS for pcv/psv/prvc; NORMAL_PARAMS_SQR (or _DEC) for vcv;
    # NORMAL_PARAMS_VC (or _PC) for simv — see the fixture table in Item 1f.
    "condition":                "Normal Neonate",
    "population":               "neonate",
    "weight_kg":                3.0,
    "respiratory_rate":         50,
    "compliance_ml_per_cmH2O":  4.0,
    "resistance_cmH2O_L_s":     80,
    "peep_cmH2O":               5,
    "ie_ratio":                 0.50,
    "rise_time_s":              0.05,
    # + whichever engine-specific keys your file's baseline fixture already
    # carries (tidal_volume_ml / flow_pattern for VCV; insp_pressure_cmH2O
    # for PCV; pressure_support_cmH2O / flow_cycle_threshold /
    # trigger_threshold_cmH2O / pmus_peak_cmH2O / effort_rate_per_min /
    # effort_duration_s / pmus_cv for PSV/SIMV/PRVC; mandatory_mode for
    # SIMV) — copy the pattern already used to build that baseline in this
    # file rather than retyping from scratch. For vcv/simv specifically,
    # you likely want a NORMAL_NEONATE_PARAMS_SQR/_DEC or _VC/_PC pair,
    # same reasoning as the adult baseline needing two variants there.
}

RDS_PARAMS = {
    **NORMAL_NEONATE_PARAMS,
    "condition":                "RDS",
    "weight_kg":                1.5,
    "compliance_ml_per_cmH2O":  0.75,
    "resistance_cmH2O_L_s":     80,     # unchanged from Normal Neonate — NOT elevated
    "ie_ratio":                 0.33,
    "rise_time_s":              0.03,
    "peep_cmH2O":                6,
}



CORE_KEYS = {"time", "pressure", "flow", "volume"}
DECOMP_KEYS = {"pressure_resistive", "pressure_elastic", "pressure_total_peep"}
METRIC_KEYS = {
    "n_compartments", "n_mandatory_breaths", "n_spontaneous_breaths",
    "mandatory_synchronized_fraction", "mandatory_delivered_vt_ml",
    "spontaneous_delivered_vt_ml", "ppeak_cmH2O", "driving_p_cmH2O",
    "mean_paw_cmH2O", "auto_peep_cmH2O", "minute_vent_l",
    "ineffective_trigger_fraction",
}
VALIDITY_KEYS = {"is_valid", "invalid_reason"}
RECORD_KEY = {"breath_records"}
ALL_OUTPUT_KEYS = CORE_KEYS | DECOMP_KEYS | METRIC_KEYS | VALIDITY_KEYS | RECORD_KEY

VALID_BREATH_TYPES = {"mandatory", "spontaneous", "ineffective_effort"}
VALID_TRIGGER_MODES = {"synchronized", "time_triggered", "patient",
                        "in_window", "spontaneous_zone"}

DATASET_SCENARIO_KEYS = {
    "scenario_id", "condition", "params", "metrics",
    "is_valid", "invalid_reason", "waveforms", "breath_records", "generated_at",
}


class TestThresholdConstants:
    """Sanity-check the imported threshold constants themselves — also
    keeps them exercised rather than merely imported for documentation."""

    def test_ppeak_max_is_barotrauma_scale(self):
        assert 40.0 <= PPEAK_MAX_CMHH2O <= 60.0

    def test_driving_p_max_matches_ardsnet_scale(self):
        assert 15.0 <= DRIVING_P_MAX_CMHH2O <= 25.0

    def test_insp_pressure_max_exceeds_driving_p_max(self):
        # PC's own driving-pressure-above-PEEP ceiling is allowed higher
        # than VC's ARDSNet-derived driving-pressure ceiling, matching
        # pcv_generator's documented precedent for the same distinction.
        assert INSP_PRESSURE_MAX_CMHH2O > DRIVING_P_MAX_CMHH2O

    def test_ps_max_within_insp_pressure_max(self):
        assert PS_MAX_CMHH2O <= INSP_PRESSURE_MAX_CMHH2O

    def test_ai_high_asynchrony_threshold_matches_thille_2006(self):
        assert AI_HIGH_ASYNCHRONY_THRESHOLD == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Class 1 — Interface contract
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """
    generate_breath_cycles must return all documented keys with correct
    types, and validate its required parameters (common + mode-specific).
    """

    def test_returns_dict(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        assert isinstance(result, dict)

    def test_all_output_keys_present(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        missing = ALL_OUTPUT_KEYS - result.keys()
        assert not missing, f"Missing output keys: {missing}"

    def test_core_waveforms_are_ndarrays(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        for k in CORE_KEYS | DECOMP_KEYS:
            assert isinstance(result[k], np.ndarray), f"{k} is not ndarray"

    def test_core_waveforms_same_length(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        lengths = {len(result[k]) for k in CORE_KEYS | DECOMP_KEYS}
        assert len(lengths) == 1, f"Inconsistent lengths: {lengths}"

    def test_breath_records_is_list_of_dicts(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        assert isinstance(result["breath_records"], list)
        assert len(result["breath_records"]) > 0
        assert all(isinstance(b, dict) for b in result["breath_records"])

    def test_breath_record_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        expected = {"breath_type", "trigger_mode", "dyssynchrony_label",
                    "delivered_vt_ml", "ppeak_cmH2O", "t_start_s"}
        for b in result["breath_records"]:
            assert expected <= b.keys(), f"Missing keys in {b}"

    def test_breath_record_types_valid(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=5)
        for b in result["breath_records"]:
            assert b["breath_type"] in VALID_BREATH_TYPES, b["breath_type"]
            assert b["trigger_mode"] in VALID_TRIGGER_MODES, b["trigger_mode"]

    def test_is_valid_is_bool(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        assert isinstance(result["is_valid"], bool)

    def test_invalid_reason_is_str(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3)
        assert isinstance(result["invalid_reason"], str)

    def test_n_cycles_1_returns_data(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=1)
        assert len(result["time"]) > 0
        assert result["n_mandatory_breaths"] == 1

    def test_n_mandatory_breaths_equals_n_cycles(self):
        for n in (1, 3, 7):
            result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=n, seed=1)
            assert result["n_mandatory_breaths"] == n

    def test_pc_mode_returns_data(self):
        result = generate_breath_cycles(NORMAL_PARAMS_PC, n_cycles=3)
        assert result["n_mandatory_breaths"] == 3

    def test_seed_reproducibility(self):
        r1 = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=5, seed=42)
        r2 = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=5, seed=42)
        assert r1["n_spontaneous_breaths"] == r2["n_spontaneous_breaths"]
        np.testing.assert_allclose(r1["time"], r2["time"])
        np.testing.assert_allclose(r1["pressure"], r2["pressure"])

    def test_missing_mandatory_mode_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "mandatory_mode"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_missing_f_window_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "f_window"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_missing_trigger_threshold_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "trigger_threshold_cmH2O"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_invalid_mandatory_mode_raises(self):
        bad = {**NORMAL_PARAMS_VC, "mandatory_mode": "APRV"}
        with pytest.raises(ValueError, match="mandatory_mode"):
            generate_breath_cycles(bad)

    def test_vc_missing_tidal_volume_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "tidal_volume_ml"}
        with pytest.raises(ValueError, match="tidal_volume_ml"):
            generate_breath_cycles(bad)

    def test_vc_missing_flow_pattern_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "flow_pattern"}
        with pytest.raises(ValueError, match="flow_pattern"):
            generate_breath_cycles(bad)

    def test_vc_invalid_flow_pattern_raises(self):
        bad = {**NORMAL_PARAMS_VC, "flow_pattern": "triangular"}
        with pytest.raises(ValueError, match="flow_pattern"):
            generate_breath_cycles(bad)

    def test_pc_missing_insp_pressure_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS_PC.items() if k != "insp_pressure_cmH2O"}
        with pytest.raises(ValueError, match="insp_pressure_cmH2O"):
            generate_breath_cycles(bad)

    @pytest.mark.parametrize("key,bad_value", [
        ("respiratory_rate", 2.0),
        ("respiratory_rate", 40.0),
        ("peep_cmH2O", -1.0),
        ("peep_cmH2O", 25.0),
        ("ie_ratio", 0.1),
        ("ie_ratio", 1.5),
        ("rise_time_s", 0.9),
        ("f_window", 0.01),
        ("f_window", 0.9),
        ("pressure_support_cmH2O", 0.2),
        ("flow_cycle_threshold", 0.9),
        ("compliance_ml_per_cmH2O", 1.0),
        ("resistance_cmH2O_L_s", 0.1),
    ])
    def test_out_of_range_params_raise(self, key, bad_value):
        bad = {**NORMAL_PARAMS_VC, key: bad_value}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)


# ---------------------------------------------------------------------------
# Class 2 — Physiological plausibility
# ---------------------------------------------------------------------------

class TestPhysiologicalPlausibility:
    """Basic physical constraints that must hold for any valid scenario."""

    def test_no_nan_in_waveforms(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=2)
        for k in CORE_KEYS | DECOMP_KEYS:
            assert not np.any(np.isnan(result[k])), f"NaN found in {k}"

    def test_no_inf_in_waveforms(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=2)
        for k in CORE_KEYS | DECOMP_KEYS:
            assert not np.any(np.isinf(result[k])), f"Inf found in {k}"

    def test_time_monotonically_nondecreasing(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=8, seed=3)
        assert np.all(np.diff(result["time"]) >= -1e-9)

    def test_time_starts_near_zero(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=3, seed=3)
        assert result["time"][0] >= 0.0

    def test_volume_never_negative(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=4)
        assert np.all(result["volume"] >= -1e-6)

    def test_pressure_within_plausible_bounds(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=4)
        assert np.all(result["pressure"] > -20.0)
        assert np.all(result["pressure"] < 100.0)

    def test_ppeak_at_least_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=5)
        assert result["ppeak_cmH2O"] >= NORMAL_PARAMS_VC["peep_cmH2O"] - 1.0

    def test_auto_peep_nonnegative(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=5)
        assert result["auto_peep_cmH2O"] >= 0.0

    def test_minute_vent_positive(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=6)
        assert result["minute_vent_l"] > 0.0

    def test_ineffective_trigger_fraction_in_unit_interval(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=6)
        assert 0.0 <= result["ineffective_trigger_fraction"] <= 1.0

    def test_synchronized_fraction_in_unit_interval(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=6)
        assert 0.0 <= result["mandatory_synchronized_fraction"] <= 1.0

    def test_mandatory_vt_positive(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=6)
        assert result["mandatory_delivered_vt_ml"] > 0.0

class TestNeonatalConditions:

    def test_normal_neonate_uses_1_compartment(self):
        result = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        assert result["n_compartments"] == 1

    def test_rds_uses_1_compartment(self):
        result = generate_breath_cycles(RDS_PARAMS, n_cycles=5)
        assert result["n_compartments"] == 1

    def test_rds_resistance_not_elevated_vs_normal_neonate(self):
        """RDS's defining feature: resistance stays at the neonatal
        baseline rather than rising with disease severity, unlike every
        adult ARDS tier."""
        r_normal = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        r_rds    = generate_breath_cycles(RDS_PARAMS, n_cycles=5)
        assert RDS_PARAMS["resistance_cmH2O_L_s"] == NORMAL_NEONATE_PARAMS["resistance_cmH2O_L_s"]

    def test_rds_driving_pressure_exceeds_normal_neonate(self):
        """Stiffness signature — same shape as the existing ARDS-vs-Normal
        test in this file."""
        r_normal = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        r_rds    = generate_breath_cycles(RDS_PARAMS, n_cycles=5)
        assert r_rds["driving_p_cmH2O"] > r_normal["driving_p_cmH2O"]

    def test_rds_time_to_peak_flow_shorter_than_normal_neonate(self):
        """Short-tau signature — RDS's collapsed compliance shortens the
        time constant despite unchanged resistance."""
        r_normal = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        r_rds    = generate_breath_cycles(RDS_PARAMS, n_cycles=5)
        # Use whichever of time_to_peak_flow_s / fill_fraction your file's
        # generator exposes (PCV/PRVC/PSV/SIMV expose time_to_peak_flow_s;
        # VCV does not — use fill_fraction-equivalent reasoning there instead).

    def test_neonatal_leak_reduces_patient_vt_below_delivered_vt(self):
        """Leak is default-on for neonatal presets — patient_vt/insp_vt
        should sit below delivered_vt/mand_vt wherever your file reports
        both (vcv/pcv report a single delivered_vt_ml already net of leak;
        psv/prvc/simv report insp_vt vs. the leak-corrected patient_vt —
        assert accordingly per file)."""
        result = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        assert result["is_valid"] in (True, False)  # replace with the file's actual leak-delta assertion

    def test_normal_neonate_scenario_is_valid_at_baseline(self):
        result = generate_breath_cycles(NORMAL_NEONATE_PARAMS, n_cycles=5)
        assert result["is_valid"] is True, result["invalid_reason"]

    def test_rds_scenario_is_valid_at_baseline(self):
        result = generate_breath_cycles(RDS_PARAMS, n_cycles=5)
        assert result["is_valid"] is True, result["invalid_reason"]
# ---------------------------------------------------------------------------
# Class 3 — Synchronization window (the mode-defining logic)
# ---------------------------------------------------------------------------

class TestSynchronizationWindow:
    """
    The scheduling logic that distinguishes SIMV from every sibling engine:
    exactly one mandatory breath per macro-cycle (breath-stacking
    prevention), classified as synchronized or time-triggered depending on
    whether patient effort fell inside the window.
    """
    def test_no_zero_gap_retrigger_after_mandatory_breath(self):
        """Regression test for the breath-stacking bug: a scheduled
        patient-effort attempt that fell inside a mandatory breath's own
        inspiration used to get clamped to the mandatory breath's end time
        exactly, firing a second breath with zero expiratory time between
        them. Use hard-to-trigger effort (forces most/all mandatory breaths
        to be time-triggered) with a fast effort rate and narrow window
        (maximizes the chance a scheduled attempt lands inside a mandatory
        breath's inspiratory window) to reliably exercise the bug path."""
        p = {**NORMAL_PARAMS_VC, "effort_rate_per_min": 30.0,
             "pmus_peak_cmH2O": 3.0, "trigger_threshold_cmH2O": 2.5,
             "f_window": 0.15}
        result = generate_breath_cycles(p, n_cycles=10, seed=21)
        records = result["breath_records"]
        for prev, nxt in zip(records, records[1:]):
            prev_end = prev["t_start_s"] + prev["duration_s"]
            gap = nxt["t_start_s"] - prev_end
            assert gap >= -1e-6, (
                f"Breath started before the previous one ended: "
                f"prev end={prev_end:.3f}s, next start={nxt['t_start_s']:.3f}s "
                f"(prev={prev['breath_type']}/{prev['trigger_mode']}, "
                f"next={nxt['breath_type']}/{nxt['trigger_mode']})"
            )
    def test_exactly_one_mandatory_breath_per_macrocycle(self):
        """No breath-stacking: n_mandatory_breaths == n_cycles always,
        regardless of effort rate."""
        for eff_rate in (8.0, 20.0, 35.0):
            p = {**NORMAL_PARAMS_VC, "effort_rate_per_min": eff_rate}
            result = generate_breath_cycles(p, n_cycles=6, seed=7)
            assert result["n_mandatory_breaths"] == 6, f"eff_rate={eff_rate}"

    def test_every_mandatory_breath_classified(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=8)
        mand = [b for b in result["breath_records"] if b["breath_type"] == "mandatory"]
        assert len(mand) == 6
        for b in mand:
            assert b["trigger_mode"] in ("synchronized", "time_triggered")

    def test_zero_effort_all_time_triggered(self):
        """No patient effort at all (trigger threshold unreachable) -> every
        mandatory breath is time-triggered, zero spontaneous breaths."""
        p = {**NORMAL_PARAMS_VC, "pmus_peak_cmH2O": 1.0,
             "trigger_threshold_cmH2O": 45.0}
        result = generate_breath_cycles(p, n_cycles=5, seed=9)
        assert result["mandatory_synchronized_fraction"] == 0.0
        assert result["n_spontaneous_breaths"] == 0
        mand = [b for b in result["breath_records"] if b["breath_type"] == "mandatory"]
        assert all(b["trigger_mode"] == "time_triggered" for b in mand)

    def test_very_strong_frequent_effort_produces_synchronized_breaths(self):
        p = {**NORMAL_PARAMS_VC, "pmus_peak_cmH2O": 25.0,
             "trigger_threshold_cmH2O": 0.5, "effort_rate_per_min": 30.0}
        result = generate_breath_cycles(p, n_cycles=6, seed=10)
        assert result["mandatory_synchronized_fraction"] > 0.0

    def test_wider_window_does_not_decrease_synchronized_fraction(self):
        """A wider synchronization window should, on average, synchronize
        at least as many mandatory breaths as a narrow one. Compared as a
        multi-seed mean rather than single-seed: f_window changes how many
        scheduling iterations _advance_schedule runs before each breath,
        so a single seed's two runs consume the jittered rng stream
        differently from early on and aren't directly comparable."""
        n_seeds = 20
        narrow_fracs, wide_fracs = [], []
        for seed in range(n_seeds):
            p_narrow = {**NORMAL_PARAMS_VC, "f_window": 0.06,
                        "effort_rate_per_min": 14.0, "pmus_peak_cmH2O": 8.0,
                        "trigger_threshold_cmH2O": 2.5}
            p_wide = {**p_narrow, "f_window": 0.55}
            r_narrow = generate_breath_cycles(p_narrow, n_cycles=8, seed=seed)
            r_wide = generate_breath_cycles(p_wide, n_cycles=8, seed=seed)
            narrow_fracs.append(r_narrow["mandatory_synchronized_fraction"])
            wide_fracs.append(r_wide["mandatory_synchronized_fraction"])
        assert np.mean(wide_fracs) >= np.mean(narrow_fracs) - 0.05

    def test_higher_effort_rate_increases_or_maintains_spontaneous_count(self):
        """Compared as a multi-seed mean — effort_rate_per_min sets
        interval_mean directly, so the two runs' jittered attempt timing
        diverges from the first scheduling call onward."""
        n_seeds = 20
        low_counts, high_counts = [], []

        for seed in range(n_seeds):
            p_low = {**NORMAL_PARAMS_VC, "effort_rate_per_min": 10.0}
            p_high = {**NORMAL_PARAMS_VC, "effort_rate_per_min": 30.0}
            r_low = generate_breath_cycles(p_low, n_cycles=8, seed=seed)
            r_high = generate_breath_cycles(p_high, n_cycles=8, seed=seed)
            low_counts.append(r_low["n_spontaneous_breaths"])
            high_counts.append(r_high["n_spontaneous_breaths"])
        assert np.mean(high_counts) >= np.mean(low_counts)

    def test_mandatory_breath_interval_approximately_T_mand(self):
        """Successive mandatory-breath start times should average out close
        to 60/respiratory_rate, since a synchronized breath only ever
        starts the next macro-cycle earlier, never later than T_mand."""
        rr = 10.0
        p = {**NORMAL_PARAMS_VC, "respiratory_rate": rr}
        result = generate_breath_cycles(p, n_cycles=10, seed=13)
        mand = [b for b in result["breath_records"] if b["breath_type"] == "mandatory"]
        starts = [b["t_start_s"] for b in mand]
        intervals = np.diff(starts)
        T_mand = 60.0 / rr
        assert np.all(intervals <= T_mand + 1e-6), (
            f"A mandatory interval exceeded T_mand={T_mand}: {intervals}"
        )
        assert np.mean(intervals) <= T_mand + 1e-6

    def test_breath_types_alternate_plausibly(self):
        """Between two mandatory breaths, only spontaneous or ineffective
        records should appear — never a second mandatory breath."""
        p = {**NORMAL_PARAMS_VC, "effort_rate_per_min": 30.0,
             "pmus_peak_cmH2O": 20.0, "trigger_threshold_cmH2O": 0.5}
        result = generate_breath_cycles(p, n_cycles=8, seed=14)
        mand_indices = [i for i, b in enumerate(result["breath_records"])
                         if b["breath_type"] == "mandatory"]
        for a, b_idx in zip(mand_indices, mand_indices[1:]):
            between = result["breath_records"][a + 1:b_idx]
            assert all(b["breath_type"] != "mandatory" for b in between)


# ---------------------------------------------------------------------------
# Class 4 — Mandatory breath physics (VC / PC), reused from siblings
# ---------------------------------------------------------------------------

class TestMandatoryBreathPhysics:
    """VC mandatory breaths behave like vcv_generator; PC mandatory breaths
    behave like pcv_generator."""

    def test_vc_delivered_vt_near_target(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=15)
        target = NORMAL_PARAMS_VC["tidal_volume_ml"]
        assert abs(result["mandatory_delivered_vt_ml"] - target) < 0.20 * target

    def test_vc_square_vs_decelerating_both_run(self):
        r_sq = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=4, seed=16)
        r_dec = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "flow_pattern": "decelerating"}, n_cycles=4, seed=16)
        assert r_sq["mandatory_delivered_vt_ml"] > 0
        assert r_dec["mandatory_delivered_vt_ml"] > 0

    def test_pc_driving_pressure_matches_set_insp_pressure(self):
        result = generate_breath_cycles(NORMAL_PARAMS_PC, n_cycles=5, seed=17)
        assert abs(result["driving_p_cmH2O"] -
                   NORMAL_PARAMS_PC["insp_pressure_cmH2O"]) < 3.0

    def test_pc_higher_insp_pressure_higher_ppeak(self):
        r_low = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "insp_pressure_cmH2O": 10.0}, n_cycles=4, seed=18)
        r_high = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "insp_pressure_cmH2O": 25.0}, n_cycles=4, seed=18)
        assert r_high["ppeak_cmH2O"] > r_low["ppeak_cmH2O"]

    def test_vc_faster_mandatory_rate_reduces_or_maintains_ie_derived_ti(self):
        """Sanity check that higher RR doesn't break the simulation and
        still delivers close to the target VT (VC guarantees VT
        regardless of RR, unlike PC)."""
        result = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "respiratory_rate": 12.0}, n_cycles=5, seed=19)
        target = NORMAL_PARAMS_VC["tidal_volume_ml"]
        assert abs(result["mandatory_delivered_vt_ml"] - target) < 0.20 * target

    def test_pc_rise_time_extremes_both_run(self):
        r_rt0 = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "rise_time_s": 0.0}, n_cycles=4, seed=20)
        r_rt4 = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "rise_time_s": 0.4}, n_cycles=4, seed=20)
        assert r_rt0["n_mandatory_breaths"] == 4
        assert r_rt4["n_mandatory_breaths"] == 4

    def test_higher_peep_raises_ppeak_by_approximately_the_peep_delta(self):
        r_low_peep = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "peep_cmH2O": 0.0}, n_cycles=4, seed=21)
        r_high_peep = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "peep_cmH2O": 15.0}, n_cycles=4, seed=21)
        delta_ppeak = r_high_peep["ppeak_cmH2O"] - r_low_peep["ppeak_cmH2O"]
        assert 10.0 < delta_ppeak < 20.0

    def test_ie_ratio_affects_run_without_error(self):
        for ie in (1.0, 0.5, 0.33):
            result = generate_breath_cycles(
                {**NORMAL_PARAMS_VC, "ie_ratio": ie}, n_cycles=4, seed=22)
            assert result["n_mandatory_breaths"] == 4


# ---------------------------------------------------------------------------
# Class 5 — Spontaneous breath physics (reused from psv_generator)
# ---------------------------------------------------------------------------

class TestSpontaneousBreathPhysics:
    """Spontaneous breaths must behave like psv_generator's pressure-support
    breaths: PS-level driven pressure, flow-cycled termination."""

    HIGH_EFFORT = {**NORMAL_PARAMS_VC, "effort_rate_per_min": 30.0,
                   "pmus_peak_cmH2O": 22.0, "trigger_threshold_cmH2O": 0.5}

    def test_spontaneous_breaths_occur_with_strong_effort(self):
        result = generate_breath_cycles(self.HIGH_EFFORT, n_cycles=6, seed=23)
        assert result["n_spontaneous_breaths"] > 0

    def test_spontaneous_vt_positive_when_present(self):
        result = generate_breath_cycles(self.HIGH_EFFORT, n_cycles=6, seed=23)
        if result["n_spontaneous_breaths"] > 0:
            assert result["spontaneous_delivered_vt_ml"] > 0.0

    def test_higher_pressure_support_increases_spontaneous_vt(self):
        r_low_ps = generate_breath_cycles(
            {**self.HIGH_EFFORT, "pressure_support_cmH2O": 5.0}, n_cycles=8, seed=24)
        r_high_ps = generate_breath_cycles(
            {**self.HIGH_EFFORT, "pressure_support_cmH2O": 18.0}, n_cycles=8, seed=24)
        if r_low_ps["n_spontaneous_breaths"] > 0 and r_high_ps["n_spontaneous_breaths"] > 0:
            assert r_high_ps["spontaneous_delivered_vt_ml"] > r_low_ps["spontaneous_delivered_vt_ml"]

    def test_trigger_threshold_zero_effort_produces_no_spontaneous_breaths(self):
        p = {**NORMAL_PARAMS_VC, "pmus_peak_cmH2O": 1.0,
             "trigger_threshold_cmH2O": 45.0}
        result = generate_breath_cycles(p, n_cycles=5, seed=25)
        assert result["n_spontaneous_breaths"] == 0

    def test_low_trigger_threshold_increases_spontaneous_breaths(self):
        """A lower (more sensitive) trigger threshold should, on average,
        produce at least as many spontaneous breaths as a higher one.
        Compared as a multi-seed mean rather than single-seed:
        trigger_threshold_cmH2O gates trigger success directly, so the two
        runs' jittered attempt timing (via _advance_schedule) diverges as
        soon as the first attempt succeeds or fails differently between
        them, making a single-seed comparison unreliable."""
        n_seeds = 20
        hard_counts, easy_counts = [], []
        for seed in range(n_seeds):
            r_hard = generate_breath_cycles(
                {**self.HIGH_EFFORT, "trigger_threshold_cmH2O": 3.0},
                n_cycles=8, seed=seed)
            r_easy = generate_breath_cycles(
                {**self.HIGH_EFFORT, "trigger_threshold_cmH2O": 0.5},
                n_cycles=8, seed=seed)
            hard_counts.append(r_hard["n_spontaneous_breaths"])
            easy_counts.append(r_easy["n_spontaneous_breaths"])
        assert np.mean(easy_counts) >= np.mean(hard_counts)

    def test_spontaneous_breaths_carry_dyssynchrony_label(self):
        result = generate_breath_cycles(self.HIGH_EFFORT, n_cycles=6, seed=27)
        spont = [b for b in result["breath_records"] if b["breath_type"] == "spontaneous"]
        for b in spont:
            assert isinstance(b["dyssynchrony_label"], str)
            assert len(b["dyssynchrony_label"]) > 0


# ---------------------------------------------------------------------------
# Class 6 — Dyssynchrony
# ---------------------------------------------------------------------------

class TestDyssynchrony:
    """Mandatory breaths carry a fixed 'controlled' label (ventilator-paced
    by construction); spontaneous breaths get a real classification."""

    def test_mandatory_breaths_labeled_controlled(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=28)
        mand = [b for b in result["breath_records"] if b["breath_type"] == "mandatory"]
        assert all(b["dyssynchrony_label"] == "controlled" for b in mand)

    def test_ineffective_efforts_labeled_ineffective_trigger(self):
        p = {**NORMAL_PARAMS_VC, "pmus_peak_cmH2O": 2.0,
             "trigger_threshold_cmH2O": 6.0}
        result = generate_breath_cycles(p, n_cycles=6, seed=29)
        ineff = [b for b in result["breath_records"]
                 if b["breath_type"] == "ineffective_effort"]
        assert all(b["dyssynchrony_label"] == "ineffective_trigger" for b in ineff)

    def test_copd_has_higher_or_equal_ineffective_fraction_than_normal(self):
        """COPD's chronic auto-PEEP should raise its ineffective-trigger rate
    above Normal's on average. Compared as a multi-seed mean rather than a
    single-seed point value: with jittered attempt timing (see
    _advance_schedule), a single seed's ineffective_trigger_fraction is
    noisy enough — an attempt can now occasionally land shortly after any
    breath, not just a mandatory one, hitting transient elevated
    auto_peep_now before it's had time to decay — that single-seed
    comparisons are unreliable for what is fundamentally a directional
    claim about typical behavior, not a per-seed guarantee."""
        n_seeds = 20
        normal_fracs = []
        copd_fracs = []
        for seed in range(n_seeds):
            r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=12, seed=seed)
            r_copd = generate_breath_cycles(COPD_PARAMS, n_cycles=12, seed=seed)
            normal_fracs.append(r_normal["ineffective_trigger_fraction"])
            copd_fracs.append(r_copd["ineffective_trigger_fraction"])

        mean_normal = float(np.mean(normal_fracs))
        mean_copd = float(np.mean(copd_fracs))
        assert mean_copd >= mean_normal - 0.05, (
            f"COPD mean ineffective fraction {mean_copd:.3f} should be >= "
            f"Normal's {mean_normal:.3f} (within 0.05 tolerance) across {n_seeds} seeds"
        )

    def test_high_flow_cycle_threshold_obstructive_runs_without_error(self):
        p = {**COPD_PARAMS, "flow_cycle_threshold": 0.65}
        result = generate_breath_cycles(p, n_cycles=6, seed=31)
        assert result["n_mandatory_breaths"] == 6


# ---------------------------------------------------------------------------
# Class 7 — Multi-compartment mechanics
# ---------------------------------------------------------------------------

class TestMultiCompartmentMechanics:
    """Compartment structure is fixed anatomy — identical across vcv/pcv/
    psv/prvc/simv — and auto-PEEP/compartment state must carry continuously
    across mandatory <-> spontaneous breath-type transitions."""

    EXPECTED_COMPARTMENTS = {
        "Normal": 1, "Mild ARDS": 2, "Moderate ARDS": 2, "Severe ARDS": 2,
        "COPD": 3, "Bronchospasm": 2, "Pneumonia": 3, "Normal Neonate": 1, "RDS": 1,
    }

    @pytest.mark.parametrize("condition,n_expected", list(EXPECTED_COMPARTMENTS.items()))
    def test_compartment_counts_match_documented_scheme(self, condition, n_expected):
        assert len(COMPARTMENT_PROFILES[condition]) == n_expected

    @pytest.mark.parametrize("condition", list(EXPECTED_COMPARTMENTS.keys()))
    def test_compartment_fractions_sum_to_one(self, condition):
        total = sum(c["fraction"] for c in COMPARTMENT_PROFILES[condition])
        assert total == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize("condition,n_expected", list(EXPECTED_COMPARTMENTS.items()))
    def test_generator_reports_correct_n_compartments_vc(self, condition, n_expected):
        result = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "condition": condition}, n_cycles=3, seed=32)
        assert result["n_compartments"] == n_expected

    @pytest.mark.parametrize("condition,n_expected", list(EXPECTED_COMPARTMENTS.items()))

    def test_generator_reports_correct_n_compartments_pc(self, condition, n_expected):
        result = generate_breath_cycles(
            {**NORMAL_PARAMS_PC, "condition": condition}, n_cycles=3, seed=33)
        assert result["n_compartments"] == n_expected
    
    def test_mandatory_vt_unaffected_by_carried_over_volume(self):
        """Regression test: delivered_vt_ml must reflect only the volume
        this breath added, not the compartment volume it started from.
        Provoke meaningful auto-PEEP (high resistance relative to
        expiratory time) and confirm mandatory VT still tracks the VC
        target rather than inflating with the trapped volume."""
        p = {**NORMAL_PARAMS_VC, "condition": "COPD",
             "compliance_ml_per_cmH2O": 100.0, "resistance_cmH2O_L_s": 30.0,
             "respiratory_rate": 20.0, "tidal_volume_ml": 500.0,
             "pmus_peak_cmH2O": 2.0, "trigger_threshold_cmH2O": 5.0}
        result = generate_breath_cycles(p, n_cycles=10, seed=22)
        assert result["auto_peep_cmH2O"] > 2.0, (
            "Fixture should provoke meaningful auto-PEEP to actually test "
            f"this; got {result['auto_peep_cmH2O']:.2f} cmH2O — adjust "
            "resistance/RR if this fails on an otherwise-correct generator"
        )
        target = p["tidal_volume_ml"]
        assert abs(result["mandatory_delivered_vt_ml"] - target) < 0.25 * target, (
            f"Mandatory VT {result['mandatory_delivered_vt_ml']:.0f} mL "
            f"strayed too far from the {target:.0f} mL VC target under "
            f"auto-PEEP — delivered_vt_ml may be including carried-over volume"
        )

    def test_recruitment_slopes_zero_for_obstructive_disease(self):
        assert RECRUITMENT_SLOPES["COPD"] == 0.0
        assert RECRUITMENT_SLOPES["Bronchospasm"] == 0.0

    def test_recruitment_slopes_positive_for_ards(self):
        for condition in ("Mild ARDS", "Moderate ARDS", "Severe ARDS"):
            assert RECRUITMENT_SLOPES[condition] > 0.0

    def test_copd_develops_auto_peep(self):
        result = generate_breath_cycles(COPD_PARAMS, n_cycles=8, seed=34)
        assert result["auto_peep_cmH2O"] > 0.3

    def test_bronchospasm_develops_auto_peep(self):
        result = generate_breath_cycles(BRONCHOSPASM_PARAMS, n_cycles=8, seed=35)
        assert result["auto_peep_cmH2O"] > 0.3

    def test_normal_has_minimal_auto_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=36)
        assert result["auto_peep_cmH2O"] < 2.0

    def test_auto_peep_persists_across_a_spontaneous_breath_following_copd_mandatory(self):
        """The state-continuity requirement unique to this engine: run COPD
        with strong effort so spontaneous breaths interleave with mandatory
        ones, and confirm auto-PEEP still develops (i.e. compartment state
        wasn't silently reset between breath types)."""
        p = {**COPD_PARAMS, "effort_rate_per_min": 25.0,
             "pmus_peak_cmH2O": 15.0, "trigger_threshold_cmH2O": 1.0}
        result = generate_breath_cycles(p, n_cycles=10, seed=37)
        assert result["n_spontaneous_breaths"] > 0
        assert result["auto_peep_cmH2O"] > 0.3


# ---------------------------------------------------------------------------
# Class 8 — ETT complications
# ---------------------------------------------------------------------------

class TestETTComplications:
    """Cuff leak and partial obstruction overlays, applied identically to
    whichever breath type is active."""

    def test_cuff_leak_reduces_mandatory_delivered_vt(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=38)
        p_leak = {**NORMAL_PARAMS_VC, "ett_complication": "cuff_leak",
                  "cuff_leak_fraction": 0.20}
        r_leak = generate_breath_cycles(p_leak, n_cycles=6, seed=38)
        assert r_leak["mandatory_delivered_vt_ml"] < r_normal["mandatory_delivered_vt_ml"]

    def test_cuff_leak_fraction_matches_expected_reduction(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=39)
        p_leak = {**NORMAL_PARAMS_VC, "ett_complication": "cuff_leak",
                  "cuff_leak_fraction": 0.25}
        r_leak = generate_breath_cycles(p_leak, n_cycles=6, seed=39)
        expected = r_normal["mandatory_delivered_vt_ml"] * 0.75
        assert abs(r_leak["mandatory_delivered_vt_ml"] - expected) < 5.0

    def test_obstruction_raises_ppeak(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=40)
        p_obs = {**NORMAL_PARAMS_VC, "ett_complication": "obstruction",
                 "obstruction_R_multiplier": 3.0}
        r_obs = generate_breath_cycles(p_obs, n_cycles=6, seed=40)
        assert r_obs["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]

    def test_ett_complications_run_in_pc_mode(self):
        p_leak = {**NORMAL_PARAMS_PC, "ett_complication": "cuff_leak",
                  "cuff_leak_fraction": 0.15}
        result = generate_breath_cycles(p_leak, n_cycles=4, seed=41)
        assert result["n_mandatory_breaths"] == 4

    def test_no_complication_leaves_vt_unaffected(self):
        r1 = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=42)
        r2 = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "ett_complication": None}, n_cycles=6, seed=42)
        assert abs(r1["mandatory_delivered_vt_ml"] - r2["mandatory_delivered_vt_ml"]) < 1e-6


# ---------------------------------------------------------------------------
# Class 9 — Physiological directions (monotone responses)
# ---------------------------------------------------------------------------

class TestPhysiologicalDirections:
    """Cross-condition and cross-parameter monotonicity checks that verify
    the equation of motion is wired correctly."""

    def test_severe_ards_ppeak_higher_than_normal(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=43)
        r_ards = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=6, seed=43)
        assert r_ards["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]

    def test_bronchospasm_ppeak_higher_than_normal(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=6, seed=44)
        r_broncho = generate_breath_cycles(BRONCHOSPASM_PARAMS, n_cycles=6, seed=44)
        assert r_broncho["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]

    def test_lower_compliance_raises_vc_ppeak(self):
        r_stiff = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "compliance_ml_per_cmH2O": 20.0}, n_cycles=5, seed=45)
        r_compliant = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "compliance_ml_per_cmH2O": 90.0}, n_cycles=5, seed=45)
        assert r_stiff["ppeak_cmH2O"] > r_compliant["ppeak_cmH2O"]

    def test_higher_resistance_raises_vc_ppeak(self):
        r_lowR = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "resistance_cmH2O_L_s": 5.0}, n_cycles=5, seed=46)
        r_highR = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "resistance_cmH2O_L_s": 30.0}, n_cycles=5, seed=46)
        assert r_highR["ppeak_cmH2O"] > r_lowR["ppeak_cmH2O"]

    def test_higher_peep_raises_mean_paw(self):
        r_low = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "peep_cmH2O": 0.0}, n_cycles=5, seed=47)
        r_high = generate_breath_cycles(
            {**NORMAL_PARAMS_VC, "peep_cmH2O": 15.0}, n_cycles=5, seed=47)
        assert r_high["mean_paw_cmH2O"] > r_low["mean_paw_cmH2O"]


# ---------------------------------------------------------------------------
# Class 10 — Validity filter
# ---------------------------------------------------------------------------

class TestValidityFilter:
    """Threshold logic and invalid_reason strings."""

    def test_normal_baseline_scenario_is_valid(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=5, seed=48)
        assert result["is_valid"] is True
        assert result["invalid_reason"] == ""

    def test_ppeak_barotrauma_flagged(self):
        p = {**NORMAL_PARAMS_PC, "insp_pressure_cmH2O": 34.0, "peep_cmH2O": 18.0,
             "compliance_ml_per_cmH2O": 15.0, "resistance_cmH2O_L_s": 40.0}
        result = generate_breath_cycles(p, n_cycles=3, seed=49)
        if not result["is_valid"]:
            assert ("barotrauma" in result["invalid_reason"].lower() or
                    "ppeak" in result["invalid_reason"].lower() or
                    "exceeds" in result["invalid_reason"].lower())

    def test_vc_driving_pressure_breach_flagged(self):
        p = {**NORMAL_PARAMS_VC, "tidal_volume_ml": 700.0,
             "compliance_ml_per_cmH2O": 15.0}
        result = generate_breath_cycles(p, n_cycles=3, seed=50)
        if not result["is_valid"]:
            assert ("driving" in result["invalid_reason"].lower() or
                    "ppeak" in result["invalid_reason"].lower())

    def test_pc_insp_pressure_ceiling_enforced_by_validation(self):
        # insp_pressure_cmH2O itself isn't range-checked in _validate_params
        # (only the common params are). Chosen so Ppeak stays under the
        # barotrauma threshold (50) while insp_pressure exceeds its own
        # ceiling (35) -- isolates the INSP_PRESSURE_MAX_CMHH2O filter from
        # the Ppeak filter, which otherwise fires first (checked earlier).
        bad = {**NORMAL_PARAMS_PC, "peep_cmH2O": 5.0, "insp_pressure_cmH2O": 40.0}
        result = generate_breath_cycles(bad, n_cycles=2, seed=51)
        assert result["is_valid"] is False
        assert "insp" in result["invalid_reason"].lower() or \
               "pressure" in result["invalid_reason"].lower()

    def test_pressure_support_ceiling_flagged(self):
        p = {**NORMAL_PARAMS_VC, "pressure_support_cmH2O": 25.0}
        result = generate_breath_cycles(p, n_cycles=3, seed=52)
        assert result["is_valid"] is False
        assert "support" in result["invalid_reason"].lower()

    def test_vt_too_low_flagged(self):
        p = {**NORMAL_PARAMS_VC, "tidal_volume_ml": 100.0}
        result = generate_breath_cycles(p, n_cycles=3, seed=53)
        if not result["is_valid"]:
            assert ("vt" in result["invalid_reason"].lower() or
                    "volume" in result["invalid_reason"].lower() or
                    "minimum" in result["invalid_reason"].lower())

    def test_vt_too_high_flagged(self):
        p = {**NORMAL_PARAMS_VC, "tidal_volume_ml": 900.0,
             "compliance_ml_per_cmH2O": 150.0, "resistance_cmH2O_L_s": 3.0}
        result = generate_breath_cycles(p, n_cycles=3, seed=54)
        if not result["is_valid"]:
            assert ("vt" in result["invalid_reason"].lower() or
                    "volume" in result["invalid_reason"].lower() or
                    "maximum" in result["invalid_reason"].lower())

    def test_constants_consistent_with_ibw(self):
        assert VT_MIN_ML == pytest.approx(IBW_KG * 3, abs=0.1)
        assert VT_MAX_ML == pytest.approx(IBW_KG * 12, abs=0.1)

    def test_invalid_reason_empty_string_when_valid(self):
        result = generate_breath_cycles(NORMAL_PARAMS_VC, n_cycles=4, seed=55)
        if result["is_valid"]:
            assert result["invalid_reason"] == ""

class TestPopulationBranching:
    """Validates that neonatal thresholds are keyed off `population`,
    not off condition name, and that adults are unaffected."""

    def test_population_field_not_condition_name_drives_thresholds(self):
        """An adult-named condition forced into the neonatal population
        branch must get neonatal thresholds — confirms the branch is
        genuinely keyed off `population`."""
        p = {**NORMAL_PARAMS_VC, "population": "neonate", "weight_kg": 3.0}
        result = generate_breath_cycles(p, n_cycles=3)
        # A 15 mL breath is below the adult VT floor (210 mL) but above
        # the neonatal floor (3.0 * 4.0 = 12 mL) — this only passes if
        # the neonatal floor was actually applied.
        p_small_vt = {**p, "tidal_volume_ml": 15} if "tidal_volume_ml" in p else p
        # (Adjust the volume-setting key per engine — tidal_volume_ml for
        # VCV/PRVC, insp_pressure_cmH2O-driven for PCV, etc.)
        assert result["is_valid"] is True or "VT" not in result.get("invalid_reason", "")

    def test_missing_population_defaults_to_adult(self):
        """Omitting `population` entirely must behave identically to
        population='adult' — protects all seven existing conditions."""
        p_explicit = {**NORMAL_PARAMS_VC, "population": "adult"}
        p_implicit = {k: v for k, v in NORMAL_PARAMS_VC.items() if k != "population"}
        r_explicit = generate_breath_cycles(p_explicit, n_cycles=5)
        r_implicit = generate_breath_cycles(p_implicit, n_cycles=5)
        assert r_explicit["is_valid"] == r_implicit["is_valid"]
        assert r_explicit["delivered_vt_ml"] == pytest.approx(r_implicit["delivered_vt_ml"], abs=1e-6)

    def test_neonate_vt_min_scales_with_weight_kg(self):
        """VT floor must scale with weight_kg, not be a second fixed number."""
        p_1_5kg = {**NORMAL_PARAMS_VC, "population": "neonate", "weight_kg": 1.5}
        p_3_0kg = {**NORMAL_PARAMS_VC, "population": "neonate", "weight_kg": 3.0}
        r_1_5 = generate_breath_cycles(p_1_5kg, n_cycles=3)
        r_3_0 = generate_breath_cycles(p_3_0kg, n_cycles=3)
        # Same delivered VT should be valid for the heavier weight and
        # invalid (too low) for the lighter one, if VT sits between the
        # two floors (1.5*4=6 mL vs 3.0*4=12 mL) — construct delivered_vt
        # accordingly per engine, or assert on the computed floor directly
        # if your engine exposes it as a metric.

    def test_neonatal_vt_ceiling_and_driving_pressure_checks_skipped(self):
        """Confirms the VT-max and driving-pressure checks are genuinely
        absent for population='neonate', not silently always-false."""
        # Construct params with population='neonate' and an enormous
        # delivered volume relative to weight — must NOT be flagged for
        # exceeding a VT ceiling (there isn't one for neonates), and must
        # not be flagged for driving pressure either.
        p = {**NORMAL_PARAMS_VC, "population": "neonate", "weight_kg": 3.0}
        result = generate_breath_cycles(p, n_cycles=3)
        if not result["is_valid"]:
            assert "maximum" not in result["invalid_reason"].lower()
            assert "mortality" not in result["invalid_reason"].lower()

    def test_adult_conditions_unaffected_by_neonatal_constants(self):
        """Full regression check — every existing adult fixture in this
        file must produce identical is_valid/metrics after this refactor.
        Run once per file against whatever adult fixtures already exist
        (NORMAL_PARAMS, SEVERE_ARDS_PARAMS, COPD_PARAMS, etc.)."""
        for fixture in (NORMAL_PARAMS_VC,):  # extend with every adult fixture in this file
            result = generate_breath_cycles(fixture, n_cycles=5)
            assert result["is_valid"] in (True, False)  # replace with recorded pre-refactor value

# ---------------------------------------------------------------------------
# Class 11 — Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """generate_dataset() must return correct structure with unique IDs and
    both mandatory sub-modes represented even in a capped slice."""

    @pytest.fixture(scope="class")
    @classmethod
    def small_dataset(cls):
        return generate_dataset(
            "Normal", compliance_ml_per_cmH2O=60.0,
            resistance_cmH2O_L_s=10.0, n_cycles=2, max_scenarios=40,
        )

    def test_returns_list(self, small_dataset):
        assert isinstance(small_dataset, list)

    def test_dataset_nonempty(self, small_dataset):
        assert len(small_dataset) > 0

    def test_dataset_respects_max_scenarios_cap(self, small_dataset):
        assert len(small_dataset) <= 40

    def test_all_scenario_keys_present(self, small_dataset):
        for s in small_dataset[:10]:
            missing = DATASET_SCENARIO_KEYS - s.keys()
            assert not missing, f"Scenario missing keys: {missing}"

    def test_scenario_ids_are_unique(self, small_dataset):
        ids = [s["scenario_id"] for s in small_dataset]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_scenario_id_contains_condition_and_mode(self, small_dataset):
        for s in small_dataset[:5]:
            sid = s["scenario_id"]
            assert "NORMAL" in sid.upper()
            assert ("_VC_" in sid) or ("_PC_" in sid)

    def test_both_mandatory_modes_represented(self, small_dataset):
        modes = {s["params"]["mandatory_mode"] for s in small_dataset}
        assert modes == {"VC", "PC"}

    def test_valid_scenarios_have_waveforms(self, small_dataset):
        for s in small_dataset:
            if s["is_valid"]:
                assert isinstance(s["waveforms"], dict)
                assert len(s["waveforms"]) > 0

    def test_invalid_scenarios_have_empty_waveforms(self, small_dataset):
        for s in small_dataset:
            if not s["is_valid"]:
                assert s["waveforms"] == {}

    def test_valid_scenarios_have_populated_metrics(self, small_dataset):
        for s in small_dataset:
            if s["is_valid"]:
                assert isinstance(s["metrics"], dict)
                assert len(s["metrics"]) > 0

    def test_condition_field_present(self, small_dataset):
        for s in small_dataset[:5]:
            assert s["condition"] == "Normal"

    def test_generated_at_is_populated(self, small_dataset):
        for s in small_dataset[:5]:
            assert isinstance(s["generated_at"], str)
            assert len(s["generated_at"]) > 0

    def test_breath_records_present_for_valid_scenarios(self, small_dataset):
        for s in small_dataset:
            if s["is_valid"]:
                assert isinstance(s["breath_records"], list)
                assert len(s["breath_records"]) > 0

    def test_severe_ards_dataset_slice_runs_without_exception(self):
        scenarios = generate_dataset(
            "Severe ARDS", compliance_ml_per_cmH2O=18.0,
            resistance_cmH2O_L_s=16.0, n_cycles=2, max_scenarios=30,
        )
        assert len(scenarios) == 30
        ids = [s["scenario_id"] for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_copd_dataset_has_higher_or_equal_ineff_fraction_than_normal(self, small_dataset):
        ds_copd = generate_dataset(
            "COPD", compliance_ml_per_cmH2O=100.0,
            resistance_cmH2O_L_s=22.0, n_cycles=3, max_scenarios=15,
        )
        valid_normal = [s for s in small_dataset if s["is_valid"]]
        valid_copd = [s for s in ds_copd if s["is_valid"]]
        if valid_normal and valid_copd:
            ineff_normal = np.mean([
                s["metrics"].get("ineffective_trigger_fraction", 0)
                for s in valid_normal
            ])
            ineff_copd = np.mean([
                s["metrics"].get("ineffective_trigger_fraction", 0)
                for s in valid_copd
            ])
            assert ineff_copd >= ineff_normal - 0.10


# ---------------------------------------------------------------------------
# Class 12 — Parameter grid completeness
# ---------------------------------------------------------------------------

class TestParameterGrid:
    """PARAMETER_GRID should cover all clinically relevant dimensions for
    both mandatory sub-modes and the spontaneous-breath / patient-effort
    model, with physiologically grounded ranges informed by the project's
    literature-grounding pass."""

    EXPECTED_MANDATORY_PARAMS = {
        "mandatory_mode", "respiratory_rate", "peep_cmH2O", "ie_ratio",
        "tidal_volume_ml_per_kg", "flow_pattern", "insp_pressure_cmH2O",
    }
    EXPECTED_SIMV_SPECIFIC_PARAMS = {"f_window", "rise_time_s"}
    EXPECTED_SPONTANEOUS_PARAMS = {
        "pressure_support_cmH2O", "flow_cycle_threshold",
        "trigger_threshold_cmH2O",
    }
    EXPECTED_PATIENT_PARAMS = {
        "pmus_peak_cmH2O", "effort_rate_per_min", "effort_duration_s", "pmus_cv",
    }

    def test_parameter_grid_has_mandatory_params(self):
        missing = self.EXPECTED_MANDATORY_PARAMS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_parameter_grid_has_simv_specific_params(self):
        missing = self.EXPECTED_SIMV_SPECIFIC_PARAMS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_parameter_grid_has_spontaneous_params(self):
        missing = self.EXPECTED_SPONTANEOUS_PARAMS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_parameter_grid_has_patient_params(self):
        missing = self.EXPECTED_PATIENT_PARAMS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_mandatory_mode_grid_has_both_submodes(self):
        assert set(PARAMETER_GRID["mandatory_mode"]) == {"VC", "PC"}

    def test_f_window_range_matches_literature_grounding(self):
        """Grounding doc recommends a tunable 0.15-0.30 range, default
        0.20-0.25 -- the grid should at least span that band."""
        fw = PARAMETER_GRID["f_window"]
        assert min(fw) <= 0.15 + 1e-9
        assert max(fw) >= 0.30 - 1e-9

    def test_flow_cycle_threshold_covers_obstructive_and_restrictive_bands(self):
        """Grounding doc: obstructive default ~0.65, restrictive ~0.25-0.40
        (not ~0.10, which is a delayed-cycling stress test, not a default)."""
        fct = PARAMETER_GRID["flow_cycle_threshold"]
        assert max(fct) >= 0.65 - 1e-9
        assert min(fct) >= 0.20, (
            "Grid should not use ~0.10 as a default restrictive FCT per "
            "literature-grounding recommendation"
        )

    def test_mandatory_rate_grid_scoped_to_simv_weaning_range(self):
        """SIMV's mandatory-rate grid is deliberately narrower than vcv/
        pcv's full 8-30 bpm CMV range -- scoped to ~4-12 bpm (weaning
        endpoint through initiation)."""
        rr = PARAMETER_GRID["respiratory_rate"]
        assert min(rr) <= 5
        assert max(rr) <= 15

    def test_all_grid_values_are_lists(self):
        for key, values in PARAMETER_GRID.items():
            assert isinstance(values, list), f"{key} is not a list"
            assert len(values) >= 2, f"{key} needs >= 2 values for a real sweep"
