"""
tests/test_prvc_generator.py
-----------------------------
Unit tests for generator/prvc_generator.py (PRVC waveform generator).

Nine test classes:
    TestInterfaceContract         — return types, keys, array shapes, validation
    TestPhysiologicalPlausibility — basic physical constraints on all outputs
    TestTestBreathBootstrap       — VC test breath, AutoFlow seeding, fallback
    TestOuterLoopControl          — adaptation step, tolerance, ceiling/floor clipping
    TestPRVCWaveformShape         — waveform morphology specific to PRVC
    TestConvergenceAndCeiling     — converged / breaths_to_converge / ceiling_limited
    TestMultiCompartmentMechanics — compartment profiles, auto-PEEP, PEEP recruitment
    TestValidityFilter            — threshold logic and invalid_reason strings
    TestDatasetGeneration         — generate_dataset() structure and correctness

Key differences from test_pcv_generator.py:
    - Breath 1 is a volume-controlled test breath, not pressure-controlled —
      it is excluded from convergence/stability tracking and always delivers
      ~vt_target by construction (it's the maneuver breath, not the mode).
    - pressure_trajectory / delivered_vt_trajectory are new per-breath arrays
      with length == n_cycles, on top of the usual 100 Hz waveform arrays.
    - adaptation_step_cmH2O and vt_tolerance_frac are uniform constants
      across conditions in this implementation (see PRVC parameter grid
      doc) — tested as overridable params, not as swept grid dimensions.
    - "Valid" now includes ceiling-limited non-convergence deliberately —
      only barotrauma (Ppeak) and out-of-range Vt *on a converged breath*
      hard-invalidate a scenario (see generate_breath_cycles docstring).
    - VT tolerance in physiological tests is wider than VCV's, matching
      PCV/PSV precedent — ODE-integrated delivery is less exact than VCV's
      analytical flow integration.

Run with:
    python -m pytest tests/test_prvc_generator.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.prvc_generator import (
    ADAPTATION_STEP_CMH2O_DEFAULT,
    COMPARTMENT_PROFILES,
    IBW_KG,
    PARAMETER_GRID,
    PPEAK_MAX_CMHH2O,
    PRESSURE_FLOOR_ABOVE_PEEP,
    RECRUITMENT_SLOPES,
    RISE_TIME_S,
    VT_AVERAGING_WINDOW_DEFAULT,
    VT_MAX_ML,
    VT_MIN_ML,
    VT_TOLERANCE_FRAC_DEFAULT,
    generate_breath_cycles,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NORMAL_PARAMS = {
    "vt_target_ml": 420.0,
    "respiratory_rate": 16,
    "peep_cmH2O": 5,
    "ie_ratio": 0.5,
    "pressure_ceiling_cmH2O": 25.0,
    "compliance_ml_per_cmH2O": 80.0,
    "resistance_cmH2O_L_s": 10.0,
    "condition": "Normal",
}

SEVERE_ARDS_PARAMS = {
    **NORMAL_PARAMS,
    "compliance_ml_per_cmH2O": 18.0,
    "resistance_cmH2O_L_s": 16.0,
    "condition": "Severe ARDS",
    "pressure_ceiling_cmH2O": 15.0,
}

MODERATE_ARDS_PARAMS = {
    **NORMAL_PARAMS,
    "compliance_ml_per_cmH2O": 32.0,
    "resistance_cmH2O_L_s": 14.0,
    "condition": "Moderate ARDS",
    "pressure_ceiling_cmH2O": 30.0,
}

COPD_PARAMS = {
    **NORMAL_PARAMS,
    "compliance_ml_per_cmH2O": 100.0,
    "resistance_cmH2O_L_s": 25.0,
    "condition": "COPD",
    "respiratory_rate": 22,
    "pressure_ceiling_cmH2O": 30.0,
}

BRONCHOSPASM_PARAMS = {
    **NORMAL_PARAMS,
    "compliance_ml_per_cmH2O": 75.0,
    "resistance_cmH2O_L_s": 37.0,
    "condition": "Bronchospasm",
    "respiratory_rate": 14,
    "ie_ratio": 0.33,
    "pressure_ceiling_cmH2O": 30.0,
}

CORE_KEYS = {"time", "pressure", "flow", "volume"}
TRAJECTORY_KEYS = {"pressure_trajectory", "delivered_vt_trajectory"}
NUMERIC_METRIC_KEYS = {
    "ppeak_cmH2O", "delivered_vt_ml", "driving_p_cmH2O", "mean_paw_cmH2O",
    "auto_peep_cmH2O", "fill_fraction", "minute_vent_l",
}
STATUS_KEYS = {"test_breath_plateau_cmH2O", "breaths_to_converge",
               "converged", "ceiling_limited"}
VALIDITY_KEYS = {"is_valid", "invalid_reason"}
ALL_KEYS = CORE_KEYS | TRAJECTORY_KEYS | NUMERIC_METRIC_KEYS | STATUS_KEYS | VALIDITY_KEYS

DATASET_SCENARIO_KEYS = {
    "scenario_id", "condition", "params", "metrics",
    "is_valid", "invalid_reason", "waveforms", "generated_at",
}


# ---------------------------------------------------------------------------
# Class 1 — Interface contract
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """
    generate_breath_cycles must return all documented keys with correct
    types, and validate its required parameters.
    """

    def test_returns_dict(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert isinstance(result, dict)

    def test_contains_all_core_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert CORE_KEYS.issubset(result.keys()), (
            f"Missing core keys: {CORE_KEYS - result.keys()}"
        )

    def test_contains_trajectory_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert TRAJECTORY_KEYS.issubset(result.keys())

    def test_contains_all_numeric_metric_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert NUMERIC_METRIC_KEYS.issubset(result.keys())

    def test_contains_status_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert STATUS_KEYS.issubset(result.keys())

    def test_contains_validity_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert VALIDITY_KEYS.issubset(result.keys())

    def test_core_arrays_are_numpy(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        for key in CORE_KEYS:
            assert isinstance(result[key], np.ndarray), (
                f"'{key}' should be np.ndarray, got {type(result[key])}"
            )

    def test_core_arrays_same_length(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        lengths = {k: len(result[k]) for k in CORE_KEYS}
        assert len(set(lengths.values())) == 1, (
            f"Core arrays have different lengths: {lengths}"
        )

    def test_trajectory_arrays_length_equals_n_cycles(self):
        n = 9
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=n, seed=0)
        assert len(result["pressure_trajectory"]) == n
        assert len(result["delivered_vt_trajectory"]) == n

    def test_numeric_metrics_are_numeric(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        for key in NUMERIC_METRIC_KEYS:
            assert isinstance(result[key], (int, float)), (
                f"Metric '{key}' should be numeric, got {type(result[key])}"
            )

    def test_converged_and_ceiling_limited_are_bool(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=0)
        assert isinstance(result["converged"], bool)
        assert isinstance(result["ceiling_limited"], bool)

    def test_breaths_to_converge_is_int_or_none(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=0)
        assert result["breaths_to_converge"] is None or isinstance(
            result["breaths_to_converge"], int
        )

    def test_missing_required_param_raises_value_error(self):
        bad = {k: v for k, v in NORMAL_PARAMS.items() if k != "vt_target_ml"}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad, n_cycles=3)

    def test_non_positive_compliance_raises_value_error(self):
        bad = {**NORMAL_PARAMS, "compliance_ml_per_cmH2O": 0.0}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad, n_cycles=3)

    def test_non_positive_resistance_raises_value_error(self):
        bad = {**NORMAL_PARAMS, "resistance_cmH2O_L_s": -1.0}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad, n_cycles=3)

    def test_non_positive_ie_ratio_raises_value_error(self):
        bad = {**NORMAL_PARAMS, "ie_ratio": 0.0}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad, n_cycles=3)

    def test_reproducible_with_same_seed(self):
        r1 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=7)
        r2 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=6, seed=7)
        np.testing.assert_array_equal(r1["pressure"], r2["pressure"])
        np.testing.assert_array_equal(r1["pressure_trajectory"], r2["pressure_trajectory"])

    def test_n_cycles_1_succeeds(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1, seed=0)
        assert isinstance(result, dict)
        assert len(result["time"]) > 0
        assert len(result["pressure_trajectory"]) == 1

    def test_more_cycles_produces_longer_waveform(self):
        r5 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=5, seed=0)
        r10 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=0)
        assert len(r10["time"]) > len(r5["time"])


# ---------------------------------------------------------------------------
# Class 2 — Physiological plausibility
# ---------------------------------------------------------------------------

class TestPhysiologicalPlausibility:
    """Physical constraints that must hold regardless of parameter settings."""

    @pytest.fixture
    def result(self):
        return generate_breath_cycles(NORMAL_PARAMS, n_cycles=8, seed=1)

    def test_time_monotonically_increasing(self, result):
        diffs = np.diff(result["time"])
        assert np.all(diffs > 0), "Time array is not strictly increasing"

    def test_pressure_never_below_peep(self, result):
        peep = NORMAL_PARAMS["peep_cmH2O"]
        assert np.all(result["pressure"] >= peep - 0.5), (
            f"Pressure dropped below PEEP ({peep}): min={result['pressure'].min():.2f}"
        )

    def test_volume_never_negative(self, result):
        assert np.all(result["volume"] >= -0.1), (
            f"Volume went negative: min={result['volume'].min():.2f} mL"
        )

    def test_flow_has_both_inspiratory_and_expiratory(self, result):
        assert result["flow"].max() > 0, "No inspiratory (positive) flow detected"
        assert result["flow"].min() < 0, "No expiratory (negative) flow detected"

    def test_ppeak_does_not_wildly_exceed_ceiling(self, result):
        ceiling_abs = NORMAL_PARAMS["peep_cmH2O"] + NORMAL_PARAMS["pressure_ceiling_cmH2O"]
        assert result["ppeak_cmH2O"] <= ceiling_abs + 5.0, (
            f"Ppeak {result['ppeak_cmH2O']:.1f} far exceeds ceiling {ceiling_abs}"
        )

    def test_delivered_vt_positive(self, result):
        assert result["delivered_vt_ml"] > 0

    def test_minute_vent_consistent_with_vt_and_rr(self, result):
        expected = result["delivered_vt_ml"] * NORMAL_PARAMS["respiratory_rate"] / 1000.0
        assert result["minute_vent_l"] == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Class 3 — Test breath bootstrap (PRVC-specific)
# ---------------------------------------------------------------------------

class TestTestBreathBootstrap:
    """
    Breath 1 is a volume-controlled maneuver breath whose measured plateau
    seeds breath 2's working pressure (AutoFlow rule). See module docstring.
    """

    def test_test_breath_plateau_recorded_when_enabled(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=5, seed=2)
        assert result["test_breath_plateau_cmH2O"] is not None
        assert result["test_breath_plateau_cmH2O"] > NORMAL_PARAMS["peep_cmH2O"]

    def test_test_breath_plateau_none_when_disabled(self):
        params = {**NORMAL_PARAMS, "use_vc_test_breath": False}
        result = generate_breath_cycles(params, n_cycles=5, seed=2)
        assert result["test_breath_plateau_cmH2O"] is None

    def test_test_breath_delivers_target_vt(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=5, seed=2)
        breath1_vt = result["delivered_vt_trajectory"][0]
        assert breath1_vt == pytest.approx(NORMAL_PARAMS["vt_target_ml"], rel=0.05), (
            f"Test breath should deliver ~target Vt (flow-prescribed), got {breath1_vt:.0f}"
        )

    def test_breath2_undershoots_measured_plateau(self):
        """AutoFlow rule: breath 2 seeds at 75% of measured driving pressure,
        so it should sit below breath 1's plateau whenever the ceiling
        doesn't force an even lower clip."""
        result = generate_breath_cycles(MODERATE_ARDS_PARAMS, n_cycles=8, seed=3)
        traj = result["pressure_trajectory"]
        assert traj[1] < traj[0], (
            f"Expected breath 2 ({traj[1]:.1f}) below test breath plateau ({traj[0]:.1f})"
        )

    def test_fallback_path_runs_without_test_breath(self):
        params = {**BRONCHOSPASM_PARAMS, "use_vc_test_breath": False}
        result = generate_breath_cycles(params, n_cycles=8, seed=4)
        assert isinstance(result, dict)
        assert len(result["time"]) > 0
        assert result["test_breath_plateau_cmH2O"] is None

    def test_stiffer_lung_produces_higher_test_breath_plateau(self):
        """The test breath is diagnostic: lower compliance -> higher measured
        plateau pressure for the same target Vt, all else equal."""
        r_normal = generate_breath_cycles(NORMAL_PARAMS, n_cycles=3, seed=5)
        r_severe = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=3, seed=5)
        assert r_severe["test_breath_plateau_cmH2O"] > r_normal["test_breath_plateau_cmH2O"], (
            f"Severe ARDS plateau {r_severe['test_breath_plateau_cmH2O']:.1f} should exceed "
            f"Normal plateau {r_normal['test_breath_plateau_cmH2O']:.1f}"
        )


# ---------------------------------------------------------------------------
# Class 4 — Outer loop control (PRVC-specific)
# ---------------------------------------------------------------------------

class TestOuterLoopControl:
    """adaptation_step_cmH2O, vt_tolerance_frac, and pressure_ceiling govern
    the breath-to-breath adaptive controller. These are uniform constants
    across conditions in this implementation, tested here as overridable
    per-call params rather than swept grid dimensions."""

    def test_pressure_never_exceeds_ceiling_from_breath_2_on(self):
        result = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=12, seed=6)
        ceiling_abs = SEVERE_ARDS_PARAMS["peep_cmH2O"] + SEVERE_ARDS_PARAMS["pressure_ceiling_cmH2O"]
        controlled = result["pressure_trajectory"][1:]  # exclude test breath
        assert np.all(controlled <= ceiling_abs + 0.01), (
            f"Controlled breaths exceeded ceiling {ceiling_abs}: max={controlled.max():.2f}"
        )

    def test_pressure_never_below_floor_from_breath_2_on(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=6)
        floor_abs = NORMAL_PARAMS["peep_cmH2O"] + PRESSURE_FLOOR_ABOVE_PEEP
        controlled = result["pressure_trajectory"][1:]
        assert np.all(controlled >= floor_abs - 0.01), (
            f"Controlled breaths dropped below floor {floor_abs}: min={controlled.min():.2f}"
        )

    def test_custom_adaptation_step_is_respected(self):
        params = {**MODERATE_ARDS_PARAMS, "adaptation_step_cmH2O": 1.0}
        result = generate_breath_cycles(params, n_cycles=10, seed=3)
        traj = result["pressure_trajectory"]
        # Look at the climbing phase (breaths 2 onward, before convergence flattens it)
        diffs = np.diff(traj[1:])
        nonzero_steps = diffs[np.abs(diffs) > 0.01]
        if len(nonzero_steps) > 0:
            assert np.all(np.abs(nonzero_steps) <= 1.0 + 0.01), (
                f"Step size exceeded configured adaptation_step=1.0: {nonzero_steps}"
            )

    def test_larger_adaptation_step_converges_in_fewer_or_equal_breaths(self):
        p_small_step = {**MODERATE_ARDS_PARAMS, "adaptation_step_cmH2O": 1.0}
        p_large_step = {**MODERATE_ARDS_PARAMS, "adaptation_step_cmH2O": 3.0}
        r_small = generate_breath_cycles(p_small_step, n_cycles=15, seed=3)
        r_large = generate_breath_cycles(p_large_step, n_cycles=15, seed=3)
        if r_small["converged"] and r_large["converged"]:
            assert r_large["breaths_to_converge"] <= r_small["breaths_to_converge"], (
                f"Larger step took {r_large['breaths_to_converge']} breaths, "
                f"smaller step took {r_small['breaths_to_converge']} breaths"
            )

    def test_default_constants_match_documented_values(self):
        assert ADAPTATION_STEP_CMH2O_DEFAULT == 2.0
        assert VT_TOLERANCE_FRAC_DEFAULT == 0.10
        assert VT_AVERAGING_WINDOW_DEFAULT == 2
        assert RISE_TIME_S == 0.10

    def test_vt_averaging_window_smooths_error_signal(self):
        """A wider averaging window should not change the final converged
        pressure by much (steady-state is steady-state either way), but
        should not error out and should still produce a valid trajectory."""
        params = {**MODERATE_ARDS_PARAMS, "vt_averaging_window": 4}
        result = generate_breath_cycles(params, n_cycles=15, seed=3)
        assert isinstance(result, dict)
        assert len(result["pressure_trajectory"]) == 15


# ---------------------------------------------------------------------------
# Class 5 — PRVC waveform shape
# ---------------------------------------------------------------------------

class TestPRVCWaveformShape:
    """PRVC's intra-breath shape should match PCV (decelerating flow, flat
    pressure plateau); its inter-breath shape is the unique staircase."""

    def test_pressure_plateaus_flat_within_a_breath(self):
        """After the rise time, pressure should hold constant for a
        stretch during a controlled (non-test) breath."""
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=3, seed=8)
        # Breath 2 is a normal PC breath; find a window well past the rise time
        # within the pressure array and check it's locally flat.
        p = result["pressure"]
        # crude locate: after the first big jump (test breath -> breath 2 rise),
        # look for a run of near-constant values
        diffs = np.abs(np.diff(p))
        flat_run = np.sum(diffs < 0.05)
        assert flat_run > len(p) * 0.2, "Expected a substantial flat plateau region"

    def test_flow_decelerating_shape_like_pcv(self):
        """Peak inspiratory flow should occur early, with flow declining
        (not increasing) over the remainder of the breath. Isolated to a
        single pressure-controlled breath via the fallback path (no VC
        test breath) so the constant-flow maneuver breath's flat profile
        doesn't contaminate the shape check -- the test breath is flow-
        prescribed VCV physics by design, not decelerating PCV physics."""
        params = {**NORMAL_PARAMS, "use_vc_test_breath": False}
        result = generate_breath_cycles(params, n_cycles=1, seed=8)
        flow = result["flow"]
        peak_idx = int(np.argmax(flow))
        assert peak_idx < len(flow) * 0.4, (
            f"Peak flow at index {peak_idx}/{len(flow)} should occur early in "
            "a pressure-controlled breath, consistent with a decelerating "
            "(not accelerating) profile"
        )
        # Flow should trend downward after the peak through the rest of
        # inspiration (allow noise via a coarse split-half comparison).
        post_peak = flow[peak_idx:]
        first_half_mean = np.mean(post_peak[:len(post_peak) // 3])
        later_mean = np.mean(post_peak[len(post_peak) // 3: 2 * len(post_peak) // 3])
        assert later_mean < first_half_mean, (
            "Flow should decline after its peak, not stay flat or rise"
        )

    def test_volume_trends_toward_target_in_converging_scenario(self):
        result = generate_breath_cycles(MODERATE_ARDS_PARAMS, n_cycles=10, seed=3)
        vt_target = MODERATE_ARDS_PARAMS["vt_target_ml"]
        errors = np.abs(result["delivered_vt_trajectory"] - vt_target)
        # Error at convergence should be smaller than error on breath 2
        # (the deliberately-undershot AutoFlow seed breath).
        assert errors[-1] <= errors[1], (
            f"Expected final error ({errors[-1]:.1f}) <= breath-2 error ({errors[1]:.1f})"
        )

    def test_pressure_trajectory_flat_after_convergence(self):
        result = generate_breath_cycles(MODERATE_ARDS_PARAMS, n_cycles=12, seed=3)
        if result["converged"] and result["breaths_to_converge"] is not None:
            idx = result["breaths_to_converge"] - 1  # 0-indexed
            tail = result["pressure_trajectory"][idx:]
            assert np.allclose(tail, tail[0], atol=0.01), (
                f"Pressure should be flat after declared convergence: {tail}"
            )


# ---------------------------------------------------------------------------
# Class 6 — Convergence and ceiling-limited failure
# ---------------------------------------------------------------------------

class TestConvergenceAndCeiling:
    """The dual terminal states: converged, or ceiling-limited non-
    convergence. Both are meaningful, labeled outcomes (see docstring)."""

    def test_normal_condition_converges(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=1)
        assert result["converged"] is True
        assert result["breaths_to_converge"] is not None

    def test_severe_ards_tight_ceiling_is_ceiling_limited(self):
        result = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=12, seed=2)
        assert result["converged"] is False
        assert result["ceiling_limited"] is True
        assert result["breaths_to_converge"] is None

    def test_ceiling_limited_scenario_not_hard_invalidated(self):
        result = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=12, seed=2)
        assert result["is_valid"] is True, (
            "Ceiling-limited non-convergence should be retained as a valid, "
            "labeled scenario, not hard-invalidated"
        )

    def test_severe_ards_final_vt_undershoots_target(self):
        result = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=12, seed=2)
        assert result["delivered_vt_ml"] < SEVERE_ARDS_PARAMS["vt_target_ml"] * 0.85

    def test_generous_ceiling_converges_even_for_severe_ards(self):
        params = {**SEVERE_ARDS_PARAMS, "pressure_ceiling_cmH2O": 35.0, "peep_cmH2O": 15.0}
        result = generate_breath_cycles(params, n_cycles=20, seed=2)
        # With a generous ceiling and recruitment-boosting PEEP, this specific
        # scenario is expected to at least get much closer to target than the
        # tight-ceiling case, even if it doesn't fully converge every seed.
        tight = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=20, seed=2)
        assert result["delivered_vt_ml"] > tight["delivered_vt_ml"]


# ---------------------------------------------------------------------------
# Class 7 — Multi-compartment mechanics
# ---------------------------------------------------------------------------

class TestMultiCompartmentMechanics:
    """PRVC reuses the same compartment architecture as psv_generator."""

    def test_compartment_counts_match_documented_scheme(self):
        expected = {
            "Normal": 1, "Mild ARDS": 2, "Moderate ARDS": 2, "Severe ARDS": 2,
            "COPD": 3, "Bronchospasm": 2, "Pneumonia": 3,
        }
        for condition, n in expected.items():
            assert len(COMPARTMENT_PROFILES[condition]) == n, (
                f"{condition} expected {n} compartments, "
                f"got {len(COMPARTMENT_PROFILES[condition])}"
            )

    def test_compartment_fractions_sum_to_one(self):
        for condition, profile in COMPARTMENT_PROFILES.items():
            total = sum(c["fraction"] for c in profile)
            assert total == pytest.approx(1.0, abs=0.01), (
                f"{condition} compartment fractions sum to {total}, not 1.0"
            )

    def test_recruitment_slopes_zero_for_obstructive_disease(self):
        assert RECRUITMENT_SLOPES["COPD"] == 0.0
        assert RECRUITMENT_SLOPES["Bronchospasm"] == 0.0

    def test_recruitment_slopes_positive_for_ards(self):
        for condition in ["Mild ARDS", "Moderate ARDS", "Severe ARDS"]:
            assert RECRUITMENT_SLOPES[condition] > 0.0

    def test_copd_develops_auto_peep(self):
        result = generate_breath_cycles(COPD_PARAMS, n_cycles=25, seed=4)
        assert result["auto_peep_cmH2O"] > 0.3, (
            f"COPD auto-PEEP {result['auto_peep_cmH2O']:.2f} too low; expected > 0.3"
        )

    def test_bronchospasm_develops_auto_peep(self):
        result = generate_breath_cycles(BRONCHOSPASM_PARAMS, n_cycles=20, seed=7)
        assert result["auto_peep_cmH2O"] > 0.3

    def test_normal_has_minimal_auto_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=15, seed=1)
        assert result["auto_peep_cmH2O"] < 1.0, (
            f"Normal auto-PEEP {result['auto_peep_cmH2O']:.2f} unexpectedly high"
        )

    def test_moderate_ards_more_peep_sensitive_than_copd(self):
        """Moderate ARDS has the strongest recruitment slope in the grid
        (0.90); COPD's is zero. Raising PEEP should move Moderate ARDS's
        converged driving pressure more than it moves COPD's."""
        p_mod_low = {**MODERATE_ARDS_PARAMS, "peep_cmH2O": 5}
        p_mod_hi = {**MODERATE_ARDS_PARAMS, "peep_cmH2O": 15}
        r_mod_low = generate_breath_cycles(p_mod_low, n_cycles=12, seed=3)
        r_mod_hi = generate_breath_cycles(p_mod_hi, n_cycles=12, seed=3)
        mod_delta = abs(r_mod_low["driving_p_cmH2O"] - r_mod_hi["driving_p_cmH2O"])

        p_copd_low = {**COPD_PARAMS, "peep_cmH2O": 5}
        p_copd_hi = {**COPD_PARAMS, "peep_cmH2O": 15}
        r_copd_low = generate_breath_cycles(p_copd_low, n_cycles=12, seed=4)
        r_copd_hi = generate_breath_cycles(p_copd_hi, n_cycles=12, seed=4)
        copd_delta = abs(r_copd_low["driving_p_cmH2O"] - r_copd_hi["driving_p_cmH2O"])

        assert mod_delta > copd_delta, (
            f"Moderate ARDS PEEP-sensitivity ({mod_delta:.1f}) should exceed "
            f"COPD's ({copd_delta:.1f}), given recruitment slopes 0.90 vs 0.0"
        )

# ---------------------------------------------------------------------------
# Class 8 — Mechanics refinement parameters
# ---------------------------------------------------------------------------

class TestMechanicsRefinementParameters:
    """
    Same refinement params as VCV/PCV. PRVC's adaptive controller targets
    vt_target_ml directly, so effects here surface as a shift in the
    converged driving_p_cmH2O rather than a direct Vt or Ppeak change.
    """

    # -- circuit_compensated ------------------------------------------------

    def test_circuit_compensated_false_reduces_delivered_vt(self):
        r_comp = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=58)
        p_uncomp = {**NORMAL_PARAMS, "circuit_compensated": False}
        r_uncomp = generate_breath_cycles(p_uncomp, n_cycles=12, seed=58)
        assert r_uncomp["delivered_vt_ml"] < r_comp["delivered_vt_ml"]

    def test_circuit_compensated_true_matches_default(self):
        r_default = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=59)
        p_explicit = {**NORMAL_PARAMS, "circuit_compensated": True}
        r_explicit = generate_breath_cycles(p_explicit, n_cycles=12, seed=59)
        assert r_default["delivered_vt_ml"] == pytest.approx(
            r_explicit["delivered_vt_ml"], abs=1e-6
        )

    # -- chest_wall_compliance_ml_per_cmH2O ---------------------------------

    def test_restrictive_chest_wall_requires_higher_driving_pressure(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=60)
        p_restricted = {**NORMAL_PARAMS,
                         "chest_wall_compliance_ml_per_cmH2O": 30.0}
        r_restricted = generate_breath_cycles(p_restricted, n_cycles=12, seed=60)
        assert r_restricted["driving_p_cmH2O"] > r_normal["driving_p_cmH2O"]

    def test_more_restrictive_chest_wall_raises_driving_pressure_further(self):
        p_mild = {**NORMAL_PARAMS, "chest_wall_compliance_ml_per_cmH2O": 60.0}
        p_severe = {**NORMAL_PARAMS, "chest_wall_compliance_ml_per_cmH2O": 20.0}
        r_mild = generate_breath_cycles(p_mild, n_cycles=12, seed=61)
        r_severe = generate_breath_cycles(p_severe, n_cycles=12, seed=61)
        assert r_severe["driving_p_cmH2O"] > r_mild["driving_p_cmH2O"]

    # -- recruitment_slope override -----------------------------------------

    def test_recruitment_slope_override_beats_copd_zero_default(self):
        p_low_peep = {**COPD_PARAMS, "peep_cmH2O": 5, "recruitment_slope": 2.0}
        p_high_peep = {**p_low_peep, "peep_cmH2O": 15}
        r_low = generate_breath_cycles(p_low_peep, n_cycles=12, seed=62)
        r_high = generate_breath_cycles(p_high_peep, n_cycles=12, seed=62)
        assert r_high["driving_p_cmH2O"] < r_low["driving_p_cmH2O"]

    def test_recruitment_slope_default_matches_condition_lookup(self):
        p_implicit = {**MODERATE_ARDS_PARAMS, "peep_cmH2O": 15}
        p_explicit = {**p_implicit,
                      "recruitment_slope": RECRUITMENT_SLOPES["Moderate ARDS"]}
        r_implicit = generate_breath_cycles(p_implicit, n_cycles=12, seed=63)
        r_explicit = generate_breath_cycles(p_explicit, n_cycles=12, seed=63)
        assert r_implicit["driving_p_cmH2O"] == pytest.approx(
            r_explicit["driving_p_cmH2O"], abs=1e-6
        )

    # -- peep_reference_cmH2O override --------------------------------------

    def test_peep_reference_override_suppresses_recruitment(self):
        base = {**MODERATE_ARDS_PARAMS, "peep_cmH2O": 10}
        r_default_ref = generate_breath_cycles(base, n_cycles=12, seed=64)
        p_high_ref = {**base, "peep_reference_cmH2O": 12.0}
        r_high_ref = generate_breath_cycles(p_high_ref, n_cycles=12, seed=64)
        assert r_high_ref["driving_p_cmH2O"] > r_default_ref["driving_p_cmH2O"]

    def test_lower_peep_reference_extends_recruitment_range(self):
        base = {**MODERATE_ARDS_PARAMS, "peep_cmH2O": 10,
                "peep_reference_cmH2O": 5.0}
        p_low_ref = {**base, "peep_reference_cmH2O": 0.0}
        r_normal_ref = generate_breath_cycles(base, n_cycles=12, seed=65)
        r_low_ref = generate_breath_cycles(p_low_ref, n_cycles=12, seed=65)
        assert r_low_ref["driving_p_cmH2O"] < r_normal_ref["driving_p_cmH2O"]
# ---------------------------------------------------------------------------
# Class 9 — Physiological directions
# ---------------------------------------------------------------------------

class TestPhysiologicalDirections:
    """
    Cross-parameter monotonicity checks for the outer-loop-adapted
    working pressure and its derived metrics. PRVC's driving pressure
    isn't set directly — it's whatever the adaptive loop converges (or
    clips) to — so these confirm the controller moves the right
    direction as mechanics and settings change.
    """

    def test_higher_peep_increases_mean_paw(self):
        p_low = {**NORMAL_PARAMS, "peep_cmH2O": 0}
        p_high = {**NORMAL_PARAMS, "peep_cmH2O": 15}
        r_low = generate_breath_cycles(p_low, n_cycles=10, seed=50)
        r_high = generate_breath_cycles(p_high, n_cycles=10, seed=50)
        assert r_high["mean_paw_cmH2O"] > r_low["mean_paw_cmH2O"]

    def test_lower_compliance_requires_higher_driving_pressure(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=51)
        r_ards = generate_breath_cycles(SEVERE_ARDS_PARAMS, n_cycles=12, seed=51)
        assert r_ards["driving_p_cmH2O"] > r_normal["driving_p_cmH2O"]

    def test_higher_vt_target_requires_higher_driving_pressure(self):
        p_low_vt = {**NORMAL_PARAMS, "vt_target_ml": 300.0}
        p_high_vt = {**NORMAL_PARAMS, "vt_target_ml": 600.0}
        r_low = generate_breath_cycles(p_low_vt, n_cycles=12, seed=52)
        r_high = generate_breath_cycles(p_high_vt, n_cycles=12, seed=52)
        assert r_high["driving_p_cmH2O"] >= r_low["driving_p_cmH2O"] - 0.5

    def test_higher_resistance_increases_auto_peep(self):
        p_low_r = {**COPD_PARAMS, "resistance_cmH2O_L_s": 15.0}
        p_high_r = {**COPD_PARAMS, "resistance_cmH2O_L_s": 30.0}
        r_low = generate_breath_cycles(p_low_r, n_cycles=15, seed=53)
        r_high = generate_breath_cycles(p_high_r, n_cycles=15, seed=53)
        assert r_high["auto_peep_cmH2O"] >= r_low["auto_peep_cmH2O"] - 0.1

    def test_higher_ie_ratio_shortens_expiratory_time_raises_auto_peep(self):
        p_long_exp = {**COPD_PARAMS, "ie_ratio": 0.33}
        p_short_exp = {**COPD_PARAMS, "ie_ratio": 1.0}
        r_long = generate_breath_cycles(p_long_exp, n_cycles=15, seed=57)
        r_short = generate_breath_cycles(p_short_exp, n_cycles=15, seed=57)
        assert r_short["auto_peep_cmH2O"] >= r_long["auto_peep_cmH2O"] - 0.1

    def test_higher_respiratory_rate_raises_minute_ventilation(self):
        p_slow = {**NORMAL_PARAMS, "respiratory_rate": 10}
        p_fast = {**NORMAL_PARAMS, "respiratory_rate": 25}
        r_slow = generate_breath_cycles(p_slow, n_cycles=10, seed=55)
        r_fast = generate_breath_cycles(p_fast, n_cycles=10, seed=55)
        assert r_fast["minute_vent_l"] > r_slow["minute_vent_l"]

    def test_bronchospasm_ppeak_exceeds_normal(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS, n_cycles=12, seed=56)
        r_broncho = generate_breath_cycles(BRONCHOSPASM_PARAMS, n_cycles=12, seed=56)
        assert r_broncho["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]

# ---------------------------------------------------------------------------
# Class 10 — Validity filter
# ---------------------------------------------------------------------------

class TestValidityFilter:
    """Validation must reject barotrauma and out-of-range converged Vt,
    while retaining ceiling-limited non-convergence as valid."""

    def test_ppeak_threshold_value(self):
        assert PPEAK_MAX_CMHH2O == 50.0

    def test_vt_min_threshold_is_3_ml_per_kg(self):
        assert VT_MIN_ML == IBW_KG * 3

    def test_vt_max_threshold_is_12_ml_per_kg(self):
        assert VT_MAX_ML == IBW_KG * 12

    def test_high_peep_high_ceiling_can_trip_barotrauma(self):
        """peep=20 + ceiling=35 allows working pressure up to 55 cmH2O,
        above the 50 cmH2O barotrauma threshold."""
        params = {**NORMAL_PARAMS, "peep_cmH2O": 20.0, "pressure_ceiling_cmH2O": 35.0,
                  "compliance_ml_per_cmH2O": 15.0, "vt_target_ml": 700.0}
        result = generate_breath_cycles(params, n_cycles=15, seed=9)
        if result["ppeak_cmH2O"] > PPEAK_MAX_CMHH2O:
            assert result["is_valid"] is False
            assert "ppeak" in result["invalid_reason"].lower() or \
                   "barotrauma" in result["invalid_reason"].lower()

    def test_invalid_scenario_has_descriptive_reason(self):
        params = {**NORMAL_PARAMS, "peep_cmH2O": 20.0, "pressure_ceiling_cmH2O": 35.0,
                  "compliance_ml_per_cmH2O": 15.0, "vt_target_ml": 700.0}
        result = generate_breath_cycles(params, n_cycles=15, seed=9)
        if not result["is_valid"]:
            assert len(result["invalid_reason"]) > 5

    def test_borderline_valid_normal_lung(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=0)
        assert result["is_valid"] is True


# ---------------------------------------------------------------------------
# Class 11 — Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """generate_dataset() must sweep PARAMETER_GRID and return correctly
    structured scenario dicts matching the codebase-wide convention."""

    @pytest.fixture(scope="class")
    @classmethod
    def small_dataset(cls):
        return generate_dataset(
            "Normal", compliance_ml_per_cmH2O=80.0, resistance_cmH2O_L_s=10.0,
            n_cycles=5, max_scenarios=12,
        )

    @pytest.fixture(scope="class")
    @classmethod
    def severe_ards_dataset(cls):
        """Tight-ceiling-prone tier — expect some non-converged/ceiling
        scenarios, though these remain is_valid=True by design."""
        return generate_dataset(
            "Severe ARDS", compliance_ml_per_cmH2O=18.0, resistance_cmH2O_L_s=16.0,
            n_cycles=8, max_scenarios=20,
        )

    def test_returns_list(self, small_dataset):
        assert isinstance(small_dataset, list)

    def test_dataset_nonempty(self, small_dataset):
        assert len(small_dataset) > 0

    def test_max_scenarios_caps_output(self, small_dataset):
        assert len(small_dataset) == 12

    def test_all_scenario_keys_present(self, small_dataset):
        for s in small_dataset:
            missing = DATASET_SCENARIO_KEYS - s.keys()
            assert not missing, f"Scenario missing keys: {missing}"

    def test_scenario_ids_are_unique(self, small_dataset):
        ids = [s["scenario_id"] for s in small_dataset]
        assert len(ids) == len(set(ids)), "Duplicate scenario IDs found"

    def test_scenario_id_contains_condition(self, small_dataset):
        for s in small_dataset:
            assert "NORMAL" in s["scenario_id"].upper()

    def test_valid_scenarios_have_waveforms(self, small_dataset):
        for s in small_dataset:
            if s["is_valid"]:
                assert isinstance(s["waveforms"], dict)
                assert len(s["waveforms"]) > 0, (
                    f"Valid scenario {s['scenario_id']} has empty waveforms"
                )
                assert "pressure_trajectory" in s["waveforms"]

    def test_invalid_scenarios_have_empty_waveforms(self, small_dataset):
        for s in small_dataset:
            if not s["is_valid"]:
                assert s["waveforms"] == {}

    def test_valid_scenarios_have_populated_metrics(self, small_dataset):
        for s in small_dataset:
            if s["is_valid"]:
                assert isinstance(s["metrics"], dict)
                assert len(s["metrics"]) > 0
                assert "converged" in s["metrics"]
                assert "ceiling_limited" in s["metrics"]

    def test_compliance_and_resistance_in_params(self, small_dataset):
        for s in small_dataset:
            assert "compliance_ml_per_cmH2O" in s["params"]
            assert "resistance_cmH2O_L_s" in s["params"]

    def test_condition_field_present(self, small_dataset):
        for s in small_dataset:
            assert s["condition"] == "Normal"

    def test_generated_at_is_populated(self, small_dataset):
        for s in small_dataset:
            assert isinstance(s["generated_at"], str)
            assert len(s["generated_at"]) > 0

    def test_severe_ards_dataset_includes_ceiling_limited_scenarios(self, severe_ards_dataset):
        """At least some scenarios in this tight-ceiling-prone tier should
        show ceiling_limited=True, and they should still be marked valid."""
        ceiling_limited = [s for s in severe_ards_dataset
                            if s["is_valid"] and s["metrics"].get("ceiling_limited")]
        assert len(ceiling_limited) > 0, (
            "Expected at least one ceiling-limited scenario in this sample; "
            "if this fails, the sample may need a larger max_scenarios or "
            "different mechanics point to reliably hit the ceiling"
        )
        for s in ceiling_limited:
            assert s["is_valid"] is True

    def test_uniform_adaptation_step_used_across_dataset(self, small_dataset):
        for s in small_dataset:
            assert s["params"]["adaptation_step_cmH2O"] == PARAMETER_GRID["adaptation_step_cmH2O"][0]
            assert s["params"]["vt_tolerance_frac"] == PARAMETER_GRID["vt_tolerance_frac"][0]

# ---------------------------------------------------------------------------
# Class 12 — Parameter grid
# ---------------------------------------------------------------------------

class TestParameterGrid:
    """
    PARAMETER_GRID must sweep every ventilator-side dimension while
    keeping the two uniform-device constants unswept, per project
    decision (see PARAMETER_GRID note in prvc_generator.py).
    """

    EXPECTED_KEYS = {
        "vt_target_ml_per_kg", "respiratory_rate", "peep_cmH2O",
        "ie_ratio", "pressure_ceiling_cmH2O",
        "adaptation_step_cmH2O", "vt_tolerance_frac",
    }

    def test_grid_has_all_required_keys(self):
        missing = self.EXPECTED_KEYS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_all_grid_values_are_lists(self):
        for key, values in PARAMETER_GRID.items():
            assert isinstance(values, list), f"{key} is not a list"

    def test_ventilator_side_dims_have_real_sweep_range(self):
        sweep_keys = ["vt_target_ml_per_kg", "respiratory_rate",
                      "peep_cmH2O", "ie_ratio", "pressure_ceiling_cmH2O"]
        for k in sweep_keys:
            assert len(PARAMETER_GRID[k]) >= 2, f"{k} needs >= 2 values"

    def test_adaptation_step_and_tolerance_are_uniform_single_values(self):
        """Per project decision, these represent one deployed-device
        algorithm and are deliberately NOT swept per condition."""
        assert len(PARAMETER_GRID["adaptation_step_cmH2O"]) == 1
        assert len(PARAMETER_GRID["vt_tolerance_frac"]) == 1

    def test_vt_target_range_spans_protective_to_standard(self):
        vt = PARAMETER_GRID["vt_target_ml_per_kg"]
        assert min(vt) <= 4
        assert max(vt) >= 10

    def test_pressure_ceiling_range_covers_conservative_to_permissive(self):
        ceil = PARAMETER_GRID["pressure_ceiling_cmH2O"]
        assert min(ceil) <= 15
        assert max(ceil) >= 30

    def test_full_ventilator_grid_combination_count(self):
        keys = ["vt_target_ml_per_kg", "respiratory_rate", "peep_cmH2O",
                "ie_ratio", "pressure_ceiling_cmH2O"]
        expected = 1
        for k in keys:
            expected *= len(PARAMETER_GRID[k])
        assert expected == 2520, (
            f"Full PRVC ventilator-side grid should be 2,520 "
            f"combinations/mechanics point (4x7x6x3x5), got {expected}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
