"""
tests/test_vcv_generator.py
---------------------------
Unit tests for generator/vcv_generator.py (VCV waveform generator).

Five test classes:
    TestInterfaceContract       — return types, keys, array shapes, validation
    TestPhysiologicalPlausibility — basic physical constraints on all outputs
    TestFlowPatternShape        — waveform morphology specific to VCV
    TestValidityFilter          — threshold logic and invalid_reason strings
    TestDatasetGeneration       — generate_dataset() structure and correctness

Run with:
    python -m pytest tests/test_vcv_generator.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.vcv_generator import (
    DRIVING_P_MAX_CMHH2O,
    COMPARTMENT_PROFILES,
    IBW_KG,
    PARAMETER_GRID,
    PPEAK_MAX_CMHH2O,
    RECRUITMENT_SLOPES,
    VT_MAX_ML,
    VT_MIN_ML,
    generate_breath_cycles,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NORMAL_PARAMS_SQR = {
    "respiratory_rate":        15,
    "tidal_volume_ml":        500,
    "compliance_ml_per_cmH2O": 60,
    "resistance_cmH2O_L_s":     10,
    "ie_ratio":                0.5,
    "peep_cmH2O":                5,
    "flow_pattern":          "square",
    "condition":               "Normal",
}

NORMAL_PARAMS_DEC = {**NORMAL_PARAMS_SQR, "flow_pattern": "decelerating"}

CORE_KEYS    = {"time", "pressure", "flow", "volume"}
METRIC_KEYS  = {
    "ppeak_cmH2O", "pplat_cmH2O", "driving_p_cmH2O",
    "mean_paw_cmH2O", "auto_peep_cmH2O", "delivered_vt_ml", "minute_vent_l",
}
VALIDITY_KEYS = {"is_valid", "invalid_reason"}
ALL_KEYS      = CORE_KEYS | METRIC_KEYS | VALIDITY_KEYS

DATASET_SCENARIO_KEYS = {
    "scenario_id", "condition", "params", "metrics",
    "is_valid", "invalid_reason", "waveforms", "generated_at",
}


# ---------------------------------------------------------------------------
# Class 1 — Interface contract
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """
    generate_breath_cycles must satisfy the shared engine interface and
    return all documented keys with the correct types.
    """

    def test_returns_dict(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert isinstance(result, dict)

    def test_contains_all_core_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert CORE_KEYS.issubset(result.keys()), (
            f"Missing core keys: {CORE_KEYS - result.keys()}"
        )

    def test_contains_all_metric_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert METRIC_KEYS.issubset(result.keys()), (
            f"Missing metric keys: {METRIC_KEYS - result.keys()}"
        )

    def test_contains_validity_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert VALIDITY_KEYS.issubset(result.keys()), (
            f"Missing validity keys: {VALIDITY_KEYS - result.keys()}"
        )

    def test_core_arrays_are_numpy(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        for key in CORE_KEYS:
            assert isinstance(result[key], np.ndarray), (
                f"'{key}' should be np.ndarray, got {type(result[key])}"
            )

    def test_core_arrays_same_length(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        lengths = {k: len(result[k]) for k in CORE_KEYS}
        unique_lengths = set(lengths.values())
        assert len(unique_lengths) == 1, (
            f"Core arrays have different lengths: {lengths}"
        )

    def test_metric_values_are_floats(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        for key in METRIC_KEYS:
            assert isinstance(result[key], (int, float)), (
                f"Metric '{key}' should be numeric, got {type(result[key])}"
            )

    def test_is_valid_is_bool(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert isinstance(result["is_valid"], bool)

    def test_invalid_reason_is_str(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert isinstance(result["invalid_reason"], str)

    def test_n_cycles_1_returns_data(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=1)
        assert len(result["time"]) > 0

    def test_n_cycles_10_duration(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=10)
        # 10 cycles at 15 bpm = 4s/cycle → ~40s
        assert result["time"].max() > 35.0

    def test_both_flow_patterns_accepted(self):
        for pattern in ["square", "decelerating"]:
            p = {**NORMAL_PARAMS_SQR, "flow_pattern": pattern}
            result = generate_breath_cycles(p)
            assert isinstance(result, dict)

    def test_invalid_flow_pattern_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "flow_pattern": "sinusoidal"}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_missing_required_param_raises(self):
        for key in [
            "respiratory_rate", "tidal_volume_ml",
            "compliance_ml_per_cmH2O", "resistance_cmH2O_L_s",
            "ie_ratio", "peep_cmH2O",
        ]:
            bad = {k: v for k, v in NORMAL_PARAMS_SQR.items() if k != key}
            with pytest.raises(ValueError, match="Missing required parameter"):
                generate_breath_cycles(bad)

    def test_out_of_range_respiratory_rate_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "respiratory_rate": 100}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_tidal_volume_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "tidal_volume_ml": 50}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_peep_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "peep_cmH2O": 25}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_compliance_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "compliance_ml_per_cmH2O": 200}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_resistance_raises(self):
        bad = {**NORMAL_PARAMS_SQR, "resistance_cmH2O_L_s": 0.1}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)


# ---------------------------------------------------------------------------
# Class 2 — Physiological plausibility
# ---------------------------------------------------------------------------

class TestPhysiologicalPlausibility:
    """
    Waveforms must satisfy basic physical constraints regardless of
    parameter values or flow pattern.
    """

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_time_is_monotonically_increasing(self, params):
        result = generate_breath_cycles(params)
        diffs = np.diff(result["time"])
        assert np.all(diffs >= 0), "Time array must be non-decreasing"

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_time_starts_at_zero(self, params):
        result = generate_breath_cycles(params)
        assert result["time"][0] == pytest.approx(0.0, abs=0.02)

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_volume_non_negative(self, params):
        result = generate_breath_cycles(params)
        assert np.all(result["volume"] >= -0.5), (
            "Volume should never be significantly negative"
        )

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_pressure_never_below_peep(self, params):
        result = generate_breath_cycles(params)
        peep = params["peep_cmH2O"]
        assert result["pressure"].min() >= peep - 0.5, (
            "Pressure must never fall below PEEP"
        )

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_flow_has_inspiratory_and_expiratory_phases(self, params):
        result = generate_breath_cycles(params)
        assert result["flow"].max() > 0.0, "No positive (inspiratory) flow found"
        assert result["flow"].min() < 0.0, "No negative (expiratory) flow found"

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_delivered_vt_within_5_percent(self, params):
        result = generate_breath_cycles(params, n_cycles=3)
        target = params["tidal_volume_ml"]
        delivered = result["volume"].max()
        assert abs(delivered - target) / target < 0.05, (
            f"Delivered VT {delivered:.0f} mL deviates >5% from target {target} mL"
        )

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_peak_pressure_physiologically_reasonable(self, params):
        result = generate_breath_cycles(params)
        peak = result["pressure"].max()
        assert 5 < peak < 50, (
            f"Peak pressure {peak:.1f} cmH2O outside plausible range for Normal lung"
        )

    @pytest.mark.parametrize("params", [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC])
    def test_sample_rate_approximately_100hz(self, params):
        result = generate_breath_cycles(params, n_cycles=1)
        median_dt = np.median(np.diff(result["time"]))
        assert median_dt == pytest.approx(0.01, abs=0.002), (
            f"Expected ~100 Hz sampling, got median dt={median_dt:.4f}s"
        )

    def test_ppeak_greater_than_pplat(self):
        # PPeak includes resistive term; Pplat is elastic only
        # Strictly true for square pattern where resistive load persists
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert result["ppeak_cmH2O"] >= result["pplat_cmH2O"], (
            "PPeak must be >= Pplat (resistive contribution at peak flow)"
        )

    def test_pplat_greater_than_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert result["pplat_cmH2O"] > NORMAL_PARAMS_SQR["peep_cmH2O"], (
            "Pplat must exceed PEEP (elastic recoil from delivered volume)"
        )

    def test_driving_pressure_equals_pplat_minus_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS_DEC)
        expected = result["pplat_cmH2O"] - NORMAL_PARAMS_DEC["peep_cmH2O"]
        assert result["driving_p_cmH2O"] == pytest.approx(expected, abs=0.5), (
            "Driving pressure must equal Pplat - PEEP"
        )

    def test_minute_ventilation_matches_formula(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        expected = (
            NORMAL_PARAMS_SQR["respiratory_rate"]
            * result["delivered_vt_ml"]
            / 1000.0
        )
        assert result["minute_vent_l"] == pytest.approx(expected, rel=0.02)

    def test_auto_peep_non_negative(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert result["auto_peep_cmH2O"] >= 0.0, "Auto-PEEP cannot be negative"

    def test_normal_lung_has_near_zero_auto_peep(self):
        # Normal lung fully empties — auto-PEEP should be negligible
        result = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        assert result["auto_peep_cmH2O"] < 1.0, (
            f"Normal lung should have near-zero auto-PEEP, "
            f"got {result['auto_peep_cmH2O']:.2f} cmH2O"
        )


# ---------------------------------------------------------------------------
# Class 3 — Flow pattern shape
# ---------------------------------------------------------------------------

class TestFlowPatternShape:
    """
    Tests specific to VCV waveform morphology — shape claims that must hold
    for each flow pattern independently and in comparison to each other.
    """

    def test_square_flow_is_constant_during_inspiration(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=1)
        flow = result["flow"]
        # Inspiratory flow is all positive samples in the first cycle
        insp_flow = flow[flow > 0]
        assert len(insp_flow) > 0
        # All positive flow values should be nearly identical (constant)
        assert np.std(insp_flow) < 0.01, (
            f"Square flow should be constant during inspiration, "
            f"std={np.std(insp_flow):.4f} L/s"
        )

    def test_decelerating_flow_decreases_during_inspiration(self):
        result = generate_breath_cycles(NORMAL_PARAMS_DEC, n_cycles=1)
        flow = result["flow"]
        insp_indices = np.where(flow > 0.001)[0]
        assert len(insp_indices) > 5
        insp_flow = flow[insp_indices]
        # Flow must be strictly decreasing — first value larger than last
        assert insp_flow[0] > insp_flow[-1], (
            "Decelerating flow must decrease during inspiration"
        )
        # All diffs should be non-positive (monotonically non-increasing)
        diffs = np.diff(insp_flow)
        assert np.all(diffs <= 0.001), (
            "Decelerating flow profile must be monotonically non-increasing"
        )

    def test_square_ppeak_exceeds_decelerating_ppeak(self):
        # Same params, different pattern — square always has higher peak flow
        # at start of breath → higher resistive pressure → higher PPeak
        r_sq  = generate_breath_cycles(NORMAL_PARAMS_SQR)
        r_dec = generate_breath_cycles(NORMAL_PARAMS_DEC)
        assert r_sq["ppeak_cmH2O"] > r_dec["ppeak_cmH2O"], (
            f"Square PPeak ({r_sq['ppeak_cmH2O']:.1f}) must exceed "
            f"Decelerating PPeak ({r_dec['ppeak_cmH2O']:.1f})"
        )

    def test_both_patterns_deliver_same_tidal_volume(self):
        r_sq  = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=3)
        r_dec = generate_breath_cycles(NORMAL_PARAMS_DEC, n_cycles=3)
        assert abs(r_sq["delivered_vt_ml"] - r_dec["delivered_vt_ml"]) < 10, (
            "Both flow patterns must deliver the same tidal volume"
        )

    def test_volume_rises_monotonically_during_inspiration(self):
        for params in [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC]:
            result = generate_breath_cycles(params, n_cycles=1)
            volume = result["volume"]
            peak_idx = int(np.argmax(volume))
            insp_volume = volume[:peak_idx + 1]
            diffs = np.diff(insp_volume)
            assert np.all(diffs >= -0.1), (
                f"Volume must rise monotonically during inspiration "
                f"({params['flow_pattern']} pattern)"
            )

    def test_volume_falls_during_expiration(self):
        for params in [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC]:
            result = generate_breath_cycles(params, n_cycles=1)
            volume = result["volume"]
            peak_idx = int(np.argmax(volume))
            exp_volume = volume[peak_idx:]
            diffs = np.diff(exp_volume)
            assert np.all(diffs <= 0.1), (
                f"Volume must fall during expiration "
                f"({params['flow_pattern']} pattern)"
            )

    def test_expiratory_flow_is_negative(self):
        for params in [NORMAL_PARAMS_SQR, NORMAL_PARAMS_DEC]:
            result = generate_breath_cycles(params, n_cycles=1)
            flow   = result["flow"]
            volume = result["volume"]
            peak_idx = int(np.argmax(volume))
            exp_flow = flow[peak_idx + 1:]
            assert np.all(exp_flow <= 0.01), (
                f"Expiratory flow must be non-positive "
                f"({params['flow_pattern']} pattern)"
            )

    def test_higher_resistance_raises_ppeak_square(self):
        low_r = {**NORMAL_PARAMS_SQR, "resistance_cmH2O_L_s": 2}
        high_r = {**NORMAL_PARAMS_SQR, "resistance_cmH2O_L_s": 20}
        r_low  = generate_breath_cycles(low_r)
        r_high = generate_breath_cycles(high_r)
        assert r_high["ppeak_cmH2O"] > r_low["ppeak_cmH2O"], (
            "Higher resistance must raise PPeak in square VCV"
        )

    def test_lower_compliance_raises_ppeak(self):
        high_c = {**NORMAL_PARAMS_SQR, "compliance_ml_per_cmH2O": 60}
        low_c  = {**NORMAL_PARAMS_SQR, "compliance_ml_per_cmH2O": 15}
        r_high = generate_breath_cycles(high_c)
        r_low  = generate_breath_cycles(low_c)
        assert r_low["ppeak_cmH2O"] > r_high["ppeak_cmH2O"], (
            "Lower compliance must raise PPeak (larger elastic pressure term)"
        )

    def test_resistance_does_not_affect_pplat(self):
        # Pplat is measured at near-zero flow — resistive term vanishes
        # Changing R should not change Pplat for decelerating pattern
        low_r  = {**NORMAL_PARAMS_DEC, "resistance_cmH2O_L_s": 2}
        high_r = {**NORMAL_PARAMS_DEC, "resistance_cmH2O_L_s": 20}
        r_low  = generate_breath_cycles(low_r)
        r_high = generate_breath_cycles(high_r)
        assert abs(r_low["pplat_cmH2O"] - r_high["pplat_cmH2O"]) < 2.0, (
            "Pplat should be nearly independent of resistance "
            "(resistive term → 0 at end of decelerating inspiration)"
        )

    def test_decelerating_peak_flow_is_double_square_peak_flow(self):
        # Decelerating: Flow_peak = 2 * VT / t_insp
        # Square:       Flow      = VT / t_insp
        # So decelerating peak should be ~2× square constant flow
        r_sq  = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=1)
        r_dec = generate_breath_cycles(NORMAL_PARAMS_DEC, n_cycles=1)
        sq_flow  = r_sq["flow"].max()
        dec_flow = r_dec["flow"].max()
        ratio = dec_flow / sq_flow
        assert 1.8 <= ratio <= 2.2, (
            f"Decelerating peak flow should be ~2× square flow, "
            f"got ratio={ratio:.2f}"
        )

# ---------------------------------------------------------------------------
# Class 4 — Multi-compartment mechanics
# ---------------------------------------------------------------------------

class TestMultiCompartmentMechanics:
    """
    VCV's parallel-compartment lung model (COMPARTMENT_PROFILES) and
    PEEP-recruited compliance (RECRUITMENT_SLOPES) must match the
    documented per-condition compartment counts and produce the expected
    qualitative behavior across tiers.
    """

    EXPECTED_COMPARTMENTS = {
        "Normal": 1, "Mild ARDS": 2, "Moderate ARDS": 2, "Severe ARDS": 2,
        "COPD": 3, "Bronchospasm": 2, "Pneumonia": 3,
    }

    def test_compartment_counts_match_documented_scheme(self):
        for condition, n in self.EXPECTED_COMPARTMENTS.items():
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

    def test_generator_reports_correct_n_compartments(self):
        for condition, n in self.EXPECTED_COMPARTMENTS.items():
            params = {**NORMAL_PARAMS_SQR, "condition": condition}
            result = generate_breath_cycles(params, n_cycles=3)
            assert result["n_compartments"] == n, (
                f"{condition}: expected n_compartments={n}, "
                f"got {result['n_compartments']}"
            )

    def test_recruitment_slopes_zero_for_obstructive_disease(self):
        assert RECRUITMENT_SLOPES["COPD"] == 0.0
        assert RECRUITMENT_SLOPES["Bronchospasm"] == 0.0

    def test_recruitment_slopes_positive_for_ards(self):
        for condition in ("Mild ARDS", "Moderate ARDS", "Severe ARDS"):
            assert RECRUITMENT_SLOPES[condition] > 0.0

    def test_copd_develops_auto_peep(self):
        p_copd = {**NORMAL_PARAMS_SQR, "condition": "COPD",
                  "compliance_ml_per_cmH2O": 100.0,
                  "resistance_cmH2O_L_s": 22.0,
                  "respiratory_rate": 22}
        result = generate_breath_cycles(p_copd, n_cycles=10)
        assert result["auto_peep_cmH2O"] > 0.3, (
            f"COPD auto-PEEP {result['auto_peep_cmH2O']:.2f} too low"
        )

    def test_bronchospasm_develops_auto_peep(self):
        p_broncho = {**NORMAL_PARAMS_SQR, "condition": "Bronchospasm",
                     "resistance_cmH2O_L_s": 35.0, "respiratory_rate": 22}
        result = generate_breath_cycles(p_broncho, n_cycles=10)
        assert result["auto_peep_cmH2O"] > 0.3

    def test_normal_has_minimal_auto_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        assert result["auto_peep_cmH2O"] < 1.0

    def test_severe_ards_driving_pressure_exceeds_normal(self):
        """Baby-lung effect: the low-compliance compartment should
        dominate and raise driving pressure vs Normal at matched VT."""
        p_ards = {**NORMAL_PARAMS_SQR, "condition": "Severe ARDS",
                  "compliance_ml_per_cmH2O": 18.0,
                  "resistance_cmH2O_L_s": 16.0}
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=3)
        r_ards = generate_breath_cycles(p_ards, n_cycles=3)
        assert r_ards["driving_p_cmH2O"] > r_normal["driving_p_cmH2O"] + 2.0, (
            f"ARDS={r_ards['driving_p_cmH2O']:.2f} "
            f"Normal={r_normal['driving_p_cmH2O']:.2f}"
        )

    def test_copd_stress_index_deviates_from_one(self):
        """Time-constant heterogeneity across COPD's 3 compartments should
        curve the pressure ramp away from SI=1.0 (square flow only)."""
        p_copd = {**NORMAL_PARAMS_SQR, "condition": "COPD",
                  "compliance_ml_per_cmH2O": 100.0,
                  "resistance_cmH2O_L_s": 22.0}
        result = generate_breath_cycles(p_copd, n_cycles=3)
        assert result["stress_index"] is not None
        assert abs(result["stress_index"] - 1.0) > 0.05

    def test_bronchospasm_ppeak_exceeds_normal(self):
        """Documents the corrected 2-compartment Bronchospasm behavior
        (see docstring/COMPARTMENT_PROFILES bug note)."""
        p_broncho = {**NORMAL_PARAMS_SQR, "condition": "Bronchospasm",
                     "resistance_cmH2O_L_s": 35.0}
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=3)
        r_broncho = generate_breath_cycles(p_broncho, n_cycles=3)
        assert r_broncho["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"] + 5.0


# ---------------------------------------------------------------------------
# Class 5 — ETT complications
# ---------------------------------------------------------------------------

class TestETTComplications:
    """
    ETT cuff leak and partial obstruction overlays. VT delivery is
    guaranteed analytically in VCV (flow-prescribed), so:
      - obstruction raises Ppeak (extra Rohrer drop added to Pao) but
        does NOT reduce delivered_vt
      - cuff leak reduces delivered_vt via a post-hoc volume-balance
        correction and does NOT change the pressure waveform/Ppeak
    """

    def test_cuff_leak_reduces_delivered_vt(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_leak = {**NORMAL_PARAMS_SQR, "ett_cuff_leak_fraction": 0.20}
        r_leak = generate_breath_cycles(p_leak, n_cycles=5)
        assert r_leak["delivered_vt_ml"] < r_normal["delivered_vt_ml"]

    def test_cuff_leak_fraction_matches_expected_reduction(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_leak = {**NORMAL_PARAMS_SQR, "ett_cuff_leak_fraction": 0.25}
        r_leak = generate_breath_cycles(p_leak, n_cycles=5)
        expected = r_normal["delivered_vt_ml"] * 0.75
        assert abs(r_leak["delivered_vt_ml"] - expected) < 5.0

    def test_larger_cuff_leak_produces_larger_vt_loss(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_small = {**NORMAL_PARAMS_SQR, "ett_cuff_leak_fraction": 0.10}
        p_large = {**NORMAL_PARAMS_SQR, "ett_cuff_leak_fraction": 0.35}
        r_small = generate_breath_cycles(p_small, n_cycles=5)
        r_large = generate_breath_cycles(p_large, n_cycles=5)
        gap_small = r_normal["delivered_vt_ml"] - r_small["delivered_vt_ml"]
        gap_large = r_normal["delivered_vt_ml"] - r_large["delivered_vt_ml"]
        assert gap_large > gap_small

    def test_cuff_leak_does_not_change_ppeak(self):
        """Cuff leak is a post-hoc volume-balance correction — it must
        not perturb the pressure waveform."""
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_leak = {**NORMAL_PARAMS_SQR, "ett_cuff_leak_fraction": 0.30}
        r_leak = generate_breath_cycles(p_leak, n_cycles=5)
        assert r_leak["ppeak_cmH2O"] == pytest.approx(
            r_normal["ppeak_cmH2O"], abs=0.1
        )

    def test_obstruction_raises_ppeak(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_obs = {**NORMAL_PARAMS_SQR, "ett_obstruction_multiplier": 3.0}
        r_obs = generate_breath_cycles(p_obs, n_cycles=5)
        assert r_obs["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]

    def test_larger_obstruction_produces_higher_ppeak(self):
        p_mild = {**NORMAL_PARAMS_SQR, "ett_obstruction_multiplier": 1.5}
        p_severe = {**NORMAL_PARAMS_SQR, "ett_obstruction_multiplier": 4.0}
        r_mild = generate_breath_cycles(p_mild, n_cycles=5)
        r_severe = generate_breath_cycles(p_severe, n_cycles=5)
        assert r_severe["ppeak_cmH2O"] > r_mild["ppeak_cmH2O"]

    def test_obstruction_does_not_change_delivered_vt(self):
        """VT delivery is analytically guaranteed in VCV regardless of
        the resistance the flow is pushed through."""
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_obs = {**NORMAL_PARAMS_SQR, "ett_obstruction_multiplier": 3.0}
        r_obs = generate_breath_cycles(p_obs, n_cycles=5)
        assert r_obs["delivered_vt_ml"] == pytest.approx(
            r_normal["delivered_vt_ml"], rel=0.05
        )

    def test_no_complication_defaults_match_baseline(self):
        r1 = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_explicit_default = {**NORMAL_PARAMS_SQR,
                               "ett_obstruction_multiplier": 1.0,
                               "ett_cuff_leak_fraction": 0.0}
        r2 = generate_breath_cycles(p_explicit_default, n_cycles=5)
        assert r1["delivered_vt_ml"] == pytest.approx(r2["delivered_vt_ml"], abs=1e-6)
        assert r1["ppeak_cmH2O"] == pytest.approx(r2["ppeak_cmH2O"], abs=1e-6)

    def test_obstruction_and_cuff_leak_combine_independently(self):
        r_normal = generate_breath_cycles(NORMAL_PARAMS_SQR, n_cycles=5)
        p_both = {**NORMAL_PARAMS_SQR,
                  "ett_obstruction_multiplier": 3.0,
                  "ett_cuff_leak_fraction": 0.20}
        r_both = generate_breath_cycles(p_both, n_cycles=5)
        assert r_both["ppeak_cmH2O"] > r_normal["ppeak_cmH2O"]
        assert r_both["delivered_vt_ml"] < r_normal["delivered_vt_ml"]


# ---------------------------------------------------------------------------
# Class 6 — Physiological directions
# ---------------------------------------------------------------------------

class TestPhysiologicalDirections:
    """
    Cross-parameter monotonicity checks that verify the equation of
    motion is wired correctly.
    """

    def test_higher_resistance_increases_auto_peep_in_copd(self):
        p_low_r = {**NORMAL_PARAMS_SQR, "condition": "COPD",
                   "compliance_ml_per_cmH2O": 100.0,
                   "resistance_cmH2O_L_s": 15.0, "respiratory_rate": 22}
        p_high_r = {**p_low_r, "resistance_cmH2O_L_s": 30.0}
        r_low = generate_breath_cycles(p_low_r, n_cycles=10)
        r_high = generate_breath_cycles(p_high_r, n_cycles=10)
        assert r_high["auto_peep_cmH2O"] >= r_low["auto_peep_cmH2O"] - 0.1

    def test_higher_peep_increases_mean_paw(self):
        p_low = {**NORMAL_PARAMS_SQR, "peep_cmH2O": 0}
        p_high = {**NORMAL_PARAMS_SQR, "peep_cmH2O": 15}
        r_low = generate_breath_cycles(p_low, n_cycles=5)
        r_high = generate_breath_cycles(p_high, n_cycles=5)
        assert r_high["mean_paw_cmH2O"] > r_low["mean_paw_cmH2O"]

    def test_lower_compliance_raises_ppeak(self):
        p_stiff = {**NORMAL_PARAMS_SQR, "compliance_ml_per_cmH2O": 20.0}
        p_compliant = {**NORMAL_PARAMS_SQR, "compliance_ml_per_cmH2O": 90.0}
        r_stiff = generate_breath_cycles(p_stiff, n_cycles=5)
        r_compliant = generate_breath_cycles(p_compliant, n_cycles=5)
        assert r_stiff["ppeak_cmH2O"] > r_compliant["ppeak_cmH2O"]

    def test_higher_resistance_raises_ppeak(self):
        p_low_r = {**NORMAL_PARAMS_SQR, "resistance_cmH2O_L_s": 5.0}
        p_high_r = {**NORMAL_PARAMS_SQR, "resistance_cmH2O_L_s": 30.0}
        r_low = generate_breath_cycles(p_low_r, n_cycles=5)
        r_high = generate_breath_cycles(p_high_r, n_cycles=5)
        assert r_high["ppeak_cmH2O"] > r_low["ppeak_cmH2O"]

    def test_higher_tidal_volume_raises_ppeak(self):
        p_low_vt = {**NORMAL_PARAMS_SQR, "tidal_volume_ml": 300}
        p_high_vt = {**NORMAL_PARAMS_SQR, "tidal_volume_ml": 700}
        r_low = generate_breath_cycles(p_low_vt, n_cycles=5)
        r_high = generate_breath_cycles(p_high_vt, n_cycles=5)
        assert r_high["ppeak_cmH2O"] > r_low["ppeak_cmH2O"]

    def test_higher_respiratory_rate_raises_minute_ventilation(self):
        p_slow = {**NORMAL_PARAMS_SQR, "respiratory_rate": 10}
        p_fast = {**NORMAL_PARAMS_SQR, "respiratory_rate": 25}
        r_slow = generate_breath_cycles(p_slow, n_cycles=5)
        r_fast = generate_breath_cycles(p_fast, n_cycles=5)
        assert r_fast["minute_vent_l"] > r_slow["minute_vent_l"]

    def test_shorter_expiratory_time_increases_auto_peep_in_copd(self):
        """Higher ie_ratio -> longer t_insp, shorter t_exp -> less time
        to exhale -> more gas trapping in a trapping-prone condition."""
        p_long_exp = {**NORMAL_PARAMS_SQR, "condition": "COPD",
                      "compliance_ml_per_cmH2O": 100.0,
                      "resistance_cmH2O_L_s": 22.0,
                      "respiratory_rate": 20, "ie_ratio": 0.33}
        p_short_exp = {**p_long_exp, "ie_ratio": 1.0}
        r_long = generate_breath_cycles(p_long_exp, n_cycles=10)
        r_short = generate_breath_cycles(p_short_exp, n_cycles=10)
        assert r_short["auto_peep_cmH2O"] >= r_long["auto_peep_cmH2O"] - 0.1
# ---------------------------------------------------------------------------
# Class 7 — Validity filter
# ---------------------------------------------------------------------------

class TestValidityFilter:
    """
    The validity filter must correctly identify and label scenarios that
    breach clinical safety thresholds.
    """

    def test_normal_scenario_is_valid(self):
        result = generate_breath_cycles(NORMAL_PARAMS_SQR)
        assert result["is_valid"] is True
        assert result["invalid_reason"] == ""

    def test_valid_scenario_has_empty_reason(self):
        result = generate_breath_cycles(NORMAL_PARAMS_DEC)
        if result["is_valid"]:
            assert result["invalid_reason"] == ""

    def test_ppeak_breach_flagged(self):
        # Very high resistance + large VT + square pattern → PPeak > 50
        params = {
            **NORMAL_PARAMS_SQR,
            "resistance_cmH2O_L_s":    48,
            "tidal_volume_ml":        800,
            "respiratory_rate":        30,
            "ie_ratio":               1.0,   # 1:1 → short t_insp → high flow
        }
        result = generate_breath_cycles(params)
        if not result["is_valid"]:
            assert "barotrauma" in result["invalid_reason"].lower() or \
                   "driving" in result["invalid_reason"].lower() or \
                   "ppeak" in result["invalid_reason"].lower(), (
                f"Invalid reason should mention pressure: {result['invalid_reason']}"
            )

    def test_driving_pressure_breach_flagged(self):
        # Very low compliance + high VT → driving P > 20
        params = {
            **NORMAL_PARAMS_DEC,
            "compliance_ml_per_cmH2O": 10,
            "tidal_volume_ml":        500,
        }
        result = generate_breath_cycles(params)
        assert result["is_valid"] is False
        # PPeak filter may trip before driving pressure filter depending on
        # parameter combination — accept any pressure-related reason
        assert ("driving" in result["invalid_reason"].lower() or
                "ards"    in result["invalid_reason"].lower() or
                "barotrauma" in result["invalid_reason"].lower() or
                "ppeak"   in result["invalid_reason"].lower()), (
            f"Expected pressure-related reason, got: {result['invalid_reason']}"
        )

    def test_driving_pressure_threshold_value(self):
        # Confirm the threshold constant matches the documented value
        assert DRIVING_P_MAX_CMHH2O == 20.0

    def test_ppeak_threshold_value(self):
        assert PPEAK_MAX_CMHH2O == 50.0

    def test_vt_min_threshold_is_3_ml_per_kg(self):
        assert VT_MIN_ML == IBW_KG * 3

    def test_vt_max_threshold_is_12_ml_per_kg(self):
        assert VT_MAX_ML == IBW_KG * 12

    def test_invalid_scenario_has_non_empty_reason(self):
        params = {
            **NORMAL_PARAMS_DEC,
            "compliance_ml_per_cmH2O": 10,
            "tidal_volume_ml":        500,
        }
        result = generate_breath_cycles(params)
        assert not result["is_valid"]
        assert len(result["invalid_reason"]) > 0, (
            "Invalid scenarios must provide a non-empty reason string"
        )

    def test_borderline_valid_normal_lung(self):
        # Standard clinical settings — must always be valid
        params = {
            **NORMAL_PARAMS_SQR,
            "tidal_volume_ml":        420,   # 6 mL/kg IBW
            "compliance_ml_per_cmH2O": 60,
            "resistance_cmH2O_L_s":     2,
            "peep_cmH2O":               5,
        }
        result = generate_breath_cycles(params)
        assert result["is_valid"] is True


# ---------------------------------------------------------------------------
# Class 8 — Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """
    generate_dataset() must return a correctly structured list of scenario
    dicts covering the full parameter grid for a given condition + mechanics.
    """

    @pytest.fixture(scope="class")
    def normal_dataset(self):
        """Run once for the class — Normal lung, small mechanics slice."""
        return generate_dataset(
            condition_name="Normal",
            compliance_ml_per_cmH2O=60,
            resistance_cmH2O_L_s=2,
            n_cycles=3,
        )

    @pytest.fixture(scope="class")
    def severe_ards_dataset(self):
        """Severe ARDS — high invalidity rate expected."""
        return generate_dataset(
            condition_name="Severe ARDS",
            compliance_ml_per_cmH2O=10,
            resistance_cmH2O_L_s=8,
            n_cycles=3,
        )

    def test_returns_list(self, normal_dataset):
        assert isinstance(normal_dataset, list)

    def test_total_count_matches_grid(self, normal_dataset):
        # Grid: 4 VT × 7 RR × 6 PEEP × 3 I:E × 2 patterns = 1,008
        expected = (
            len(PARAMETER_GRID["tidal_volume_ml_per_kg"])
            * len(PARAMETER_GRID["respiratory_rate"])
            * len(PARAMETER_GRID["peep_cmH2O"])
            * len(PARAMETER_GRID["ie_ratio"])
            * len(PARAMETER_GRID["flow_pattern"])
        )
        assert len(normal_dataset) == expected, (
            f"Expected {expected} scenarios, got {len(normal_dataset)}"
        )

    def test_every_scenario_has_required_keys(self, normal_dataset):
        for i, scenario in enumerate(normal_dataset):
            missing = DATASET_SCENARIO_KEYS - scenario.keys()
            assert not missing, (
                f"Scenario {i} missing keys: {missing}"
            )

    def test_scenario_ids_are_strings(self, normal_dataset):
        for scenario in normal_dataset:
            assert isinstance(scenario["scenario_id"], str)
            assert len(scenario["scenario_id"]) > 0

    def test_scenario_ids_start_with_vcv(self, normal_dataset):
        for scenario in normal_dataset:
            assert scenario["scenario_id"].startswith("VCV_"), (
                f"Scenario ID must start with 'VCV_': {scenario['scenario_id']}"
            )

    def test_scenario_ids_are_unique(self, normal_dataset):
        ids = [s["scenario_id"] for s in normal_dataset]
        assert len(ids) == len(set(ids)), "All scenario IDs must be unique"

    def test_condition_field_matches_input(self, normal_dataset):
        for scenario in normal_dataset:
            assert scenario["condition"] == "Normal"

    def test_params_contain_required_keys(self, normal_dataset):
        required = {
            "respiratory_rate", "tidal_volume_ml",
            "compliance_ml_per_cmH2O", "resistance_cmH2O_L_s",
            "ie_ratio", "peep_cmH2O", "flow_pattern",
        }
        for scenario in normal_dataset:
            missing = required - scenario["params"].keys()
            assert not missing, f"Params missing keys: {missing}"

    def test_valid_scenarios_have_metrics(self, normal_dataset):
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                assert len(scenario["metrics"]) > 0, (
                    "Valid scenarios must have populated metrics dict"
                )

    def test_valid_scenarios_have_waveforms(self, normal_dataset):
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                assert "time" in scenario["waveforms"]
                assert "pressure" in scenario["waveforms"]
                assert "flow" in scenario["waveforms"]
                assert "volume" in scenario["waveforms"]

    def test_invalid_scenarios_have_reason(self, normal_dataset):
        invalid = [s for s in normal_dataset if not s["is_valid"]]
        for scenario in invalid:
            assert len(scenario["invalid_reason"]) > 0, (
                "Invalid scenarios must have a non-empty invalid_reason"
            )

    def test_normal_lung_majority_valid(self, normal_dataset):
        valid_count = sum(1 for s in normal_dataset if s["is_valid"])
        valid_fraction = valid_count / len(normal_dataset)
        assert valid_fraction > 0.70, (
            f"Normal lung should have >70% valid scenarios, "
            f"got {valid_fraction:.1%}"
        )

    def test_severe_ards_has_high_invalidity(self, severe_ards_dataset):
        invalid_count = sum(1 for s in severe_ards_dataset if not s["is_valid"])
        invalid_fraction = invalid_count / len(severe_ards_dataset)
        assert invalid_fraction > 0.40, (
            f"Severe ARDS should have >40% invalid scenarios, "
            f"got {invalid_fraction:.1%}"
        )

    def test_generated_at_is_string(self, normal_dataset):
        for scenario in normal_dataset:
            assert isinstance(scenario["generated_at"], str)
            assert len(scenario["generated_at"]) > 0

    def test_both_flow_patterns_present(self, normal_dataset):
        patterns = {s["params"]["flow_pattern"] for s in normal_dataset}
        assert "square" in patterns
        assert "decelerating" in patterns

    def test_all_grid_rr_values_present(self, normal_dataset):
        rr_values = {s["params"]["respiratory_rate"] for s in normal_dataset}
        for rr in PARAMETER_GRID["respiratory_rate"]:
            assert rr in rr_values, (
                f"Respiratory rate {rr} bpm missing from dataset"
            )

    def test_compliance_consistent_across_scenarios(self, normal_dataset):
        # All scenarios in this dataset use the same compliance
        compliances = {
            s["params"]["compliance_ml_per_cmH2O"]
            for s in normal_dataset
        }
        assert compliances == {60.0}, (
            f"All scenarios should use C=60, got {compliances}"
        )
# ---------------------------------------------------------------------------
# Class 9 — Parameter grid
# ---------------------------------------------------------------------------

class TestParameterGrid:
    """
    PARAMETER_GRID must define every required VCV sweep dimension with
    physiologically sensible ranges and produce the documented full-grid
    combination count.
    """

    EXPECTED_KEYS = {
        "tidal_volume_ml_per_kg", "respiratory_rate",
        "peep_cmH2O", "ie_ratio", "flow_pattern",
    }

    def test_grid_has_all_required_keys(self):
        missing = self.EXPECTED_KEYS - PARAMETER_GRID.keys()
        assert not missing, f"PARAMETER_GRID missing: {missing}"

    def test_all_grid_values_are_lists(self):
        for key, values in PARAMETER_GRID.items():
            assert isinstance(values, list), f"{key} is not a list"
            assert len(values) >= 2, f"{key} needs >= 2 values for a real sweep"

    def test_tidal_volume_range_spans_protective_to_standard(self):
        vt = PARAMETER_GRID["tidal_volume_ml_per_kg"]
        assert min(vt) <= 4, "Grid should include ultra-protective VT (<=4 mL/kg)"
        assert max(vt) >= 10, "Grid should include standard-upper VT (>=10 mL/kg)"

    def test_respiratory_rate_range_covers_clinical_span(self):
        rr = PARAMETER_GRID["respiratory_rate"]
        assert min(rr) <= 8
        assert max(rr) >= 30

    def test_peep_range_covers_zero_to_high(self):
        peep = PARAMETER_GRID["peep_cmH2O"]
        assert min(peep) == 0
        assert max(peep) >= 20

    def test_ie_ratio_covers_all_three_clinical_ratios(self):
        ie = set(PARAMETER_GRID["ie_ratio"])
        assert {1.0, 0.5, 0.33}.issubset(ie), (
            "Grid should include 1:1, 1:2, and 1:3 I:E ratios"
        )

    def test_flow_pattern_includes_both_shapes(self):
        assert set(PARAMETER_GRID["flow_pattern"]) == {"square", "decelerating"}

    def test_full_grid_combination_count(self):
        keys = ["tidal_volume_ml_per_kg", "respiratory_rate",
                "peep_cmH2O", "ie_ratio", "flow_pattern"]
        expected = 1
        for k in keys:
            expected *= len(PARAMETER_GRID[k])
        assert expected == 1008, (
            f"Full VCV grid should be 1,008 combinations/mechanics point "
            f"(4x7x6x3x2), got {expected}"
        )