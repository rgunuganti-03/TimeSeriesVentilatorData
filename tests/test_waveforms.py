"""
tests/test_waveforms.py
-----------------------
Unit tests for generator/waveforms.py (Phase 1 rule-based waveform generator).

Run with:
    python -m pytest tests/test_waveforms.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.waveforms import generate_breath_cycles
from generator.conditions import get_condition, list_conditions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NORMAL_PARAMS = {
    "respiratory_rate":          15,
    "tidal_volume_mL":          500,
    "compliance_mL_per_cmH2O":   60,
    "resistance_cmH2O_L_s":       2,
    "ie_ratio":                 0.5,
    "peep_cmH2O":                 5,
}


# ---------------------------------------------------------------------------
# Interface contract tests
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """Rule-based generator must satisfy the shared waveform interface."""

    def test_returns_dict(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert set(result.keys()) == {"time", "pressure", "flow", "volume"}

    def test_all_values_are_numpy_arrays(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        for key, arr in result.items():
            assert isinstance(arr, np.ndarray), f"'{key}' is not a numpy array"

    def test_all_arrays_same_length(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        lengths = [len(v) for v in result.values()]
        assert len(set(lengths)) == 1, f"Array lengths differ: {lengths}"

    def test_n_cycles_1(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        assert len(result["time"]) > 0

    def test_n_cycles_10(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=10)
        # 10 cycles at 15 bpm = 4s/cycle → ~40s total
        assert result["time"].max() > 35.0

    def test_missing_param_raises_value_error(self):
        bad_params = {k: v for k, v in NORMAL_PARAMS.items() if k != "peep_cmH2O"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad_params)

    def test_out_of_range_rr_raises(self):
        bad = {**NORMAL_PARAMS, "respiratory_rate": 100}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_tidal_volume_raises(self):
        bad = {**NORMAL_PARAMS, "tidal_volume_mL": 50}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_peep_raises(self):
        bad = {**NORMAL_PARAMS, "peep_cmH2O": 25}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)


# ---------------------------------------------------------------------------
# Physiological plausibility tests
# ---------------------------------------------------------------------------

class TestPhysiologicalPlausibility:
    """Generated waveforms must satisfy basic physiological constraints."""

    def test_time_is_monotonically_increasing(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        diffs = np.diff(result["time"])
        assert np.all(diffs >= 0), "Time array is not monotonically increasing"

    def test_time_starts_near_zero(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["time"][0] == pytest.approx(0.0, abs=0.02)

    def test_volume_is_non_negative(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert np.all(result["volume"] >= -0.5), "Volume should not be significantly negative"

    def test_pressure_above_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        peep = NORMAL_PARAMS["peep_cmH2O"]
        assert result["pressure"].min() >= peep - 0.1

    def test_peak_pressure_is_physiologically_reasonable(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        peak = result["pressure"].max()
        assert 10 <= peak <= 50, f"Peak pressure {peak:.1f} cmH2O outside plausible range"

    def test_flow_has_positive_and_negative_phases(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["flow"].max() > 0.0, "No positive (inspiratory) flow found"
        assert result["flow"].min() < 0.0, "No negative (expiratory) flow found"

    def test_tidal_volume_delivered_within_tolerance(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=3)
        tv_target = NORMAL_PARAMS["tidal_volume_mL"]
        peak_v = result["volume"].max()
        assert abs(peak_v - tv_target) / tv_target < 0.10, (
            f"Delivered TV {peak_v:.0f} mL deviates >10% from target {tv_target} mL"
        )

    def test_sample_rate_approximately_100hz(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        dt = np.diff(result["time"])
        median_dt = np.median(dt)
        assert median_dt == pytest.approx(0.01, abs=0.002), (
            f"Expected ~100 Hz (dt=0.01s), got median dt={median_dt:.4f}s"
        )

    def test_total_duration_matches_respiratory_rate(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=5)
        expected_duration = 5 * (60.0 / NORMAL_PARAMS["respiratory_rate"])
        actual_duration = result["time"].max()
        assert actual_duration == pytest.approx(expected_duration, rel=0.02)


# ---------------------------------------------------------------------------
# Rule-based waveform shape tests
# ---------------------------------------------------------------------------

class TestWaveformShape:
    """
    Tests specific to the rule-based generator's waveform morphology:
    decelerating inspiratory flow and passive exponential expiration.
    """

    def test_inspiratory_flow_is_decelerating(self):
        """
        Peak inspiratory flow should occur at the start of inspiration,
        not the middle or end (decelerating profile).
        """
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        flow = result["flow"]
        # Find the inspiratory segment (positive flow)
        insp_indices = np.where(flow > 0)[0]
        assert len(insp_indices) > 0, "No inspiratory flow found"
        insp_flow = flow[insp_indices]
        # First sample should be larger than the last sample
        assert insp_flow[0] > insp_flow[-1], (
            "Inspiratory flow is not decelerating (peak not at start)"
        )

    def test_expiratory_flow_is_negative(self):
        """All expiratory-phase flow values should be negative."""
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        flow = result["flow"]
        volume = result["volume"]
        # After peak volume, we are in expiration
        peak_idx = np.argmax(volume)
        exp_flow = flow[peak_idx + 1:]
        assert np.all(exp_flow <= 0.0), "Expiratory flow contains positive values"

    def test_volume_rises_during_inspiration(self):
        """Volume must increase monotonically during the inspiratory phase."""
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        volume = result["volume"]
        peak_idx = int(np.argmax(volume))
        insp_volume = volume[:peak_idx + 1]
        diffs = np.diff(insp_volume)
        assert np.all(diffs >= -0.1), "Volume is not monotonically rising during inspiration"

    def test_volume_falls_during_expiration(self):
        """Volume must decrease after the inspiratory peak."""
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        volume = result["volume"]
        peak_idx = int(np.argmax(volume))
        exp_volume = volume[peak_idx:]
        diffs = np.diff(exp_volume)
        assert np.all(diffs <= 0.1), "Volume is not falling during expiration"

    def test_higher_resistance_lowers_peak_flow(self):
        """
        Increasing airway resistance should reduce peak inspiratory flow
        for the same tidal volume target.
        """
        low_r = {**NORMAL_PARAMS, "resistance_cmH2O_L_s": 2}
        high_r = {**NORMAL_PARAMS, "resistance_cmH2O_L_s": 20}
        result_low  = generate_breath_cycles(low_r,  n_cycles=3)
        result_high = generate_breath_cycles(high_r, n_cycles=3)
        assert result_high["flow"].max() <= result_low["flow"].max(), (
            "Higher resistance should reduce peak inspiratory flow"
        )

    def test_lower_compliance_raises_peak_pressure(self):
        """
        Reduced compliance (stiffer lung) must produce higher peak airway
        pressure for the same tidal volume.
        """
        normal_c = {**NORMAL_PARAMS, "compliance_mL_per_cmH2O": 60}
        low_c    = {**NORMAL_PARAMS, "compliance_mL_per_cmH2O": 15}
        result_normal = generate_breath_cycles(normal_c, n_cycles=3)
        result_low_c  = generate_breath_cycles(low_c,    n_cycles=3)
        assert result_low_c["pressure"].max() > result_normal["pressure"].max(), (
            "Lower compliance should produce higher peak pressure"
        )


# ---------------------------------------------------------------------------
# Condition preset tests — all five conditions must run without error
# ---------------------------------------------------------------------------

class TestConditionPresets:

    @pytest.mark.parametrize("condition", list_conditions())
    def test_condition_runs(self, condition):
        params = get_condition(condition)
        result = generate_breath_cycles(params, n_cycles=5)
        assert len(result["time"]) > 0

    @pytest.mark.parametrize("condition", list_conditions())
    def test_condition_returns_four_arrays(self, condition):
        params = get_condition(condition)
        result = generate_breath_cycles(params, n_cycles=5)
        assert set(result.keys()) == {"time", "pressure", "flow", "volume"}

    @pytest.mark.parametrize("condition", list_conditions())
    def test_condition_pressures_positive(self, condition):
        params = get_condition(condition)
        result = generate_breath_cycles(params, n_cycles=5)
        assert result["pressure"].min() >= 0.0, (
            f"Negative pressure in condition '{condition}'"
        )

    @pytest.mark.parametrize("condition", list_conditions())
    def test_condition_volumes_non_negative(self, condition):
        params = get_condition(condition)
        result = generate_breath_cycles(params, n_cycles=5)
        assert result["volume"].min() >= -1.0, (
            f"Significantly negative volume in condition '{condition}'"
        )


# ---------------------------------------------------------------------------
# Condition differentiation tests — conditions must be physiologically distinct
# ---------------------------------------------------------------------------

class TestConditionDifferentiation:

    def test_ards_peak_pressure_higher_than_normal(self):
        """ARDS (low compliance) must produce higher peak pressure than Normal."""
        normal = generate_breath_cycles(get_condition("Normal"), n_cycles=5)
        ards   = generate_breath_cycles(get_condition("ARDS"),   n_cycles=5)
        assert ards["pressure"].max() > normal["pressure"].max()

    def test_copd_peak_flow_lower_than_normal(self):
        """COPD (high resistance) must have lower peak expiratory flow than Normal."""
        normal = generate_breath_cycles(get_condition("Normal"), n_cycles=5)
        copd   = generate_breath_cycles(get_condition("COPD"),   n_cycles=5)
        assert abs(copd["flow"].min()) < abs(normal["flow"].min()), (
            "COPD should have lower peak expiratory flow magnitude than Normal"
        )

    def test_bronchospasm_peak_pressure_higher_than_normal(self):
        """Bronchospasm (very high resistance) must produce higher peak pressure."""
        normal       = generate_breath_cycles(get_condition("Normal"),      n_cycles=5)
        bronchospasm = generate_breath_cycles(get_condition("Bronchospasm"), n_cycles=5)
        assert bronchospasm["pressure"].max() > normal["pressure"].max()
