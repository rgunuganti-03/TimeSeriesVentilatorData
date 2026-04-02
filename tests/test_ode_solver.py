"""
tests/test_ode_solver.py
------------------------
Unit tests for generator/ode_solver.py (Phase 2 ODE lung mechanics model).

Run with:
    python -m pytest tests/test_ode_solver.py -v
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.ode_solver import generate_breath_cycles
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
# Interface contract tests — must match generator/waveforms.py exactly
# ---------------------------------------------------------------------------

class TestInterfaceContract:
    """ODE solver must be a drop-in replacement for the rule-based generator."""

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
        # Duration should be approximately 10 cycles × 4s/cycle = 40s
        assert result["time"].max() > 35.0

    def test_missing_param_raises_value_error(self):
        bad_params = {k: v for k, v in NORMAL_PARAMS.items() if k != "peep_cmH2O"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad_params)

    def test_out_of_range_rr_raises(self):
        bad = {**NORMAL_PARAMS, "respiratory_rate": 100}
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
        # Pressure should never fall below PEEP
        assert result["pressure"].min() >= peep - 0.1

    def test_peak_pressure_is_physiologically_reasonable(self):
        # For normal params, peak pressure should be in clinical range 10–40 cmH2O
        result = generate_breath_cycles(NORMAL_PARAMS)
        peak = result["pressure"].max()
        assert 10 <= peak <= 50, f"Peak pressure {peak:.1f} cmH2O outside plausible range"

    def test_flow_has_positive_and_negative_phases(self):
        # Inspiration = positive flow, expiration = negative flow
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["flow"].max() > 0.0, "No positive (inspiratory) flow found"
        assert result["flow"].min() < 0.0, "No negative (expiratory) flow found"

    def test_tidal_volume_delivered_within_tolerance(self):
        # Peak volume per cycle should be close to target TV (within 10%)
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
# Auto-PEEP detection — COPD with high resistance should show air-trapping
# ---------------------------------------------------------------------------

class TestAutoPEEP:

    def test_copd_shows_higher_residual_volume_than_normal(self):
        """
        COPD (high resistance, short I:E) should result in more residual
        volume at end of expiration than a normal lung.
        """
        normal = generate_breath_cycles(get_condition("Normal"), n_cycles=10)
        copd   = generate_breath_cycles(get_condition("COPD"),   n_cycles=10)

        # Compare volume in the last sample of each (end of final expiration)
        v_end_normal = normal["volume"][-1]
        v_end_copd   = copd["volume"][-1]

        assert v_end_copd >= v_end_normal, (
            f"Expected COPD residual volume ({v_end_copd:.1f} mL) >= "
            f"Normal ({v_end_normal:.1f} mL)"
        )
