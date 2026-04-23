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

class TestConditionDifferentiation:
    """
    Verifies that conditions produce outputs that are directionally correct
    relative to each other — not just that they run, but that the ODE engine
    reflects the right pathophysiology.

    Note on COPD: the ODE engine expresses high resistance primarily through
    the expiratory time constant (slow decay), not peak expiratory flow.
    The COPD test therefore measures mid-expiration flow rather than peak flow,
    which is the correct physiological signature for an RC circuit with high R.
    """

    def test_ards_peak_pressure_higher_than_normal(self):
        """
        ARDS (low compliance) must produce higher peak pressure than Normal
        for a similar tidal volume target. Lower C means V/C is larger,
        driving pressure must be higher to deliver the target volume.
        """
        normal = generate_breath_cycles(get_condition("Normal"), n_cycles=5)
        ards   = generate_breath_cycles(get_condition("ARDS"),   n_cycles=5)
        assert ards["pressure"].max() > normal["pressure"].max(), (
            "ARDS peak pressure should exceed Normal due to lower compliance"
        )

    def test_copd_slower_expiratory_decay_than_normal(self):
        """
        COPD (high resistance) must show a slower expiratory flow decay than
        Normal. At 1 second into expiration, COPD should have more remaining
        flow magnitude than Normal — its expiratory tail persists longer.

        This tests the RC time constant effect directly:
            tau_COPD  = 18 × 55 / 1000 = 0.99s
            tau_Normal =  2 × 60 / 1000 = 0.12s
        COPD's time constant is ~8× longer, so at t=1s into expiration
        COPD retains exp(-1/0.99) ≈ 37% of peak expiratory flow,
        while Normal retains exp(-1/0.12) ≈ 0.02% — effectively zero.
        """
        normal = generate_breath_cycles(get_condition("Normal"), n_cycles=5)
        copd   = generate_breath_cycles(get_condition("COPD"),   n_cycles=5)

        def flow_at_1s_into_expiration(result, params):
            """
            Find the index where expiration begins (peak volume) in the
            first breath cycle, then sample flow 1 second later.
            """
            rr       = params["respiratory_rate"]
            ie       = params["ie_ratio"]
            t_cycle  = 60.0 / rr
            t_insp   = t_cycle * ie / (1.0 + ie)
            # Expiration starts at t_insp in the first cycle
            t_target = t_insp + 1.0
            idx      = np.searchsorted(result["time"], t_target)
            idx      = min(idx, len(result["flow"]) - 1)
            return abs(result["flow"][idx])

        flow_normal_at_1s = flow_at_1s_into_expiration(normal, get_condition("Normal"))
        flow_copd_at_1s   = flow_at_1s_into_expiration(copd,   get_condition("COPD"))

        assert flow_copd_at_1s > flow_normal_at_1s, (
            f"COPD expiratory flow at 1s ({flow_copd_at_1s:.4f} L/s) should exceed "
            f"Normal ({flow_normal_at_1s:.4f} L/s) — COPD decays more slowly"
        )

    def test_bronchospasm_peak_pressure_higher_than_normal(self):
        """
        Bronchospasm (very high resistance) must produce higher peak pressure
        than Normal. The resistive term R × Flow is large at high R, driving
        peak airway pressure well above the elastic component alone.
        """
        normal       = generate_breath_cycles(get_condition("Normal"),      n_cycles=5)
        bronchospasm = generate_breath_cycles(get_condition("Bronchospasm"), n_cycles=5)
        assert bronchospasm["pressure"].max() > normal["pressure"].max(), (
            "Bronchospasm peak pressure should exceed Normal due to very high resistance"
        )

    def test_ards_peak_pressure_higher_than_pneumonia(self):
        """
        ARDS compliance (18) is lower than Pneumonia compliance (35),
        so for similar tidal volume targets ARDS must produce higher
        peak pressure than Pneumonia.
        """
        ards     = generate_breath_cycles(get_condition("ARDS"),     n_cycles=5)
        pneumonia = generate_breath_cycles(get_condition("Pneumonia"), n_cycles=5)
        assert ards["pressure"].max() > pneumonia["pressure"].max(), (
            "ARDS peak pressure should exceed Pneumonia — ARDS has lower compliance"
        )

    def test_bronchospasm_slower_expiratory_decay_than_copd(self):
        """
        Bronchospasm resistance (30) exceeds COPD resistance (18), so
        Bronchospasm must show an even slower expiratory decay than COPD.

        We compare normalised flow — flow as a fraction of each condition's
        own peak expiratory flow — to isolate the decay rate from the
        difference in starting tidal volumes (COPD=550mL, Bronchospasm=420mL).

            tau_Bronchospasm = 30 × 50 / 1000 = 1.50s
            tau_COPD         = 18 × 55 / 1000 = 0.99s

        At t=0.5s into expiration:
            Bronchospasm retains exp(-0.5/1.50) ≈ 71% of peak flow
            COPD         retains exp(-0.5/0.99) ≈ 60% of peak flow
        """
        copd         = generate_breath_cycles(get_condition("COPD"),        n_cycles=5)
        bronchospasm = generate_breath_cycles(get_condition("Bronchospasm"), n_cycles=5)

        def normalised_flow_at_0_5s(result, params):
            """
            Return flow magnitude at 0.5s into expiration, divided by the
            peak expiratory flow magnitude for that condition.
            """
            rr       = params["respiratory_rate"]
            ie       = params["ie_ratio"]
            t_cycle  = 60.0 / rr
            t_insp   = t_cycle * ie / (1.0 + ie)
            t_target = t_insp + 0.5
            idx      = np.searchsorted(result["time"], t_target)
            idx      = min(idx, len(result["flow"]) - 1)

            flow_at_0_5s   = abs(result["flow"][idx])
            peak_exp_flow  = abs(result["flow"].min())   # most negative = peak expiratory

            return flow_at_0_5s / peak_exp_flow if peak_exp_flow > 0 else 0.0

        norm_copd         = normalised_flow_at_0_5s(copd,         get_condition("COPD"))
        norm_bronchospasm = normalised_flow_at_0_5s(bronchospasm, get_condition("Bronchospasm"))

        assert norm_bronchospasm > norm_copd, (
            f"Bronchospasm normalised flow at 0.5s ({norm_bronchospasm:.4f}) "
            f"should exceed COPD ({norm_copd:.4f}) — higher resistance means "
            f"slower proportional decay"
        )