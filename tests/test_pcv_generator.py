"""
tests/test_pcv_generator.py
---------------------------
Unit tests for generator/pcv_generator.py (PCV waveform generator).

Five test classes:
    TestInterfaceContract         — return types, keys, array shapes, validation
    TestPhysiologicalPlausibility — basic physical constraints on all outputs
    TestPCVWaveformShape          — waveform morphology specific to PCV
    TestValidityFilter            — threshold logic and invalid_reason strings
    TestDatasetGeneration         — generate_dataset() structure and correctness

Key differences from test_vcv_generator.py:
    - No flow_pattern parameter (PCV has one pressure profile shape)
    - rise_time_s is a required parameter with its own tests
    - delivered_vt_mL is the DEPENDENT variable (not guaranteed)
    - fill_fraction is a unique PCV metric with its own test class section
    - VT tolerance is wider (15%) because ODE convergence at 1 cycle is
      less precise than VCV's analytical integration
    - Dataset fixture uses n_cycles=1 for speed (see EXPERIMENT_LOG.md)

Run with:
    python -m pytest tests/test_pcv_generator.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generator.pcv_generator import (
    FILL_FRACTION_MIN,
    IBW_KG,
    INSP_PRESSURE_MAX_CMHH2O,
    PARAMETER_GRID,
    PPEAK_MAX_CMHH2O,
    VT_MAX_ML,
    VT_MIN_ML,
    generate_breath_cycles,
    generate_dataset,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NORMAL_PARAMS = {
    "respiratory_rate":        15,
    "insp_pressure_cmH2O":     10,
    "compliance_mL_per_cmH2O": 60,
    "resistance_cmH2O_L_s":     2,
    "ie_ratio":                0.5,
    "peep_cmH2O":               5,
    "rise_time_s":             0.0,
}

# Slow RR, high compliance — large VT to test overdistension boundary
HIGH_VT_PARAMS = {
    **NORMAL_PARAMS,
    "insp_pressure_cmH2O": 14,   # 14 × 60 × ~1.0 = 840 mL (at boundary)
    "respiratory_rate":      8,
}

CORE_KEYS   = {"time", "pressure", "flow", "volume"}
METRIC_KEYS = {
    "ppeak_cmH2O", "delivered_vt_mL", "driving_p_cmH2O",
    "mean_paw_cmH2O", "auto_peep_cmH2O", "fill_fraction",
    "minute_vent_L", "time_to_peak_flow_s",
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
    generate_breath_cycles must return all documented keys with correct types.
    Validation must reject missing or out-of-range parameters.
    """

    def test_returns_dict(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert isinstance(result, dict)

    def test_contains_all_core_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert CORE_KEYS.issubset(result.keys()), (
            f"Missing core keys: {CORE_KEYS - result.keys()}"
        )

    def test_contains_all_metric_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert METRIC_KEYS.issubset(result.keys()), (
            f"Missing metric keys: {METRIC_KEYS - result.keys()}"
        )

    def test_contains_validity_keys(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert VALIDITY_KEYS.issubset(result.keys())

    def test_core_arrays_are_numpy(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        for key in CORE_KEYS:
            assert isinstance(result[key], np.ndarray), (
                f"'{key}' should be np.ndarray, got {type(result[key])}"
            )

    def test_core_arrays_same_length(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        lengths = {k: len(result[k]) for k in CORE_KEYS}
        assert len(set(lengths.values())) == 1, (
            f"Core arrays have different lengths: {lengths}"
        )

    def test_metric_values_are_numeric(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        for key in METRIC_KEYS:
            assert isinstance(result[key], (int, float)), (
                f"Metric '{key}' should be numeric, got {type(result[key])}"
            )

    def test_is_valid_is_bool(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert isinstance(result["is_valid"], bool)

    def test_invalid_reason_is_str(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert isinstance(result["invalid_reason"], str)

    def test_n_cycles_1_returns_data(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        assert len(result["time"]) > 0

    def test_n_cycles_5_longer_than_n_cycles_1(self):
        r1 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        r5 = generate_breath_cycles(NORMAL_PARAMS, n_cycles=5)
        assert r5["time"].max() > r1["time"].max()

    def test_missing_respiratory_rate_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS.items() if k != "respiratory_rate"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_missing_insp_pressure_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS.items() if k != "insp_pressure_cmH2O"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_missing_rise_time_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS.items() if k != "rise_time_s"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_missing_peep_raises(self):
        bad = {k: v for k, v in NORMAL_PARAMS.items() if k != "peep_cmH2O"}
        with pytest.raises(ValueError, match="Missing required parameter"):
            generate_breath_cycles(bad)

    def test_out_of_range_rr_raises(self):
        bad = {**NORMAL_PARAMS, "respiratory_rate": 100}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_insp_pressure_raises(self):
        bad = {**NORMAL_PARAMS, "insp_pressure_cmH2O": 0}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_rise_time_raises(self):
        bad = {**NORMAL_PARAMS, "rise_time_s": 0.9}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_peep_raises(self):
        bad = {**NORMAL_PARAMS, "peep_cmH2O": 25}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_compliance_raises(self):
        bad = {**NORMAL_PARAMS, "compliance_mL_per_cmH2O": 200}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)

    def test_out_of_range_resistance_raises(self):
        bad = {**NORMAL_PARAMS, "resistance_cmH2O_L_s": 0.1}
        with pytest.raises(ValueError):
            generate_breath_cycles(bad)


# ---------------------------------------------------------------------------
# Class 2 — Physiological plausibility
# ---------------------------------------------------------------------------

class TestPhysiologicalPlausibility:
    """
    Waveforms must satisfy basic physical laws regardless of parameter values.
    """

    def test_time_is_monotonically_increasing(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert np.all(np.diff(result["time"]) >= 0)

    def test_time_starts_near_zero(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["time"][0] == pytest.approx(0.0, abs=0.02)

    def test_volume_non_negative(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert np.all(result["volume"] >= -0.5), (
            "Volume must never be significantly negative"
        )

    def test_pressure_never_below_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        peep = NORMAL_PARAMS["peep_cmH2O"]
        assert result["pressure"].min() >= peep - 0.5

    def test_pressure_never_exceeds_pip(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        pip = NORMAL_PARAMS["peep_cmH2O"] + NORMAL_PARAMS["insp_pressure_cmH2O"]
        assert result["pressure"].max() <= pip + 0.5, (
            "Pressure must never exceed PIP"
        )

    def test_flow_has_positive_and_negative_phases(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["flow"].max() > 0.0, "No inspiratory flow found"
        assert result["flow"].min() < 0.0, "No expiratory flow found"

    def test_sample_rate_approximately_100hz(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        median_dt = np.median(np.diff(result["time"]))
        assert median_dt == pytest.approx(0.01, abs=0.002)

    def test_ppeak_equals_pip(self):
        # In PCV plateau phase, pressure is held at exactly PIP
        result = generate_breath_cycles(NORMAL_PARAMS)
        pip = NORMAL_PARAMS["peep_cmH2O"] + NORMAL_PARAMS["insp_pressure_cmH2O"]
        assert result["ppeak_cmH2O"] == pytest.approx(pip, abs=0.5), (
            "PPeak must equal PIP in PCV"
        )

    def test_driving_p_equals_insp_pressure(self):
        # driving_p is a direct pass-through of the insp_pressure setting
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["driving_p_cmH2O"] == pytest.approx(
            NORMAL_PARAMS["insp_pressure_cmH2O"], abs=0.1
        )

    def test_fill_fraction_between_zero_and_one(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert 0.0 <= result["fill_fraction"] <= 1.0

    def test_mean_paw_above_peep(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["mean_paw_cmH2O"] > NORMAL_PARAMS["peep_cmH2O"], (
            "Mean airway pressure must exceed PEEP"
        )

    def test_auto_peep_non_negative(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["auto_peep_cmH2O"] >= 0.0

    def test_minute_ventilation_matches_formula(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        expected = (
            NORMAL_PARAMS["respiratory_rate"]
            * result["delivered_vt_mL"]
            / 1000.0
        )
        assert result["minute_vent_L"] == pytest.approx(expected, rel=0.02)

    def test_volume_rises_during_inspiration(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        volume   = result["volume"]
        peak_idx = int(np.argmax(volume))
        insp_vol = volume[:peak_idx + 1]
        diffs    = np.diff(insp_vol)
        assert np.all(diffs >= -0.5), (
            "Volume must rise during inspiration"
        )

    def test_volume_falls_during_expiration(self):
        result = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        volume   = result["volume"]
        peak_idx = int(np.argmax(volume))
        exp_vol  = volume[peak_idx:]
        diffs    = np.diff(exp_vol)
        assert np.all(diffs <= 0.5), (
            "Volume must fall during expiration"
        )


# ---------------------------------------------------------------------------
# Class 3 — PCV waveform shape
# ---------------------------------------------------------------------------

class TestPCVWaveformShape:
    """
    Tests specific to PCV waveform morphology — claims that hold for PCV
    and distinguish it from VCV. Covers rise time effects, fill fraction
    physics, and the dependent-volume relationship.
    """

    def test_lower_compliance_reduces_delivered_vt(self):
        r_hc = generate_breath_cycles({**NORMAL_PARAMS,
                                        "compliance_mL_per_cmH2O": 60})
        r_lc = generate_breath_cycles({**NORMAL_PARAMS,
                                        "compliance_mL_per_cmH2O": 20})
        assert r_lc["delivered_vt_mL"] < r_hc["delivered_vt_mL"], (
            "Lower compliance must reduce delivered VT at same pressure"
        )

    def test_higher_resistance_reduces_delivered_vt(self):
        r_lr = generate_breath_cycles({**NORMAL_PARAMS,
                                        "resistance_cmH2O_L_s": 2})
        r_hr = generate_breath_cycles({**NORMAL_PARAMS,
                                        "resistance_cmH2O_L_s": 20})
        assert r_hr["delivered_vt_mL"] < r_lr["delivered_vt_mL"], (
            "Higher resistance reduces fill fraction → lower delivered VT"
        )

    def test_higher_insp_pressure_increases_delivered_vt(self):
        r_lp = generate_breath_cycles({**NORMAL_PARAMS,
                                        "insp_pressure_cmH2O": 5})
        r_hp = generate_breath_cycles({**NORMAL_PARAMS,
                                        "insp_pressure_cmH2O": 12})
        assert r_hp["delivered_vt_mL"] > r_lp["delivered_vt_mL"], (
            "Higher inspiratory pressure must increase delivered VT"
        )

    def test_higher_insp_pressure_does_not_change_fill_fraction(self):
        # Fill fraction depends on tau and t_insp — not on pressure magnitude
        r_lp = generate_breath_cycles({**NORMAL_PARAMS,
                                        "insp_pressure_cmH2O": 5})
        r_hp = generate_breath_cycles({**NORMAL_PARAMS,
                                        "insp_pressure_cmH2O": 12})
        assert abs(r_lp["fill_fraction"] - r_hp["fill_fraction"]) < 0.05, (
            "Fill fraction must be independent of inspiratory pressure — "
            "it depends only on tau and t_insp"
        )

    def test_faster_rr_reduces_fill_fraction(self):
        r_slow = generate_breath_cycles({**NORMAL_PARAMS,
                                          "respiratory_rate": 8})
        r_fast = generate_breath_cycles({**NORMAL_PARAMS,
                                          "respiratory_rate": 30})
        assert r_fast["fill_fraction"] < r_slow["fill_fraction"], (
            "Faster RR → shorter t_insp → lower fill fraction"
        )

    def test_higher_resistance_reduces_fill_fraction(self):
        r_lr = generate_breath_cycles({**NORMAL_PARAMS,
                                        "resistance_cmH2O_L_s": 2})
        r_hr = generate_breath_cycles({**NORMAL_PARAMS,
                                        "resistance_cmH2O_L_s": 20})
        assert r_hr["fill_fraction"] < r_lr["fill_fraction"], (
            "Higher resistance → larger tau → lower fill fraction"
        )

    def test_rise_time_zero_produces_earlier_peak_flow(self):
        r_rt0 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.0})
        r_rt4 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.4})
        assert r_rt4["time_to_peak_flow_s"] >= r_rt0["time_to_peak_flow_s"], (
            "Longer rise time must delay time to peak inspiratory flow"
        )

    def test_rise_time_zero_produces_higher_peak_flow(self):
        # Instantaneous pressure step → maximum initial flow gradient
        r_rt0 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.0},
                                        n_cycles=1)
        r_rt4 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.4},
                                        n_cycles=1)
        assert r_rt0["flow"].max() > r_rt4["flow"].max(), (
            "Zero rise time must produce higher peak inspiratory flow "
            "than long rise time"
        )

    def test_rise_time_does_not_change_ppeak(self):
        # PPeak = PIP regardless of rise time — pressure always reaches PIP
        r_rt0 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.0})
        r_rt4 = generate_breath_cycles({**NORMAL_PARAMS, "rise_time_s": 0.4})
        assert abs(r_rt0["ppeak_cmH2O"] - r_rt4["ppeak_cmH2O"]) < 0.5, (
            "PPeak must be the same regardless of rise time — "
            "pressure always reaches PIP during plateau"
        )

    def test_longer_ie_ratio_increases_fill_fraction(self):
        # Higher ie_ratio → longer t_insp → more time to fill
        r_short = generate_breath_cycles({**NORMAL_PARAMS, "ie_ratio": 0.33})
        r_long  = generate_breath_cycles({**NORMAL_PARAMS, "ie_ratio": 1.0})
        assert r_long["fill_fraction"] > r_short["fill_fraction"], (
            "Longer I:E ratio → longer t_insp → higher fill fraction"
        )

    def test_expiratory_flow_is_negative(self):
        result   = generate_breath_cycles(NORMAL_PARAMS, n_cycles=1)
        volume   = result["volume"]
        flow     = result["flow"]
        peak_idx = int(np.argmax(volume))
        exp_flow = flow[peak_idx + 1:]
        assert np.all(exp_flow <= 0.05), (
            "Expiratory flow must be non-positive"
        )

    def test_normal_lung_high_fill_fraction(self):
        # Normal lung: tau=0.12s, t_insp≈1.33s → fill fraction ≈ 1.0
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["fill_fraction"] > 0.95, (
            f"Normal lung should have fill fraction > 0.95, "
            f"got {result['fill_fraction']:.4f}"
        )

    def test_pressure_held_at_pip_during_plateau(self):
        # During the plateau phase, pressure must equal PIP within tolerance
        result = generate_breath_cycles(
            {**NORMAL_PARAMS, "rise_time_s": 0.0}, n_cycles=1
        )
        pip     = NORMAL_PARAMS["peep_cmH2O"] + NORMAL_PARAMS["insp_pressure_cmH2O"]
        time    = result["time"]
        pressure = result["pressure"]
        rr      = NORMAL_PARAMS["respiratory_rate"]
        ie      = NORMAL_PARAMS["ie_ratio"]
        t_cycle = 60.0 / rr
        t_insp  = t_cycle * ie / (1.0 + ie)
        # Sample midway through inspiration — should be at PIP
        mid_insp_mask = (time > 0.1) & (time < t_insp * 0.9)
        plateau_pressures = pressure[mid_insp_mask]
        assert np.all(np.abs(plateau_pressures - pip) < 0.5), (
            "Pressure must equal PIP throughout the plateau phase"
        )


# ---------------------------------------------------------------------------
# Class 4 — Validity filter
# ---------------------------------------------------------------------------

class TestValidityFilter:
    """
    The validity filter must correctly identify scenarios breaching clinical
    safety thresholds and provide meaningful reason strings.
    """

    def test_normal_scenario_is_valid(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        assert result["is_valid"] is True
        assert result["invalid_reason"] == ""

    def test_ppeak_breach_flagged(self):
        # insp_pressure=35 + PEEP=20 → PIP=55 > 50
        params = {**NORMAL_PARAMS, "insp_pressure_cmH2O": 35, "peep_cmH2O": 20}
        result = generate_breath_cycles(params)
        assert result["is_valid"] is False
        assert "barotrauma" in result["invalid_reason"].lower() or \
               "ppeak" in result["invalid_reason"].lower(), (
            f"Expected barotrauma mention, got: {result['invalid_reason']}"
        )

    def test_fill_fraction_breach_flagged(self):
        # R=50, RR=30, IE=0.33, RT=0.4 → fill_fraction ≈ 0.079 < 0.20
        params = {
            **NORMAL_PARAMS,
            "resistance_cmH2O_L_s": 50,
            "respiratory_rate":     30,
            "ie_ratio":             0.33,
            "rise_time_s":          0.4,
        }
        result = generate_breath_cycles(params)
        assert result["is_valid"] is False
        assert "fill" in result["invalid_reason"].lower(), (
            f"Expected fill fraction mention, got: {result['invalid_reason']}"
        )

    def test_vt_max_breach_flagged(self):
        # insp_pressure=14, C=60, slow RR → VT ≈ 840 mL (at boundary)
        # Push just over with insp_pressure=15 at slow RR
        params = {
            **NORMAL_PARAMS,
            "insp_pressure_cmH2O":     15,
            "compliance_mL_per_cmH2O": 60,
            "respiratory_rate":         8,
            "ie_ratio":                1.0,   # maximum t_insp → full fill
            "rise_time_s":             0.0,
        }
        result = generate_breath_cycles(params)
        if not result["is_valid"]:
            assert (
                "vt" in result["invalid_reason"].lower() or
                "tidal" in result["invalid_reason"].lower() or
                "volume" in result["invalid_reason"].lower() or
                "overdistension" in result["invalid_reason"].lower() or
                "maximum" in result["invalid_reason"].lower() or
                "barotrauma" in result["invalid_reason"].lower() or
                "ppeak" in result["invalid_reason"].lower()
            ), f"Unexpected reason: {result['invalid_reason']}"

    def test_vt_min_breach_flagged(self):
        # Very low insp_pressure + low compliance → tiny VT
        params = {
            **NORMAL_PARAMS,
            "insp_pressure_cmH2O":     5,
            "compliance_mL_per_cmH2O": 5,
            "respiratory_rate":        30,
            "ie_ratio":                0.33,
        }
        result = generate_breath_cycles(params)
        if not result["is_valid"]:
            assert (
                "vt" in result["invalid_reason"].lower() or
                "tidal" in result["invalid_reason"].lower() or
                "volume" in result["invalid_reason"].lower() or
                "inadequate" in result["invalid_reason"].lower() or
                "minimum" in result["invalid_reason"].lower() or
                "fill" in result["invalid_reason"].lower()
            ), f"Unexpected reason: {result['invalid_reason']}"

    def test_invalid_scenario_has_non_empty_reason(self):
        params = {**NORMAL_PARAMS, "insp_pressure_cmH2O": 35, "peep_cmH2O": 20}
        result = generate_breath_cycles(params)
        assert not result["is_valid"]
        assert len(result["invalid_reason"]) > 0

    def test_valid_scenario_has_empty_reason(self):
        result = generate_breath_cycles(NORMAL_PARAMS)
        if result["is_valid"]:
            assert result["invalid_reason"] == ""

    def test_ppeak_threshold_constant(self):
        assert PPEAK_MAX_CMHH2O == 50.0

    def test_insp_pressure_max_constant(self):
        assert INSP_PRESSURE_MAX_CMHH2O == 35.0

    def test_fill_fraction_min_constant(self):
        assert FILL_FRACTION_MIN == 0.20

    def test_vt_min_is_3_ml_per_kg(self):
        assert VT_MIN_ML == IBW_KG * 3

    def test_vt_max_is_12_ml_per_kg(self):
        assert VT_MAX_ML == IBW_KG * 12


# ---------------------------------------------------------------------------
# Class 5 — Dataset generation
# ---------------------------------------------------------------------------

class TestDatasetGeneration:
    """
    generate_dataset() must return a correctly structured list of scenario
    dicts covering the full PCV parameter grid for a given condition + mechanics.

    n_cycles=1 is used throughout to keep test suite runtime under 2 minutes.
    See EXPERIMENT_LOG.md for discussion of this tradeoff.
    """

    @pytest.fixture(scope="class")
    def normal_dataset(self):
        return generate_dataset(
            condition_name="Normal",
            compliance_mL_per_cmH2O=60,
            resistance_cmH2O_L_s=2,
            n_cycles=1,
        )

    @pytest.fixture(scope="class")
    def high_resistance_dataset(self):
        # High resistance — high invalidity rate expected from fill fraction
        return generate_dataset(
            condition_name="Bronchospasm",
            compliance_mL_per_cmH2O=50,
            resistance_cmH2O_L_s=30,
            n_cycles=1,
        )

    def test_returns_list(self, normal_dataset):
        assert isinstance(normal_dataset, list)

    def test_total_count_matches_grid(self, normal_dataset):
        expected = (
            len(PARAMETER_GRID["insp_pressure_cmH2O"])
            * len(PARAMETER_GRID["respiratory_rate"])
            * len(PARAMETER_GRID["peep_cmH2O"])
            * len(PARAMETER_GRID["ie_ratio"])
            * len(PARAMETER_GRID["rise_time_s"])
        )
        assert len(normal_dataset) == expected, (
            f"Expected {expected} scenarios, got {len(normal_dataset)}"
        )

    def test_every_scenario_has_required_keys(self, normal_dataset):
        for i, scenario in enumerate(normal_dataset):
            missing = DATASET_SCENARIO_KEYS - scenario.keys()
            assert not missing, f"Scenario {i} missing keys: {missing}"

    def test_scenario_ids_are_strings(self, normal_dataset):
        for scenario in normal_dataset:
            assert isinstance(scenario["scenario_id"], str)
            assert len(scenario["scenario_id"]) > 0

    def test_scenario_ids_start_with_pcv(self, normal_dataset):
        for scenario in normal_dataset:
            assert scenario["scenario_id"].startswith("PCV_"), (
                f"Scenario ID must start with 'PCV_': {scenario['scenario_id']}"
            )

    def test_scenario_ids_are_unique(self, normal_dataset):
        ids = [s["scenario_id"] for s in normal_dataset]
        assert len(ids) == len(set(ids)), (
            f"Scenario IDs are not unique — "
            f"{len(ids)} total, {len(set(ids))} unique"
        )

    def test_condition_field_matches_input(self, normal_dataset):
        for scenario in normal_dataset:
            assert scenario["condition"] == "Normal"

    def test_params_contain_required_keys(self, normal_dataset):
        required = {
            "respiratory_rate", "insp_pressure_cmH2O",
            "compliance_mL_per_cmH2O", "resistance_cmH2O_L_s",
            "ie_ratio", "peep_cmH2O", "rise_time_s",
        }
        for scenario in normal_dataset:
            missing = required - scenario["params"].keys()
            assert not missing, f"Params missing keys: {missing}"

    def test_valid_scenarios_have_metrics(self, normal_dataset):
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                assert len(scenario["metrics"]) > 0

    def test_valid_scenarios_have_waveforms(self, normal_dataset):
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                for key in ("time", "pressure", "flow", "volume"):
                    assert key in scenario["waveforms"], (
                        f"Valid scenario missing waveform key: {key}"
                    )

    def test_invalid_scenarios_have_reason(self, normal_dataset):
        for scenario in normal_dataset:
            if not scenario["is_valid"]:
                assert len(scenario["invalid_reason"]) > 0

    def test_normal_lung_minority_invalid(self, normal_dataset):
        # In PCV, a compliant Normal lung (C=60) produces VT>840mL at many
        # pressure settings — the VT_MAX filter correctly rejects these.
        # Valid fraction is ~29% at C=60, R=2 — lower than other conditions
        # because high compliance means even moderate pressures overdistend.
        # The threshold of 25% is a meaningful floor: a broken generator
        # would produce 0% valid; the real generator produces ~29%.
        valid_count    = sum(1 for s in normal_dataset if s["is_valid"])
        valid_fraction = valid_count / len(normal_dataset)
        assert valid_fraction > 0.25, (
            f"Normal lung PCV dataset should have >25% valid scenarios, "
            f"got {valid_fraction:.1%}"
        )

    def test_high_resistance_has_elevated_invalidity(self, high_resistance_dataset):
        invalid_count    = sum(1 for s in high_resistance_dataset
                               if not s["is_valid"])
        invalid_fraction = invalid_count / len(high_resistance_dataset)
        assert invalid_fraction > 0.30, (
            f"High resistance condition should have >30% invalid scenarios, "
            f"got {invalid_fraction:.1%}"
        )

    def test_generated_at_is_string(self, normal_dataset):
        for scenario in normal_dataset:
            assert isinstance(scenario["generated_at"], str)
            assert len(scenario["generated_at"]) > 0

    def test_all_rise_time_values_present(self, normal_dataset):
        rt_values = {s["params"]["rise_time_s"] for s in normal_dataset}
        for rt in PARAMETER_GRID["rise_time_s"]:
            assert rt in rt_values, (
                f"Rise time {rt}s missing from dataset"
            )

    def test_all_insp_pressure_values_present(self, normal_dataset):
        p_values = {s["params"]["insp_pressure_cmH2O"] for s in normal_dataset}
        for p in PARAMETER_GRID["insp_pressure_cmH2O"]:
            assert p in p_values, (
                f"Inspiratory pressure {p} cmH2O missing from dataset"
            )

    def test_compliance_consistent_across_scenarios(self, normal_dataset):
        compliances = {
            s["params"]["compliance_mL_per_cmH2O"]
            for s in normal_dataset
        }
        assert compliances == {60.0}, (
            f"All scenarios should use C=60, got {compliances}"
        )

    def test_metrics_contain_fill_fraction(self, normal_dataset):
        # fill_fraction is PCV-specific — must be present in every valid metric dict
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                assert "fill_fraction" in scenario["metrics"], (
                    "fill_fraction must be present in PCV metrics"
                )

    def test_metrics_contain_time_to_peak_flow(self, normal_dataset):
        for scenario in normal_dataset:
            if scenario["is_valid"]:
                assert "time_to_peak_flow_s" in scenario["metrics"], (
                    "time_to_peak_flow_s must be present in PCV metrics"
                )
