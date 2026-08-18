"""
generator/conditions.py
-----------------------
Respiratory condition presets for the ventilator waveform simulator.

Each preset defines a complete parameter dictionary that can be passed
directly to any generator (waveforms.py or ode_solver.py) via
generate_breath_cycles().

Conditions defined:
    - Normal          : Healthy adult baseline
    - Mild ARDS       : Berlin Definition — mild tier (P/F 200–300)
    - Moderate ARDS   : Berlin Definition — moderate tier (P/F 100–200)
    - Severe ARDS     : Berlin Definition — severe tier (P/F < 100)
    - COPD            : Chronic Obstructive Pulmonary Disease (high resistance)
    - Bronchospasm    : Acute airway narrowing (very high resistance, fast RR)
    - Pneumonia       : Partially consolidated lung (moderate compliance drop)

Usage:
    from generator.conditions import get_condition, list_conditions

    params = get_condition("Moderate ARDS")
    # -> returns dict ready for generate_breath_cycles(params)
"""


# ---------------------------------------------------------------------------
# Condition definitions
# ---------------------------------------------------------------------------
# Parameter reference:
#   respiratory_rate          bpm       — breaths per minute
#   tidal_volume_mL           mL        — target tidal volume per breath
#   compliance_mL_per_cmH2O   mL/cmH2O  — lung compliance (stiffness inverse)
#   resistance_cmH2O_L_s      cmH2O/L/s — airway resistance
#   ie_ratio                  unitless  — t_insp / t_exp  (0.5 = 1:2 ratio)
#   peep_cmH2O                cmH2O     — positive end-expiratory pressure

CONDITIONS = {

    "Normal": {
        "label":                     "Normal",
        "description":               (
            "Healthy adult lung. Normal compliance and resistance. "
            "Standard tidal volume and respiratory rate."
        ),
        "condition":   "Normal",
        "population":  "adult",
        "respiratory_rate":         15,
        "stress_index":             1.00, 
        "tidal_volume_ml":          500,
        "compliance_ml_per_cmH2O":  70,
        "resistance_cmH2O_L_s":     10,
        "ie_ratio":                 0.5,
        "rise_time_s":              0.10,
        "peep_cmH2O":               5,
        "pressure_support_cmH2O":   10,
        "flow_cycle_threshold":     0.25,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":          8,
        "effort_rate_per_min":      15,
        "effort_duration_s":        0.8,
        "pmus_cv":                  0.20,
        "pressure_ceiling_cmH2O":   25
    },

    "Mild ARDS": {
        "label":                     "Mild ARDS",
        "description":               (
            "Mild ARDS (Berlin Definition: P/F 200–300). Moderately stiff lungs. "
            "Lung-protective tidal volume is still achievable without dangerous "
            "driving pressures in most VCV scenarios. Compliance 45 mL/cmH₂O."
        ),
        "condition":   "Mild ARDS",
        "population":  "adult",
        "respiratory_rate":          20,
        "stress_index": 0.90, 
        "tidal_volume_ml":          420,
        "compliance_ml_per_cmH2O":   45,
        "resistance_cmH2O_L_s":      12,
        "ie_ratio":                 0.5,
        "rise_time_s":              0.10,
        "peep_cmH2O":                 8,
        "pressure_support_cmH2O":  12,
        "flow_cycle_threshold":     0.20,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":         11,
        "effort_rate_per_min":     24,
        "effort_duration_s":        0.7,
        "pmus_cv":                  0.25,
        "pressure_ceiling_cmH2O":   28
    },

    "Moderate ARDS": {
        "label":                     "Moderate ARDS",
        "description":               (
            "Moderate ARDS (Berlin Definition: P/F 100–200). Significantly reduced "
            "compliance. Strict lung-protective ventilation required. "
            "Compliance 30 mL/cmH₂O."
        ),
        "condition":   "Moderate ARDS",
        "population":  "adult",
        "respiratory_rate":          24,
        "stress_index": 0.85,
        "tidal_volume_ml":          380,
        "compliance_ml_per_cmH2O":   30,
        "resistance_cmH2O_L_s":      14,
        "rise_time_s":              0.10,
        "ie_ratio":                 0.5,
        "peep_cmH2O":                12,
        "pressure_support_cmH2O":  12,
        "flow_cycle_threshold":     0.15,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":         13,
        "effort_rate_per_min":     30,
        "effort_duration_s":        0.60,
        "pmus_cv":                  0.25,
        "pressure_ceiling_cmH2O":   30
    },

    "Severe ARDS": {
        "label":                     "Severe ARDS",
        "description":               (
            "Severe ARDS (Berlin Definition: P/F < 100). Critically reduced "
            "compliance — the 'baby lung'. Ultra-protective ventilation and "
            "permissive hypercapnia required. Compliance 18 mL/cmH₂O."
        ),
        "condition":   "Severe ARDS",
        "population":  "adult",
        "respiratory_rate":          28,
        "stress_index": 0.80,
        "tidal_volume_ml":          300,
        "compliance_ml_per_cmH2O":   18,
        "resistance_cmH2O_L_s":      16,
        "rise_time_s":              0.10,
        "ie_ratio":                 0.5,
        "peep_cmH2O":                16,
        "pressure_support_cmH2O":  8,
        "flow_cycle_threshold":     0.10,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":          14,
        "effort_rate_per_min":     35,
        "effort_duration_s":        0.50,
        "pmus_cv":                  0.30,
        "pressure_ceiling_cmH2O":   30
    },

    "COPD": {
        "label":                     "COPD",
        "description":               (
            "Chronic Obstructive Pulmonary Disease. High compliance from "
            "emphysematous destruction of elastic tissue. Very high resistance "
            "from dynamic airway collapse. Slow RR and extended I:E required "
            "to prevent dynamic hyperinflation."
        ),
        "condition":   "COPD",
        "population":  "adult",
        "respiratory_rate":          12,
        "stress_index": 1.20,
        "tidal_volume_ml":          550,
        "compliance_ml_per_cmH2O":  100,
        "resistance_cmH2O_L_s":      22,
        "ie_ratio":                 0.25,
        "rise_time_s":              0.10,
        "peep_cmH2O":                 5,
        "pressure_support_cmH2O":  10,
        "flow_cycle_threshold":     0.55,
        "trigger_threshold_cmH2O":  1.0,
        "pmus_peak_cmH2O":         15,
        "effort_rate_per_min":     24,
        "effort_duration_s":        0.75,
        "pmus_cv":                  0.28,
        "pressure_ceiling_cmH2O":   30
    },

    "Bronchospasm": {
        "label":                     "Bronchospasm",
        "description":               (
            "Acute severe bronchospasm (status asthmaticus). Near-normal compliance "
            "but dramatically elevated resistance from bronchoconstriction. "
            "Very low RR and high inspiratory flow required to maximise "
            "expiratory time and prevent air trapping."
        ),
        "condition":   "Bronchospasm",
        "population":  "adult",
        "respiratory_rate":          10,
        "stress_index": 1.00,
        "tidal_volume_ml":          420,
        "compliance_ml_per_cmH2O":   70,
        "resistance_cmH2O_L_s":      35,
        "rise_time_s":              0.10,
        "ie_ratio":                 0.20,
        "peep_cmH2O":                 3,
        "pressure_support_cmH2O":  14,
        "flow_cycle_threshold":     0.65,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":         8,
        "effort_rate_per_min":     12,
        "effort_duration_s":        0.85,
        "pmus_cv":                  0.15,
        "pressure_ceiling_cmH2O":   30
    },

    "Pneumonia": {
        "label":                     "Pneumonia",
        "description":               (
            "Bacterial or viral pneumonia with lobar consolidation. Moderately "
            "reduced compliance from consolidated and oedematous lung units. "
            "Mildly elevated resistance from secretions and airway inflammation."
        ),
        "condition":   "Pneumonia",
        "population":  "adult",
        "respiratory_rate":          22,
        "stress_index": 0.95,
        "tidal_volume_ml":          450,
        "compliance_ml_per_cmH2O":   50,
        "resistance_cmH2O_L_s":      12,
        "ie_ratio":                 0.5,
        "rise_time_s":              0.10,
        "peep_cmH2O":                 8,
        "pressure_support_cmH2O":  12,
        "flow_cycle_threshold":     0.25,
        "trigger_threshold_cmH2O":  1.5,
        "pmus_peak_cmH2O":         11,
        "effort_rate_per_min":     24,
        "effort_duration_s":        0.75,
        "pmus_cv":                  0.22,
        "pressure_ceiling_cmH2O":   26
    },

    "Normal Neonate": {
    "label":       "Normal Neonate",
    "description": (
        "Healthy term neonate (~3 kg) on an uncuffed ETT. Small absolute "
        "compliance and tidal volume, high absolute resistance from the "
        "narrow tube, fast rate, short time constant. No lung pathology — "
        "the neonatal analog of the adult Normal preset."
    ),
    "condition":                "Normal Neonate",
    "population":               "neonate",          # NEW field — see Blocker 1/2
    "weight_kg":                3.0,
    "respiratory_rate":         50,
    "stress_index":             1.00,
    "tidal_volume_ml":          15,                  # ~5 mL/kg
    "compliance_ml_per_cmH2O":  4.0,
    "resistance_cmH2O_L_s":     80,
    "ie_ratio":                 0.50,                # ~1:2, Ti ~0.4s @ RR50
    "rise_time_s":              0.05,                # ASSUMPTION — not sourced
    "peep_cmH2O":               5,
    "pressure_support_cmH2O":   8,
    "flow_cycle_threshold":     0.15,                # neonatal range is 5-20%, vs adult ~25%
    "trigger_threshold_cmH2O":  0.5,                 # ASSUMPTION — weak effort, not sourced
    "pmus_peak_cmH2O":          5,
    "effort_rate_per_min":      50,
    "effort_duration_s":        0.35,
    "pmus_cv":                  0.20,
    "pressure_ceiling_cmH2O":   20,
    "ett_cuff_leak_fraction":   0.15,                # NEW field — see Open Decision 1
    "cuff_leak_fraction":     0.15,     # ADD — PSV/SIMV convention
    "ett_complication":       "cuff_leak",  # ADD — PSV/SIMV convention
    },

    "RDS": {
        "label":       "RDS (Respiratory Distress Syndrome)",
        "description": (
            "Preterm surfactant deficiency. Severely reduced compliance, "
            "resistance at the neonatal baseline (not elevated by disease), "
            "short time constant. Distinct from Severe ARDS: resistance stays "
            "normal here, and compliance can improve rapidly post-surfactant."
        ),
        "condition":                "RDS",
        "population":               "neonate",
        "weight_kg":                1.5,                 # preterm — flag if you want a term RDS variant too
        "respiratory_rate":         50,
        "stress_index":             0.85,                # ASSUMPTION, ARDS-style recruitable tissue
        "tidal_volume_ml":          6,                    # ~4 mL/kg floor, preterm
        "compliance_ml_per_cmH2O":  0.75,                 # 0.5-1 mL/cmH2O, Kumar/PMC7874283
        "resistance_cmH2O_L_s":     80,                   # NOT elevated — IJRC
        "ie_ratio":                 0.33,                 # short Ti ~0.3s
        "rise_time_s":              0.03,                 # ASSUMPTION
        "peep_cmH2O":                6,
        "pressure_support_cmH2O":   10,
        "flow_cycle_threshold":     0.15,
        "trigger_threshold_cmH2O":  0.5,                  # ASSUMPTION
        "pmus_peak_cmH2O":          4,                    # weak preterm effort — ASSUMPTION
        "effort_rate_per_min":      50,
        "effort_duration_s":        0.30,
        "pmus_cv":                  0.25,                 # ASSUMPTION
        "pressure_ceiling_cmH2O":   20,
        "cuff_leak_fraction":       0.15,
        "ett_complication":         "cuff_leak",
        "ett_cuff_leak_fraction":   0.15,  
    },
}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_condition(name: str) -> dict:
    """
    Return the parameter dict for a named respiratory condition.

    Parameters
    ----------
    name : str
        Condition name. Case-insensitive. Use list_conditions() to see options.

    Returns
    -------
    dict
        Parameter dict ready to pass to generate_breath_cycles().
        Does NOT include 'label' or 'description' keys — only waveform params.

    Raises
    ------
    ValueError
        If the condition name is not found.
    """
    key = _resolve_key(name)
    raw = CONDITIONS[key]
    # Strip metadata keys — return only waveform parameters
    return {k: v for k, v in raw.items() if k not in ("label", "description")}


def get_condition_meta(name: str) -> dict:
    """
    Return the full condition entry including label and description.
    Useful for building UI dropdowns and tooltips.
    """
    key = _resolve_key(name)
    return CONDITIONS[key].copy()


def list_conditions() -> list:
    """
    Return a list of available condition names (preserves insertion order).
    """
    return list(CONDITIONS.keys())


def get_all_meta() -> dict:
    """
    Return metadata (label + description) for all conditions.
    Useful for populating UI elements.

    Returns
    -------
    dict  {name: {"label": str, "description": str}}
    """
    return {
        name: {"label": v["label"], "description": v["description"]}
        for name, v in CONDITIONS.items()
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_key(name: str) -> str:
    """Case-insensitive lookup of condition name."""
    # Direct match
    if name in CONDITIONS:
        return name
    # Case-insensitive match
    name_lower = name.strip().lower()
    for key in CONDITIONS:
        if key.lower() == name_lower:
            return key
    raise ValueError(
        f"Unknown condition: '{name}'. "
        f"Available conditions: {list_conditions()}"
    )


# ---------------------------------------------------------------------------
# Quick smoke test — run directly: python generator/conditions.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from vcv_generator import generate_breath_cycles
    from pcv_generator import generate_breath_cycles
    from psv_generator import generate_breath_cycles

    print("Available conditions:", list_conditions())
    print()

    for name in list_conditions():
        meta   = get_condition_meta(name)
        params = get_condition(name)
        result = generate_breath_cycles(params, n_cycles=3)

        peak_p = result["pressure"].max()
        peak_f = result["flow"].max()
        peak_v = result["volume"].max()
        min_f  = result["flow"].min()

        print(f"{'─' * 55}")
        print(f"  {meta['label']}")
        print(f"  {meta['description'][:70]}...")
        print(f"  RR={params['respiratory_rate']} bpm | "
              f"TV={params['tidal_volume_ml']} ml | "
              f"C={params['compliance_ml_per_cmH2O']} | "
              f"R={params['resistance_cmH2O_L_s']}")
        print(f"  Peak pressure : {peak_p:.1f} cmH2O")
        print(f"  Peak flow     : {peak_f:.3f} l/s  |  Min flow: {min_f:.3f} l/s")
        print(f"  Peak volume   : {peak_v:.1f} ml")
        print()

    print("Smoke test passed.")