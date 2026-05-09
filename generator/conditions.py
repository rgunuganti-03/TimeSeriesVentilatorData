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
    "respiratory_rate":          15,
    "tidal_volume_mL":          500,
    "compliance_mL_per_cmH2O":   70,   # healthy: 60–100 mL/cmH2O
    "resistance_cmH2O_L_s":      10,   # total system: ETT (~5–7) + normal airways (~3–5)
    "ie_ratio":                 0.5,
    "peep_cmH2O":                 5,
},

"Mild ARDS": {
    "respiratory_rate":          20,   # elevated drive; Berlin RR typically 18–25
    "tidal_volume_mL":          420,   # ~6 mL/kg IBW for 70 kg patient
    "compliance_mL_per_cmH2O":   45,   # moderately reduced (40–55 range)
    "resistance_cmH2O_L_s":      12,   # ETT + mild airway edema/inflammation
    "ie_ratio":                 0.5,
    "peep_cmH2O":                 8,
},

"Moderate ARDS": {
    "respiratory_rate":          24,
    "tidal_volume_mL":          380,   # strict lung-protective
    "compliance_mL_per_cmH2O":   30,   # significantly reduced (28–40 range)
    "resistance_cmH2O_L_s":      14,   # ETT + peribronchial edema
    "ie_ratio":                 0.5,
    "peep_cmH2O":                12,
},

"Severe ARDS": {
    "respiratory_rate":          28,
    "tidal_volume_mL":          300,   # ultra-protective: ~4 mL/kg IBW
    "compliance_mL_per_cmH2O":   18,   # severely reduced (15–28 range)
    "resistance_cmH2O_L_s":      16,   # ETT + significant airway edema
    "ie_ratio":                 0.5,
    "peep_cmH2O":                16,
},

"COPD": {
    "respiratory_rate":          12,   # slow RR — permissive hypercapnia strategy
    "tidal_volume_mL":          550,
    "compliance_mL_per_cmH2O":  100,   # HIGH — emphysema destroys elastic recoil
    "resistance_cmH2O_L_s":      22,   # ETT + severely obstructed airways
    "ie_ratio":                 0.33,  # 1:3 — extended expiratory time
    "peep_cmH2O":                 5,
},

"Bronchospasm": {
    "respiratory_rate":          10,   # deliberately slow — maximize Te
    "tidal_volume_mL":          420,   # protective; avoid overdistension
    "compliance_mL_per_cmH2O":   70,   # near-normal (problem is resistance, not compliance)
    "resistance_cmH2O_L_s":      35,   # very high — severe bronchoconstriction
    "ie_ratio":                 0.30,  # 1:3.3 — maximize expiratory time (Tuxen strategy)
    "peep_cmH2O":                 3,   # low/zero extrinsic PEEP in acute bronchospasm
},

"Pneumonia": {
    "respiratory_rate":          22,
    "tidal_volume_mL":          450,
    "compliance_mL_per_cmH2O":   50,   # moderately reduced from consolidation
    "resistance_cmH2O_L_s":      12,   # ETT + secretions + inflamed airways
    "ie_ratio":                 0.5,
    "peep_cmH2O":                 8,
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
    from waveforms import generate_breath_cycles

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
              f"TV={params['tidal_volume_mL']} mL | "
              f"C={params['compliance_mL_per_cmH2O']} | "
              f"R={params['resistance_cmH2O_L_s']}")
        print(f"  Peak pressure : {peak_p:.1f} cmH2O")
        print(f"  Peak flow     : {peak_f:.3f} L/s  |  Min flow: {min_f:.3f} L/s")
        print(f"  Peak volume   : {peak_v:.1f} mL")
        print()

    print("Smoke test passed.")