"""
generator/conditions.py
-----------------------
Respiratory condition presets for the ventilator waveform simulator.

Each preset defines a complete parameter dictionary that can be passed
directly to generator/waveforms.py::generate_breath_cycles().

Conditions defined:
    - Normal        : Healthy adult baseline
    - ARDS          : Acute Respiratory Distress Syndrome (low compliance)
    - COPD          : Chronic Obstructive Pulmonary Disease (high resistance)
    - Bronchospasm  : Acute airway narrowing (very high resistance, fast RR)
    - Pneumonia     : Partially consolidated lung (moderate compliance drop)

Usage:
    from generator.conditions import get_condition, list_conditions

    params = get_condition("ARDS")
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
        "label":                    "Normal",
        "description": (
            "Healthy adult lung. Normal compliance and resistance. "
            "Standard tidal volume and respiratory rate."
        ),
        "respiratory_rate":          15,
        "tidal_volume_mL":          500,
        "compliance_mL_per_cmH2O":   60,   # healthy: 60–100 mL/cmH2O
        "resistance_cmH2O_L_s":       2,   # healthy: 1–3 cmH2O/L/s
        "ie_ratio":                 0.5,   # 1:2 — standard
        "peep_cmH2O":                 5,
    },

    "ARDS": {
        "label":                    "ARDS",
        "description": (
            "Acute Respiratory Distress Syndrome. Severely reduced compliance "
            "due to fluid-filled, collapsed alveoli. Higher RR and PEEP are "
            "used clinically. Expect elevated peak pressures."
        ),
        "respiratory_rate":          20,   # faster RR to compensate
        "tidal_volume_mL":          380,   # lung-protective: 6 mL/kg IBW
        "compliance_mL_per_cmH2O":   18,   # severely reduced (normal: 60)
        "resistance_cmH2O_L_s":       4,   # mildly elevated
        "ie_ratio":                 0.5,
        "peep_cmH2O":                10,   # higher PEEP to recruit alveoli
    },

    "COPD": {
        "label":                    "COPD",
        "description": (
            "Chronic Obstructive Pulmonary Disease. Markedly elevated airway "
            "resistance due to narrowed airways. Prolonged expiration needed — "
            "reduced IE ratio. Risk of auto-PEEP and air trapping."
        ),
        "respiratory_rate":          14,   # slower RR to allow full expiration
        "tidal_volume_mL":          550,
        "compliance_mL_per_cmH2O":   55,   # near normal (emphysema can increase C)
        "resistance_cmH2O_L_s":      18,   # markedly elevated (normal: 2)
        "ie_ratio":                 0.35,  # more time for slow expiration (1:2.9)
        "peep_cmH2O":                 5,
    },

    "Bronchospasm": {
        "label":                    "Bronchospasm",
        "description": (
            "Acute bronchospasm (e.g. asthma attack). Very high resistance "
            "from bronchoconstriction. Rapid respiratory rate. Expect high "
            "peak pressures and slow expiratory flow."
        ),
        "respiratory_rate":          22,
        "tidal_volume_mL":          420,
        "compliance_mL_per_cmH2O":   50,
        "resistance_cmH2O_L_s":      30,   # very high
        "ie_ratio":                 0.35,
        "peep_cmH2O":                 5,
    },

    "Pneumonia": {
        "label":                    "Pneumonia",
        "description": (
            "Consolidating pneumonia. Moderately reduced compliance due to "
            "fluid and inflammatory exudate filling alveolar spaces. "
            "Mild-to-moderate compliance drop with slightly elevated resistance."
        ),
        "respiratory_rate":          18,
        "tidal_volume_mL":          450,
        "compliance_mL_per_cmH2O":   35,   # moderately reduced
        "resistance_cmH2O_L_s":       6,   # mildly elevated
        "ie_ratio":                 0.5,
        "peep_cmH2O":                 6,
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