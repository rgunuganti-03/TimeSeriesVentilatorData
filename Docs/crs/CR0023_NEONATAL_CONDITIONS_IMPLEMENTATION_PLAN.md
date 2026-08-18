# CR0023 (draft) — Incorporating Normal Neonate, RDS, and MAS Across All Five Modes

**Author:** Riya Gunuganti (draft prepared with Claude)
**Status:** Proposed — Blockers 1 and 2 decided. Also decided: explicit `"population": "adult"` backfill on the seven existing conditions; omit the driving-pressure and neonatal-VT-ceiling checks rather than invent unsourced thresholds; no separate `NEONATE_ETT_K1`/`K2`/`PS_MAX` constants (reuse adult values until a tube-specific source exists); leak stays the existing fixed-fraction mechanism, default-on; RDS ships as 1 compartment; scenario IDs get population-gated decimal precision; dataset generation uses parallel per-population scripts. One open item remains: whether MAS ships this round with provisional compartment numbers or is deferred until those numbers have a source (see Section 2).
**Project:** Time Series Ventilator Data

---

## Read this part first — two blockers, not three conditions

Before touching `conditions.py`, there are two things in the current architecture that were built assuming an adult patient, and both will silently break or hard-reject neonatal scenarios if left alone. Neither is a "new condition" problem — they're population problems, and they show up once, everywhere, rather than once per condition. Fix these first; the condition-specific work in the sections below assumes they're resolved.

### Blocker 1 — every safety/validity constant is a hardcoded adult number, duplicated five times

`IBW_KG = 70.0` is a module-level constant in `vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`, `prvc_generator.py`, and `simv_generator.py`. Everything downstream is derived from it:

```python
IBW_KG: float                  = 70.0
VT_MIN_ML: float                = IBW_KG * 3       # 210 mL
VT_MAX_ML: float                = IBW_KG * 12      # 840 mL
PPEAK_MAX_CMHH2O: float         = 50.0
DRIVING_P_MAX_CMHH2O: float     = 20.0
INSP_PRESSURE_MAX_CMHH2O: float = 35.0
PS_MAX_CMHH2O: float            = 20.0
CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 2.5
DEFAULT_CHEST_WALL_COMPLIANCE: float   = 250.0     # ~inert for an adult
ETT_K1: float = 5.0   # sized for a 7.5 mm ID tube
ETT_K2: float = 3.0
```

A term neonate's whole tidal volume is ~15 mL. `VT_MIN_ML = 210` means **every physiologically correct neonatal breath will be flagged `is_valid = False`** the moment it hits the validity filter — not because the physics is wrong, but because the filter is checking it against a 70 kg adult's floor. Same story for `DEFAULT_CHEST_WALL_COMPLIANCE = 250` (fine as "effectively infinite" for an adult; wrong for a neonate where chest wall compliance is the *dominant*, not negligible, term) and `ETT_K1`/`ETT_K2` (calibrated for a 7.5 mm tube; a neonatal 2.5–3.5 mm tube has much higher Rohrer coefficients).

**DECIDED: option 2 — a second constants block gated by a `population` field.** This section now spells out exactly what that means in each file, because "add NEONATE_ constants" undersells it — there are four distinct consequences that fall out of this choice, and one of them (weight) changes the shape of the constants, not just their values.

#### 1a. `population` becomes a real, load-bearing field — read it once, near the top

In every `generate_breath_cycles()`, alongside the existing `condition = params.get("condition", "Normal")` line, add:
```python
population = params.get("population", "adult")
```
Note this is deliberately keyed off `population`, **not** off `condition` name-matching (e.g. `if condition == "RDS"`). That decoupling is what makes the design sound — a test can construct `{"condition": "Normal", "population": "neonate", ...}` and correctly get neonatal thresholds without the generator needing to know the names of all three neonatal conditions. It also means `conditions.py`'s three new entries **must** carry `"population": "neonate"` (already included in the entries drafted below) or they'll silently get adult thresholds despite everything else being correct — worth a dedicated test (see Section 3 below).

**DECIDED: yes, explicit `"population": "adult"` on all seven existing entries.** Every entry already has a `"condition": "<Name>",` line — that's a reliable anchor that appears once per entry, so the mechanical edit is the same in all seven places: add one new line directly after it.

```python
"Normal": {
    "label":                     "Normal",
    "description":               ( ... ),
    "condition":   "Normal",
    "population":  "adult",                      # ADD THIS LINE
    "respiratory_rate":         15,
    ...
```

```python
"COPD": {
    "label":                     "COPD",
    "description":               ( ... ),
    "condition":   "COPD",
    "population":  "adult",                      # ADD THIS LINE
    "respiratory_rate":          12,
    ...
```

Apply the same single-line insertion (`"population": "adult",` immediately after `"condition": "<Name>",`) to Mild ARDS, Moderate ARDS, Severe ARDS, Bronchospasm, and Pneumonia — every entry has that same anchor line, so it's the same edit seven times, not seven different edits. This is a good candidate for `str_replace` calls in your editor scoped to each `"condition": "X",` line specifically, so you don't need to retype each full dict.

#### 1b. The weight-dependent constants can't just be a second fixed number — they need to read `weight_kg`

This is the one place where mirroring the adult pattern exactly would be wrong. `IBW_KG = 70.0` works as a single fixed constant because every adult condition in this simulator implicitly assumes the same ~70 kg reference body. The three neonatal conditions do **not** share one weight — the RDS preset (preterm, 1.5 kg) and the MAS preset (term, 3.2 kg) are genuinely different patients, and `conditions.py` already carries a per-condition `weight_kg` field for exactly this reason. So `VT_MIN_ML`/`VT_MAX_ML` shouldn't become a second *fixed* module constant (`NEONATE_VT_MIN_ML = 9.0`) — that would silently misjudge RDS's 1.5 kg baby against a 3 kg reference. Instead, keep the **per-kg multiplier** as the constant and compute the bound from the scenario's own weight:

```python
# Module-level — multipliers only, not absolute volumes
VT_MIN_ML_PER_KG_ADULT:    float = 3.0    # existing behavior, unchanged
VT_MAX_ML_PER_KG_ADULT:    float = 12.0
VT_MIN_ML_PER_KG_NEONATE:  float = 4.0    # lung-protective floor — Spaeth 2022 / neonatal consensus
NEONATE_IBW_KG_DEFAULT:    float = 3.0    # fallback only if weight_kg is somehow absent

# Inside generate_breath_cycles(), after population is read:
if population == "neonate":
    weight = float(params.get("weight_kg", NEONATE_IBW_KG_DEFAULT))
    vt_min_ml = weight * VT_MIN_ML_PER_KG_NEONATE
    vt_max_ml = None   # DECIDED: no ceiling check for neonates — see below
else:
    vt_min_ml = IBW_KG * VT_MIN_ML_PER_KG_ADULT   # identical to current VT_MIN_ML
    vt_max_ml = IBW_KG * VT_MAX_ML_PER_KG_ADULT
```

**DECIDED: no `VT_MAX_ML_PER_KG_NEONATE`.** Same reasoning as the driving-pressure decision below — there's no sourced neonatal overdistension ceiling in the research done so far, and 8 mL/kg was a guess, not a finding. Rather than ship a fabricated ceiling, the neonatal validity filter checks the floor only (`delivered_vt < vt_min_ml`) and skips the `delivered_vt > vt_max_ml` branch entirely when `population == "neonate"`. This is a real, if modest, gap in safety coverage — a neonatal scenario with mechanically absurd overdistension wouldn't get caught by this specific check — worth a one-line comment in the validity filter noting the gap is intentional and pending a source, not an oversight. `PPEAK_MAX_CMHH2O` (which you're keeping — see 1c) still catches the most dangerous overdistension cases indirectly, since volume and pressure are coupled.

This also means `_validate_params()` should treat `weight_kg` as optional-with-a-population-appropriate-default rather than adding it to `REQUIRED_PARAMS` — simpler than building a new conditionally-required-field validation path that doesn't exist anywhere else in the codebase yet.

#### 1c. Constants that are NOT weight-dependent — only three of the original five survive contact with "is this actually sourced?"

Sorted by what happened to each one:

**Keep — reasonably grounded, add a `NEONATE_` counterpart:**
```python
NEONATE_PPEAK_MAX_CMHH2O:                float = 30.0  # neonatal barotrauma risk — MSD Manual PIP ranges
NEONATE_DEFAULT_CHEST_WALL_COMPLIANCE:   float = 12.0  # NOT ~inert here — first-order term, see turn 2 discussion
NEONATE_CIRCUIT_COMPLIANCE_ML_PER_CMH2O: float = 0.6   # dedicated low-compliance neonatal circuit, not the adult 2.5 value
```

**DECIDED: drop these — no `NEONATE_` version, reuse the adult constant for both populations:**
- `ETT_K1` / `ETT_K2` — no neonatal-tube-specific Rohrer coefficients turned up in research. This doesn't block anything: `resistance_cmH2O_L_s` (the global parameter, already correctly sourced at 50–150 for neonates) already carries the dominant physiology. `ETT_K1`/`ETT_K2` only refine the *nonlinear/turbulent* component of flow-dependent resistance on top of that — reusing the adult 7.5mm-tube values there means that one refinement is somewhat off for a 2.5–3.5mm tube, but the headline resistance number is right either way. Worth a one-line code comment noting this is a known simplification, not a silent gap.
- `PS_MAX_CMHH2O` — no neonatal-specific ceiling found either; it was already going to equal the adult value, so there's no actual second constant to define. Just don't branch this one.

**DECIDED: `DRIVING_P_MAX_CMHH2O` — omit the check for neonates, no `NEONATE_` version.** That number is grounded in Amato et al. 2015, an **adult** ARDS mortality study, with no neonatal counterpart found. The validity filter skips the driving-pressure branch entirely when `population == "neonate"` and relies on `PPEAK_MAX` + the VT floor instead. A missing check is more honest than a fabricated threshold, and it's easy to add back if a source turns up later.

So the actual net change to 1c versus the original draft: **three new `NEONATE_` constants, not five** — `PPEAK_MAX`, `DEFAULT_CHEST_WALL_COMPLIANCE`, `CIRCUIT_COMPLIANCE_ML_PER_CMH2O`. Everything else either reuses the adult constant unbranched, or the check is skipped outright.

#### 1d. The helper, and exactly where each surviving constant gets selected

```python
def _neonate_or_adult(population: str, neonate_val, adult_val):
    """Return neonate_val if population == 'neonate', else adult_val.
    Works for any type — floats, None, whatever a given constant needs."""
    return neonate_val if population == "neonate" else adult_val
```
Put this near the top of each of the five generator files, right after the constants block (Section 2, where `IBW_KG`/`PPEAK_MAX_CMHH2O`/etc. already live) — same location in all five files, so it's easy to find by analogy across the codebase.

With only three constants actually branching, the call sites are:

```python
# Inside generate_breath_cycles(), after population = params.get("population", "adult"):

ppeak_max = _neonate_or_adult(population, NEONATE_PPEAK_MAX_CMHH2O, PPEAK_MAX_CMHH2O)
C_chest   = float(params.get(
    "chest_wall_compliance_ml_per_cmH2O",
    _neonate_or_adult(population, NEONATE_DEFAULT_CHEST_WALL_COMPLIANCE, DEFAULT_CHEST_WALL_COMPLIANCE),
))
circuit_c = _neonate_or_adult(population, NEONATE_CIRCUIT_COMPLIANCE_ML_PER_CMH2O, CIRCUIT_COMPLIANCE_ML_PER_CMH2O)
```
— then use `ppeak_max` in place of the bare `PPEAK_MAX_CMHH2O` in the validity filter's `if ppeak > ppeak_max:` check, and pass `circuit_c` into `_circuit_vt_correction()` instead of the module constant directly (that function currently reads `CIRCUIT_COMPLIANCE_ML_PER_CMH2O` directly rather than taking it as an argument, so it needs a small signature change — `def _circuit_vt_correction(vt_raw, ppeak, peep, compensated=True, circuit_compliance=CIRCUIT_COMPLIANCE_ML_PER_CMH2O):` — to accept the population-selected value rather than always reading the adult module constant).

`K1_base`/`K2_base` and `PS_MAX_CMHH2O` need **no `_neonate_or_adult` call at all**, since they're not branching — just leave `ETT_K1`/`ETT_K2`/`PS_MAX_CMHH2O` exactly as they are today and let both populations flow through the same lines unchanged. `DRIVING_P_MAX_CMHH2O`'s validity-filter branch needs an `if population != "neonate":` guard around the existing check instead of a value substitution, since the whole check — not just its threshold — is being skipped.

### Blocker 2 — every dashboard slider's min/max/step is hardcoded to adult magnitude

```python
compliance = st.slider("Compliance (ml/cmH2O)", 5, 150, ...)   # RDS is 0.5–1
resistance = st.slider("Resistance (cmH2O/L/s)", 1, 50, ...)   # Neonate baseline is 50–150
tv         = st.slider("Tidal Volume (ml)", 100, 900, step=10, ...)  # Neonate VT is ~12–24
effort_rate = st.slider("Effort Rate (breaths/min)", 8, 40, ...)     # Neonate RR is 40–60+
```

Selecting "Normal Neonate" from the condition dropdown will load a preset whose values are **outside the slider's own min/max** for compliance, resistance, and tidal volume, and RDS's compliance (0.5–1) is below the slider floor entirely. Streamlit will clip `value=` to the nearest bound, so the dashboard will silently display and simulate the wrong number rather than erroring — the worst kind of failure for a project built around a validity filter that's supposed to catch exactly this class of problem.

**DECIDED: population branch in `render_sidebar()`.** Now that Blocker 1 confirmed `population` as the actual field name flowing through `conditions.py`, the branch should read it directly off the loaded preset rather than hardcoding the three neonatal condition names as a string list — that's both less code and means a fourth neonatal condition added to `conditions.py` later gets correct slider treatment automatically, with zero changes to `dashboard.py`:

```python
preset = get_condition(condition_name)
is_neonatal = preset.get("population", "adult") == "neonate"

compliance = st.slider(
    "Compliance (ml/cmH2O)",
    0.1, 8.0, value=float(preset["compliance_ml_per_cmH2O"]), step=0.1,
    key=f"compliance_{condition_name}_{engine_name}",
) if is_neonatal else st.slider(
    "Compliance (ml/cmH2O)", 5, 150, value=int(preset["compliance_ml_per_cmH2O"]), step=1,
    key=f"compliance_{condition_name}_{engine_name}",
)
```

Same treatment needed for resistance (extend range up to ~200), tidal volume (down to ~5–40 mL, step 1), and effort rate (up to ~70). This is mechanical but touches every mode's slider block in `render_sidebar()`, since compliance/resistance/PEEP are shared across all five engines and tidal volume/effort-rate sliders are duplicated per-engine within the function.

**New consequence of this approach, not obvious until you trace it through — and the actual shape is worth getting right.** `render_sidebar()` doesn't hand the whole `preset` dict to the generator, and it isn't one incremental `params["key"] = value` sequence either. It's an `if engine_key == "vcv": params = {...} elif engine_key == "pcv": params = {...} elif engine_key == "psv": params = {...} elif engine_key == "prvc": params = {...} elif engine_key == "simv": params = {...}` chain — five separate dict *literals*, one per engine, each with its own `"condition": condition_name,` entry written out by hand inside that engine's literal. `population` and `weight_kg` aren't in any of the five, so unless they're added, they'll be silently dropped from `params` even though they're sitting right there in `preset` — the generator falls back to `population = params.get("population", "adult")` and gets every threshold wrong, on the dashboard path specifically (direct calls, tests, and dataset scripts that pass `get_condition()`'s output straight through don't have this problem — only the dashboard's hand-built reconstruction does).

The useful part of this structure: all five branches assign to the same `params` variable, and all five fall through to one shared `return` at the end of the function — there's no need to add the fix inside each of the five literals. Add it once, after the `if/elif` chain closes, immediately before the existing `return params, condition_name, engine_name, n_cycles`:

```python
        elif engine_key == "simv":
            params = { ... }               # existing SIMV dict literal, unchanged
            if mode == "VC":
                params["tidal_volume_ml"] = tv
                params["flow_pattern"]    = flow_pattern
            else:
                params["insp_pressure_cmH2O"] = insp_pressure

        # ADD THESE TWO LINES — applies to all five engine branches at once,
        # since every branch above assigns to this same `params` variable.
        params["population"] = preset.get("population", "adult")
        params["weight_kg"]  = preset.get("weight_kg", 3.0 if is_neonatal else 70.0)

        return params, condition_name, engine_name, n_cycles
```
The `3.0`/`70.0` fallback is a local literal rather than an import of `NEONATE_IBW_KG_DEFAULT`/`IBW_KG` from a generator module — those constants live inside `vcv_generator.py` etc., and importing a generator-internal constant into `dashboard.py` just to use as a defensive fallback is more coupling than it's worth. In practice this fallback shouldn't ever fire once `conditions.py` has `population`/`weight_kg` set on all ten entries (per Section 1a and the neonatal entries below) — it only matters if a future condition is added to `conditions.py` without those fields.

---

## File-by-file changes

### 1. `generator/conditions.py`

Add three new entries to `CONDITIONS`, following the exact same shape as the existing seven (every field every mode needs, even though a given mode may ignore some of them — that's the existing convention, not something new). Proposed starting values, drawn from the physiology write-up already produced for this project (cite: neonatal parameter tables), **with every number that isn't directly literature-sourced flagged inline**:

```python
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
    "ett_leak_fraction":        0.15,                # NEW field — see Open Decision 1
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
    "ett_leak_fraction":        0.15,
},

"Meconium Aspiration Syndrome": {
    "label":       "Meconium Aspiration Syndrome",
    "description": (
        "Term/post-term infant with heterogeneous lung: ball-valve "
        "obstruction and gas trapping in some units, atelectatic/"
        "surfactant-inactivated collapse in others. Requires a "
        "two-compartment profile — NOT a rescaled COPD/Bronchospasm preset."
    ),
    "condition":                "Meconium Aspiration Syndrome",
    "population":               "neonate",
    "weight_kg":                3.2,
    "respiratory_rate":         45,                   # <50 to protect exp. time — Dargaville
    "stress_index":             1.10,                 # ASSUMPTION — heterogeneity proxy
    "tidal_volume_ml":          18,                    # ~5.5 mL/kg — Dargaville
    "compliance_ml_per_cmH2O":  2.5,                   # DIRECTION sourced, MAGNITUDE not — flag
    "resistance_cmH2O_L_s":     130,                   # DIRECTION sourced, MAGNITUDE not — flag
    "ie_ratio":                 0.80,                  # long Ti 0.5-0.7s — Goel & Nangia
    "rise_time_s":              0.05,                  # ASSUMPTION
    "peep_cmH2O":                5,
    "pressure_support_cmH2O":  14,
    "flow_cycle_threshold":     0.20,
    "trigger_threshold_cmH2O":  0.5,                   # ASSUMPTION
    "pmus_peak_cmH2O":          6,
    "effort_rate_per_min":     45,
    "effort_duration_s":        0.45,
    "pmus_cv":                  0.25,                  # ASSUMPTION
    "pressure_ceiling_cmH2O":  25,                     # PIP up to 30-40 reported — leaves headroom
    "ett_leak_fraction":        0.15,
},
```

`get_condition()`, `get_condition_meta()`, `list_conditions()`, `get_all_meta()`, and `_resolve_key()` need **no code changes** — they're already generic over `CONDITIONS.keys()`. Adding these three entries automatically populates the dashboard's condition dropdown. That's the one piece of this that's genuinely free.

---

### 2. `generator/{vcv,pcv,psv,prvc,simv}_generator.py` — all five, same edits in each

This is the part with real duplication risk. `COMPARTMENT_PROFILES` and `RECRUITMENT_SLOPES` are copy-pasted across all five files already (that's the same duplication your SIMV control-loop doc flagged as a `lung_physics.py` refactor candidate) — which means **each new condition has to be added correctly in five places**, and a miss in any one of them doesn't crash, it silently falls back to the adult `"Normal"` compartment profile (`COMPARTMENT_PROFILES.get(condition, COMPARTMENT_PROFILES["Normal"])`). That's a silent-wrong-physics bug, not a loud one — exactly the failure mode your docstring/compartment-count cross-referencing already caught once for Bronchospasm.

**Add to `COMPARTMENT_PROFILES` in all five files:**

```python
"Normal Neonate": [
    {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
     "R_exp_ratio": 1.2, "tethering": 0.80},   # identical shape to adult Normal
],
"RDS": [
    {"fraction": 1.00, "C_frac": 1.00, "R_frac": 1.00,
     "R_exp_ratio": 1.3, "tethering": 0.30},   # single compartment — OPEN DECISION, see below
],
"Meconium Aspiration Syndrome": [
    {"fraction": 0.50, "C_frac": 0.90, "R_frac": 2.60,
     "R_exp_ratio": 5.5, "tethering": 0.00},   # obstructive / ball-valve — no tethering, like Bronchospasm
    {"fraction": 0.50, "C_frac": 0.25, "R_frac": 1.00,
     "R_exp_ratio": 1.8, "tethering": 0.20},   # atelectatic / surfactant-inactivated — ARDS-like
],
```

**STILL OPEN — this is the one decision left in the whole document.** Every number in the MAS split above is a proposed engineering assumption, not a literature value — the physiology write-up was explicit that quantitative two-compartment MAS parameters don't exist for human neonates, only the qualitative structure (obstructive + atelectatic compartments) does. Given your stated preference to omit unsourced numbers rather than ship guesses, MAS genuinely doesn't fit the same "add the constant, flag it as an assumption" treatment the other items got — the *numbers* aren't sourced, but the *shape* (2 compartments) is, which makes MAS different from something like `NEONATE_ETT_K1` where both the number and the need for a separate value were equally ungrounded. Two honest ways to handle that split:

- **(a) Defer MAS entirely this round.** Ship Normal Neonate and RDS only — both have properly sourced or clearly-reasoned parameters throughout. Add MAS's `COMPARTMENT_PROFILES`/`RECRUITMENT_SLOPES`/`conditions.py` entries once the compartment split has either a literature source or explicit mentor sign-off that engineering estimates are acceptable for a first version.
- **(b) Ship the structure, mark the numbers.** Keep the 2-compartment entry (the compartment *count* is reasonably grounded — Yeh 2017 describes exactly this structure), but don't treat the fraction/`C_frac`/`R_frac`/`R_exp_ratio`/`tethering` values as anything but placeholders pending validation — the same way PRVC's ceiling-limited non-convergence is retained as a labeled, real state rather than hidden.

This document is written assuming **(a)** — the build order below now ships Normal Neonate and RDS as this round's deliverable, with MAS as a clearly separate follow-up once its compartment numbers have a real basis — but this is genuinely a judgment call between "ship nothing on MAS until it's sourced" and "ship the grounded structure with clearly-labeled placeholder numbers," and it's worth explicitly confirming which one you want rather than inferring it from the rest of this message's pattern.

**DECIDED: RDS ships as 1 compartment.** Matches the physiology write-up's description of RDS as comparatively homogeneous, and is the lower-risk starting point — revisit if validation shows the single-compartment version doesn't reproduce the expected pressure-time steepness.

**Add to `RECRUITMENT_SLOPES` in all five files (Normal Neonate and RDS now; MAS if/when Open Decision above resolves toward shipping it):**
```python
"Normal Neonate":               0.30,   # ASSUMPTION — modest PEEP recruitment, like adult Normal
"RDS":                          0.60,   # higher than adult ARDS — RDS is the textbook recruitable lung
```
Both are positive — neither is an obstructive-only condition where PEEP fails to recruit (that's the COPD/Bronchospasm=0.0 pattern), so both get a nonzero slope like the ARDS tiers. RDS's is set higher than any adult ARDS tier's because RDS is, more than any adult analog, *the* textbook recruitable lung — the whole disease is under-aerated-but-recruitable tissue, which is a stronger version of the same mechanism that gives Moderate ARDS the highest slope among the adult conditions. (If MAS ships later: its atelectatic compartment recruits the same way, but its obstructive compartment doesn't — same logic as why COPD/Bronchospasm are zero — so a MAS slope would land somewhere between RDS's and COPD's, weighted by compartment fraction, roughly the 0.20 proposed earlier, though that number inherits the same "not yet sourced" caveat as the rest of MAS's compartment data.)

**DECIDED: leak stays the existing fixed-fraction `cuff_leak` machinery, default-on.** Set `ett_leak_fraction`/`cuff_leak_fraction` as a parameter that's *on by default* for the neonatal presets (today, for adults, it's an opt-in complication that defaults off — for neonates it becomes the default state, worth a one-line docstring note since it's a small semantic shift for that parameter). This reuses existing code with zero new physics. The pressure-proportional version (`Q_leak(t) = k_leak * P_ao(t)`, built into the equation of motion rather than applied as a flat post-hoc percentage) stays a Phase 2 idea, worth revisiting only if the fixed-fraction approximation visibly fails to reproduce the "volume doesn't return to baseline within a breath" signature that's the most diagnostic neonatal waveform feature.

**Add to `CONDITION_TIERS`** (referenced from `prvc_generator.py` and shared across the thinning scripts) — two new tier rows this round; the MAS row is shown for reference and lands with its own CR:
```python
{"name": "Normal Neonate",              "compliance_range": (3, 6),    "compliance_step": 0.5,
 "resistance_range": (60, 100),  "resistance_step": 10, "n_cycles": 15},
{"name": "RDS",                         "compliance_range": (0.4, 1.2), "compliance_step": 0.1,
 "resistance_range": (60, 100),  "resistance_step": 10, "n_cycles": 15},
{"name": "Meconium Aspiration Syndrome","compliance_range": (1.5, 3.5), "compliance_step": 0.5,
 "resistance_range": (90, 160),  "resistance_step": 10, "n_cycles": 25},   # more cycles — auto-PEEP needs time to develop, like COPD/Bronchospasm
```

---

### 3. `tests/test_{vcv,pcv,psv,prvc,simv}_generator.py` — all five

Every test file that currently hardcodes the full condition list needs the new keys added, or those tests will keep passing without ever exercising the new conditions — this is exactly the "docstring says one thing, dictionary says another" class of bug your project has already caught once. This round that means **Normal Neonate and RDS**; add the equivalent MAS lines when its follow-on CR lands. Specifically, in each file:

- **`test_compartment_counts_match_documented_scheme`** (or equivalent) — extend the `expected` dict: `"Normal Neonate": 1, "RDS": 1`. Add `"Meconium Aspiration Syndrome": 2` when that CR lands.
- **`test_recruitment_slopes_positive_for_ards`**-style loops — add Normal Neonate and RDS to the "should have positive recruitment slope" group (both are, per the answer above) rather than leaving them untested by omission.
- **Add two new fixtures** per file (`NORMAL_NEONATE_PARAMS`, `RDS_PARAMS`), built from the `conditions.py` entries above, following the existing `NORMAL_PARAMS` / `SEVERE_ARDS_PARAMS` pattern already in each file. Add `MAS_PARAMS` alongside its CR.
- **New physiological-direction tests**, matching the project's existing style of asserting *relationships* rather than exact numbers (e.g. `test_bronchospasm_ppeak_exceeds_normal`):
  - `test_rds_time_to_peak_flow_shorter_than_normal_adult` — the short-τ signature.
  - `test_rds_driving_pressure_exceeds_normal_neonate` — stiffness signature, same shape as the existing ARDS-vs-Normal test.
  - `test_rds_uses_1_compartment`, `test_normal_neonate_uses_1_compartment`.
  - `test_neonatal_leak_reduces_patient_vt_below_delivered_vt` — reuses the existing `TestETTComplications` pattern, just asserted as always-on for these conditions rather than opt-in.
  - (Follow-on, with MAS: `test_mas_develops_auto_peep`, same pattern as `test_copd_develops_auto_peep`/`test_bronchospasm_develops_auto_peep`; `test_mas_uses_2_compartments`.)
- **Validity filter tests** need neonatal-scale boundary cases added alongside the existing adult ones (`test_vt_too_low_flagged` currently uses `tidal_volume_ml: 100` as a "too low" adult example — that number is *above* a real neonate's target VT, so a neonatal-scale version needs its own boundary values), and a test confirming the VT-ceiling and driving-pressure checks are genuinely skipped (not silently zero'd out) for `population == "neonate"`.
- Add a test analogous to `test_constants_consistent_with_ibw`, but for the neonatal weight-based floor: `test_neonate_vt_min_scales_with_weight_kg` — construct two RDS-population scenarios differing only in `weight_kg` (e.g. 1.5 vs 3.0) and assert the effective VT_MIN bound scales proportionally, not that it equals some fixed number.
- **`test_population_field_not_condition_name_drives_thresholds`** — this is the test that actually validates the design decision, not just its numbers: construct `{**NORMAL_PARAMS, "population": "neonate", "weight_kg": 3.0}` (an adult `"Normal"` condition string forced into the neonatal population branch) and assert the neonatal VT bounds/constants apply. This confirms the branch is genuinely keyed off `population`, not accidentally off `condition` name-matching somewhere.
- **`test_missing_population_defaults_to_adult`** — omit `population` entirely from an otherwise-valid params dict and assert behavior is identical to `population="adult"`, protecting the seven existing conditions from any regression.
- **`test_adult_conditions_unaffected_by_neonatal_constants`** — run all seven existing condition fixtures through the full test suite post-change and confirm identical `is_valid`/metric outputs to before the refactor (this is really a full-suite regression run rather than one new test, but worth stating as an explicit pass/fail gate before merging).

---

### 4. Dataset generation scripts (`generate_{vcv,pcv,psv,prvc,simv}_dataset_thinned.py`)

Two changes, one straightforward and one worth a decision:

**Straightforward:** each script's condition loop needs to pick up the three new `CONDITION_TIERS` rows automatically once those are added in the generator file — no script-level change needed if the scripts already iterate `CONDITION_TIERS` generically (they do, per the existing PRVC/PSV scripts).

**Worth a decision — scenario ID collisions.** `_make_scenario_id()` rounds compliance to the nearest whole number:
```python
f"_C{int(round(params['compliance_ml_per_cmH2O'])):03d}"
```
This is fine at adult scale (70 vs 71 are meaningfully different scenarios) but **breaks at neonatal/RDS scale**, where compliance values like 0.5, 0.6, 0.7, 0.8, and 0.9 mL/cmH₂O — a full sweep of RDS's clinically meaningful range — all round to `C001` and silently collide into the same scenario ID. This is precisely the "missing parameters from scenario ID encoding caused silent duplicate overwriting" failure mode already documented from PSV/PRVC development, just triggered by rounding precision instead of a missing key.

**DECIDED: population-gated ×10 scaling (`int(round(params['compliance_ml_per_cmH2O'] * 10)):03d` when `population == "neonate"`), not a uniform hundredths change.** Both were on the table; here's why the gated version won:

| | Population-gated ×10 (chosen) | Uniform ×100 for everyone |
|---|---|---|
| Existing adult datasets | Untouched — every ID already generated stays valid and comparable to new runs | Every adult ID's compliance field changes meaning; existing manifests/exports become inconsistent with anything regenerated after the change, effectively forcing a full adult dataset regeneration to stay consistent |
| Field width | Fits in the existing `:03d` (RDS's max ~1.2 × 10 = 12; Normal Neonate's max ~6 × 10 = 60) | Adult compliance up to 150 × 100 = 15,000 needs `:05d` — a wider schema change, not just added precision |
| Code | One `if population == "neonate":` branch inside `_make_scenario_id()` | No branch, but only because the whole ID scheme moved under it |
| Readability risk | A neonatal `C008` (=0.8) and an adult `C008` (=8) are only distinguishable by knowing which population you're looking at | None — one consistent rule everywhere |

That readability risk is the one real drawback, but it's mitigated by something already true of every scenario ID in this project: the condition name is embedded right in it (`PCV_RDS_C008_...`, `PCV_Normal_C008_...`). Nobody reading a scenario ID decodes the compliance field without also reading the condition — and the condition name alone already tells you which decoding rule applies. Given that, disturbing zero existing adult data outweighs a readability concern that the ID format already mostly covers for.

**DECIDED: parallel neonatal grids, and yes — separate `generate_{mode}_neonatal_dataset_thinned.py` scripts per mode, not a population branch inside the existing five.** Reasoning, since you asked specifically whether the separate-scripts structure is the cleaner one:

- **It matches the codebase's existing organizing principle.** You already have one script per *mode* rather than one script branching across modes internally; extending that to one script per *mode × population* is the same pattern applied one level further, not a new one.
- **It keeps the sweep loop linear and readable.** A single script branching `PARAMETER_GRID` vs. a hypothetical `NEONATAL_PARAMETER_GRID` mid-loop (different RR ranges, different VT/kg ranges, different `n_cycles` defaults for MAS-style auto-PEEP accumulation) turns the current clean `for combo in itertools.product(*values): ...` into something with a conditional generating its own `values` depending on which `CONDITION_TIERS` row is currently active — more branching exactly where the code most benefits from staying simple.
- **Independent regeneration.** You can re-run and iterate on the neonatal grid without re-running the (slower) full adult sweep every time, which matters more here than usual since the neonatal parameters are the ones still being validated.
- **The real cost:** boilerplate duplication — manifest writing, CSV export, the summary-printing block at the end of each script are effectively identical across all five *existing* scripts already, and doubling the script count doubles that duplication too. Given you're already carrying this duplication five times over for the boilerplate (separate from the `COMPARTMENT_PROFILES`-style duplication that actually varies by file), this is a reasonable moment to factor the shared manifest/CSV/summary code into one small `dataset_io_helpers.py` imported by all ten scripts — worth doing alongside this change rather than after, since you're touching every script's structure anyway.

---

### 5. `ui/dashboard.py`

Beyond the slider-range fix in Blocker 2:

- **No changes needed** to `render_header()`, `render_export()`, or the plotting code in `render_waveform_plot()` — these are already condition-agnostic and will handle the new presets correctly once the generators return valid data.
- **`render_metrics()`** — check whether `patient_vt_ml` (vs `delivered_vt_ml`) is already surfaced as a metric card for modes where cuff leak is active; if leak becomes default-on for neonatal conditions (per the Phase 1 leak decision above), that gap between the two numbers becomes a headline metric for these three conditions specifically and probably deserves a visible card rather than being buried in the returned dict.
- **`_pcv_default_driving_pressure()`** — this helper's math (fill-fraction formula) is condition-agnostic and should compute a sensible default at neonatal scale automatically, no change needed, but worth a manual check once real neonatal numbers flow through it given how small τ and C get.

---

## Suggested build order

Matching the staged approach already used for prior modes (control loop → parameter grid → generator → test → dataset → dashboard), and given that the two blockers above are shared infrastructure rather than per-condition work:

1. Implement Blocker 1 (population-gated constants, Sections 1a–1d) and Blocker 2 (population-branched sliders, including the corrected `params["population"]`/`params["weight_kg"]` insertion point above) as their own small CR — infrastructure, not condition work, and every later step depends on it. Land the tests from Section 3 that validate the *design* (population-vs-condition decoupling, missing-population default, weight-scaling, adult-regression) as part of this same CR, before any neonatal condition data exists to test against. Includes the seven `"population": "adult"` backfills.
2. Add the `conditions.py` entries for **Normal Neonate and RDS** (no dependencies beyond #1). MAS's entry waits on the open decision below.
3. Implement **Normal Neonate** end-to-end across all five generators + tests first — it's the simplest (one compartment, no new physics beyond scale + leak), and validates that the constants/slider fix actually works before RDS adds complexity on top.
4. Implement **RDS** — parameter variant of Normal Neonate (drop compliance, keep resistance, shorten τ), still no new architecture.
5. Dataset thinning: the two new `generate_{mode}_neonatal_dataset_thinned.py` scripts per mode, plus the shared `dataset_io_helpers.py` extraction, plus the scenario-ID precision fix — once Normal Neonate and RDS are validated in the generators.
6. Dashboard metric-card update for leak visibility.
7. **MAS, as its own follow-on CR**, once the compartment-parameter sourcing question is resolved — implement using whichever of the two paths (defer vs. ship-with-labeled-placeholders) gets confirmed.

---

## Open decisions

Everything raised in the original draft is now resolved except one:

**MAS compartment parameters.** The fraction/`C_frac`/`R_frac`/`R_exp_ratio`/`tethering` values are engineering estimates, not literature values — only the 2-compartment *structure* is grounded (Yeh 2017), not these specific numbers. This document is written assuming MAS is deferred to a follow-on CR (Section 2's "STILL OPEN" note and the build order above both reflect that), but confirm that's actually what you want rather than the alternative (ship the grounded 2-compartment structure now with the numbers explicitly labeled as placeholders pending validation, the same way PRVC's ceiling-limited non-convergence is retained as a labeled real state rather than hidden).

---

## Acceptance criteria (draft)

**This round (Normal Neonate + RDS):**
- `conditions.py` contains two new entries (plus `"population": "adult"` added to all seven existing ones) with every field the five generators require, and `list_conditions()` includes them with no other code changes to that file.
- All five `COMPARTMENT_PROFILES` dictionaries (vcv/pcv/psv/prvc/simv) contain matching 1-compartment entries for both new conditions, cross-checked against each file's own compartment-count test (closing the exact gap that caused the earlier Bronchospasm docstring/dictionary mismatch).
- Neonatal-scale scenarios pass the VT floor check and are no longer rejected by adult-sized `VT_MIN_ML`/`PPEAK_MAX_CMHH2O` thresholds; the VT-ceiling and driving-pressure checks are intentionally absent for `population == "neonate"`, not silently broken.
- Dashboard sliders can express the full literature-sourced range for both new conditions without value-clipping on selection, and `params["population"]`/`params["weight_kg"]` reach the generator on every engine.
- RDS scenarios show resistance at the neonatal baseline (not elevated) and a shorter time-to-peak-flow than Normal Neonate at matched mechanics.
- Leak is on by default for both neonatal conditions across every mode, and `patient_vt_ml` is measurably below `delivered_vt_ml`/`insp_vt` wherever the existing cuff-leak metric is reported.
- Scenario IDs remain unique across the full neonatal parameter grid, including RDS's sub-1-unit compliance sweep (verifies the ×10 precision fix).
- All existing adult-condition tests continue to pass unchanged (no regression from the constants refactor), including a full-suite run confirming identical `is_valid`/metric outputs to before the refactor.

**Follow-on (MAS), once the open decision above resolves:**
- MAS scenarios develop measurable auto-PEEP over a multi-cycle run and use exactly 2 compartments.
- Compartment parameters are either literature-sourced or explicitly signed off as documented placeholders — not silently treated as validated.

---

## Files likely to be touched

- **Update:** `generator/conditions.py` — two new CONDITIONS entries (Normal Neonate, RDS) plus `"population": "adult"` added to the existing seven; MAS entry follows in its own CR
- **Update:** `generator/vcv_generator.py`, `pcv_generator.py`, `psv_generator.py`, `prvc_generator.py`, `simv_generator.py` — `_neonate_or_adult()` helper, three new `NEONATE_` constants (`PPEAK_MAX`, `DEFAULT_CHEST_WALL_COMPLIANCE`, `CIRCUIT_COMPLIANCE_ML_PER_CMH2O`), weight-scaled VT floor, `COMPARTMENT_PROFILES`/`RECRUITMENT_SLOPES`/`CONDITION_TIERS` entries for the two new conditions, default-on leak for neonatal conditions, driving-pressure and VT-ceiling checks skipped for `population == "neonate"`
- **Update:** `tests/test_vcv_generator.py`, `test_pcv_generator.py`, `test_psv_generator.py`, `test_prvc_generator.py`, `test_simv_generator.py` — new fixtures, extended compartment-count/recruitment-slope tables, population-vs-condition design tests, neonatal-scale validity boundary tests
- **Create:** `generate_vcv_neonatal_dataset_thinned.py`, `generate_pcv_neonatal_dataset_thinned.py`, `generate_psv_neonatal_dataset_thinned.py`, `generate_prvc_neonatal_dataset_thinned.py`, `generate_simv_neonatal_dataset_thinned.py` — parallel to the existing five, own `NEONATAL_PARAMETER_GRID`, scenario IDs use the ×10 compliance-precision branch
- **Create:** `dataset_io_helpers.py` — manifest/CSV/summary-printing code factored out of all ten dataset scripts
- **Update:** `ui/dashboard.py` — population-aware slider bounds in `render_sidebar()`, the `params["population"]`/`params["weight_kg"]` insertion before the shared `return`, possible leak metric card in `render_metrics()`
- **Update:** `EXPERIMENT_LOG.md` — record this CR's decisions
- **Update:** `ARCHITECTURE.md` — document the neonatal population extension as a scoped addition, not a rewrite

---

## Status

**Decided, not yet implemented.** Every architectural fork raised in the original draft is resolved except one (MAS compartment parameters — see Open Decisions). Ready to start with Blocker 1 + Blocker 2 as the first CR, per the build order above.
