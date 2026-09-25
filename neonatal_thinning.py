"""
generator/neonatal_thinning.py
-------------------------------
Shared config and helpers for the five generate_<mode>_neonatal_dataset_thinned.py
scripts (vcv, pcv, psv, prvc, simv).

WHY THIS FILE EXISTS
Two different kinds of values feed the neonatal thinned scripts, and they
are handled very differently on purpose:

  1. Facts that already live elsewhere in the codebase -- which conditions
     count as neonatal, what each one weighs, its compliance/resistance
     sweep range, its n_cycles, its recruitment slope, its scenario-ID
     formatting, and all of the underlying population-gated physics --
     are NEVER duplicated here or in the five scripts. Each script
     imports them directly from the relevant generator/<mode>_generator.py
     module at call time. If conditions.py or a generator file changes one
     of these later (a revised weight_kg, a corrected recruitment slope,
     a new neonatal condition such as MAS once its two-compartment model
     lands), every neonatal script picks up the change on its next run
     with zero edits, here or anywhere else.

  2. The THINNED ventilator/patient-effort grid is a dataset-generation
     *design decision*, not a fact that lives anywhere else in the
     codebase -- exactly like the adult generate_<mode>_dataset_thinned.py
     scripts, which each define their own THINNED_PARAMETER_GRID locally
     rather than importing the generator's full PARAMETER_GRID. Those
     values are defined ONCE, here, so the five neonatal scripts share one
     copy instead of five. Every value below is commented with its source:
     the two neonatal condition presets (Normal Neonate, RDS) where a
     direct anchor exists, or ASSUMPTION where the project's own
     literature audit hasn't sourced a neonatal-specific number -- the
     same ASSUMPTION-tagging convention conditions.py and every adult
     thinned script already use.

This file contains no compliance/resistance/Rohrer/recruitment physics.
That all still lives in each generator/<mode>_generator.py, population-
gated, exactly as CR0023 built it.
"""

from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Tier selection and mechanics grid -- derived, never hardcoded
# ---------------------------------------------------------------------------

def neonatal_tiers(condition_tiers: List[Dict],
                    neonate_weight_map: Dict[str, float]) -> List[Dict]:
    """
    Return only the CONDITION_TIERS entries belonging to the neonatal
    population, identified by membership in NEONATE_CONDITION_WEIGHT_KG
    rather than a hardcoded tier-name list ("Normal Neonate", "RDS").

    This is the single mechanism that keeps the five neonatal scripts in
    sync with the generator files without editing: add a condition to
    both CONDITION_TIERS and NEONATE_CONDITION_WEIGHT_KG in any generator
    module (MAS, once its two-compartment model lands, being the obvious
    next one) and every neonatal thinned script sweeps it on the next
    run, automatically.
    """
    return [t for t in condition_tiers if t["name"] in neonate_weight_map]


def mechanics_grid(tier: Dict) -> List[Tuple[float, float]]:
    """
    Expand one CONDITION_TIERS entry's compliance_range/resistance_range
    into the explicit (compliance, resistance) pairs to sweep. Mirrors
    the _mechanics_grid() helper every adult thinned script already
    defines locally -- kept here once instead of five times.
    """
    c_lo, c_hi = tier["compliance_range"]
    c_step = tier["compliance_step"]
    r_lo, r_hi = tier["resistance_range"]
    r_step = tier["resistance_step"]

    compliances = _float_range(c_lo, c_hi, c_step)
    resistances = _float_range(r_lo, r_hi, r_step)

    return [(c, r) for c in compliances for r in resistances]


def _float_range(lo: float, hi: float, step: float) -> List[float]:
    """
    Inclusive float range. Plain arange-style iteration drifts on the
    sub-1.0 steps RDS's own compliance grid needs (0.4 to 1.2 step 0.1),
    so the step count is computed once and each value rebuilt from it.
    """
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 6) for i in range(n + 1)]


# ---------------------------------------------------------------------------
# Dyssynchrony label helpers (PSV, SIMV spontaneous breaths)
# ---------------------------------------------------------------------------

def dyssynchrony_counts(labels: List[str]) -> Counter:
    return Counter(labels)


def dominant_dyssync(labels: List[str]) -> str:
    if not labels:
        return ""
    return Counter(labels).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Shared thinned grid dimensions
# ---------------------------------------------------------------------------
# Every value below is commented with its source. Where the two neonatal
# presets (Normal Neonate, RDS) agree or nearly agree, that value anchors
# the grid. Where no primary source exists for a neonatal-specific
# number, the nearest adult thinned-script convention is carried over and
# flagged ASSUMPTION.

# Tidal volume, mL PER KG OF THE CONDITION-SPECIFIC weight_kg (never a
# fixed adult IBW_KG -- see NEONATE_CONDITION_WEIGHT_KG in each generator
# module, applied by each script at combo-build time). Anchors: RDS
# preset = 6 mL / 1.5 kg = 4.0 mL/kg (preterm protective floor); Normal
# Neonate preset = 15 mL / 3.0 kg = 5.0 mL/kg. 6.0 mL/kg added as the
# upper bookend, mirroring the adult grid's low/standard/upper-standard
# spread (4/6/10 mL/kg IBW).
TIDAL_VOLUME_ML_PER_KG = [4.0, 5.0, 6.0]

# Respiratory rate, bpm (VCV/PCV/PRVC's fully-controlling rate). Both
# presets use 50 bpm. 40/60 added as ASSUMPTION bookends spanning the
# term-to-preterm neonatal range ARCHITECTURE.md's neonatal section
# frames as 30-60+ bpm.
RESPIRATORY_RATE = [40.0, 50.0, 60.0]

# SIMV's own mandatory (backup) rate is a distinct dimension -- kept
# separate from RESPIRATORY_RATE above rather than reused, exactly as
# adult simv_generator's own thinned script gives its RR dimension a
# different range than vcv/pcv/prvc's. Neonates don't tolerate the long
# apnea an adult weaning backup rate implies, so the bookends sit higher
# than adult SIMV's [4.0, 12.0] -- ASSUMPTION, ordinary NICU SIMV backup
# range.
SIMV_BACKUP_RATE = [15.0, 30.0]

# PEEP, cmH2O. Presets: Normal Neonate = 5, RDS = 6. Kept as two values
# (not thinned to one the way several adult grids are) because RDS
# management routinely titrates PEEP upward from the term-neonate
# baseline -- a clinically distinct strategy, not just a vertical shift
# within one condition.
PEEP_CMH2O = [5.0, 7.0]

# I:E ratio. Presets: Normal Neonate = 0.50 (1:2), RDS = 0.33 (1:3, short
# Ti for the stiff preterm lung). Both preset values kept directly rather
# than the adult grid's three-point [1.0, 0.5, 0.33] spread -- 1:1 has no
# neonatal preset anchor and was dropped rather than guessed.
IE_RATIO = [0.50, 0.33]

# Rise time, s. Presets: Normal Neonate = 0.05 (ASSUMPTION, unsourced in
# its own preset comment), RDS = 0.03 (ASSUMPTION, unsourced). Thinned to
# one value, matching the adult PSV/SIMV thinned-script precedent for
# this dimension (full range stays available via each generator's own
# PARAMETER_GRID for targeted runs). 0.05 s chosen as the less aggressive
# of the two unsourced preset values.
RISE_TIME_S = 0.05

# Flow pattern (VCV/PRVC/SIMV-VC). No population-dependent physics
# difference is documented anywhere for this dimension, so the adult
# convention is reused as-is.
FLOW_PATTERN = ["square", "decelerating"]

# Inspiratory (driving) pressure above PEEP, cmH2O -- PCV and SIMV-PC.
# Both neonatal presets cap pressure_ceiling_cmH2O at 20; bookends chosen
# to bracket typical neonatal PIP-PEEP driving pressures while staying at
# or under that ceiling. ASSUMPTION -- insp_pressure_cmH2O is a PCV-only
# field, so neither preset anchors it directly.
INSP_PRESSURE_CMH2O = [8.0, 14.0, 18.0]

# Pressure ceiling, cmH2O -- PRVC's own signature parameter, thinned
# least aggressively of any dimension here, mirroring how adult prvc's
# own thinned script treats pressure_ceiling and adult simv's treats
# f_window. Center value (20) matches both neonatal presets exactly;
# 16/24 are ASSUMPTION bookends.
PRESSURE_CEILING_CMH2O = [16.0, 20.0, 24.0]

# Pressure support, cmH2O -- PSV and SIMV spontaneous breaths. Presets:
# Normal Neonate = 8, RDS = 10. 6/14 added as ASSUMPTION weaning/high-
# support bookends, mirroring the adult PS grid's weaning/standard/high
# framing.
PRESSURE_SUPPORT_CMH2O = [6.0, 10.0, 14.0]

# Flow-cycle threshold -- PSV and SIMV spontaneous breaths. Both presets
# use 0.15, and the Normal Neonate preset's own comment states the
# neonatal range is 5-20% vs. the adult ~25% default -- so this grid
# brackets that explicitly-documented neonatal range directly, rather
# than reusing the adult grid's own values.
FLOW_CYCLE_THRESHOLD = [0.10, 0.15, 0.20]

# Trigger threshold, cmH2O. Both presets use 0.5 (ASSUMPTION, weak
# effort, unsourced in either preset's own comment). Kept as a single
# value, matching the adult PSV/SIMV thinned-script convention for this
# dimension.
TRIGGER_THRESHOLD_CMH2O = 0.5

# Patient inspiratory effort, Pmus peak, cmH2O. Presets: Normal Neonate =
# 5, RDS = 6 (both flagged ASSUMPTION in their own preset comments). Two
# values kept as weak/stronger-effort bookends, mirroring the adult
# grid's own two-point Pmus-peak spread.
PMUS_PEAK_CMH2O = [4.0, 8.0]

# Effort rate, breaths/min. Both presets use 50 -- kept as a single
# value, matching the adult thinned-script convention that this
# dimension has the weakest case for multiple values.
EFFORT_RATE_PER_MIN = 50.0

# Effort duration, s. Presets: Normal Neonate = 0.35, RDS = 0.30.
# Averaged to a single representative value (0.32), matching the adult
# convention of thinning this dimension to one point.
EFFORT_DURATION_S = 0.32

# Pmus coefficient of variation. Presets: Normal Neonate = 0.20, RDS =
# 0.25 (both ASSUMPTION). Averaged to a single representative value
# (0.22), matching adult convention.
PMUS_CV = 0.22
