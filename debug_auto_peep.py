# NORMAL_PARAMS = {
#     "vt_target_ml": 420.0,
#     "respiratory_rate": 16,
#     "peep_cmH2O": 5,
#     "ie_ratio": 0.5,
#     "pressure_ceiling_cmH2O": 25.0,
#     "compliance_ml_per_cmH2O": 80.0,
#     "resistance_cmH2O_L_s": 10.0,
#     "condition": "Normal",
# }

# SEVERE_ARDS_PARAMS = {
#     **NORMAL_PARAMS,
#     "compliance_ml_per_cmH2O": 18.0,
#     "resistance_cmH2O_L_s": 16.0,
#     "condition": "Severe ARDS",
#     "pressure_ceiling_cmH2O": 15.0,
# }
# NORMAL_NEONATE_PARAMS = {
#     "condition":                "Normal Neonate",
#     "population":               "neonate",
#     "weight_kg":                3.0,
#     "respiratory_rate":         50,
#     "compliance_ml_per_cmH2O":  4.0,
#     "resistance_cmH2O_L_s":     80,
#     "peep_cmH2O":               5,
#     "ie_ratio":                 0.50,
#     "rise_time_s":              0.05,
#     "pressure_support_cmH2O":   6.0,    
#     "flow_cycle_threshold":     0.15,   
#     "trigger_threshold_cmH2O":  0.5,    
#     "pmus_peak_cmH2O":          5.0,    
#     "effort_rate_per_min":      50,     
#     "effort_duration_s":        0.35,   
#     "pmus_cv":                  0.20,   
# }

# RDS_PARAMS = {
#     **NORMAL_NEONATE_PARAMS,
#     "condition":                "RDS",
#     "weight_kg":                1.5,
#     "compliance_ml_per_cmH2O":  0.75,
#     "resistance_cmH2O_L_s":     80,     # unchanged from Normal Neonate — NOT elevated
#     "ie_ratio":                 0.33,
#     "rise_time_s":              0.03,
#     "peep_cmH2O":                6,
#     "pressure_support_cmH2O":   6.0,     
#     "pmus_peak_cmH2O":          6,      
#     "effort_duration_s":        0.30,   
#     "pmus_cv":                  0.25,   
#     "stress_index":             0.85,   
# }

# import numpy as np

# from generator.conditions import get_condition
# from generator.conditions import get_condition
# import generator.pcv_generator  as pcv
# import generator.psv_generator  as psv
# import generator.prvc_generator as prvc
# import generator.simv_generator as simv




import itertools
import numpy as np
from collections import Counter
from generator.prvc_generator import (
    generate_breath_cycles, PARAMETER_GRID, IBW_KG,
    PRESSURE_FLOOR_ABOVE_PEEP, ADAPTATION_STEP_CMH2O_DEFAULT, VT_TOLERANCE_FRAC_DEFAULT,
)

def classify(r, params):
    if r["converged"]:
        return "converged"
    if r["ceiling_limited"]:
        return "ceiling_limited"
    traj = np.asarray(r["pressure_trajectory"])[1:]
    if traj[-1] <= params["peep_cmH2O"] + PRESSURE_FLOOR_ABOVE_PEEP + 0.01:
        return "FLOOR_PINNED"
    d = np.diff(traj[-5:])
    d = d[np.abs(d) > 1e-9]
    if len(d) >= 2 and np.all(np.sign(d[1:]) != np.sign(d[:-1])):
        return "OSCILLATING"
    return "DRIFTING_OR_OTHER"

targets = PARAMETER_GRID["vt_target_ml_per_kg"]
other_keys = ["respiratory_rate", "peep_cmH2O", "ie_ratio", "pressure_ceiling_cmH2O"]
other_combos = list(itertools.product(*[PARAMETER_GRID[k] for k in other_keys]))
SAMPLE_PER_TARGET = 40



def build_params(C, target, combo_idx=0):
    combo = other_combos[combo_idx]
    params = dict(zip(other_keys, combo))
    params.update({
        "vt_target_ml": target * IBW_KG,
        "compliance_ml_per_cmH2O": C,
        "resistance_cmH2O_L_s": 10.0,
        "condition": "Normal",
        "adaptation_step_cmH2O": ADAPTATION_STEP_CMH2O_DEFAULT,
        "vt_tolerance_frac": VT_TOLERANCE_FRAC_DEFAULT,
    })
    return params

cases = [("C=60 target=4",  build_params(60.0, 4)),
         ("C=100 target=6", build_params(100.0, 6)),
         ("C=100 target=8", build_params(100.0, 8))]







for condition, C_range in [("Mild ARDS", (40.0, 55.0)), ("COPD", (50.0, 130.0))]:
    for C in C_range:
        tally = Counter()
        for target in targets:
            for combo in other_combos[:40]:
                params = dict(zip(other_keys, combo))
                params.update({"vt_target_ml": target * IBW_KG, "compliance_ml_per_cmH2O": C,
                               "resistance_cmH2O_L_s": 10.0, "condition": condition,
                               "adaptation_step_cmH2O": ADAPTATION_STEP_CMH2O_DEFAULT,
                               "vt_tolerance_frac": VT_TOLERANCE_FRAC_DEFAULT})
                try:
                    r = generate_breath_cycles(params, n_cycles=12)
                except Exception:
                    continue
                tally[(target, classify(r, params))] += 1
        print(condition, C)
        for k in sorted(tally): print("  ", k, tally[k])