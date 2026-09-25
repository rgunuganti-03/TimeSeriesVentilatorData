NORMAL_PARAMS = {
    "vt_target_ml": 420.0,
    "respiratory_rate": 16,
    "peep_cmH2O": 5,
    "ie_ratio": 0.5,
    "pressure_ceiling_cmH2O": 25.0,
    "compliance_ml_per_cmH2O": 80.0,
    "resistance_cmH2O_L_s": 10.0,
    "condition": "Normal",
}

SEVERE_ARDS_PARAMS = {
    **NORMAL_PARAMS,
    "compliance_ml_per_cmH2O": 18.0,
    "resistance_cmH2O_L_s": 16.0,
    "condition": "Severe ARDS",
    "pressure_ceiling_cmH2O": 15.0,
}
NORMAL_NEONATE_PARAMS = {
    "condition":                "Normal Neonate",
    "population":               "neonate",
    "weight_kg":                3.0,
    "respiratory_rate":         50,
    "compliance_ml_per_cmH2O":  4.0,
    "resistance_cmH2O_L_s":     80,
    "peep_cmH2O":               5,
    "ie_ratio":                 0.50,
    "rise_time_s":              0.05,
    "pressure_support_cmH2O":   6.0,    
    "flow_cycle_threshold":     0.15,   
    "trigger_threshold_cmH2O":  0.5,    
    "pmus_peak_cmH2O":          5.0,    
    "effort_rate_per_min":      50,     
    "effort_duration_s":        0.35,   
    "pmus_cv":                  0.20,   
}

RDS_PARAMS = {
    **NORMAL_NEONATE_PARAMS,
    "condition":                "RDS",
    "weight_kg":                1.5,
    "compliance_ml_per_cmH2O":  0.75,
    "resistance_cmH2O_L_s":     80,     # unchanged from Normal Neonate — NOT elevated
    "ie_ratio":                 0.33,
    "rise_time_s":              0.03,
    "peep_cmH2O":                6,
    "pressure_support_cmH2O":   6.0,     
    "pmus_peak_cmH2O":          6,      
    "effort_duration_s":        0.30,   
    "pmus_cv":                  0.25,   
    "stress_index":             0.85,   
}

import numpy as np

from generator.conditions import get_condition
from generator.conditions import get_condition
import generator.pcv_generator  as pcv
import generator.psv_generator  as psv
import generator.prvc_generator as prvc
import generator.simv_generator as simv


def compare_leak(engine_name, generate_fn, extra_params=None, n_cycles=5):
    params = get_condition("RDS")
    if extra_params:
        params.update(extra_params)

    r_leak = generate_fn(params, n_cycles=n_cycles)
    r_no_leak = generate_fn({**params, "ett_cuff_leak_fraction": 0.0}, n_cycles=n_cycles)

    vt_leak    = r_leak["delivered_vt_ml"]
    vt_no_leak = r_no_leak["delivered_vt_ml"]

    print(f"--- {engine_name} ---")
    print(f"delivered_vt_ml, leak on  : {vt_leak:.2f} mL")
    print(f"delivered_vt_ml, leak off : {vt_no_leak:.2f} mL")
    print(f"reduction: {1.0 - vt_leak / vt_no_leak:.1%}\n")


# --- PCV ---
# PCV is pressure-controlled, so it needs a commanded inspiratory pressure --
# the same gap VCV had with flow_pattern. RDS's preset doesn't carry
# insp_pressure_cmH2O, so I'm supplying a guess (PEEP=6 + ~10 driving
# pressure). I haven't directly confirmed pcv_generator's REQUIRED_PARAMS
# in this conversation, so if this throws "Missing required parameter(s)",
# check generator/pcv_generator.py's _validate_params for the exact name(s)
# it wants and adjust here.
compare_leak("PCV", pcv.generate_breath_cycles,
             extra_params={"insp_pressure_cmH2O": 16.0})

# --- PSV ---
# psv_generator's REQUIRED_PARAMS is fully covered by RDS's preset already
# (pressure_support_cmH2O, peep_cmH2O, rise_time_s, flow_cycle_threshold,
# trigger_threshold_cmH2O, pmus_peak_cmH2O, effort_rate_per_min,
# effort_duration_s, pmus_cv, compliance_ml_per_cmH2O, resistance_cmH2O_L_s)
# -- confirmed, no extra params needed.
compare_leak("PSV", psv.generate_breath_cycles)

# --- PRVC ---
# prvc_generator.REQUIRED_PARAMS uses "vt_target_ml", not the
# "tidal_volume_ml" key conditions.py stores RDS's target volume under --
# map it across. Also give it more cycles since PRVC needs several breaths
# to converge (the RDS smoke test elsewhere in the codebase uses 15).
compare_leak("PRVC", prvc.generate_breath_cycles,
             extra_params={"vt_target_ml": get_condition("RDS")["tidal_volume_ml"]},
             n_cycles=15)



# compare_leak("SIMV", simv.generate_breath_cycles,
#              extra_params={
#                  "mandatory_mode":   "VC",
#                  "flow_pattern":     "square",
#                  "f_window":         0.20,
#                  "respiratory_rate": 30,   # TEST-ONLY override to clear the 10–40 bpm
#                                            # validity gate — RDS's real rate (50) isn't
#                                            # physiologically represented by this run
#              })

params = get_condition("RDS")
params.update({
    "mandatory_mode":   "VC",
    "flow_pattern":     "square",
    "f_window":         0.20,
    "respiratory_rate": 30,   # TEST-ONLY override, same caveat as before
})
r = simv.generate_breath_cycles(params, n_cycles=5)
print(sorted(r.keys()))

def compare_leak_simv(engine_name, generate_fn, extra_params, n_cycles=5):
    params = get_condition("RDS")
    params.update(extra_params)

    r_leak    = generate_fn({**params, "ett_cuff_leak_fraction": 0.15}, n_cycles=n_cycles)
    r_no_leak = generate_fn({**params, "ett_cuff_leak_fraction": 0.0},  n_cycles=n_cycles)

    vt_leak    = r_leak["mandatory_delivered_vt_ml"]
    vt_no_leak = r_no_leak["mandatory_delivered_vt_ml"]

    print(f"--- {engine_name} ---")
    print(f"mandatory_delivered_vt_ml, leak on  : {vt_leak:.2f} mL")
    print(f"mandatory_delivered_vt_ml, leak off : {vt_no_leak:.2f} mL")
    print(f"reduction: {1.0 - vt_leak / vt_no_leak:.1%}\n")


compare_leak_simv("SIMV", simv.generate_breath_cycles,
                   extra_params={
                       "mandatory_mode":   "VC",
                       "flow_pattern":     "square",
                       "f_window":         0.20,
                       "respiratory_rate": 30,
                   })

# --- SIMV ---
# simv_generator needs mandatory_mode set explicitly, and for VC mode,
# tidal_volume_ml/flow_pattern (already present/added) plus f_window --
# none of which live in the base conditions.py preset, same category of
# gap as VCV's flow_pattern.
# compare_leak("SIMV", simv.generate_breath_cycles,
#              extra_params={
#                  "mandatory_mode": "VC",
#                  "flow_pattern":   "square",
#                  "f_window":       0.20,
#              })