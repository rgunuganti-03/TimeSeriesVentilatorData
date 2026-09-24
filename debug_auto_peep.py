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





import generator.psv_generator as psv

p_stress = {**NORMAL_NEONATE_PARAMS, "stress_index": 0.85,
            "pressure_support_cmH2O": 15.0, "pmus_peak_cmH2O": 15.0}
r = psv.generate_breath_cycles(p_stress, n_cycles=10, seed=50)
print("delivered_vt_ml:", r["delivered_vt_ml"])
print("patient_vt_ml:", r["patient_vt_ml"])
print("ppeak_cmH2O:", r["ppeak_cmH2O"])
print("is_valid:", r["is_valid"], r["invalid_reason"])

r_rds = psv.generate_breath_cycles(RDS_PARAMS, n_cycles=10, seed=42)
print("RDS:", r_rds["is_valid"], r_rds["delivered_vt_ml"], r_rds["ppeak_cmH2O"])
#                             V_turnover_ratio=2.5, stress_index_decline=15.0):
#     if abs(stress_index - 1.0) < 0.01 or V_mL <= 0.0:
#         return C_base
#     V_turnover = V_turnover_ratio * max(V_ref, 1.0)
#     if V_mL <= V_turnover:
#         V_norm = max(V_mL / max(V_ref, 1.0), 0.01)
#         return float(C_base * (V_norm ** (1.0 - stress_index)))
#     V_norm_at_turnover = V_turnover / max(V_ref, 1.0)
#     C_turnover = C_base * (V_norm_at_turnover ** (1.0 - stress_index))
#     V_norm_past_turnover = V_mL / V_turnover
#     return float(C_turnover * (V_norm_past_turnover ** (1.0 - stress_index_decline)))


# import generator.simv_generator as simv
# _original_simv = simv._compliance_nonlinear
# simv._compliance_nonlinear = _compliance_two_regime

# r_copd = simv.generate_breath_cycles(
#     {**COPD_PARAMS, "stress_index": 0.85, "effort_rate_per_min": 25.0,
#      "pmus_peak_cmH2O": 15.0, "trigger_threshold_cmH2O": 1.0},
#     n_cycles=10, seed=37)

# simv._compliance_nonlinear = _original_simv

# print("spontaneous_delivered_vt_ml:", r_copd["spontaneous_delivered_vt_ml"])
# print("(for reference: 782.85 original bug, 746.76 with si_decline=2.5, "
#       "579.72 bell-curve d=0.3, 384-457 bell-curve final)")

# import generator.pcv_generator as pcv
# _original_pcv = pcv._compliance_nonlinear
# pcv._compliance_nonlinear = _compliance_two_regime

# p_neo_low_c = {**NORMAL_NEONATE_PARAMS, "stress_index": 0.85, "rise_time_s": 0.0}
# r1 = pcv.generate_breath_cycles(p_neo_low_c, n_cycles=5)

# p_neo_stress = {**NORMAL_NEONATE_PARAMS, "stress_index": 0.85, "rise_time_s": 0.0,
#                  "respiratory_rate": 20, "ie_ratio": 1.0}
# r2 = pcv.generate_breath_cycles(p_neo_stress, n_cycles=5)

# pcv._compliance_nonlinear = _original_pcv

# print("PCV case 1:", r1["delivered_vt_ml"], r1["is_valid"], r1["invalid_reason"])
# print("PCV case 2:", r2["delivered_vt_ml"], r2["is_valid"], r2["invalid_reason"])