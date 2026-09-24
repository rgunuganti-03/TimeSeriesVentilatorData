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





# import generator.prvc_generator as prvc

# r = prvc.generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=1)
# print("converged:", r["converged"], "delivered_vt_ml (final):", r["delivered_vt_trajectory"][-1])
# print("pressure_trajectory:", r["pressure_trajectory"])

# # Estimate V_target_per_comp's scale at convergence for comparison:
# comps = prvc._build_compartments("Normal", 80.0, 10.0, 5.0, 5.0, 0.0, prvc.DEFAULT_CHEST_WALL_COMPLIANCE, "adult")
# offset = 50.0 * (prvc.IBW_KG / prvc.IBW_KG)  # adult: weight_kg == IBW_KG, offset == 50.0
# print("V_target_per_comp scale (offset, adult):", offset, " vs. converged delivered VT:", r["delivered_vt_trajectory"][-1])

# r = prvc.generate_breath_cycles(NORMAL_PARAMS, n_cycles=10, seed=1)
# print("converged:", r["converged"], "delivered_vt_ml (final):", r["delivered_vt_trajectory"][-1])
# print("V_target_per_comp scale (new, adult):", 420.0 * 1.0, " vs. converged delivered VT:", r["delivered_vt_trajectory"][-1])

# print(NORMAL_PARAMS.get("stress_index", "not set — check default"))

# r_stress = prvc.generate_breath_cycles({**NORMAL_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)
# print("converged:", r_stress["converged"], "delivered_vt_ml (final):", r_stress["delivered_vt_trajectory"][-1])
# print("pressure_trajectory:", r_stress["pressure_trajectory"])

import generator.prvc_generator as prvc

# _original = prvc._compliance_two_regime
# def _instrumented(V_mL, C_base, V_ref, stress_index):
#     V_turnover = 1.4 * max(V_ref, 1.0)
#     if V_mL > V_turnover:
#         print(f"REGIME 2 HIT: V={V_mL:.2f} V_ref={V_ref:.2f} V/V_ref={V_mL/V_ref:.2f}")
#     return _original(V_mL, C_base, V_ref, stress_index)
# prvc._compliance_two_regime = _instrumented

# r_ards_stress = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)

# prvc._compliance_two_regime = _original  # restore immediately after use

# print("converged:", r_ards_stress["converged"])
# print("delivered_vt_ml (final):", r_ards_stress["delivered_vt_trajectory"][-1])
# print("pressure_trajectory:", r_ards_stress["pressure_trajectory"])
r_unbounded = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 1.0}, n_cycles=10, seed=1)
print("flat converged:", r_unbounded["converged"])
print("flat trajectory:", r_unbounded["delivered_vt_trajectory"])

prvc._compliance_nonlinear = lambda V_mL, C_base, V_ref, stress_index=1.0: (
    C_base if abs(stress_index - 1.0) < 0.01 or V_mL <= 0.0
    else C_base * (max(V_mL / max(V_ref, 1.0), 0.01) ** (1.0 - stress_index))
)
r_original = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)
print("original-unbounded converged:", r_original["converged"])
print("original-unbounded trajectory:", r_original["delivered_vt_trajectory"])

# restore the real dispatch before continuing anything else:
import importlib
importlib.reload(prvc)

r_two_regime = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)
print("two-regime converged:", r_two_regime["converged"])
print("two-regime trajectory:", r_two_regime["delivered_vt_trajectory"])
# _original = prvc._compliance_two_regime
# def _instrumented(V_mL, C_base, V_ref, stress_index):
#     V_turnover = 1.4 * max(V_ref, 1.0)
#     if V_mL > V_turnover:
#         print(f"REGIME 2 HIT: V={V_mL:.2f} V_ref={V_ref:.2f} V/V_ref={V_mL/V_ref:.2f}")
#     return _original(V_mL, C_base, V_ref, stress_index)
# prvc._compliance_two_regime = _instrumented

# r_stress = prvc.generate_breath_cycles({**NORMAL_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)

# prvc._compliance_two_regime = _original
# print("delivered_vt_ml (final):", r_stress["delivered_vt_trajectory"][-1])

# r_ards_stress = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)

r_flat = prvc.generate_breath_cycles({**SEVERE_ARDS_PARAMS, "stress_index": 1.0}, n_cycles=10, seed=1)
print("flat (SI=1.0) delivered_vt_ml (final):", r_flat["delivered_vt_trajectory"][-1])

# import generator.prvc_generator as prvc
# r_stress = prvc.generate_breath_cycles({**NORMAL_PARAMS, "stress_index": 0.85}, n_cycles=10, seed=1)
# print("converged:", r_stress["converged"], "delivered_vt_ml (final):", r_stress["delivered_vt_trajectory"][-1])
# print("pressure_trajectory:", r_stress["pressure_trajectory"])

# r_rds = prvc.generate_breath_cycles(RDS_PARAMS, n_cycles=10, seed=42)
# print("RDS:", r_rds["converged"], r_rds["delivered_vt_trajectory"][-1], r_rds.get("ppeak_cmH2O"))
# restore afterward:
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