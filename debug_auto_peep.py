from generator.simv_generator import generate_breath_cycles
from generator.conditions import CONDITIONS

N_CYCLES_TO_CHECK = 20

for name in ["COPD", "Bronchospasm"]:
    raw = CONDITIONS[name]
    params = {k: v for k, v in raw.items() if k not in ("label", "description")}
    params["insp_pressure_cmH2O"] = params["tidal_volume_ml"] / params["compliance_ml_per_cmH2O"]
    params["mandatory_mode"] = "PC"
    params["f_window"] = 0.25
    print(f"\n=== {name} (SIMV) ===")
    generate_breath_cycles(params, n_cycles=N_CYCLES_TO_CHECK)