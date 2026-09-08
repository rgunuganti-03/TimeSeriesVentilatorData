import numpy as np
from generator.simv_generator import generate_breath_cycles
from generator.conditions import CONDITIONS

N_CYCLES = 20
N_SEEDS = 30

params = {k: v for k, v in CONDITIONS["Bronchospasm"].items() if k not in ("label", "description")}
params["mandatory_mode"] = "PC"
params["f_window"] = 0.25
params["insp_pressure_cmH2O"] = params["tidal_volume_ml"] / params["compliance_ml_per_cmH2O"]
params["effort_rate_per_min"] = 28

values = np.array([
    generate_breath_cycles(params, n_cycles=N_CYCLES, seed=seed)["auto_peep_cmH2O"]
    for seed in range(N_SEEDS)
])
print(f"SIMV Bronchospasm (effort_rate=28): mean={values.mean():.4f}  std={values.std():.4f}")