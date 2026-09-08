import numpy as np
from generator.simv_generator import generate_breath_cycles
from generator.conditions import get_condition_for_mode

N_CYCLES = 20
N_SEEDS = 30

def run_averaged(name):
    params = get_condition_for_mode(name, "simv")
    params["mandatory_mode"] = "PC"
    params["f_window"] = 0.25
    params["insp_pressure_cmH2O"] = params["tidal_volume_ml"] / params["compliance_ml_per_cmH2O"]

    values = np.array([
        generate_breath_cycles(params, n_cycles=N_CYCLES, seed=seed)["auto_peep_cmH2O"]
        for seed in range(N_SEEDS)
    ])
    print(f"{name} (SIMV): mean={values.mean():.4f}  std={values.std():.4f}  "
          f"min={values.min():.4f}  max={values.max():.4f}")
    return values

run_averaged("COPD")
run_averaged("Bronchospasm")