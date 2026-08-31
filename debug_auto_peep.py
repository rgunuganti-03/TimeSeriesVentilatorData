from generator.pcv_generator import generate_breath_cycles
from generator.conditions import get_condition

N_CYCLES_TO_CHECK = 20   # production default is 10 (per CR0008) — go well past it

for name, params in [("COPD", "Bronchospasm")]:
    print(f"\n=== {name} ===")
    generate_breath_cycles(params, n_cycles=N_CYCLES_TO_CHECK)