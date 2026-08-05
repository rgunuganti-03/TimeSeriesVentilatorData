"""
checks_gap.py
--------------
Diagnostic: print the gap between every consecutive pair of breaths in a
SIMV scenario, to check for the zero-gap re-trigger bug (should now show
strictly positive gaps everywhere after the patch).

Run from the project root:
    python checks_gap.py
"""
"""
check_rate_distortion.py
--------------------------
Post-fix verification: for each RR/effort-rate/f_window combination,
run the real generator across many seeds and measure the actual
mandatory-breath interval distribution, rather than predicting it.
"""

import numpy as np

from generator.conditions import get_condition
from generator.simv_generator import generate_breath_cycles


def check_rate_distortion(rr, eff, fw, n_seeds=100, n_cycles=15,
                           condition_name="Normal", **overrides):
    base = get_condition(condition_name)
    params = {
        **base,
        "mandatory_mode": "VC",
        "flow_pattern": "square",
        "respiratory_rate": rr,
        "effort_rate_per_min": eff,
        "f_window": fw,
    }
    params.update(overrides)

    T_mand_nominal = 60.0 / rr
    intervals = []
    for seed in range(n_seeds):
        result = generate_breath_cycles(params, n_cycles=n_cycles, seed=seed)
        mand_starts = [b["t_start_s"] for b in result["breath_records"]
                       if b["breath_type"] == "mandatory"]
        if len(mand_starts) >= 2:
            intervals.extend(np.diff(mand_starts))

    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    print(f"RR={rr} eff={eff} fw={fw}: nominal T_mand={T_mand_nominal:.2f}s, "
          f"observed mean={mean_interval:.2f}s (std={std_interval:.2f}s), "
          f"distortion={mean_interval - T_mand_nominal:+.2f}s")


if __name__ == "__main__":
    # The confirmed worst offenders from the deterministic pass
    for rr, eff, fw in [(4, 16, 0.25), (4, 16, 0.30),
                         (6, 16, 0.25), (6, 16, 0.30),
                         (12, 16, 0.25), (12, 16, 0.30)]:
        check_rate_distortion(rr, eff, fw)

# from generator.conditions import get_condition
# from generator.simv_generator import generate_breath_cycles
# from generator.simv_generator import PARAMETER_GRID


# def check_gaps(condition_name, seed=42, n_cycles=10, **overrides):
#     params = get_condition(condition_name)

#     # These aren't in conditions.py -- they're set by dashboard.py's SIMV
#     # controls, not the condition preset. Match the dashboard's defaults:
#     params["mandatory_mode"] = "VC"
#     params["flow_pattern"] = "decelerating"
#     params["f_window"] = 0.25
#     # dashboard.py halves the preset RR specifically for SIMV
#     params["respiratory_rate"] = max(4, round(params["respiratory_rate"] * 0.5))

#     params.update(overrides)  # let you override anything for a specific test

#     result = generate_breath_cycles(params, n_cycles=n_cycles, seed=seed)
#     records = result["breath_records"]

#     # print(f"\n{condition_name} — {len(records)} breaths, seed={seed}")
#     # for b in records:
#     #     end = b["t_start_s"] + b["duration_s"]
#     #     print(f"  t={b['t_start_s']:6.2f}s -> {end:6.2f}s  "
#     #           f"{b['breath_type']:>12}/{b['trigger_mode']}")

#     locked = 0
#     total = 0
#     for rr in PARAMETER_GRID["respiratory_rate"]:
#         for eff in PARAMETER_GRID["effort_rate_per_min"]:
#             for fw in PARAMETER_GRID["f_window"]:
#                 total += 1
#                 T_mand = 60.0 / rr
#                 window_open = T_mand * (1 - fw)
#                 interval = 60.0 / eff
#                 # first attempt time that clears window_open, as a fraction of T_mand
#                 k = -(-window_open // interval)  # ceil
#                 first_after = k * interval
#                 actual_period = first_after  # what the mandatory rate collapses to if this attempt always wins
#                 if abs(actual_period - T_mand) > 0.5:
#                     locked += 1
#                     print(f"RR={rr} eff={eff} fw={fw}: nominal T_mand={T_mand:.1f}s, "
#                         f"collapses toward {actual_period:.1f}s")
#     print(f"\n{locked}/{total} combinations show >0.5s rate distortion")
#     # min_gap = float("inf")
#     # for prev, nxt in zip(records, records[1:]):
#     #     gap = nxt["t_start_s"] - (prev["t_start_s"] + prev["duration_s"])
#     #     min_gap = min(min_gap, gap)
#     #     flag = "  <-- SUSPICIOUSLY SMALL" if gap < 0.05 else ""
#     #     print(f"  {prev['breath_type']:>12}/{prev['trigger_mode']:<14} -> "
#     #           f"{nxt['breath_type']:<12}/{nxt['trigger_mode']:<14} "
#     #           f"gap={gap:6.3f}s{flag}")
#     # print(f"  min gap: {min_gap:.3f}s")
#     # return min_gap


# if __name__ == "__main__":
#     check_gaps("COPD")
    