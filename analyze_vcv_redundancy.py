"""
analyze_vcv_redundancy.py
-------------------------
Diagnostic script — identifies redundant scenarios in the VCV dataset
by measuring similarity in the derived metrics space.
 
Two scenarios are considered functionally redundant if their derived
metrics vectors are very close after normalisation — meaning a downstream
model would learn the same thing from either one.
 
Run from the project root:
    python analyze_vcv_redundancy.py
 
Dependencies:
    pip install scikit-learn
 
Output:
    - Prints redundancy summary to terminal
    - Writes data/exports/vcv_hdf5/vcv_redundancy_report.csv
      One row per valid scenario with a redundancy flag and
      nearest-neighbour distance
"""
 
import os
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
 
MANIFEST_PATH  = Path("data/exports/vcv_hdf5/vcv_manifest.csv")
REPORT_PATH    = Path("data/exports/vcv_hdf5/vcv_redundancy_report.csv")
 
# Derived metrics used for similarity comparison
# These are the columns that define what a scenario "means" to a model
METRIC_COLS = [
    "ppeak_cmH2O",
    "pplat_cmH2O",
    "driving_p_cmH2O",
    "mean_paw_cmH2O",
    "auto_peep_cmH2O",
    "delivered_vt_mL",
    "minute_vent_L",
]
 
# Distance threshold below which two scenarios are considered redundant
# 0.5 in standardised space means the scenarios differ by less than
# half a standard deviation across all metrics combined
REDUNDANCY_THRESHOLD = 3
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def run():
    print("=" * 65)
    print("  VCV Dataset Redundancy Analysis")
    print("=" * 65)
 
    # --- Load manifest ----------------------------------------------------
    if not MANIFEST_PATH.exists():
        print(f"\n  ERROR: Manifest not found at {MANIFEST_PATH}")
        print("  Run generate_vcv_dataset_hdf5.py first.")
        sys.exit(1)
 
    df = pd.read_csv(MANIFEST_PATH)
    print(f"\n  Total scenarios in manifest : {len(df):,}")
 
    # Work only with valid scenarios — invalid ones have no metrics
    valid = df[df["is_valid"] == True].copy()
    print(f"  Valid scenarios             : {len(valid):,}")
    print(f"  Invalid scenarios           : {len(df) - len(valid):,}")
 
    # --- Build metric matrix ----------------------------------------------
    # Drop any rows with missing metric values (should not occur for valid)
    before = len(valid)
    valid  = valid.dropna(subset=METRIC_COLS)
    if len(valid) < before:
        print(f"  Dropped {before - len(valid)} rows with missing metrics")
 
    X_raw = valid[METRIC_COLS].values.astype(np.float64)
 
    # Standardise — each metric gets mean=0, std=1
    # This prevents high-magnitude metrics (like delivered_vt_mL in mL)
    # from dominating over low-magnitude ones (like auto_peep_cmH2O)
    scaler = StandardScaler()
    X      = scaler.fit_transform(X_raw)
 
    print(f"\n  Metric columns used ({len(METRIC_COLS)}):")
    for col in METRIC_COLS:
        print(f"    {col}")
 
    # --- Nearest neighbour search -----------------------------------------
    print(f"\n  Running nearest-neighbour search...")
    print(f"  Redundancy threshold : {REDUNDANCY_THRESHOLD} "
          f"(standardised distance)")
 
    # n_neighbors=2 because the first neighbour of any point is itself
    nbrs = NearestNeighbors(
        n_neighbors=2,
        algorithm="ball_tree",
        metric="euclidean",
    ).fit(X)
 
    distances, indices = nbrs.kneighbors(X)
 
    # Second column = nearest other scenario (first is self at distance 0)
    nn_distances = distances[:, 1]
    nn_indices   = indices[:, 1]
 
    # Flag scenarios whose nearest neighbour is within the threshold
    is_redundant = nn_distances < REDUNDANCY_THRESHOLD
 
    # --- Per-condition breakdown -------------------------------------------
    valid = valid.copy()
    valid["nn_distance"]  = nn_distances
    valid["nn_scenario_id"] = valid["scenario_id"].iloc[nn_indices].values
    valid["is_redundant"] = is_redundant
 
    print(f"\n  {'─' * 55}")
    print(f"  {'Condition':<20} {'Total':>8} {'Redundant':>10} {'Pct':>8}")
    print(f"  {'─' * 55}")
 
    for condition in df["condition"].unique():
        cond_mask  = valid["condition"] == condition
        cond_total = cond_mask.sum()
        cond_redun = (valid.loc[cond_mask, "is_redundant"]).sum()
        pct        = 100 * cond_redun / cond_total if cond_total > 0 else 0
        print(f"  {condition:<20} {cond_total:>8,} {cond_redun:>10,} "
              f"{pct:>7.1f}%")
 
    print(f"  {'─' * 55}")
    total_redun = is_redundant.sum()
    total_valid = len(valid)
    print(f"  {'TOTAL':<20} {total_valid:>8,} {total_redun:>10,} "
          f"{100*total_redun/total_valid:>7.1f}%")
 
    # --- Distance distribution --------------------------------------------
    print(f"\n  Nearest-neighbour distance distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(nn_distances, p)
        print(f"    p{p:>2}  :  {val:.4f}")
 
    print(f"    min  :  {nn_distances.min():.4f}")
    print(f"    max  :  {nn_distances.max():.4f}")
 
    # --- Recommended keep set ---------------------------------------------
    # Greedy deduplication: sort by nn_distance ascending (most redundant
    # first), mark as redundant if nearest neighbour is already kept
    # Simple approach: keep all non-redundant + sample redundant
    kept        = valid[~valid["is_redundant"]]
    redundant   = valid[valid["is_redundant"]]
 
    print(f"\n  Recommended keep set:")
    print(f"    Non-redundant (always keep) : {len(kept):,}")
    print(f"    Redundant (can thin)        : {len(redundant):,}")
    print(f"    Reduction if thinned 100%   : "
          f"{100*len(redundant)/total_valid:.1f}%")
    print(f"    Reduction if thinned 50%    : "
          f"{100*(len(redundant)*0.5)/total_valid:.1f}%")
 
    # --- Write report CSV -------------------------------------------------
    report_cols = [
        "scenario_id", "condition",
        "compliance_mL_per_cmH2O", "resistance_cmH2O_L_s",
        "respiratory_rate", "tidal_volume_mL", "ie_ratio",
        "peep_cmH2O", "flow_pattern",
    ] + METRIC_COLS + ["nn_distance", "nn_scenario_id", "is_redundant"]
 
    # Keep only columns that exist in valid
    report_cols = [c for c in report_cols if c in valid.columns]
    valid[report_cols].to_csv(REPORT_PATH, index=False)
 
    print(f"\n  Report written to : {REPORT_PATH}")
    print(f"  Columns           : scenario_id, condition, params, "
          f"metrics, nn_distance, is_redundant")
 
    print(f"\n{'=' * 65}")
    print(f"  Redundancy analysis complete.")
    print(f"{'=' * 65}")
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    run()
 