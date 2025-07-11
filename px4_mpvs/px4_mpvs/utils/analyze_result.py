#!/usr/bin/env python3
"""
Combine many *.pkl experiment result files and print summary stats.
Usage:
    python combine_results.py /path/to/results_dir
"""
from pathlib import Path
import pickle
import numpy as np
import sys
import pandas as pd

from px4_mpvs.utils.plot_utils import plot_features, plot_weights

desired_points = np.array([[82, 123], [563, 123], [176, 337], [505, 218]]).flatten()


def load_pickle(path: Path):
    """Return the dictionary stored in one pickle file."""
    with path.open("rb") as f:
        dict = pickle.load(f)
        return dict


def flatten_dicts(dicts):
    """
    Merge a list of result-dicts into one dict whose values
    are lists of all observations.
    """
    merged = {}
    for d in dicts:
        for k, v in d.items():
            # Coerce scalars to 1-element list for uniformity
            v_arr = np.atleast_1d(v)
            merged.setdefault(k, []).extend(v_arr)
    return merged


def summarize(merged):
    """Return a DataFrame with count, mean, std, min, max for each metric."""
    rows = {}
    for k, vals in merged.items():
        arr = np.asarray(vals, dtype=float)
        rows[k] = {
            "count": arr.size,
            "mean": arr.mean(),
            "std": arr.std(ddof=1) if arr.size > 1 else 0.0,
            "min": arr.min(),
            "max": arr.max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def main(results_dir):
    paths = sorted(Path(results_dir).glob("*.pickle"))
    if not paths:
        sys.exit(f"No .pkl files found in {results_dir}")

    dicts = [load_pickle(p) for p in paths]



    merged = flatten_dicts(dicts)
    
    summary_df = summarize(merged)

    # choose statistic with shortest "hybrid_duration"
    hybrid_durations = np.array(merged["hybrid_duration"])
    min_duration_index = np.argmin(hybrid_durations)
    plot_features(dicts[min_duration_index]["recorded_features"], desired_points)
    plot_weights(dicts[min_duration_index]["recorded_wp"], dicts[min_duration_index]["recorded_ws"], dicts[min_duration_index]["full_docking_duration"])

    # plot lyapunovs
    plot_weights(
        dicts[min_duration_index]["Vp_dot"],
        dicts[min_duration_index]["Vs_dot"],
        dicts[min_duration_index]["full_docking_duration"],
        lyapunov=True,
    )

    print(f"\nLoaded {len(paths)} files:")
    for p in paths:
        print(" •", p.name)
    print("\nSummary statistics:")
    print(summary_df)


if __name__ == "__main__":
    results_dir = (
        "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/simulation_data/ratio"

    )
    main(results_dir)
