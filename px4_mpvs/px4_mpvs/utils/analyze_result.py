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
import matplotlib.pyplot as plt

from px4_mpvs.utils.plot_utils import plot_features, plot_weights



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
    # check if results_dir is path or file, if a file, immediately load it
    if not Path(results_dir).is_dir():
        file_analysis = True
        if Path(results_dir).suffix != ".pickle":
            sys.exit(f"Expected a directory or a .pickle file, got {results_dir}")
        # Load single pickle file
        dicts = [load_pickle(Path(results_dir))]
    else:
        file_analysis = False
        paths = sorted(Path(results_dir).glob("*.pickle"))
        if not paths:
            sys.exit(f"No .pkl files found in {results_dir}")

        dicts = [load_pickle(p) for p in paths]
        print(f"\nLoaded {len(paths)} files:")
        for p in paths:
            print(" •", p.name)



    merged = flatten_dicts(dicts)
    
    summary_df = summarize(merged)

    # choose statistic with shortest "hybrid_duration"
    hybrid_durations = np.array(merged["hybrid_duration"])
    min_duration_index = np.argmin(hybrid_durations)
    plot_features(dicts[min_duration_index]["recorded_features"], dicts[min_duration_index]["desired_points"])
    plot_weights(dicts[min_duration_index]["recorded_wp"], dicts[min_duration_index]["recorded_ws"], dicts[min_duration_index]["full_docking_duration"])

    # plot lyapunovs
    plot_weights(
        dicts[min_duration_index]["Vp_dot"],
        dicts[min_duration_index]["Vs_dot"],
        dicts[min_duration_index]["full_docking_duration"],
        lyapunov=True,
    )

    plt.show()

    
    print("\nSummary statistics:")
    print(summary_df)
    
    #print full docking duration, hybrid duration
    #show the name of the file with the shortest hybrid duration
    if not file_analysis:
        print("\nThe best result is from file:", paths[min_duration_index].name)
    print(f"Full docking duration : {dicts[min_duration_index]['full_docking_duration']}")
    print(f"Hybrid duration : {dicts[min_duration_index]['hybrid_duration']}")

    #Show SSE using the last features_error
    last_features_error = dicts[min_duration_index]['features_error'][-1]
    print(f"Last features error (SSE) : {last_features_error}")

    # print final pose
    final_pos = dicts[min_duration_index]['robot_pose'][-1]
    final_att = dicts[min_duration_index]['robot_att'][-1]
    print(f"Final position: {final_pos}")
    print(f"Final attitude: {final_att}")

    print("desired pose : ", dicts[min_duration_index]['desired_pos'])
    print("desired attitude : ", dicts[min_duration_index]['desired_att'])

if __name__ == "__main__":
    results_dir = (
        "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/simulation_data/softmax/"

    )
    main(results_dir)
