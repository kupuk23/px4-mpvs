#!/usr/bin/env python3
"""
Combine many *.pkl experiment result files and print summary stats.
Usage:
    python analyze_result.py /path/to/results_dir
"""
from pathlib import Path
import pickle
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

from px4_mpvs.utils.plot_utils import plot_features, plot_weights, plot_pose_error


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
        if k == "robot_att":
            continue
        arr = np.asarray(vals, dtype=float)
        rows[k] = {
            "count": arr.size,
            "mean": arr.mean(),
            "std": arr.std(ddof=1) if arr.size > 1 else 0.0,
            "min": arr.min(),
            "max": arr.max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def load_mode_results(mode_dir: Path):
    """Load all pickle files from a specific mode directory."""
    pickle_files = sorted(mode_dir.glob("*.pickle"))
    if not pickle_files:
        print(f"Warning: No .pickle files found in {mode_dir}")
        return [], []
    
    dicts = [load_pickle(p) for p in pickle_files]
    return dicts, pickle_files


def find_best_across_modes(results_dir: Path):
    """Find the best result across all modes based on hybrid_duration."""
    modes = ["softmax", "ratio", "discrete"]
    
    all_dicts = []
    all_files = []
    mode_info = []  # Store (mode_name, file_index_within_mode, original_file_path)
    
    for mode in modes:
        mode_dir = results_dir / mode
        if not mode_dir.exists():
            print(f"Warning: Mode directory {mode_dir} does not exist")
            continue
        
        mode_dicts, mode_files = load_mode_results(mode_dir)
        
        if mode_dicts:
            print(f"\nLoaded {len(mode_files)} files from {mode} mode:")
            for file_path in mode_files:
                print(f" • {file_path.name}")
                mode_info.append((mode, len(all_files), file_path))
                
            all_dicts.extend(mode_dicts)
            all_files.extend(mode_files)
    
    if not all_dicts:
        sys.exit(f"No .pickle files found in any mode directories under {results_dir}")
    
    # Find the best result across all modes
    hybrid_durations = np.array([d["hybrid_duration"] for d in all_dicts])
    min_duration_index = np.argmin(hybrid_durations)
    
    # Get mode information for the best result
    best_mode, _, best_file_path = mode_info[min_duration_index]
    
    return all_dicts, all_files, min_duration_index, best_mode, best_file_path


def analyze_mode_performance(all_dicts, mode_info):
    """Analyze performance by mode and return summary statistics."""
    mode_stats = {}
    
    for mode, file_idx, file_path in mode_info:
        if mode not in mode_stats:
            mode_stats[mode] = {
                'hybrid_durations': [],
                'full_durations': [],
                'final_errors': [],
                'file_paths': []
            }
        
        dict_data = all_dicts[file_idx]
        mode_stats[mode]['hybrid_durations'].append(dict_data['hybrid_duration'])
        mode_stats[mode]['full_durations'].append(dict_data['full_docking_duration'])
        mode_stats[mode]['final_errors'].append(dict_data['features_error'][-1])
        mode_stats[mode]['file_paths'].append(file_path)
    
    # Create summary DataFrame
    summary_rows = {}
    for mode, stats in mode_stats.items():
        hybrid_arr = np.array(stats['hybrid_durations'])
        full_arr = np.array(stats['full_durations'])
        error_arr = np.array(stats['final_errors'])
        
        summary_rows[mode] = {
            'count': len(hybrid_arr),
            'hybrid_mean': hybrid_arr.mean(),
            'hybrid_std': hybrid_arr.std() if len(hybrid_arr) > 1 else 0.0,
            'hybrid_min': hybrid_arr.min(),
            'full_mean': full_arr.mean(),
            'full_std': full_arr.std() if len(full_arr) > 1 else 0.0,
            'error_mean': error_arr.mean(),
            'error_std': error_arr.std() if len(error_arr) > 1 else 0.0,
        }
    
    return pd.DataFrame.from_dict(summary_rows, orient='index'), mode_stats


def main(results_dir):
    results_path = Path(results_dir)
    
    # Check if results_dir is a single file or directory
    if results_path.is_file():
        if results_path.suffix != ".pickle":
            sys.exit(f"Expected a directory or a .pickle file, got {results_dir}")
        
        # Load single pickle file
        dicts = [load_pickle(results_path)]
        min_duration_index = 0
        best_mode = "single_file"
        best_file_path = results_path
        best_dicts_per_mode = {"single_file": dicts[0]}
        
        print(f"Loaded single file: {results_path.name}")
        
    elif results_path.is_dir():
        # Check if this is a mode directory (contains .pickle files) or parent directory (contains mode subdirs)
        pickle_files = list(results_path.glob("*.pickle"))
        mode_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name in ["softmax", "ratio", "discrete"]]
        
        if pickle_files and not mode_dirs:
            # This is a single mode directory
            dicts = [load_pickle(p) for p in sorted(pickle_files)]
            hybrid_durations = np.array([d["hybrid_duration"] for d in dicts])
            min_duration_index = np.argmin(hybrid_durations)
            best_mode = results_path.name
            best_file_path = sorted(pickle_files)[min_duration_index]
            best_dicts_per_mode = {best_mode: dicts[min_duration_index]}
            
            print(f"Loaded {len(pickle_files)} files from {best_mode} mode")
            
        elif mode_dirs:
            # This is a parent directory with mode subdirectories
            dicts, all_files, min_duration_index, best_mode, best_file_path = find_best_across_modes(results_path)
            
            # Create mode info for analysis
            mode_info = []
            file_idx = 0
            for mode in ["softmax", "ratio", "discrete"]:
                mode_dir = results_path / mode
                if mode_dir.exists():
                    mode_files = sorted(mode_dir.glob("*.pickle"))
                    for file_path in mode_files:
                        mode_info.append((mode, file_idx, file_path))
                        file_idx += 1
            
            # Analyze performance by mode
            mode_summary_df, mode_stats = analyze_mode_performance(dicts, mode_info)
            print(f"\n{'='*60}")
            print("PERFORMANCE BY MODE:")
            print(f"{'='*60}")
            print(mode_summary_df)
            
            # Get best result for each mode
            best_dicts_per_mode = {}
            print(f"\n{'='*60}")
            print("BEST RESULT FOR EACH MODE:")
            print(f"{'='*60}")
            for mode, stats in mode_stats.items():
                best_idx = np.argmin(stats['hybrid_durations'])
                print(f"{mode.upper()}:")
                print(f"  Best hybrid duration: {stats['hybrid_durations'][best_idx]:.2f}s")
                print(f"  File: {stats['file_paths'][best_idx].name}")
                print(f"  Final error: {stats['final_errors'][best_idx]:.4f}")
                
                # Find the corresponding dict in all_dicts
                for i, (m, _, _) in enumerate(mode_info):
                    if m == mode and stats['file_paths'][best_idx] == mode_info[i][2]:
                        best_dicts_per_mode[mode] = dicts[i]
                        break
        else:
            sys.exit(f"No .pickle files or mode directories found in {results_dir}")
    else:
        sys.exit(f"Path {results_dir} does not exist")

    # Plot results for the overall best result
    print(f"\n{'='*60}")
    print("OVERALL BEST RESULT:")
    print(f"{'='*60}")
    print(f"Mode: {best_mode}")
    print(f"File: {best_file_path.name}")
    print(f"Hybrid duration: {dicts[min_duration_index]['hybrid_duration']:.2f}s")
    print(f"Full docking duration: {dicts[min_duration_index]['full_docking_duration']:.2f}s")
    
    # Generate plots for the best result
    best_dict = dicts[min_duration_index]
    plot_features(best_dict)
    plot_weights(best_dict["recorded_wp"], best_dict["recorded_ws"], best_dict["full_docking_duration"])
    plot_pose_error(best_dicts_per_mode)  # Pass all best results from each mode
    
    # Plot Lyapunov derivatives
    plot_weights(
        best_dict["Vp_dot"],
        best_dict["Vs_dot"],
        best_dict["full_docking_duration"],
        lyapunov=True,
    )
    
    plt.show()
    
    # Print detailed results for overall best
    last_features_error = best_dict['features_error'][-1]
    print(f"Last features error (SSE): {last_features_error:.6f}")
    
    final_pos = best_dict['robot_pose'][-1]
    final_att = best_dict['robot_att'][-1]
    print(f"Final position: {final_pos}")
    print(f"Final attitude: {final_att}")
    print(f"Desired position: {best_dict['desired_pos']}")
    print(f"Desired attitude: {best_dict['desired_att']}")


if __name__ == "__main__":
    # Default results directory - parent directory containing mode subdirectories
    results_dir = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/px4_mpvs/hw_exp"
    
    # You can also pass a specific mode directory or single file
    # results_dir = "/home/tafarrel/discower_ws/src/px4_mpvs/px4_mpvs/px4_mpvs/hw_exp/softmax"
    # results_dir = "/path/to/single/result.pickle"
    
    main(results_dir)