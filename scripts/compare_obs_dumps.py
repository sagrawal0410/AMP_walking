#!/usr/bin/env python3
"""
Compare C++ observation dumps with Python-generated observations.
Usage:
    python compare_obs_dumps.py --cpp-dump /tmp/amp_debug/obs_obs_step_000001.csv --python-dump obs_python.npy
"""

import numpy as np
import argparse
import csv
import os

# Slice boundaries for 585-dim obs
SLICE_BOUNDARIES = {
    "base_ang_vel": (0, 15),
    "root_local_rot_tan_norm": (15, 45),
    "keyboard_velocity_commands": (45, 60),
    "joint_pos": (60, 205),
    "joint_vel": (205, 350),
    "actions": (350, 495),
    "key_body_pos_b": (495, 585),
}

def load_csv_vector(filepath):
    """Load a CSV file containing a single row of floats."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        row = next(reader)
        return np.array([float(x) for x in row])

def print_slice_stats(vec, start, end, name):
    """Print statistics for a slice of the vector."""
    slice_vec = vec[start:end]
    print(f"\n{name} [{start}:{end}):")
    print(f"  Size: {len(slice_vec)}")
    print(f"  Min: {np.min(slice_vec):.6f}")
    print(f"  Max: {np.max(slice_vec):.6f}")
    print(f"  Mean: {np.mean(slice_vec):.6f}")
    print(f"  Std: {np.std(slice_vec):.6f}")
    print(f"  L2 norm: {np.linalg.norm(slice_vec):.6f}")
    print(f"  First 6: {slice_vec[:6]}")
    
    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(slice_vec))
    inf_count = np.sum(np.isinf(slice_vec))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values!")
    if inf_count > 0:
        print(f"  WARNING: {inf_count} Inf values!")

def compare_vectors(vec1, vec2, name="vectors", tolerance=1e-3):
    """Compare two vectors and print differences."""
    if vec1.shape != vec2.shape:
        print(f"ERROR: Shape mismatch: {vec1.shape} vs {vec2.shape}")
        return False
    
    diff = np.abs(vec1 - vec2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_diff_idx = np.argmax(diff)
    
    print(f"\n{name} comparison:")
    print(f"  Max absolute difference: {max_diff:.6f} at index {max_diff_idx}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Values at max diff: vec1[{max_diff_idx}]={vec1[max_diff_idx]:.6f}, vec2[{max_diff_idx}]={vec2[max_diff_idx]:.6f}")
    
    if max_diff > tolerance:
        print(f"  WARNING: Differences exceed tolerance {tolerance}!")
        return False
    else:
        print(f"  OK: Differences within tolerance {tolerance}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python observation dumps")
    parser.add_argument("--cpp-dump", type=str, required=True, help="Path to C++ CSV dump")
    parser.add_argument("--python-dump", type=str, help="Path to Python numpy dump (optional)")
    parser.add_argument("--step", type=int, default=1, help="Step number for analysis")
    parser.add_argument("--dump-dir", type=str, default="/tmp/amp_debug", help="Dump directory")
    
    args = parser.parse_args()
    
    # Load C++ dump
    if not os.path.exists(args.cpp_dump):
        print(f"Error: C++ dump file not found: {args.cpp_dump}")
        return
    
    print(f"Loading C++ dump: {args.cpp_dump}")
    cpp_obs = load_csv_vector(args.cpp_dump)
    print(f"C++ obs shape: {cpp_obs.shape}")
    
    # Verify size
    if len(cpp_obs) != 585:
        print(f"WARNING: C++ obs size is {len(cpp_obs)}, expected 585!")
    
    # Print slice stats for C++ obs
    print("\n" + "="*60)
    print("C++ OBSERVATION SLICE STATISTICS")
    print("="*60)
    for name, (start, end) in SLICE_BOUNDARIES.items():
        print_slice_stats(cpp_obs, start, end, name)
    
    # Compare with Python if provided
    if args.python_dump:
        if not os.path.exists(args.python_dump):
            print(f"Error: Python dump file not found: {args.python_dump}")
            return
        
        print("\n" + "="*60)
        print("LOADING PYTHON DUMP")
        print("="*60)
        python_obs = np.load(args.python_dump)
        if python_obs.ndim > 1:
            python_obs = python_obs.flatten()
        print(f"Python obs shape: {python_obs.shape}")
        
        if len(python_obs) != 585:
            print(f"WARNING: Python obs size is {len(python_obs)}, expected 585!")
        
        # Compare overall
        print("\n" + "="*60)
        print("OVERALL COMPARISON")
        print("="*60)
        compare_vectors(cpp_obs, python_obs, "Overall obs")
        
        # Compare each slice
        print("\n" + "="*60)
        print("SLICE-BY-SLICE COMPARISON")
        print("="*60)
        for name, (start, end) in SLICE_BOUNDARIES.items():
            cpp_slice = cpp_obs[start:end]
            python_slice = python_obs[start:end]
            compare_vectors(cpp_slice, python_slice, name, tolerance=1e-2)
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
If you see mismatches:
1. Check obs_order in deploy.yaml matches Python training order
2. Check history_length=5 for all terms
3. Check history flattening order (oldest->newest)
4. Check normalization/scaling matches training
5. Check joint_ids_map is correct
6. Check key_body_pos_b body names match
7. Check root_local_rot_tan_norm quaternion convention (wxyz vs xyzw)
8. Check control rate (policy Hz) matches training
    """)

if __name__ == "__main__":
    main()
