#!/usr/bin/env python3
"""
Observation Shift Analysis Script

This script analyzes the actual perturbation values added to observations
across different levels in continual learning experiments. It plots the
shift values for each observation dimension across levels, with confidence
intervals calculated across seeds.

Usage:
    python observation_shift_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style to match Figure 6 plots
plt.style.use('default')
sns.set_palette("husl")

# Coolors palette for consistent colors
coolors_palette = ['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93']

def load_shift_data(data_folder: str) -> Dict:
    """
    Load shift data from all HYB_* experiments in the specified folder.
    
    Args:
        data_folder: Path to the folder containing experiments (e.g., "cartpole/N0002/S256")
    
    Returns:
        Dictionary with structure: {seed: {level: [obs_dim_0, obs_dim_1, ...]}}
    """
    print(f"🔄 Loading shift data from {data_folder} folder...")
    
    shift_data = {}
    base_path = Path(data_folder)
    
    if not base_path.exists():
        print(f"❌ Folder {data_folder} not found!")
        return {}
    
    # Find all HYB_* experiment directories
    hyb_experiments = [run_dir for run_dir in base_path.iterdir() 
                      if run_dir.is_dir() and run_dir.name.startswith('HYB_')]
    
    print(f"   📁 Found {len(hyb_experiments)} HYB experiments")
    
    for exp_dir in hyb_experiments:
        try:
            # Extract seed from directory name
            seed = None
            for part in exp_dir.name.split('_'):
                if part.startswith('seed'):
                    seed = int(part[4:])  # Remove 'seed' prefix
                    break
            
            if seed is None:
                print(f"⚠️ Could not extract seed from {exp_dir.name}")
                continue
            
            # Load shift data
            shift_file = exp_dir / "data" / "shift_data.csv"
            if shift_file.exists():
                shift_df = pd.read_csv(shift_file)
                
                # Parse the offset_repr column (string representation of list)
                shift_values = []
                for _, row in shift_df.iterrows():
                    try:
                        # Parse the string representation of the list
                        offset_list = ast.literal_eval(row['offset_repr'])
                        shift_values.append({
                            'level': row['shift_id'],
                            'values': offset_list
                        })
                    except (ValueError, SyntaxError) as e:
                        print(f"⚠️ Error parsing shift data for seed {seed}, level {row['shift_id']}: {e}")
                        continue
                
                if shift_values:
                    shift_data[seed] = shift_values
                    print(f"   ✅ Loaded {len(shift_values)} levels for seed {seed}")
                else:
                    print(f"   ⚠️ No valid shift data for seed {seed}")
            else:
                print(f"   ⚠️ No shift_data.csv found in {exp_dir}")
                
        except Exception as e:
            print(f"⚠️ Error loading {exp_dir}: {e}")
            continue
    
    print(f"✅ Loaded shift data for {len(shift_data)} seeds")
    return shift_data

def calculate_bootstrap_confidence_intervals(data: np.ndarray, n_bootstrap: int = 1000, 
                                           confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate bootstrap confidence intervals for the given data.
    
    Args:
        data: Array of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound) arrays
    """
    if len(data) == 0:
        return np.array([]), np.array([])
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return lower_bound, upper_bound

def prepare_shift_plot_data(shift_data: Dict) -> Dict:
    """
    Prepare data for plotting by organizing shifts by level and observation dimension.
    
    Args:
        shift_data: Dictionary from load_shift_data()
    
    Returns:
        Dictionary with structure: {obs_dim: {level: [values_across_seeds]}}
    """
    print("🔄 Preparing shift plot data...")
    
    # First, determine the number of observation dimensions
    obs_dims = None
    for seed_data in shift_data.values():
        if seed_data:
            obs_dims = len(seed_data[0]['values'])
            break
    
    if obs_dims is None:
        print("❌ Could not determine observation dimensions")
        return {}
    
    print(f"   📊 Found {obs_dims} observation dimensions")
    
    # Organize data by observation dimension and level
    plot_data = {i: {} for i in range(obs_dims)}
    
    for seed, seed_data in shift_data.items():
        for level_data in seed_data:
            level = level_data['level']
            values = level_data['values']
            
            for obs_dim in range(obs_dims):
                if level not in plot_data[obs_dim]:
                    plot_data[obs_dim][level] = []
                plot_data[obs_dim][level].append(values[obs_dim])
    
    print(f"✅ Prepared data for {obs_dims} observation dimensions")
    return plot_data

def plot_observation_shifts_across_levels(shift_data: Dict, data_folder: str):
    """
    Plot observation shifts across levels with confidence intervals.
    
    Args:
        shift_data: Dictionary from load_shift_data()
        data_folder: Name of the data folder for title/filename
    """
    print(f"🔄 Creating observation shift plot for {data_folder}...")
    
    # Prepare data for plotting
    plot_data = prepare_shift_plot_data(shift_data)
    
    if not plot_data:
        print("❌ No data to plot")
        return
    
    # Set the color palette for matplotlib (same as Figure 6)
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Create figure with same dimensions as Figure 6 plots
    plt.figure(figsize=(3.15*1.5, 1.97))
    
    # Plot each observation dimension
    obs_dims = len(plot_data)
    colors = coolors_palette[:obs_dims]
    
    for obs_dim in range(obs_dims):
        levels = sorted(plot_data[obs_dim].keys())
        means = []
        lower_bounds = []
        upper_bounds = []
        
        for level in levels:
            values = np.array(plot_data[obs_dim][level])
            mean_val = np.mean(values)
            lower, upper = calculate_bootstrap_confidence_intervals(values)
            
            means.append(mean_val)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        # Convert to numpy arrays
        levels = np.array(levels)
        means = np.array(means)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        
        # Plot mean line (same linewidth as Figure 6)
        plt.plot(levels, means, color=colors[obs_dim], linewidth=2, 
                label=f'obs[{obs_dim}]')
        
        # Plot confidence interval (same alpha as Figure 6)
        plt.fill_between(levels, lower_bounds, upper_bounds, alpha=0.3, 
                        color=colors[obs_dim])
    
    # Add grey dotted vertical lines at level transitions (same as Figure 6)
    if levels.size > 0:
        max_level = int(levels.max())
        for level in range(1, max_level + 1):
            plt.axvline(x=level, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot (same styling as Figure 6)
    plt.xlabel('Level', fontsize=9, color='black')
    plt.ylabel('Observation Shift Value', fontsize=9, color='black')
    plt.title(f'Observation Shifts - {data_folder.split("/")[0].title()}', 
              fontsize=10, color='black', fontweight='bold')
    plt.grid(True, alpha=0.3, color=coolors_palette[2])
    plt.gca().set_facecolor('white')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    safe_folder = data_folder.replace('/', '_')
    plt.savefig(f'observation_shifts_{safe_folder}.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(f'observation_shifts_{safe_folder}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✅ Observation shift plot saved for {data_folder}!")

def main():
    """Main function to run the observation shift analysis."""
    print("🔬 Starting Observation Shift Analysis...")
    
    # Load data from cartpole/N0002/S256
    data_folder = "cartpole/N0002/S256"
    shift_data = load_shift_data(data_folder)
    
    if not shift_data:
        print("❌ No shift data found!")
        return
    
    # Create the plot
    plot_observation_shifts_across_levels(shift_data, data_folder)
    
    print("\n🎉 Observation shift analysis completed!")
    print("📁 Files saved:")
    print("   - observation_shifts_{folder}.png/pdf")

if __name__ == "__main__":
    main()
