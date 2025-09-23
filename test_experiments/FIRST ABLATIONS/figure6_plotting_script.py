#!/usr/bin/env python3
"""
FIGURE 6 PLOTTING SCRIPT - MULTI-TASK WITH FOLDER SELECTION
Adapted from example plotting style with bootstrap confidence intervals
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List
from pathlib import Path

# Import the Coolors palette (same as example)
coolors_palette = [
    '#ff595e',  # Vibrant red
    '#ffca3a',  # Bright yellow  
    '#8ac926',  # Fresh green
    '#1982c4',  # Deep blue
    '#6a4c93'   # Rich purple
]

# Define consistent color mapping for each topology type
topology_colors = {
    'standard_mlp': '#ff595e',    # Vibrant red
    'hybrid': '#ffca3a',          # Bright yellow
    'modular': '#8ac926',         # Fresh green
    'small_world': '#1982c4'      # Deep blue
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_topology_data(data_folder: str) -> Dict:
    """Load topology data from specified folder (recursively searches through noise_level/network_size structure)."""
    print(f"🔄 Loading data from {data_folder} folder...")
    
    topology_groups = {}
    base_path = Path(data_folder)  # Data folders are in current directory
    
    if not base_path.exists():
        print(f"❌ Folder {data_folder} not found!")
        return {}
    
    # Check if this is a specific subfolder (contains noise level and/or network size)
    # If so, search only in this specific folder, not recursively
    if any(part.startswith(('N00', 'S')) for part in base_path.parts):
        # This is a specific subfolder, search for experiment directories inside it
        search_paths = [run_dir for run_dir in base_path.iterdir() if run_dir.is_dir() and run_dir.name.startswith(('HYB_', 'MOD_', 'SW_', 'STANDARD_MLP_'))]
        print(f"   📁 Searching in specific subfolder: {base_path}")
    else:
        # This is a task folder, search recursively through all subfolders
        search_paths = [run_dir for run_dir in base_path.rglob("*") if run_dir.is_dir() and run_dir.name.startswith(('HYB_', 'MOD_', 'SW_', 'STANDARD_MLP_'))]
        print(f"   📁 Searching recursively in task folder: {base_path}")
    
    # Find all experiment directories
    for search_path in search_paths:
            try:
                # Parse run name to extract topology
                run_name = search_path.name
                if run_name.startswith('HYB_'):
                    topology = 'hybrid'
                elif run_name.startswith('MOD_'):
                    topology = 'modular'
                elif run_name.startswith('SW_'):
                    topology = 'small_world'
                elif run_name.startswith('STANDARD_MLP_'):
                    topology = 'standard_mlp'
                else:
                    continue
                
                # Load episode data
                episode_file = search_path / "data" / "episode_data.csv"
                if episode_file.exists():
                    episode_data = pd.read_csv(episode_file)
                    
                    if 'episode_return_raw' in episode_data.columns:
                        rewards = episode_data['episode_return_raw']
                        
                        # Group episodes by iteration (every 2 episodes) and calculate mean per iteration
                        iteration_rewards = []
                        for i in range(0, len(rewards), 2):
                            if i + 1 < len(rewards):
                                iteration_mean = rewards.iloc[i:i+2].mean()
                                iteration_rewards.append(iteration_mean)
                            elif i < len(rewards):
                                iteration_rewards.append(rewards.iloc[i])
                        
                        if topology not in topology_groups:
                            topology_groups[topology] = []
                        topology_groups[topology].append(np.array(iteration_rewards))
                        
            except Exception as e:
                print(f"⚠️ Error loading {search_path}: {e}")
                continue
    
    print(f"✅ Loaded data for {len(topology_groups)} topology types")
    for topology, data_list in topology_groups.items():
        print(f"   {topology}: {len(data_list)} runs")
    
    return topology_groups

def calculate_bootstrap_confidence_intervals(data_arrays: List[np.ndarray], confidence: float = 0.95, n_bootstrap: int = 1000) -> Dict:
    """Calculate bootstrap confidence intervals (distribution-free)."""
    if not data_arrays:
        return {}
    
    # Stack arrays
    stacked = np.vstack(data_arrays)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0, ddof=1)
    var = np.var(stacked, axis=0, ddof=1)
    
    # Bootstrap confidence intervals
    n_seeds = len(data_arrays)
    n_iterations = stacked.shape[1]
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_seeds, size=n_seeds, replace=True)
        bootstrap_data = stacked[bootstrap_indices]
        bootstrap_mean = np.mean(bootstrap_data, axis=0)
        bootstrap_means.append(bootstrap_mean)
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate percentiles for confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile, axis=0)
    upper_bound = np.percentile(bootstrap_means, upper_percentile, axis=0)
    
    return {
        'mean': mean,
        'std': std,
        'var': var,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_seeds': len(data_arrays),
        'cv': std / np.abs(mean + 1e-8)
    }

def apply_window_smoothing(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply window smoothing to data."""
    if len(data) < window:
        return data
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain length
    padding = np.full(window-1, smoothed[0])
    return np.concatenate([padding, smoothed])

def prepare_bootstrap_figure6_data(topology_groups: Dict) -> Dict:
    """Prepare data for Figure6 plots with bootstrap confidence intervals."""
    figure6_data = {}
    
    for topology_key, group in topology_groups.items():
        if not group:
            continue
            
        # Filter out empty runs first
        valid_runs = [run_data for run_data in group if len(run_data) > 0]
        if not valid_runs:
            continue
            
        # Find maximum iterations across all valid runs
        max_iterations = max(len(run_data) for run_data in valid_runs)
        
        # Pad shorter runs with their last value
        padded_rewards = []
        for run_data in valid_runs:
            if len(run_data) < max_iterations:
                # Pad with last value
                padded = np.pad(run_data, (0, max_iterations - len(run_data)), mode='edge')
            else:
                padded = run_data[:max_iterations]
            
            # Apply smoothing
            smoothed = apply_window_smoothing(padded, window=5)
            padded_rewards.append(smoothed)
        
        # Calculate bootstrap confidence intervals
        stats = calculate_bootstrap_confidence_intervals(padded_rewards)
        
        figure6_data[topology_key] = {
            'iterations': np.arange(1, max_iterations + 1),
            'mean_rewards': stats['mean'],
            'std_rewards': stats['std'],
            'lower_bound': stats['lower_bound'],
            'upper_bound': stats['upper_bound'],
            'cv': stats['cv'],
            'topology_type': topology_key.replace('_', ' ').title(),
            'num_seeds': stats['n_seeds']
        }
    
    return figure6_data

def create_and_save_legend(figure6_data: Dict, task_name: str):
    """Create and save legend as separate file (same as example)."""
    fig_legend = plt.figure(figsize=(4, 1))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    
    # Create legend with the same styling
    legend_elements = []
    for topology_key, data in figure6_data.items():
        # Use consistent color mapping for each topology type
        color = topology_colors.get(topology_key, '#6a4c93')  # Default purple if not found
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                        label=f"{data['topology_type']}"))
    
    legend_fig = ax_legend.legend(handles=legend_elements, frameon=True, 
                                 facecolor='white', edgecolor=coolors_palette[4],
                                 loc='center', fontsize=9, ncol=4)
    
    # Save legend figure
    fig_legend.savefig(f'{task_name}_figure6_legend.png', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    fig_legend.savefig(f'{task_name}_figure6_legend.pdf', dpi=300, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
    plt.close(fig_legend)

# ============================================================================
# CELL 1: CARTPOLE ANALYSIS
# ============================================================================

def plot_cartpole_figure6(data_folder="cartpole"):
    """Plot Figure 6 for Cartpole task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(topology_groups)
    
    # Create figure with same dimensions as example
    plt.figure(figsize=(3.15*1.5, 1.97))
    
    # Plot each topology with different colors from the palette
    for topology_key, data in figure6_data.items():
        iterations = data['iterations']
        mean_rewards = data['mean_rewards']
        lower_bound = data['lower_bound']
        upper_bound = data['upper_bound']
        
        # Use consistent color mapping for each topology type
        color = topology_colors.get(topology_key, '#6a4c93')  # Default purple if not found
        
        # Plot the mean line
        plt.plot(iterations, mean_rewards, color=color, linewidth=2, 
                label=f"{data['topology_type']}")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # Add vertical dashed lines at steps 0, 200, 400, 600, etc. (same as example)
    if len(figure6_data) > 0:
        max_iterations = max(data['iterations'][-1] for data in figure6_data.values())
        vertical_steps = list(range(0, int(max_iterations) + 1, 200))
        for step in vertical_steps:
            plt.axvline(x=step, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Mean Episode reward', fontsize=9, color='black')
    plt.title(f'Cartpole', fontsize=10, color='black', fontweight='bold')
    plt.grid(True, alpha=0.3, color=coolors_palette[2])
    plt.gca().set_facecolor('white')
    
    # Create and save legend
    create_and_save_legend(figure6_data, 'cartpole')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot (sanitize filename)
    safe_folder = data_folder.replace('/', '_')
    plt.savefig(f'cartpole_{safe_folder}_figure6.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(f'cartpole_{safe_folder}_figure6.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✅ Cartpole plot saved from {data_folder} folder!")

# Run Cartpole analysis
print("🔄 Running Cartpole Figure 6 analysis...")
plot_cartpole_figure6(data_folder="cartpole/N0002/S128")  # Specific subfolder

# ============================================================================
# CELL 2: ACROBOT ANALYSIS  
# ============================================================================

def plot_acrobot_figure6(data_folder="acrobot"):
    """Plot Figure 6 for Acrobot task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(topology_groups)
    
    # Create figure with same dimensions as example
    plt.figure(figsize=(3.15*1.5, 1.97))
    
    # Plot each topology with different colors from the palette
    for topology_key, data in figure6_data.items():
        iterations = data['iterations']
        mean_rewards = data['mean_rewards']
        lower_bound = data['lower_bound']
        upper_bound = data['upper_bound']
        
        # Use consistent color mapping for each topology type
        color = topology_colors.get(topology_key, '#6a4c93')  # Default purple if not found
        
        # Plot the mean line
        plt.plot(iterations, mean_rewards, color=color, linewidth=2, 
                label=f"{data['topology_type']}")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # Add vertical dashed lines at steps 0, 200, 400, 600, etc. (same as example)
    if len(figure6_data) > 0:
        max_iterations = max(data['iterations'][-1] for data in figure6_data.values())
        vertical_steps = list(range(0, int(max_iterations) + 1, 200))
        for step in vertical_steps:
            plt.axvline(x=step, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Episode reward', fontsize=9, color='black')
    plt.title(f'Acrobot', fontsize=10, color='black', fontweight='bold')
    plt.grid(True, alpha=0.3, color=coolors_palette[2])
    plt.gca().set_facecolor('white')
    
    # Create and save legend
    create_and_save_legend(figure6_data, 'acrobot')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot (sanitize filename)
    safe_folder = data_folder.replace('/', '_')
    plt.savefig(f'acrobot_{safe_folder}_figure6.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(f'acrobot_{safe_folder}_figure6.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✅ Acrobot plot saved from {data_folder} folder!")

# Run Acrobot analysis
print("🔄 Running Acrobot Figure 6 analysis...")
plot_acrobot_figure6(data_folder="acrobot/N0001/S128")  # Specific subfolder

# ============================================================================
# CELL 3: LUNARLANDER ANALYSIS
# ============================================================================

def plot_lunarlander_figure6(data_folder="lunarlander"):
    """Plot Figure 6 for Lunarlander task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(topology_groups)
    
    # Create figure with same dimensions as example
    plt.figure(figsize=(3.15*1.5, 1.97))
    
    # Plot each topology with different colors from the palette
    for topology_key, data in figure6_data.items():
        iterations = data['iterations']
        mean_rewards = data['mean_rewards']
        lower_bound = data['lower_bound']
        upper_bound = data['upper_bound']
        
        # Use consistent color mapping for each topology type
        color = topology_colors.get(topology_key, '#6a4c93')  # Default purple if not found
        
        # Plot the mean line
        plt.plot(iterations, mean_rewards, color=color, linewidth=2, 
                label=f"{data['topology_type']}")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # Add vertical dashed lines at steps 0, 200, 400, 600, etc. (same as example)
    if len(figure6_data) > 0:
        max_iterations = max(data['iterations'][-1] for data in figure6_data.values())
        vertical_steps = list(range(0, int(max_iterations) + 1, 200))
        for step in vertical_steps:
            plt.axvline(x=step, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Episode reward', fontsize=9, color='black')
    plt.ylim(-1000, 200)
    plt.title(f'Lunarlander', fontsize=10, color='black', fontweight='bold')
    plt.grid(True, alpha=0.3, color=coolors_palette[2])
    plt.gca().set_facecolor('white')
    
    # Create and save legend
    create_and_save_legend(figure6_data, 'lunarlander')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot (sanitize filename)
    safe_folder = data_folder.replace('/', '_')
    plt.savefig(f'lunarlander_{safe_folder}_figure6.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(f'lunarlander_{safe_folder}_figure6.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✅ Lunarlander plot saved from {data_folder} folder!")

# Run Lunarlander analysis
print("🔄 Running Lunarlander Figure 6 analysis...")
plot_lunarlander_figure6(data_folder="lunarlander/N0002/S128")  # Specific subfolder

# ============================================================================
# CELL 4: COMBINED ANALYSIS (OPTIONAL)
# ============================================================================

def plot_combined_figure6(all_data_folders):
    """Plot combined Figure 6 for all tasks from specified data folders."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Create figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(3.15*1.5, 1.97*3), sharex=True)
    
    tasks = ['cartpole', 'acrobot', 'lunarlander']
    figure6_data = {}  # Initialize for legend creation
    
    # Plot each task
    for task_idx, (task_name, data_folder) in enumerate(zip(tasks, all_data_folders)):
        ax = axes[task_idx]
        
        # Load data from specified folder
        topology_groups = load_topology_data(data_folder)
        if not topology_groups:
            print(f"❌ No data found in {data_folder}")
            continue
            
        figure6_data = prepare_bootstrap_figure6_data(topology_groups)
        
        # Plot each topology with different colors from the palette
        for topology_key, data in figure6_data.items():
            iterations = data['iterations']
            mean_rewards = data['mean_rewards']
            lower_bound = data['lower_bound']
            upper_bound = data['upper_bound']
            
            # Use consistent color mapping for each topology type
            color = topology_colors.get(topology_key, '#6a4c93')  # Default purple if not found
            
            # Plot the mean line
            ax.plot(iterations, mean_rewards, color=color, linewidth=2, 
                   label=f"{data['topology_type']}")
            
            # Plot bootstrap confidence intervals
            ax.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                           color=color)
        
        # Add vertical dashed lines at steps 0, 200, 400, 600, etc.
        if len(iterations) > 0:
            vertical_steps = list(range(0, int(max(iterations)) + 1, 200))
            for step in vertical_steps:
                ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Customize each subplot
        ax.set_ylabel('Mean Episode Reward', fontsize=9, color='black')
        ax.set_title(f'{task_name.title()} - {data_folder}', fontsize=10, color='black', fontweight='bold')
        ax.grid(True, alpha=0.3, color=coolors_palette[2])
        ax.set_facecolor('white')
        
        # Set x-axis ticks
        if len(iterations) > 0:
            ax.set_xticks(vertical_steps)
            ax.tick_params(axis="x", labelsize=7)
        
        # Only show x-axis label on bottom subplot
        if task_idx == len(tasks) - 1:
            ax.set_xlabel('Iteration', fontsize=9, color='black')
    
    # Create and save legend (use the last successful figure6_data)
    if figure6_data:
        create_and_save_legend(figure6_data, 'combined')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    
    # Save the combined plot
    fig.savefig('combined_figure6_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    fig.savefig('combined_figure6_comparison.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print("✅ Combined Figure 6 plot saved!")

# Run Combined analysis (optional)
print("🔄 Running Combined Figure 6 analysis...")
plot_combined_figure6(all_data_folders=["cartpole/N0002/S128", "acrobot/N0001/S128", "lunarlander/N0002/S128"])

print("\n🎉 All Figure 6 plots completed!")
print("📁 Files saved:")
print("   - Individual task plots: {task}_{folder}_figure6.png/pdf")
print("   - Legends: {task}_figure6_legend.png/pdf")
print("   - Combined plot: combined_figure6_comparison.png/pdf")
