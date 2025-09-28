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
    'trac': '#1982c4',            # Deep blue for TRAC results
    'standard_mlp': '#ff595e',    # Vibrant red
    'hybrid': '#ffca3a',          # Bright yellow
    'modular': '#8ac926',         # Fresh green
    'small_world': '#1982c4'      # Deep blue
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_topology_data(data_folder: str) -> Dict:
    """Load topology data from iclr/original/control directory, filtering out no_noise files."""
    print(f"🔄 Loading data from {data_folder} folder...")
    
    topology_groups = {}
    base_path = Path(data_folder)
    
    if not base_path.exists():
        print(f"❌ Folder {data_folder} not found!")
        return {}
    
    # Find all reward log files (excluding no_noise files)
    reward_files = []
    for file_path in base_path.glob("trac_reward_log_*.txt"):
        if "no_noise" not in file_path.name:
            reward_files.append(file_path)
    
    print(f"   📁 Found {len(reward_files)} reward log files (with noise)")
    
    # Group files by task
    task_files = {}
    for file_path in reward_files:
        # Extract task name from filename (e.g., "trac_reward_log_CartPole-v1_0.txt" -> "CartPole-v1")
        filename = file_path.name
        if "CartPole-v1" in filename:
            task = "cartpole"
        elif "Acrobot-v1" in filename:
            task = "acrobot"
        elif "MountainCar-v0" in filename:
            task = "mountaincar"
        else:
            continue
            
        if task not in task_files:
            task_files[task] = []
        task_files[task].append(file_path)
    
    # Load data for each task
    for task, files in task_files.items():
        print(f"   📊 Loading {len(files)} files for {task}")
        
        task_rewards = []
        for file_path in files:
            try:
                # Read reward data
                with open(file_path, 'r') as f:
                    rewards = [float(line.strip()) for line in f if line.strip()]
                
                if rewards:
                    # Use all episodes directly (no grouping)
                    task_rewards.append(np.array(rewards))
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
        
        if task_rewards:
            # Use 'trac' as the topology type for all TRAC results
            topology_groups['trac'] = task_rewards
    
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

def plot_cartpole_figure6(data_folder="iclr/original/control"):
    """Plot Figure 6 for Cartpole task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    # Filter for cartpole data only
    cartpole_data = {}
    for topology_key, data_list in topology_groups.items():
        if topology_key == 'trac':
            # Filter cartpole files
            cartpole_files = []
            base_path = Path(data_folder)
            for file_path in base_path.glob("trac_reward_log_CartPole-v1_*.txt"):
                if "no_noise" not in file_path.name:
                    cartpole_files.append(file_path)
            
            if cartpole_files:
                cartpole_rewards = []
                for file_path in cartpole_files:
                    try:
                        with open(file_path, 'r') as f:
                            rewards = [float(line.strip()) for line in f if line.strip()]
                        
                        if rewards:
                            # Group episodes by iteration (every 2 episodes) and calculate mean per iteration
                            iteration_rewards = []
                            for i in range(0, len(rewards), 2):
                                if i + 1 < len(rewards):
                                    iteration_mean = (rewards[i] + rewards[i+1]) / 2
                                    iteration_rewards.append(iteration_mean)
                                elif i < len(rewards):
                                    iteration_rewards.append(rewards[i])
                            
                            cartpole_rewards.append(np.array(iteration_rewards))
                            
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
                        continue
                
                if cartpole_rewards:
                    cartpole_data[topology_key] = cartpole_rewards
    
    if not cartpole_data:
        print(f"❌ No cartpole data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(cartpole_data)
    
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
                label=f"TRAC ({data['num_seeds']} seeds)")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # No vertical reference lines for clean background
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Mean Episode reward', fontsize=9, color='black')
    plt.title(f'Cartpole', fontsize=10, color='black', fontweight='bold')
    plt.grid(False)
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
plot_cartpole_figure6(data_folder="iclr/original/control")

# ============================================================================
# CELL 2: ACROBOT ANALYSIS  
# ============================================================================

def plot_acrobot_figure6(data_folder="iclr/original/control"):
    """Plot Figure 6 for Acrobot task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    # Filter for acrobot data only
    acrobot_data = {}
    for topology_key, data_list in topology_groups.items():
        if topology_key == 'trac':
            # Filter acrobot files
            acrobot_files = []
            base_path = Path(data_folder)
            for file_path in base_path.glob("trac_reward_log_Acrobot-v1_*.txt"):
                if "no_noise" not in file_path.name:
                    acrobot_files.append(file_path)
            
            if acrobot_files:
                acrobot_rewards = []
                for file_path in acrobot_files:
                    try:
                        with open(file_path, 'r') as f:
                            rewards = [float(line.strip()) for line in f if line.strip()]
                        
                        if rewards:
                            # Group episodes by iteration (every 2 episodes) and calculate mean per iteration
                            iteration_rewards = []
                            for i in range(0, len(rewards), 2):
                                if i + 1 < len(rewards):
                                    iteration_mean = (rewards[i] + rewards[i+1]) / 2
                                    iteration_rewards.append(iteration_mean)
                                elif i < len(rewards):
                                    iteration_rewards.append(rewards[i])
                            
                            acrobot_rewards.append(np.array(iteration_rewards))
                            
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
                        continue
                
                if acrobot_rewards:
                    acrobot_data[topology_key] = acrobot_rewards
    
    if not acrobot_data:
        print(f"❌ No acrobot data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(acrobot_data)
    
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
                label=f"TRAC ({data['num_seeds']} seeds)")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # No vertical reference lines for clean background
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Episode reward', fontsize=9, color='black')
    plt.title(f'Acrobot', fontsize=10, color='black', fontweight='bold')
    plt.grid(False)
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
plot_acrobot_figure6(data_folder="iclr/original/control")

# ============================================================================
# CELL 3: MOUNTAINCAR ANALYSIS
# ============================================================================

def plot_mountaincar_figure6(data_folder="iclr/original/control"):
    """Plot Figure 6 for MountainCar task from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Load data from specified folder
    topology_groups = load_topology_data(data_folder)
    if not topology_groups:
        print(f"❌ No data found in {data_folder}")
        return
    
    # Filter for mountaincar data only
    mountaincar_data = {}
    for topology_key, data_list in topology_groups.items():
        if topology_key == 'trac':
            # Filter mountaincar files
            mountaincar_files = []
            base_path = Path(data_folder)
            for file_path in base_path.glob("trac_reward_log_MountainCar-v0_*.txt"):
                if "no_noise" not in file_path.name:
                    mountaincar_files.append(file_path)
            
            if mountaincar_files:
                mountaincar_rewards = []
                for file_path in mountaincar_files:
                    try:
                        with open(file_path, 'r') as f:
                            rewards = [float(line.strip()) for line in f if line.strip()]
                        
                        if rewards:
                            # Group episodes by iteration (every 2 episodes) and calculate mean per iteration
                            iteration_rewards = []
                            for i in range(0, len(rewards), 2):
                                if i + 1 < len(rewards):
                                    iteration_mean = (rewards[i] + rewards[i+1]) / 2
                                    iteration_rewards.append(iteration_mean)
                                elif i < len(rewards):
                                    iteration_rewards.append(rewards[i])
                            
                            mountaincar_rewards.append(np.array(iteration_rewards))
                            
                    except Exception as e:
                        print(f"⚠️ Error loading {file_path}: {e}")
                        continue
                
                if mountaincar_rewards:
                    mountaincar_data[topology_key] = mountaincar_rewards
    
    if not mountaincar_data:
        print(f"❌ No mountaincar data found in {data_folder}")
        return
    
    figure6_data = prepare_bootstrap_figure6_data(mountaincar_data)
    
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
                label=f"TRAC ({data['num_seeds']} seeds)")
        
        # Plot bootstrap confidence intervals
        plt.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                        color=color)
    
    # No vertical reference lines for clean background
    
    # Customize the plot
    plt.xlabel('Iteration', fontsize=9, color='black')
    plt.ylabel('Episode reward', fontsize=9, color='black')
    plt.title(f'MountainCar', fontsize=10, color='black', fontweight='bold')
    plt.grid(False)
    plt.gca().set_facecolor('white')
    
    # Create and save legend
    create_and_save_legend(figure6_data, 'mountaincar')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot (sanitize filename)
    safe_folder = data_folder.replace('/', '_')
    plt.savefig(f'mountaincar_{safe_folder}_figure6.png', dpi=300, bbox_inches='tight', 
                facecolor='white')
    plt.savefig(f'mountaincar_{safe_folder}_figure6.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white')
    print(f"✅ MountainCar plot saved from {data_folder} folder!")

# Run MountainCar analysis
print("🔄 Running MountainCar Figure 6 analysis...")
plot_mountaincar_figure6(data_folder="iclr/original/control")

# ============================================================================
# CELL 4: COMBINED ANALYSIS (OPTIONAL)
# ============================================================================

def plot_combined_figure6(data_folder="iclr/original/control"):
    """Plot combined Figure 6 for all tasks from specified data folder."""
    
    # Set the color palette for matplotlib
    plt.style.use('default')
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=coolors_palette)
    
    # Create figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(3.15*1.5, 1.97*3), sharex=True)
    
    tasks = ['cartpole', 'acrobot', 'mountaincar']
    figure6_data = {}  # Initialize for legend creation
    
    # Plot each task
    for task_idx, task_name in enumerate(tasks):
        ax = axes[task_idx]
        
        # Load data for specific task
        if task_name == 'cartpole':
            task_files = list(Path(data_folder).glob("trac_reward_log_CartPole-v1_*.txt"))
        elif task_name == 'acrobot':
            task_files = list(Path(data_folder).glob("trac_reward_log_Acrobot-v1_*.txt"))
        elif task_name == 'mountaincar':
            task_files = list(Path(data_folder).glob("trac_reward_log_MountainCar-v0_*.txt"))
        else:
            continue
        
        # Filter out no_noise files
        task_files = [f for f in task_files if "no_noise" not in f.name]
        
        if not task_files:
            print(f"❌ No {task_name} data found in {data_folder}")
            continue
        
        # Load task data
        task_rewards = []
        for file_path in task_files:
            try:
                with open(file_path, 'r') as f:
                    rewards = [float(line.strip()) for line in f if line.strip()]
                
                if rewards:
                    # Use all episodes directly (no grouping)
                    task_rewards.append(np.array(rewards))
                    
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
        
        if not task_rewards:
            continue
            
        # Prepare data for plotting
        task_data = {'trac': task_rewards}
        figure6_data = prepare_bootstrap_figure6_data(task_data)
        
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
                   label=f"TRAC ({data['num_seeds']} seeds)")
            
            # Plot bootstrap confidence intervals
            ax.fill_between(iterations, lower_bound, upper_bound, alpha=0.3, 
                           color=color)
        
        # No vertical reference lines for clean background
        
        # Customize each subplot
        ax.set_ylabel('Mean Episode Reward', fontsize=9, color='black')
        ax.set_title(f'{task_name.title()}', fontsize=10, color='black', fontweight='bold')
        ax.grid(False)
        ax.set_facecolor('white')
        
        # Set x-axis ticks
        if len(iterations) > 0:
            # Set reasonable x-axis ticks for 2000 episodes
            x_ticks = list(range(0, int(max(iterations)) + 1, 400))
            ax.set_xticks(x_ticks)
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
plot_combined_figure6(data_folder="iclr/original/control")

print("\n🎉 All Figure 6 plots completed!")
print("📁 Files saved:")
print("   - Individual task plots: {task}_{folder}_figure6.png/pdf")
print("   - Legends: {task}_figure6_legend.png/pdf")
print("   - Combined plot: combined_figure6_comparison.png/pdf")
