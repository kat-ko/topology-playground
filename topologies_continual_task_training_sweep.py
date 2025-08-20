#!/usr/bin/env python3
"""
Triple-Task Topology Networks with Weights & Biases Sweep Support

This script is a modified version of the triple-task training script that can work with wandb sweeps
for hyperparameter optimization. It reads hyperparameters from wandb.config and runs training accordingly.
Includes reward scaling and task normalization for fair comparison across tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import sys
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import wandb
import networkx as nx
import io
import base64
from PIL import Image
import csv
import argparse

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.standard_mlp import StandardMLPTopology
from src.networks.ffn import FeedForwardNetwork
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager
from src.utils.capacity_matching_helper import pre_calculate_capacity_matching
from src.utils.device_manager import get_device_manager, get_device_info
from src.utils.task_normalization import (
    compute_multi_task_metrics, log_normalized_metrics, print_normalized_summary,
    get_task_thresholds, get_normalization_constants, normalize_reward,
    calculate_reward_completion_percentage
)
from src.utils.advanced_plotting import (
    log_streamlined_plots_for_run, create_multi_phase_learning_curves
)
from src.utils.task_training_config import get_task_timesteps, create_convergence_callback
from src.utils.topology_logging_handler import (
    SimplifiedLoggingHandler, SimplifiedCallback, create_logging_handler
)

# ============================================================================
# LOCAL DATA COLLECTION SYSTEM (No W&B Dependency)
# ============================================================================

class LocalDataCollector:
    """
    Collects training data locally without W&B dependency.
    Focuses on episode and shift data for Figure-6 style plots.
    """
    
    def __init__(self, base_path="test_experiments", run_id=None):
        self.base_path = base_path
        self.run_id = run_id or f"test_{int(time.time())}"
        self.run_dir = f"{self.base_path}/{self.run_id}"
        self.episode_data = []
        self.shift_data = []
        
        # Batch writing for performance optimization
        self.episode_batch = []
        self.shift_batch = []
        self.batch_size = 200  # Write to CSV every 200 episodes instead of every episode
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(f"{self.run_dir}/data", exist_ok=True)
        os.makedirs(f"{self.run_dir}/plots", exist_ok=True)
        
        # Initialize CSV files
        self._init_csv_files()
        
        print(f"📁 Local data collection initialized: {self.run_dir}")
        print(f"   Batch writing enabled: CSV updates every {self.batch_size} episodes")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Episode data CSV
        episode_headers = [
            'global_step_end', 'episode_length', 'episode_return_raw',
            'episode_return_scaled', 'shift_id', 'seed', 'env', 'topology'
        ]
        with open(f"{self.run_dir}/data/episode_data.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(episode_headers)
        
        # Shift data CSV
        shift_headers = [
            'shift_step', 'shift_id', 'offset_repr', 'seed', 'env', 'topology'
        ]
        with open(f"{self.run_dir}/data/shift_data.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(shift_headers)
    
    def log_episode(self, episode_info):
        """Log episode completion data with batch writing for performance."""
        self.episode_data.append(episode_info)
        self.episode_batch.append(episode_info)
        
        # Write to CSV in batches for performance
        if len(self.episode_batch) >= self.batch_size:
            self._flush_episode_batch()
    
    def _flush_episode_batch(self):
        """Flush episode batch to CSV file."""
        if not self.episode_batch:
            return
            
        with open(f"{self.run_dir}/data/episode_data.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            for episode_info in self.episode_batch:
                writer.writerow([
                    episode_info['global_step_end'],
                    episode_info['episode_length'],
                    episode_info['episode_return_raw'],
                    episode_info['episode_return_scaled'],
                    episode_info['shift_id'],
                    episode_info['seed'],
                    episode_info['env'],
                    episode_info['topology']
                ])
        
        # Clear batch after writing
        self.episode_batch = []
    
    def log_shift(self, shift_info):
        """Log shift boundary data with batch writing for performance."""
        self.shift_data.append(shift_info)
        self.shift_batch.append(shift_info)
        
        # Write to CSV in batches for performance
        if len(self.shift_batch) >= self.batch_size:
            self._flush_shift_batch()
    
    def _flush_shift_batch(self):
        """Flush shift batch to CSV file."""
        if not self.shift_batch:
            return
            
        with open(f"{self.run_dir}/data/shift_data.csv", 'a', newline='') as f:
            writer = csv.writer(f)
            for shift_info in self.shift_batch:
                writer.writerow([
                    shift_info['shift_step'],
                    shift_info['shift_id'],
                    shift_info['offset_repr'],
                    shift_info['seed'],
                    shift_info['env'],
                    shift_info['topology']
                ])
        
        # Clear batch after writing
        self.shift_batch = []
    
    def finalize_run(self):
        """Save run metadata and create summary."""
        # Flush any remaining batches before finalizing
        if self.episode_batch:
            self._flush_episode_batch()
        if self.shift_batch:
            self._flush_shift_batch()
        
        metadata = {
            'run_id': self.run_id,
            'timestamp': time.time(),
            'total_episodes': len(self.episode_data),
            'total_shifts': len(self.shift_data),
            'episode_data_file': f"{self.run_dir}/data/episode_data.csv",
            'shift_data_file': f"{self.run_dir}/data/shift_data.csv"
        }
        
        with open(f"{self.run_dir}/run_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Test run completed: {self.run_id}")
        print(f"📊 Data saved to: {self.run_dir}/data/")
        print(f"📁 Ready for plotting: {self.run_dir}/plots/")
        
        return self.run_dir

# ============================================================================
# FIGURE-6 STYLE PLOTTING SYSTEM
# ============================================================================

class Figure6Plotter:
    """
    Creates Figure-6 style plots for topology comparison.
    Implements the exact plotting protocol from the checklist.
    """
    
    def __init__(self, data_path="test_experiments"):
        self.data_path = data_path
        self.plot_style = self._setup_plot_style()
        
        # Create plots directory if it doesn't exist
        plots_dir = f"{self.data_path}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        print(f"🎨 Figure-6 Plotter initialized for: {data_path}")
        print(f"📁 Plots will be saved to: {plots_dir}")
    
    def _setup_plot_style(self):
        """Setup matplotlib style for publication-quality plots."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        
        return {
            'small_world': {'color': 'red', 'label': 'Small World'},
            'fully_connected': {'color': 'blue', 'label': 'Fully Connected'}
        }
    
    def create_topology_comparison_plots(self, env_name, seeds=[42, 123, 456, 789, 999]):
        """
        Create Figure-6 style plots for Small World vs Fully Connected comparison.
        Creates both raw and scaled reward versions.
        
        Args:
            env_name: Environment name (CartPole-v1, Acrobot-v1, LunarLander-v2)
            seeds: List of seeds to aggregate over
        """
        print(f"🎨 Creating Figure-6 style plots for {env_name}")
        print(f"📊 Generating both raw and scaled reward versions")
        
        # Load data for both topologies
        small_world_data = self._load_topology_data(env_name, 'small_world', seeds)
        fully_connected_data = self._load_topology_data(env_name, 'fully_connected', seeds)
        
        if small_world_data is None or fully_connected_data is None:
            print(f"❌ Failed to load data for one or both topologies")
            return
        
        # Create individual plots for both raw and scaled
        self._create_individual_plot(env_name, 'small_world', small_world_data['raw'], 'red', 'raw')
        self._create_individual_plot(env_name, 'small_world', small_world_data['scaled'], 'red', 'scaled')
        self._create_individual_plot(env_name, 'fully_connected', fully_connected_data['raw'], 'blue', 'raw')
        self._create_individual_plot(env_name, 'fully_connected', fully_connected_data['scaled'], 'blue', 'scaled')
        
        # Create comparison plots for both raw and scaled
        self._create_comparison_plot(env_name, small_world_data['raw'], fully_connected_data['raw'], 'raw')
        self._create_comparison_plot(env_name, small_world_data['scaled'], fully_connected_data['scaled'], 'scaled')
        
        # Create combined Figure-6 style plots for both raw and scaled
        self._create_combined_figure6_plot(env_name, small_world_data['raw'], fully_connected_data['raw'], 'raw')
        self._create_combined_figure6_plot(env_name, small_world_data['scaled'], fully_connected_data['scaled'], 'scaled')
        
        print(f"✅ All Figure-6 style plots created for {env_name}")
        print(f"📊 Generated: Individual plots (raw/scaled), comparison plots (raw/scaled), and combined plots (raw/scaled)")
    
    def _load_topology_data(self, env_name, topology, seeds):
        """Load and aggregate data for a specific topology across seeds."""
        all_seed_data = []
        
        for seed in seeds:
            # Find run directory for this seed/topology/env combination
            run_dir = self._find_run_directory(env_name, topology, seed)
            if run_dir:
                episode_file = f"{run_dir}/data/episode_data.csv"
                if os.path.exists(episode_file):
                    try:
                        seed_data = pd.read_csv(episode_file)
                        # Process individual seed data (now returns dict with 'raw' and 'scaled')
                        processed_seed_data = self._process_episode_data(seed_data)
                        all_seed_data.append(processed_seed_data)
                        print(f"📊 Loaded data for {topology} seed {seed}: {len(seed_data)} episodes")
                    except Exception as e:
                        print(f"⚠️  Failed to load {episode_file}: {e}")
        
        if not all_seed_data:
            print(f"⚠️  No data found for {topology} on {env_name}")
            return None
        
        # Aggregate across seeds to get Figure-6 style statistics (now returns dict with 'raw' and 'scaled')
        aggregated_data = self._aggregate_across_seeds(all_seed_data)
        
        if aggregated_data is None or aggregated_data['raw'].empty:
            print(f"⚠️  Failed to aggregate data for {topology} on {env_name}")
            return None
        
        print(f"📊 Aggregated data for {topology}: {len(aggregated_data['raw'])} steps, {len(all_seed_data)} seeds")
        return aggregated_data
    
    def _find_run_directory(self, env_name, topology, seed):
        """Find the run directory for a specific experiment."""
        # Look for directories matching the pattern
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path):
                # Check if this directory contains data for our experiment
                episode_file = f"{item_path}/data/episode_data.csv"
                if os.path.exists(episode_file):
                    try:
                        # Read a few lines to check if it's the right experiment
                        sample_data = pd.read_csv(episode_file, nrows=5)
                        if (sample_data['env'].iloc[0] == env_name and 
                            sample_data['topology'].iloc[0] == topology and
                            sample_data['seed'].iloc[0] == seed):
                            return item_path
                    except:
                        continue
        return None
    
    def _process_episode_data(self, episode_data):
        """
        Process episode data according to Figure-6 protocol:
        1. Sort by global_step_end
        2. Apply 5-episode moving average smoothing per seed
        3. Align to common step axis
        4. Process both raw and scaled returns
        """
        # Sort by step
        episode_data = episode_data.sort_values('global_step_end')
        
        # Apply 5-episode moving average smoothing per seed for both raw and scaled
        episode_data['episode_return_raw_smoothed'] = (
            episode_data['episode_return_raw']
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        episode_data['episode_return_scaled_smoothed'] = (
            episode_data['episode_return_scaled']
            .rolling(window=5, min_periods=1)
            .mean()
        )
        
        # Create common step grid (0, 10, 20, ..., 3000)
        step_grid = np.arange(0, 3001, 10)
        
        # Align data to grid for both raw and scaled
        aligned_data_raw = []
        aligned_data_scaled = []
        
        for step in step_grid:
            # Find most recent episode ending before or at this step
            valid_episodes = episode_data[episode_data['global_step_end'] <= step]
            if not valid_episodes.empty:
                latest_episode = valid_episodes.iloc[-1]
                
                # Raw return data
                aligned_data_raw.append({
                    'step': step,
                    'episode_return': latest_episode['episode_return_raw_smoothed'],
                    'episode_return_raw': latest_episode['episode_return_raw']
                })
                
                # Scaled return data
                aligned_data_scaled.append({
                    'step': step,
                    'episode_return': latest_episode['episode_return_scaled_smoothed'],
                    'episode_return_scaled': latest_episode['episode_return_scaled']
                })
        
        return {
            'raw': pd.DataFrame(aligned_data_raw),
            'scaled': pd.DataFrame(aligned_data_scaled)
        }
    
    def _aggregate_across_seeds(self, all_seed_data):
        """
        Aggregate data across multiple seeds to create Figure-6 style statistics.
        
        Args:
            all_seed_data: List of DataFrames, one per seed
            
        Returns:
            Dictionary with 'raw' and 'scaled' DataFrames, each with mean ± SD at each step
        """
        if not all_seed_data:
            return None
        
        # Separate raw and scaled data
        raw_seed_data = [seed_data['raw'] for seed_data in all_seed_data if 'raw' in seed_data]
        scaled_seed_data = [seed_data['scaled'] for seed_data in all_seed_data if 'scaled' in seed_data]
        
        # Get common step grid
        step_grid = np.arange(0, 3001, 10)
        
        # Aggregate raw data
        aggregated_raw = []
        for step in step_grid:
            step_values = []
            for seed_df in raw_seed_data:
                if not seed_df.empty:
                    step_data = seed_df[seed_df['step'] == step]
                    if not step_data.empty:
                        step_values.append(step_data.iloc[0]['episode_return'])
            
            if step_values:
                mean_val = np.mean(step_values)
                std_val = np.std(step_values)
                n_seeds = len(step_values)
                
                aggregated_raw.append({
                    'step': step,
                    'mean': mean_val,
                    'std': std_val,
                    'n_seeds': n_seeds,
                    'min_val': mean_val - std_val,
                    'max_val': mean_val + std_val
                })
        
        # Aggregate scaled data
        aggregated_scaled = []
        for step in step_grid:
            step_values = []
            for seed_df in scaled_seed_data:
                if not seed_df.empty:
                    step_data = seed_df[seed_df['step'] == step]
                    if not step_data.empty:
                        step_values.append(step_data.iloc[0]['episode_return'])
            
            if step_values:
                mean_val = np.mean(step_values)
                std_val = np.std(step_values)
                n_seeds = len(step_values)
                
                aggregated_scaled.append({
                    'step': step,
                    'mean': mean_val,
                    'std': std_val,
                    'n_seeds': n_seeds,
                    'min_val': mean_val - std_val,
                    'max_val': mean_val + std_val
                })
        
        return {
            'raw': pd.DataFrame(aggregated_raw),
            'scaled': pd.DataFrame(aggregated_scaled)
        }
    
    def _create_individual_plot(self, env_name, topology, data, color, version):
        """Create Figure-6 style individual plot for one topology with mean ± SD."""
        if data is None or data.empty:
            print(f"⚠️  Skipping {topology} {version} plot - no data")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot mean performance curve
        ax.plot(data['step'], data['mean'], 
                color=color, linewidth=2, label=f'{topology.replace("_", " ").title()} ({version.title()})')
        
        # Add shaded area for mean ± SD
        ax.fill_between(data['step'], 
                       data['min_val'], 
                       data['max_val'], 
                       color=color, alpha=0.3, 
                       label=f'±1 SD ({data["n_seeds"].iloc[0]} seeds)')
        
        # Add shift boundary markers
        shift_steps = np.arange(0, 3001, 200)
        for shift_step in shift_steps:
            ax.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.7)
        
        # Customize plot based on version
        ax.set_xlabel('Environment Steps')
        if version == 'raw':
            ax.set_ylabel('Episode Return (Raw)')
            ax.set_title(f'{env_name} - {topology.replace("_", " ").title()} - Raw Rewards\nMean ± SD across {data["n_seeds"].iloc[0]} seeds')
        else:  # scaled
            ax.set_ylabel('Episode Return (Scaled)')
            ax.set_title(f'{env_name} - {topology.replace("_", " ").title()} - Scaled Rewards (×20)\nMean ± SD across {data["n_seeds"].iloc[0]} seeds')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save plot
        plot_path = f"{self.data_path}/plots/{env_name}_{topology}_{version}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Figure-6 style {version} individual plot saved: {plot_path}")
    
    def _create_comparison_plot(self, env_name, small_world_data, fully_connected_data, version):
        """Create Figure-6 style combined comparison plot with mean ± SD."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        
        # Plot Small World (top) with mean ± SD
        if small_world_data is not None and not small_world_data.empty:
            # Mean line
            ax1.plot(small_world_data['step'], small_world_data['mean'], 
                     color='red', linewidth=2, label=f'Small World (Mean) - {version.title()}')
            # Shaded area
            ax1.fill_between(small_world_data['step'], 
                            small_world_data['min_val'], 
                            small_world_data['max_val'], 
                            color='red', alpha=0.3, 
                            label=f'±1 SD ({small_world_data["n_seeds"].iloc[0]} seeds)')
        
        # Plot Fully Connected (bottom) with mean ± SD
        if fully_connected_data is not None and not fully_connected_data.empty:
            # Mean line
            ax2.plot(fully_connected_data['step'], fully_connected_data['mean'], 
                     color='blue', linewidth=2, label=f'Fully Connected (Mean) - {version.title()}')
            # Shaded area
            ax2.fill_between(fully_connected_data['step'], 
                            fully_connected_data['min_val'], 
                            fully_connected_data['max_val'], 
                            color='blue', alpha=0.3, 
                            label=f'±1 SD ({fully_connected_data["n_seeds"].iloc[0]} seeds)')
        
        # Add shift boundaries to both subplots
        shift_steps = np.arange(0, 3001, 200)
        for ax in [ax1, ax2]:
            for shift_step in shift_steps:
                ax.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.legend()
            # Set y-label based on version
            if version == 'raw':
                ax.set_ylabel('Episode Return (Raw)')
            else:  # scaled
                ax.set_ylabel('Episode Return (Scaled)')
        
        # Set x-label only for bottom subplot
        ax2.set_xlabel('Environment Steps')
        
        # Set titles based on version
        if version == 'raw':
            ax1.set_title(f'{env_name} - Small World vs Fully Connected Comparison - Raw Rewards')
            ax2.set_title('Fully Connected (Mean ± SD) - Raw Rewards')
        else:  # scaled
            ax1.set_title(f'{env_name} - Small World vs Fully Connected Comparison - Scaled Rewards (×20)')
            ax2.set_title('Fully Connected (Mean ± SD) - Scaled Rewards (×20)')
        
        # Save plot
        plot_path = f"{self.data_path}/plots/{env_name}_comparison_{version}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Figure-6 style {version} comparison plot saved: {plot_path}")
    
    def _create_combined_figure6_plot(self, env_name, small_world_data, fully_connected_data, version):
        """Create single Figure-6 style plot with both topologies on same axes."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot Small World with mean ± SD
        if small_world_data is not None and not small_world_data.empty:
            # Mean line
            ax.plot(small_world_data['step'], small_world_data['mean'], 
                    color='red', linewidth=3, label=f'Small World (Mean) - {version.title()}')
            # Shaded area
            ax.fill_between(small_world_data['step'], 
                           small_world_data['min_val'], 
                           small_world_data['max_val'], 
                           color='red', alpha=0.2, 
                           label=f'Small World ±1 SD - {version.title()}')
        
        # Plot Fully Connected with mean ± SD
        if fully_connected_data is not None and not fully_connected_data.empty:
            # Mean line
            ax.plot(fully_connected_data['step'], fully_connected_data['mean'], 
                    color='blue', linewidth=3, label=f'Fully Connected (Mean) - {version.title()}')
            # Shaded area
            ax.fill_between(fully_connected_data['step'], 
                           fully_connected_data['min_val'], 
                           fully_connected_data['max_val'], 
                           color='blue', alpha=0.2, 
                           label=f'Fully Connected ±1 SD - {version.title()}')
        
        # Add shift boundaries
        shift_steps = np.arange(0, 3001, 200)
        for shift_step in shift_steps:
            ax.axvline(x=shift_step, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # Customize plot based on version
        ax.set_xlabel('Environment Steps', fontsize=14)
        if version == 'raw':
            ax.set_ylabel('Episode Return (Raw)', fontsize=14)
            ax.set_title(f'{env_name} - Topology Comparison (Figure-6 Style) - Raw Rewards\nMean ± SD across 5 seeds', 
                         fontsize=16, fontweight='bold')
        else:  # scaled
            ax.set_ylabel('Episode Return (Scaled)', fontsize=14)
            ax.set_title(f'{env_name} - Topology Comparison (Figure-6 Style) - Scaled Rewards (×20)\nMean ± SD across 5 seeds', 
                         fontsize=16, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='upper right')
        
        # Add shift labels
        for i, shift_step in enumerate(shift_steps):
            if i % 2 == 0:  # Label every other shift to avoid clutter
                ax.text(shift_step, ax.get_ylim()[1] * 0.95, f'Shift {i}', 
                       rotation=90, ha='center', va='top', fontsize=10, alpha=0.7)
        
        # Save plot
        plot_path = f"{self.data_path}/plots/{env_name}_figure6_combined_{version}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Figure-6 style {version} combined plot saved: {plot_path}")
        return fig

# 🚨 CRITICAL: Set W&B environment variables BEFORE any W&B usage
# This prevents W&B from logging internal metrics with step 0
os.environ['WANDB_SILENT'] = 'false'  # Allow W&B to log legitimate training metrics
os.environ['WANDB_DISABLE_CODE'] = 'true'  # Disable code logging
os.environ['WANDB_DISABLE_GIT'] = 'true'  # Disable git logging
os.environ['WANDB_DISABLE_ARTIFACTS'] = 'false'  # Allow artifacts for training data
os.environ['WANDB_DISABLE_SERVICE'] = 'false'  # Allow W&B service for logging
print("🔧 Set W&B environment variables at module level to allow legitimate training metrics")

# 🚨 GPU SUPPORT: Initialize device manager early
try:
    DEVICE_MANAGER = get_device_manager()
    DEVICE_INFO = get_device_info()
    print(f"🔧 GPU Support: {DEVICE_INFO['device']}")
    if DEVICE_INFO['is_cuda']:
        print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
        print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
except Exception as e:
    print(f"⚠️  GPU Support: Failed to initialize device manager: {e}")
    DEVICE_MANAGER = None
    DEVICE_INFO = {'device': 'cpu', 'is_cuda': False, 'is_gpu_available': False}

# 🚨 CONFIGURATION INTEGRATION: Import unified configuration system
try:
    # from wandb_sweep_config import get_config_by_name, generate_parameter_combinations
    print("🔧 Configuration system: Temporarily disabled for Phase 1 testing")
    CONFIG_SYSTEM_AVAILABLE = False
except ImportError as e:
    print(f"⚠️  Configuration system: Failed to import unified config: {e}")
    CONFIG_SYSTEM_AVAILABLE = False

# ============================================================================
# CONTINUAL LEARNING WRAPPER
# ============================================================================

class ContinualLearningWrapper(gym.Wrapper):
    """
    Wrapper for continual learning with piecewise-constant observation shifts.
    
    PAPER-ACCURATE IMPLEMENTATION:
    - Iteration-based training (not step-based)
    - Pre-generated perturbations for all 15 levels
    - Level 0 (iterations 0-199): NO NOISE - Clean baseline learning
    - Level 1+ (iterations 200+): Perturbations applied every 200 iterations
    - Reward scaling: Division by 20 (creates small gradients)
    - Episode capping at 400 steps maximum
    """
    
    def __init__(self, env, task_name, max_iterations=3000, level_switch=200, shift_range=[0, 2], seed=None, reward_scale=20.0, episode_cap=400, logging_callback=None, num_levels=15):
        super().__init__(env)
        self.task_name = task_name
        self.max_iterations = max_iterations
        self.level_switch = level_switch
        self.shift_range = shift_range
        self.seed = seed
        self.reward_scale = reward_scale  # This will be used for division, not multiplication
        self.episode_cap = episode_cap
        self.logging_callback = logging_callback
        self.num_levels = num_levels  # Number of distribution shift levels
        
        # Initialize iteration and level tracking
        self.current_iteration = 0
        self.current_level = 0
        self.episodes_in_current_iteration = 0
        self.max_episodes_per_iteration = 2
        
        # Pre-generate all perturbation levels using seed
        if seed is not None:
            self.perturbation_rng = np.random.RandomState(seed)
        else:
            self.perturbation_rng = np.random.RandomState(42)
        
        # Generate perturbations for all levels
        obs_dim = self.observation_space.shape[0]
        self.perturbations = []
        for level in range(self.num_levels):  # Dynamic number of levels
            if level == 0:
                # Level 0: NO NOISE - Clean baseline
                perturbation = np.zeros(obs_dim)
            else:
                # Levels 1+: Random perturbations
                perturbation = self.perturbation_rng.uniform(
                    low=self.shift_range[0], 
                    high=self.shift_range[1], 
                    size=obs_dim
                )
            self.perturbations.append(perturbation)
        
        # Set initial perturbation (Level 0 = no noise)
        self.current_perturbation = self.perturbations[0]
        
        # Episode tracking for capping
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_returns = []
        
        # Environment step counting (for logging)
        self.total_env_steps = 0
        
        print(f"🎲 Continual Learning Wrapper initialized (Paper-Accurate):")
        print(f"   Task: {task_name}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Level switch: {level_switch} iterations")
        print(f"   Number of levels: {self.num_levels}")
        print(f"   Shift range: {shift_range}")
        print(f"   Reward scale: {reward_scale} (division factor)")
        print(f"   Episode cap: {episode_cap} steps")
        print(f"   Max episodes per iteration: {self.max_episodes_per_iteration}")
        print(f"   Initial perturbation (Level 0): {self.current_perturbation}")
        print(f"   Total perturbation levels: {len(self.perturbations)}")
        if logging_callback:
            print(f"   Enhanced logging: Enabled")
    
    def set_iteration(self, iteration):
        """Set the current iteration and update perturbation level accordingly."""
        self.current_iteration = iteration
        
        # Calculate which perturbation level we're in
        # Switch every 200 iterations, not every 200 env steps
        new_level = iteration // self.level_switch
        
        # Only update and log if the level actually changed
        if new_level != self.current_level:
            self.current_level = new_level
            
            # Ensure we don't exceed the number of pre-generated perturbations
            if self.current_level < len(self.perturbations):
                self.current_perturbation = self.perturbations[self.current_level]
            else:
                # If we exceed, use the last perturbation
                self.current_perturbation = self.perturbations[-1]
                self.current_level = len(self.perturbations) - 1
            
            # Log the level activation (only when it changes)
            if self.current_level == 0:
                print(f"\n🎯 NEW NOISE LEVEL ACTIVATED:")
                print(f"   🧹 Level {self.current_level}: Clean Baseline (NO NOISE)")
                print(f"   📍 Iteration: {iteration}")
                print(f"   📊 Environment Steps: ~{iteration * 800:,}")
            else:
                print(f"\n🎯 NEW NOISE LEVEL ACTIVATED:")
                print(f"   📊 Level {self.current_level}: Noise Vector Applied")
                print(f"   📍 Iteration: {iteration}")
                print(f"   📊 Environment Steps: ~{iteration * 800:,}")
                print(f"   🔧 Perturbation: {self.current_perturbation}")
            
            # Log level change to callback if available
            if self.logging_callback and hasattr(self.logging_callback, '_log_perturbation_level_change'):
                try:
                    self.logging_callback._log_perturbation_level_change(iteration, self.current_level, self.current_perturbation)
                except Exception as e:
                    print(f"⚠️  Level change logging failed: {e}")
        
        # Show progress for current level
        self._show_progress()
    
    def step(self, action):
        """Step environment and apply current observation shift with reward scaling and episode capping."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Apply current perturbation to observation
        shifted_obs = obs + self.current_perturbation
        
        # Apply reward scaling (divide by 20 for training, as per notebook)
        scaled_reward = reward / self.reward_scale
        
        # Store raw reward for logging (we'll multiply back by 20 when logging)
        self.episode_reward += reward
        
        # Update episode tracking
        self.episode_step += 1
        self.total_env_steps += 1
        
        # Check episode termination (cap at episode_cap steps)
        episode_ended = done or truncated or self.episode_step >= self.episode_cap
        
        if episode_ended:
            self._log_episode()
            self._reset_episode()
            
            # Increment episode counter for current iteration
            self.episodes_in_current_iteration += 1
        
        return shifted_obs, scaled_reward, episode_ended, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and maintain perturbation state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self._reset_episode()
        
        # Apply current perturbation to reset observation
        shifted_obs = obs + self.current_perturbation
        
        return shifted_obs, info
    
    def _log_episode(self):
        """Log episode completion with raw returns and iteration information."""
        episode_info = {
            'global_step_end': self.total_env_steps,
            'episode_length': self.episode_step,
            'episode_return_raw': self.episode_reward,  # Raw return (×1)
            'episode_return_scaled': self.episode_reward * self.reward_scale,  # Scaled return (×20)
            'shift_id': self.current_level,
            'iteration': self.current_iteration,
            'level': self.current_level,
            'perturbation_applied': self.current_perturbation.tolist(),
            'shift_boundary': (self.current_iteration % self.level_switch == 0)
        }
        
        # Store episode data
        self.episode_returns.append(episode_info)
        
        # Use enhanced logging callback if available
        if self.logging_callback and hasattr(self.logging_callback, '_log_episode_completion'):
            try:
                self.logging_callback._log_episode_completion(episode_info)
                # Only show episode logging in verbose mode or for milestone episodes
                if self.total_env_steps % 10000 == 0:  # Every 10k steps
                    print(f"📊 Episode {len(self.episode_returns)}: Raw={episode_info['episode_return_raw']:.0f}, Steps={self.total_env_steps:,}")
            except Exception as e:
                print(f"⚠️  Episode logging callback failed: {e}")
    
    def _show_progress(self):
        """Show progress using tqdm for smooth, continuous tracking."""
        # Update progress every iteration for continuous tracking
        if self.current_iteration % 1 == 0:  # Every iteration
            # Create new progress bar when level changes
            if not hasattr(self, 'current_pbar') or self.current_pbar is None:
                level_desc = "🧹 Clean Baseline" if self.current_level == 0 else f"📊 Level {self.current_level}"
                self.current_pbar = tqdm(
                    total=self.level_switch,
                    desc=f"Level {self.current_level} {level_desc}",
                    leave=False,
                    ncols=80,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )
            
            # Update progress bar
            current_progress = self.current_iteration % self.level_switch
            self.current_pbar.n = current_progress
            self.current_pbar.refresh()
            
            # Close progress bar when level is complete
            if current_progress == self.level_switch - 1:
                self.current_pbar.close()
                self.current_pbar = None
    
    def _reset_episode(self):
        """Reset episode tracking."""
        self.episode_step = 0
        self.episode_reward = 0.0
    
    def get_current_info(self):
        """Get current wrapper state information."""
        return {
            'current_iteration': self.current_iteration,
            'current_level': self.current_level,
            'current_perturbation': self.current_perturbation.copy(),
            'episodes_in_iteration': self.episodes_in_current_iteration,
            'total_env_steps': self.total_env_steps,
            'total_episodes': len(self.episode_returns)
        }

# ============================================================================
# UNIVERSAL ACTION SPACE WRAPPER
# ============================================================================

class UniversalActionWrapper(gym.Wrapper):
    """
    Wrapper to create universal action space (3 actions) and universal observation space (6 dimensions) for all tasks.
    Maps universal actions to task-specific actions using action masking.
    Pads observations to universal dimensions.
    """
    
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        
        # Universal action space: 4 actions for all tasks (LunarLander-v2 needs 4)
        self.action_space = gym.spaces.Discrete(4)
        
        # Universal observation space: 8 dimensions for all tasks (LunarLander-v2 needs 8)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(8,),  # Universal 8-dimensional observation space
            dtype=np.float32
        )
        
        # Task-specific action masks and mappings
        self.action_masks = {
            'CartPole-v1': [True, True, False, False],    # Actions 0,1 valid, 2,3 invalid
            'Acrobot-v1': [True, True, True, False],      # Actions 0,1,2 valid, 3 invalid
            'LunarLander-v2': [True, True, True, True]    # All 4 actions valid
        }
        
        # Action mappings for invalid actions (fallback to valid action)
        self.action_mappings = {
            'CartPole-v1': {2: 0, 3: 0},      # Map actions 2,3 to action 0
            'Acrobot-v1': {3: 0},             # Map action 3 to action 0
            'LunarLander-v2': {}              # No mapping needed (all valid)
        }
        
        self.current_mask = self.action_masks.get(task_name, [True, True, True])
        self.current_mapping = self.action_mappings.get(task_name, {})
    
    def step(self, action):
        """
        Map universal action to task-specific action and step the environment.
        Pad observations to universal dimensions.
        """
        # Convert numpy array to integer for dictionary lookup
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Map universal action to task-specific action
        if action in self.current_mapping:
            mapped_action = self.current_mapping[action]
        else:
            mapped_action = action
        
        # Step the environment with mapped action
        obs, reward, done, truncated, info = self.env.step(mapped_action)
        
        # Pad observation to universal dimensions
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to universal 8-dimensional space."""
        if isinstance(obs, np.ndarray):
            obs = obs.flatten()
        else:
            obs = np.array(obs).flatten()
        
        # Pad with zeros to reach 8 dimensions
        if len(obs) < 8:
            padded_obs = np.zeros(8, dtype=np.float32)
            padded_obs[:len(obs)] = obs
            return padded_obs
        elif len(obs) > 8:
            # Truncate to 8 dimensions
            return obs[:8].astype(np.float32)
        else:
            return obs.astype(np.float32)
    
    def reset(self, **kwargs):
        """Reset environment and pad observation."""
        obs, info = self.env.reset(**kwargs)
        padded_obs = self._pad_observation(obs)
        return padded_obs, info
    
    def get_action_mask(self):
        """Get action mask for current task."""
        return self.current_mask

class DebugTopologyPolicy(ActorCriticPolicy):
    """
    Debug Topology Policy for triple-task training with sweep support.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=128, num_layers=3, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Get hyperparameters from wandb config if available
        if config is None and wandb.run is not None:
            config = wandb.config
        
        # Extract parameters from config or use defaults
        self.topology_type = config.get('topology_type', topology_type) if config else topology_type
        self.hidden_size = config.get('hidden_size', hidden_size) if config else hidden_size
        self.num_layers = config.get('num_layers', num_layers) if config else num_layers
        self.activation = config.get('activation', 'relu') if config else 'relu'
        self.dropout = config.get('dropout', 0.0) if config else 0.0
        
        # Create topology networks for actor and critic
        self.actor_topology = self._create_topology_network('actor')
        self.critic_topology = self._create_topology_network('critic')
        
        # Debug network structure
        self._debug_network_structure()
    
    def _create_topology_network(self, network_type):
        """Create topology network based on type and parameters."""
        # 1. Define input/output nodes (task-specific, no fallbacks) - FIRST
        try:
            from stable_baselines3.common.preprocessing import get_flattened_obs_dim
            # Observation dim - use actual observation space
            if hasattr(self, 'observation_space') and self.observation_space is not None:
                input_dim = int(get_flattened_obs_dim(self.observation_space))
            else:
                raise ValueError("Observation space not available - cannot create network")
            
            # Action dim - use actual action space
            if hasattr(self, 'action_space') and self.action_space is not None:
                action_space = self.action_space
                if hasattr(action_space, 'n'):
                    output_dim = int(action_space.n)
                elif hasattr(action_space, 'shape') and len(action_space.shape) > 0:
                    output_dim = int(action_space.shape[0])
                else:
                    raise ValueError("Action space not properly configured - cannot create network")
            else:
                raise ValueError("Action space not available - cannot create network")
        except Exception as e:
            # No fallbacks - fail explicitly if we can't determine dimensions
            raise ValueError(f"Cannot determine network dimensions: {e}")
        
        # 2. Create topology object
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=self.hidden_size,  # Total network size (matches other topologies)
                seed=42  # For reproducibility
            )
        elif self.topology_type == 'standard_mlp':
            topology = StandardMLPTopology(
                size=self.hidden_size,
                num_layers=self.num_layers,  # MLP supports multiple layers
                activation=self.activation
            )
        elif self.topology_type == 'small_world':
            k = getattr(wandb.config, 'small_world_k', 4) if wandb.run else 4
            p = getattr(wandb.config, 'small_world_p', 0.2) if wandb.run else 0.2
            topology = SmallWorldTopology(
                size=self.hidden_size,
                k=k,
                p=p
            )
        elif self.topology_type == 'modular':
            num_modules = getattr(wandb.config, 'modular_num_modules', 4) if wandb.run else 4
            inter_prob = getattr(wandb.config, 'modular_inter_module_prob', 0.1) if wandb.run else 0.1
            intra_prob = getattr(wandb.config, 'modular_intra_module_prob', 0.8) if wandb.run else 0.8
            topology = ModularTopology(
                size=self.hidden_size,
                num_modules=num_modules,
                inter_module_prob=inter_prob,
                intra_module_prob=intra_prob
            )
        elif self.topology_type == 'hybrid':
            num_modules = getattr(wandb.config, 'hybrid_num_modules', 4) if wandb.run else 4
            k = getattr(wandb.config, 'hybrid_k', 4) if wandb.run else 4
            p = getattr(wandb.config, 'hybrid_p', 0.2) if wandb.run else 0.2
            inter_prob = getattr(wandb.config, 'hybrid_inter_module_prob', 0.1) if wandb.run else 0.1
            topology = HybridTopology(
                size=self.hidden_size,
                num_modules=num_modules,
                k=k,
                p=p,
                inter_module_prob=inter_prob
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # 3. Generate graph from topology with input/output dimensions
        graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
        
        input_nodes = list(range(input_dim))
        output_nodes = list(range(input_dim + self.hidden_size, input_dim + self.hidden_size + output_dim))
        
        # 4. Create FeedForwardNetwork
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        network = FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
        
        # 5. Return actual network (not topology object)
        return network
    
    def get_effective_num_layers(self):
        """Get the effective number of layers for the current topology."""
        if self.topology_type == 'standard_mlp':
            return self.num_layers  # MLP supports multiple layers
        else:
            return 1  # All other topologies always create single-layer networks
    
    def _get_topology_params(self, topology_network):
        """Get number of parameters in topology network."""
        total_params = 0
        try:
            # For FeedForwardNetwork, count weights and biases from node_states
            if hasattr(topology_network, 'node_states'):
                for node, state in topology_network.node_states.items():
                    # Count bias
                    if 'bias' in state:
                        total_params += 1
                    # Count weights
                    if 'weights' in state:
                        total_params += len(state['weights'])
            elif hasattr(topology_network, 'parameters'):
                # Fallback to PyTorch parameters
                for param in topology_network.parameters():
                    total_params += param.numel()
            else:
                # For topology objects, create actual networks to count real parameters
                if hasattr(topology_network, 'generate'):
                    # Generate the actual graph from topology
                    graph = topology_network.generate()
                    
                    # Define input/output nodes using TASK-SPECIFIC dimensions
                    try:
                        from stable_baselines3.common.preprocessing import get_flattened_obs_dim
                        # Observation dim
                        if hasattr(self, 'observation_space') and self.observation_space is not None:
                            input_dim = int(get_flattened_obs_dim(self.observation_space))
                        else:
                            input_dim = 6  # safe fallback
                        
                        # Action dim (Discrete or Box)
                        if hasattr(self, 'action_space') and self.action_space is not None:
                            action_space = self.action_space
                            if hasattr(action_space, 'n'):
                                output_dim = int(action_space.n)
                            elif hasattr(action_space, 'shape') and len(action_space.shape) > 0:
                                output_dim = int(action_space.shape[0])
                            else:
                                output_dim = 1
                        else:
                            output_dim = 3  # safe fallback
                    except Exception:
                        # As a last resort, keep previous safe defaults
                        input_dim = 6
                        output_dim = 3
                    
                    hidden_size = getattr(topology_network, 'size', 128)
                    
                    input_nodes = list(range(input_dim))
                    output_nodes = list(range(input_dim + hidden_size, input_dim + hidden_size + output_dim))
                    
                    # Create actual network to count real parameters
                    from src.networks.ffn import FeedForwardNetwork
                    network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
                    
                    # Generate extended graph with input/output dimensions for accurate parameter counting
                    extended_graph = topology_network.generate(input_dim=input_dim, output_dim=output_dim)
                    network = FeedForwardNetwork(extended_graph, input_nodes, output_nodes, network_params)
                    
                    # Now count actual parameters from the real network
                    if hasattr(network, 'node_states'):
                        for node, state in network.node_states.items():
                            # Count bias
                            if 'bias' in state:
                                total_params += 1
                            # Count weights
                            if 'weights' in state:
                                total_params += len(state['weights'])
                    else:
                        # Fallback to PyTorch parameters
                        for param in network.parameters():
                            total_params += param.numel()
                else:
                    # If we can't create a network, we can't count real parameters
                    print(f"       ⚠️  Cannot create network from topology for parameter counting")
                    total_params = 0
        except Exception as e:
            print(f"       ⚠️  Error counting parameters: {e}")
            total_params = 0
        return total_params
    
    def _debug_network_structure(self):
        """Debug and log network structure."""
        # Removed all W&B logging - keeping only standard training metrics
        pass
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create input mask matching current observation dimensionality."""
        # Mask all observed dimensions (no universal padding)
        return torch.ones_like(x, device=x.device)
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply input masking to observations."""
        return x * mask
    
    def forward_actor(self, obs):
        """Forward pass for actor network."""
        # Apply input masking
        mask = self._create_input_mask(obs)
        masked_obs = self._apply_input_masking(obs, mask)
        
        # Forward through topology network
        features = self.actor_topology(masked_obs)
        
        # Apply action masking for task-specific actions
        action_mask = self.get_action_mask()
        if action_mask is not None:
            # Create action mask tensor
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=obs.device)
            # Apply mask to logits (set invalid actions to -inf)
            features = torch.where(mask_tensor, features, torch.tensor(-1e8, device=obs.device))
        
        return features
    
    def forward_critic(self, obs):
        """Forward pass for critic network."""
        # Apply input masking
        mask = self._create_input_mask(obs)
        masked_obs = self._apply_input_masking(obs, mask)
        
        # Forward through topology network
        features = self.critic_topology(masked_obs)
        
        return features
    
    def get_action_mask(self):
        """Get action mask for current task."""
        # This will be set by the environment wrapper
        return None

# ============================================================================
# SIMPLIFIED LOGGING SYSTEM
# ============================================================================
# All complex logging methods have been removed.
# Only standard training metrics are logged via the SimplifiedCallback.

# ============================================================================
# IMPROVED LOGGING SYSTEM WITH REWARD SCALING
# ============================================================================

def initialize_wandb_run(config, topology_type, training_type='triple_task'):
    """Initialize wandb with placeholder naming that will be updated later."""
    
    # Create initial placeholder run name (will be updated with actual values)
    initial_run_name = create_initial_run_name(config, topology_type, training_type)
    
    # Create tags for easy filtering
    tags = create_run_tags(config, topology_type, training_type)
    
    # Initialize wandb
    wandb.init(
        project="topologies--triple-task-training",
        entity="katko-it-universitetet-i-k-benhavn",
        config=config,
        name=initial_run_name,
        tags=tags
    )

def create_initial_run_name(config, topology_type, training_type):
    """Create initial run name with available config values."""
    
    # Topology abbreviation
    topology_abbrev = {
        'small_world': 'SW',
        'modular': 'MOD', 
        'hybrid': 'HYB',
        'fully_connected': 'FC'
    }.get(topology_type, topology_type.upper())
    
    # Get initial size from config
    initial_size = config.get('hidden_size', 'unknown')
    
    # Task abbreviations
    task_abbrev = {
        'LunarLander-v2': 'LL',
        'Acrobot-v1': 'AC', 
        'CartPole-v1': 'CP',
        'MountainCar-v0': 'MC'
    }
    
    # Build initial name parts
    name_parts = [topology_abbrev]
    
    # Add placeholder capacity (will be updated later)
    name_parts.append("C?")
    
    # Add initial size
    name_parts.append(f"S{initial_size}")
    
    # Add task sequence (in correct order)
    if training_type == 'triple_task':
        task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
        tasks = task_order.split('_')
        task_abbrevs = [task_abbrev.get(task, task) for task in tasks]
        name_parts.append("-".join(task_abbrevs))
    
    return "_".join(name_parts)

def create_run_name(config, topology_type, training_type, model=None, total_params=None):
    """Create descriptive run name with TARGET capacity and ACTUAL size from the model."""
    
    # Topology abbreviation
    topology_abbrev = {
        'small_world': 'SW',
        'modular': 'MOD', 
        'hybrid': 'HYB',
        'fully_connected': 'FC'
    }.get(topology_type, topology_type.upper())
    
    # Use TARGET capacity for run name (as requested by user)
    target_capacity = '?'
    if 'target_capacity' in config:
        target_capacity = config['target_capacity']
    elif 'hidden_size' in config:
        # For fixed size runs, estimate capacity
        hidden_size = config['hidden_size']
        num_layers = config.get('num_layers', 3)
        # Rough estimate: input_size * hidden_size + hidden_size * hidden_size * (num_layers-1) + hidden_size * output_size
        # This is approximate and will be corrected with actual values later
        estimated_capacity = hidden_size * 64 + hidden_size * hidden_size * (num_layers - 1) + hidden_size * 64
        target_capacity = int(estimated_capacity)
    
    # Get ACTUAL size from the model for run name
    actual_size = 'unknown'
    if model is not None and hasattr(model, 'policy'):
        try:
            policy = model.policy
            
            # Get actual hidden size from the model
            if hasattr(policy, 'actor_topology') and hasattr(policy.actor_topology, 'hidden_size'):
                actual_size = policy.actor_topology.hidden_size
            elif hasattr(policy, 'critic_topology') and hasattr(policy.critic_topology, 'hidden_size'):
                actual_size = policy.critic_topology.hidden_size
                
        except Exception as e:
            print(f"   ⚠️  Could not get actual size from model: {e}")
            # Fallback to config values
            actual_size = config.get('hidden_size', 'unknown')
    
    # Task abbreviations
    task_abbrev = {
        'LunarLander-v2': 'LL',
        'Acrobot-v1': 'AC', 
        'CartPole-v1': 'CP',
        'MountainCar-v0': 'MC'
    }
    
    # Build name parts
    name_parts = [topology_abbrev]
    
    # Add TARGET capacity (as requested)
    name_parts.append(f"C{target_capacity}")
    
    # Add ACTUAL size
    name_parts.append(f"S{actual_size}")
    
    # Add task sequence (in correct order)
    if training_type == 'triple_task':
        task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
        tasks = task_order.split('_')
        task_abbrevs = [task_abbrev.get(task, task) for task in tasks]
        name_parts.append("-".join(task_abbrevs))
    
    return "_".join(name_parts)

def create_run_tags(config, topology_type, training_type, model=None, total_params=None):
    """Create enhanced tags for easy filtering and organization with actual capacity tracking."""
    
    # Primary tags
    tags = [
        topology_type,
        training_type,
        "normalized_metrics"
    ]
    
    # Capacity and size tags
    if 'target_capacity' in config:
        tags.extend([
            "fixed_capacity",
            f"target_capacity_{config.get('target_capacity')}",
            "capacity_matched"
        ])
        
        # Add actual capacity tag if available
        if model is not None and total_params is not None:
            tags.extend([
                f"actual_capacity_{total_params}",
                "capacity_achieved"
            ])
            
    elif 'hidden_size' in config:
        tags.extend([
            "fixed_size", 
            f"size_{config.get('hidden_size')}",
            "size_matched"
        ])
        
        # Add actual capacity tag if available
        if model is not None and total_params is not None:
            tags.extend([
                f"actual_capacity_{total_params}",
                "capacity_achieved"
            ])
    
    # Task tags
    if training_type == 'triple_task':
        task_order = config.get('task_order', 'LunarLander-v2_Acrobot-v1_CartPole-v1')
        tasks = task_order.split('_')
        tags.extend(tasks)
        
        # Add task order tag for easy filtering
        tags.append(f"order_{task_order.replace('_', '-')}")
    
    # Add sweep type tag
    if 'target_capacity' in config:
        tags.append("sweep_fixed_capacity")
    else:
        tags.append("sweep_fixed_size")
    
    return tags

def create_continual_learning_run_name(config, topology_type, task_name, seed, model=None, shift_range=None):
    """
    Create enhanced run name for continual learning experiments.
    
    Format: {topology_type}_{network_details}_{task_abbrev}_{seed}_{experiment_details}
    Example: SW_L3_S128_P12345_CP_seed42_L15_I3000_LS200_N02
    """
    # Topology abbreviation
    topology_abbrev = {
        'small_world': 'SW',
        'modular': 'MOD', 
        'hybrid': 'HYB',
        'fully_connected': 'FC'
    }.get(topology_type, topology_type.upper())
    
    # Get actual topology parameters for more detailed naming
    topology_params = ''
    if model is not None and hasattr(model, 'policy'):
        try:
            policy = model.policy
            if topology_type == 'small_world':
                k = config.get('small_world_k', 4)
                p = config.get('small_world_p', 0.2)
                topology_params = f"_k{k}_p{p:.1f}"
            elif topology_type == 'modular':
                num_modules = config.get('modular_num_modules', 4)
                inter_prob = config.get('modular_inter_module_prob', 0.1)
                intra_prob = config.get('modular_intra_module_prob', 0.8)
                topology_params = f"_m{num_modules}_i{inter_prob:.1f}_a{intra_prob:.1f}"
            elif topology_type == 'hybrid':
                num_modules = config.get('hybrid_num_modules', 4)
                k = config.get('hybrid_k', 4)
                p = config.get('hybrid_p', 0.2)
                inter_prob = config.get('hybrid_inter_module_prob', 0.1)
                topology_params = f"_m{num_modules}_k{k}_p{p:.1f}_i{inter_prob:.1f}"
        except Exception:
            topology_params = ''
    
    # Get actual parameter count from the model
    total_params = 0
    if model is not None and hasattr(model, 'policy'):
        try:
            policy = model.policy
            # Extract actual topology parameters using the policy's method
            if hasattr(policy, '_get_topology_params'):
                actor_params = policy._get_topology_params(policy.actor_topology)
                critic_params = policy._get_topology_params(policy.critic_topology)
                total_params = actor_params + critic_params
            else:
                # Fallback to PyTorch parameter counting
                total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        except Exception:
            # Fallback to estimated parameters
            hidden_size = config.get('hidden_size', 128)
            num_layers = config.get('num_layers', 1)
            estimated_params = hidden_size * 64 + hidden_size * hidden_size * (num_layers - 1) + hidden_size * 64
            total_params = int(estimated_params)
    
    # Get network details
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 1)
    
    # Task abbreviation
    task_abbrev = {
        'LunarLander-v2': 'LL',
        'Acrobot-v1': 'AC', 
        'CartPole-v1': 'CP',
        'MountainCar-v0': 'MC'
    }.get(task_name, task_name[:2].upper())
    
    # Experiment details
    num_levels = config.get('max_iterations', 3000) // config.get('level_switch', 200)
    max_iterations = config.get('max_iterations', 3000)
    level_switch = config.get('level_switch', 200)
    
    # Noise interval from shift_range
    noise_interval = 'N00'
    if shift_range and len(shift_range) == 2:
        noise_interval = f"N{int(shift_range[0]):02d}{int(shift_range[1]):02d}"
    
    # Build name parts
    name_parts = [
        topology_abbrev,
        f"L{num_layers}",
        f"S{hidden_size}",
        f"P{total_params}",
        task_abbrev,
        f"seed{seed}",
        f"L{num_levels}",
        f"I{max_iterations}",
        f"LS{level_switch}",
        noise_interval
    ]
    
    # Add topology-specific parameters if available
    if topology_params:
        name_parts.append(topology_params.lstrip('_'))  # Remove leading underscore
    
    return "_".join(name_parts)

def log_baseline_results(wandb_run, baseline_results, topology_type):
    """Log baseline evaluation results with streamlined structure."""
    
    for task, results in baseline_results.items():
        # Streamlined baseline metrics
        wandb_run.log({
            f'baseline/{task}/mean_reward': results['mean_reward'],
            f'baseline/{task}/success_rate': results['success_rate'],
            f'baseline/{task}/mean_length': np.mean(results['lengths']),
            f'baseline/{task}/std_reward': np.std(results['rewards']),
            f'baseline/{task}/std_length': np.std(results['lengths'])
        })

def log_phase_results(wandb_run, phase_results, phase_idx, topology_type, task_order=None):
    """Log results after each training phase with streamlined structure."""
    
    for task, results in phase_results.items():
        # Streamlined phase metrics
        base_path = f"phase_results/{topology_type}"
        if task_order:
            base_path += f"/{task_order}"
        
        wandb_run.log({
            f'{base_path}/phase_{phase_idx}_{task}/mean_reward': results['mean_reward'],
            f'{base_path}/phase_{phase_idx}_{task}/success_rate': results['success_rate'],
            f'{base_path}/phase_{phase_idx}_{task}/mean_length': np.mean(results['lengths']),
            f'{base_path}/phase_{phase_idx}_{task}/std_reward': np.std(results['rewards']),
            f'{base_path}/phase_{phase_idx}_{task}/std_length': np.std(results['lengths'])
        })

def log_normalized_metrics(wandb_run, task_metrics, phase_idx, topology_type, task_order=None):
    """Log normalized metrics with streamlined structure."""
    
    base_path = f"normalized_metrics/{topology_type}"
    if task_order:
        base_path += f"/{task_order}"
    
    # Task-specific normalized metrics
    for task, metrics in task_metrics.items():
        wandb_run.log({
            f'{base_path}/phase_{phase_idx}_{task}/normalized_reward': metrics['normalized_reward'],
            f'{base_path}/phase_{phase_idx}_{task}/steps_to_threshold': metrics['steps_to_threshold'],
            f'{base_path}/phase_{phase_idx}_{task}/final_reward': metrics['final_reward']
        })
    
    # Aggregated normalized metrics
    final_normalized_score = np.mean([metrics['normalized_reward'] for metrics in task_metrics.values()])
    efficiency_score = np.mean([metrics['steps_to_threshold'] for metrics in task_metrics.values()])
    
    wandb_run.log({
        f'{base_path}/phase_{phase_idx}/final_normalized_score': final_normalized_score,
        f'{base_path}/phase_{phase_idx}/efficiency_score': efficiency_score
    })

def log_transfer_metrics(wandb_run, transfer_metrics, phase_idx, topology_type, task_order=None):
    """Log transfer learning metrics with streamlined structure."""
    
    base_path = f"transfer_metrics/{topology_type}"
    if task_order:
        base_path += f"/{task_order}"
    
    # Log all transfer metrics
    for metric_name, value in transfer_metrics.items():
        wandb_run.log({
            f'{base_path}/phase_{phase_idx}/{metric_name}': value
        })

def log_final_analysis(wandb_run, final_analysis, topology_type, task_order=None):
    """Log final analysis with streamlined structure."""
    
    base_path = f"final_analysis/{topology_type}"
    if task_order:
        base_path += f"/{task_order}"
    
    # Log all final analysis metrics
    for metric_name, value in final_analysis.items():
        wandb_run.log({
            f'{base_path}/{metric_name}': value
        })

# ============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ============================================================================

def create_debug_config(num_levels=15, num_layers=1):
    """Create a basic configuration for testing and debugging (Paper-Accurate)."""
    return {
        'max_iterations': num_levels * 200,  # Total iterations = num_levels × 200
        'level_switch': 200,           # Switch perturbation every 200 iterations
        'shift_range': [0, 2],        # Uniform[0, 2] per dimension (increased from [0, 2])
        'episode_cap': 400,            # Max episode length
        'reward_scale': 20.0,          # Division factor (creates small gradients)
        'n_steps': 800,                # PPO rollout size
        'n_epochs': 5,                 # PPO training epochs
        'batch_size': 32,              # PPO batch size
        'learning_rate': 0.01,       # PPO learning rate (paper-aligned)
        'gamma': 0.99,                 # PPO gamma
        'gae_lambda': 0.95,            # PPO GAE lambda
        'clip_range': 0.2,             # PPO clip range
        'ent_coef': 0.01,             # PPO entropy coefficient
        'max_grad_norm': 0.5,          # PPO max gradient norm
        'num_layers': num_layers       # Number of layers for topology networks
    }

# 🚨 CONVENIENT TRAINING FUNCTIONS: Using unified configuration system
def run_single_training(config_name='single', topology_type=None, seed=42, task_name=None, **overrides):
    """
    Run a single training session using unified configuration.
    
    Args:
        config_name (str): 'single', 'batch', or 'fixed_capacity_batch'
        topology_type (str): Override topology type if specified
        seed (int): Random seed for reproducibility
        task_name (str): Task to train on (for continual learning)
        **overrides: Additional parameter overrides
    """
    if not CONFIG_SYSTEM_AVAILABLE:
        print("❌ Configuration system not available. Using debug config.")
        config = create_debug_config()
        if topology_type:
            config['topology_type'] = topology_type
        if task_name:
            config['task_name'] = task_name
            # Force continual learning mode when task is specified
            config['continual_learning'] = True
        config.update(overrides)
        
        # Add seed to config for proper run naming
        config['seed'] = seed
        
        # Choose training function based on config
        if config.get('continual_learning', False):
            return continual_learning_training(DebugTopologyPolicy, config['topology_type'], config, seed=seed, task_name=task_name)
        else:
        return triple_task_training(DebugTopologyPolicy, config['topology_type'], config, seed=seed)
    
    print(f"🚀 Starting single training with config: {config_name}")
    
    # Get configuration
    config = get_config_by_name(config_name)
    
    # Apply overrides
    if topology_type:
        config['topology_type'] = topology_type
    if task_name:
        config['task_name'] = task_name
        # Force continual learning mode when task is specified
        config['continual_learning'] = True
    config.update(overrides)
    
    # Add seed to config for proper run naming
    config['seed'] = seed
    
    print(f"📋 Configuration: {config}")
    
    # Choose training function based on config
    if config.get('continual_learning', False):
        return continual_learning_training(DebugTopologyPolicy, config['topology_type'], config, seed=seed, task_name=task_name)
    else:
    return triple_task_training(DebugTopologyPolicy, config['topology_type'], config, seed=seed)

def run_batch_training(config_name='batch', max_runs=None, **overrides):
    """
    Run batch training with multiple parameter combinations.
    
    Args:
        config_name (str): 'batch' or 'fixed_capacity_batch'
        max_runs (int): Maximum number of runs to execute
        **overrides: Additional parameter overrides
    """

    if not CONFIG_SYSTEM_AVAILABLE:
        print("❌ Configuration system not available. Cannot run batch training.")
        return
    
    print(f"🚀 Starting batch training with config: {config_name}")
    
    # Get batch configuration
    batch_config = get_config_by_name(config_name)
    
    # Apply overrides
    batch_config.update(overrides)
    
    # Generate all combinations
    combinations = generate_parameter_combinations(batch_config)
    
    if max_runs:
        combinations = combinations[:max_runs]
        print(f"📊 Limited to {max_runs} runs out of {len(combinations)} total combinations")
    
    print(f"📊 Total combinations to run: {len(combinations)}")
    
    # Run each combination
    results = []
    for i, combo_config in enumerate(combinations):
        print(f"\n🔄 Running combination {i+1}/{len(combinations)}")
        print(f"   Topology: {combo_config['topology_type']}")
        print(f"   Hidden size: {combo_config.get('hidden_size', 'N/A')}")
        print(f"   Task order: {combo_config.get('task_order', 'N/A')}")
        print(f"   Seed: {combo_config.get('seed', 'N/A')}")
        
        try:
            # Ensure seed is in the config for proper run naming
            combo_config['seed'] = combo_config.get('seed', 42)
            result = triple_task_training(DebugTopologyPolicy, combo_config['topology_type'], combo_config, seed=combo_config['seed'])
            results.append(result)
            print(f"✅ Combination {i+1} completed successfully")
        except Exception as e:
            print(f"❌ Combination {i+1} failed: {e}")
            results.append(None)
    
    print(f"\n🎯 Batch training completed: {len([r for r in results if r is not None])}/{len(combinations)} successful")
    return results

def run_sweep_training(sweep_type='fixed_network_sizes'):
    """
    Launch W&B sweep training.
    
    Args:
        sweep_type (str): 'fixed_network_sizes', 'fixed_capacities', or 'continual_learning'
    """
    if not CONFIG_SYSTEM_AVAILABLE:
        print("❌ Configuration system not available. Cannot launch sweep.")
        return
    
    print(f"🚀 Launching W&B sweep: {sweep_type}")
    
    try:
        if sweep_type == 'fixed_network_sizes':
            from wandb_sweep_config import create_fixed_network_sizes_triple_task_sweep
            sweep_config = create_fixed_network_sizes_triple_task_sweep()
        elif sweep_type == 'fixed_capacities':
            from wandb_sweep_config import create_fixed_capacities_triple_task_sweep
            sweep_config = create_fixed_capacities_triple_task_sweep()
        elif sweep_type == 'continual_learning':
            from wandb_sweep_config import create_continual_learning_sweep
            sweep_config = create_continual_learning_sweep()
        else:
            print(f"❌ Unknown sweep type: {sweep_type}")
            return
        
        # Launch sweep
        import wandb
        sweep_id = wandb.sweep(sweep_config, project='topology-research')
        print(f"✅ Sweep launched with ID: {sweep_id}")
        print(f"🔗 View at: https://wandb.ai/sweeps/{sweep_id}")
        
        return sweep_id
        
    except Exception as e:
        print(f"❌ Failed to launch sweep: {e}")
        return None

def make_env(env_name, seed=None, continual_learning=False, max_iterations=3000, level_switch=200, shift_range=[0, 2], reward_scale=20.0, episode_cap=400, logging_callback=None, num_levels=15):
    """Create an environment with optional continual learning wrapper."""
    def _make_env():
            env = gym.make(env_name)
        
        # Set seed for reproducibility using modern Gymnasium API
        if seed is not None:
            # Use the modern reset(seed=seed) method
            env.reset(seed=seed)
            # Seed action and observation spaces
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
        # Apply continual learning wrapper if requested
        if continual_learning:
            env = ContinualLearningWrapper(env, env_name, max_iterations, level_switch, shift_range, seed, reward_scale, episode_cap, logging_callback, num_levels)
        
        return env
    
    return _make_env

def evaluate_model(model, env, n_eval_episodes=3):
    """Evaluate model on environment."""
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_eval_episodes):
        # Handle different reset return signatures
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            if len(reset_result) == 2:
                obs, _ = reset_result
            else:
                obs = reset_result[0]
        else:
            obs = reset_result
        
        done = False
        total_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Handle different step return signatures
            step_result = env.step(action)
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, reward, done, truncated, _ = step_result
                elif len(step_result) == 4:
                    obs, reward, done, _ = step_result
                    truncated = False
                else:
                    obs, reward, done = step_result
                    truncated = False
            else:
                # Handle case where step returns a single value
                obs, reward, done = step_result, 0, True
                truncated = False
            
            total_reward += reward
            episode_length += 1
            done = done or truncated
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths

def evaluate_model_enhanced(model, env, task_name, n_eval_episodes=3):
    """Enhanced evaluation with task-specific metrics."""
    episode_rewards, episode_lengths = evaluate_model(model, env, n_eval_episodes)
    
    # Calculate both success rate and completion percentage
    success_rate, completion_pct = calculate_success_rate_with_completion(episode_rewards, episode_lengths, task_name)
    
    # Evaluation metrics logging removed - keeping only standard training metrics
    pass
    
    return episode_rewards, episode_lengths, success_rate, completion_pct

def calculate_success_rate(rewards, episode_lengths, task_name):
    """Calculate success rate based on task-specific criteria."""
    if task_name == 'CartPole-v1':
        # Success: reward >= 500 (actual solved threshold) - consistent with completion
        return np.mean([reward >= 500 for reward in rewards])
    elif task_name == 'Acrobot-v1':
        # Success: reward >= -80 (actual solved threshold)
        return np.mean([reward >= -80 for reward in rewards])
    elif task_name == 'LunarLander-v2':  # Replace MountainCar-v0
        # Success: reward >= 200 (actual solved threshold)
        return np.mean([reward >= 200 for reward in rewards])
    else:
        # Default: above average performance
        mean_reward = np.mean(rewards)
        return np.mean([reward >= mean_reward for reward in rewards])


def calculate_success_rate_with_completion(rewards, episode_lengths, task_name):
    """Calculate success rate with completion percentage for different tasks."""
    if not rewards:
        return 0.0, 0.0
    
    # Task-specific thresholds
    thresholds = {
        'CartPole-v1': 195.0,  # CartPole solved threshold
        'Acrobot-v1': -100.0,  # Acrobot solved threshold (negative reward)
        'LunarLander-v2': 200.0,  # LunarLander solved threshold
        'MountainCar-v0': -110.0,  # MountainCar solved threshold
    }
    
    # Default threshold if task not found
    threshold = thresholds.get(task_name, 0.0)
    
    # Calculate success rate
    successful_episodes = sum(1 for r in rewards if r >= threshold)
    success_rate = successful_episodes / len(rewards)
    
    # Calculate completion percentage (how close to optimal performance)
    if task_name == 'CartPole-v1':
        # CartPole: 500 is max, 195 is solved, 0 is worst
        max_reward = 500.0
        min_reward = 0.0
        completion_pct = np.mean([(r - min_reward) / (max_reward - min_reward) for r in rewards])
    elif task_name == 'Acrobot-v1':
        # Acrobot: 0 is best, -500 is worst, -100 is solved
        max_reward = 0.0
        min_reward = -500.0
        completion_pct = np.mean([(r - min_reward) / (max_reward - min_reward) for r in rewards])
    elif task_name == 'LunarLander-v2':
        # LunarLander: 250+ is excellent, 200 is solved, -1000 is worst
        max_reward = 250.0
        min_reward = -1000.0
        completion_pct = np.mean([(r - min_reward) / (max_reward - min_reward) for r in rewards])
    elif task_name == 'MountainCar-v0':
        # MountainCar: -110 is solved, -200 is worst
        max_reward = -110.0
        min_reward = -200.0
        completion_pct = np.mean([(r - min_reward) / (max_reward - min_reward) for r in rewards])
    else:
        # Default: normalize to [0, 1] based on observed range
        min_obs, max_obs = min(rewards), max(rewards)
        if max_obs > min_obs:
            completion_pct = np.mean([(r - min_obs) / (max_obs - min_obs) for r in rewards])
        else:
            completion_pct = 0.5  # Default to 50% if no variation
    
    return success_rate, completion_pct

def calculate_reward_completion_percentage(rewards, task_name):
    """Calculate completion percentage for reward-based tasks."""
    if not rewards:
        return 0.0
    
    # Task-specific completion calculation
    if task_name == 'CartPole-v1':
        # CartPole: 500 is max (100%), 195 is solved (39%), 0 is worst (0%)
        # Use 195 as the "solved" threshold for meaningful completion
        solved_threshold = 195.0
        max_reward = 500.0
        min_reward = 0.0
        
        # Calculate completion relative to solved threshold
        completion_pcts = []
        for r in rewards:
            if r >= solved_threshold:
                # Above solved threshold: 100% completion
                completion_pcts.append(100.0)
            else:
                # Below solved threshold: linear scale from 0% to 100%
                completion_pct = max(0.0, (r - min_reward) / (solved_threshold - min_reward) * 100.0)
                completion_pcts.append(completion_pct)
        
        completion_pct = np.mean(completion_pcts)
        
    elif task_name == 'Acrobot-v1':
        # Acrobot: 0 is best (100%), -100 is solved (80%), -500 is worst (0%)
        # Use -100 as the "solved" threshold for meaningful completion
        solved_threshold = -100.0
        max_reward = 0.0
        min_reward = -500.0
        
        # Calculate completion relative to solved threshold
        completion_pcts = []
        for r in rewards:
            if r >= solved_threshold:
                # Above solved threshold: 80% to 100% completion
                completion_pct = 80.0 + (r - solved_threshold) / (max_reward - solved_threshold) * 20.0
                completion_pcts.append(completion_pct)
            else:
                # Below solved threshold: linear scale from 0% to 80%
                completion_pct = max(0.0, (r - min_reward) / (solved_threshold - min_reward) * 80.0)
                completion_pcts.append(completion_pct)
        
        completion_pct = np.mean(completion_pcts)
        
    elif task_name == 'LunarLander-v2':
        # LunarLander: 250+ is excellent (100%), 200 is solved (80%), -1000 is worst (0%)
        # Use 200 as the "solved" threshold for meaningful completion
        solved_threshold = 200.0
        max_reward = 250.0
        min_reward = -1000.0
        
        # Calculate completion relative to solved threshold
        completion_pcts = []
        for r in rewards:
            if r >= solved_threshold:
                # Above solved threshold: 80% to 100% completion
                completion_pct = 80.0 + (r - solved_threshold) / (max_reward - solved_threshold) * 20.0
                completion_pcts.append(completion_pct)
            else:
                # Below solved threshold: linear scale from 0% to 80%
                completion_pct = max(0.0, (r - min_reward) / (solved_threshold - min_reward) * 80.0)
                completion_pcts.append(completion_pct)
        
        completion_pct = np.mean(completion_pcts)
        
    elif task_name == 'MountainCar-v0':
        # MountainCar: -110 is solved (100%), -200 is worst (0%)
        solved_threshold = -110.0
        max_reward = -110.0
        min_reward = -200.0
        
        # Calculate completion relative to solved threshold
        completion_pcts = []
        for r in rewards:
            if r >= solved_threshold:
                # Above solved threshold: 100% completion
                completion_pcts.append(100.0)
            else:
                # Below solved threshold: linear scale from 0% to 100%
                completion_pct = max(0.0, (r - min_reward) / (solved_threshold - min_reward) * 100.0)
                completion_pcts.append(completion_pct)
        
        completion_pct = np.mean(completion_pcts)
        
    else:
        # Default: normalize to [0, 100] based on observed range
        min_obs, max_obs = min(rewards), max(rewards)
        if max_obs > min_obs:
            completion_pct = np.mean([(r - min_obs) / (max_obs - min_obs) * 100.0 for r in rewards])
        else:
            completion_pct = 50.0  # Default to 50% if no variation
    
    return completion_pct

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def triple_task_training(policy_class, topology_type, config, seed=42, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None, train_task_3=None):
    """
    Triple-task training function with intermediate testing after each phase.
    
    Sequential training: Train on task 1, test on all tasks, then train on task 2, test on all tasks, then train on task 3, test on all tasks.
    
    Args:
        policy_class: Policy class to use
        topology_type: Type of topology network
        config: Configuration dictionary
        seed (int): Random seed for reproducibility
        num_layers: Number of layers
        hidden_size: Hidden layer size
        train_task_1: First training task
        train_task_2: Second training task
        train_task_3: Third training task
    """
    # 🚨 CRITICAL: Import os at the top of the function
    import os
    
    # 🚨 CRITICAL: Extract task information from config if not provided
    if train_task_1 is None:
        train_task_1 = config.get('train_task_1', 'CartPole-v1')
    if train_task_2 is None:
        train_task_2 = config.get('train_task_2', 'Acrobot-v1')
    if train_task_3 is None:
        train_task_3 = config.get('train_task_3', 'LunarLander-v2')
    
    # 🚨 CRITICAL: Extract topology parameters from config if not provided
    if hidden_size is None:
        hidden_size = config.get('hidden_size', 128)
    if num_layers is None:
        num_layers = config.get('num_layers', 2)
    
    print("=" * 80)
    print(f"🎯 TRIPLE-TASK SEQUENTIAL TRAINING: {topology_type.upper()} TOPOLOGY")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   • Task Sequence: {train_task_1} → {train_task_2} → {train_task_3}")
    print(f"   • Topology Type: {topology_type}")
    print(f"   • Hidden Size: {hidden_size}")
    print(f"   • Layers: {num_layers}")
    print(f"   • Total Timesteps per Phase: {config['total_timesteps']:,}")
    print(f"   • Learning Rate: {config['learning_rate']}")
    print(f"   • Batch Size: {config['batch_size']}")
    print(f"🔧 Device: {DEVICE_INFO['device']}")
    if DEVICE_INFO['is_cuda']:
        print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
        print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
    print("=" * 80)
    
    # 🚨 CRITICAL: Apply comprehensive seeding for reproducibility
    print(f"🎲 Setting random seed: {seed}")
    
    # PyTorch seeding
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # NumPy seeding
    np.random.seed(seed)
    
    # Python random seeding
    import random
    random.seed(seed)
    
    # Gym seeding
    try:
        gym.utils.seeding.np_random(seed)
    except:
        pass  # Some gym versions don't have this function
    
    print(f"✅ All random states seeded with seed: {seed}")
    print("=" * 80)
    
    # Initialize wandb if not already done
    if wandb.run is None:
        # Create proper naming for this specific training run
        run_name = create_run_name(config, topology_type, 'triple_task')
        tags = create_run_tags(config, topology_type, 'triple_task')
        
        wandb.init(
            project="topologies--triple-task-training",
            entity="katko-it-universitetet-i-k-benhavn",
            config=config,
            name=run_name,
            tags=tags,
            mode="disabled" if os.environ.get('WANDB_DISABLE', 'false').lower() == 'true' else "online"
            # Removed silent=True to allow legitimate training metrics
        )
        
        # 🚨 CRITICAL FIX: Ensure W&B's internal step counter never starts at 0
        # This prevents the "step 0" warnings by ensuring W&B starts at step 1
        if wandb.run and hasattr(wandb.run, 'step'):
            # Force W&B to start at step 1 instead of 0
            try:
                # Initialization logging removed - keeping only standard training metrics
                print("🔧 W&B Step Fix: Initialized internal step counter at step 1")
            except Exception as e:
                print(f"⚠️  W&B Step Fix: Could not initialize step counter: {e}")
    
    # 🚨 CRITICAL FIX: Ensure W&B's step counter is always > 0
    # This prevents step 0 warnings throughout training
    if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step <= 0:
        try:
            # Step reset logging removed - keeping only standard training metrics
            print("🔧 W&B Step Fix: Reset internal step counter from 0 to 1")
        except Exception as e:
            print(f"⚠️  W&B Step Fix: Could not reset step counter: {e}")
    
    # Calculate timesteps for each phase
    task1_timesteps = config['total_timesteps'] // 3  # 200,000 timesteps per phase
    task2_timesteps = config['total_timesteps'] // 3  # 200,000 timesteps per phase  
    task3_timesteps = config['total_timesteps'] // 3  # 200,000 timesteps per phase
    
    # Create task order string for topology-aware logging
    task_order = f"{train_task_1}_{train_task_2}_{train_task_3}"
    
    # Create environments for sequential training
    env1 = DummyVecEnv([make_env(train_task_1, seed=seed)])
    env2 = DummyVecEnv([make_env(train_task_2, seed=seed)])
    env3 = DummyVecEnv([make_env(train_task_3, seed=seed)])
    
    # Create ONE model for sequential training
    # 🚨 CRITICAL: Disable ALL internal SB3 logging to ensure clean output
    os.environ['SB3_VERBOSE'] = '0'  # Force disable SB3 verbose logging
    
    model = PPO(
        policy_class,
        env1,  # Start with first task environment
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=0,  # 🚨 CRITICAL: Disable internal Stable-Baselines3 logging
        tensorboard_log=None,  # Disable tensorboard logging
        policy_kwargs={
            'topology_type': topology_type,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'config': config
        }
    )
    
    # Create logging handler and callback FIRST
    logging_handler = create_logging_handler(config, topology_type, 'triple_task')
    logging_handler.initialize_run()
    
    # Create callback using the simplified logging handler
    callback = SimplifiedCallback(logging_handler=logging_handler, log_freq=1000)
    
    # Calculate actual capacity and update run name if needed
    if wandb.run is not None:
        try:
            # Calculate actual capacity from the policy using PyTorch's built-in method
            # This automatically counts total parameters (actor + critic) from the actual model
            policy = model.policy
            total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            
            # Get individual network info for logging (optional, for debugging)
            actor_params = policy._get_topology_params(policy.actor_topology)
            critic_params = policy._get_topology_params(policy.critic_topology)
            
            # Update run name using the logging handler
            updated_run_name = logging_handler.update_run_name(model, total_params)
            
            # 🚨 CRITICAL: Don't log network info at step 0 - wait until after first training step
            # This prevents W&B step warnings about logging to step 0
            print(f"📊 Actual network capacity: {total_params:,} parameters")
            
        except Exception as e:
            print(f"   ⚠️  Could not calculate actual capacity: {e}")
    
    # ============================================================================
    # PHASE 1: Train on task 1
    # ============================================================================
    callback.set_task_phase(train_task_1, 1)  # Set phase 1
    
    # Get task-specific training configuration
    task1_timesteps = get_task_timesteps(train_task_1, config)
    convergence_callback = create_convergence_callback(train_task_1, config, verbose=0)  # Disable verbose output
    
    print(f"📋 Task-specific training: {train_task_1} for {task1_timesteps:,} timesteps")
    
    # Create a callback to update our progress bar
    class ProgressBarCallback(BaseCallback):
        def __init__(self, total_timesteps, task_name):
            super().__init__()
            self.total_timesteps = total_timesteps
            self.task_name = task_name
            self.progress_bar = None
            self.last_update = 0
        
        def _on_training_start(self) -> None:
            # Create a fresh progress bar for each training phase
            self.progress_bar = tqdm(
                total=self.total_timesteps, 
                desc=f"Training {self.task_name}", 
                unit="steps", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        def _on_step(self) -> bool:
            # Update progress bar with current training progress
            if self.progress_bar:
                current_step = self.num_timesteps
                if current_step > self.last_update:
                    self.progress_bar.update(current_step - self.last_update)
                    self.last_update = current_step
            return True
        
        def _on_training_end(self) -> None:
            # Ensure progress bar is complete and closed
            if self.progress_bar:
                self.progress_bar.close()
                self.progress_bar = None
    
    # Create a callback to monitor training rewards in real-time (silent version)
    class RewardMonitorCallback(BaseCallback):
        def __init__(self, task_name, log_interval=10000):
            super().__init__()
            self.task_name = task_name
            self.log_interval = log_interval
            self.last_log_step = 0
            self.episode_rewards = []
            self.episode_lengths = []
        
        def _on_step(self) -> bool:
            # Collect episode rewards from the environment
            if hasattr(self.training_env, 'get_episode_rewards'):
                rewards = self.training_env.get_episode_rewards()
                if rewards:
                    self.episode_rewards.extend(rewards)
            
            # Log rewards periodically (silent - only to W&B)
            if self.num_timesteps - self.last_log_step >= self.log_interval:
                self.last_log_step = self.num_timesteps
                # Silent - no terminal output
            
            return True
    
    # Create a callback that integrates convergence monitoring with periodic evaluation (silent version)
    class ConvergenceEvaluationCallback(BaseCallback):
        def __init__(self, convergence_callback, model, env, task_name, eval_interval=20000):
            super().__init__()
            self.convergence_callback = convergence_callback
            self.model = model
            self.env = env
            self.task_name = task_name
            self.eval_interval = eval_interval
            self.last_eval_step = 0
        
        def _on_step(self) -> bool:
            # Check if we should do a quick evaluation
            if self.num_timesteps - self.last_eval_step >= self.eval_interval:
                self.last_eval_step = self.num_timesteps
                
                # Quick evaluation to check convergence
                try:
                    rewards, lengths, success, completion = evaluate_model_enhanced(
                        self.model, self.env, self.task_name, 5  # Quick eval with 5 episodes
                    )
                    mean_reward = np.mean(rewards)
                    
                    # Update convergence callback with evaluation results
                    self.convergence_callback.update_with_evaluation(mean_reward, success)
                    
                    # Silent - no terminal output, only W&B logging
                    
                    # Check convergence status (silent)
                    if self.convergence_callback.should_stop:
                        pass  # Silent early stopping
                
                except Exception as e:
                    # Silent error handling
                    pass
            
            return True
    
    # RewardTrackingCallback class definition moved to continual_learning_training function
    
    # Combine all callbacks
    combined_callback = CallbackList([
        callback,  # Main logging callback
        ProgressBarCallback(task1_timesteps, train_task_1),  # Progress bar
        RewardMonitorCallback(train_task_1, log_interval=10000),  # Reward monitoring
        ConvergenceEvaluationCallback(convergence_callback, model, env1, train_task_1, eval_interval=20000),  # Convergence evaluation
        RewardTrackingCallback(train_task_1, log_frequency=5)  # Fine-grained reward tracking
    ])
    # Phase 1
    # 🚨 CRITICAL: Disable SB3's internal progress bar to prevent conflicts with our TQDM bar
    model.learn(total_timesteps=task1_timesteps, callback=combined_callback, progress_bar=False)
    
    # 🚨 CREATIVE FIX: Use W&B's internal step reset mechanism
    # Since we can't directly set wandb.run.step, we use alternative methods
    if wandb.run:
        print("🔄 W&B step counter reset not needed - using environment variables")
    
    # 🚨 CREATIVE FIX: W&B-Native Step Management
    # Instead of fighting W&B's step counter, we work with it
    # This eliminates step warnings while maintaining proper timestep tracking
    if wandb.run:
        print("🔄 Using W&B-native step management for seamless synchronization")
        # Let W&B manage its own step counter naturally
        # We'll use wandb.run.step for all logging to ensure alignment
    
    # 🚨 CRITICAL: Update global timesteps after Phase 1 training
    # This ensures continuous progression across tasks for learning curves
    actual_task1_duration = model.num_timesteps if hasattr(model, 'num_timesteps') else task1_timesteps
    logging_handler.update_global_timesteps(actual_task1_duration)
    
    # 🚨 CRITICAL: Evaluate on all tasks after Phase 1 training
    print("\n" + "=" * 80)
    print("🔍 PHASE 1 EVALUATION: Testing performance on all tasks after CartPole-v1 training")
    print("=" * 80)
    
    # Evaluate on all tasks to see transfer learning effects
    for task_name in [train_task_1, train_task_2, train_task_3]:
        print(f"\n📊 Evaluating on {task_name} after CartPole-v1 training...")
        try:
            # Create evaluation environment
            eval_env = DummyVecEnv([make_env(task_name, seed=seed)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Evaluation logging removed - keeping only standard training metrics
            pass
            
            print(f"   ✅ {task_name}: Reward={np.mean(rewards):.2f}, Success={success:.2%}, Completion={completion:.2%}")
            
        except Exception as e:
            print(f"   ❌ {task_name} evaluation failed: {e}")
    
    # 🚨 CRITICAL: Phase 2 - Train on Acrobot-v1
    print("\n" + "=" * 80)
    print("🎯 PHASE 2: Training on Acrobot-v1")
    print("=" * 80)
    
    # Switch to Acrobot environment for Phase 2
    model.set_env(env2)
    logging_handler.set_task_phase(train_task_2, 2)
    
    # Get task-specific training configuration for Phase 2
    task2_timesteps = get_task_timesteps(train_task_2, config)
    convergence_callback_2 = create_convergence_callback(train_task_2, config)
    
    # Create Phase 2 callback
    phase2_callback = CallbackList([
        callback,  # Main logging callback
        convergence_callback_2,  # Convergence monitoring for Phase 2
        RewardMonitorCallback(train_task_2, log_interval=10000),  # Reward monitoring
        ConvergenceEvaluationCallback(convergence_callback_2, model, env2, train_task_2, eval_interval=20000),  # Convergence evaluation
        ProgressBarCallback(task2_timesteps, train_task_2)  # 🚨 CRITICAL: Fresh progress bar for Phase 2
    ])
    
    # Train on Acrobot-v1
    print(f"📋 Task-specific training: {train_task_2} for {task2_timesteps:,} timesteps")
    model.learn(total_timesteps=task2_timesteps, callback=phase2_callback, progress_bar=False)
    
    # Update global timesteps after Phase 2
    actual_task2_duration = model.num_timesteps if hasattr(model, 'num_timesteps') else task2_timesteps
    logging_handler.update_global_timesteps(actual_task2_duration)
    
    # 🚨 CRITICAL: Evaluate on all tasks after Phase 2 training
    print("\n" + "=" * 80)
    print("🔍 PHASE 2 EVALUATION: Testing performance on all tasks after Acrobot-v1 training")
    print("=" * 80)
    
    # Evaluate on all tasks to see transfer learning effects
    for task_name in [train_task_1, train_task_2, train_task_3]:
        print(f"\n📊 Evaluating on {task_name} after Acrobot-v1 training...")
        try:
            # Create evaluation environment
            eval_env = DummyVecEnv([make_env(task_name, seed=seed)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Evaluation logging removed - keeping only standard training metrics
            pass
            
            print(f"   ✅ {task_name}: Reward={np.mean(rewards):.2f}, Success={success:.2%}, Completion={completion:.2%}")
            
        except Exception as e:
            print(f"   ❌ {task_name} evaluation failed: {e}")
    
    # 🚨 CRITICAL: Phase 3 - Train on LunarLander-v2
    print("\n" + "=" * 80)
    print("🎯 PHASE 3: Training on LunarLander-v2")
    print("=" * 80)
    
    # Switch to LunarLander environment for Phase 3
    model.set_env(env3)
    logging_handler.set_task_phase(train_task_3, 3)
    
    # Get task-specific training configuration for Phase 3
    task3_timesteps = get_task_timesteps(train_task_3, config)
    convergence_callback_3 = create_convergence_callback(train_task_3, config)
    
    # Create Phase 3 callback
    phase3_callback = CallbackList([
        callback,  # Main logging callback
        convergence_callback_3,  # Convergence monitoring for Phase 3
        RewardMonitorCallback(train_task_3, log_interval=10000),  # Reward monitoring
        ConvergenceEvaluationCallback(convergence_callback_3, model, env3, train_task_3, eval_interval=20000),  # Convergence evaluation
        ProgressBarCallback(task3_timesteps, train_task_3)  # 🚨 CRITICAL: Fresh progress bar for Phase 3
    ])
    
    # Train on LunarLander-v2
    print(f"📋 Task-specific training: {train_task_3} for {task3_timesteps:,} timesteps")
    model.learn(total_timesteps=task3_timesteps, callback=phase3_callback, progress_bar=False)
    
    # Update global timesteps after Phase 3
    actual_task3_duration = model.num_timesteps if hasattr(model, 'num_timesteps') else task3_timesteps
    logging_handler.update_global_timesteps(actual_task3_duration)
    
    # 🚨 CRITICAL: Final evaluation on all tasks after Phase 3 training
    print("\n" + "=" * 80)
    print("🔍 FINAL EVALUATION: Testing performance on all tasks after complete training")
    print("=" * 80)
    
    # Evaluate on all tasks to see final transfer learning effects
    for task_name in [train_task_1, train_task_2, train_task_3]:
        print(f"\n📊 Final evaluation on {task_name}...")
        try:
            # Create evaluation environment
            eval_env = DummyVecEnv([make_env(task_name, seed=seed)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Evaluation logging removed - keeping only standard training metrics
            pass
            
            print(f"   ✅ {task_name}: Reward={np.mean(rewards):.2f}, Success={success:.2%}, Completion={completion:.2%}")
            
        except Exception as e:
            print(f"   ❌ {task_name} evaluation failed: {e}")
    
    # 🚨 CRITICAL: Log final training summary
    print("\n" + "=" * 80)
    print("🎯 TRIPLE-TASK TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Total training timesteps: {logging_handler.global_timesteps:,}")
    print(f"📊 Task 1 ({train_task_1}): {actual_task1_duration:,} timesteps")
    print(f"📊 Task 2 ({train_task_2}): {actual_task2_duration:,} timesteps")
    print(f"📊 Task 3 ({train_task_3}): {actual_task3_duration:,} timesteps")
    print(f"🔧 Topology: {topology_type}")
    print(f"🔧 Network capacity: {total_params:,} parameters")
    print("=" * 80)
    
    # Final summary logging removed - keeping only standard training metrics
    pass
    
    return {
        'model': model,
        'total_timesteps': logging_handler.global_timesteps,
        'task_durations': [actual_task1_duration, actual_task2_duration, actual_task3_duration],
        'topology_type': topology_type,
        'network_capacity': total_params
    }

# ============================================================================
# CONTINUAL LEARNING TRAINING FUNCTION
# ============================================================================

def continual_learning_training(config, task_name, topology_type, seed, use_wandb=True, enable_phase3=False):
    """
    Train a model using continual learning with observation shifts.
    
    PAPER-ACCURATE IMPLEMENTATION:
    - Iteration-based training (not step-based)
    - 3000 iterations total
    - 200 iterations per perturbation level
    - Level 0 (0-199): NO NOISE - Clean baseline
    - Level 1+ (200+): Perturbations every 200 iterations
    - Reward scaling: Division by 20 (creates small gradients)
    """
    print(f"🚀 Starting Continual Learning Training (Paper-Accurate)")
    print(f"   Task: {task_name}")
    print(f"   Topology: {topology_type}")
    print(f"   Seed: {seed}")
    print(f"   W&B: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"   Phase 3 Analysis: {'Enabled' if enable_phase3 else 'Disabled'}")
    
    # Extract configuration parameters for paper-accurate approach
    max_iterations = config.get('max_iterations', 3000)  # Total iterations
    level_switch = config.get('level_switch', 200)       # Iterations per level
    shift_range = config.get('shift_range', [0, 2])     # Perturbation range
    reward_scale = config.get('reward_scale', 20.0)      # Division factor
    episode_cap = config.get('episode_cap', 400)         # Max steps per episode
    
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Level Switch: {level_switch} iterations")
    print(f"   Shift Range: {shift_range}")
    print(f"   Reward Scale: {reward_scale} (division factor)")
    print(f"   Episode Cap: {episode_cap} steps")
    print(f"   PPO Steps: {config.get('n_steps', 800)}")
    print(f"   PPO Epochs: {config.get('n_epochs', 5)}")
    print(f"   Expected Total Env Steps: ~{max_iterations * 800:,}")
    
    # Initialize W&B if requested
    if use_wandb:
        # Create initial run name (will be updated with actual parameters later)
        initial_run_name = f"continual_learning_{task_name}_{topology_type}_seed{seed}"
        
        wandb.init(
            project="topologies--continual-learning-training",
            name=initial_run_name,
            config={
                'task_name': task_name,
                'topology_type': topology_type,
                'seed': seed,
                'max_iterations': max_iterations,
                'level_switch': level_switch,
                'shift_range': shift_range,
                'reward_scale': reward_scale,
                'episode_cap': episode_cap,
                **{k: v for k, v in config.items() if k not in ['max_iterations', 'level_switch', 'shift_range', 'reward_scale', 'episode_cap']}
            },
            tags=["continual_learning", "paper_accurate", "iteration_based"]
        )
    
    # Initialize local data collector for offline analysis
    local_data_collector = LocalDataCollector(
        base_path="test_experiments",
        run_id=f"{task_name}_{topology_type}_seed{seed}_{int(time.time())}"
    )
    
    # Create enhanced logging callback with local data collection
    enhanced_logging_callback = EnhancedLoggingCallback(
        task_name=task_name,
        topology_type=topology_type,
        seed=seed,
        reward_scale=reward_scale,
        local_data_collector=local_data_collector
    )
    
    # Create environment with enhanced logging
    env = make_env(
        env_name=task_name,
        seed=seed,
        continual_learning=True,
        max_iterations=max_iterations,
        level_switch=level_switch,
        shift_range=shift_range,
        reward_scale=reward_scale,
        episode_cap=episode_cap,
        logging_callback=enhanced_logging_callback,
        num_levels=config.get('max_iterations', 3000) // 200  # Calculate from max_iterations
    )()
    
    # Create model with custom topology policy
    model = PPO(
        policy=DebugTopologyPolicy,  # Use custom topology-aware policy
        env=env,
        policy_kwargs={
            'topology_type': topology_type,
            'hidden_size': config.get('hidden_size', 128),
            'num_layers': config.get('num_layers', 1),
            'config': config
        },
        verbose=0,  # Disable verbose output to reduce terminal spam and speed up training
        learning_rate=config.get('learning_rate', 0.01),  # Paper-aligned learning rate
        n_steps=config.get('n_steps', 800),
        batch_size=config.get('batch_size', 32),
        n_epochs=config.get('n_epochs', 5),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        max_grad_norm=config.get('max_grad_norm', 0.5)
    )
    
    # Update W&B run name with actual parameters if W&B is enabled
    if use_wandb and wandb.run is not None:
        try:
            # Create sophisticated run name with actual parameters
            sophisticated_run_name = create_continual_learning_run_name(
                config, topology_type, task_name, seed, model, shift_range
            )
            
            # Update the run name
            wandb.run.name = sophisticated_run_name
            print(f"✅ Updated run name with actual parameters: {sophisticated_run_name}")
            
        except Exception as e:
            print(f"⚠️  Could not update run name with actual parameters: {e}")
            print(f"   Using initial run name: {initial_run_name}")
    
    # PAPER-ACCURATE ITERATION-BASED TRAINING
    print(f"🎯 Starting iteration-based training for {max_iterations} iterations...")
    print(f"   Each iteration will run ~800 environment steps (2 episodes × 400 steps)")
    print(f"   Perturbation levels will switch every {level_switch} iterations")
    print(f"   Total expected environment steps: ~{max_iterations * 800:,}")
    
    # Initialize iteration tracking
    current_iteration = 0
    total_env_steps = 0
    
    # Main iteration loop
    while current_iteration < max_iterations:
        # Set current iteration in environment wrapper (this will show progress and level changes)
        env.set_iteration(current_iteration)
        
        # Run training for this iteration (approximately 800 env steps)
        # This will run 2 episodes with max 400 steps each
        iteration_steps = 0
        episodes_in_iteration = 0
        
        while iteration_steps < 800 and episodes_in_iteration < 2:
            # Train for one episode or until we reach 800 steps
            model.learn(
                total_timesteps=min(400, 800 - iteration_steps),  # Max 400 steps per episode
                callback=enhanced_logging_callback,
                progress_bar=False,  # Disable progress bar for cleaner output
                reset_num_timesteps=False  # Don't reset timestep counter
            )
            
            # Update iteration tracking
            if hasattr(env, 'get_current_info'):
                info = env.get_current_info()
                iteration_steps = info['total_env_steps'] - total_env_steps
                episodes_in_iteration = info['episodes_in_iteration']
                total_env_steps = info['total_env_steps']
        
        # Show iteration completion summary (only every 50 iterations to reduce noise)
        # if current_iteration % 50 == 0:
        #     print(f"   ✅ Iteration {current_iteration + 1}/{max_iterations} completed: {episodes_in_iteration} episodes, ~{iteration_steps} steps")
        
        # Move to next iteration
        current_iteration += 1
    
    print(f"\n🎯 Training completed! Total iterations: {max_iterations}")
    print(f"   Total environment steps: {total_env_steps:,}")
    print(f"   Total perturbation levels: {max_iterations // level_switch + 1}")
    
    # Get final episode returns for analysis
    if hasattr(env, 'episode_returns'):
        episode_returns = env.episode_returns
        print(f"📊 Total episodes completed: {len(episode_returns)}")
        
        if episode_returns:
            final_raw_return = np.mean([ep['episode_return_raw'] for ep in episode_returns])
            final_scaled_return = np.mean([ep['episode_return_scaled'] for ep in episode_returns])
            print(f"   Final mean raw return: {final_raw_return:.2f}")
            print(f"   Final mean scaled return: {final_scaled_return:.2f}")
    
    # Finalize local data collection
    print("\n📁 Finalizing local data collection...")
    run_dir = local_data_collector.finalize_run()
    print(f"✅ Local data saved to: {run_dir}")
    
    # Phase 3: Advanced Analysis & Visualization
    if enable_phase3:
        print("\n🎨 Phase 3: Creating Advanced Analysis & Visualization...")
        
        # Create advanced plotter
        advanced_plotter = AdvancedContinualLearningPlotter(
            task_name=task_name,
            topology_type=topology_type,
            seed=seed,
            reward_scale=reward_scale
        )
        
        # Extract data from enhanced logging callback
        episode_data = enhanced_logging_callback.episode_buffer if hasattr(enhanced_logging_callback, 'episode_buffer') else []
        shift_data = enhanced_logging_callback.shift_buffer if hasattr(enhanced_logging_callback, 'shift_buffer') else []
        
        # Create update data from PPO training
        update_data = []
        if hasattr(enhanced_logging_callback, 'update_index'):
            for i in range(enhanced_logging_callback.update_index):
                # Get actual update data if available
                if hasattr(enhanced_logging_callback, 'episode_buffer'):
                    # Calculate mean returns for this update period
                    start_step = i * 800
                    end_step = (i + 1) * 800
                    
                    # Find episodes in this update period
                    update_episodes = [ep for ep in enhanced_logging_callback.episode_buffer 
                                     if start_step <= ep['global_step_end'] <= end_step]
                    
                    if update_episodes:
                        mean_scaled = np.mean([ep['episode_return_scaled'] for ep in update_episodes])
                        mean_raw = np.mean([ep['episode_return_raw'] for ep in update_episodes])
                        std_scaled = np.std([ep['episode_return_scaled'] for ep in update_episodes])
                        std_raw = np.std([ep['episode_return_raw'] for ep in update_episodes])
                    else:
                        mean_scaled = mean_raw = std_scaled = std_raw = 0
                else:
                    mean_scaled = mean_raw = std_scaled = std_raw = 0
                
                update_data.append({
                    'update_index': i,
                    'global_step_end': (i + 1) * 800,
                    'rollout_size': 800,
                    'epochs_per_update': 5,
                    'reward_scale': reward_scale,
                    'mean_scaled_return': mean_scaled,
                    'mean_raw_return': mean_raw,
                    'std_scaled_return': std_scaled,
                    'std_raw_return': std_raw,
                    'episodes_in_update': len(update_episodes) if 'update_episodes' in locals() else 0
                })
        
        # Create comprehensive analysis
        if episode_data:
            try:
                # Create comprehensive analysis plots
                analysis_fig = advanced_plotter.create_comprehensive_analysis(
                    episode_data=episode_data,
                    shift_data=shift_data,
                    update_data=update_data
                )
                print("✅ Comprehensive analysis plots created successfully!")
                
                # Create detailed shift impact analysis
                shift_impact_fig, impact_metrics = advanced_plotter.create_shift_impact_analysis(
                    episode_data=episode_data,
                    shift_data=shift_data
                )
                print("✅ Shift impact analysis created successfully!")
                
                # Log analysis metrics to W&B
                if wandb.run:
                    # Log key analysis metrics
                    if impact_metrics:
                        mean_stability = np.mean([m['stability'] for m in impact_metrics.values() if m['stability'] != float('inf')])
                        mean_performance_change = np.mean([m['performance_change'] for m in impact_metrics.values()])
                        
                        wandb.log({
                            'analysis/mean_stability': mean_stability,
                            'analysis/mean_performance_change': mean_performance_change,
                            'analysis/total_shifts_analyzed': len(impact_metrics),
                            'analysis/analysis_completed': True
                        })
                        print("📊 Analysis metrics logged to W&B")
                    
            except Exception as e:
                print(f"⚠️  Advanced analysis failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️  No episode data available for advanced analysis")
    else:
        print("\n⏭️  Phase 3 analysis skipped (use --phase3 to enable)")
    
    # Close W&B run
    if use_wandb:
        wandb.finish()
    
    return model, env

# ============================================================================
# ENHANCED LOGGING SYSTEM (Phase 2)
# ============================================================================

class EnhancedLoggingCallback(BaseCallback):
    """
    Enhanced logging callback for Phase 2: Multi-granularity metrics.
    
    Logs at three levels:
    1. Per-shift (every 200 steps): Shift boundaries and IDs
    2. Per-episode (individual completion): Returns, lengths, timestamps
    3. Per-update (every 800 steps): PPO diagnostics and mean returns
    """
    
    def __init__(self, task_name, topology_type, seed, reward_scale=20.0, local_data_collector=None):
        super().__init__()
        self.task_name = task_name
        self.topology_type = topology_type
        self.seed = seed
        self.reward_scale = reward_scale
        self.local_data_collector = local_data_collector
        
        # Tracking
        self.update_index = 0
        self.last_update_step = 0
        self.episode_buffer = []
        self.shift_buffer = []
        
        # Episode tracking for PPO updates
        self.episodes_since_last_update = []
        
        if local_data_collector:
            print(f"📁 Enhanced logging with local data collection enabled")
    
    def _on_step(self) -> bool:
        """Called every step during training."""
        # Per-shift logging (every 200 steps)
        if self.num_timesteps % 200 == 0:
            self._log_shift_event()
        
        # Per-update logging (every 800 steps)
        if self.num_timesteps - self.last_update_step >= 800:
            self._log_update_event()
            self.last_update_step = self.num_timesteps
            self.update_index += 1
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called when a PPO rollout ends (every n_steps)."""
        # This is called more reliably than _on_step during PPO training
        if wandb.run:
            # Log rollout completion
            wandb.log({
                'ppo/rollout_completed': True,
                'ppo/rollout_step': self.num_timesteps,
                'ppo/rollout_size': 800
            }, step=self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        # Log final summary silently (no spam)
        # Note: Plots are now logged on level changes, not at training end
    
    def _log_shift_event(self):
        """Log shift boundary events when perturbation levels actually change."""
        # Only log shifts when perturbation levels change, not every 200 env steps
        # This method should be called from the wrapper when levels change
        pass
    
    def _log_perturbation_level_change(self, iteration, level, perturbation):
        """Log when perturbation levels actually change (every 200 iterations) - simplified for minimal W&B logging."""
        shift_id = level
        
        # Store level change event for analysis
        level_change_data = {
            'iteration': iteration,
            'level': level,
            'global_step': self.num_timesteps,
            'perturbation': perturbation.tolist() if level > 0 else [0, 0, 0, 0],
            'timestamp': time.time()
        }
        self.shift_buffer.append(level_change_data)
        
        # Log to local data collector if available
        if self.local_data_collector:
            local_level_info = {
                'shift_step': self.num_timesteps,
                'shift_id': shift_id,
                'iteration': iteration,
                'level': level,
                'offset_repr': str(perturbation.tolist()) if level > 0 else "[0, 0, 0, 0]",
                'seed': self.seed,
                'env': self.task_name,
                'topology': self.topology_type
            }
            self.local_data_collector.log_shift(local_level_info)
        
        # Log the iteration vs. rewards plot to W&B only when levels change
        if wandb.run and hasattr(self, 'update_buffer') and self.update_buffer:
            self._create_and_log_iteration_plot()
        
        if level == 0:
            print(f"🔄 Level {level} change logged at iteration {iteration}: Clean baseline (NO NOISE)")
    else:
            print(f"🔄 Level {level} change logged at iteration {iteration}: Perturbation applied")
    
    def _log_update_event(self):
        """Log PPO update events - simplified for minimal W&B logging and performance."""
        # Only process episode data every 20 updates to improve performance
        if self.update_index % 20 == 0:
            # Calculate mean episode return over last update for local tracking
            recent_episodes = self._get_recent_episodes(800)
            
            if recent_episodes:
                # Calculate both scaled and raw returns for local tracking
                scaled_returns = [ep['episode_return_scaled'] for ep in recent_episodes]
                raw_returns = [ep['episode_return_raw'] for ep in recent_episodes]
                
                mean_scaled_return = np.mean(scaled_returns)
                mean_raw_return = np.mean(raw_returns)
                
                # Only show PPO update logging every 10 updates to reduce noise
                # if self.update_index % 10 == 0:
                #     print(f"📊 PPO Update {self.update_index}: Mean raw return={mean_raw_return:.1f}")
                
                # Store update data for the iteration vs. rewards plot
                update_data = {
                    'update_index': self.update_index,
                    'global_step': self.num_timesteps,
                    'mean_raw_return': mean_raw_return,
                    'episodes_in_update': len(recent_episodes)
                }
    else:
                # Store update data for the iteration vs. rewards plot (no episodes)
                update_data = {
                    'update_index': self.update_index,
                    'global_step': self.num_timesteps,
                    'mean_raw_return': 0.0,
                    'episodes_in_update': 0
                }
            
            # Store for the iteration vs. rewards plot
            if not hasattr(self, 'update_buffer'):
                self.update_buffer = []
            self.update_buffer.append(update_data)
    
    def _log_episode_completion(self, episode_data):
        """Log individual episode completion - simplified for minimal W&B logging."""
        # Store episode for update calculations
        self.episodes_since_last_update.append(episode_data)
        
        # Store for analysis
        self.episode_buffer.append(episode_data)
        
        # Log to local data collector if available
        if self.local_data_collector:
            local_episode_info = {
                'global_step_end': episode_data['global_step_end'],
                'episode_length': episode_data['episode_length'],
                'episode_return_raw': episode_data['episode_return_raw'],
                'episode_return_scaled': episode_data['episode_return_scaled'],
                'shift_id': episode_data['shift_id'],
                'seed': self.seed,
                'env': self.task_name,
                'topology': self.topology_type
            }
            self.local_data_collector.log_episode(local_episode_info)
        
        # Only show episode logging for milestone episodes (every 100 episodes)
        # Removed to keep terminal clean between levels
    
    def _get_recent_episodes(self, window_steps):
        """Get episodes that ended within the last window_steps."""
        current_step = self.num_timesteps
        return [ep for ep in self.episodes_since_last_update 
                if current_step - ep['global_step_end'] <= window_steps]
    
    def _create_and_log_iteration_plot(self):
        """Create the single iteration vs. rewards plot and log it to W&B."""
        try:
            # Extract iteration data from update buffer
            iterations = []
            mean_rewards = []
            
            # Group updates by iteration (each iteration has ~800 steps, so group updates accordingly)
            current_iteration = 0
            current_iteration_rewards = []
            
            for update in self.update_buffer:
                # Estimate which iteration this update belongs to
                estimated_iteration = update['global_step'] // 800
                
                if estimated_iteration != current_iteration:
                    # Save previous iteration data
                    if current_iteration_rewards:
                        iterations.append(current_iteration)
                        mean_rewards.append(np.mean(current_iteration_rewards))
                    
                    # Start new iteration
                    current_iteration = estimated_iteration
                    current_iteration_rewards = [update['mean_raw_return']]
                    else:
                    current_iteration_rewards.append(update['mean_raw_return'])
            
            # Add final iteration
            if current_iteration_rewards:
                iterations.append(current_iteration)
                mean_rewards.append(np.mean(current_iteration_rewards))
            
            if iterations and mean_rewards:
                # Create the plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(iterations, mean_rewards, 'b-', linewidth=2, marker='o', markersize=4)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Mean Episode Reward (Raw)')
                ax.set_title(f'Iteration vs. Mean Episode Rewards\n{self.task_name} - {self.topology_type}')
                ax.grid(True, alpha=0.3)
                
                # Add level change markers
                level_switch = 200  # Each level is 200 iterations
                for level in range(1, max(iterations) // level_switch + 1):
                    iteration_marker = level * level_switch
                    if iteration_marker <= max(iterations):
                        ax.axvline(x=iteration_marker, color='r', linestyle='--', alpha=0.7, 
                                  label=f'Level {level}' if level == 1 else "")
                
                if max(iterations) > level_switch:
                    ax.legend()
                
                plt.tight_layout()
                
                # Log to W&B
                wandb.log({"iteration_vs_rewards_plot": wandb.Image(fig)})
                print("📊 Iteration vs. rewards plot logged to W&B")
                
                plt.close(fig)
                
        except Exception as e:
            print(f"⚠️  Failed to create iteration plot: {e}")
            import traceback
            traceback.print_exc()

# ============================================================================
# ADVANCED ANALYSIS & VISUALIZATION SYSTEM (Phase 3)
# ============================================================================

class AdvancedContinualLearningPlotter:
    """
    Advanced plotting system for Phase 3: Multi-granularity visualization.
    
    Features:
    1. Shift-level analysis: Performance across observation shifts
    2. Episode-level analysis: Individual episode performance trends
    3. Update-level analysis: PPO learning dynamics
    4. Catastrophic forgetting metrics: Performance degradation analysis
    5. Comparative analysis: Cross-topology performance comparison
    """
    
    def __init__(self, task_name, topology_type, seed, reward_scale=20.0):
        self.task_name = task_name
        self.topology_type = topology_type
        self.seed = seed
        self.reward_scale = reward_scale
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'shifts': '#1f77b4',
            'episodes': '#ff7f0e', 
            'updates': '#2ca02c',
            'performance': '#d62728',
            'forgetting': '#9467bd'
        }
        
    def create_comprehensive_analysis(self, episode_data, shift_data, update_data):
        """
        Create comprehensive analysis plots for continual learning research.
        
        Args:
            episode_data: List of episode dictionaries from EnhancedLoggingCallback
            shift_data: List of shift dictionaries from EnhancedLoggingCallback
            update_data: List of update dictionaries from EnhancedLoggingCallback
        """
        print("🎨 Creating comprehensive continual learning analysis...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Continual Learning Analysis: {self.task_name} - {self.topology_type} (Seed {self.seed})', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Performance Over Time
        self._plot_episode_performance(axes[0, 0], episode_data)
        
        # Plot 2: Shift-Level Performance Analysis
        self._plot_shift_performance(axes[0, 1], episode_data, shift_data)
        
        # Plot 3: PPO Update Learning Curves
        self._plot_ppo_learning_curves(axes[0, 2], update_data)
        
        # Plot 4: Catastrophic Forgetting Analysis
        self._plot_catastrophic_forgetting(axes[1, 0], episode_data, shift_data)
        
        # Plot 5: Performance Distribution by Shift
        self._plot_performance_distribution(axes[1, 1], episode_data, shift_data)
        
        # Plot 6: Learning Efficiency Metrics
        self._plot_learning_efficiency(axes[1, 2], episode_data, update_data)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"continual_learning_analysis_{self.task_name}_{self.topology_type}_seed{self.seed}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Analysis saved as: {filename}")
        
        # Log to W&B if available
        if wandb.run:
            wandb.log({"analysis/comprehensive_plots": wandb.Image(fig)})
            print("📊 Analysis logged to W&B")
        
        plt.show()
        return fig
    
    def _plot_episode_performance(self, ax, episode_data):
        """Plot individual episode performance over time."""
        if not episode_data:
            ax.text(0.5, 0.5, 'No episode data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Episode Performance Over Time')
            return
        
        # Extract data
        steps = [ep['global_step_end'] for ep in episode_data]
        scaled_returns = [ep['episode_return'] for ep in episode_data]
        raw_returns = [ep['raw_episode_return'] for ep in episode_data]
        shift_ids = [ep['shift_id'] for ep in episode_data]
        
        # Create scatter plot with color coding by shift
        scatter = ax.scatter(steps, scaled_returns, c=shift_ids, cmap='viridis', alpha=0.7, s=50)
        
        # Add trend line
        z = np.polyfit(steps, scaled_returns, 1)
        p = np.poly1d(z)
        ax.plot(steps, p(steps), "--", color='red', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Episode Return (Scaled)')
        ax.set_title('Episode Performance Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for shift IDs
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Shift ID')
        
        # Add performance statistics
        mean_return = np.mean(scaled_returns)
        ax.axhline(y=mean_return, color='orange', linestyle=':', alpha=0.7, label=f'Mean: {mean_return:.1f}')
        ax.legend()
    
    def _plot_shift_performance(self, ax, episode_data, shift_data):
        """Plot performance analysis across observation shifts."""
        if not episode_data or not shift_data:
            ax.text(0.5, 0.5, 'No shift data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Shift-Level Performance Analysis')
            return
        
        # Group episodes by shift
        shift_performance = {}
        for ep in episode_data:
            shift_id = ep['shift_id']
            if shift_id not in shift_performance:
                shift_performance[shift_id] = []
            shift_performance[shift_id].append(ep['raw_episode_return'])
        
        # Calculate statistics per shift
        shift_ids = sorted(shift_performance.keys())
        mean_returns = [np.mean(shift_performance[sid]) for sid in shift_ids]
        std_returns = [np.std(shift_performance[sid]) for sid in shift_ids]
        
        # Create bar plot with error bars
        bars = ax.bar(shift_ids, mean_returns, yerr=std_returns, capsize=5, 
                      color=self.colors['shifts'], alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Shift ID')
        ax.set_ylabel('Mean Raw Return')
        ax.set_title('Performance Across Observation Shifts')
        ax.grid(True, alpha=0.3)
        
        # Add shift boundary indicators
        for i, (shift_id, mean_return) in enumerate(zip(shift_ids, mean_returns)):
            ax.text(shift_id, mean_return + std_returns[i] + 2, f'{len(shift_performance[shift_id])}', 
                   ha='center', va='bottom', fontsize=8)
        
        # Add overall trend line
        if len(shift_ids) > 1:
            z = np.polyfit(shift_ids, mean_returns, 1)
            p = np.poly1d(z)
            ax.plot(shift_ids, p(shift_ids), "--", color='red', alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
    
    def _plot_ppo_learning_curves(self, ax, update_data):
        """Plot PPO learning dynamics across updates."""
        if not update_data:
            ax.text(0.5, 0.5, 'No update data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PPO Learning Curves')
            return
        
        # Extract data
        update_indices = [up['update_index'] for up in update_data if 'update_index' in up]
        mean_scaled = [up['mean_scaled_return'] for up in update_data if 'mean_scaled_return' in up]
        mean_raw = [up['mean_raw_return'] for up in update_data if 'mean_raw_return' in up]
        
        if not update_indices:
            ax.text(0.5, 0.5, 'No PPO update data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PPO Learning Curves')
            return
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        # Plot scaled returns (left axis)
        line1 = ax.plot(update_indices, mean_scaled, 'o-', color=self.colors['updates'], 
                        linewidth=2, markersize=8, label='Scaled Returns')
        ax.set_xlabel('PPO Update Index')
        ax.set_ylabel('Mean Scaled Return', color=self.colors['updates'])
        ax.tick_params(axis='y', labelcolor=self.colors['updates'])
        
        # Plot raw returns (right axis)
        line2 = ax2.plot(update_indices, mean_raw, 's-', color=self.colors['performance'], 
                         linewidth=2, markersize=8, label='Raw Returns')
        ax2.set_ylabel('Mean Raw Return', color=self.colors['performance'])
        ax2.tick_params(axis='y', labelcolor=self.colors['performance'])
        
        # Customize plot
        ax.set_title('PPO Learning Dynamics')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    def _plot_catastrophic_forgetting(self, ax, episode_data, shift_data):
        """Plot catastrophic forgetting analysis."""
        if not episode_data or not shift_data:
            ax.text(0.5, 0.5, 'No data available for forgetting analysis', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Catastrophic Forgetting Analysis')
            return
        
        # Group episodes by shift and calculate forgetting metrics
        shift_performance = {}
        for ep in episode_data:
            shift_id = ep['shift_id']
            if shift_id not in shift_performance:
                shift_performance[shift_id] = []
            shift_performance[shift_id].append(ep['raw_episode_return'])
        
        # Calculate forgetting metrics
        shift_ids = sorted(shift_performance.keys())
        mean_returns = [np.mean(shift_performance[sid]) for sid in shift_ids]
        
        # Calculate forgetting rate (performance drop between consecutive shifts)
        forgetting_rates = []
        for i in range(1, len(mean_returns)):
            if mean_returns[i-1] > 0:  # Avoid division by zero
                forgetting_rate = (mean_returns[i-1] - mean_returns[i]) / mean_returns[i-1]
                forgetting_rates.append(forgetting_rate)
                    else:
                forgetting_rates.append(0)
        
        # Create forgetting analysis plot
        x_pos = range(len(forgetting_rates))
        bars = ax.bar(x_pos, forgetting_rates, color=self.colors['forgetting'], alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Shift Transition')
        ax.set_ylabel('Forgetting Rate')
        ax.set_title('Catastrophic Forgetting Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add shift transition labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{shift_ids[i]}→{shift_ids[i+1]}' for i in range(len(shift_ids)-1)])
        
        # Add statistics
        mean_forgetting = np.mean(forgetting_rates) if forgetting_rates else 0
        ax.axhline(y=mean_forgetting, color='red', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_forgetting:.3f}')
        ax.legend()
        
        # Color code bars based on forgetting severity
        for i, bar in enumerate(bars):
            if forgetting_rates[i] > mean_forgetting:
                bar.set_color('red')
            elif forgetting_rates[i] < mean_forgetting:
                bar.set_color('green')
    
    def _plot_performance_distribution(self, ax, episode_data, shift_data):
        """Plot performance distribution analysis by shift."""
        if not episode_data:
            ax.text(0.5, 0.5, 'No episode data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Distribution by Shift')
            return
        
        # Group episodes by shift
        shift_performance = {}
        for ep in episode_data:
            shift_id = ep['shift_id']
            if shift_id not in shift_performance:
                shift_performance[shift_id] = []
            shift_performance[shift_id].append(ep['raw_episode_return'])
        
        # Create box plot
        shift_ids = sorted(shift_performance.keys())
        performance_lists = [shift_performance[sid] for sid in shift_ids]
        
        box_plot = ax.boxplot(performance_lists, labels=shift_ids, patch_artist=True)
        
        # Color code boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(shift_ids)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot
        ax.set_xlabel('Shift ID')
        ax.set_ylabel('Raw Episode Return')
        ax.set_title('Performance Distribution by Shift')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        overall_mean = np.mean([ep['raw_episode_return'] for ep in episode_data])
        ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Mean: {overall_mean:.2f}')
        ax.legend()
    
    def _plot_learning_efficiency(self, ax, episode_data, update_data):
        """Plot learning efficiency metrics."""
        if not episode_data:
            ax.text(0.5, 0.5, 'No episode data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Efficiency Metrics')
            return
        
        # Calculate learning efficiency metrics
        steps = [ep['global_step_end'] for ep in episode_data]
        returns = [ep['raw_episode_return'] for ep in episode_data]
        
        # Calculate moving average for smooth trend
        window_size = min(10, len(returns) // 4)  # Adaptive window size
        if window_size > 1:
            moving_avg = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
            moving_avg_steps = steps[window_size-1:]
                else:
            moving_avg = returns
            moving_avg_steps = steps
        
        # Calculate learning rate (improvement per episode)
        if len(returns) > 1:
            learning_rates = np.diff(returns)
            learning_rate_steps = steps[1:]
        else:
            learning_rates = [0]
            learning_rate_steps = steps
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        # Plot moving average (left axis)
        line1 = ax.plot(moving_avg_steps, moving_avg, 'o-', color=self.colors['performance'], 
                        linewidth=2, markersize=6, label='Moving Average')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Moving Average Return', color=self.colors['performance'])
        ax.tick_params(axis='y', labelcolor=self.colors['performance'])
        
        # Plot learning rates (right axis)
        line2 = ax2.plot(learning_rate_steps, learning_rates, 's-', color=self.colors['performance'], 
                         linewidth=1, markersize=4, alpha=0.7, label='Learning Rate')
        ax2.set_ylabel('Learning Rate (Δ Return)', color=self.colors['performance'])
        ax2.tick_params(axis='y', labelcolor=self.colors['performance'])
        
        # Customize plot
        ax.set_title('Learning Efficiency Analysis')
        ax.grid(True, alpha=0.3)
        
        # Add zero line for learning rate
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
    
    def create_shift_impact_analysis(self, episode_data, shift_data):
        """Create detailed shift impact analysis."""
        if not episode_data or not shift_data:
            print("⚠️  No data available for shift impact analysis")
            return None
        
        print("🔍 Creating shift impact analysis...")
        
        # Group episodes by shift
        shift_performance = {}
        for ep in episode_data:
            shift_id = ep['shift_id']
            if shift_id not in shift_performance:
                shift_performance[shift_id] = []
            shift_performance[shift_id].append(ep['raw_episode_return'])
        
        # Calculate shift impact metrics
        shift_ids = sorted(shift_performance.keys())
        impact_metrics = {}
        
        for i, shift_id in enumerate(shift_ids):
            returns = shift_performance[shift_id]
            
            # Basic statistics
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            episode_count = len(returns)
            
            # Performance change from previous shift
            if i > 0:
                prev_mean = np.mean(shift_performance[shift_ids[i-1]])
                performance_change = (mean_return - prev_mean) / prev_mean if prev_mean != 0 else 0
            else:
                performance_change = 0
            
            # Stability metric (inverse of coefficient of variation)
            stability = mean_return / std_return if std_return != 0 else float('inf')
            
            impact_metrics[shift_id] = {
                'mean_return': mean_return,
                'std_return': std_return,
                'episode_count': episode_count,
                'performance_change': performance_change,
                'stability': stability
            }
        
        # Create impact analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Shift Impact Analysis: {self.task_name} - {self.topology_type}', fontsize=16, fontweight='bold')
        
        # Plot 1: Performance change between shifts
        changes = [impact_metrics[sid]['performance_change'] for sid in shift_ids[1:]]
        axes[0, 0].bar(range(len(changes)), changes, color=self.colors['shifts'], alpha=0.8)
        axes[0, 0].set_xlabel('Shift Transition')
        axes[0, 0].set_ylabel('Performance Change (%)')
        axes[0, 0].set_title('Performance Change Between Shifts')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(len(changes)))
        axes[0, 0].set_xticklabels([f'{shift_ids[i]}→{shift_ids[i+1]}' for i in range(len(shift_ids)-1)])
        
        # Plot 2: Stability across shifts
        stabilities = [impact_metrics[sid]['stability'] for sid in shift_ids]
        # Handle infinite stability values
        stabilities = [s if s != float('inf') else max(stabilities) * 1.1 for s in stabilities]
        axes[0, 1].plot(shift_ids, stabilities, 'o-', color=self.colors['performance'], linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Shift ID')
        axes[0, 1].set_ylabel('Stability (Mean/Std)')
        axes[0, 1].set_title('Performance Stability Across Shifts')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Episode count per shift
        episode_counts = [impact_metrics[sid]['episode_count'] for sid in shift_ids]
        axes[1, 0].bar(shift_ids, episode_counts, color=self.colors['updates'], alpha=0.8)
        axes[1, 0].set_xlabel('Shift ID')
        axes[1, 0].set_ylabel('Number of Episodes')
        axes[1, 0].set_title('Episode Distribution Across Shifts')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Return variance across shifts
        std_returns = [impact_metrics[sid]['std_return'] for sid in shift_ids]
        axes[0, 1].plot(shift_ids, std_returns, 's-', color=self.colors['forgetting'], linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('Shift ID')
        axes[0, 1].set_ylabel('Return Standard Deviation')
        axes[0, 1].set_title('Performance Variance Across Shifts')
        axes[0, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"shift_impact_analysis_{self.task_name}_{self.topology_type}_seed{self.seed}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Shift impact analysis saved as: {filename}")
        
        # Log to W&B if available
        if wandb.run:
            wandb.log({"analysis/shift_impact_analysis": wandb.Image(fig)})
            print("📊 Shift impact analysis logged to W&B")
        
        plt.show()
        return fig, impact_metrics

# ============================================================================
# TEST EXPERIMENT RUNNER
# ============================================================================

def run_test_experiment(task_name="CartPole-v1", seeds=[42, 123, 456, 789, 999], use_wandb=False, num_levels=15):
    """
    Run complete test experiment with local data collection.
    
    Args:
        task_name: Environment to test (CartPole-v1, Acrobot-v1, LunarLander-v2)
        seeds: List of seeds to test
        use_wandb: Whether to enable W&B logging
        num_levels: Number of distribution shift levels
    """
    print("🧪 Starting Continual Learning Test Experiment")
    print("=" * 80)
    print(f"🎯 Configuration:")
    print(f"   Task: {task_name}")
    print(f"   Seeds: {len(seeds)} seeds")
    print(f"   Number of Levels: {num_levels}")
    print(f"   W&B: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"   Topologies: Small World, Fully Connected")
    print("=" * 80)
    
    # Test configuration using the new system
    test_config = create_debug_config(num_levels)
    
    # Test each topology
    for topology in ['small_world', 'fully_connected']:
        print(f"\n🎯 Testing {topology} topology...")
        
        for seed in seeds:
            print(f"\n   🌱 Running seed {seed}...")
            
            try:
                # Run single training with local data collection
                model, env = continual_learning_training(
                    config=test_config,
                    task_name=task_name,
                    topology_type=topology,
                    seed=seed,
                    use_wandb=use_wandb,
                    enable_phase3=False  # Disable Phase 3 for test runs
                )
                
                print(f"   ✅ {topology} seed {seed} completed successfully!")
                
            except Exception as e:
                print(f"   ❌ {topology} seed {seed} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Create Figure-6 style plots
    print(f"\n🎨 Creating Figure-6 style plots for {task_name}...")
    plotter = Figure6Plotter(data_path="test_experiments")
    plotter.create_topology_comparison_plots(task_name, seeds)
    
    print("\n🎉 Test experiment completed!")
    print("📊 Check test_experiments/plots/ for Figure-6 style plots")
    print("📁 Check test_experiments/ for raw data files")

if __name__ == "__main__":
    """
    Main execution for continual learning training with enhanced logging.
    
    Usage:
        # Quick test with 5 levels (800K env steps)
        python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5
        
        # Medium test with 10 levels (1.6M env steps)  
        python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 10
        
        # Full experiment with 15 levels (2.4M env steps, default)
        python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42
        
        # Test experiment with 5 levels
        python topologies_continual_task_training_sweep.py --test --task CartPole-v1 --num_levels 5
    """
    parser = argparse.ArgumentParser(description="Continual Learning Training with Enhanced Logging")
    parser.add_argument("--single", action="store_true", help="Run single training instead of sweep")
    parser.add_argument("--topology", type=str, default="small_world", 
                       choices=["small_world", "modular", "hybrid", "fully_connected", "standard_mlp"],
                       help="Network topology type")
    parser.add_argument("--task", type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v2"],
                       help="Environment to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--phase3", action="store_true", help="Enable Phase 3 advanced analysis")
    parser.add_argument("--test", action="store_true", help="Run test experiment with multiple seeds and topologies")
    parser.add_argument("--num_levels", type=int, default=15, 
                       help="Number of distribution shift levels (default: 15, each level = 200 iterations × 800 env steps)")
    parser.add_argument("--num_layers", type=int, default=1, 
                       help="Number of layers for topology networks (default: 1, standard_mlp supports multiple layers)")
    
    args = parser.parse_args()
    
    print("🚀 Topology Playground - Continual Learning Training (Phase 2)")
    print("=" * 80)
    print(f"🎯 Configuration:")
    print(f"   Topology: {args.topology}")
    print(f"   Task: {args.task}")
    print(f"   Seed: {args.seed}")
    print(f"   Number of Levels: {args.num_levels}")
    print(f"   Number of Layers: {args.num_layers}")
    print(f"   W&B: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"   Phase 3 Analysis: {'Enabled' if args.phase3 else 'Disabled'}")
    print(f"   Mode: {'Single Training' if args.single else 'Sweep'}")
    print("=" * 80)
    
    # Create debug configuration
    config = create_debug_config(args.num_levels, args.num_layers)
    
    if args.test:
        # Test experiment mode - run multiple seeds and topologies
        print("🧪 Starting test experiment mode...")
        
        try:
            run_test_experiment(
                task_name=args.task,
                seeds=[42, 123, 456, 789, 999],  # 5 seeds for testing
                use_wandb=not args.no_wandb,
                num_levels=args.num_levels
            )
            
            print("🎉 Test experiment completed successfully!")
            
        except Exception as e:
            print(f"❌ Test experiment failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.single:
        # Single training run with enhanced logging
        print("🎯 Starting single training run...")
        
        try:
            model, env = continual_learning_training(
                config=config,
                task_name=args.task,
                topology_type=args.topology,
                seed=args.seed,
                use_wandb=not args.no_wandb,
                enable_phase3=args.phase3
            )
            
            print("✅ Single training completed successfully!")
            print(f"📊 Model: {type(model).__name__}")
            print(f"📊 Environment: {type(env).__name__}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
                
    else:
        # Sweep mode (placeholder for future implementation)
        print("🔄 Sweep mode not yet implemented in Phase 2")
        print("   Use --single flag for individual training runs")
        sys.exit(1)
    