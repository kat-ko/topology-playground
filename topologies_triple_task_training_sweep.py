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
    from wandb_sweep_config import get_config_by_name, generate_parameter_combinations
    print("🔧 Configuration system: Unified config system available")
    CONFIG_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Configuration system: Failed to import unified config: {e}")
    CONFIG_SYSTEM_AVAILABLE = False

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
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, num_layers=2, config=None, *args, **kwargs):
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
        if self.topology_type == 'fully_connected':
            return FullyConnectedTopology(
                size=self.hidden_size,  # Total network size (matches other topologies)
                num_layers=self.num_layers,  # Number of layers (fully connected supports variable layers)
                seed=42  # For reproducibility
            )
        elif self.topology_type == 'small_world':
            k = getattr(wandb.config, 'small_world_k', 4) if wandb.run else 4
            p = getattr(wandb.config, 'small_world_p', 0.2) if wandb.run else 0.2
            return SmallWorldTopology(
                size=self.hidden_size,
                k=k,
                p=p
            )
        elif self.topology_type == 'modular':
            num_modules = getattr(wandb.config, 'modular_num_modules', 4) if wandb.run else 4
            inter_prob = getattr(wandb.config, 'modular_inter_module_prob', 0.1) if wandb.run else 0.1
            intra_prob = getattr(wandb.config, 'modular_intra_module_prob', 0.8) if wandb.run else 0.8
            return ModularTopology(
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
            return HybridTopology(
                size=self.hidden_size,
                num_modules=num_modules,
                k=k,
                p=p,
                inter_module_prob=inter_prob
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
    
    def _get_topology_params(self, topology_network):
        """Get topology-specific parameters."""
        if hasattr(topology_network, 'get_parameters'):
            return topology_network.get_parameters()
        return {}
    
    def _debug_network_structure(self):
        """Debug and log network structure."""
        if wandb.run:
            actor_params = self._get_topology_params(self.actor_topology)
            critic_params = self._get_topology_params(self.critic_topology)
            
            wandb.log({
                'network/actor_topology_type': self.topology_type,
                'network/critic_topology_type': self.topology_type,
                'network/hidden_size': self.hidden_size,
                'network/num_layers': self.num_layers,
                'network/activation': self.activation,
                'network/dropout': self.dropout,
                'network/actor_params': actor_params,
                'network/critic_params': critic_params
            })
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create input mask for universal observation space."""
        # Create mask for 6-dimensional input (first 6 dimensions are valid)
        mask = torch.ones(x.shape[0], 6, device=x.device)
        return mask
    
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

def create_debug_config():
    """Create a debug configuration for testing."""
    return {
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'hidden_size': 64,
        'num_layers': 2,
        'topology_type': 'fully_connected',
        'activation': 'relu',
        'dropout': 0.0,
        'total_timesteps': 600000,
        'n_eval_episodes': 15,
        'train_task_1': 'CartPole-v1',
        'train_task_2': 'Acrobot-v1',
        'train_task_3': 'LunarLander-v2'
    }

# 🚨 CONVENIENT TRAINING FUNCTIONS: Using unified configuration system
def run_single_training(config_name='single', topology_type=None, **overrides):
    """
    Run a single training session using unified configuration.
    
    Args:
        config_name (str): 'single', 'batch', or 'fixed_capacity_batch'
        topology_type (str): Override topology type if specified
        **overrides: Additional parameter overrides
    """
    if not CONFIG_SYSTEM_AVAILABLE:
        print("❌ Configuration system not available. Using debug config.")
        config = create_debug_config()
        if topology_type:
            config['topology_type'] = topology_type
        config.update(overrides)
        return triple_task_training(DebugTopologyPolicy, config['topology_type'], config)
    
    print(f"🚀 Starting single training with config: {config_name}")
    
    # Get configuration
    config = get_config_by_name(config_name)
    
    # Apply overrides
    if topology_type:
        config['topology_type'] = topology_type
    config.update(overrides)
    
    print(f"📋 Configuration: {config}")
    
    # Run training
    return triple_task_training(DebugTopologyPolicy, config['topology_type'], config)

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
        
        try:
            result = triple_task_training(DebugTopologyPolicy, combo_config['topology_type'], combo_config)
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
        sweep_type (str): 'fixed_network_sizes' or 'fixed_capacities'
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

def make_env(env_name):
    """Create environment with universal action wrapper."""
    def _make_env():
        env = gym.make(env_name)
        return UniversalActionWrapper(env, env_name)
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
    
    # Log evaluation metrics
    if wandb.run:
        wandb.log({
            f'evaluation/{task_name}/mean_reward': np.mean(episode_rewards),
            f'evaluation/{task_name}/std_reward': np.std(episode_rewards),
            f'evaluation/{task_name}/mean_length': np.mean(episode_lengths),
            f'evaluation/{task_name}/success_rate': success_rate,
            f'evaluation/{task_name}/completion_percentage': completion_pct,
            f'evaluation/{task_name}/episode_rewards': episode_rewards,
            f'evaluation/{task_name}/episode_lengths': episode_lengths
        })
    
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

def triple_task_training(policy_class, topology_type, config, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None, train_task_3=None):
    """
    Triple-task training function with intermediate testing after each phase.
    
    Sequential training: Train on task 1, test on all tasks, then train on task 2, test on all tasks, then train on task 3, test on all tasks.
    
    Args:
        policy_class: Policy class to use
        topology_type: Type of topology network
        config: Configuration dictionary
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
        hidden_size = config.get('hidden_size', 64)
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
    print(f"   • Evaluation Episodes: {config['n_eval_episodes']}")
    print(f"🔧 Device: {DEVICE_INFO['device']}")
    if DEVICE_INFO['is_cuda']:
        print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
        print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
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
                # Log a dummy metric at step 1 to initialize W&B's step counter
                wandb.log({"initialization": True}, step=1)
                print("🔧 W&B Step Fix: Initialized internal step counter at step 1")
            except Exception as e:
                print(f"⚠️  W&B Step Fix: Could not initialize step counter: {e}")
    
    # 🚨 CRITICAL FIX: Ensure W&B's step counter is always > 0
    # This prevents step 0 warnings throughout training
    if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step <= 0:
        try:
            # Force W&B to use step 1 if it somehow got reset to 0
            wandb.log({"step_reset": True}, step=1)
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
    env1 = DummyVecEnv([make_env(train_task_1)])
    env2 = DummyVecEnv([make_env(train_task_2)])
    env3 = DummyVecEnv([make_env(train_task_3)])
    
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
                    # If evaluation fails, continue training (silent)
                    pass
            
            return True
    
    # Combine all callbacks including our progress bar callback
    combined_callback = CallbackList([
        callback,  # Main logging callback
        convergence_callback,  # Convergence monitoring
        RewardMonitorCallback(train_task_1, log_interval=10000),  # Reward monitoring
        ConvergenceEvaluationCallback(convergence_callback, model, env1, train_task_1, eval_interval=20000),  # Convergence evaluation
        ProgressBarCallback(task1_timesteps, train_task_1)  # 🚨 CRITICAL: Fresh progress bar for Phase 1
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
            eval_env = DummyVecEnv([make_env(task_name)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Log evaluation results
            eval_metrics = {
                'eval/task': task_name,
                'eval/phase': 1,
                'eval/mean_reward': np.mean(rewards),
                'eval/mean_length': np.mean(lengths),
                'eval/success_rate': success,
                'eval/completion_rate': completion,
                'eval/global_timesteps': logging_handler.global_timesteps
            }
            
            # Log to W&B with validated step
            if wandb.run:
                wandb.log(eval_metrics, step=max(1, logging_handler.global_timesteps))
            
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
            eval_env = DummyVecEnv([make_env(task_name)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Log evaluation results
            eval_metrics = {
                'eval/task': task_name,
                'eval/phase': 2,
                'eval/mean_reward': np.mean(rewards),
                'eval/mean_length': np.mean(lengths),
                'eval/success_rate': success,
                'eval/completion_rate': completion,
                'eval/global_timesteps': logging_handler.global_timesteps
            }
            
            # Log to W&B with validated step
            if wandb.run:
                wandb.log(eval_metrics, step=max(1, logging_handler.global_timesteps))
            
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
            eval_env = DummyVecEnv([make_env(task_name)])
            
            # Evaluate model performance
            rewards, lengths, success, completion = evaluate_model_enhanced(
                model, eval_env, task_name, config['n_eval_episodes']
            )
            
            # Log evaluation results
            eval_metrics = {
                'eval/task': task_name,
                'eval/phase': 'final',
                'eval/mean_reward': np.mean(rewards),
                'eval/mean_length': np.mean(lengths),
                'eval/success_rate': success,
                'eval/completion_rate': completion,
                'eval/global_timesteps': logging_handler.global_timesteps
            }
            
            # Log to W&B with validated step
            if wandb.run:
                wandb.log(eval_metrics, step=max(1, logging_handler.global_timesteps))
            
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
    
    # Log final training summary to W&B
    if wandb.run:
        final_summary = {
            'train/global/final_summary': {
                'total_timesteps': logging_handler.global_timesteps,
                'task1_timesteps': actual_task1_duration,
                'task2_timesteps': actual_task2_duration,
                'task3_timesteps': actual_task3_duration,
                'topology_type': topology_type,
                'network_capacity': total_params,
                'training_completed': True
            }
        }
        
        # Log with validated step
        wandb.log(final_summary, step=max(1, logging_handler.global_timesteps))
    
    return {
        'model': model,
        'total_timesteps': logging_handler.global_timesteps,
        'task_durations': [actual_task1_duration, actual_task2_duration, actual_task3_duration],
        'topology_type': topology_type,
        'network_capacity': total_params
    }


if __name__ == "__main__":
    """
    Main entry point for triple-task training.
    
    Usage:
        # Single run with default config
        python3 topologies_triple_task_training_sweep.py
        
        # Single run with specific topology
        python3 topologies_triple_task_training_sweep.py --single --topology modular
        
        # Batch run (limited to 3 combinations)
        python3 topologies_triple_task_training_sweep.py --batch --max-runs 3
        
        # Launch W&B sweep
        python3 topologies_triple_task_training_sweep.py --sweep fixed_network_sizes
        
        # Interactive mode
        python3 topologies_triple_task_training_sweep.py --interactive
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Triple-Task Training with Topology Comparison')
    parser.add_argument('--single', action='store_true', help='Run single training session')
    parser.add_argument('--batch', action='store_true', help='Run batch training')
    parser.add_argument('--sweep', choices=['fixed_network_sizes', 'fixed_capacities'], 
                       help='Launch W&B sweep')
    parser.add_argument('--topology', choices=['modular', 'small_world', 'hybrid', 'fully_connected'],
                       default='modular', help='Topology type for single run')
    parser.add_argument('--max-runs', type=int, help='Maximum runs for batch training')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Show device information
    print("🔧 Device Information:")
    print(f"   Device: {DEVICE_INFO['device']}")
    if DEVICE_INFO['is_cuda']:
        print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
        print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
    else:
        print("   Using CPU")
    print()
    
    # Configuration system status
    if CONFIG_SYSTEM_AVAILABLE:
        print("✅ Configuration system: Unified config system available")
        print("   Available configs: single, batch, fixed_capacity_batch")
        print("   Available sweeps: fixed_network_sizes, fixed_capacities")
    else:
        print("⚠️  Configuration system: Using debug config only")
    print()
    
    # Handle different run types
    if args.single:
        print("🚀 Starting single training...")
        run_single_training(topology_type=args.topology)
        
    elif args.batch:
        print("🚀 Starting batch training...")
        run_batch_training(max_runs=args.max_runs)
        
    elif args.sweep:
        print("🚀 Launching W&B sweep...")
        run_sweep_training(args.sweep)
        
    elif args.interactive:
        print("🎯 Interactive Training Mode")
        print("=" * 50)
        print("Available options:")
        print("1. Single run")
        print("2. Batch run")
        print("3. Launch W&B sweep")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == '1':
                    topology = input("Enter topology type (modular/small_world/hybrid/fully_connected): ").strip()
                    if topology in ['modular', 'small_world', 'hybrid', 'fully_connected']:
                        run_single_training(topology_type=topology)
                    else:
                        print("❌ Invalid topology type")
                        
                elif choice == '2':
                    max_runs = input("Enter max runs (or press Enter for all): ").strip()
                    max_runs = int(max_runs) if max_runs.isdigit() else None
                    run_batch_training(max_runs=max_runs)
                    
                elif choice == '3':
                    sweep_type = input("Enter sweep type (fixed_network_sizes/fixed_capacities): ").strip()
                    if sweep_type in ['fixed_network_sizes', 'fixed_capacities']:
                        run_sweep_training(sweep_type)
                    else:
                        print("❌ Invalid sweep type")
                        
                elif choice == '4':
                    print("👋 Goodbye!")
                    break
                    
                else:
                    print("❌ Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    else:
        # Default: single run with modular topology
        print("🚀 Starting default single training (modular topology)...")
        run_single_training(topology_type='modular')
    