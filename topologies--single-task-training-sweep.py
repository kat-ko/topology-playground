#!/usr/bin/env python3
"""
Topology Networks with Weights & Biases Sweep Support

This script is a modified version of the single-task training script that can work with wandb sweeps
for hyperparameter optimization. It reads hyperparameters from wandb.config and runs training accordingly.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
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

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.networks.ffn import FeedForwardNetwork
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager

# Import the original classes and functions from the single-task training script
# (You'll need to copy the UniversalActionWrapper, DebugTopologyPolicy, etc. from the original file)

def create_sweep_config():
    """Create configuration for sweep runs using wandb.config."""
    config = {
        # ============================================================================
        # EXPERIMENT PARAMETERS (from wandb.config)
        # ============================================================================
        'tasks': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'],
        'total_timesteps': wandb.config.get('total_timesteps', 400000),
        'n_eval_episodes': wandb.config.get('n_eval_episodes', 15),
        
        # ============================================================================
        # TOPOLOGY CONFIGURATION
        # ============================================================================
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
        'topology_type': wandb.config.get('topology_type', 'small_world'),
        
        # Layer configuration based on sweep parameters
        'layer_configs': {
            'small_world': [wandb.config.get('num_layers', 1)],
            'modular': [wandb.config.get('num_layers', 1)],
            'hybrid': [wandb.config.get('num_layers', 1)],
            'fully_connected': [wandb.config.get('num_layers', 1)]
        },
        
        # ============================================================================
        # NETWORK DIMENSIONS
        # ============================================================================
        'universal_input_dim': 6,
        'universal_output_dim': 3,
        'universal_action_dim': 3,
        'hidden_size': wandb.config.get('hidden_size', 128),
        'network_sizes': [wandb.config.get('hidden_size', 128)],
        'network_types': ['ffn'],
        'num_io_nodes': 4,
        
        # ============================================================================
        # EXPERIMENT TYPES
        # ============================================================================
        'experiment_types': ['same_size'],
        'capacity_matching_reference': 'small_world',
        
        # ============================================================================
        # TOPOLOGY-SPECIFIC PARAMETERS (from wandb.config)
        # ============================================================================
        'topology_params': {
            'small_world': {
                'k': wandb.config.get('small_world_k', 4),
                'p': wandb.config.get('small_world_p', 0.3),
                'inter_layer_prob': 0.5
            },
            'modular': {
                'num_modules': wandb.config.get('modular_num_modules', 4),
                'inter_module_prob': wandb.config.get('modular_inter_module_prob', 0.2),
                'intra_module_prob': wandb.config.get('modular_intra_module_prob', 0.8),
                'inter_layer_prob': 0.5
            },
            'hybrid': {
                'num_modules': wandb.config.get('hybrid_num_modules', 4),
                'k': wandb.config.get('hybrid_k', 4),
                'p': wandb.config.get('hybrid_p', 0.3),
                'inter_module_prob': wandb.config.get('hybrid_inter_module_prob', 0.2),
                'inter_layer_prob': 0.5
            },
            'fully_connected': {
                'inter_layer_prob': 1.0,
                'intra_layer_prob': 1.0
            }
        },
        
        # ============================================================================
        # NETWORK PARAMETERS (from wandb.config)
        # ============================================================================
        'network_params': {
            'ffn': {
                'activation': wandb.config.get('activation', 'relu'),
                'dropout': wandb.config.get('dropout', 0.0)
            }
        },
        
        # ============================================================================
        # PPO TRAINING PARAMETERS (from wandb.config)
        # ============================================================================
        'ppo_params': {
            'learning_rate': wandb.config.get('learning_rate', 3e-4),
            'n_steps': wandb.config.get('n_steps', 2048),
            'batch_size': wandb.config.get('batch_size', 128),
            'n_epochs': wandb.config.get('n_epochs', 5),
            'gamma': wandb.config.get('gamma', 0.99),
            'gae_lambda': wandb.config.get('gae_lambda', 0.95),
            'clip_range': wandb.config.get('clip_range', 0.2),
            'ent_coef': wandb.config.get('ent_coef', 0.05),
            'max_grad_norm': wandb.config.get('max_grad_norm', 0.5)
        },
        
        # ============================================================================
        # CAPACITY MATCHING CONFIGURATION
        # ============================================================================
        'capacity_matching': {
            'enabled': True,
            'use_capacity_mapping': False,
            'min_search_size': 10,
            'max_search_size': 2000,
            'seeds': [42]
        },
        
        # ============================================================================
        # PARAMETER BUDGET CONFIGURATION
        # ============================================================================
        'parameter_budget': {
            'enabled': True,
            'budget_type': 'weights',
            'padding_strategy': 'random'
        },
        
        # ============================================================================
        # NODE SELECTION CONFIGURATION
        # ============================================================================
        'node_selection_strategies': ['random']
    }
    
    # Add backward compatibility aliases
    config['use_capacity_matching'] = config['capacity_matching']['enabled']
    config['use_capacity_mapping'] = config['capacity_matching']['use_capacity_mapping']
    config['min_search_size'] = config['capacity_matching']['min_search_size']
    config['max_search_size'] = config['capacity_matching']['max_search_size']
    config['seeds'] = config['capacity_matching']['seeds']
    
    # Add layer configuration aliases
    config['num_layers'] = [wandb.config.get('num_layers', 1)]
    config['fc_num_layers'] = config['layer_configs']['fully_connected']
    
    # Add topology-specific parameter aliases
    config['small_world_params'] = config['topology_params']['small_world']
    config['modular_params'] = config['topology_params']['modular']
    config['hybrid_params'] = config['topology_params']['hybrid']
    config['fully_connected_params'] = config['topology_params']['fully_connected']
    
    return config

def train_with_sweep():
    """Main training function for wandb sweep."""
    
    # Initialize wandb run
    wandb.init(
        entity="katko-it-universitetet-i-k-benhavn",
        project="topologies--hyperparameter-optimization",
        config={
            # These will be overridden by sweep parameters
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 128,
            'n_epochs': 5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.05,
            'max_grad_norm': 0.5,
            'hidden_size': 128,
            'num_layers': 1,
            'topology_type': 'small_world',
            'train_task': 'CartPole-v1',
            'total_timesteps': 400000,
            'n_eval_episodes': 15,
            'activation': 'relu',
            'dropout': 0.0,
            # Topology-specific parameters
            'small_world_k': 4,
            'small_world_p': 0.3,
            'modular_num_modules': 4,
            'modular_inter_module_prob': 0.2,
            'modular_intra_module_prob': 0.8,
            'hybrid_num_modules': 4,
            'hybrid_k': 4,
            'hybrid_p': 0.3,
            'hybrid_inter_module_prob': 0.2,
        }
    )
    
    print(f"🎯 Starting sweep run with configuration:")
    print(f"   • Topology: {wandb.config.topology_type}")
    print(f"   • Hidden size: {wandb.config.hidden_size}")
    print(f"   • Layers: {wandb.config.num_layers}")
    print(f"   • Learning rate: {wandb.config.learning_rate}")
    print(f"   • Train task: {wandb.config.train_task}")
    print(f"   • Total timesteps: {wandb.config.total_timesteps}")
    
    # Create configuration from wandb.config
    config = create_sweep_config()
    
    # Get training task from sweep
    train_task = wandb.config.train_task
    
    # Run single-task training with cross-task evaluation
    try:
        # Import the cross_task_testing function from the original script
        # You'll need to copy this function here or import it
        result = cross_task_testing(
            DebugTopologyPolicy,  # You'll need to copy this class
            wandb.config.topology_type,
            config,
            num_layers=wandb.config.num_layers,
            hidden_size=wandb.config.hidden_size,
            train_task=train_task
        )
        
        # Log the best result for the sweep metric
        best_reward = 0.0
        for task in config['tasks']:
            task_reward = result.get(f'{task}_final_mean_reward', 0.0)
            if task_reward > best_reward:
                best_reward = task_reward
        
        # Log sweep metric
        wandb.log({
            'testing/mean_reward': best_reward,
            'sweep/best_reward': best_reward,
            'sweep/topology_type': wandb.config.topology_type,
            'sweep/hidden_size': wandb.config.hidden_size,
            'sweep/num_layers': wandb.config.num_layers,
            'sweep/learning_rate': wandb.config.learning_rate,
            'sweep/train_task': wandb.config.train_task,
        })
        
        print(f"✅ Sweep run completed successfully!")
        print(f"   • Best reward: {best_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Error in sweep run: {e}")
        import traceback
        traceback.print_exc()
        
        # Log error for sweep
        wandb.log({
            'testing/mean_reward': -1000,  # Penalty for failed runs
            'sweep/error': str(e)
        })
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    train_with_sweep() 