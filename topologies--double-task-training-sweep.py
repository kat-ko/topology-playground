#!/usr/bin/env python3
"""
Double-Task Topology Networks with Weights & Biases Sweep Support

This script is a modified version of the double-task training script that can work with wandb sweeps
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

# Import topology modules
from src.topologies import (
    SmallWorldTopology, ModularTopology, HybridTopology, FullyConnectedTopology
)
from src.networks.ffn import FeedForwardNetwork
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager
from src.utils.capacity_matching_helper import pre_calculate_capacity_matching
from src.utils.task_normalization import (
    compute_multi_task_metrics, log_normalized_metrics, print_normalized_summary,
    get_task_thresholds, get_normalization_constants
)
from src.utils.advanced_plotting import (
    log_comprehensive_plots_for_run, create_multi_phase_learning_curves
)
from src.utils.task_training_config import get_task_timesteps, create_convergence_callback

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
    Debug Topology Policy for double-task training with sweep support.
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
                input_size=6,  # Universal observation space
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                num_layers=self.num_layers,
                activation=self.activation,
                dropout=self.dropout
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
                input_size=6,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                num_layers=self.num_layers,
                num_modules=num_modules,
                inter_module_prob=inter_prob,
                intra_module_prob=intra_prob,
                activation=self.activation,
                dropout=self.dropout
            )
        elif self.topology_type == 'hybrid':
            num_modules = getattr(wandb.config, 'hybrid_num_modules', 4) if wandb.run else 4
            k = getattr(wandb.config, 'hybrid_k', 4) if wandb.run else 4
            p = getattr(wandb.config, 'hybrid_p', 0.2) if wandb.run else 0.2
            inter_prob = getattr(wandb.config, 'hybrid_inter_module_prob', 0.1) if wandb.run else 0.1
            return HybridTopology(
                input_size=6,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                num_layers=self.num_layers,
                num_modules=num_modules,
                k=k,
                p=p,
                inter_module_prob=inter_prob,
                activation=self.activation,
                dropout=self.dropout
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
# CALLBACK FOR WANDB INTEGRATION
# ============================================================================

class EnhancedDebugCallback(BaseCallback):
    """Enhanced callback for tracking training progress with wandb integration and sequential training support."""
    
    def __init__(self, verbose=0, wandb_run=None, log_freq=100):
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        self.step_count = 0
        self.rollout_count = 0
        
        # Global step counter for wandb (continues across phases)
        self.global_timesteps = 0
        
        # Sequential training tracking
        self.current_task_phase = 0
        self.task_phases = []
        self.phase_start_timesteps = []
        self.phase_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'learning_rates': []
        }
        
        # Overall training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'learning_rates': []
        }
        
        # Task-specific reward tracking for normalized metrics
        self.task_rewards = {
            'CartPole-v1': [],
            'Acrobot-v1': [],
            'LunarLander-v2': []
        }
        self.current_task = None
    
    def set_task_phase(self, task_name, phase_number):
        """Set the current task phase for sequential training."""
        self.current_task_phase = phase_number
        self.current_task = task_name  # Track current task for reward collection
        
        # Initialize global timesteps if this is the first phase
        if phase_number == 0:
            self.global_timesteps = 0
        
        self.task_phases.append({
            'phase': phase_number,
            'task': task_name,
            'start_timesteps': self.global_timesteps
        })
        self.phase_start_timesteps.append(self.global_timesteps)
        
        # Reset phase-specific metrics
        self.phase_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'learning_rates': []
        }
        
        if wandb.run:
            wandb.log({
                'sequential_training/phase_start': phase_number,
                'sequential_training/current_task': task_name,
                'sequential_training/total_timesteps': self.global_timesteps
            }, step=self.global_timesteps)
    
    def _on_step(self) -> bool:
        """Log metrics on each step."""
        self.step_count += 1
        self.global_timesteps += 1
        
        if self.num_timesteps % self.log_freq == 0 and wandb.run:
            self._log_training_metrics()
            
            # Log overall training metrics
            wandb.log({
                'training/timesteps': self.num_timesteps,
                'training/episodes': len(self.episode_rewards),
                'training/mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'training/mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
                'sequential_training/current_phase': self.current_task_phase,
                'sequential_training/phase_timesteps': self.num_timesteps - (self.phase_start_timesteps[-1] if self.phase_start_timesteps else 0)
            }, step=self.global_timesteps)
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout."""
        self.rollout_count += 1
        
        # Collect episodic rewards for current task
        if self.current_task and self.current_task in self.task_rewards:
            # Get the latest episode rewards from the rollout
            if hasattr(self, 'episode_rewards') and self.episode_rewards:
                latest_rewards = self.episode_rewards[-self.n_envs:] if hasattr(self, 'n_envs') else self.episode_rewards[-1:]
                self.task_rewards[self.current_task].extend(latest_rewards)
        
        if wandb.run:
            self._log_rollout_metrics()
            
            # Log overall rollout metrics
            wandb.log({
                'rollout/mean_reward': np.mean(self.episode_rewards[-self.n_envs:]) if self.episode_rewards else 0,
                'rollout/mean_length': np.mean(self.episode_lengths[-self.n_envs:]) if self.episode_lengths else 0,
                'sequential_training/phase': self.current_task_phase
            })
    
    def _on_training_end(self) -> None:
        """Log final training summary."""
        if wandb.run:
            self._log_final_training_summary()
            
            # Log sequential training summary
            if self.task_phases:
                wandb.log({
                    'sequential_training/total_phases': len(self.task_phases),
                    'sequential_training/final_phase': self.current_task_phase,
                    'sequential_training/total_timesteps': self.num_timesteps
                })
    
    def _log_training_metrics(self):
        """Log detailed training metrics with phase tracking."""
        try:
            # Get metrics from the model's logger
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = self.model.logger.name_to_value
                
                # Base metrics
                metrics = {
                    "train/step": self.step_count,
                    "train/total_timesteps": self.num_timesteps,
                    "train/rollout_count": self.rollout_count,
                    "sequential_training/phase": self.current_task_phase,
                }
                
                # Add specific PPO metrics if available
                for key, value in name_to_value.items():
                    if any(term in key.lower() for term in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip', 'explained']):
                        metrics[f"train/{key}"] = value
                        # Also log phase-specific metrics
                        metrics[f"phase_{self.current_task_phase}/{key}"] = value
                
                # Add learning rate if available
                if hasattr(self.model, 'lr_schedule'):
                    current_lr = self.model.lr_schedule(self.num_timesteps)
                    metrics["train/learning_rate"] = current_lr
                    metrics[f"phase_{self.current_task_phase}/learning_rate"] = current_lr
                
                # Add network-specific metrics if available
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                    actor_params = self.model.policy._get_topology_params(self.model.policy.actor_topology)
                    critic_params = self.model.policy._get_topology_params(self.model.policy.critic_topology)
                    metrics.update({
                        "network/actor_parameters": actor_params,
                        "network/critic_parameters": critic_params,
                    })
                    
                    # Calculate total parameters if both are dictionaries with 'size' key
                    if isinstance(actor_params, dict) and isinstance(critic_params, dict):
                        actor_size = actor_params.get('size', 0)
                        critic_size = critic_params.get('size', 0)
                        total_params = actor_size + critic_size
                        metrics["network/total_parameters"] = total_params
                    
                    # Add enhanced graph metrics
                    self._log_graph_metrics()
                    self._log_depth_analysis()
                    self._log_sample_efficiency()
                    self._log_hyperparameter_correlation()
                
                wandb.log(metrics, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging training metrics: {e}")
    
    def _log_rollout_metrics(self):
        """Log metrics at the end of each rollout with phase tracking."""
        try:
            # Get rollout statistics
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                buffer = self.model.rollout_buffer
                
                # Calculate rollout statistics
                if hasattr(buffer, 'observations') and buffer.observations is not None:
                    obs_mean = np.mean(buffer.observations)
                    obs_std = np.std(buffer.observations)
                    
                    metrics = {
                        'rollout/obs_mean': obs_mean,
                        'rollout/obs_std': obs_std,
                        'sequential_training/phase': self.current_task_phase,
                    }
                    
                    # Add phase-specific rollout metrics
                    if self.task_phases:
                        current_task = self.task_phases[-1]['task']
                        metrics[f'phase_{self.current_task_phase}/rollout_obs_mean'] = obs_mean
                        metrics[f'phase_{self.current_task_phase}/rollout_obs_std'] = obs_std
                        metrics[f'phase_{self.current_task_phase}/task'] = current_task
                    
                    wandb.log(metrics, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging rollout metrics: {e}")
    
    def _log_final_training_summary(self):
        """Log final training summary with sequential training details."""
        try:
            # Calculate overall statistics
            if self.episode_rewards:
                final_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                final_lengths = self.episode_lengths[-100:]
                
                summary = {
                    'final/mean_reward': np.mean(final_rewards),
                    'final/std_reward': np.std(final_rewards),
                    'final/mean_length': np.mean(final_lengths),
                    'final/std_length': np.std(final_lengths),
                    'final/total_episodes': len(self.episode_rewards),
                    'final/total_timesteps': self.num_timesteps,
                    'sequential_training/total_phases': len(self.task_phases),
                }
                
                # Add phase-specific final metrics
                if self.task_phases:
                    for i, phase in enumerate(self.task_phases):
                        summary[f'phase_{i}/task'] = phase['task']
                        summary[f'phase_{i}/start_timesteps'] = phase['start_timesteps']
                
                wandb.log(summary, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging final summary: {e}")
    
    def _log_graph_metrics(self):
        """Log graph metrics with phase tracking."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                actor_topology = self.model.policy.actor_topology
                critic_topology = self.model.policy.critic_topology
                
                # Generate graphs from topology objects
                actor_graph = actor_topology.generate() if hasattr(actor_topology, 'generate') else None
                critic_graph = critic_topology.generate() if hasattr(critic_topology, 'generate') else None
                
                # Actor metrics
                actor_metrics = self._calculate_graph_metrics(actor_graph, 'actor')
                for key, value in actor_metrics.items():
                    wandb.log({f'graph/actor/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.global_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/actor/{key}': value}, step=self.global_timesteps)
                
                # Critic metrics
                critic_metrics = self._calculate_graph_metrics(critic_graph, 'critic')
                for key, value in critic_metrics.items():
                    wandb.log({f'graph/critic/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.global_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/critic/{key}': value}, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging graph metrics: {e}")
    
    def _calculate_graph_metrics(self, G, network_type):
        """Calculate graph metrics for the network."""
        try:
            if G is None:
                return {}
            
            # Convert to undirected for metrics that don't support directed graphs
            G_undirected = G.to_undirected() if G.is_directed() else G
            
            metrics = {
                'clustering_coefficient': nx.average_clustering(G_undirected),
                'density': nx.density(G),
                'avg_degree': np.mean([d for n, d in G.degree()]),
                'diameter': nx.diameter(G_undirected),
                'avg_shortest_path': nx.average_shortest_path_length(G_undirected),
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            }
            
            return metrics
        except Exception as e:
            print(f"   ⚠️  Error calculating graph metrics: {e}")
            return {}
    
    def _log_depth_analysis(self):
        """Log depth analysis with phase tracking."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                actor_topology = self.model.policy.actor_topology
                critic_topology = self.model.policy.critic_topology
                
                # Generate graphs from topology objects
                actor_graph = actor_topology.generate() if hasattr(actor_topology, 'generate') else None
                critic_graph = critic_topology.generate() if hasattr(critic_topology, 'generate') else None
                
                # Actor depth analysis
                actor_depth = self._calculate_depth_metrics(actor_graph, 'actor')
                for key, value in actor_depth.items():
                    wandb.log({f'depth/actor/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.global_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/actor_depth/{key}': value}, step=self.global_timesteps)
                
                # Critic depth analysis
                critic_depth = self._calculate_depth_metrics(critic_graph, 'critic')
                for key, value in critic_depth.items():
                    wandb.log({f'depth/critic/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.global_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/critic_depth/{key}': value}, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging depth analysis: {e}")
    
    def _calculate_depth_metrics(self, G, network_type):
        """Calculate depth metrics for the network."""
        try:
            if G is None:
                return {}
            
            # Calculate depth-related metrics
            if nx.is_directed_acyclic_graph(G):
                # For DAGs, calculate longest path
                longest_path = nx.dag_longest_path(G)
                depth = len(longest_path) - 1  # Number of edges in longest path
            else:
                # For non-DAGs, use diameter as approximation
                depth = nx.diameter(G.to_undirected())
            
            metrics = {
                'depth': depth,
                'max_depth': depth,
                'avg_depth': depth,  # Simplified for now
            }
            
            return metrics
        except Exception as e:
            print(f"   ⚠️  Error calculating depth metrics: {e}")
            return {}
    
    def _log_sample_efficiency(self):
        """Log sample efficiency metrics with phase tracking."""
        try:
            if self.episode_rewards:
                # Calculate sample efficiency metrics
                recent_rewards = self.episode_rewards[-50:]  # Last 50 episodes
                if len(recent_rewards) >= 10:
                    sample_efficiency = np.mean(recent_rewards) / max(1, len(recent_rewards))
                    
                    wandb.log({
                        'sample_efficiency/recent_mean_reward': np.mean(recent_rewards),
                        'sample_efficiency/efficiency_score': sample_efficiency,
                        'sequential_training/phase': self.current_task_phase
                    }, step=self.global_timesteps)
                    
                    wandb.log({
                        f'phase_{self.current_task_phase}/sample_efficiency/recent_mean_reward': np.mean(recent_rewards),
                        f'phase_{self.current_task_phase}/sample_efficiency/efficiency_score': sample_efficiency
                    }, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging sample efficiency: {e}")
    
    def _log_hyperparameter_correlation(self):
        """Log hyperparameter correlation metrics with phase tracking."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                # Get current hyperparameters
                current_lr = self.model.lr_schedule(self.num_timesteps) if hasattr(self.model, 'lr_schedule') else 0
                
                # Calculate correlation metrics (simplified)
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-20:]  # Last 20 episodes
                    if len(recent_rewards) >= 5:
                        reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0]
                        
                        wandb.log({
                            'hyperparameter_correlation/learning_rate': current_lr,
                            'hyperparameter_correlation/reward_trend': reward_trend,
                            'sequential_training/phase': self.current_task_phase
                        }, step=self.global_timesteps)
                        
                        wandb.log({
                            f'phase_{self.current_task_phase}/hyperparameter_correlation/learning_rate': current_lr,
                            f'phase_{self.current_task_phase}/hyperparameter_correlation/reward_trend': reward_trend
                        }, step=self.global_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging hyperparameter correlation: {e}")

    def get_task_rewards(self):
        """Get collected task rewards for normalized metrics calculation."""
        return self.task_rewards.copy()

# ============================================================================
# IMPROVED LOGGING SYSTEM WITH REWARD SCALING
# ============================================================================

def initialize_wandb_run(config, topology_type, training_type='double_task'):
    """Initialize wandb with proper naming and configuration."""
    
    # Create descriptive run name
    run_name = create_run_name(config, topology_type, training_type)
    
    # Create tags for easy filtering
    tags = create_run_tags(config, topology_type, training_type)
    
    # Initialize wandb
    wandb.init(
        project="topology-playground",
        entity="katko-it-universitetet-i-k-benhavn",
        config=config,
        name=run_name,
        tags=tags
    )

def create_run_name(config, topology_type, training_type):
    """Create descriptive run name."""
    
    # Base name
    name_parts = [training_type, topology_type]
    
    # Add capacity or size information
    if 'target_capacity' in config:
        name_parts.append(f"cap{config.get('target_capacity')}")
    elif 'hidden_size' in config:
        name_parts.append(f"size{config.get('hidden_size')}")
    
    # Add task information
    if training_type == 'double_task':
        task_order = config.get('task_order', 'LunarLander-v2_CartPole-v1')
        tasks = task_order.split('_')
        name_parts.extend(tasks)
    
    return "_".join(name_parts)

def create_run_tags(config, topology_type, training_type):
    """Create tags for easy filtering and organization."""
    
    tags = [
        training_type,
        topology_type,
        f"capacity_{config.get('target_capacity', 'variable')}" if 'target_capacity' in config else f"size_{config.get('hidden_size', 'variable')}",
        "comparison_sweep" if 'target_capacity' in config or config.get('hidden_size') in [64, 128, 256, 512] else "optimization_sweep",
        "normalized_metrics"  # Indicate that normalized metrics are used
    ]
    
    # Add task tags
    if training_type == 'double_task':
        task_order = config.get('task_order', 'LunarLander-v2_CartPole-v1')
        tasks = task_order.split('_')
        tags.extend(tasks)
    
    return tags

def log_baseline_results(wandb_run, baseline_results, topology_type):
    """Log baseline evaluation results with hierarchical structure and normalization."""
    
    for task, results in baseline_results.items():
        # Raw metrics
        wandb_run.log({
            f'baseline/{task}/raw/mean_reward': results['mean_reward'],
            f'baseline/{task}/raw/success_rate': results['success_rate'],
            f'baseline/{task}/raw/mean_length': np.mean(results['lengths']),
            f'baseline/{task}/raw/std_reward': np.std(results['rewards']),
            f'baseline/{task}/raw/std_length': np.std(results['lengths'])
        })
        
        # Normalized metrics
        normalized_reward = normalize_reward(results['mean_reward'], task)
        wandb_run.log({
            f'baseline/{task}/normalized/reward': normalized_reward,
            f'baseline/{task}/normalized/efficiency': results.get('efficiency_score', 1.0)
        })

def log_phase_results(wandb_run, phase_results, phase_idx, topology_type, task_order=None):
    """Log results after each training phase with topology-aware naming."""
    
    for task, results in phase_results.items():
        context = results['context']
        
        # Topology-aware metric names for easy comparison
        base_path = f"{topology_type}/{task_order}/{context}" if task_order else f"{topology_type}/phase{phase_idx}"
        
        # Raw metrics with topology context
        wandb_run.log({
            f'{base_path}/{task}/raw/mean_reward': results['mean_reward'],
            f'{base_path}/{task}/raw/success_rate': results['success_rate'],
            f'{base_path}/{task}/raw/mean_length': np.mean(results['lengths']),
            f'{base_path}/{task}/raw/std_reward': np.std(results['rewards']),
            f'{base_path}/{task}/raw/std_length': np.std(results['lengths'])
        })
        
        # Normalized metrics with topology context
        normalized_reward = (normalize_rewardresults['mean_reward'], task)
        wandb_run.log({
            f'{base_path}/{task}/normalized/reward': normalized_reward,
            f'{base_path}/{task}/normalized/efficiency': results.get('efficiency_score', 1.0),
            f'{base_path}/{task}/normalized/steps_to_threshold': results.get('steps_to_threshold', 0)
        })
        
        # Legacy phase-based metrics for backward compatibility
        wandb_run.log({
            f'phase{phase_idx}/{task}/{context}/raw/mean_reward': results['mean_reward'],
            f'phase{phase_idx}/{task}/{context}/raw/success_rate': results['success_rate'],
            f'phase{phase_idx}/{task}/{context}/raw/mean_length': np.mean(results['lengths']),
            f'phase{phase_idx}/{task}/{context}/raw/std_reward': np.std(results['rewards']),
            f'phase{phase_idx}/{task}/{context}/raw/std_length': np.std(results['lengths']),
            f'phase{phase_idx}/{task}/{context}/normalized/reward': normalized_reward,
            f'phase{phase_idx}/{task}/{context}/normalized/efficiency': results.get('efficiency_score', 1.0),
            f'phase{phase_idx}/{task}/{context}/normalized/steps_to_threshold': results.get('steps_to_threshold', 0)
        })
        
        # Add task order context if provided
        if task_order:
            wandb_run.log({
                f'phase{phase_idx}/task_order': task_order,
                f'phase{phase_idx}/current_task': task,
                f'{base_path}/task_order': task_order,
                f'{base_path}/current_task': task
            })

def log_normalized_metrics(wandb_run, task_metrics, phase_idx, topology_type, task_order=None):
    """Log comprehensive normalized metrics with topology context."""
    
    base_path = f"{topology_type}/{task_order}" if task_order else f"{topology_type}/phase{phase_idx}"
    
    # Task-specific normalized metrics with topology context
    for task, metrics in task_metrics.items():
        wandb_run.log({
            f'{base_path}/normalized/{task}/normalized_reward': metrics['normalized_reward'],
            f'{base_path}/normalized/{task}/steps_to_threshold': metrics['steps_to_threshold'],
            f'{base_path}/normalized/{task}/final_reward': metrics['final_reward'],
            f'{base_path}/normalized/{task}/rolling_mean_final': metrics['rolling_mean_final']
        })
    
    # Aggregated normalized metrics with topology context
    final_normalized_score = np.mean([metrics['normalized_reward'] for metrics in task_metrics.values()])
    efficiency_score = np.mean([metrics['steps_to_threshold'] for metrics in task_metrics.values()])
    
    wandb_run.log({
        f'{base_path}/normalized/final_normalized_score': final_normalized_score,
        f'{base_path}/normalized/efficiency_score': efficiency_score
    })
    
    # Legacy metrics for backward compatibility
    wandb_run.log({
        f'normalized/phase{phase_idx}/final_normalized_score': final_normalized_score,
        f'normalized/phase{phase_idx}/efficiency_score': efficiency_score
    })

def log_transfer_metrics(wandb_run, transfer_metrics, phase_idx, topology_type, task_order=None):
    """Log transfer learning metrics with topology context."""
    
    base_path = f"{topology_type}/{task_order}/transfer" if task_order else f"{topology_type}/phase{phase_idx}/transfer"
    
    for metric_name, value in transfer_metrics.items():
        # Topology-aware transfer metrics
        wandb.log({
            f'{base_path}/{metric_name}': value
        })
        
        # Legacy metrics for backward compatibility
        wandb.log({
            f'phase{phase_idx}/transfer/{metric_name}': value
        })
    
    # Normalized transfer metrics with topology context
    if 'forward_transfer_score' in transfer_metrics:
        wandb.log({
            f'{base_path}/normalized_forward_transfer': transfer_metrics['forward_transfer_score']
        })
    
    if 'backward_transfer_score' in transfer_metrics:
        wandb.log({
            f'{base_path}/normalized_backward_transfer': transfer_metrics['backward_transfer_score']
        })

def log_final_analysis(wandb_run, final_analysis, topology_type, task_order=None):
    """Log final comprehensive analysis with topology context."""
    
    base_path = f"{topology_type}/{task_order}/final" if task_order else f"{topology_type}/final"
    
    # Raw performance metrics with topology context
    wandb.log({
        f'{base_path}/raw/mean_reward': final_analysis['final_mean_reward'],
        f'{base_path}/raw/success_rate': final_analysis['final_success_rate'],
        f'{base_path}/raw/mean_length': final_analysis['final_mean_length']
    })
    
    # Normalized performance metrics with topology context
    wandb.log({
        f'{base_path}/normalized/final_normalized_score': final_analysis['final_normalized_score'],
        f'{base_path}/normalized/efficiency_score': final_analysis['efficiency_score'],
        f'{base_path}/normalized/parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0)
    })
    
    # Transfer learning summary with topology context
    wandb.log({
        f'{base_path}/transfer/normalized_forward_transfer_score': final_analysis['forward_transfer_score'],
        f'{base_path}/transfer/normalized_backward_transfer_score': final_analysis['backward_transfer_score'],
        f'{base_path}/transfer/normalized_total_transfer_score': final_analysis['total_transfer_score']
    })
    
    # Topology-specific normalized metrics
    wandb.log({
        f'{base_path}/topology/normalized_parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'{base_path}/topology/normalized_learning_stability': final_analysis.get('learning_stability', 0.0)
    })
    
    # Legacy metrics for backward compatibility
    wandb.log({
        'final/raw/mean_reward': final_analysis['final_mean_reward'],
        'final/raw/success_rate': final_analysis['final_success_rate'],
        'final/raw/mean_length': final_analysis['final_mean_length'],
        'final/normalized/final_normalized_score': final_analysis['final_normalized_score'],
        'final/normalized/efficiency_score': final_analysis['efficiency_score'],
        'final/normalized/parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'final/transfer/{topology_type}/normalized_forward_transfer_score': final_analysis['forward_transfer_score'],
        f'final/transfer/{topology_type}/normalized_backward_transfer_score': final_analysis['backward_transfer_score'],
        f'final/transfer/{topology_type}/normalized_total_transfer_score': final_analysis['total_transfer_score'],
        f'final/topology/{topology_type}/normalized_parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'final/topology/{topology_type}/normalized_learning_stability': final_analysis.get('learning_stability', 0.0)
    })

# ============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ============================================================================

def create_debug_config():
    """Create configuration for double-task training with sweep support."""
    # Get hyperparameters from wandb config if available
    if wandb.run:
        config = wandb.config
    else:
        # Default configuration for testing
        config = {
            'learning_rate': 3e-4,
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
            'train_task_2': 'Acrobot-v1'
        }
    
    return config

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
        obs, _ = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            episode_length += 1
            done = done or truncated
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths

def evaluate_model_enhanced(model, env, task_name, n_eval_episodes=3):
    """Enhanced evaluation with task-specific metrics."""
    episode_rewards, episode_lengths = evaluate_model(model, env, n_eval_episodes)
    
    # Calculate task-specific success rate and completion percentage
    from src.utils.task_normalization import calculate_success_rate_with_completion
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

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def double_task_training(policy_class, topology_type, config, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None):
    """
    Double-task training function with intermediate testing after each phase.
    
    Sequential training: Train on task 1, test on all tasks, then train on task 2, test on all tasks.
    
    Args:
        policy_class: Policy class to use
        topology_type: Type of topology network
        config: Configuration dictionary
        num_layers: Number of layers
        hidden_size: Hidden layer size
        train_task_1: First training task
        train_task_2: Second training task
    """
    print(f"🎯 DOUBLE-TASK SEQUENTIAL TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"   • Task 1: {train_task_1}")
    print(f"   • Task 2: {train_task_2}")
    print(f"   • Hidden Size: {hidden_size}")
    print(f"   • Layers: {num_layers}")
    
    # Initialize wandb if not already done
    if wandb.run is None:
        wandb.init(
            project="topologies--double-task-training",
            entity="katko-it-universitetet-i-k-benhavn",
            config=config,
            name=f"double_task_{topology_type}_{train_task_1}_{train_task_2}"
        )
    
    # Create task order string for topology-aware logging
    task_order = f"{train_task_1}_{train_task_2}"
    
    # Create environments for sequential training
    env1 = DummyVecEnv([make_env(train_task_1)])
    env2 = DummyVecEnv([make_env(train_task_2)])
    
    # Create ONE model for sequential training
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
        verbose=1,
        policy_kwargs={
            'topology_type': topology_type,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'config': config
        }
    )
    
    # Create callback
    callback = EnhancedDebugCallback(wandb_run=wandb.run, log_freq=1000)
    
    # ============================================================================
    # PHASE 1: Train on task 1
    # ============================================================================
    print(f"🚀 Phase 1: Training on {train_task_1}...")
    callback.set_task_phase(train_task_1, 1)  # Set phase 1
    
    # Get task-specific training configuration
    task1_timesteps = get_task_timesteps(train_task_1, config)
    convergence_callback = create_convergence_callback(train_task_1, config)
    
    print(f"📋 Task-specific training: {train_task_1} for {task1_timesteps:,} timesteps")
    
    # Create a callback that integrates convergence monitoring with periodic evaluation
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
                    rewards, lengths, success = evaluate_model_enhanced(
                        self.model, self.env, self.task_name, 5  # Quick eval with 5 episodes
                    )
                    mean_reward = np.mean(rewards)
                    
                    # Update convergence callback with evaluation results
                    self.convergence_callback.update_with_evaluation(mean_reward, success)
                    
                    if self.convergence_callback.verbose > 0:
                        print(f"📊 {self.task_name}: Quick eval - Reward: {mean_reward:.2f}, Success: {success:.1%}, Completion: {completion:.1f}%")
                
                except Exception as e:
                    # If evaluation fails, continue training
                    if self.convergence_callback.verbose > 0:
                        print(f"⚠️  {self.task_name}: Evaluation failed: {e}")
            
            return True
    
    # Combine callbacks (convergence callback will handle early stopping)
    combined_callback = [callback, convergence_callback, ConvergenceEvaluationCallback(convergence_callback, model, env1, train_task_1)]
    model.learn(total_timesteps=task1_timesteps, callback=combined_callback)
    
    # ============================================================================
    # INTERMEDIATE TESTING: Test on all tasks after Phase 1
    # ============================================================================
    print(f"📊 Phase 1 Testing: Evaluating on all tasks after training on {train_task_1}...")
    
    # Test on all available tasks
    all_tasks = ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
    phase1_results = {}
    
    for task in all_tasks:
        eval_env = make_env(task)()
        rewards, lengths, success, completion = evaluate_model_enhanced(
            model, eval_env, task, config['n_eval_episodes']
        )
        phase1_results[task] = {
            'rewards': rewards,
            'lengths': lengths,
            'success_rate': success,
            'completion_percentage': completion,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }
        eval_env.close()
        
        # Print immediate results for this task
        print(f"   • {task}: {phase1_results[task]['mean_reward']:.2f} ± {phase1_results[task]['std_reward']:.2f} "
              f"(Success: {phase1_results[task]['success_rate']:.1%}, Completion: {phase1_results[task]['completion_percentage']:.1f}%)")
    
    # Log Phase 1 results with topology-aware naming
    if wandb.run:
        for task, results in phase1_results.items():
            wandb.log({
                f'{topology_type}/{task_order}/phase1/testing/{task}/mean_reward': results['mean_reward'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/std_reward': results['std_reward'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/mean_length': results['mean_length'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/std_length': results['std_length'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/success_rate': results['success_rate'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/completion_percentage': results['completion_percentage'],
                f'{topology_type}/{task_order}/phase1/testing/{task}/n_eval_episodes': config['n_eval_episodes'],
                
                # Legacy metrics for backward compatibility
                f'phase1/{task}/testing/mean_reward': results['mean_reward'],
                f'phase1/{task}/testing/success_rate': results['success_rate'],
            })
    
    # ============================================================================
    # PHASE 2: Train on task 2
    # ============================================================================
    print(f"🚀 Phase 2: Continuing training on {train_task_2}...")
    callback.set_task_phase(train_task_2, 2)  # Set phase 2
    model.set_env(env2)  # Switch environment for second task
    
    # Get task-specific training configuration
    task2_timesteps = get_task_timesteps(train_task_2, config)
    convergence_callback = create_convergence_callback(train_task_2, config)
    
    print(f"📋 Task-specific training: {train_task_2} for {task2_timesteps:,} timesteps")
    
    # Combine callbacks (convergence callback will handle early stopping)
    combined_callback = [callback, convergence_callback, ConvergenceEvaluationCallback(convergence_callback, model, env2, train_task_2)]
    model.learn(total_timesteps=task2_timesteps, callback=combined_callback)
    
    # ============================================================================
    # FINAL TESTING: Test on all tasks after Phase 2
    # ============================================================================
    print(f"📊 Phase 2 Testing: Evaluating on all tasks after training on {train_task_2}...")
    
    # Test on all available tasks
    phase2_results = {}
    
    for task in all_tasks:
        eval_env = make_env(task)()
        rewards, lengths, success, completion = evaluate_model_enhanced(
            model, eval_env, task, config['n_eval_episodes']
        )
        phase2_results[task] = {
            'rewards': rewards,
            'lengths': lengths,
            'success_rate': success,
            'completion_percentage': completion,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }
        eval_env.close()
        
        # Print immediate results for this task
        print(f"   • {task}: {phase2_results[task]['mean_reward']:.2f} ± {phase2_results[task]['std_reward']:.2f} "
              f"(Success: {phase2_results[task]['success_rate']:.1%}, Completion: {phase2_results[task]['completion_percentage']:.1f}%)")
    
    # Log Phase 2 results with topology-aware naming
    if wandb.run:
        for task, results in phase2_results.items():
        wandb.log({
                f'{topology_type}/{task_order}/phase2/testing/{task}/mean_reward': results['mean_reward'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/std_reward': results['std_reward'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/mean_length': results['mean_length'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/std_length': results['std_length'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/success_rate': results['success_rate'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/completion_percentage': results['completion_percentage'],
                f'{topology_type}/{task_order}/phase2/testing/{task}/n_eval_episodes': config['n_eval_episodes'],
                
                # Legacy metrics for backward compatibility
                f'phase2/{task}/testing/mean_reward': results['mean_reward'],
                f'phase2/{task}/testing/success_rate': results['success_rate'],
            })
    
    # ============================================================================
    # TRANSFER LEARNING ANALYSIS
    # ============================================================================
    print(f"🔄 Analyzing transfer learning patterns...")
    
    # Calculate transfer learning metrics
    transfer_metrics = {}
    
    # Forward transfer: How well does training on task 1 help with task 2?
    if train_task_1 in phase1_results and train_task_2 in phase2_results:
        task1_baseline = phase1_results[train_task_1]['mean_reward']
        task2_final = phase2_results[train_task_2]['mean_reward']
        task2_baseline = phase1_results[train_task_2]['mean_reward']
        
        forward_transfer = task2_final - task2_baseline if task2_baseline > 0 else task2_final
        transfer_metrics['forward_transfer_score'] = forward_transfer
    
    # Backward transfer: How well does training on task 2 affect task 1 retention?
    if train_task_1 in phase1_results and train_task_1 in phase2_results:
        task1_baseline = phase1_results[train_task_1]['mean_reward']
        task1_final = phase2_results[train_task_1]['mean_reward']
        
        retention_rate = task1_final / task1_baseline if task1_baseline > 0 else 0
        transfer_metrics['backward_transfer_score'] = retention_rate
        transfer_metrics['catastrophic_forgetting'] = 1 - retention_rate
    
    # Log transfer learning metrics with topology context
    if wandb.run and transfer_metrics:
        for metric_name, value in transfer_metrics.items():
            wandb.log({
                f'{topology_type}/{task_order}/transfer/{metric_name}': value,
                # Legacy metrics for backward compatibility
                f'transfer/{metric_name}': value,
            })
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"✅ Double-task training completed!")
    print(f"📊 Final Results:")
    print(f"   • {train_task_1}: {phase2_results[train_task_1]['mean_reward']:.2f} (Phase 1: {phase1_results[train_task_1]['mean_reward']:.2f})")
    print(f"   • {train_task_2}: {phase2_results[train_task_2]['mean_reward']:.2f} (Phase 1: {phase1_results[train_task_2]['mean_reward']:.2f})")
    
    if transfer_metrics:
        print(f"🔄 Transfer Learning:")
        if 'forward_transfer_score' in transfer_metrics:
            print(f"   • Forward Transfer: {transfer_metrics['forward_transfer_score']:.2f}")
        if 'backward_transfer_score' in transfer_metrics:
            print(f"   • Retention Rate: {transfer_metrics['backward_transfer_score']:.3f}")
    
    # ============================================================================
    # ADVANCED PLOTTING INTEGRATION
    # ============================================================================
    if wandb.run:
        print(f"📊 Generating advanced plots for {topology_type} - {task_order}...")
        
        # Combine all phase results for plotting
        all_phase_results = {}
        for task in [train_task_1, train_task_2, 'LunarLander-v2']:
            # Phase 1 results
            if task in phase1_results:
                all_phase_results[f'{topology_type}/{task_order}/phase1/testing/{task}/mean_reward'] = phase1_results[task]['mean_reward']
            # Phase 2 results
            if task in phase2_results:
                all_phase_results[f'{topology_type}/{task_order}/phase2/testing/{task}/mean_reward'] = phase2_results[task]['mean_reward']
        
        # Log comprehensive plots
        log_comprehensive_plots_for_run(
            wandb_run=wandb.run,
            phase_results=all_phase_results,
            transfer_metrics=transfer_metrics,
            topology_type=topology_type,
            task_sequence=task_order,
            sweep_results=None  # Will be populated when sweep results are available
        )
        
        print(f"✅ Advanced plots logged to wandb!")
    
    # Clean up
    env1.close()
    env2.close()
    
    return {
        'phase1_results': phase1_results,
        'phase2_results': phase2_results,
        'transfer_metrics': transfer_metrics,
        'task_order': task_order,
        'sequential_training': True
    }

# ============================================================================
# SWEEP TRAINING FUNCTION
# ============================================================================

def train_with_sweep():
    """Main function for sweep training."""
    
    # ============================================================================
    # PRE-CALCULATE CAPACITY MATCHING (BEFORE wandb.init)
    # ============================================================================
    
    effective_hidden_size, target_capacity, args = pre_calculate_capacity_matching()
    
    # Initialize wandb run if not already done
    if wandb.run is None:
        wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--double-task-training",
            config={
                # These will be overridden by sweep parameters
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_grad_norm': 0.5,
                'hidden_size': effective_hidden_size,  # Use pre-calculated size
                'num_layers': 2,
                'topology_type': 'fully_connected',
                'train_task_1': 'CartPole-v1',
                'train_task_2': 'Acrobot-v1',
                'total_timesteps': 600000,
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
                # Capacity matching parameters
                'target_capacity': None,  # Will be set by sweep if capacity matching is enabled
            }
        )
    
    print(f"🎯 Starting double-task sweep run with configuration:")
    try:
        print(f"   • Topology: {wandb.config.topology_type}")
        print(f"   • Hidden size: {wandb.config.hidden_size}")
        print(f"   • Layers: {wandb.config.num_layers}")
        print(f"   • Learning rate: {wandb.config.learning_rate}")
        print(f"   • Task 1: {wandb.config.train_task_1}")
        print(f"   • Task 2: {wandb.config.train_task_2}")
        print(f"   • Total timesteps: {wandb.config.total_timesteps}")
        
        # Check for capacity matching
        target_capacity = wandb.config.get('target_capacity', None)
        if target_capacity is not None:
            print(f"   • Target capacity: {target_capacity:,} parameters")
            print(f"   • Capacity matching: ENABLED")
        else:
            print(f"   • Capacity matching: DISABLED")
            
    except:
        print(f"   • Using default configuration (not in sweep mode)")
    
    # Get configuration from wandb
    config = create_debug_config()
    
    # Log capacity matching results if applicable
    if target_capacity is not None:
        wandb.log({
            'capacity_matching/target_capacity': target_capacity,
            'capacity_matching/calculated_size': effective_hidden_size,
            'capacity_matching/original_hidden_size': args.hidden_size,
            'capacity_matching/topology_type': args.topology_type,
            'capacity_matching/num_layers': args.num_layers,
        })
    
    # Extract parameters
    topology_type = wandb.config.get('topology_type', 'fully_connected')
    hidden_size = wandb.config.get('hidden_size', 64)
    num_layers = wandb.config.get('num_layers', 2)
    train_task_1 = wandb.config.get('train_task_1', 'CartPole-v1')
    train_task_2 = wandb.config.get('train_task_2', 'Acrobot-v1')
    
    # Run double-task training
    results = double_task_training(
        policy_class=DebugTopologyPolicy,
        topology_type=topology_type,
        config=config,
        num_layers=num_layers,
        hidden_size=hidden_size,
        train_task_1=train_task_1,
        train_task_2=train_task_2
    )
    
    print("✅ Double-task training completed!")
    print(f"   • Task 1 ({train_task_1}) success rate: {results['task1_success']:.3f}")
    print(f"   • Task 2 ({train_task_2}) success rate: {results['task2_success']:.3f}")
    print(f"   • Overall success rate: {(results['task1_success'] + results['task2_success']) / 2:.3f}")

def unified_training_function():
    """
    Unified training function for double-task training with reward scaling.
    This is the main entry point for wandb sweeps.
    """
    
    # Check if we're in a wandb sweep or running standalone
    if wandb.run is None:
        # Standalone execution - use default configuration
        print("🚀 Running double-task training in standalone mode...")
        
        # Default configuration for standalone execution
        config = {
            'topology_type': 'small_world',
                    'hidden_size': 128,
        'num_layers': 2,
        'task_order': 'LunarLander-v2_CartPole-v1',
        'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
            'total_timesteps': 600000,
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
        
        # Initialize wandb with default config
        wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--double-task-training",
            config=config
        )
    else:
        # Sweep execution - use wandb.config
        config = wandb.config
    
    # Determine topology type
    topology_type = config.get('topology_type', 'fully_connected')
    
    # Determine hidden size or capacity
    hidden_size = config.get('hidden_size', 64)
    num_layers = config.get('num_layers', 2)
    
    # Determine tasks from task_order parameter
    task_order = config.get('task_order', 'LunarLander-v2_CartPole-v1')
    tasks = task_order.split('_')
    train_task_1 = tasks[0]
    train_task_2 = tasks[1]
    
    # Initialize wandb with proper naming (if not already done)
    if wandb.run is not None:
        initialize_wandb_run(config, topology_type, 'double_task')
    
    # Create configuration
    debug_config = create_debug_config()
    
    # Run double-task training
    return double_task_training(
        DebugTopologyPolicy,
        topology_type,
        debug_config,
        num_layers=num_layers,
        hidden_size=hidden_size,
        train_task_1=train_task_1,
        train_task_2=train_task_2
    )

if __name__ == "__main__":
    # Run the unified training function for wandb sweeps
    unified_training_function()