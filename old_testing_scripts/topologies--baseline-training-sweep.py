#!/usr/bin/env python3
"""
Baseline Topology Training with Weights & Biases Sweep Support

This script is a sweep-enabled version of the baseline training script focused on:
- Single task training (no cross-task evaluation)
- Topology network verification and debugging
- Hyperparameter optimization for basic performance
- Simplified configuration for baseline experiments
- Reward scaling and task normalization for fair comparison
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
from src.utils.capacity_matching_helper import pre_calculate_capacity_matching
from src.utils.task_normalization import (
    compute_multi_task_metrics, log_normalized_metrics, print_normalized_summary,
    get_task_thresholds, get_normalization_constants, normalize_reward,
    calculate_reward_completion_percentage
)
from src.utils.advanced_plotting import (
    log_streamlined_plots_for_run, create_multi_phase_learning_curves
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
        
        self.current_mask = self.action_masks.get(task_name, [True, True, True, True])
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
        
        # Pad observation to universal dimensions (8)
        obs = self._pad_observation(obs)
        
        # Add action masking info to info dict
        info['universal_action'] = action
        info['mapped_action'] = mapped_action
        info['action_mask'] = self.current_mask
        
        return obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to universal 8-dimensional space."""
        obs = np.array(obs, dtype=np.float32)
        
        if len(obs.shape) == 1:
            # Single observation
            if obs.shape[0] < 8:
                # Pad with zeros
                padded_obs = np.zeros(8, dtype=np.float32)
                padded_obs[:obs.shape[0]] = obs
                return padded_obs
            elif obs.shape[0] > 8:
                # Truncate
                return obs[:8]
            else:
                return obs
        else:
            # Vectorized observation
            batch_size = obs.shape[0]
            if obs.shape[1] < 8:
                # Pad with zeros
                padded_obs = np.zeros((batch_size, 8), dtype=np.float32)
                padded_obs[:, :obs.shape[1]] = obs
                return padded_obs
            elif obs.shape[1] > 8:
                # Truncate
                return obs[:, :8]
            else:
                return obs
    
    def reset(self, **kwargs):
        """Reset the environment and pad observation."""
        result = self.env.reset(**kwargs)
        
        # Handle different reset return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Pad the observation
        padded_obs = self._pad_observation(obs)
        
        # Return in the same format as received
        if isinstance(result, tuple):
            return padded_obs, info
        else:
            return padded_obs
    
    def get_action_mask(self):
        """Get the action mask for the current task."""
        return self.current_mask

# ============================================================================
# BASELINE TOPOLOGY POLICY CLASS
# ============================================================================

class BaselineTopologyPolicy(ActorCriticPolicy):
    """
    Baseline Topology Policy for single-task training with sweep support.
    Simplified version focused on topology verification and basic performance.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, num_layers=2, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_baseline_config()
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        
        # Universal dimensions from config
        self.universal_input_dim = config['universal_input_dim']
        self.universal_output_dim = config['universal_output_dim']
        self.universal_action_dim = config['universal_action_dim']
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.topology_type = topology_type
        self.config = config
        
        print(f"\n🔍 BASELINE: Creating {topology_type} topology policy")
        print(f"   • Task dimensions: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"   • Universal dimensions: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"   • Hidden size: {self.hidden_size}")
        print(f"   • Number of layers: {self.num_layers}")
        
        # Create separate topology networks for actor and critic
        print(f"   • Creating actor topology network...")
        self.actor_topology = self._create_topology_network('actor')
        print(f"   • Creating critic topology network...")
        self.critic_topology = self._create_topology_network('critic')
        
        # Get weight statistics
        actor_params = self._get_topology_params(self.actor_topology)
        critic_params = self._get_topology_params(self.critic_topology)
        
        # Calculate total parameters safely
        if isinstance(actor_params, (int, float)) and isinstance(critic_params, (int, float)):
        total_params = actor_params + critic_params
        elif isinstance(actor_params, dict) and isinstance(critic_params, dict):
            actor_size = actor_params.get('size', 0)
            critic_size = critic_params.get('size', 0)
            total_params = actor_size + critic_size
        else:
            total_params = 0
            
        print(f"   • Actor topology parameters: {actor_params}")
        print(f"   • Critic topology parameters: {critic_params}")
        print(f"   • Total parameters: {total_params:,}")
    
    def _create_topology_network(self, network_type):
        """Create topology network for actor or critic."""
        print(f"   • Creating {network_type} topology network...")
        
        # Use the hidden_size that was passed to the policy
        total_size = self.hidden_size
        
        print(f"     🔧 Creating {network_type} topology ({self.topology_type})")
        print(f"       • Network type: ffn")
        print(f"       • Number of layers: {self.num_layers}")
        print(f"       • Using size: {total_size} nodes")
        
        # Create topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=self.num_layers,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,
                k=self.config.get('small_world_params', {}).get('k', 4),
                p=self.config.get('small_world_params', {}).get('p', 0.3),
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=total_size,
                num_modules=self.config.get('modular_params', {}).get('num_modules', 4),
                inter_module_prob=self.config.get('modular_params', {}).get('inter_module_prob', 0.2),
                intra_module_prob=self.config.get('modular_params', {}).get('intra_module_prob', 0.8),
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=total_size,
                num_modules=self.config.get('hybrid_params', {}).get('num_modules', 4),
                k=self.config.get('hybrid_params', {}).get('k', 4),
                p=self.config.get('hybrid_params', {}).get('p', 0.3),
                inter_module_prob=self.config.get('hybrid_params', {}).get('inter_module_prob', 0.2),
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate graph
        print(f"       • Generating graph...")
        graph = topology.generate()
        print(f"       • Graph generated with {len(graph.edges())} edges")
        
        # Define input/output nodes
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        print(f"       • Input nodes: {input_nodes}")
        print(f"       • Output nodes: {output_nodes}")
        
        # Create network
        network_params = self.config.get('network_params', {}).get('ffn', {
            'learning_rate': 0.001,
            'activation': 'tanh'
        })
        
        network = FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
        print(f"       • Network created successfully")
        
        # Count actual parameters
        actual_params = self._get_topology_params(network)
        print(f"       • Actual parameters: {actual_params}")
        
        return network
    
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
            else:
                # Fallback to PyTorch parameters
                for param in topology_network.parameters():
                    total_params += param.numel()
        except Exception as e:
            print(f"       ⚠️  Error counting parameters: {e}")
            total_params = 0
        return total_params
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create input mask for masking strategy."""
        batch_size = x.shape[0]
        mask = torch.zeros(batch_size, self.universal_input_dim, dtype=torch.bool, device=x.device)
        mask[:, :x.shape[1]] = True
        return mask
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply input masking."""
        masked_input = torch.zeros(x.shape[0], self.universal_input_dim, dtype=x.dtype, device=x.device)
        masked_input[mask] = x.flatten()
        return masked_input
    
    def forward_actor(self, obs):
        """Forward pass for actor - topology network directly outputs action logits."""
        features = self.extract_features(obs)
        
        # Create input mask and apply masking
        input_mask = self._create_input_mask(features)
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Convert to dictionary format for topology network
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through actor topology network
        topology_output = self.actor_topology.forward(input_dict)
        
        # Convert topology output to action logits
        output_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = topology_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        
        # Stack outputs and project to action space
        universal_output = torch.stack(output_values, dim=1)
        
        # Project to action space (universal_output_dim → action_space.n)
        action_logits = universal_output[:, :self.action_space.n]
        
        return action_logits
    
    def forward_critic(self, obs):
        """Forward pass for critic - topology network directly outputs value."""
        features = self.extract_features(obs)
        
        # Create input mask and apply masking
        input_mask = self._create_input_mask(features)
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Convert to dictionary format for topology network
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through critic topology network
        topology_output = self.critic_topology.forward(input_dict)
        
        # Convert topology output to value
        output_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = topology_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        
        # Stack outputs and take mean as value
        universal_output = torch.stack(output_values, dim=1)
        
        # Take mean of outputs as value (or use first output)
        value = universal_output.mean(dim=1, keepdim=True)
        
        return value

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class BaselineCallback(BaseCallback):
    """Baseline callback for tracking training progress."""
    
    def __init__(self, verbose=0, wandb_run=None, log_freq=1000):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_count = 0
        self.wandb_run = wandb_run
        self.log_freq = log_freq
        self.rollout_count = 0
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'learning_rates': []
        }
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log detailed metrics every log_freq steps
        if self.step_count % self.log_freq == 0 and self.wandb_run is not None:
            self._log_training_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout."""
        self.rollout_count += 1
        
        if self.wandb_run is not None:
            self._log_rollout_metrics()
        
        super()._on_rollout_end()
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if self.wandb_run is not None:
            self._log_final_training_summary()
        
        super()._on_training_end()
    
    def _log_training_metrics(self):
        """Log detailed training metrics."""
        try:
            # Get metrics from the model's logger
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = self.model.logger.name_to_value
                
                metrics = {
                    "train/step": self.step_count,
                    "train/total_timesteps": self.num_timesteps,
                    "train/rollout_count": self.rollout_count,
                }
                
                # Add specific PPO metrics if available
                for key, value in name_to_value.items():
                    if any(term in key.lower() for term in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip', 'explained']):
                        metrics[f"train/{key}"] = value
                
                # Add learning rate if available
                if hasattr(self.model, 'lr_schedule'):
                    current_lr = self.model.lr_schedule(self.num_timesteps)
                    metrics["train/learning_rate"] = current_lr
                
                # NEW: Add learning progression metrics
                if len(self.training_metrics['episode_rewards']) > 0:
                    recent_rewards = self.training_metrics['episode_rewards'][-100:]  # Last 100 episodes
                    recent_lengths = self.training_metrics['episode_lengths'][-100:]  # Last 100 episodes
                    
                    metrics.update({
                        "learning_progression/episode_reward_mean": np.mean(recent_rewards),
                        "learning_progression/episode_reward_std": np.std(recent_rewards),
                        "learning_progression/episode_length_mean": np.mean(recent_lengths),
                        "learning_progression/episode_length_std": np.std(recent_lengths),
                        "learning_progression/training_progress_ratio": self.num_timesteps / self.model.total_timesteps if hasattr(self.model, 'total_timesteps') else 0.0
                    })
                    
                    # Calculate current success rate and completion percentage if we have task info
                    if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs') and len(self.model.env.envs) > 0:
                        env = self.model.env.envs[0]
                        if hasattr(env, 'spec') and env.spec is not None:
                            task_name = env.spec.id
                            success_rate = calculate_success_rate(recent_rewards, recent_lengths, task_name)
                            completion_pct = calculate_reward_completion_percentage(recent_rewards, task_name)
                            metrics.update({
                                "learning_progression/success_rate_current": success_rate,
                                "learning_progression/completion_percentage_current": completion_pct
                            })
                
                # Add network-specific metrics if available
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                    actor_params = self.model.policy._get_topology_params(self.model.policy.actor_topology)
                    critic_params = self.model.policy._get_topology_params(self.model.policy.critic_topology)
                    metrics.update({
                        "network/actor_parameters": actor_params,
                        "network/critic_parameters": critic_params,
                    })
                    
                    # Calculate total parameters safely
                    if isinstance(actor_params, (int, float)) and isinstance(critic_params, (int, float)):
                        total_params = actor_params + critic_params
                        metrics["network/total_parameters"] = total_params
                    elif isinstance(actor_params, dict) and isinstance(critic_params, dict):
                        actor_size = actor_params.get('size', 0)
                        critic_size = critic_params.get('size', 0)
                        total_params = actor_size + critic_size
                        metrics["network/total_parameters"] = total_params
                    
                    # REMOVED: Graph metrics logging (too expensive during training)
                    # REMOVED: Depth analysis (too expensive during training)
                    # REMOVED: Sample efficiency (redundant metrics)
                    # REMOVED: Hyperparameter correlation (redundant metrics)
                
                self.wandb_run.log(metrics, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging training metrics: {e}")
    
    def _log_rollout_metrics(self):
        """Log metrics at the end of each rollout."""
        try:
            # Get rollout statistics
            if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
                buffer = self.model.rollout_buffer
                
                # Calculate episode statistics
                if hasattr(buffer, 'rewards') and len(buffer.rewards) > 0:
                    episode_rewards = buffer.rewards.flatten()
                    episode_lengths = np.ones_like(episode_rewards) * self.model.n_steps
                    
                    metrics = {
                        "rollout/mean_reward": np.mean(episode_rewards),
                        "rollout/std_reward": np.std(episode_rewards),
                        "rollout/max_reward": np.max(episode_rewards),
                        "rollout/min_reward": np.min(episode_rewards),
                        "rollout/mean_length": np.mean(episode_lengths),
                        "rollout/episode_count": len(episode_rewards),
                    }
                    
                    self.wandb_run.log(metrics, step=self.num_timesteps)
                    
                    # Store for final summary
                    self.training_metrics['episode_rewards'].extend(episode_rewards.tolist())
                    self.training_metrics['episode_lengths'].extend(episode_lengths.tolist())
        except Exception as e:
            print(f"   ⚠️  Error logging rollout metrics: {e}")
    
    def _log_graph_metrics(self):
        """Log real-time graph metrics during training."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                actor_G = self.model.policy.actor_topology.topology
                critic_G = self.model.policy.critic_topology.topology
                
                # Calculate graph metrics
                actor_metrics = self._calculate_graph_metrics(actor_G, 'actor')
                critic_metrics = self._calculate_graph_metrics(critic_G, 'critic')
                
                # Log with timestep correlation
                metrics = {
                    **actor_metrics,
                    **critic_metrics,
                    'graph/timestep': self.num_timesteps,
                    'graph/topology_type': self.model.policy.topology_type
                }
                
                self.wandb_run.log(metrics, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging graph metrics: {e}")

    def _calculate_graph_metrics(self, G, network_type):
        """Calculate focused graph metrics (removed redundant ones)."""
        try:
            G_undirected = G.to_undirected() if G.is_directed() else G
            
            # Keep only high-value metrics
            metrics = {
                f'graph/{network_type}/density': nx.density(G),
            }
            
            # Connectivity-dependent metrics
            if nx.is_connected(G_undirected):
                metrics.update({
                    f'graph/{network_type}/diameter': nx.diameter(G_undirected),
                    f'graph/{network_type}/avg_path_length': nx.average_shortest_path_length(G_undirected),
                    f'graph/{network_type}/clustering_coefficient': nx.average_clustering(G_undirected),
                })
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                largest_cc_graph = G_undirected.subgraph(largest_cc)
                metrics.update({
                    f'graph/{network_type}/diameter': nx.diameter(largest_cc_graph),
                    f'graph/{network_type}/avg_path_length': nx.average_shortest_path_length(largest_cc_graph),
                    f'graph/{network_type}/clustering_coefficient': nx.average_clustering(G_undirected),
                })
            
            return metrics
        except Exception as e:
            print(f"   ⚠️  Error calculating graph metrics for {network_type}: {e}")
            return {}

    def _log_depth_analysis(self):
        """Log depth efficiency analysis correlating graph structure with performance."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                actor_G = self.model.policy.actor_topology.topology
                critic_G = self.model.policy.critic_topology.topology
                
                # Calculate depth metrics
                actor_depth = self._calculate_depth_metrics(actor_G, 'actor')
                critic_depth = self._calculate_depth_metrics(critic_G, 'critic')
                
                # Performance correlation
                current_reward = np.mean(self.training_metrics['episode_rewards'][-100:]) if self.training_metrics['episode_rewards'] else 0
                
                # Calculate efficiency ratios
                actor_avg_path = actor_depth.get('depth/actor/avg_path_length', 1)
                actor_density = actor_depth.get('depth/actor/density', 1)
                
                depth_analysis = {
                    'depth/current_reward': current_reward,
                    'depth/depth_efficiency': current_reward / (actor_avg_path + 1e-6),
                    'depth/density_efficiency': current_reward / (actor_density + 1e-6),
                }
                
                self.wandb_run.log(depth_analysis, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging depth analysis: {e}")

    def _calculate_depth_metrics(self, G, network_type):
        """Calculate depth efficiency metrics only (removed redundant ones)."""
        try:
            G_undirected = G.to_undirected() if G.is_directed() else G
            
            if nx.is_connected(G_undirected):
                avg_path_length = nx.average_shortest_path_length(G_undirected)
                density = nx.density(G)
                
                return {
                    f'depth/{network_type}/avg_path_length': avg_path_length,
                    f'depth/{network_type}/density': density,
                }
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                largest_cc_graph = G_undirected.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(largest_cc_graph)
                density = nx.density(G)
                
                return {
                    f'depth/{network_type}/avg_path_length': avg_path_length,
                    f'depth/{network_type}/density': density,
                }
        except Exception as e:
            print(f"   ⚠️  Error calculating depth metrics for {network_type}: {e}")
            return {}

    def _log_sample_efficiency(self):
        """Log focused sample efficiency metrics (removed redundant ones)."""
        try:
            if len(self.training_metrics['episode_rewards']) > 0:
                # Calculate sample efficiency metrics
                total_timesteps = self.num_timesteps
                total_episodes = len(self.training_metrics['episode_rewards'])
                
                # Recent performance (last 100 episodes)
                recent_rewards = self.training_metrics['episode_rewards'][-100:] if len(self.training_metrics['episode_rewards']) >= 100 else self.training_metrics['episode_rewards']
                recent_mean_reward = np.mean(recent_rewards)
                
                # Keep only high-value sample efficiency metrics
                efficiency_metrics = {
                    'efficiency/reward_per_timestep': recent_mean_reward / (total_timesteps + 1e-6),
                    'efficiency/reward_per_episode': recent_mean_reward,
                    'efficiency/timesteps_per_episode': total_timesteps / (total_episodes + 1e-6),
                }
                
                self.wandb_run.log(efficiency_metrics, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging sample efficiency: {e}")

    def _log_hyperparameter_correlation(self):
        """Log focused hyperparameter correlation metrics (removed redundant ones)."""
        try:
            if hasattr(self.model, 'policy'):
                # Get current hyperparameters
                config = wandb.config if wandb.run else {}
                
                # Get graph metrics
                if hasattr(self.model.policy, 'actor_topology'):
                    actor_G = self.model.policy.actor_topology.topology
                    G_undirected = actor_G.to_undirected() if actor_G.is_directed() else actor_G
                    
                    if nx.is_connected(G_undirected):
                        avg_path_length = nx.average_shortest_path_length(G_undirected)
                        diameter = nx.diameter(G_undirected)
                        
                        # Keep only high-value hyperparameter correlation metrics
                        correlation_metrics = {
                            'correlation/lr_path_length_ratio': config.get('learning_rate', 0) / (avg_path_length + 1e-6),
                            'correlation/lr_diameter_ratio': config.get('learning_rate', 0) / (diameter + 1e-6),
                        }
                        
                        self.wandb_run.log(correlation_metrics, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging hyperparameter correlation: {e}")
    
    def _log_final_training_summary(self):
        """Log final training summary with comprehensive metrics."""
        if self.wandb_run is None:
            return
        
        # Log final training summary
        self.wandb_run.log({
            'training/final_summary/total_timesteps': self.num_timesteps,
            'training/final_summary/total_episodes': len(self.episode_rewards),
            'training/final_summary/mean_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'training/final_summary/std_reward': np.std(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.std(self.episode_rewards),
            'training/final_summary/max_reward': np.max(self.episode_rewards),
            'training/final_summary/min_reward': np.min(self.episode_rewards),
        })

# ============================================================================
# IMPROVED LOGGING SYSTEM WITH REWARD SCALING
# ============================================================================

def initialize_wandb_run(config, topology_type, training_type='baseline'):
    """Initialize wandb with proper naming and configuration."""
    
    # Create descriptive run name
    run_name = create_run_name(config, topology_type, training_type)
    
    # Create tags for easy filtering
    tags = create_run_tags(config, topology_type, training_type)
    
    # Initialize wandb
    wandb.init(
        project="topologies--baseline-training",
        entity="katko-it-universitetet-i-k-benhavn",
        config=config,
        name=run_name,
        tags=tags
    )

def create_run_name(config, topology_type, training_type, model=None):
    """Create descriptive run name with exact capacity and size."""
    
    # Topology abbreviation
    topology_abbrev = {
        'small_world': 'SW',
        'modular': 'MOD', 
        'hybrid': 'HYB',
        'fully_connected': 'FC'
    }.get(topology_type, topology_type.upper())
    
    # Get actual capacity and size
    actual_capacity = None
    actual_size = config.get('hidden_size', 'unknown')
    
    # Handle capacity calculation based on sweep type
    if 'target_capacity' in config:
        # For capacity-matched sweeps, use target capacity directly
        actual_capacity = config.get('target_capacity')
        
        # For capacity-matched sweeps, we need to show the adjusted hidden size
        # This will be calculated during model creation, but for naming we can estimate
        # or use a placeholder that will be updated later
        if model is not None and hasattr(model, 'policy'):
            try:
                # Try to get the actual adjusted size from the model
                policy = model.policy
                if hasattr(policy, 'actor_topology') and hasattr(policy.actor_topology, 'hidden_size'):
                    actual_size = policy.actor_topology.hidden_size
                elif hasattr(policy, 'critic_topology') and hasattr(policy.critic_topology, 'hidden_size'):
                    actual_size = policy.critic_topology.hidden_size
            except Exception as e:
                # If we can't get the adjusted size, use a placeholder
                actual_size = f"adj_{config.get('hidden_size', 'unknown')}"
        else:
            # For initial naming, use a placeholder that indicates it will be adjusted
            actual_size = f"adj_{config.get('hidden_size', 'unknown')}"
            
    elif model is not None and hasattr(model, 'policy'):
        # Calculate actual capacity from the model
        try:
            policy = model.policy
            actor_params = policy._get_topology_params(policy.actor_topology)
            critic_params = policy._get_topology_params(policy.critic_topology)
            
            # Calculate total parameters safely
            if isinstance(actor_params, (int, float)) and isinstance(critic_params, (int, float)):
                actual_capacity = actor_params + critic_params
            elif isinstance(actor_params, dict) and isinstance(critic_params, dict):
                actor_size = actor_params.get('size', 0)
                critic_size = critic_params.get('size', 0)
                actual_capacity = actor_size + critic_size
            else:
                actual_capacity = None
        except Exception as e:
            print(f"   ⚠️  Could not calculate actual capacity: {e}")
            actual_capacity = None
    elif 'hidden_size' in config:
        # For size-matched sweeps without model, we can't calculate actual capacity yet
        actual_capacity = None
    
    # Task abbreviations
    task_abbrev = {
        'LunarLander-v2': 'LL',
        'Acrobot-v1': 'AC', 
        'CartPole-v1': 'CP',
        'MountainCar-v0': 'MC'
    }
    
    # Build name parts
    name_parts = [topology_abbrev]
    
    # Add capacity (exact number)
    if actual_capacity is not None:
        name_parts.append(f"C{actual_capacity}")
    else:
        name_parts.append("C?")
    
    # Add size
    name_parts.append(f"S{actual_size}")
    
    # Add task information
    if training_type == 'baseline' or training_type == 'single_task':
        task_name = config.get('train_task', 'unknown')
        task_abbrev_name = task_abbrev.get(task_name, task_name)
        name_parts.append(task_abbrev_name)
    
    return "_".join(name_parts)

def create_run_tags(config, topology_type, training_type):
    """Create enhanced tags for easy filtering and organization."""
    
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
    elif 'hidden_size' in config:
        tags.extend([
            "fixed_size", 
            f"size_{config.get('hidden_size')}",
            "size_matched"
        ])
    
    # Task tags
    if training_type == 'baseline' or training_type == 'single_task':
        tags.append(config.get('train_task', 'unknown'))
    
    return tags

def log_baseline_results(wandb_run, baseline_results, topology_type, task_order=None):
    """Log baseline evaluation results with topology-aware naming."""
    
    for task, results in baseline_results.items():
        # Topology-aware metric names for easy comparison
        base_path = f"{topology_type}/{task_order}/baseline" if task_order else f"{topology_type}/baseline"
        
        # Raw metrics with topology context
        wandb_run.log({
            f'{base_path}/{task}/raw/mean_reward': results['mean_reward'],
            f'{base_path}/{task}/raw/success_rate': results['success_rate'],
            f'{base_path}/{task}/raw/mean_length': np.mean(results['lengths']),
            f'{base_path}/{task}/raw/std_reward': np.std(results['rewards']),
            f'{base_path}/{task}/raw/std_length': np.std(results['lengths'])
        })
        
        # Normalized metrics with topology context
        normalized_reward = normalize_reward(results['mean_reward'], task)
        wandb_run.log({
            f'{base_path}/{task}/normalized/reward': normalized_reward,
            f'{base_path}/{task}/normalized/efficiency': results.get('efficiency_score', 1.0)
        })
        
        # Legacy metrics for backward compatibility
        wandb_run.log({
            f'baseline/{task}/raw/mean_reward': results['mean_reward'],
            f'baseline/{task}/raw/success_rate': results['success_rate'],
            f'baseline/{task}/raw/mean_length': np.mean(results['lengths']),
            f'baseline/{task}/raw/std_reward': np.std(results['rewards']),
            f'baseline/{task}/raw/std_length': np.std(results['lengths']),
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
        normalized_reward = normalize_reward(results['mean_reward'], task)
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

def log_final_analysis(wandb_run, final_analysis, topology_type, task_order=None):
    """Log final comprehensive analysis with topology context."""
    
    base_path = f"{topology_type}/{task_order}/final" if task_order else f"{topology_type}/final"
    
    # Raw performance metrics with topology context
    wandb_run.log({
        f'{base_path}/raw/mean_reward': final_analysis['final_mean_reward'],
        f'{base_path}/raw/success_rate': final_analysis['final_success_rate'],
        f'{base_path}/raw/mean_length': final_analysis['final_mean_length']
    })
    
    # Normalized performance metrics with topology context
    wandb_run.log({
        f'{base_path}/normalized/final_normalized_score': final_analysis['final_normalized_score'],
        f'{base_path}/normalized/efficiency_score': final_analysis['efficiency_score'],
        f'{base_path}/normalized/parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0)
    })
    
    # Topology-specific normalized metrics
    wandb_run.log({
        f'{base_path}/topology/normalized_parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'{base_path}/topology/normalized_learning_stability': final_analysis.get('learning_stability', 0.0)
    })
    
    # Legacy metrics for backward compatibility
    wandb_run.log({
        'final/raw/mean_reward': final_analysis['final_mean_reward'],
        'final/raw/success_rate': final_analysis['final_success_rate'],
        'final/raw/mean_length': final_analysis['final_mean_length'],
        'final/normalized/final_normalized_score': final_analysis['final_normalized_score'],
        'final/normalized/efficiency_score': final_analysis['efficiency_score'],
        'final/normalized/parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'final/topology/{topology_type}/normalized_parameter_efficiency': final_analysis.get('parameter_efficiency', 0.0),
        f'final/topology/{topology_type}/normalized_learning_stability': final_analysis.get('learning_stability', 0.0)
    })

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def create_baseline_config():
    """Create configuration for baseline training with sweep support."""
    config = {
        # ============================================================================
        # EXPERIMENT PARAMETERS
        # ============================================================================
        'total_timesteps': 50000,  # Shorter training for baseline
        'n_eval_episodes': 10,     # Fewer evaluation episodes
        
        # ============================================================================
        # NETWORK DIMENSIONS
        # ============================================================================
        'universal_input_dim': 6,
        'universal_output_dim': 3,
        'universal_action_dim': 3,
        'hidden_size': 64,  # Default size
        'network_types': ['ffn'],
        
        # ============================================================================
        # TOPOLOGY-SPECIFIC PARAMETERS
        # ============================================================================
        'topology_params': {
            'small_world': {
                'k': 4,
                'p': 0.3
            },
            'modular': {
                'num_modules': 4,
                'inter_module_prob': 0.2,
                'intra_module_prob': 0.8
            },
            'hybrid': {
                'num_modules': 4,
                'k': 4,
                'p': 0.3,
                'inter_module_prob': 0.2
            },
            'fully_connected': {}
        },
        
        # ============================================================================
        # NETWORK PARAMETERS
        # ============================================================================
        'network_params': {
            'ffn': {
                'activation': 'relu',
                'dropout': 0.0
            }
        },
        
        # ============================================================================
        # PPO TRAINING PARAMETERS (defaults, will be overridden by sweep)
        # ============================================================================
        'ppo_params': {
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 32,
            'n_epochs': 5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5
        }
    }
    
    # Add backward compatibility aliases
    config['small_world_params'] = config['topology_params']['small_world']
    config['modular_params'] = config['topology_params']['modular']
    config['hybrid_params'] = config['topology_params']['hybrid']
    config['fully_connected_params'] = config['topology_params']['fully_connected']
    
    return config

def create_debug_config():
    """Alias for create_baseline_config for compatibility."""
    return create_baseline_config()

def evaluate_model_enhanced(model, env, task_name, n_eval_episodes=3):
    """Enhanced evaluation with task-specific metrics."""
    episode_rewards, episode_lengths = evaluate_model(model, env, n_eval_episodes)
    
    # Calculate task-specific success rate
    success_rate = calculate_success_rate(episode_rewards, episode_lengths, task_name)
    
    # Log evaluation metrics
    if wandb.run:
        wandb.log({
            f'evaluation/{task_name}/mean_reward': np.mean(episode_rewards),
            f'evaluation/{task_name}/std_reward': np.std(episode_rewards),
            f'evaluation/{task_name}/mean_length': np.mean(episode_lengths),
            f'evaluation/{task_name}/success_rate': success_rate,
            f'evaluation/{task_name}/episode_rewards': episode_rewards,
            f'evaluation/{task_name}/episode_lengths': episode_lengths
        })
    
    return episode_rewards, episode_lengths, success_rate

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

def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate a model on an environment."""
    rewards = []
    
    # Use tqdm for progress bar
    for episode in tqdm(range(n_eval_episodes), desc="Evaluating", leave=False):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle (obs, info) format
        
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # Handle both old and new gym step return formats
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                truncated = False
            else:
                obs, reward, done, truncated, info = step_result
            
            episode_reward += reward[0] if hasattr(reward, '__len__') else reward
            step_count += 1
            
            # Safety check to prevent infinite loops
            if step_count > 500:
                print(f"      ⚠️  Episode {episode} exceeded 500 steps, terminating")
                break
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    return mean_reward, std_reward

def baseline_training(policy_class, topology_type, config, num_layers=1, hidden_size=None, task=None):
    """
    Baseline training function with improved logging and reward scaling.
    
    Args:
        policy_class: Policy class to use
        topology_type: Type of topology network
        config: Configuration dictionary
        num_layers: Number of layers (for fully connected)
        hidden_size: Hidden size (if not using capacity matching)
        task: Task to train on
    """
    
    # Initialize wandb with proper naming
    initialize_wandb_run(config, topology_type, 'baseline')
    
    # Determine task from config if not provided
    if task is None:
        task = config.get('train_task', 'CartPole-v1')
    
    print(f"🚀 Starting baseline training on {task} with {topology_type} topology")
    
    # Pre-calculate capacity matching if target_capacity is specified
    effective_hidden_size = hidden_size
    if 'target_capacity' in config:
        effective_hidden_size = pre_calculate_capacity_matching(
            topology_type, config.get('target_capacity'), config
        )
        print(f"📊 Capacity matching: target={config.get('target_capacity')}, effective_size={effective_hidden_size}")
        
        # Update config with effective hidden size for correct naming
        config['hidden_size'] = effective_hidden_size
    
    # Create environment
    env = DummyVecEnv([make_env(task)])
    
    # Create policy with topology
    policy_kwargs = {
        'topology_type': topology_type,
        'hidden_size': effective_hidden_size,
        'num_layers': num_layers,
        'config': config
    }
    
    # Create model
    model = PPO(
        policy_class,
        env,
        learning_rate=config.get('learning_rate', 3e-4),
        batch_size=config.get('batch_size', 64),
        n_steps=config.get('n_steps', 2048),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        ent_coef=config.get('ent_coef', 0.01),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Calculate actual capacity and update run name if needed
    if wandb.run is not None and 'target_capacity' not in config:
        try:
            # Calculate actual capacity from the model
    policy = model.policy
    actor_params = policy._get_topology_params(policy.actor_topology)
    critic_params = policy._get_topology_params(policy.critic_topology)
            
            # Calculate total parameters safely
            if isinstance(actor_params, (int, float)) and isinstance(critic_params, (int, float)):
    total_params = actor_params + critic_params
            elif isinstance(actor_params, dict) and isinstance(critic_params, dict):
                actor_size = actor_params.get('size', 0)
                critic_size = critic_params.get('size', 0)
                total_params = actor_size + critic_size
            else:
                total_params = 0
            
            # Update run name with actual capacity
            updated_run_name = create_run_name(config, topology_type, 'baseline', model)
            if updated_run_name != wandb.run.name:
                print(f"📝 Updating run name with actual capacity: {wandb.run.name} → {updated_run_name}")
                wandb.run.name = updated_run_name
            
            # Log the actual capacity
            wandb.log({
                'network/actual_capacity': total_params,
                'network/actor_params': actor_params,
                'network/critic_params': critic_params
            })
            
            print(f"📊 Actual network capacity: {total_params:,} parameters")
            
    except Exception as e:
            print(f"   ⚠️  Could not calculate actual capacity: {e}")
    
    # Create callback for logging
    callback = BaselineCallback(wandb_run=wandb.run, log_freq=1000)
    
    # Get task-specific training configuration
    task_timesteps = get_task_timesteps(task, config)
    convergence_callback = create_convergence_callback(task, config)
    
    print(f"📋 Task-specific training: {task} for {task_timesteps:,} timesteps")
    
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
    
    # Train the model with convergence monitoring
    combined_callback = [callback, convergence_callback, ConvergenceEvaluationCallback(convergence_callback, model, env, task)]
    model.learn(total_timesteps=task_timesteps, callback=combined_callback)
    
    # Evaluate the model
    n_eval_episodes = config.get('n_eval_episodes', 15)
    eval_results = evaluate_model(model, env, n_eval_episodes)
    
    # Calculate normalized metrics
    normalized_reward = normalize_reward(eval_results['mean_reward'], task)
    
    # Create task order string for topology-aware logging (baseline = just the training task)
    task_order = task
    
    # Log final results with topology-aware naming
    wandb.log({
        # Topology-aware metrics for easy comparison
        f'{topology_type}/{task_order}/baseline/raw/mean_reward': eval_results['mean_reward'],
        f'{topology_type}/{task_order}/baseline/raw/success_rate': eval_results['success_rate'],
        f'{topology_type}/{task_order}/baseline/raw/mean_length': np.mean(eval_results['lengths']),
        f'{topology_type}/{task_order}/baseline/raw/std_reward': np.std(eval_results['rewards']),
        f'{topology_type}/{task_order}/baseline/raw/std_length': np.std(eval_results['lengths']),
        f'{topology_type}/{task_order}/baseline/normalized/reward': normalized_reward,
        f'{topology_type}/{task_order}/baseline/normalized/efficiency': eval_results.get('efficiency_score', 1.0),
        
        # Legacy metrics for backward compatibility
        'baseline/raw/mean_reward': eval_results['mean_reward'],
        'baseline/raw/success_rate': eval_results['success_rate'],
        'baseline/raw/mean_length': np.mean(eval_results['lengths']),
        'baseline/raw/std_reward': np.std(eval_results['rewards']),
        'baseline/raw/std_length': np.std(eval_results['lengths']),
        'baseline/normalized/reward': normalized_reward,
        'baseline/normalized/efficiency': eval_results.get('efficiency_score', 1.0)
    })
    
    # Final analysis
    final_analysis = {
        'final_mean_reward': eval_results['mean_reward'],
        'final_success_rate': eval_results['success_rate'],
        'final_mean_length': np.mean(eval_results['lengths']),
        'final_normalized_score': normalized_reward,
        'efficiency_score': eval_results.get('efficiency_score', 1.0),
        'parameter_efficiency': eval_results.get('parameter_efficiency', 0.0),
        'learning_stability': eval_results.get('learning_stability', 0.0)
    }
    
    # ============================================================================
    # ADVANCED PLOTTING INTEGRATION
    # ============================================================================
    if wandb.run:
        print(f"📊 Generating advanced plots for {topology_type} - {task_order}...")
        
        # Combine all results for plotting (baseline has only one phase)
        all_phase_results = {}
        all_phase_results[f'{topology_type}/{task_order}/phase1/testing/{task}/mean_reward'] = eval_results['mean_reward']
        
        # Log comprehensive plots
        log_streamlined_plots_for_run(
            wandb_run=wandb.run,
            phase_results=all_phase_results,
            transfer_metrics={},  # No transfer metrics for baseline
            topology_type=topology_type,
            task_sequence=task_order,
            sweep_results=None  # Will be populated when sweep results are available
        )
        
        print(f"✅ Advanced plots logged to wandb!")
    
    log_final_analysis(wandb.run, final_analysis, topology_type, task_order)
    
    print(f"✅ Baseline training completed on {task}")
    print(f"📊 Final normalized score: {normalized_reward:.4f}")
    
    return final_analysis

def unified_training_function():
    """
    Unified training function for baseline training with reward scaling.
    This is the main entry point for wandb sweeps.
    """
    
    # Check if we're in a wandb sweep or running standalone
    if wandb.run is None:
        # Standalone execution - use default configuration
        print("🚀 Running baseline training in standalone mode...")
        
        # Default configuration for standalone execution
        config = {
            'topology_type': 'small_world',
            'hidden_size': 128,
            'num_layers': 1,
            'train_task': 'CartPole-v1',
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
            # Universal dimensions
            'universal_input_dim': 6,
            'universal_output_dim': 3,
            'universal_action_dim': 3,
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
        
        # Determine topology type for naming
        topology_type = config.get('topology_type', 'fully_connected')
        
        # Initialize wandb with proper naming
        initialize_wandb_run(config, topology_type, 'baseline')
        else:
        # Sweep execution - use wandb.config
        config = wandb.config
    
    # Determine topology type
    topology_type = config.get('topology_type', 'fully_connected')
    
    # Determine hidden size or capacity
    hidden_size = config.get('hidden_size', 64)
    num_layers = config.get('num_layers', 1)
    
    # Determine task
    task = config.get('train_task', 'CartPole-v1')
    
    # Run baseline training
    return baseline_training(
            BaselineTopologyPolicy,
            topology_type,
            config,
        num_layers=num_layers,
        hidden_size=hidden_size,
        task=task
    )

if __name__ == "__main__":
    # Run the unified training function for wandb sweeps
    unified_training_function()