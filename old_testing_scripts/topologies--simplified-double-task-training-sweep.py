#!/usr/bin/env python3
"""
Simplified Double-Task Training Sweep for Topology Networks

This script trains topology networks sequentially on two tasks (CartPole and Acrobot only).
Sequential training: train on task A, then task B, then evaluate on both tasks.

Simplified version: Only uses CartPole-v1 and Acrobot-v1 (MountainCar-v0 removed).
"""

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
import wandb
import networkx as nx
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from tqdm import tqdm
import time # Added for timing training runs

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.fully_connected import FullyConnectedTopology
from src.utils.capacity_matching_helper import pre_calculate_capacity_matching
from src.utils.task_normalization import (
    compute_multi_task_metrics, log_normalized_metrics, print_normalized_summary,
    get_task_thresholds, get_normalization_constants
)

# ============================================================================
# UNIVERSAL ACTION WRAPPER (Simplified for CartPole + Acrobot only)
# ============================================================================

class UniversalActionWrapper(gym.Wrapper):
    """Universal action wrapper for CartPole and Acrobot environments."""
    
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        
        # Simplified task set: only CartPole and Acrobot
        if task_name not in ['CartPole-v1', 'Acrobot-v1']:
            raise ValueError(f"Unsupported task: {task_name}. Only CartPole-v1 and Acrobot-v1 are supported in simplified mode.")
        
        # Standardize observation space to 6 dimensions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Standardize action space to 3 actions
        self.action_space = gym.spaces.Discrete(3)
        
        # Action mappings for invalid actions (fallback to valid action)
        self.action_mappings = {
            'CartPole-v1': {2: 0},      # Map action 2 to action 0
            'Acrobot-v1': {}            # No mapping needed (all valid)
        }
        
        self.current_mapping = self.action_mappings.get(task_name, {})
    
    def step(self, action):
        """Execute action and return standardized observation."""
        # Convert action to int if it's a numpy array
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        # Map universal action to task-specific action
        if action in self.current_mapping:
            mapped_action = self.current_mapping[action]
        else:
            mapped_action = action
        
        # Step the environment with mapped action
        obs, reward, done, truncated, info = self.env.step(mapped_action)
        
        # Pad observation to 6 dimensions if needed
        obs = self._pad_observation(obs)
        
        # Add action mapping info to info dict
        info['universal_action'] = action
        info['mapped_action'] = mapped_action
        
        return obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to 6 dimensions."""
        if len(obs) < 6:
            # Pad with zeros
            obs = np.concatenate([obs, np.zeros(6 - len(obs))])
        elif len(obs) > 6:
            # Truncate to 6 dimensions
            obs = obs[:6]
        return obs
    
    def reset(self, **kwargs):
        """Reset environment and return standardized observation."""
        obs, info = self.env.reset(**kwargs)
        obs = self._pad_observation(obs)
        return obs, info
    
    def get_action_mask(self):
        """Get action mask for the current task."""
        if self.task_name == 'CartPole-v1':
            # CartPole: actions 0 and 1 are valid
            return [True, True, False]
        elif self.task_name == 'Acrobot-v1':
            # Acrobot: actions 0, 1, and 2 are valid
            return [True, True, True]
        else:
            # Default: all actions valid
            return [True, True, True]

# ============================================================================
# DEBUG TOPOLOGY POLICY (Simplified)
# ============================================================================

class DebugTopologyPolicy(ActorCriticPolicy):
    """Debug topology policy with simplified task support."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, num_layers=2, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.topology_type = topology_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.config = config or {}
        self.current_task = None  # Track current task for action masking
        
        # Create topology networks
        self.actor_topology = self._create_topology_network('actor')
        self.critic_topology = self._create_topology_network('critic')
        
        # Debug network structure
        self._debug_network_structure()
    
    def set_current_task(self, task_name):
        """Set the current task for action masking."""
        self.current_task = task_name
    
    def _create_topology_network(self, network_type):
        """Create topology network based on type."""
        # Calculate total size for topology (input + hidden + output)
        input_size = self.observation_space.shape[0]
        output_size = self.action_space.n if network_type == 'actor' else 1
        total_size = input_size + self.hidden_size + output_size
        
        # Define input/output nodes
        input_nodes = list(range(input_size))
        output_nodes = list(range(input_size + self.hidden_size, total_size))
        
        # Create topology
        if self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,
                k=self.config.get('small_world_k', 4),
                p=self.config.get('small_world_p', 0.1),
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=total_size,
                num_modules=self.config.get('modular_num_modules', 4),
                inter_module_prob=self.config.get('modular_inter_module_prob', 0.05),
                intra_module_prob=self.config.get('modular_intra_module_prob', 0.7),
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=total_size,
                num_modules=self.config.get('hybrid_num_modules', 4),
                k=self.config.get('hybrid_k', 4),
                p=self.config.get('hybrid_p', 0.1),
                inter_module_prob=self.config.get('hybrid_inter_module_prob', 0.05),
                seed=42
            )
        elif self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=self.num_layers,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate the network structure
        network_structure = topology.generate()
        
        # Create network parameters
        network_params = {
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'activation': self.config.get('activation', 'relu')
        }
        
        # Create FeedForwardNetwork
        from src.networks.ffn import FeedForwardNetwork
        return FeedForwardNetwork(
            topology=network_structure,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            network_params=network_params
        )
    
    def _get_topology_params(self, topology_network):
        """Get number of parameters in topology network."""
        if hasattr(topology_network, 'get_parameter_count'):
            return topology_network.get_parameter_count()
        else:
            # Fallback: estimate parameters
            return sum(p.numel() for p in topology_network.parameters())
    
    def _debug_network_structure(self):
        """Debug network structure."""
        if wandb.run:
            actor_params = self._get_topology_params(self.actor_topology)
            critic_params = self._get_topology_params(self.critic_topology)
            
            wandb.log({
                'network/actor_parameters': actor_params,
                'network/critic_parameters': critic_params,
                'network/total_parameters': actor_params + critic_params,
                'network/topology_type': self.topology_type,
                'network/hidden_size': self.hidden_size,
                'network/num_layers': self.num_layers
            })
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create input mask for the current task."""
        # Simplified: no input masking needed for CartPole and Acrobot
        return torch.ones_like(x)
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply input masking."""
        return x * mask
    
    def forward_actor(self, obs):
        """Forward pass through actor network."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Apply input masking if needed
        mask = self._create_input_mask(obs)
        obs = self._apply_input_masking(obs, mask)
        
        # Convert to dictionary format for topology network
        input_dict = {i: obs[:, i] for i in range(obs.shape[1])}
        
        # Forward through topology network
        output_dict = self.actor_topology.forward(input_dict)
        
        # Convert output dictionary to tensor
        output_values = list(output_dict.values())
        if output_values:
            return torch.stack(output_values, dim=1)
        else:
            # Fallback: return zeros
            return torch.zeros(obs.shape[0], len(output_dict), device=obs.device)
    
    def forward_critic(self, obs):
        """Forward pass through critic network."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Apply input masking if needed
        mask = self._create_input_mask(obs)
        obs = self._apply_input_masking(obs, mask)
        
        # Convert to dictionary format for topology network
        input_dict = {i: obs[:, i] for i in range(obs.shape[1])}
        
        # Forward through topology network
        output_dict = self.critic_topology.forward(input_dict)
        
        # Convert output dictionary to tensor
        output_values = list(output_dict.values())
        if output_values:
            return torch.stack(output_values, dim=1)
        else:
            # Fallback: return zeros
            return torch.zeros(obs.shape[0], len(output_dict), device=obs.device)
    
    def get_action_mask(self):
        """Get action mask for the current task."""
        if self.current_task == 'CartPole-v1':
            # CartPole: actions 0 and 1 are valid
            return [True, True, False]
        elif self.current_task == 'Acrobot-v1':
            # Acrobot: actions 0, 1, and 2 are valid
            return [True, True, True]
        else:
            # Default: all actions valid
            return [True, True, True]

# ============================================================================
# ENHANCED DEBUG CALLBACK (Simplified with Sequential Training Support)
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
        
        # Sequential training tracking
        self.current_task_phase = 0
        self.current_task = None # Track current task for reward collection
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
            'Acrobot-v1': []
        }
        self.current_task = None
    
    def set_task_phase(self, task_name, phase_number):
        """Set the current task phase for sequential training."""
        self.current_task_phase = phase_number
        self.current_task = task_name  # Track current task for reward collection
        self.task_phases.append({
            'phase': phase_number,
            'task': task_name,
            'start_timesteps': self.num_timesteps if hasattr(self, 'num_timesteps') else 0
        })
        self.phase_start_timesteps.append(self.num_timesteps if hasattr(self, 'num_timesteps') else 0)
        
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
                'sequential_training/total_timesteps': self.num_timesteps if hasattr(self, 'num_timesteps') else 0
            })
    
    def _on_step(self) -> bool:
        """Called at each step during training."""
        # Only log detailed metrics every log_freq steps to avoid step warnings
        if self.num_timesteps % self.log_freq == 0:
            self._log_training_metrics()
            self._log_graph_metrics()
            self._log_depth_analysis()
            self._log_sample_efficiency()
            self._log_hyperparameter_correlation()
        
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
    
    def get_task_rewards(self):
        """Get collected task rewards for normalized metrics calculation."""
        return self.task_rewards.copy()
    
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
                        "network/total_parameters": actor_params + critic_params,
                    })
                    
                    # Add enhanced graph metrics
                    self._log_graph_metrics()
                    self._log_depth_analysis()
                    self._log_sample_efficiency()
                    self._log_hyperparameter_correlation()
                
                wandb.log(metrics, step=self.num_timesteps)
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
                    
                    wandb.log(metrics, step=self.num_timesteps)
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
                
                wandb.log(summary, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging final summary: {e}")
    
    def _log_graph_metrics(self):
        """Log graph metrics with phase tracking."""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                actor_topology = self.model.policy.actor_topology
                critic_topology = self.model.policy.critic_topology
                
                # Actor metrics
                actor_metrics = self._calculate_graph_metrics(actor_topology, 'actor')
                for key, value in actor_metrics.items():
                    wandb.log({f'graph/actor/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.num_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/actor/{key}': value}, step=self.num_timesteps)
                
                # Critic metrics
                critic_metrics = self._calculate_graph_metrics(critic_topology, 'critic')
                for key, value in critic_metrics.items():
                    wandb.log({f'graph/critic/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.num_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/critic/{key}': value}, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging graph metrics: {e}")
    
    def _calculate_graph_metrics(self, topology_network, network_type):
        """Calculate graph metrics for the network."""
        try:
            if topology_network is None:
                return {}
            
            # Get the underlying networkx graph from FeedForwardNetwork
            if hasattr(topology_network, 'topology'):
                G = topology_network.topology
            else:
                print(f"   ⚠️  No topology graph found in {network_type} network")
                return {}
            
            # Check if graph is connected
            is_connected = nx.is_connected(G.to_undirected()) if G.is_directed() else nx.is_connected(G)
            
            # Basic metrics that always work
            metrics = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'avg_degree': np.mean([d for n, d in G.degree()]),
            }
            
            # Metrics that require connected graphs
            if is_connected:
                G_undirected = G.to_undirected() if G.is_directed() else G
                try:
                    metrics.update({
                        'clustering_coefficient': nx.average_clustering(G_undirected),
                        'diameter': nx.diameter(G_undirected),
                        'avg_shortest_path': nx.average_shortest_path_length(G_undirected),
                    })
                except Exception as e:
                    print(f"   ⚠️  Error calculating connected graph metrics: {e}")
                    metrics.update({
                        'clustering_coefficient': 0.0,
                        'diameter': 0,
                        'avg_shortest_path': 0.0,
                    })
            else:
                # For disconnected graphs, calculate metrics on largest component
                largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
                if len(largest_cc) > 1:
                    G_largest = G.subgraph(largest_cc).to_undirected()
                    try:
                        metrics.update({
                            'clustering_coefficient': nx.average_clustering(G_largest),
                            'diameter': nx.diameter(G_largest),
                            'avg_shortest_path': nx.average_shortest_path_length(G_largest),
                        })
                    except Exception as e:
                        print(f"   ⚠️  Error calculating largest component metrics: {e}")
                        metrics.update({
                            'clustering_coefficient': 0.0,
                            'diameter': 0,
                            'avg_shortest_path': 0.0,
                        })
                else:
                    metrics.update({
                        'clustering_coefficient': 0.0,
                        'diameter': 0,
                        'avg_shortest_path': 0.0,
                    })
            
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
                
                # Actor depth analysis
                actor_depth = self._calculate_depth_metrics(actor_topology, 'actor')
                for key, value in actor_depth.items():
                    wandb.log({f'depth/actor/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.num_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/actor_depth/{key}': value}, step=self.num_timesteps)
                
                # Critic depth analysis
                critic_depth = self._calculate_depth_metrics(critic_topology, 'critic')
                for key, value in critic_depth.items():
                    wandb.log({f'depth/critic/{key}': value, 'sequential_training/phase': self.current_task_phase}, step=self.num_timesteps)
                    wandb.log({f'phase_{self.current_task_phase}/critic_depth/{key}': value}, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging depth analysis: {e}")
    
    def _calculate_depth_metrics(self, topology_network, network_type):
        """Calculate depth metrics for the network."""
        try:
            if topology_network is None:
                return {}
            
            # Get the underlying networkx graph from FeedForwardNetwork
            if hasattr(topology_network, 'topology'):
                G = topology_network.topology
            else:
                print(f"   ⚠️  No topology graph found in {network_type} network")
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
                    }, step=self.num_timesteps)
                    
                    wandb.log({
                        f'phase_{self.current_task_phase}/sample_efficiency/recent_mean_reward': np.mean(recent_rewards),
                        f'phase_{self.current_task_phase}/sample_efficiency/efficiency_score': sample_efficiency
                    }, step=self.num_timesteps)
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
                        }, step=self.num_timesteps)
                        
                        wandb.log({
                            f'phase_{self.current_task_phase}/hyperparameter_correlation/learning_rate': current_lr,
                            f'phase_{self.current_task_phase}/hyperparameter_correlation/reward_trend': reward_trend
                        }, step=self.num_timesteps)
        except Exception as e:
            print(f"   ⚠️  Error logging hyperparameter correlation: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_debug_config():
    """Create debug configuration for testing."""
    return {
        'topology_type': 'small_world',
        'hidden_size': 64,
        'num_layers': 2,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 2048,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'activation': 'relu',
        'dropout': 0.0,
        'total_timesteps': 10000,  # Short for testing
        'n_eval_episodes': 3,
        'train_task_1': 'CartPole-v1',
        'train_task_2': 'Acrobot-v1',
        # Topology-specific parameters
        'small_world_k': 4,
        'small_world_p': 0.1,
        'modular_num_modules': 4,
        'modular_inter_module_prob': 0.05,
        'modular_intra_module_prob': 0.7,
        'hybrid_num_modules': 4,
        'hybrid_k': 4,
        'hybrid_p': 0.1,
        'hybrid_inter_module_prob': 0.05,
    }

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
    
    # Use tqdm for progress bar
    for episode in tqdm(range(n_eval_episodes), desc="Evaluating", leave=False):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, done, truncated, _ = step_result
            else:
                obs, reward, done, truncated = step_result
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return episode_rewards, episode_lengths

def evaluate_model_enhanced(model, env, task_name, n_eval_episodes=3):
    """Enhanced evaluation with task-specific metrics."""
    episode_rewards, episode_lengths = evaluate_model(model, env, n_eval_episodes)
    
    # Calculate success rate
    success_rate = calculate_success_rate(episode_rewards, episode_lengths, task_name)
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    return episode_rewards, episode_lengths, success_rate, mean_reward, std_reward, mean_length

def calculate_success_rate(rewards, episode_lengths, task_name):
    """Calculate success rate based on task-specific criteria."""
    if task_name == 'CartPole-v1':
        # Success: episode length >= 195 (close to max of 500)
        return np.mean([length >= 195 for length in episode_lengths])
    elif task_name == 'Acrobot-v1':
        # Success: reward >= -100 (close to optimal)
        return np.mean([reward >= -100 for reward in rewards])
    else:
        # Default: above average performance
        mean_reward = np.mean(rewards)
        return np.mean([reward >= mean_reward for reward in rewards])

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def simplified_double_task_training(policy_class, topology_type, config, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None):
    """
    Simplified double-task training: train sequentially on two different tasks.
    Only supports CartPole-v1 and Acrobot-v1.
    Creates separate training and testing runs for better interpretability.
    """
    # Skip same-task combinations
    if train_task_1 == train_task_2:
        print(f"⏭️  SKIPPING: Same-task combination {train_task_1} → {train_task_2}")
        print("   This would train on the same task twice, which is not the intended experiment.")
        return {
            'skipped': True,
            'reason': 'same_task_combination',
            'train_task_1': train_task_1,
            'train_task_2': train_task_2,
            'sequential_training': True,
            'simplified_mode': True
        }
    
    # Validate task names
    valid_tasks = ['CartPole-v1', 'Acrobot-v1']
    if train_task_1 not in valid_tasks or train_task_2 not in valid_tasks:
        print(f"⏭️  SKIPPING: Invalid task combination {train_task_1} → {train_task_2}")
        print(f"   Only {valid_tasks} are supported in simplified mode.")
        return {
            'skipped': True,
            'reason': 'invalid_task_combination',
            'train_task_1': train_task_1,
            'train_task_2': train_task_2,
            'sequential_training': True,
            'simplified_mode': True
        }
    
    print(f"🎯 SIMPLIFIED DOUBLE-TASK TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"   • Task 1: {train_task_1}")
    print(f"   • Task 2: {train_task_2}")
    print(f"   • Hidden Size: {hidden_size}")
    print(f"   • Layers: {num_layers}")
    print(f"   • Mode: Sequential training with separate testing runs")
    
    # Create environments
    env1 = DummyVecEnv([make_env(train_task_1)])
    env2 = DummyVecEnv([make_env(train_task_2)])
    
    # Get network parameters for run naming
    temp_model = PPO(
        policy_class,
        env1,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=0,
        tensorboard_log=None
    )
    
    # Get parameter counts
    actor_params = temp_model.policy._get_topology_params(temp_model.policy.actor_topology)
    critic_params = temp_model.policy._get_topology_params(temp_model.policy.critic_topology)
    total_params = actor_params + critic_params
    
    # Create descriptive run names
    training_run_name = f"training_{topology_type}_{num_layers}_{hidden_size}_{total_params}_{train_task_1}_{train_task_2}"
    
    # Initialize training run
    try:
        training_wandb_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--simplified-double-task-training",
            name=training_run_name,
            config={
                "run_type": "sequential_training",
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "total_params": total_params,
                "actor_params": actor_params,
                "critic_params": critic_params,
                "train_task_1": train_task_1,
                "train_task_2": train_task_2,
                "total_timesteps": config['total_timesteps'],
                "n_eval_episodes": config['n_eval_episodes'],
                "learning_rate": config['learning_rate'],
                "batch_size": config['batch_size'],
                "n_steps": config['n_steps'],
                "n_epochs": config['n_epochs'],
                "gamma": config['gamma'],
                "gae_lambda": config['gae_lambda'],
                "clip_range": config['clip_range'],
                "ent_coef": config['ent_coef'],
                "max_grad_norm": config['max_grad_norm'],
                "activation": config['activation'],
                "dropout": config['dropout'],
                "experiment_type": "sequential_double_task",
            },
            tags=[topology_type, f"layers_{num_layers}", f"size_{hidden_size}", f"params_{total_params}", train_task_1, train_task_2, "sequential_training"],
            reinit=True
        )
        training_wandb_enabled = True
        print(f"   ✅ Training run initialized: {training_run_name}")
    except Exception as e:
        print(f"   ⚠️  Training WandB logging disabled: {e}")
        training_wandb_run = None
        training_wandb_enabled = False
    
    # Create callback for training tracking
    callback = EnhancedDebugCallback(verbose=1, wandb_run=training_wandb_run, log_freq=1000)
    
    # Initialize model with first environment
    model = PPO(
        policy_class,
        env1,
        learning_rate=config['learning_rate'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        gamma=config['gamma'],
        gae_lambda=config['gae_lambda'],
        clip_range=config['clip_range'],
        ent_coef=config['ent_coef'],
        max_grad_norm=config['max_grad_norm'],
        verbose=0,
        tensorboard_log=None
    )
    
    # Set current task for action masking
    model.policy.set_current_task(train_task_1)
    
    # PHASE 1: Train on Task 1
    print(f"🚀 PHASE 1: Training on {train_task_1}...")
    callback.set_task_phase(train_task_1, 1)
    
    # Train with progress bar
    start_time = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    phase1_time = time.time() - start_time
    
    # Log Phase 1 training metrics
    if training_wandb_enabled:
        training_wandb_run.log({
            "training/phase_1_task1_training_time": phase1_time,
            "training/phase_1_task1_timesteps_per_second": config['total_timesteps'] / phase1_time,
            "training/phase_1_task1_completed": True
        })
    
    # Evaluate both tasks after Phase 1 (create separate testing run)
    print(f"📊 EVALUATION AFTER PHASE 1: Testing both tasks after training on {train_task_1}...")
    
    # Evaluate on Task 1 (baseline performance)
    rewards1_after_task1, lengths1_after_task1, success1_after_task1, mean1_after_task1, std1_after_task1, mean_len1_after_task1 = evaluate_model_enhanced(
        model, env1, train_task_1, config['n_eval_episodes']
    )
    
    # Evaluate on Task 2 (initial transfer)
    rewards2_after_task1, lengths2_after_task1, success2_after_task1, mean2_after_task1, std2_after_task1, mean_len2_after_task1 = evaluate_model_enhanced(
        model, env2, train_task_2, config['n_eval_episodes']
    )
    
    print(f"   • {train_task_1} (trained): {mean1_after_task1:.2f} (success: {success1_after_task1:.1%})")
    print(f"   • {train_task_2} (untrained): {mean2_after_task1:.2f} (success: {success2_after_task1:.1%})")
    
    # Create Phase 1 testing run
    phase1_testing_run_name = f"testing_{topology_type}_{num_layers}_{hidden_size}_{total_params}_{train_task_1}_{train_task_2}_phase1"
    try:
        phase1_testing_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--simplified-double-task-training",
            name=phase1_testing_run_name,
            config={
                "run_type": "phase1_testing",
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "total_params": total_params,
                "train_task_1": train_task_1,
                "train_task_2": train_task_2,
                "phase": 1,
                "n_eval_episodes": config['n_eval_episodes'],
                "experiment_type": "sequential_double_task",
            },
            tags=[topology_type, f"layers_{num_layers}", f"size_{hidden_size}", f"params_{total_params}", train_task_1, train_task_2, "phase1_testing"],
            reinit=True
        )
        
        # Log Phase 1 testing metrics
        phase1_testing_run.log({
            # Task 1 (trained) metrics
            "testing/phase1_task1_trained_mean_reward": mean1_after_task1,
            "testing/phase1_task1_trained_std_reward": std1_after_task1,
            "testing/phase1_task1_trained_success_rate": success1_after_task1,
            "testing/phase1_task1_trained_mean_length": mean_len1_after_task1,
            
            # Task 2 (transfer) metrics
            "testing/phase1_task2_transfer_mean_reward": mean2_after_task1,
            "testing/phase1_task2_transfer_std_reward": std2_after_task1,
            "testing/phase1_task2_transfer_success_rate": success2_after_task1,
            "testing/phase1_task2_transfer_mean_length": mean_len2_after_task1,
            
            # Transfer learning metrics
            "transfer/phase1_task2_transfer_ratio": mean2_after_task1 / mean1_after_task1 if mean1_after_task1 != 0 else 0,
            "transfer/phase1_task2_relative_performance": (mean2_after_task1 / mean1_after_task1 * 100) if mean1_after_task1 != 0 else 0,
            
            # Network architecture
            "network/topology_type": topology_type,
            "network/layers": num_layers,
            "network/size": hidden_size,
            "network/total_parameters": total_params,
        })
        
        phase1_testing_run.finish()
        print(f"   ✅ Phase 1 testing run completed: {phase1_testing_run_name}")
        
    except Exception as e:
        print(f"   ⚠️  Error creating Phase 1 testing run: {e}")
    
    # PHASE 2: Train on Task 2
    print(f"🚀 PHASE 2: Training on {train_task_2}...")
    
    # Switch to second environment and task
    model.set_env(env2)
    model.policy.set_current_task(train_task_2)
    callback.set_task_phase(train_task_2, 2)
    
    # Train with progress bar
    start_time = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    phase2_time = time.time() - start_time
    
    # Log Phase 2 training metrics
    if training_wandb_enabled:
        training_wandb_run.log({
            "training/phase_2_task2_training_time": phase2_time,
            "training/phase_2_task2_timesteps_per_second": config['total_timesteps'] / phase2_time,
            "training/phase_2_task2_completed": True
        })
    
    # Evaluate both tasks after Phase 2
    print(f"📊 EVALUATION AFTER PHASE 2: Testing both tasks after training on {train_task_2}...")
    
    # Evaluate on Task 1 (retention)
    rewards1_after_task2, lengths1_after_task2, success1_after_task2, mean1_after_task2, std1_after_task2, mean_len1_after_task2 = evaluate_model_enhanced(
        model, env1, train_task_1, config['n_eval_episodes']
    )
    
    # Evaluate on Task 2 (final performance)
    rewards2_after_task2, lengths2_after_task2, success2_after_task2, mean2_after_task2, std2_after_task2, mean_len2_after_task2 = evaluate_model_enhanced(
        model, env2, train_task_2, config['n_eval_episodes']
    )
    
    print(f"   • {train_task_1} (retention): {mean1_after_task2:.2f} (success: {success1_after_task2:.1%})")
    print(f"   • {train_task_2} (trained): {mean2_after_task2:.2f} (success: {success2_after_task2:.1%})")
    
    # Create Phase 2 testing run
    phase2_testing_run_name = f"testing_{topology_type}_{num_layers}_{hidden_size}_{total_params}_{train_task_1}_{train_task_2}_phase2"
    try:
        phase2_testing_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--simplified-double-task-training",
            name=phase2_testing_run_name,
            config={
                "run_type": "phase2_testing",
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "total_params": total_params,
                "train_task_1": train_task_1,
                "train_task_2": train_task_2,
                "phase": 2,
                "n_eval_episodes": config['n_eval_episodes'],
                "experiment_type": "sequential_double_task",
            },
            tags=[topology_type, f"layers_{num_layers}", f"size_{hidden_size}", f"params_{total_params}", train_task_1, train_task_2, "phase2_testing"],
            reinit=True
        )
        
        # Calculate retention and learning metrics
        retention_ratio = mean1_after_task2 / mean1_after_task1 if mean1_after_task1 != 0 else 0
        forgetting_rate = 1 - retention_ratio
        learning_ratio = mean2_after_task2 / mean2_after_task1 if mean2_after_task1 != 0 else 0
        task_similarity = mean2_after_task1 / mean1_after_task1 if mean1_after_task1 != 0 else 0
        
        # Calculate normalized scores (0-1 scale)
        # Task-specific maximum rewards for normalization
        max_rewards = {
            'CartPole-v1': 500.0,
            'Acrobot-v1': -80.0,
        }
        
        def normalize_reward(reward, task):
            max_reward = max_rewards.get(task, 500.0)
            if max_reward < 0:  # Negative rewards (Acrobot)
                return (reward - max_reward) / abs(max_reward)
            else:  # Positive rewards (CartPole)
                return reward / max_reward
        
        task1_baseline_normalized = normalize_reward(mean1_after_task1, train_task_1)
        task1_retention_normalized = normalize_reward(mean1_after_task2, train_task_1)
        task2_final_normalized = normalize_reward(mean2_after_task2, train_task_2)
        
        # Log Phase 2 testing metrics
        phase2_testing_run.log({
            # Task 1 (retention) metrics
            "testing/phase2_task1_retention_mean_reward": mean1_after_task2,
            "testing/phase2_task1_retention_std_reward": std1_after_task2,
            "testing/phase2_task1_retention_success_rate": success1_after_task2,
            "testing/phase2_task1_retention_mean_length": mean_len1_after_task2,
            
            # Task 2 (final) metrics
            "testing/phase2_task2_final_mean_reward": mean2_after_task2,
            "testing/phase2_task2_final_std_reward": std2_after_task2,
            "testing/phase2_task2_final_success_rate": success2_after_task2,
            "testing/phase2_task2_final_mean_length": mean_len2_after_task2,
            
            # Retention analysis
            "retention/task1_retention_ratio": retention_ratio,
            "retention/task1_forgetting_rate": forgetting_rate,
            "retention/task1_retention_success_ratio": success1_after_task2 / success1_after_task1 if success1_after_task1 > 0 else 0,
            
            # Learning analysis
            "learning/task2_learning_ratio": learning_ratio,
            "learning/task2_learning_success_ratio": success2_after_task2 / success2_after_task1 if success2_after_task1 > 0 else 0,
            
            # Task similarity
            "task_similarity/phase1_transfer_ratio": task_similarity,
            "task_similarity/phase1_transfer_success_ratio": success2_after_task1 / success1_after_task1 if success1_after_task1 > 0 else 0,
            
            # Normalized performance scores
            "normalized/task1_baseline_score": task1_baseline_normalized,
            "normalized/task1_retention_score": task1_retention_normalized,
            "normalized/task2_final_score": task2_final_normalized,
            "normalized/overall_performance_score": (task1_retention_normalized + task2_final_normalized) / 2,
            
            # Network architecture
            "network/topology_type": topology_type,
            "network/layers": num_layers,
            "network/size": hidden_size,
            "network/total_parameters": total_params,
        })
        
        phase2_testing_run.finish()
        print(f"   ✅ Phase 2 testing run completed: {phase2_testing_run_name}")
        
    except Exception as e:
        print(f"   ⚠️  Error creating Phase 2 testing run: {e}")
    
    # Finish training run
    if training_wandb_enabled:
        total_training_time = phase1_time + phase2_time
        training_wandb_run.log({
            "training/total_training_time": total_training_time,
            "training/total_timesteps": config['total_timesteps'] * 2,
            "training/timesteps_per_second": (config['total_timesteps'] * 2) / total_training_time,
            "training/sequential_training_completed": True
        })
        
        # Create training summary table
        training_summary_table = wandb.Table(columns=["Metric", "Value", "Description"])
        training_summary_table.add_data("Topology Type", topology_type, "Network topology used")
        training_summary_table.add_data("Number of Layers", str(num_layers), "Network depth")
        training_summary_table.add_data("Hidden Size", str(hidden_size), "Hidden units per layer")
        training_summary_table.add_data("Total Parameters", f"{total_params:,}", "Trainable parameters")
        training_summary_table.add_data("Task 1", train_task_1, "First training task")
        training_summary_table.add_data("Task 2", train_task_2, "Second training task")
        training_summary_table.add_data("Total Training Time", f"{total_training_time:.2f}s", "Total training duration")
        training_summary_table.add_data("Timesteps/sec", f"{(config['total_timesteps'] * 2) / total_training_time:.2f}", "Training speed")
        
        training_wandb_run.log({"training_summary": training_summary_table})
        training_wandb_run.finish()
        print(f"   ✅ Training run completed: {training_run_name}")
    
    # Calculate comprehensive analysis
    print(f"📈 COMPREHENSIVE ANALYSIS:")
    
    # Retention analysis (Task 1 performance after training on Task 2)
    retention_rate = mean1_after_task2 / mean1_after_task1 if mean1_after_task1 != 0 else 0
    forgetting_rate = 1 - retention_rate
    print(f"   • Task 1 Retention: {retention_rate:.1%} (forgetting: {forgetting_rate:.1%})")
    
    # Learning analysis (Task 2 performance improvement)
    learning_rate = mean2_after_task2 / mean2_after_task1 if mean2_after_task1 != 0 else 0
    print(f"   • Task 2 Learning: {learning_rate:.1%}")
    
    # Task similarity analysis (Task 2 baseline vs Task 1 baseline)
    task_similarity = mean2_after_task1 / mean1_after_task1 if mean1_after_task1 != 0 else 0
    print(f"   • Task Similarity (Task2/Task1 baseline): {task_similarity:.1%}")
    
    # Collect results for return
    result = {
        'sequential_training': True,
        'simplified_mode': True,
        'topology_type': topology_type,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'total_params': total_params,
        'train_task_1': train_task_1,
        'train_task_2': train_task_2,
        'phase1_task1_baseline_reward': mean1_after_task1,
        'phase1_task2_transfer_reward': mean2_after_task1,
        'phase2_task1_retention_reward': mean1_after_task2,
        'phase2_task2_final_reward': mean2_after_task2,
        'retention_ratio': retention_rate,
        'forgetting_rate': forgetting_rate,
        'learning_ratio': learning_rate,
        'task_similarity': task_similarity,
        'total_training_time': phase1_time + phase2_time
    }
    
    print(f"✅ Simplified double-task training completed!")
    print(f"   • Total training time: {phase1_time + phase2_time:.2f}s")
    print(f"   • Training runs: {training_run_name}")
    print(f"   • Testing runs: {phase1_testing_run_name}, {phase2_testing_run_name}")
    
    return result

# ============================================================================
# SWEEP TRAINING FUNCTION
# ============================================================================

def train_with_sweep():
    """Main training function for sweep execution."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--topology_type', type=str, default='small_world', help='Topology type')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_range', type=float, default=0.2, help='Clip range')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='Max gradient norm')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--total_timesteps', type=int, default=10000, help='Total timesteps')
    parser.add_argument('--n_eval_episodes', type=int, default=3, help='Number of evaluation episodes')
    parser.add_argument('--train_task_1', type=str, default='CartPole-v1', help='First training task')
    parser.add_argument('--train_task_2', type=str, default='Acrobot-v1', help='Second training task')
    
    # Topology-specific parameters
    parser.add_argument('--small_world_k', type=int, default=4, help='Small world k parameter')
    parser.add_argument('--small_world_p', type=float, default=0.1, help='Small world p parameter')
    parser.add_argument('--modular_num_modules', type=int, default=4, help='Modular number of modules')
    parser.add_argument('--modular_inter_module_prob', type=float, default=0.05, help='Modular inter-module probability')
    parser.add_argument('--modular_intra_module_prob', type=float, default=0.7, help='Modular intra-module probability')
    parser.add_argument('--hybrid_num_modules', type=int, default=4, help='Hybrid number of modules')
    parser.add_argument('--hybrid_k', type=int, default=4, help='Hybrid k parameter')
    parser.add_argument('--hybrid_p', type=float, default=0.1, help='Hybrid p parameter')
    parser.add_argument('--hybrid_inter_module_prob', type=float, default=0.05, help='Hybrid inter-module probability')
    
    # Parse known args to ignore WandB arguments
    args, unknown = parser.parse_known_args()
    
    if args.debug:
        print("🐛 Running in debug mode...")
        config = create_debug_config()
    else:
        # Configuration is handled within the training function for separate runs
        config = {
            'topology_type': args.topology_type,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'n_steps': args.n_steps,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'max_grad_norm': args.max_grad_norm,
            'activation': args.activation,
            'dropout': args.dropout,
            'total_timesteps': args.total_timesteps,
            'n_eval_episodes': args.n_eval_episodes,
            'train_task_1': args.train_task_1,
            'train_task_2': args.train_task_2,
            'small_world_k': args.small_world_k,
            'small_world_p': args.small_world_p,
            'modular_num_modules': args.modular_num_modules,
            'modular_inter_module_prob': args.modular_inter_module_prob,
            'modular_intra_module_prob': args.modular_intra_module_prob,
            'hybrid_num_modules': args.hybrid_num_modules,
            'hybrid_k': args.hybrid_k,
            'hybrid_p': args.hybrid_p,
            'hybrid_inter_module_prob': args.hybrid_inter_module_prob,
        }
    
    # Run training with progress tracking
    print(f"🎯 Starting simplified double-task training...")
    print(f"   • Topology: {config['topology_type']}")
    print(f"   • Tasks: {config['train_task_1']} → {config['train_task_2']}")
    print(f"   • Timesteps: {config['total_timesteps']}")
    
    try:
        result = simplified_double_task_training(
            policy_class=DebugTopologyPolicy,
            topology_type=config['topology_type'],
            config=config,
            num_layers=config['num_layers'],
            hidden_size=config['hidden_size'],
            train_task_1=config['train_task_1'],
            train_task_2=config['train_task_2']
        )
        
        # Handle skipped runs
        if result.get('skipped', False):
            print(f"✅ Simplified double-task training completed!")
            print(f"   • Status: SKIPPED ({result.get('reason', 'unknown')})")
            print(f"   • Tasks: {result['train_task_1']} → {result['train_task_2']}")
            return result
        
        print(f"✅ Simplified double-task training completed!")
        return result
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        if wandb.run:
            wandb.log({'error': str(e)})
        return {'error': str(e)}

if __name__ == "__main__":
    result = train_with_sweep() 