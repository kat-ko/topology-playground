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

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.networks.topology_networks import (
    SmallWorldTopology, ModularTopology, HybridTopology, FullyConnectedTopology
)
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
    
    def step(self, action):
        """Execute action and return standardized observation."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Pad observation to 6 dimensions if needed
        obs = self._pad_observation(obs)
        
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
        
        # Create topology networks
        self.actor_topology = self._create_topology_network('actor')
        self.critic_topology = self._create_topology_network('critic')
        
        # Debug network structure
        self._debug_network_structure()
    
    def _create_topology_network(self, network_type):
        """Create topology network based on type."""
        if self.topology_type == 'small_world':
            return SmallWorldTopology(
                input_size=self.observation_space.shape[0],
                hidden_size=self.hidden_size,
                output_size=self.action_space.n if network_type == 'actor' else 1,
                k=self.config.get('small_world_k', 4),
                p=self.config.get('small_world_p', 0.1)
            )
        elif self.topology_type == 'modular':
            return ModularTopology(
                input_size=self.observation_space.shape[0],
                hidden_size=self.hidden_size,
                output_size=self.action_space.n if network_type == 'actor' else 1,
                num_modules=self.config.get('modular_num_modules', 4),
                inter_module_prob=self.config.get('modular_inter_module_prob', 0.05),
                intra_module_prob=self.config.get('modular_intra_module_prob', 0.7)
            )
        elif self.topology_type == 'hybrid':
            return HybridTopology(
                input_size=self.observation_space.shape[0],
                hidden_size=self.hidden_size,
                output_size=self.action_space.n if network_type == 'actor' else 1,
                num_modules=self.config.get('hybrid_num_modules', 4),
                k=self.config.get('hybrid_k', 4),
                p=self.config.get('hybrid_p', 0.1),
                inter_module_prob=self.config.get('hybrid_inter_module_prob', 0.05)
            )
        elif self.topology_type == 'fully_connected':
            return FullyConnectedTopology(
                input_size=self.observation_space.shape[0],
                hidden_size=self.hidden_size,
                output_size=self.action_space.n if network_type == 'actor' else 1,
                num_layers=self.num_layers
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
    
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
        
        return self.actor_topology(obs)
    
    def forward_critic(self, obs):
        """Forward pass through critic network."""
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        
        # Apply input masking if needed
        mask = self._create_input_mask(obs)
        obs = self._apply_input_masking(obs, mask)
        
        return self.critic_topology(obs)
    
    def get_action_mask(self):
        """Get action mask for the current task."""
        # This will be set by the environment wrapper
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
        """Log metrics on each step."""
        self.step_count += 1
        
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
            })
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
    Simplified double-task training function with sweep support and comprehensive retention measurement.
    
    This function trains sequentially on two tasks and evaluates both tasks after each training phase
    to measure retention, transfer learning, and task similarity effects.
    
    Args:
        policy_class: Policy class to use
        topology_type: Type of topology network
        config: Configuration dictionary
        num_layers: Number of layers
        hidden_size: Hidden layer size
        train_task_1: First training task (CartPole-v1 or Acrobot-v1)
        train_task_2: Second training task (CartPole-v1 or Acrobot-v1)
    """
    # Skip same-task combinations (we don't want to train on the same task twice)
    if train_task_1 == train_task_2:
        print(f"⏭️  SKIPPING: Same-task combination {train_task_1} → {train_task_2}")
        print(f"   This would train on the same task twice, which is not the intended experiment.")
        return {
            'skipped': True,
            'reason': 'same_task_combination',
            'train_task_1': train_task_1,
            'train_task_2': train_task_2
        }
    
    print(f"🎯 SIMPLIFIED DOUBLE-TASK TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"   • Task 1: {train_task_1}")
    print(f"   • Task 2: {train_task_2}")
    print(f"   • Hidden Size: {hidden_size}")
    print(f"   • Layers: {num_layers}")
    print(f"   • Mode: Sequential training with comprehensive evaluation")
    
    # Validate tasks
    if train_task_1 not in ['CartPole-v1', 'Acrobot-v1'] or train_task_2 not in ['CartPole-v1', 'Acrobot-v1']:
        raise ValueError(f"Invalid tasks: {train_task_1}, {train_task_2}. Only CartPole-v1 and Acrobot-v1 are supported in simplified mode.")
    
    # Initialize wandb if not already done
    if wandb.run is None:
        wandb.init(
            project="topologies--simplified-double-task-training",
            entity="katko-it-universitetet-i-k-benhavn",
            config=config,
            name=f"simplified_double_task_{topology_type}_{train_task_1}_{train_task_2}"
        )
    
    # Create environments
    env1 = make_env(train_task_1)()
    env2 = make_env(train_task_2)()
    
    # Create model
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
        device='cpu',
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
    # PHASE 1: Train on Task 1
    # ============================================================================
    print(f"🚀 PHASE 1: Training on {train_task_1}...")
    callback.set_task_phase(train_task_1, 1)  # Set phase 1
    model.learn(total_timesteps=config['total_timesteps'], callback=callback)
    
    # ============================================================================
    # EVALUATION AFTER PHASE 1: Test both tasks after Task 1 training
    # ============================================================================
    print(f"📊 EVALUATION AFTER PHASE 1: Testing both tasks after training on {train_task_1}...")
    
    # Test on task 1 (baseline performance)
    eval_env1_after_task1 = make_env(train_task_1)()
    rewards1_after_task1, lengths1_after_task1, success1_after_task1 = evaluate_model_enhanced(
        model, eval_env1_after_task1, train_task_1, config['n_eval_episodes']
    )
    eval_env1_after_task1.close()
    
    # Test on task 2 (transfer learning check)
    eval_env2_after_task1 = make_env(train_task_2)()
    rewards2_after_task1, lengths2_after_task1, success2_after_task1 = evaluate_model_enhanced(
        model, eval_env2_after_task1, train_task_2, config['n_eval_episodes']
    )
    eval_env2_after_task1.close()
    
    print(f"   • {train_task_1} (trained): {np.mean(rewards1_after_task1):.2f} (success: {success1_after_task1:.1%})")
    print(f"   • {train_task_2} (untrained): {np.mean(rewards2_after_task1):.2f} (success: {success2_after_task1:.1%})")
    
    # Compute normalized metrics for Phase 1
    phase1_task_rewards = {
        train_task_1: callback.task_rewards.get(train_task_1, []),
        train_task_2: callback.task_rewards.get(train_task_2, [])
    }
    
    if any(len(rewards) > 0 for rewards in phase1_task_rewards.values()):
        phase1_metrics = compute_multi_task_metrics(phase1_task_rewards, config['total_timesteps'])
        log_normalized_metrics(wandb.run, phase1_metrics['task_metrics'], 
                             phase1_metrics['final_normalized_score'], 
                             phase1_metrics['efficiency_score'], "phase1")
        print_normalized_summary(phase1_metrics['task_metrics'], 
                               phase1_metrics['final_normalized_score'], 
                               phase1_metrics['efficiency_score'], "Phase 1")
    
    # ============================================================================
    # PHASE 2: Train on Task 2
    # ============================================================================
    print(f"🚀 PHASE 2: Training on {train_task_2}...")
    callback.set_task_phase(train_task_2, 2)  # Set phase 2
    model.set_env(env2)  # Switch environment for second task
    model.learn(total_timesteps=config['total_timesteps'], callback=callback)
    
    # ============================================================================
    # EVALUATION AFTER PHASE 2: Test both tasks after Task 2 training
    # ============================================================================
    print(f"📊 EVALUATION AFTER PHASE 2: Testing both tasks after training on {train_task_2}...")
    
    # Test on task 1 (retention check)
    eval_env1_after_task2 = make_env(train_task_1)()
    rewards1_after_task2, lengths1_after_task2, success1_after_task2 = evaluate_model_enhanced(
        model, eval_env1_after_task2, train_task_1, config['n_eval_episodes']
    )
    
    # Test on task 2 (final performance)
    eval_env2_after_task2 = make_env(train_task_2)()
    rewards2_after_task2, lengths2_after_task2, success2_after_task2 = evaluate_model_enhanced(
        model, eval_env2_after_task2, train_task_2, config['n_eval_episodes']
    )
    
    print(f"   • {train_task_1} (retention): {np.mean(rewards1_after_task2):.2f} (success: {success1_after_task2:.1%})")
    print(f"   • {train_task_2} (trained): {np.mean(rewards2_after_task2):.2f} (success: {success2_after_task2:.1%})")
    
    # Compute normalized metrics for Phase 2
    phase2_task_rewards = {
        train_task_1: callback.task_rewards.get(train_task_1, []),
        train_task_2: callback.task_rewards.get(train_task_2, [])
    }
    
    if any(len(rewards) > 0 for rewards in phase2_task_rewards.values()):
        phase2_metrics = compute_multi_task_metrics(phase2_task_rewards, config['total_timesteps'])
        log_normalized_metrics(wandb.run, phase2_metrics['task_metrics'], 
                             phase2_metrics['final_normalized_score'], 
                             phase2_metrics['efficiency_score'], "phase2")
        print_normalized_summary(phase2_metrics['task_metrics'], 
                               phase2_metrics['final_normalized_score'], 
                               phase2_metrics['efficiency_score'], "Phase 2")
    
    # ============================================================================
    # CALCULATE COMPREHENSIVE METRICS
    # ============================================================================
    
    # Task 1 metrics
    task1_baseline_reward = np.mean(rewards1_after_task1)
    task1_final_reward = np.mean(rewards1_after_task2)
    task1_baseline_success = success1_after_task1
    task1_final_success = success1_after_task2
    
    # Task 2 metrics
    task2_baseline_reward = np.mean(rewards2_after_task1)
    task2_final_reward = np.mean(rewards2_after_task2)
    task2_baseline_success = success2_after_task1
    task2_final_success = success2_after_task2
    
    # Retention metrics (Task 1 forgetting)
    retention_reward_task1 = task1_final_reward / task1_baseline_reward if task1_baseline_reward > 0 else 0
    retention_success_task1 = task1_final_success / task1_baseline_success if task1_baseline_success > 0 else 0
    forgetting_reward_task1 = 1.0 - retention_reward_task1
    forgetting_success_task1 = 1.0 - retention_success_task1
    
    # Learning metrics (Task 2 improvement from training)
    learning_reward_task2 = task2_final_reward / task2_baseline_reward if task2_baseline_reward > 0 else 0
    learning_success_task2 = task2_final_success / task2_baseline_success if task2_baseline_success > 0 else 0
    
    # Task similarity metrics (Task 2 baseline vs Task 1 baseline)
    task_similarity_reward = task2_baseline_reward / task1_baseline_reward if task1_baseline_reward > 0 else 0
    task_similarity_success = task2_baseline_success / task1_baseline_success if task1_baseline_success > 0 else 0
    
    print(f"📈 COMPREHENSIVE ANALYSIS:")
    print(f"   • Task 1 Retention: {retention_reward_task1:.1%} (forgetting: {forgetting_reward_task1:.1%})")
    print(f"   • Task 2 Learning: {learning_reward_task2:.1%}")
    print(f"   • Task Similarity (Task2/Task1 baseline): {task_similarity_reward:.1%}")
    
    # Log comprehensive results
    if wandb.run:
        wandb.log({
            # Overall performance
            'testing/mean_reward': (task1_final_reward + task2_final_reward) / 2,
            'testing/task1_final_mean_reward': task1_final_reward,
            'testing/task2_final_mean_reward': task2_final_reward,
            'testing/task1_final_success_rate': task1_final_success,
            'testing/task2_final_success_rate': task2_final_success,
            'testing/overall_success_rate': (task1_final_success + task2_final_success) / 2,
            
            # Phase 1 results (after Task 1 training)
            'phase1/task1_reward': task1_baseline_reward,
            'phase1/task1_success': task1_baseline_success,
            'phase1/task2_reward': task2_baseline_reward,
            'phase1/task2_success': task2_baseline_success,
            
            # Phase 2 results (after Task 2 training)
            'phase2/task1_reward': task1_final_reward,
            'phase2/task1_success': task1_final_success,
            'phase2/task2_reward': task2_final_reward,
            'phase2/task2_success': task2_final_success,
            
            # Retention metrics (Task 1 forgetting)
            'retention/task1_baseline_reward': task1_baseline_reward,
            'retention/task1_baseline_success': task1_baseline_success,
            'retention/task1_final_reward': task1_final_reward,
            'retention/task1_final_success': task1_final_success,
            'retention/reward_retention_rate': retention_reward_task1,
            'retention/success_retention_rate': retention_success_task1,
            'retention/reward_forgetting_rate': forgetting_reward_task1,
            'retention/success_forgetting_rate': forgetting_success_task1,
            
            # Learning metrics (Task 2 improvement from training)
            'learning/task2_baseline_reward': task2_baseline_reward,
            'learning/task2_baseline_success': task2_baseline_success,
            'learning/task2_final_reward': task2_final_reward,
            'learning/task2_final_success': task2_final_success,
            'learning/reward_improvement_rate': learning_reward_task2,
            'learning/success_improvement_rate': learning_success_task2,
            
            # Task similarity metrics (Task 2 baseline vs Task 1 baseline)
            'task_similarity/reward_ratio': task_similarity_reward,
            'task_similarity/success_ratio': task_similarity_success,
            
            # Training metadata
            'training/sequential_training': True,
            'training/task1': train_task_1,
            'training/task2': train_task_2,
            'simplified_mode': True,
        })
    
    # Clean up
    env1.close()
    env2.close()
    eval_env1_after_task2.close()
    eval_env2_after_task2.close()
    
    return {
        'task1_baseline_rewards': rewards1_after_task1,
        'task1_final_rewards': rewards1_after_task2,
        'task2_baseline_rewards': rewards2_after_task1,
        'task2_final_rewards': rewards2_after_task2,
        'task1_baseline_success': task1_baseline_success,
        'task1_final_success': task1_final_success,
        'task2_baseline_success': task2_baseline_success,
        'task2_final_success': task2_final_success,
        'retention_reward_task1': retention_reward_task1,
        'retention_success_task1': retention_success_task1,
        'forgetting_reward_task1': forgetting_reward_task1,
        'forgetting_success_task1': forgetting_success_task1,
        'transfer_reward_task2': learning_reward_task2,
        'transfer_success_task2': learning_success_task2,
        'cross_transfer_reward': task_similarity_reward,
        'cross_transfer_success': task_similarity_success,
        'sequential_training': True,
        'simplified_mode': True
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
            project="topologies--simplified-double-task-training",
            entity="katko-it-universitetet-i-k-benhavn",
            config=wandb.config
        )
    
    # Get configuration from wandb
    config = wandb.config
    
    # Override hidden_size with capacity-matched value if available
    if effective_hidden_size is not None:
        config['hidden_size'] = effective_hidden_size
    
    # Log capacity matching results
    if wandb.run and target_capacity is not None:
        wandb.log({
            'capacity_matching/target_capacity': target_capacity,
            'capacity_matching/effective_hidden_size': effective_hidden_size,
            'capacity_matching/actual_hidden_size': config['hidden_size']
        })
    
    # Run training
    result = simplified_double_task_training(
        policy_class=DebugTopologyPolicy,
        topology_type=config['topology_type'],
        config=config,
        num_layers=config.get('num_layers', 2),
        hidden_size=config['hidden_size'],
        train_task_1=config['train_task_1'],
        train_task_2=config['train_task_2']
    )
    
    print("✅ Simplified double-task training completed!")
    return result

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Double-Task Training Sweep")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        print("🐛 Running in debug mode...")
        config = create_debug_config()
        result = simplified_double_task_training(
            policy_class=DebugTopologyPolicy,
            topology_type=config['topology_type'],
            config=config,
            num_layers=config['num_layers'],
            hidden_size=config['hidden_size'],
            train_task_1=config['train_task_1'],
            train_task_2=config['train_task_2']
        )
        print(f"Debug result: {result}")
    else:
        train_with_sweep() 