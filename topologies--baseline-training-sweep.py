#!/usr/bin/env python3
"""
Baseline Topology Training with Weights & Biases Sweep Support

This script is a sweep-enabled version of the baseline training script focused on:
- Single task training (no cross-task evaluation)
- Topology network verification and debugging
- Hyperparameter optimization for basic performance
- Simplified configuration for baseline experiments
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
        
        # Universal action space: 3 actions for all tasks
        self.action_space = gym.spaces.Discrete(3)
        
        # Universal observation space: 6 dimensions for all tasks
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,),  # Universal 6-dimensional observation space
            dtype=np.float32
        )
        
        # Task-specific action masks and mappings
        self.action_masks = {
            'CartPole-v1': [True, True, False],    # Actions 0,1 valid, 2 invalid
            'MountainCar-v0': [True, True, True],  # All 3 actions valid
            'Acrobot-v1': [True, True, False]      # Actions 0,1 valid, 2 invalid
        }
        
        # Action mappings for invalid actions (fallback to valid action)
        self.action_mappings = {
            'CartPole-v1': {2: 0},      # Map action 2 to action 0
            'MountainCar-v0': {},       # No mapping needed (all valid)
            'Acrobot-v1': {2: 0}        # Map action 2 to action 0
        }
        
        self.current_mask = self.action_masks.get(task_name, [True, True, True])
        self.current_mapping = self.action_mappings.get(task_name, {})
    
    def step(self, action):
        """
        Map universal action to task-specific action and step the environment.
        Pad observations to universal dimensions.
        """
        # Map universal action to task-specific action
        if action in self.current_mapping:
            mapped_action = self.current_mapping[action]
        else:
            mapped_action = action
        
        # Step the environment with mapped action
        obs, reward, done, truncated, info = self.env.step(mapped_action)
        
        # Pad observation to universal dimensions (6)
        obs = self._pad_observation(obs)
        
        # Add action masking info to info dict
        info['universal_action'] = action
        info['mapped_action'] = mapped_action
        info['action_mask'] = self.current_mask
        
        return obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to universal 6-dimensional space."""
        obs = np.array(obs, dtype=np.float32)
        
        if len(obs.shape) == 1:
            # Single observation
            if obs.shape[0] < 6:
                # Pad with zeros
                padded_obs = np.zeros(6, dtype=np.float32)
                padded_obs[:obs.shape[0]] = obs
                return padded_obs
            elif obs.shape[0] > 6:
                # Truncate
                return obs[:6]
            else:
                return obs
        else:
            # Vectorized observation
            batch_size = obs.shape[0]
            if obs.shape[1] < 6:
                # Pad with zeros
                padded_obs = np.zeros((batch_size, 6), dtype=np.float32)
                padded_obs[:, :obs.shape[1]] = obs
                return padded_obs
            elif obs.shape[1] > 6:
                # Truncate
                return obs[:, :6]
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
        total_params = actor_params + critic_params
        print(f"   • Actor topology parameters: {actor_params:,}")
        print(f"   • Critic topology parameters: {critic_params:,}")
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
    
    def __init__(self, verbose=0, wandb_run=None, log_freq=100):
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
    
    def _log_final_training_summary(self):
        """Log final training summary."""
        try:
            if len(self.training_metrics['episode_rewards']) > 0:
                # Log final statistics
                final_metrics = {
                    "final/mean_reward": np.mean(self.training_metrics['episode_rewards']),
                    "final/std_reward": np.std(self.training_metrics['episode_rewards']),
                    "final/max_reward": np.max(self.training_metrics['episode_rewards']),
                    "final/min_reward": np.min(self.training_metrics['episode_rewards']),
                    "final/total_episodes": len(self.training_metrics['episode_rewards']),
                    "final/total_steps": self.step_count,
                }
                
                self.wandb_run.log(final_metrics)
        except Exception as e:
            print(f"   ⚠️  Error logging final summary: {e}")

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
    Baseline training function for single task with sweep support.
    """
    print(f"\n{'='*80}")
    print(f"🎯 BASELINE TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"{'='*80}")
    
    # Create environment for single task
    train_env = DummyVecEnv([make_env(task)])
    
    # Use provided hidden_size or fall back to config
    actual_hidden_size = hidden_size if hidden_size is not None else config['hidden_size']
    
    # Create policy with debugging
    SpecificPolicyClass = lambda obs_space, action_space, lr_schedule, **kwargs: policy_class(
        obs_space, action_space, lr_schedule, 
        topology_type=topology_type, 
        hidden_size=actual_hidden_size,
        num_layers=num_layers,
        config=config,
        **kwargs
    )
    
    # Initialize model with PPO parameters from sweep
    ppo_params = config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/baseline_{topology_type}/",
        **ppo_params
    )
    
    # Get network size and parameter count for run name
    policy = model.policy
    actor_params = policy._get_topology_params(policy.actor_topology)
    critic_params = policy._get_topology_params(policy.critic_topology)
    total_params = actor_params + critic_params
    
    # Create descriptive run name
    run_name = f"{topology_type}_{num_layers}_{actual_hidden_size}_{total_params}_{task}"
    
    # Initialize wandb run for this topology
    try:
        wandb_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--baseline-training",
            name=run_name,
            config={
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": actual_hidden_size,
                "total_params": total_params,
                "actor_params": actor_params,
                "critic_params": critic_params,
                "total_timesteps": config['total_timesteps'],
                "n_eval_episodes": config['n_eval_episodes'],
                "task": task,
                "ppo_params": config['ppo_params'],
                "universal_input_dim": config['universal_input_dim'],
                "universal_output_dim": config['universal_output_dim'],
                "universal_action_dim": config['universal_action_dim'],
            },
            tags=[topology_type, f"layers_{num_layers}", f"size_{actual_hidden_size}", f"params_{total_params}", task, "baseline"],
            reinit=True
        )
        topology_wandb_enabled = True
    except Exception as e:
        print(f"   ⚠️  WandB logging disabled for {topology_type}: {e}")
        wandb_run = None
        topology_wandb_enabled = False
    
    print(f"📋 Configuration:")
    print(f"   • Task: {task}")
    print(f"   • Topology: {topology_type}")
    print(f"   • Hidden size: {actual_hidden_size}")
    print(f"   • Number of layers: {num_layers}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Training timesteps: {config['total_timesteps']:,}")
    print(f"   • WandB Run: {run_name}")
    
    # Setup callback with wandb
    callback = BaselineCallback(wandb_run=wandb_run if topology_wandb_enabled else None, log_freq=500)
    
    # Train the model with progress bar
    print(f"\n🎯 Training Phase:")
    print(f"   • Training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Test the model
    print(f"\n🧪 Testing Phase:")
    print(f"   • Evaluating on {task}...")
    mean_reward, std_reward = evaluate_model(model, train_env, n_eval_episodes=config['n_eval_episodes'])
    print(f"   • Results: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Log final results to wandb
    if topology_wandb_enabled and wandb_run is not None:
        try:
            wandb_run.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/training_time": training_time,
                "eval/topology_type": topology_type,
                "eval/num_layers": num_layers,
                "eval/timesteps_per_second": config['total_timesteps'] / training_time,
                "eval/performance_score": mean_reward / training_time,  # Reward per second
            })
            
            # Finish wandb run
            wandb_run.finish()
        except Exception as e:
            print(f"   ⚠️  Error logging to WandB: {e}")
    
    train_env.close()
    
    result = {
        'topology_type': topology_type,
        'num_layers': num_layers,
        'task': task,
        'network_size': actual_hidden_size,
        'total_params': total_params,
        'actor_params': actor_params,
        'critic_params': critic_params,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time
    }
    
    return result

def train_with_sweep():
    """Main training function for wandb sweep."""
    
    # Initialize wandb run
    wandb.init(
        entity="katko-it-universitetet-i-k-benhavn",
        project="topologies--baseline-training",
        config={
            # These will be overridden by sweep parameters
            'learning_rate': 3e-4,
            'n_steps': 1024,
            'batch_size': 32,
            'n_epochs': 5,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5,
            'hidden_size': 64,
            'num_layers': 1,
            'topology_type': 'small_world',
            'train_task': 'CartPole-v1',
            'total_timesteps': 50000,
            'n_eval_episodes': 10,
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
    
    print(f"🎯 Starting baseline sweep run with configuration:")
    try:
        print(f"   • Topology: {wandb.config.topology_type}")
        print(f"   • Hidden size: {wandb.config.hidden_size}")
        print(f"   • Layers: {wandb.config.num_layers}")
        print(f"   • Learning rate: {wandb.config.learning_rate}")
        print(f"   • Train task: {wandb.config.train_task}")
        print(f"   • Total timesteps: {wandb.config.total_timesteps}")
    except:
        print(f"   • Using default configuration (not in sweep mode)")
    
    # Create configuration from wandb.config
    config = create_baseline_config()
    
    # Update PPO parameters from sweep
    config['ppo_params'].update({
        'learning_rate': wandb.config.get('learning_rate', 3e-4),
        'n_steps': wandb.config.get('n_steps', 1024),
        'batch_size': wandb.config.get('batch_size', 32),
        'n_epochs': wandb.config.get('n_epochs', 5),
        'gamma': wandb.config.get('gamma', 0.99),
        'gae_lambda': wandb.config.get('gae_lambda', 0.95),
        'clip_range': wandb.config.get('clip_range', 0.2),
        'ent_coef': wandb.config.get('ent_coef', 0.01),
        'max_grad_norm': wandb.config.get('max_grad_norm', 0.5),
    })
    
    # Update training parameters from sweep
    config['total_timesteps'] = wandb.config.get('total_timesteps', 50000)
    config['n_eval_episodes'] = wandb.config.get('n_eval_episodes', 10)
    
    # Update topology-specific parameters from sweep
    topology_type = wandb.config.get('topology_type', 'small_world')
    if topology_type == 'small_world':
        config['topology_params']['small_world'].update({
            'k': wandb.config.get('small_world_k', 4),
            'p': wandb.config.get('small_world_p', 0.3),
        })
    elif topology_type == 'modular':
        config['topology_params']['modular'].update({
            'num_modules': wandb.config.get('modular_num_modules', 4),
            'inter_module_prob': wandb.config.get('modular_inter_module_prob', 0.2),
            'intra_module_prob': wandb.config.get('modular_intra_module_prob', 0.8),
        })
    elif topology_type == 'hybrid':
        config['topology_params']['hybrid'].update({
            'num_modules': wandb.config.get('hybrid_num_modules', 4),
            'k': wandb.config.get('hybrid_k', 4),
            'p': wandb.config.get('hybrid_p', 0.3),
            'inter_module_prob': wandb.config.get('hybrid_inter_module_prob', 0.2),
        })
    
    # Update network parameters from sweep
    config['network_params']['ffn'].update({
        'activation': wandb.config.get('activation', 'relu'),
        'dropout': wandb.config.get('dropout', 0.0),
    })
    
    # Get training task from sweep
    try:
        train_task = wandb.config.train_task
    except:
        train_task = 'CartPole-v1'  # Default fallback
    
    # Run baseline training
    try:
        result = baseline_training(
            BaselineTopologyPolicy,
            topology_type,
            config,
            num_layers=wandb.config.get('num_layers', 1),
            hidden_size=wandb.config.get('hidden_size', 64),
            task=train_task
        )
        
        # Log the result for the sweep metric
        mean_reward = result['mean_reward']
        
        # Log sweep metric
        wandb.log({
            'testing/mean_reward': mean_reward,
            'sweep/best_reward': mean_reward,
            'sweep/topology_type': topology_type,
            'sweep/hidden_size': wandb.config.get('hidden_size', 64),
            'sweep/num_layers': wandb.config.get('num_layers', 1),
            'sweep/learning_rate': wandb.config.get('learning_rate', 3e-4),
            'sweep/train_task': train_task,
            'sweep/total_params': result['total_params'],
            'sweep/training_time': result['training_time'],
        })
        
        print(f"✅ Baseline sweep run completed successfully!")
        print(f"   • Mean reward: {mean_reward:.2f}")
        
    except Exception as e:
        print(f"❌ Error in baseline sweep run: {e}")
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