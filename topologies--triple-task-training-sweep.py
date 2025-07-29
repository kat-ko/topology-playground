#!/usr/bin/env python3
"""
Triple-Task Topology Networks with Weights & Biases Sweep Support

This script is a modified version of the triple-task training script that can work with wandb sweeps
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
        
        # Pad observation to universal dimensions
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to universal 6-dimensional space."""
        if isinstance(obs, np.ndarray):
            obs = obs.flatten()
        else:
            obs = np.array(obs).flatten()
        
        # Pad with zeros to reach 6 dimensions
        if len(obs) < 6:
            padded_obs = np.zeros(6, dtype=np.float32)
            padded_obs[:len(obs)] = obs
            return padded_obs
        elif len(obs) > 6:
            # Truncate to 6 dimensions
            return obs[:6].astype(np.float32)
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
                input_size=6,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                num_layers=self.num_layers,
                k=k,
                p=p,
                activation=self.activation,
                dropout=self.dropout
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
    """Enhanced callback for tracking training progress with wandb integration."""
    
    def __init__(self, verbose=0, wandb_run=None, log_freq=100):
        super().__init__(verbose)
        self.wandb_run = wandb_run
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
    
    def _on_step(self) -> bool:
        """Log metrics on each step."""
        if self.num_timesteps % self.log_freq == 0 and wandb.run:
            wandb.log({
                'training/timesteps': self.num_timesteps,
                'training/episodes': len(self.episode_rewards),
                'training/mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'training/mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
            })
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout."""
        if wandb.run:
            wandb.log({
                'rollout/mean_reward': np.mean(self.episode_rewards[-self.n_envs:]) if self.episode_rewards else 0,
                'rollout/mean_length': np.mean(self.episode_lengths[-self.n_envs:]) if self.episode_lengths else 0
            })
    
    def _on_training_end(self) -> None:
        """Log final training summary."""
        if wandb.run:
            self._log_final_training_summary()

# ============================================================================
# CONFIGURATION AND UTILITY FUNCTIONS
# ============================================================================

def create_debug_config():
    """Create configuration for triple-task training with sweep support."""
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
            'total_timesteps': 500000,
            'n_eval_episodes': 15,
            'train_task_1': 'CartPole-v1',
            'train_task_2': 'Acrobot-v1',
            'train_task_3': 'MountainCar-v0'
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
    elif task_name == 'MountainCar-v0':
        # Success: reached the goal (reward >= -110)
        return np.mean([reward >= -110 for reward in rewards])
    else:
        # Default: above average performance
        mean_reward = np.mean(rewards)
        return np.mean([reward >= mean_reward for reward in rewards])

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def triple_task_training(policy_class, topology_type, config, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None, train_task_3=None):
    """
    Triple-task training function with sweep support.
    
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
    print(f"🎯 TRIPLE-TASK TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"   • Task 1: {train_task_1}")
    print(f"   • Task 2: {train_task_2}")
    print(f"   • Task 3: {train_task_3}")
    print(f"   • Hidden Size: {hidden_size}")
    print(f"   • Layers: {num_layers}")
    
    # Initialize wandb if not already done
    if wandb.run is None:
        wandb.init(
            project="topologies--hyperparameter-optimization",
            entity="katko-it-universitetet-i-k-benhavn",
            config=config,
            name=f"triple_task_{topology_type}_{train_task_1}_{train_task_2}_{train_task_3}"
        )
    
    # Create environments
    env1 = DummyVecEnv([make_env(train_task_1)])
    env2 = DummyVecEnv([make_env(train_task_2)])
    env3 = DummyVecEnv([make_env(train_task_3)])
    
    # Create model for task 1
    model1 = PPO(
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
        verbose=1,
        policy_kwargs={
            'topology_type': topology_type,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'config': config
        }
    )
    
    # Create model for task 2
    model2 = PPO(
        policy_class,
        env2,
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
    
    # Create model for task 3
    model3 = PPO(
        policy_class,
        env3,
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
    
    # Train model 1
    print(f"🚀 Training on {train_task_1}...")
    model1.learn(total_timesteps=config['total_timesteps'], callback=callback)
    
    # Evaluate model 1
    print(f"📊 Evaluating on {train_task_1}...")
    eval_env1 = make_env(train_task_1)()
    rewards1, lengths1, success1 = evaluate_model_enhanced(
        model1, eval_env1, train_task_1, config['n_eval_episodes']
    )
    
    # Train model 2
    print(f"🚀 Training on {train_task_2}...")
    model2.learn(total_timesteps=config['total_timesteps'], callback=callback)
    
    # Evaluate model 2
    print(f"📊 Evaluating on {train_task_2}...")
    eval_env2 = make_env(train_task_2)()
    rewards2, lengths2, success2 = evaluate_model_enhanced(
        model2, eval_env2, train_task_2, config['n_eval_episodes']
    )
    
    # Train model 3
    print(f"🚀 Training on {train_task_3}...")
    model3.learn(total_timesteps=config['total_timesteps'], callback=callback)
    
    # Evaluate model 3
    print(f"📊 Evaluating on {train_task_3}...")
    eval_env3 = make_env(train_task_3)()
    rewards3, lengths3, success3 = evaluate_model_enhanced(
        model3, eval_env3, train_task_3, config['n_eval_episodes']
    )
    
    # Log combined results
    if wandb.run:
        wandb.log({
            'testing/mean_reward': (np.mean(rewards1) + np.mean(rewards2) + np.mean(rewards3)) / 3,
            'testing/task1_mean_reward': np.mean(rewards1),
            'testing/task2_mean_reward': np.mean(rewards2),
            'testing/task3_mean_reward': np.mean(rewards3),
            'testing/task1_success_rate': success1,
            'testing/task2_success_rate': success2,
            'testing/task3_success_rate': success3,
            'testing/overall_success_rate': (success1 + success2 + success3) / 3
        })
    
    # Clean up
    env1.close()
    env2.close()
    env3.close()
    eval_env1.close()
    eval_env2.close()
    eval_env3.close()
    
    return {
        'task1_rewards': rewards1,
        'task2_rewards': rewards2,
        'task3_rewards': rewards3,
        'task1_success': success1,
        'task2_success': success2,
        'task3_success': success3
    }

# ============================================================================
# SWEEP TRAINING FUNCTION
# ============================================================================

def train_with_sweep():
    """Main function for sweep training."""
    # Get configuration from wandb
    config = create_debug_config()
    
    # Extract parameters
    topology_type = config['topology_type']
    hidden_size = config['hidden_size']
    num_layers = config['num_layers']
    train_task_1 = config['train_task_1']
    train_task_2 = config['train_task_2']
    train_task_3 = config['train_task_3']
    
    # Run triple-task training
    results = triple_task_training(
        policy_class=DebugTopologyPolicy,
        topology_type=topology_type,
        config=config,
        num_layers=num_layers,
        hidden_size=hidden_size,
        train_task_1=train_task_1,
        train_task_2=train_task_2,
        train_task_3=train_task_3
    )
    
    print("✅ Triple-task training completed!")
    print(f"   • Task 1 ({train_task_1}) success rate: {results['task1_success']:.3f}")
    print(f"   • Task 2 ({train_task_2}) success rate: {results['task2_success']:.3f}")
    print(f"   • Task 3 ({train_task_3}) success rate: {results['task3_success']:.3f}")
    print(f"   • Overall success rate: {(results['task1_success'] + results['task2_success'] + results['task3_success']) / 3:.3f}")

if __name__ == "__main__":
    train_with_sweep()