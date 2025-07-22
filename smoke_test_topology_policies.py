#!/usr/bin/env python3
"""
Smoke Test: Topology Networks as Actor and Critic Functions
Tests cross-task transfer with topology networks as the actual policy and value functions.
Option A: Separate topology networks for actor and critic.
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
# TOPOLOGY POLICY CLASSES (Separate Actor and Critic Topologies)
# ============================================================================

class TopologyPolicy_Padding(ActorCriticPolicy):
    """
    Topology Policy with Padding Strategy.
    Option A: Separate topology networks for actor and critic.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_config()
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        
        # Universal dimensions from config
        self.universal_input_dim = config['universal_input_dim']
        self.universal_output_dim = config['universal_output_dim']
        self.universal_action_dim = config['universal_action_dim']
        self.hidden_size = hidden_size
        self.topology_type = topology_type
        self.config = config
        
        print(f"[Topology Policy - {topology_type}] Task: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"[Topology Policy - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"[Topology Policy - {topology_type}] Hidden Size: {self.hidden_size}")
        
        # Create separate topology networks for actor and critic
        self.actor_topology = self._create_topology_network('actor')
        self.critic_topology = self._create_topology_network('critic')
        
        # Debug: Check topology network types
        print(f"[Topology Policy - {topology_type}] Actor Topology Type: {type(self.actor_topology)}")
        print(f"[Topology Policy - {topology_type}] Critic Topology Type: {type(self.critic_topology)}")
        
        # Get weight statistics
        actor_params = self._get_topology_params(self.actor_topology)
        critic_params = self._get_topology_params(self.critic_topology)
        total_params = actor_params + critic_params
        print(f"[Topology Policy - {topology_type}] Actor Topology Parameters: {actor_params:,}")
        print(f"[Topology Policy - {topology_type}] Critic Topology Parameters: {critic_params:,}")
        print(f"[Topology Policy - {topology_type}] Total Parameters: {total_params:,}")
    
    def _create_topology_network(self, network_type):
        """Create topology network for actor or critic."""
        # Total nodes: input + hidden + output
        total_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=2,
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,
                k=4,
                p=0.1,
                num_layers=2,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=total_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=2,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=total_size,
                num_modules=4,
                k=4,
                p=0.1,
                inter_module_prob=0.1,
                num_layers=2,
                inter_layer_prob=0.1,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate graph
        graph = topology.generate()
        
        # Define input/output nodes
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        return FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
    
    def _get_topology_params(self, topology_network):
        """Get number of parameters in topology network."""
        total_params = 0
        try:
            for param in topology_network.parameters():
                total_params += param.numel()
        except:
            # If parameters() doesn't work, try to get parameters differently
            try:
                for module in topology_network.modules():
                    if hasattr(module, 'weight'):
                        total_params += module.weight.numel()
                    if hasattr(module, 'bias') and module.bias is not None:
                        total_params += module.bias.numel()
            except:
                # If all else fails, return 0
                total_params = 0
        return total_params
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to universal dimensions."""
        batch_size = x.shape[0]
        if x.shape[1] < self.universal_input_dim:
            # Pad with zeros
            padded = torch.zeros(batch_size, self.universal_input_dim, dtype=x.dtype, device=x.device)
            padded[:, :x.shape[1]] = x
            return padded
        elif x.shape[1] > self.universal_input_dim:
            # Truncate
            return x[:, :self.universal_input_dim]
        else:
            return x
    
    def forward_actor(self, obs):
        """Forward pass for actor - topology network directly outputs action logits."""
        features = self.extract_features(obs)
        
        # Pad input to universal dimensions
        universal_input = self._pad_input(features)
        
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
        
        # Pad input to universal dimensions
        universal_input = self._pad_input(features)
        
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

class TopologyPolicy_Masking(ActorCriticPolicy):
    """
    Topology Policy with Masking Strategy.
    Option A: Separate topology networks for actor and critic.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_config()
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        
        # Universal dimensions from config
        self.universal_input_dim = config['universal_input_dim']
        self.universal_output_dim = config['universal_output_dim']
        self.universal_action_dim = config['universal_action_dim']
        self.hidden_size = hidden_size
        self.topology_type = topology_type
        self.config = config
        
        print(f"[Topology Policy - {topology_type}] Task: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"[Topology Policy - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"[Topology Policy - {topology_type}] Hidden Size: {self.hidden_size}")
        
        # Create separate topology networks for actor and critic
        self.actor_topology = self._create_topology_network('actor')
        self.critic_topology = self._create_topology_network('critic')
        
        # Get weight statistics
        actor_params = self._get_topology_params(self.actor_topology)
        critic_params = self._get_topology_params(self.critic_topology)
        total_params = actor_params + critic_params
        print(f"[Topology Policy - {topology_type}] Actor Topology Parameters: {actor_params:,}")
        print(f"[Topology Policy - {topology_type}] Critic Topology Parameters: {critic_params:,}")
        print(f"[Topology Policy - {topology_type}] Total Parameters: {total_params:,}")
    
    def _create_topology_network(self, network_type):
        """Create topology network for actor or critic."""
        # Total nodes: input + hidden + output
        total_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=1,
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,
                k=4,
                p=0.1,
                num_layers=1,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=total_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=1,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=total_size,
                num_modules=4,
                k=4,
                p=0.1,
                inter_module_prob=0.1,
                num_layers=1,
                inter_layer_prob=0.1,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate graph
        graph = topology.generate()
        
        # Define input/output nodes
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        return FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
    
    def _get_topology_params(self, topology_network):
        """Get number of parameters in topology network."""
        total_params = 0
        for param in topology_network.parameters():
            total_params += param.numel()
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

class CrossTaskCallback(BaseCallback):
    """Callback to track training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Track episode rewards and lengths (simplified for DummyVecEnv)
        return True

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def create_config():
    """Create configuration for smoke test."""
    config = {
        # ============================================================================
        # EXPERIMENT PARAMETERS
        # ============================================================================
        'tasks': ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'],
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
        'experiment_types': ['same_size', 'match_small_world'],
        'total_timesteps': 10000,  # Reduced for smoke test
        'n_eval_episodes': 5,     # Reduced for smoke test
        
        # ============================================================================
        # CAPACITY MATCHING PARAMETERS
        # ============================================================================
        'network_sizes': [64],  # Reduced for smoke test
        'network_types': ['ffn'],
        'num_layers': [1],
        'seeds': [42],
        'node_selection_strategies': ['random'],
        'num_io_nodes': 2,
        'use_capacity_mapping': False,
        'min_search_size': 10,
        'max_search_size': 200,
        
        # ============================================================================
        # UNIVERSAL TOPOLOGY PARAMETERS
        # ============================================================================
        'universal_input_dim': 6,
        'universal_output_dim': 3,
        'universal_action_dim': 3,
        'hidden_size': 32,  # Reduced for smoke test
        'adapter_type': 'linear',
        'adapter_hidden_dim': 8,
        'freeze_adapters': False,
        
        # ============================================================================
        # TOPOLOGY-SPECIFIC PARAMETERS
        # ============================================================================
        'small_world_params': {
            'k': 4,
            'p': 0.3,
            'inter_layer_prob': 0.1
        },
        'modular_params': {
            'num_modules': 4,
            'inter_module_prob': 0.1,
            'intra_module_prob': 0.8,
            'inter_layer_prob': 0.1
        },
        'hybrid_params': {
            'num_modules': 4,
            'k': 4,
            'p': 0.3,
            'inter_module_prob': 0.1,
            'inter_layer_prob': 0.1
        },
        'fully_connected_params': {
            'inter_layer_prob': 1.0,
            'intra_layer_prob': 1.0
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
        # PPO TRAINING PARAMETERS
        # ============================================================================
        'ppo_params': {
            'learning_rate': 3e-4,
            'n_steps': 1024,  # Reduced for smoke test
            'batch_size': 32,  # Reduced for smoke test
            'n_epochs': 5,     # Reduced for smoke test
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5
        }
    }
    return config

def train_and_test_policy(policy_class, policy_name, train_task, test_tasks, topology_type, 
                         config, experiment_type='same_size', total_timesteps=5000):
    """
    Train a policy on one task and test on multiple tasks.
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING: {policy_name}")
    print(f"{'='*80}")
    print(f"📋 Configuration:")
    print(f"   • Train Task: {train_task}")
    print(f"   • Topology Type: {topology_type}")
    print(f"   • Experiment Type: {experiment_type}")
    print(f"   • Policy Class: {policy_class.__name__}")
    print(f"   • Total Timesteps: {total_timesteps:,}")
    print(f"   • Test Tasks: {', '.join(test_tasks)}")
    print(f"{'='*80}")
    
    # Initialize capacity matching calculator
    calculator = ParameterBudgetCalculator(config)
    
    # Get the network size for capacity matching
    if experiment_type.startswith('match_'):
        network_size = config['network_sizes'][0]
        reference_topology = experiment_type[len('match_'):]
        target_capacity = calculator._get_reference_capacity(reference_topology, network_size, 'ffn', 1)
        matching_size = calculator.calculate_matching_size(topology_type, target_capacity, 'ffn', 1)
        actual_size = matching_size
        print(f"🔧 Capacity Matching:")
        print(f"   • Reference Topology: {reference_topology}")
        print(f"   • Target Capacity: {target_capacity:,} parameters")
        print(f"   • Matching Size: {matching_size} nodes")
    else:
        actual_size = config['network_sizes'][0]
        print(f"🔧 Same Size: {actual_size} nodes")
    
    # Create training environment
    train_env = DummyVecEnv([make_env(train_task)])
    
    # Create policy with capacity matching
    SpecificPolicyClass = lambda obs_space, action_space, lr_schedule, **kwargs: policy_class(
        obs_space, action_space, lr_schedule, 
        topology_type=topology_type, 
        hidden_size=actual_size,
        config=config,
        **kwargs
    )
    
    # Initialize model with centralized PPO parameters
    ppo_params = config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/{train_task}_smoke_test/",
        **ppo_params
    )
    
    # Setup callback
    callback = CrossTaskCallback()
    
    # Train the model with progress bar
    print(f"🎯 Training Phase:")
    print(f"   • Training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Test on all tasks
    print(f"\n🧪 Testing Phase:")
    print(f"   • Number of test tasks: {len(test_tasks)}")
    print(f"   • Evaluation episodes per task: {config['n_eval_episodes']}")
    
    results = {}
    for i, test_task in enumerate(test_tasks, 1):
        print(f"\n📊 Test {i}/{len(test_tasks)}: {test_task}")
        print(f"   • Creating environment...")
        
        # Create new environment for test task
        test_env = DummyVecEnv([make_env(test_task)])
        
        print(f"   • Reusing trained model...")
        # REUSE THE EXACT SAME MODEL - No new policy creation!
        # The padding/masking strategies handle dimension differences automatically
        
        print(f"   • Evaluating model...")
        # Evaluate the model directly (no weight transfer needed)
        mean_reward, std_reward = evaluate_model(model, test_env, n_eval_episodes=config['n_eval_episodes'])
        
        print(f"   • Results: {mean_reward:.2f} ± {std_reward:.2f}")
        
        results[test_task] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'train_task': train_task,
            'policy_name': policy_name,
            'topology_type': topology_type,
            'experiment_type': experiment_type,
            'capacity_matched_size': actual_size
        }
        
        test_env.close()
    
    train_env.close()
    
    return results, callback

def evaluate_model(model, env, n_eval_episodes=5):
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
            if step_count > 1000:
                print(f"      ⚠️  Episode {episode} exceeded 1000 steps, terminating")
                break
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    return mean_reward, std_reward

def main():
    """Main function to run smoke test."""
    print("🚀 Smoke Test: Topology Networks as Actor and Critic Functions")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create configuration
    config = create_config()
    
    # Define tasks (smoke test: only train on CartPole, test on others)
    train_task = 'CartPole-v1'
    test_tasks = ['MountainCar-v0', 'Acrobot-v1']
    
    # Define topologies
    topologies = config['topologies']
    
    # Define capacity matching experiment types
    experiment_types = config['experiment_types']
    
    print(f"\n📋 Smoke Test Configuration:")
    print(f"   • Train Task: {train_task}")
    print(f"   • Test Tasks: {', '.join(test_tasks)}")
    print(f"   • Topologies: {', '.join(topologies)}")
    print(f"   • Experiment Types: {', '.join(experiment_types)}")
    print(f"   • Total Experiments: {len(experiment_types) * len(topologies) * 2} (2 strategies per config)")
    print(f"   • Training Timesteps: {config['total_timesteps']:,}")
    print(f"   • Evaluation Episodes: {config['n_eval_episodes']} per task")
    
    # Store all results
    all_results = []
    
    # Run experiments for each capacity matching type
    total_experiments = len(experiment_types) * len(topologies) * 2
    experiment_count = 0
    
    for experiment_type in experiment_types:
        print(f"\n{'='*80}")
        print(f"🎯 EXPERIMENT TYPE: {experiment_type.upper()}")
        print(f"{'='*80}")
        
        # Run experiments for each topology
        for topology in topologies:
            print(f"\n🔬 Testing {topology} topology")
            print(f"{'─'*60}")
            
            experiment_count += 2  # Padding and Masking strategies
            
            print(f"\n📚 Training on {train_task} → Testing on {', '.join(test_tasks)}")
            print(f"   Progress: {experiment_count}/{total_experiments} experiments completed")
            
            # Run Padding strategy
            results_padding, callback_padding = train_and_test_policy(
                TopologyPolicy_Padding,
                f"Padding_{topology}",
                train_task,
                test_tasks,
                topology,
                config,
                experiment_type=experiment_type,
                total_timesteps=config['total_timesteps']
            )
            
            # Run Masking strategy
            results_masking, callback_masking = train_and_test_policy(
                TopologyPolicy_Masking,
                f"Masking_{topology}",
                train_task,
                test_tasks,
                topology,
                config,
                experiment_type=experiment_type,
                total_timesteps=config['total_timesteps']
            )
            
            # Combine results
            results = {**results_padding, **results_masking}
            
            # Store results
            for test_task, result in results.items():
                all_results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/smoke_test_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(f"{results_dir}/smoke_test_results.csv", index=False)
    
    # Save configuration
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🎉 SMOKE TEST COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Results Summary:")
    print(f"   • Total Experiments: {len(all_results)}")
    print(f"   • Results saved to: {results_dir}")
    print(f"   • CSV file: smoke_test_results.csv")
    print(f"   • Config file: config.json")
    
    # Print some key results
    if not df.empty:
        print(f"\n📈 Key Results:")
        print(f"   • Best transfer performance:")
        best_result = df.loc[df['mean_reward'].idxmax()]
        print(f"     {best_result['policy_name']} on {best_result.name}: {best_result['mean_reward']:.2f}")
        
        print(f"   • Average performance by topology:")
        for topology in topologies:
            topology_results = df[df['topology_type'] == topology]
            if not topology_results.empty:
                avg_reward = topology_results['mean_reward'].mean()
                print(f"     {topology}: {avg_reward:.2f}")
    
    print(f"\n✅ Smoke test completed successfully!")

if __name__ == "__main__":
    main() 