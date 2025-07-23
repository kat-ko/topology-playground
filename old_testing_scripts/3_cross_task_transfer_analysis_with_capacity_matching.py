#!/usr/bin/env python3
"""
Cross-Task Transfer Analysis with Capacity Matching
Analyzes knowledge transfer between different RL tasks using universal topology networks.
Enhanced with capacity matching for fair comparison across topologies.
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
        # Handle different observation formats
        if isinstance(obs, (list, tuple)):
            obs = np.array(obs, dtype=np.float32)
        elif not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # Ensure it's a 1D array
        if len(obs.shape) == 0:
            obs = obs.reshape(1)
        
        # Pad or truncate to 6 dimensions
        if obs.shape[0] < 6:
            # Pad with zeros
            padded_obs = np.zeros(6, dtype=np.float32)
            padded_obs[:obs.shape[0]] = obs.flatten()
            return padded_obs
        elif obs.shape[0] > 6:
            # Truncate
            return obs.flatten()[:6]
        else:
            return obs.flatten()
    
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
        """Get the current action mask for this task."""
        return self.current_mask

# ============================================================================
# UNIVERSAL ACTION POLICY CLASSES
# ============================================================================

class VersionB_PaddingPolicy_2Layers_Universal(ActorCriticPolicy):
    """Version B: Universal topology with padding - established pattern from previous files."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_capacity_matching_config()
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        
        # Universal dimensions from config
        self.universal_input_dim = config['universal_input_dim']
        self.universal_output_dim = config['universal_output_dim']
        self.universal_action_dim = config['universal_action_dim']
        self.hidden_size = hidden_size
        self.topology_type = topology_type
        self.config = config
        
        print(f"[Version B - {topology_type}] Task: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"[Version B - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"[Version B - {topology_type}] Hidden Size: {self.hidden_size} | Total Network Size: {self.universal_input_dim + self.hidden_size + self.universal_output_dim}")
        
        # Create topology network
        self.topology_network = self._create_topology_network()
        
        # Get weight statistics after network creation
        weight_stats = self.get_weight_stats()
        print(f"[Version B - {topology_type}] Topology Parameters: {weight_stats['total_params']:,}")
        
        # Actor head (policy) - UNIVERSAL ACTION SPACE
        head_params = config['head_params']
        self.actor_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, head_params['actor_hidden_dim']),
            nn.ReLU(),
            nn.Linear(head_params['actor_hidden_dim'], self.universal_action_dim)  # Always 3 outputs
        )
        
        # Critic head (value) - established pattern
        self.critic_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, head_params['critic_hidden_dim']),
            nn.ReLU(),
            nn.Linear(head_params['critic_hidden_dim'], 1)
        )
    
    def _create_topology_network(self):
        """Create the topology network with universal dimensions - established pattern."""
        # Total nodes: input + hidden + output
        total_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create topology based on type - established pattern
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
        
        # Define input/output nodes - established pattern
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network - established pattern
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        return FeedForwardNetwork(
            topology=graph,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            network_params=network_params
        )
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to universal dimensions."""
        if x.shape[-1] < self.universal_input_dim:
            padding = torch.zeros(x.shape[0], self.universal_input_dim - x.shape[-1], device=x.device)
            return torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > self.universal_input_dim:
            return x[:, :self.universal_input_dim]
        else:
            return x
    
    def _pad_output(self, x: torch.Tensor) -> torch.Tensor:
        """Pad output to task-specific dimensions."""
        if x.shape[-1] < self.task_output_dim:
            padding = torch.zeros(x.shape[0], self.task_output_dim - x.shape[-1], device=x.device)
            return torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > self.task_output_dim:
            return x[:, :self.task_output_dim]
        else:
            return x
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create input mask for masking strategy."""
        mask = torch.zeros(self.universal_input_dim, device=x.device)
        mask[:x.shape[-1]] = 1.0
        return mask
    
    def _create_output_mask(self) -> torch.Tensor:
        """Create output mask for masking strategy."""
        mask = torch.zeros(self.universal_output_dim)
        mask[:self.task_output_dim] = 1.0
        return mask
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply input masking."""
        return x * mask.unsqueeze(0)
    
    def _apply_output_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply output masking."""
        return x * mask.unsqueeze(0)
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to universal dimensions - established pattern."""
        if x.shape[-1] < self.universal_input_dim:
            padding = torch.zeros(x.shape[0], self.universal_input_dim - x.shape[-1], device=x.device)
            return torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > self.universal_input_dim:
            return x[:, :self.universal_input_dim]
        else:
            return x
    
    def _truncate_output(self, x: torch.Tensor) -> torch.Tensor:
        """Truncate output to task-specific dimensions - established pattern."""
        if x.shape[-1] > self.task_output_dim:
            return x[:, :self.task_output_dim]
        else:
            return x
    
    def forward_actor(self, obs):
        """Forward pass for actor with universal action space."""
        features = self.extract_features(obs)
        
        # Pad input
        universal_input = self._pad_input(features)
        
        # Convert to dictionary format for topology network
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through topology network
        topology_output = self.topology_network.forward(input_dict)
        
        # Convert topology output to tensor
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
        universal_output = torch.stack(output_values, dim=1)
        
        # Forward through actor head (universal action space)
        return self.actor_head(universal_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic with pure universal architecture."""
        features = self.extract_features(obs)
        
        # Pad input
        universal_input = self._pad_input(features)
        
        # Convert to dictionary format for topology network
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through topology network
        topology_output = self.topology_network.forward(input_dict)
        
        # Convert topology output to tensor
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
        universal_output = torch.stack(output_values, dim=1)
        
        # Forward through critic head (pure universal - no truncation)
        return self.critic_head(universal_output)
    
    def get_weight_stats(self):
        """Get statistics about topology network weights."""
        total_params = 0
        for param in self.topology_network.parameters():
            total_params += param.numel()
        return {'total_params': total_params}

class VersionD_MaskingPolicy_2Layers_Universal(ActorCriticPolicy):
    """Version D: Universal topology with masking - established pattern from previous files."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_capacity_matching_config()
        
        self.task_input_dim = observation_space.shape[0]
        self.universal_input_dim = config['universal_input_dim']
        self.universal_output_dim = config['universal_output_dim']
        self.universal_action_dim = config['universal_action_dim']
        self.hidden_size = hidden_size
        self.topology_type = topology_type
        self.config = config
        
        print(f"[Version D - {topology_type}] Task: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"[Version D - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"[Version D - {topology_type}] Hidden Size: {self.hidden_size} | Total Network Size: {self.universal_input_dim + self.hidden_size + self.universal_output_dim}")
        
        self._create_topology_network()
        
        # Get weight statistics after network creation
        weight_stats = self.get_weight_stats()
        print(f"[Version D - {topology_type}] Topology Parameters: {weight_stats['total_params']:,}")
        
        # Actor head (policy) - UNIVERSAL ACTION SPACE
        head_params = config['head_params']
        self.actor_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, head_params['actor_hidden_dim']),
            nn.ReLU(),
            nn.Linear(head_params['actor_hidden_dim'], self.universal_action_dim)  # Always 3 outputs
        )
        
        # Critic head (value) - established pattern
        self.critic_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, head_params['critic_hidden_dim']),
            nn.ReLU(),
            nn.Linear(head_params['critic_hidden_dim'], 1)
        )
    
    def _create_topology_network(self):
        """Create the topology network - established pattern."""
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create topology based on type - established pattern
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=layer_size,
                num_layers=2,
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=layer_size,
                k=4,
                p=0.1,
                num_layers=2,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=layer_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=2,
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=layer_size,
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
        
        graphs = topology.generate(2)
        
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create a mask for input nodes based on actual input dimensions - established pattern."""
        batch_size = x.shape[0]
        mask = torch.zeros(batch_size, self.universal_input_dim, device=x.device)
        
        # Set mask to 1 for actual input dimensions, 0 for padding
        actual_dim = min(x.shape[1], self.universal_input_dim)
        mask[:, :actual_dim] = 1.0
        
        return mask
    
    def _create_output_mask(self) -> torch.Tensor:
        """Create a mask for output nodes based on task output dimensions - established pattern."""
        # For now, we'll use a simple approach - mask out unused output dimensions
        mask = torch.ones(self.universal_output_dim)
        if self.task_output_dim < self.universal_output_dim:
            mask[self.task_output_dim:] = 0.0
        return mask
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masking to input, setting masked values to 0 - established pattern."""
        # Pad or truncate input to universal dimension
        batch_size = x.shape[0]
        if x.shape[1] < self.universal_input_dim:
            padding = torch.zeros(batch_size, self.universal_input_dim - x.shape[1], device=x.device)
            padded_input = torch.cat([x, padding], dim=1)
        else:
            padded_input = x[:, :self.universal_input_dim]
        
        # Apply mask
        return padded_input * mask
    
    def _apply_output_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masking to output, keeping only relevant dimensions - established pattern."""
        # Apply mask to output
        masked_output = x * mask.unsqueeze(0).expand(x.shape[0], -1)
        
        # Return only the task-relevant dimensions
        return masked_output[:, :self.task_output_dim]
    
    def forward_actor(self, obs):
        """Forward pass for actor - established pattern."""
        features = self.extract_features(obs)
        
        # Create masks
        input_mask = self._create_input_mask(features)
        output_mask = self._create_output_mask().to(features.device)
        
        # Apply input masking
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
        output_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        layer1_tensor = torch.stack(output_values, dim=1)
        
        # Layer 2
        input_dict = {i: layer1_tensor[:, i] for i in range(self.universal_output_dim)}
        layer2_output = self.layer2_network.forward(input_dict)
        
        # Convert layer2 output to tensor
        output_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        layer2_tensor = torch.stack(output_values, dim=1)
        
        # Forward through actor head (pure universal - no masking)
        return self.actor_head(layer2_tensor)
    
    def forward_critic(self, obs):
        """Forward pass for critic - established pattern."""
        features = self.extract_features(obs)
        
        # Create masks
        input_mask = self._create_input_mask(features)
        output_mask = self._create_output_mask().to(features.device)
        
        # Apply input masking
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
        output_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        layer1_tensor = torch.stack(output_values, dim=1)
        
        # Layer 2
        input_dict = {i: layer1_tensor[:, i] for i in range(self.universal_output_dim)}
        layer2_output = self.layer2_network.forward(input_dict)
        
        # Convert layer2 output to tensor
        output_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        layer2_tensor = torch.stack(output_values, dim=1)
        
        # Forward through critic head (pure universal - no masking)
        return self.critic_head(layer2_tensor)
    
    def get_weight_stats(self):
        """Get statistics about topology network weights."""
        total_params = 0
        for param in self.layer1_network.parameters():
            total_params += param.numel()
        for param in self.layer2_network.parameters():
            total_params += param.numel()
        return {'total_params': total_params}

class CrossTaskCallback(BaseCallback):
    """Callback to track training metrics across tasks."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards and lengths
        if len(self.training_env.buf_rews) > 0:
            self.current_episode_reward += self.training_env.buf_rews[0]
            self.current_episode_length += 1
            
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        # Track losses if available
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            if 'train/value_loss' in self.model.logger.name_to_value:
                self.value_losses.append(self.model.logger.name_to_value['train/value_loss'])
            if 'train/policy_loss' in self.model.logger.name_to_value:
                self.policy_losses.append(self.model.logger.name_to_value['train/policy_loss'])
            if 'train/entropy_loss' in self.model.logger.name_to_value:
                self.entropy_losses.append(self.model.logger.name_to_value['train/entropy_loss'])
        
        return True

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def create_capacity_matching_config():
    """Create configuration for capacity matching experiments."""
    config = {
        # ============================================================================
        # EXPERIMENT PARAMETERS
        # ============================================================================
        'tasks': ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'],
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
        'experiment_types': ['same_size', 'match_small_world'],
        'total_timesteps': 10000,
        'n_eval_episodes': 10,
        
        # ============================================================================
        # CAPACITY MATCHING PARAMETERS
        # ============================================================================
        'network_sizes': [64],  # Reduced for faster experiments
        'network_types': ['ffn'],
        'num_layers': [1],
        'seeds': [42],
        'node_selection_strategies': ['random'],
        'num_io_nodes': 2,
        'use_capacity_mapping': False,  # Use incremental adjustment
        'min_search_size': 10,
        'max_search_size': 2000,
        
        # ============================================================================
        # UNIVERSAL TOPOLOGY PARAMETERS
        # ============================================================================
        'universal_input_dim': 6,   # Maximum input dimension across all tasks
        'universal_output_dim': 3,  # Maximum output dimension across all tasks
        'universal_action_dim': 3,  # Universal action space
        'hidden_size': 64,
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
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'max_grad_norm': 0.5
        },
        
        # ============================================================================
        # ACTOR/CRITIC HEAD PARAMETERS
        # ============================================================================
        'head_params': {
            'actor_hidden_dim': 32,
            'critic_hidden_dim': 32,
            'activation': 'relu'
        }
    }
    return config

def train_and_test_policy_with_capacity_matching(policy_class, policy_name, train_task, test_tasks, topology_type, 
                                                capacity_matching_config, experiment_type='same_size', total_timesteps=10000, strategy='padding'):
    """
    Train a policy on one task and test on multiple tasks with capacity matching.
    
    Args:
        policy_class: The policy class to use
        policy_name: Name of the policy for logging
        train_task: Task to train on
        test_tasks: List of tasks to test on
        topology_type: Type of topology to use
        capacity_matching_config: Configuration for capacity matching
        experiment_type: Type of capacity matching experiment
        total_timesteps: Number of training timesteps
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
    calculator = ParameterBudgetCalculator(capacity_matching_config)
    
    # Get the network size for capacity matching
    if experiment_type.startswith('match_'):
        # For capacity matching, get the matching size
        network_size = capacity_matching_config['network_sizes'][0]
        matching_size = calculator.get_matching_size(experiment_type, topology_type, network_size)
        print(f"Capacity matching: {network_size} -> {matching_size} nodes")
        actual_size = matching_size
    else:
        # For same_size, use the original size
        actual_size = capacity_matching_config['network_sizes'][0]
    
    # Create training environment
    train_env = DummyVecEnv([make_env(train_task)])
    
    # Create policy with capacity matching
    # Use the specified policy class
    SpecificPolicyClass = lambda obs_space, action_space, lr_schedule, **kwargs: policy_class(
        obs_space, action_space, lr_schedule, 
        topology_type=topology_type, 
        hidden_size=actual_size,  # Use the capacity-matched size
        config=capacity_matching_config,  # Pass centralized config
        **kwargs
    )
    
    # Initialize model with centralized PPO parameters
    ppo_params = capacity_matching_config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/{train_task}_versionD/",
        **ppo_params
    )
    
    # Setup callback
    callback = CrossTaskCallback()
    
    # Train the model with progress bar
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Save trained weights for reuse
    trained_weights = {}
    for name, param in model.policy.named_parameters():
        trained_weights[name] = param.data.clone()
    
    # Test on all tasks
    print(f"\n🧪 Testing Phase:")
    print(f"   • Number of test tasks: {len(test_tasks)}")
    print(f"   • Evaluation episodes per task: {capacity_matching_config['n_eval_episodes']}")
    print(f"   • Weight transfer: Enabled")
    
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
        mean_reward, std_reward = evaluate_model(model, test_env, n_eval_episodes=capacity_matching_config['n_eval_episodes'])
        
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

def train_and_test_policy_true_reuse_with_capacity_matching(policy_class, policy_name, train_task, test_tasks, topology_type,
                                                           capacity_matching_config, experiment_type='same_size', total_timesteps=10000):
    """
    Train a policy on one task and test on multiple tasks with TRUE network reuse and capacity matching.
    This version reuses the exact same network weights across all tasks.
    
    Args:
        policy_class: The policy class to use
        policy_name: Name of the policy for logging
        train_task: Task to train on
        test_tasks: List of tasks to test on
        topology_type: Type of topology to use
        capacity_matching_config: Configuration for capacity matching
        experiment_type: Type of capacity matching experiment
        total_timesteps: Number of training timesteps
    """
    print(f"\n{'='*60}")
    print(f"TRUE REUSE: Training {policy_name} on {train_task} with {topology_type} topology")
    print(f"Capacity matching: {experiment_type}")
    print(f"{'='*60}")
    
    # Initialize capacity matching calculator
    calculator = ParameterBudgetCalculator(capacity_matching_config)
    
    # Get the network size for capacity matching
    if experiment_type.startswith('match_'):
        # For capacity matching, get the matching size
        network_size = capacity_matching_config['network_sizes'][0]
        matching_size = calculator.get_matching_size(experiment_type, topology_type, network_size)
        print(f"Capacity matching: {network_size} -> {matching_size} nodes")
        actual_size = matching_size
    else:
        # For same_size, use the original size
        actual_size = capacity_matching_config['network_sizes'][0]
    
    # Create training environment
    train_env = DummyVecEnv([make_env(train_task)])
    
    # Create policy with capacity matching
    # Use the specified policy class
    SpecificPolicyClass = lambda obs_space, action_space, lr_schedule, **kwargs: policy_class(
        obs_space, action_space, lr_schedule, 
        topology_type=topology_type, 
        hidden_size=actual_size,  # Use the capacity-matched size
        config=capacity_matching_config,  # Pass centralized config
        **kwargs
    )
    
    # Initialize model with centralized PPO parameters
    ppo_params = capacity_matching_config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/{train_task}_versionD/",
        **ppo_params
    )
    
    # Setup callback
    callback = CrossTaskCallback()
    
    # Train the model with progress bar
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Save trained weights for reuse
    trained_weights = {}
    for name, param in model.policy.named_parameters():
        trained_weights[name] = param.data.clone()
    
    # Test on all tasks with TRUE reuse
    results = {}
    for test_task in test_tasks:
        print(f"\nTesting on {test_task} with TRUE reuse...")
        
        # Create new environment for test task
        test_env = DummyVecEnv([make_env(test_task)])
        
        # Create new model for test task (to handle different observation/action spaces)
        test_model = PPO(
            SpecificPolicyClass,
            test_env,
            verbose=0
        )
        
        # Copy trained weights to test model
        for name, param in test_model.policy.named_parameters():
            if name in trained_weights:
                param.data.copy_(trained_weights[name])
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_model(test_model, test_env, n_eval_episodes=capacity_matching_config['n_eval_episodes'])
        
        results[test_task] = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'train_task': train_task,
            'policy_name': policy_name,
            'topology_type': topology_type,
            'experiment_type': experiment_type,
            'capacity_matched_size': actual_size,
            'reuse_type': 'TRUE_REUSE'
        }
        
        test_env.close()
    
    train_env.close()
    
    return results, callback

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
            if step_count > 10000:
                print(f"      ⚠️  Episode {episode} exceeded 10000 steps, terminating")
                break
        
        rewards.append(episode_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    return mean_reward, std_reward

def verify_capacity_matching(config):
    """
    Verify capacity matching for all experiment types and configurations.
    Uses a two-phase approach: baseline measurement and capacity matching verification.
    """
    print("="*80)
    print("🔍 CAPACITY MATCHING VERIFICATION")
    print("="*80)
    print(f"📋 Verification Configuration:")
    print(f"   • Network Sizes: {config['network_sizes']}")
    print(f"   • Topologies: {['small_world', 'modular', 'hybrid', 'fully_connected']}")
    print(f"   • Network Types: {config['network_types']}")
    print(f"   • Num Layers: {config['num_layers']}")
    print(f"   • Seeds: {config['seeds']}")
    print(f"   • Experiment Types: {config['experiment_types']}")
    print(f"   • Divergence Threshold: 10.0%")
    print("="*80)
    
    # Disable capacity mapping to force incremental adjustment
    config['use_capacity_mapping'] = False
    
    # Initialize measurement manager for baseline measurements
    measurement_manager = CapacityMeasurementManager(config)
    
    # Extract configuration parameters
    sizes = config['network_sizes']
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    network_types = config['network_types']
    num_layers_list = config['num_layers']
    seeds = config['seeds']
    experiment_types = config['experiment_types']  # Test all experiment types including same_size
    node_selection_strategies = config['node_selection_strategies']

    # --- Baseline measurement phase ---
    print("\n📊 Baseline measurement phase: measuring all required capacities...")
    
    total_measurements = len(topologies) * len(sizes) * len(network_types) * len(num_layers_list) * len(seeds)
    measurement_count = 0
    
    with tqdm(total=total_measurements, desc="Measuring capacities") as pbar:
        for topology in topologies:
            for size in sizes:
                for network_type in network_types:
                    for num_layers in num_layers_list:
                        for seed in seeds:
                            if measurement_manager.get_measurement(topology, size, network_type, num_layers) is None:
                                actual_capacity = measurement_manager.measure_capacity(topology, size, network_type, num_layers, seed)
                                measurement_manager.store_measurement(topology, size, network_type, num_layers, actual_capacity, seed)
                                measurement_count += 1
                            pbar.update(1)
    
    measurement_manager._save_measurements()
    print(f"✅ Baseline measurement phase complete. Measured {measurement_count} new capacities.\n")

    # Track results for summary
    results_summary = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {}
    }
    
    # Create calculator AFTER disabling capacity mapping
    calculator = ParameterBudgetCalculator(config)
    
    for exp_type in experiment_types:
        if exp_type.startswith('match_'):
            reference_topology = exp_type[len('match_'):]
            print(f"\n--- Testing {exp_type} (matching to {reference_topology}) ---")
            
            for topology in topologies:
                if topology == reference_topology:
                    continue  # Skip matching topology to itself
                    
                for size in sizes:
                    for network_type in network_types:
                        for num_layers in num_layers_list:
                            print(f"  Testing {topology} (size {size}) matching to {reference_topology}...")
                            
                            try:
                                # Get target capacity from reference topology - working pattern
                                target_capacity = calculator._get_reference_capacity(
                                    reference_topology, size, network_type, num_layers
                                )
                                
                                # Get matching size for current topology - working pattern
                                matching_size = calculator.calculate_matching_size(topology, target_capacity, network_type, num_layers)
                                
                                # Measure actual capacity of matched topology
                                actual_capacity = measurement_manager.measure_capacity(
                                    topology, matching_size, network_type, num_layers, 42
                                )
                                
                                # Calculate divergence
                                divergence = abs(actual_capacity - target_capacity) / target_capacity * 100
                                
                                # Check if within threshold
                                threshold = 10.0  # 10% threshold
                                if divergence <= threshold:
                                    print(f"    ✓ PASSED: {divergence:.1f}% divergence")
                                    print(f"      Size adjustment: {size} → {matching_size} nodes")
                                    print(f"      Target: {target_capacity:,} parameters (from {reference_topology})")
                                    print(f"      Actual: {actual_capacity:,} parameters")
                                    results_summary['passed'] += 1
                                else:
                                    print(f"    ✗ FAILED: {divergence:.1f}% divergence (threshold: {threshold}%)")
                                    print(f"      Size adjustment: {size} → {matching_size} nodes")
                                    print(f"      Target: {target_capacity:,} parameters (from {reference_topology})")
                                    print(f"      Actual: {actual_capacity:,} parameters")
                                    results_summary['failed'] += 1
                                
                                # Store details
                                key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'reference_topology': reference_topology,
                                    'target_capacity': target_capacity,
                                    'matching_size': matching_size,
                                    'actual_capacity': actual_capacity,
                                    'divergence': divergence,
                                    'passed': divergence <= threshold
                                }
                                
                            except Exception as e:
                                print(f"    ✗ ERROR: {e}")
                                results_summary['errors'] += 1
                                
                                # Store error details
                                key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'reference_topology': reference_topology,
                                    'error': str(e),
                                    'passed': False
                                }
        else:
            # For same_size experiments, test all topologies
            print(f"\n--- Testing {exp_type} (same size for all topologies) ---")
            
            for topology in topologies:
                for size in sizes:
                    for network_type in network_types:
                        for num_layers in num_layers_list:
                            print(f"  Testing {topology} (size {size}) with {exp_type}...")
                            
                            try:
                                # For same_size, just verify the network can be created
                                network = calculator.create_network(
                                    topology=topology,
                                    size=size,
                                    experiment_type=exp_type,
                                    network_type=network_type,
                                    num_layers=num_layers,
                                    seed=42
                                )
                                
                                # Get actual capacity
                                metrics = network.get_network_metrics()
                                actual_capacity = sum(
                                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                )
                                
                                print(f"    ✓ PASSED: {actual_capacity:,} parameters")
                                results_summary['passed'] += 1
                                
                                # Store details
                                key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'actual_capacity': actual_capacity,
                                    'passed': True
                                }
                                
                            except Exception as e:
                                print(f"    ✗ ERROR: {e}")
                                results_summary['errors'] += 1
                                
                                # Store error details
                                key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'error': str(e),
                                    'passed': False
                                                                 }
    
    # Print summary
    print(f"\n{'='*80}")
    print("CAPACITY MATCHING VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Passed: {results_summary['passed']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Errors: {results_summary['errors']}")
    print(f"Total: {results_summary['passed'] + results_summary['failed'] + results_summary['errors']}")
    
    if results_summary['failed'] > 0 or results_summary['errors'] > 0:
        print("\nWARNING: Some capacity matching tests failed or had errors.")
        print("This may indicate issues with the capacity matching implementation.")
        print("Check the details above for specific failures.")
    
    return results_summary

def main():
    """Main function to run cross-task transfer analysis with capacity matching."""
    print("🚀 Cross-Task Transfer Analysis with Capacity Matching")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create capacity matching configuration
    config = create_capacity_matching_config()
    
    # Verify capacity matching first
    print("\n🔍 Verifying capacity matching implementation...")
    verification_results = verify_capacity_matching(config)
    
    # Use centralized configuration
    tasks = config['tasks']
    topologies = config['topologies']
    experiment_types = config['experiment_types']
    total_timesteps = config['total_timesteps']
    n_eval_episodes = config['n_eval_episodes']
    
    print(f"\n📋 Experiment Configuration:")
    print(f"   • Tasks: {', '.join(tasks)}")
    print(f"   • Topologies: {', '.join(topologies)}")
    print(f"   • Experiment Types: {', '.join(experiment_types)}")
    print(f"   • Total Experiments: {len(experiment_types) * len(topologies) * len(tasks) * 2} (2 versions per config)")
    print(f"   • Training Timesteps: {total_timesteps:,}")
    print(f"   • Evaluation Episodes: {n_eval_episodes} per task")
    
    # Store all results
    all_results = []
    
    # Run experiments for each capacity matching type
    total_experiments = len(experiment_types) * len(topologies) * len(tasks) * 2
    experiment_count = 0
    
    for experiment_type in experiment_types:
        print(f"\n{'='*80}")
        print(f"🎯 EXPERIMENT TYPE: {experiment_type.upper()}")
        print(f"{'='*80}")
        
        # Run experiments for each topology
        for topology in topologies:
            print(f"\n🔬 Testing {topology} topology")
            print(f"{'─'*60}")
            
            # Train on each task and test on all tasks
            for train_task in tasks:
                test_tasks = [task for task in tasks if task != train_task]
                experiment_count += 2  # Version B and Version D
                
                print(f"\n📚 Training on {train_task} → Testing on {', '.join(test_tasks)}")
                print(f"   Progress: {experiment_count}/{total_experiments} experiments completed")
                
                # Run Version B (padding) with capacity matching
                results_b, callback_b = train_and_test_policy_with_capacity_matching(
                    VersionB_PaddingPolicy_2Layers_Universal,
                    f"VersionB_{topology}",
                    train_task,
                    test_tasks,
                    topology,
                    config,
                    experiment_type=experiment_type,
                    total_timesteps=total_timesteps
                )
                
                # Run Version D (masking) with capacity matching
                results_d, callback_d = train_and_test_policy_with_capacity_matching(
                    VersionD_MaskingPolicy_2Layers_Universal,
                    f"VersionD_{topology}",
                    train_task,
                    test_tasks,
                    topology,
                    config,
                    experiment_type=experiment_type,
                    total_timesteps=total_timesteps
                )
                
                # Combine results
                results = {**results_b, **results_d}
                
                # Store results
                for test_task, result in results.items():
                    all_results.append(result)
                
                # Run TRUE reuse version
                true_reuse_results, true_reuse_callback = train_and_test_policy_true_reuse_with_capacity_matching(
                    VersionB_PaddingPolicy_2Layers_Universal,  # Use Version B as default
                    f"Universal_{topology}_TRUE_REUSE",
                    train_task,
                    test_tasks,
                    topology,
                    config,
                    experiment_type=experiment_type,
                    total_timesteps=total_timesteps
                )
                
                # Store TRUE reuse results
                for test_task, result in true_reuse_results.items():
                    all_results.append(result)
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/cross_task_transfer_with_capacity_matching_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save results
    results_file = os.path.join(results_dir, "cross_task_transfer_results.json")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert results for JSON serialization
    serializable_results = []
    for result in all_results:
        serializable_result = {}
        for key, value in result.items():
            serializable_result[key] = convert_numpy(value)
        serializable_results.append(serializable_result)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create summary table
    df = pd.DataFrame(serializable_results)
    print("\nCross-Task Transfer Results Summary:")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save summary table
    summary_file = os.path.join(results_dir, "summary_table.csv")
    df.to_csv(summary_file, index=False)
    print(f"\nSummary table saved to: {summary_file}")
    
    # Create visualization
    create_transfer_visualization(df, results_dir)
    
    print(f"\nAnalysis complete! Results saved to: {results_dir}")

def create_transfer_visualization(df, results_dir):
    """Create visualization of transfer results."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Transfer performance by topology
    ax1 = axes[0, 0]
    topology_performance = df.groupby('topology_type')['mean_reward'].mean().sort_values(ascending=False)
    topology_performance.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Average Transfer Performance by Topology')
    ax1.set_ylabel('Mean Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Transfer performance by experiment type
    ax2 = axes[0, 1]
    exp_performance = df.groupby('experiment_type')['mean_reward'].mean().sort_values(ascending=False)
    exp_performance.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('Average Transfer Performance by Capacity Matching Type')
    ax2.set_ylabel('Mean Reward')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Transfer performance heatmap
    ax3 = axes[1, 0]
    pivot_data = df.pivot_table(
        values='mean_reward', 
        index='train_task', 
        columns='test_task', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu', ax=ax3)
    ax3.set_title('Transfer Performance Heatmap\n(Train Task → Test Task)')
    
    # 4. Capacity matching effect
    ax4 = axes[1, 1]
    capacity_effect = df.groupby(['topology_type', 'experiment_type'])['mean_reward'].mean().unstack()
    capacity_effect.plot(kind='bar', ax=ax4, color=['lightblue', 'lightcoral'])
    ax4.set_title('Capacity Matching Effect on Transfer Performance')
    ax4.set_ylabel('Mean Reward')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Capacity Matching')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'transfer_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 