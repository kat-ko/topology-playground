#!/usr/bin/env python3
"""
Comprehensive 2-Layer Topology Test with All Tasks

This script provides comprehensive analysis including:
- All 4 topology versions (A, B, C, D) + Standard MLP
- All 3 tasks (CartPole, Acrobot, MountainCar)
- Fitness metrics (reward curves, convergence)
- Trajectory analysis (episode lengths, action distributions)
- Weight values and gradients
- Performance comparisons across tasks
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import sys
import os
import time
from typing import Dict, List, Tuple, Any
import json

# Add src to path
sys.path.append('src')

from src.topologies.fully_connected import FullyConnectedTopology
from src.networks.ffn import FeedForwardNetwork

class DetailedTrainingCallback(BaseCallback):
    """Enhanced callback to track detailed training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.gradient_norms = []
        self.weight_norms = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.action_distributions = []
        
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

class StandardMLPPolicy(ActorCriticPolicy):
    """Standard 2-layer MLP policy for comparison."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # 2-layer architecture
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.action_net(shared_features)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.value_net(shared_features)
    
    def get_weight_stats(self):
        """Get weight statistics for analysis."""
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats

class MinimalAdapter(nn.Module):
    """Minimal adapter with configurable complexity."""
    
    def __init__(self, input_dim: int, output_dim: int, adapter_type: str = 'linear'):
        super().__init__()
        self.adapter_type = adapter_type
        
        if adapter_type == 'linear':
            self.projection = nn.Linear(input_dim, output_dim)
        elif adapter_type == 'tiny_mlp':
            self.projection = nn.Sequential(
                nn.Linear(input_dim, 8),
                nn.ReLU(),
                nn.Linear(8, output_dim)
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class VersionA_AdapterPolicy_2Layers(ActorCriticPolicy):
    """Version A: Universal 2-layer topology with minimal task adapters."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        
        print(f"[Version A - 2 Layers] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version A - 2 Layers] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        # Create adapters
        self.input_adapter = MinimalAdapter(self.task_input_dim, self.universal_input_dim, 'linear')
        self.output_adapter = MinimalAdapter(self.universal_output_dim, self.task_output_dim, 'linear')
        
        # Create 2-layer topology network
        self._create_topology_network()
        
        # Actor/Critic heads
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        """Create the 2-layer topology network."""
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=42
        )
        
        graphs = topology.generate(2)
        
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        universal_input = self.input_adapter(features)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input = self.input_adapter(layer1_tensor)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        task_output = self.output_adapter(universal_output)
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        universal_input = self.input_adapter(features)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input = self.input_adapter(layer1_tensor)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        task_output = self.output_adapter(universal_output)
        return self.critic_head(task_output)
    
    def get_weight_stats(self):
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats
    
    def _get_topology_weight_stats(self, network):
        """Get weight statistics for topology network."""
        stats = {}
        for name, param in network.named_parameters():
            if param.requires_grad:
                stats[f"topology_{name}"] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats

class VersionB_PaddingPolicy_2Layers(ActorCriticPolicy):
    """Version B: Universal 2-layer topology with padding/truncation."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        
        print(f"[Version B - 2 Layers] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version B - 2 Layers] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        self._create_topology_network()
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=42
        )
        
        graphs = topology.generate(2)
        
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if x.shape[1] < self.universal_input_dim:
            padding = torch.zeros(batch_size, self.universal_input_dim - x.shape[1], device=x.device)
            return torch.cat([x, padding], dim=1)
        else:
            return x[:, :self.universal_input_dim]
    
    def _truncate_output(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :self.task_output_dim]
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        universal_input = self._pad_input(features)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input = self._pad_input(layer1_tensor)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        task_output = self._truncate_output(universal_output)
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        universal_input = self._pad_input(features)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input = self._pad_input(layer1_tensor)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        task_output = self._truncate_output(universal_output)
        return self.critic_head(task_output)
    
    def get_weight_stats(self):
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats

class VersionC_DirectPolicy_2Layers(ActorCriticPolicy):
    """Version C: Direct 2-layer topology with task-specific dimensions."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n
        self.hidden_size = 64
        
        print(f"[Version C - 2 Layers] Direct: {self.input_dim}→{self.output_dim}")
        
        self._create_topology_network()
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        layer_size = self.input_dim + self.hidden_size + self.output_dim
        
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=42
        )
        
        graphs = topology.generate(2)
        
        input_nodes = list(range(self.input_dim))
        output_nodes = list(range(self.input_dim + self.hidden_size, 
                                 self.input_dim + self.hidden_size + self.output_dim))
        
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        
        # Layer 1
        input_dict = {i: features[:, i] for i in range(self.input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input_dict = {i: layer1_tensor[:, i] for i in range(self.input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        output_tensor = torch.stack(layer2_values, dim=1)
        
        return self.actor_head(output_tensor)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        
        # Layer 1
        input_dict = {i: features[:, i] for i in range(self.input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Layer 2
        layer2_input_dict = {i: layer1_tensor[:, i] for i in range(self.input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        output_tensor = torch.stack(layer2_values, dim=1)
        
        return self.critic_head(output_tensor)
    
    def get_weight_stats(self):
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats

class VersionD_MaskingPolicy_2Layers(ActorCriticPolicy):
    """Version D: Universal 2-layer topology with masking instead of padding."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        
        print(f"[Version D - 2 Layers] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version D - 2 Layers] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        self._create_topology_network()
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=42
        )
        
        graphs = topology.generate(2)
        
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        network_params = {'learning_rate': 0.001, 'activation': 'tanh'}
        
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def _create_input_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create a mask for input nodes based on actual input dimensions."""
        batch_size = x.shape[0]
        mask = torch.zeros(batch_size, self.universal_input_dim, device=x.device)
        
        # Set mask to 1 for actual input dimensions, 0 for padding
        actual_dim = min(x.shape[1], self.universal_input_dim)
        mask[:, :actual_dim] = 1.0
        
        return mask
    
    def _create_output_mask(self) -> torch.Tensor:
        """Create a mask for output nodes based on task output dimensions."""
        # For now, we'll use a simple approach - mask out unused output dimensions
        # In practice, this could be more sophisticated
        mask = torch.ones(self.universal_output_dim)
        if self.task_output_dim < self.universal_output_dim:
            mask[self.task_output_dim:] = 0.0
        return mask
    
    def _apply_input_masking(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masking to input, setting masked values to 0."""
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
        """Apply masking to output, keeping only relevant dimensions."""
        # Apply mask to output
        masked_output = x * mask.unsqueeze(0).expand(x.shape[0], -1)
        
        # Return only the task-relevant dimensions
        return masked_output[:, :self.task_output_dim]
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        
        # Create masks
        input_mask = self._create_input_mask(features)
        output_mask = self._create_output_mask().to(features.device)
        
        # Apply input masking
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Apply output masking to layer 1
        layer1_masked = self._apply_output_masking(layer1_tensor, output_mask)
        
        # Layer 2 - use masked layer 1 output as input
        layer2_input = self._apply_input_masking(layer1_masked, input_mask)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        # Apply final output masking
        task_output = self._apply_output_masking(universal_output, output_mask)
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        
        # Create masks
        input_mask = self._create_input_mask(features)
        output_mask = self._create_output_mask().to(features.device)
        
        # Apply input masking
        universal_input = self._apply_input_masking(features, input_mask)
        
        # Layer 1
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        layer1_output = self.layer1_network.forward(input_dict)
        
        layer1_values = []
        batch_size = features.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer1_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer1_values.append(val)
        layer1_tensor = torch.stack(layer1_values, dim=1)
        
        # Apply output masking to layer 1
        layer1_masked = self._apply_output_masking(layer1_tensor, output_mask)
        
        # Layer 2 - use masked layer 1 output as input
        layer2_input = self._apply_input_masking(layer1_masked, input_mask)
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)
        
        # Apply final output masking
        task_output = self._apply_output_masking(universal_output, output_mask)
        return self.critic_head(task_output)
    
    def get_weight_stats(self):
        stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'norm': param.data.norm().item()
                }
        return stats 