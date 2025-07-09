#!/usr/bin/env python3
"""
Extended Simple Topology Test: 2-Layer Fully Connected Topology

This script tests 3 different approaches to using 2-layer topology networks with PPO on CartPole:

1. Version A: Minimal Task Adapters (Universal approach)
   - Uses input/output adapters for future multi-task compatibility
   - CartPole: 4→6 input, 2→3 output adapters
   - 2-layer fully connected topology

2. Version B: IO Padding/Truncation (Universal approach)
   - Uses padding/truncation instead of adapters
   - CartPole: pad 4→6 input, truncate 3→2 output
   - 2-layer fully connected topology

3. Version C: Single Task Direct (Simple approach)
   - No adapters, no padding, direct CartPole dimensions (4→2)
   - Minimal complexity, just 2-layer topology network as MLP replacement

4. Standard MLP (Baseline comparison)
   - Traditional 2-layer MLP for comparison

All versions use 2-layer fully connected topology and should achieve similar performance to standard MLP.
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

# Add src to path
sys.path.append('src')

from src.topologies.fully_connected import FullyConnectedTopology
from src.networks.ffn import FeedForwardNetwork

class TrainingCallback(BaseCallback):
    """Callback to track training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.training_env.buf_rews) > 0:
            self.current_episode_reward += self.training_env.buf_rews[0]
            self.current_episode_length += 1
            
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        return True

class StandardMLPPolicy(ActorCriticPolicy):
    """Standard 2-layer MLP policy for comparison."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # 2-layer architecture matching the topology networks
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),  # 4 → 64
            nn.ReLU(),
            nn.Linear(64, 64),                 # 64 → 64
            nn.ReLU(),
            nn.Linear(64, 64),                 # 64 → 64 (second layer)
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),                 # 64 → 32
            nn.ReLU(),
            nn.Linear(32, action_space.n)      # 32 → 2
        )
        
        # Critic head (value)
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),                 # 64 → 32
            nn.ReLU(),
            nn.Linear(32, 1)                   # 32 → 1
        )
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.action_net(shared_features)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.value_net(shared_features)

# ============================================================================
# VERSION A: Minimal Task Adapters (Universal approach) - 2 LAYERS
# ============================================================================

class MinimalAdapter(nn.Module):
    """Minimal adapter with configurable complexity."""
    
    def __init__(self, input_dim: int, output_dim: int, adapter_type: str = 'linear'):
        super().__init__()
        self.adapter_type = adapter_type
        
        if adapter_type == 'linear':
            # Simple linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        elif adapter_type == 'tiny_mlp':
            # Tiny MLP with minimal hidden layer
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
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]  # 4 for CartPole
        self.task_output_dim = action_space.n  # 2 for CartPole
        
        # Universal dimensions (fixed across all tasks)
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        
        print(f"[Version A - 2 Layers] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version A - 2 Layers] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        # Create minimal input adapter (4 → 6)
        self.input_adapter = MinimalAdapter(
            self.task_input_dim, 
            self.universal_input_dim, 
            adapter_type='linear'
        )
        
        # Create 2-layer topology network
        self.topology_network = self._create_topology_network()
        
        # Create minimal output adapter (3 → 2)
        self.output_adapter = MinimalAdapter(
            self.universal_output_dim, 
            self.task_output_dim, 
            adapter_type='linear'
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        """Create the 2-layer topology network with universal dimensions."""
        # Total nodes per layer: input + hidden + output
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create 2-layer fully connected topology
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,  # 2 layers
            inter_layer_prob=1.0,  # Fully connected between layers
            intra_layer_prob=1.0,  # Fully connected within layers
            seed=42
        )
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes_layer1 = list(range(self.universal_input_dim))
        output_nodes_layer1 = list(range(self.universal_input_dim + self.hidden_size, 
                                        self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create first layer network
        self.layer1_network = FeedForwardNetwork(
            topology=graphs[0],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
        
        # Create second layer network
        self.layer2_network = FeedForwardNetwork(
            topology=graphs[1],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Input adapter: project to universal input space
        universal_input = self.input_adapter(features)  # (batch_size, 6)
        
        # Convert to dictionary format for first layer
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        layer1_tensor = torch.stack(layer1_values, dim=1)  # (batch_size, 3)
        
        # Use layer1 output as input to layer2 (with adapter if needed)
        layer2_input = self.input_adapter(layer1_tensor)  # (batch_size, 6)
        
        # Convert to dictionary format for second layer
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)  # (batch_size, 3)
        
        # Output adapter: project to task output space
        task_output = self.output_adapter(universal_output)  # (batch_size, 2)
        
        # Actor head
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Input adapter: project to universal input space
        universal_input = self.input_adapter(features)
        
        # Convert to dictionary format for first layer
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        
        # Use layer1 output as input to layer2
        layer2_input = self.input_adapter(layer1_tensor)
        
        # Convert to dictionary format for second layer
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
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
        
        # Output adapter: project to task output space
        task_output = self.output_adapter(universal_output)
        
        # Critic head
        return self.critic_head(task_output)

# ============================================================================
# VERSION B: IO Padding/Truncation (Universal approach) - 2 LAYERS
# ============================================================================

class VersionB_PaddingPolicy_2Layers(ActorCriticPolicy):
    """Version B: Universal 2-layer topology with padding/truncation."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]  # 4 for CartPole
        self.task_output_dim = action_space.n  # 2 for CartPole
        
        # Universal dimensions (fixed across all tasks)
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        
        print(f"[Version B - 2 Layers] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version B - 2 Layers] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        # Create 2-layer topology network
        self.topology_network = self._create_topology_network()
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        """Create the 2-layer topology network with universal dimensions."""
        # Total nodes per layer: input + hidden + output
        layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        # Create 2-layer fully connected topology
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,  # 2 layers
            inter_layer_prob=1.0,  # Fully connected between layers
            intra_layer_prob=1.0,  # Fully connected within layers
            seed=42
        )
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes_layer1 = list(range(self.universal_input_dim))
        output_nodes_layer1 = list(range(self.universal_input_dim + self.hidden_size, 
                                        self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create first layer network
        self.layer1_network = FeedForwardNetwork(
            topology=graphs[0],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
        
        # Create second layer network
        self.layer2_network = FeedForwardNetwork(
            topology=graphs[1],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input from task dimension to universal dimension."""
        batch_size = x.shape[0]
        if x.shape[1] < self.universal_input_dim:
            # Pad with zeros
            padding = torch.zeros(batch_size, self.universal_input_dim - x.shape[1], device=x.device)
            return torch.cat([x, padding], dim=1)
        else:
            # Truncate if needed
            return x[:, :self.universal_input_dim]
    
    def _truncate_output(self, x: torch.Tensor) -> torch.Tensor:
        """Truncate output from universal dimension to task dimension."""
        return x[:, :self.task_output_dim]
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Pad input: 4 → 6
        universal_input = self._pad_input(features)
        
        # Convert to dictionary format for first layer
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        layer1_tensor = torch.stack(layer1_values, dim=1)  # (batch_size, 3)
        
        # Pad layer1 output for layer2 input: 3 → 6
        layer2_input = self._pad_input(layer1_tensor)
        
        # Convert to dictionary format for second layer
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
        layer2_values = []
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        universal_output = torch.stack(layer2_values, dim=1)  # (batch_size, 3)
        
        # Truncate output: 3 → 2
        task_output = self._truncate_output(universal_output)
        
        # Actor head
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Pad input: 4 → 6
        universal_input = self._pad_input(features)
        
        # Convert to dictionary format for first layer
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        
        # Pad layer1 output for layer2 input: 3 → 6
        layer2_input = self._pad_input(layer1_tensor)
        
        # Convert to dictionary format for second layer
        layer2_input_dict = {i: layer2_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
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
        
        # Truncate output: 3 → 2
        task_output = self._truncate_output(universal_output)
        
        # Critic head
        return self.critic_head(task_output)

# ============================================================================
# VERSION C: Single Task Direct (Simple approach) - 2 LAYERS
# ============================================================================

class VersionC_DirectPolicy_2Layers(ActorCriticPolicy):
    """Version C: Direct 2-layer topology with CartPole dimensions (4→2)."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Direct CartPole dimensions
        self.input_dim = observation_space.shape[0]  # 4 for CartPole
        self.output_dim = action_space.n  # 2 for CartPole
        self.hidden_size = 64
        
        print(f"[Version C - 2 Layers] Direct: {self.input_dim}→{self.output_dim}")
        
        # Create 2-layer topology network
        self.topology_network = self._create_topology_network()
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self):
        """Create the 2-layer topology network with direct CartPole dimensions."""
        # Total nodes per layer: input + hidden + output
        layer_size = self.input_dim + self.hidden_size + self.output_dim
        
        # Create 2-layer fully connected topology
        topology = FullyConnectedTopology(
            size=layer_size,
            num_layers=2,  # 2 layers
            inter_layer_prob=1.0,  # Fully connected between layers
            intra_layer_prob=1.0,  # Fully connected within layers
            seed=42
        )
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes_layer1 = list(range(self.input_dim))
        output_nodes_layer1 = list(range(self.input_dim + self.hidden_size, 
                                        self.input_dim + self.hidden_size + self.output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create first layer network
        self.layer1_network = FeedForwardNetwork(
            topology=graphs[0],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
        
        # Create second layer network
        self.layer2_network = FeedForwardNetwork(
            topology=graphs[1],
            input_nodes=input_nodes_layer1,
            output_nodes=output_nodes_layer1,
            network_params=network_params
        )
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Convert to dictionary format for first layer
        input_dict = {i: features[:, i] for i in range(self.input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        layer1_tensor = torch.stack(layer1_values, dim=1)  # (batch_size, 2)
        
        # Use layer1 output as input to layer2
        layer2_input_dict = {i: layer1_tensor[:, i] for i in range(self.input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
        layer2_values = []
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        task_output = torch.stack(layer2_values, dim=1)  # (batch_size, 2)
        
        # Actor head
        return self.actor_head(task_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Convert to dictionary format for first layer
        input_dict = {i: features[:, i] for i in range(self.input_dim)}
        
        # Forward through first layer
        layer1_output = self.layer1_network.forward(input_dict)
        
        # Convert layer1 output to tensor
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
        
        # Use layer1 output as input to layer2
        layer2_input_dict = {i: layer1_tensor[:, i] for i in range(self.input_dim)}
        
        # Forward through second layer
        layer2_output = self.layer2_network.forward(layer2_input_dict)
        
        # Convert layer2 output to tensor
        layer2_values = []
        for i in range(self.output_dim):
            output_node_idx = self.input_dim + self.hidden_size + i
            val = layer2_output.get(output_node_idx, 0.0)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=features.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            layer2_values.append(val)
        task_output = torch.stack(layer2_values, dim=1)
        
        # Critic head
        return self.critic_head(task_output)

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_policy(policy_class, policy_name, total_timesteps=50000):
    """Test a specific policy."""
    print(f"\n=== Testing {policy_name} ===")
    
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        policy_class,
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.15,
        ent_coef=0.02,
        verbose=0
    )
    
    callback = TrainingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Test performance
    test_rewards = []
    for i in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        test_rewards.append(total_reward)
    
    avg_reward = np.mean(test_rewards)
    print(f"{policy_name} - Average test reward: {avg_reward:.2f}")
    print(f"{policy_name} - All test rewards: {test_rewards}")
    
    return {
        'type': policy_name,
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def plot_results(results):
    """Plot training curves for all policies."""
    plt.figure(figsize=(15, 5))
    
    # Plot episode rewards
    plt.subplot(1, 3, 1)
    for result in results:
        if result['episode_rewards']:
            plt.plot(result['episode_rewards'], label=result['type'], alpha=0.7)
    plt.title('Training Episode Rewards (2-Layer Topologies)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot test rewards distribution
    plt.subplot(1, 3, 2)
    test_rewards_data = [result['test_rewards'] for result in results]
    plt.boxplot(test_rewards_data, labels=[result['type'] for result in results])
    plt.title('Test Rewards Distribution (2-Layer Topologies)')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot average test rewards
    plt.subplot(1, 3, 3)
    avg_rewards = [result['avg_reward'] for result in results]
    policy_names = [result['type'] for result in results]
    bars = plt.bar(policy_names, avg_rewards)
    plt.title('Average Test Rewards (2-Layer Topologies)')
    plt.ylabel('Average Reward')
    
    # Add value labels on bars
    for bar, reward in zip(bars, avg_rewards):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{reward:.1f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('topology_comparison_2layers_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all tests and compare results."""
    print("🚀 Starting Extended Simple Topology Test with 2-Layer Fully Connected")
    print("=" * 70)
    
    # Test all policies
    results = []
    
    # Test Standard MLP (baseline)
    results.append(test_policy(StandardMLPPolicy, "Standard MLP (2-Layer)"))
    
    # Test Version A: Minimal Task Adapters (2-Layer)
    results.append(test_policy(VersionA_AdapterPolicy_2Layers, "Version A: Adapters (2-Layer)"))
    
    # Test Version B: IO Padding/Truncation (2-Layer)
    results.append(test_policy(VersionB_PaddingPolicy_2Layers, "Version B: Padding (2-Layer)"))
    
    # Test Version C: Single Task Direct (2-Layer)
    results.append(test_policy(VersionC_DirectPolicy_2Layers, "Version C: Direct (2-Layer)"))
    
    # Plot results
    plot_results(results)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY (2-Layer Topologies)")
    print("=" * 70)
    for result in results:
        print(f"{result['type']:35} | Avg Reward: {result['avg_reward']:6.2f} | "
              f"Min: {min(result['test_rewards']):4.0f} | Max: {max(result['test_rewards']):4.0f}")
    
    print("\n✅ All 2-layer topology tests completed!")
    print("📁 Check 'topology_comparison_2layers_results.png' for visual results.")

if __name__ == "__main__":
    main() 