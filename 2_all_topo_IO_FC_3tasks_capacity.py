#!/usr/bin/env python3
"""
Comprehensive Topology Test: All Topologies on All Tasks (2-Layer Networks)
WITH CAPACITY MATCHING

This script tests ALL topology types on ALL tasks using ALL three versions with 2-layer networks,
ensuring all networks have comparable parameter counts for fair comparison.

Topologies:
- Fully Connected
- Small World  
- Modular
- Hybrid

Tasks:
- CartPole-v1
- MountainCar-v0
- Acrobot-v1

Versions:
- Version A: Minimal Task Adapters (Universal approach)
- Version B: IO Padding/Truncation (Universal approach)  
- Version C: Single Task Direct (Simple approach)
- Standard MLP (Baseline comparison)

Total: 4 topologies × 3 tasks × 4 versions = 48 experiments
All using 2-layer topology networks with capacity matching
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

# Add src to path
sys.path.append('src')

from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.networks.ffn import FeedForwardNetwork 
from src.utils.capacity_mapping import CapacityMapper
from src.utils.capacity_measurement import CapacityMeasurementManager

# ============================================================================
# CAPACITY MATCHING CONFIGURATION
# ============================================================================

def get_capacity_matching_config():
    """Get configuration for capacity matching."""
    return {
        'network_types': ['ffn'],
        'num_layers': [2],
        'num_io_nodes': [6, 3],  # Universal input/output dimensions
        'small_world_params': {
            'k': 4,
            'p': 0.1,
            'inter_layer_prob': 0.1
        },
        'modular_params': {
            'num_modules': 4,
            'inter_module_prob': 0.1,
            'intra_module_prob': 0.3,
            'inter_layer_prob': 0.1
        },
        'hybrid_params': {
            'num_modules': 4,
            'k': 4,
            'p': 0.1,
            'inter_module_prob': 0.1,
            'inter_layer_prob': 0.1
        },
        'fully_connected_params': {
            'inter_layer_prob': 1.0,
            'intra_layer_prob': 1.0
        },
        'network_params': {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
    }

def calculate_standard_mlp_capacity(observation_space, action_space):
    """Calculate the exact parameter count of the standard MLP."""
    # Create a temporary model to count parameters
    temp_model = StandardMLPPolicy(observation_space, action_space, lambda _: 0.001)
    
    # Count parameters
    total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    
    print(f"Standard MLP capacity: {total_params} parameters")
    return total_params

def get_capacity_matched_size(topology_type: str, target_capacity: int, config: Dict) -> int:
    """Get the appropriate size for a topology to match target capacity."""
    try:
        # Create capacity mapper
        mapper = CapacityMapper(config)
        
        # Find matching size for this topology
        matched_size = mapper.find_matching_size(
            topology=topology_type,
            target_capacity=target_capacity,
            network_type='ffn',
            num_layers=2
        )
        
        print(f"Capacity-matched size for {topology_type}: {matched_size} (target: {target_capacity})")
        return matched_size
        
    except Exception as e:
        print(f"Warning: Could not find capacity-matched size for {topology_type}: {e}")
        print(f"Using default size of 100")
        return 100

# ============================================================================
# TRAINING CALLBACK
# ============================================================================

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

# ============================================================================
# STANDARD MLP (BASELINE)
# ============================================================================

class StandardMLPPolicy(ActorCriticPolicy):
    """Standard 2-layer MLP policy for comparison."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # 2-layer architecture matching the topology networks
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),  # Input → 64
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
            nn.Linear(32, action_space.n)      # 32 → action_dim
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
    """Version A: Universal topology with minimal task adapters (2-layer) with capacity matching."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', target_capacity=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Universal dimensions (fixed across all tasks)
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        self.topology_type = topology_type
        
        print(f"[Version A 2L - {topology_type}] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version A 2L - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        # Create minimal input adapter
        self.input_adapter = MinimalAdapter(
            self.task_input_dim, 
            self.universal_input_dim, 
            adapter_type='linear'
        )
        
        # Create 2-layer topology network with capacity matching
        self.topology_network = self._create_topology_network(target_capacity)
        
        # Create minimal output adapter
        self.output_adapter = MinimalAdapter(
            self.universal_output_dim, 
            self.task_output_dim, 
            adapter_type='linear'
        )
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n if hasattr(action_space, 'n') else action_space.shape[0])
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self, target_capacity=None):
        """Create the 2-layer topology network with capacity matching."""
        # Get capacity-matched size if target_capacity is provided
        if target_capacity is not None:
            config = get_capacity_matching_config()
            layer_size = get_capacity_matched_size(self.topology_type, target_capacity, config)
        else:
            # Fallback to default size
            layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        print(f"[Version A 2L - {self.topology_type}] Using layer size: {layer_size}")
        
        # Create 2-layer topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=layer_size,
                num_layers=2,  # 2 layers
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=layer_size,
                k=4,
                p=0.1,
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=layer_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=2,  # 2 layers
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
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create 2-layer network
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Input adapter
        adapted_input = self.input_adapter(features)
        
        # First layer
        layer1_output = self.layer1_network.forward(adapted_input)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Output adapter
        adapted_output = self.output_adapter(layer2_output)
        
        # Actor head
        return self.actor_head(adapted_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Input adapter
        adapted_input = self.input_adapter(features)
        
        # First layer
        layer1_output = self.layer1_network.forward(adapted_input)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Output adapter
        adapted_output = self.output_adapter(layer2_output)
        
        # Critic head
        return self.critic_head(adapted_output)

# ============================================================================
# VERSION B: IO Padding/Truncation (Universal approach) - 2 LAYERS
# ============================================================================

class VersionB_PaddingPolicy_2Layers(ActorCriticPolicy):
    """Version B: Universal topology with IO padding/truncation (2-layer) with capacity matching."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', target_capacity=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Universal dimensions (fixed across all tasks)
        self.universal_input_dim = 6
        self.universal_output_dim = 3
        self.hidden_size = 64
        self.topology_type = topology_type
        
        print(f"[Version B 2L - {topology_type}] Task: {self.task_input_dim}→{self.task_output_dim}")
        print(f"[Version B 2L - {topology_type}] Universal: {self.universal_input_dim}→{self.universal_output_dim}")
        
        # Create 2-layer topology network with capacity matching
        self.topology_network = self._create_topology_network(target_capacity)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n if hasattr(action_space, 'n') else action_space.shape[0])
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.universal_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self, target_capacity=None):
        """Create the 2-layer topology network with capacity matching."""
        # Get capacity-matched size if target_capacity is provided
        if target_capacity is not None:
            config = get_capacity_matching_config()
            layer_size = get_capacity_matched_size(self.topology_type, target_capacity, config)
        else:
            # Fallback to default size
            layer_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        print(f"[Version B 2L - {self.topology_type}] Using layer size: {layer_size}")
        
        # Create 2-layer topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=layer_size,
                num_layers=2,  # 2 layers
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=layer_size,
                k=4,
                p=0.1,
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=layer_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=2,  # 2 layers
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
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create 2-layer network
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        """Pad or truncate input to universal dimensions."""
        if x.shape[-1] < self.universal_input_dim:
            # Pad with zeros
            padding = torch.zeros(x.shape[:-1] + (self.universal_input_dim - x.shape[-1],), device=x.device)
            return torch.cat([x, padding], dim=-1)
        else:
            # Truncate
            return x[..., :self.universal_input_dim]
    
    def _truncate_output(self, x: torch.Tensor) -> torch.Tensor:
        """Truncate output to task dimensions."""
        return x[..., :self.task_output_dim]
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Pad input to universal dimensions
        padded_input = self._pad_input(features)
        
        # First layer
        layer1_output = self.layer1_network.forward(padded_input)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Truncate output to task dimensions
        truncated_output = self._truncate_output(layer2_output)
        
        # Actor head
        return self.actor_head(truncated_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Pad input to universal dimensions
        padded_input = self._pad_input(features)
        
        # First layer
        layer1_output = self.layer1_network.forward(padded_input)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Truncate output to task dimensions
        truncated_output = self._truncate_output(layer2_output)
        
        # Critic head
        return self.critic_head(truncated_output)

# ============================================================================
# VERSION C: Single Task Direct (Simple approach) - 2 LAYERS
# ============================================================================

class VersionC_DirectPolicy_2Layers(ActorCriticPolicy):
    """Version C: Single task direct mapping (2-layer) with capacity matching."""
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', target_capacity=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.task_input_dim = observation_space.shape[0]
        self.task_output_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        self.hidden_size = 64
        self.topology_type = topology_type
        
        print(f"[Version C 2L - {topology_type}] Task: {self.task_input_dim}→{self.task_output_dim}")
        
        # Create 2-layer topology network with capacity matching
        self.topology_network = self._create_topology_network(target_capacity)
        
        # Actor head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n if hasattr(action_space, 'n') else action_space.shape[0])
        )
        
        # Critic head (value)
        self.critic_head = nn.Sequential(
            nn.Linear(self.task_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def _create_topology_network(self, target_capacity=None):
        """Create the 2-layer topology network with capacity matching."""
        # Get capacity-matched size if target_capacity is provided
        if target_capacity is not None:
            config = get_capacity_matching_config()
            layer_size = get_capacity_matched_size(self.topology_type, target_capacity, config)
        else:
            # Fallback to default size
            layer_size = self.task_input_dim + self.hidden_size + self.task_output_dim
        
        print(f"[Version C 2L - {self.topology_type}] Using layer size: {layer_size}")
        
        # Create 2-layer topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=layer_size,
                num_layers=2,  # 2 layers
                inter_layer_prob=1.0,
                intra_layer_prob=1.0,
                seed=42
            )
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=layer_size,
                k=4,
                p=0.1,
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=layer_size,
                num_modules=4,
                inter_module_prob=0.1,
                intra_module_prob=0.3,
                num_layers=2,  # 2 layers
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
                num_layers=2,  # 2 layers
                inter_layer_prob=0.1,
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Generate 2-layer graph
        graphs = topology.generate(2)  # Generate 2 layers
        
        # Define input/output nodes for each layer
        input_nodes = list(range(self.task_input_dim))
        output_nodes = list(range(self.task_input_dim + self.hidden_size, 
                                 self.task_input_dim + self.hidden_size + self.task_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        # Create 2-layer network
        self.layer1_network = FeedForwardNetwork(graphs[0], input_nodes, output_nodes, network_params)
        self.layer2_network = FeedForwardNetwork(graphs[1], input_nodes, output_nodes, network_params)
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # First layer
        layer1_output = self.layer1_network.forward(features)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Actor head
        return self.actor_head(layer2_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # First layer
        layer1_output = self.layer1_network.forward(features)
        
        # Second layer
        layer2_output = self.layer2_network.forward(layer1_output)
        
        # Critic head
        return self.critic_head(layer2_output)

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def test_policy(policy_class, policy_name, env_name, total_timesteps=10000, target_capacity=None):
    """Test a policy on a specific environment."""
    
    def make_env():
        env = gym.make(env_name)
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(1)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create policy with capacity matching if specified
    if target_capacity is not None and 'topology' in policy_name.lower():
        policy = policy_class(env.observation_space, env.action_space, lambda _: 0.001, target_capacity=target_capacity)
    else:
        policy = policy_class(env.observation_space, env.action_space, lambda _: 0.001)
    
    # Create callback for tracking
    callback = TrainingCallback()
    
    # Create model
    model = PPO(policy, env, verbose=0, learning_rate=0.001, n_steps=2048, batch_size=64, n_epochs=10)
    
    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    training_time = time.time() - start_time
    
    # Evaluate
    eval_env = gym.make(env_name)
    obs = eval_env.reset()[0]
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(10):  # 10 evaluation episodes
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if truncated:
                done = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        obs = eval_env.reset()[0]
    
    eval_env.close()
    
    # Calculate metrics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)
    
    # Check if solved (task-specific criteria)
    solved = False
    if env_name == 'CartPole-v1':
        solved = avg_reward >= 195.0  # CartPole solved criteria
    elif env_name == 'MountainCar-v0':
        solved = avg_reward >= -110.0  # MountainCar solved criteria
    elif env_name == 'Acrobot-v1':
        solved = avg_length <= 100  # Acrobot solved criteria (shorter episodes = better)
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    
    return {
        'policy_name': policy_name,
        'env_name': env_name,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'std_reward': std_reward,
        'solved': solved,
        'training_time': training_time,
        'total_params': total_params,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths
    }

# ============================================================================
# PLOTTING AND ANALYSIS
# ============================================================================

def plot_comprehensive_results(results):
    """Plot comprehensive results from all experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Topology Test Results (2-Layer Networks with Capacity Matching)', fontsize=16)
    
    # Group results by environment
    envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
    
    for i, env in enumerate(envs):
        env_results = [r for r in results if r['env_name'] == env]
        
        # Plot average rewards
        policy_names = [r['policy_name'] for r in env_results]
        avg_rewards = [r['avg_reward'] for r in env_results]
        colors = ['red' if 'MLP' in name else 'blue' for name in policy_names]
        
        axes[0, i].bar(range(len(policy_names)), avg_rewards, color=colors, alpha=0.7)
        axes[0, i].set_title(f'{env} - Average Reward')
        axes[0, i].set_ylabel('Average Reward')
        axes[0, i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(avg_rewards):
            axes[0, i].text(j, v + max(avg_rewards) * 0.01, f'{v:.1f}', ha='center', va='bottom')
        
        # Plot parameter counts
        param_counts = [r['total_params'] for r in env_results]
        axes[1, i].bar(range(len(policy_names)), param_counts, color=colors, alpha=0.7)
        axes[1, i].set_title(f'{env} - Parameter Count')
        axes[1, i].set_ylabel('Total Parameters')
        axes[1, i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(param_counts):
            axes[1, i].text(j, v + max(param_counts) * 0.01, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_topology_test_results_2layers_capacity_matched.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results):
    """Create a summary table of results."""
    print("\n" + "="*100)
    print("COMPREHENSIVE TOPOLOGY TEST RESULTS (2-Layer Networks with Capacity Matching)")
    print("="*100)
    
    # Group by environment
    for env in ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']:
        print(f"\n{env}:")
        print("-" * 80)
        print(f"{'Policy':<40} {'Avg Reward':<12} {'Solved':<8} {'Params':<10} {'Time(s)':<8}")
        print("-" * 80)
        
        env_results = [r for r in results if r['env_name'] == env]
        env_results.sort(key=lambda x: x['avg_reward'], reverse=True)
        
        for result in env_results:
            policy_name = result['policy_name'][:39]  # Truncate if too long
            avg_reward = f"{result['avg_reward']:.1f}"
            solved = "✓" if result['solved'] else "✗"
            params = f"{result['total_params']:,}"
            time_str = f"{result['training_time']:.1f}"
            
            print(f"{policy_name:<40} {avg_reward:<12} {solved:<8} {params:<10} {time_str:<8}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Run all experiments with capacity matching."""
    print("Starting Comprehensive Topology Test (2-Layer Networks with Capacity Matching)")
    print("="*80)
    
    # Test environments
    envs = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
    
    # Topology types
    topologies = ['fully_connected', 'small_world', 'modular', 'hybrid']
    
    # Policy classes with their names
    policy_configs = [
        (StandardMLPPolicy, "Standard MLP (2L)"),
        (VersionA_AdapterPolicy_2Layers, "Version A - Adapters"),
        (VersionB_PaddingPolicy_2Layers, "Version B - Padding"),
        (VersionC_DirectPolicy_2Layers, "Version C - Direct")
    ]
    
    # First, calculate the standard MLP capacity for each environment
    print("\nCalculating standard MLP capacities...")
    target_capacities = {}
    
    for env_name in envs:
        env = gym.make(env_name)
        target_capacity = calculate_standard_mlp_capacity(env.observation_space, env.action_space)
        target_capacities[env_name] = target_capacity
        env.close()
    
    print(f"\nTarget capacities: {target_capacities}")
    
    # Run all experiments
    all_results = []
    
    for env_name in envs:
        print(f"\n{'='*20} Testing {env_name} {'='*20}")
        
        # Get target capacity for this environment
        target_capacity = target_capacities[env_name]
        
        for policy_class, policy_base_name in policy_configs:
            if 'Standard MLP' in policy_base_name:
                # Standard MLP doesn't need capacity matching
                for _ in range(1):  # Only one standard MLP per environment
                    result = test_policy(policy_class, policy_base_name, env_name, total_timesteps=10000)
                    all_results.append(result)
                    print(f"✓ {policy_base_name} on {env_name}: {result['avg_reward']:.1f} reward, {result['total_params']:,} params")
            else:
                # Topology networks with capacity matching
                for topology in topologies:
                    policy_name = f"{policy_base_name} ({topology})"
                    result = test_policy(policy_class, policy_name, env_name, total_timesteps=10000, target_capacity=target_capacity)
                    all_results.append(result)
                    print(f"✓ {policy_name} on {env_name}: {result['avg_reward']:.1f} reward, {result['total_params']:,} params")
    
    # Create summary and plots
    create_summary_table(all_results)
    plot_comprehensive_results(all_results)
    
    # Save results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_test_results_2layers_capacity_matched_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = []
    for result in all_results:
        serializable_result = {k: v for k, v in result.items() if k not in ['episode_rewards', 'episode_lengths']}
        serializable_results.append(serializable_result)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("\nTest completed!")

if __name__ == "__main__":
    main() 