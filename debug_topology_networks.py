#!/usr/bin/env python3
"""
Debug Topology Networks: Simplified smoke test for debugging topology network implementation.
Focuses on verifying that actor and critic topology networks are implemented correctly.
Runs only one configuration across 4 topologies with masking strategy.
Designed for continuous debugging and layer exploration.
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
# DEBUG TOPOLOGY POLICY CLASS
# ============================================================================

class DebugTopologyPolicy(ActorCriticPolicy):
    """
    Debug Topology Policy with Masking Strategy.
    Focused on debugging and verifying topology network implementation.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, topology_type='fully_connected', hidden_size=64, num_layers=2, config=None, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Use centralized configuration
        if config is None:
            config = create_debug_config()
        
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
        
        print(f"\n🔍 DEBUG: Creating {topology_type} topology policy")
        print(f"   • Task dimensions: {self.task_input_dim}→{self.universal_action_dim}")
        print(f"   • Universal dimensions: {self.universal_input_dim}→{self.universal_output_dim}")
        print(f"   • Hidden size: {self.hidden_size}")
        print(f"   • Number of layers: {self.num_layers}")
        
        # Create separate topology networks for actor and critic
        print(f"   • Creating actor topology network...")
        self.actor_topology = self._create_topology_network('actor')
        print(f"   • Creating critic topology network...")
        self.critic_topology = self._create_topology_network('critic')
        
        # Debug: Check topology network types and properties
        print(f"   • Actor topology type: {type(self.actor_topology)}")
        print(f"   • Critic topology type: {type(self.critic_topology)}")
        
        # Get weight statistics
        actor_params = self._get_topology_params(self.actor_topology)
        critic_params = self._get_topology_params(self.critic_topology)
        total_params = actor_params + critic_params
        print(f"   • Actor topology parameters: {actor_params:,}")
        print(f"   • Critic topology parameters: {critic_params:,}")
        print(f"   • Total parameters: {total_params:,}")
        
        # Debug: Check if networks have forward methods
        print(f"   • Actor topology has forward: {hasattr(self.actor_topology, 'forward')}")
        print(f"   • Critic topology has forward: {hasattr(self.critic_topology, 'forward')}")
        
        # Debug: Check network structure
        self._debug_network_structure()
    
    def _create_topology_network(self, network_type):
        """Create topology network for actor or critic with capacity matching."""
        # Initialize capacity matching if enabled
        if self.config and self.config.get('use_capacity_matching', False):
            print(f"     🔧 Creating {network_type} topology ({self.topology_type}) with CAPACITY MATCHING")
            
            # Simple capacity matching approach
            reference_topology = self.config.get('capacity_matching_reference', 'small_world')
            base_size = self.config['network_sizes'][0]
            
            # Calculate target capacity based on small_world at base size
            # This is a simplified approach - in practice you'd use the full capacity mapping system
            if reference_topology == 'small_world':
                # Estimate small_world capacity: ~0.135 * size^1.92
                target_capacity = int(0.135 * base_size**1.92)
            else:
                target_capacity = 1000  # Default fallback
            
            print(f"       • Reference topology: {reference_topology}")
            print(f"       • Target capacity: {target_capacity:,} parameters")
            
            # Calculate matching size for this topology using empirical formulas
            if self.topology_type != reference_topology:
                if self.topology_type == 'modular':
                    # Modular: ~2.8 * size * (size/num_modules)
                    num_modules = self.config.get('modular_params', {}).get('num_modules', 4)
                    # Use a more conservative estimate for modular
                    matching_size = int((target_capacity / 2.8)**0.5)
                elif self.topology_type == 'hybrid':
                    # Hybrid: ~1.6 * size * (size/num_modules)
                    num_modules = self.config.get('hybrid_params', {}).get('num_modules', 4)
                    # Use a more conservative estimate for hybrid
                    matching_size = int((target_capacity / 1.6)**0.5)
                elif self.topology_type == 'fully_connected':
                    # Fully connected: ~2.05 * size^2
                    matching_size = int((target_capacity / 2.05)**0.5)
                else:
                    matching_size = base_size
                
                # Ensure minimum viable size
                min_size = max(30, self.universal_input_dim + self.hidden_size + self.universal_output_dim)
                matching_size = max(matching_size, min_size)
                
                print(f"       • Matching size for {self.topology_type}: {matching_size} nodes")
                total_size = matching_size
            else:
                total_size = base_size
                print(f"       • Using base size: {base_size} nodes")
        else:
            # Use original fixed size approach
            total_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
            print(f"     🔧 Creating {network_type} topology ({self.topology_type}) with FIXED SIZE")
            print(f"       • Total size: {total_size} nodes")
        
        print(f"       • Input nodes: {self.universal_input_dim}")
        print(f"       • Hidden nodes: {self.hidden_size}")
        print(f"       • Output nodes: {self.universal_output_dim}")
        
        # Create topology based on type
        if self.topology_type == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=self.num_layers,  # Only FC uses num_layers
                seed=42
            )
            print(f"       • FC layers: {self.num_layers}")
            layer_info = topology.get_layer_info()
            print(f"       • Layer sizes: {layer_info['layer_sizes']}")
        elif self.topology_type == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,  # Remove num_layers
                k=self.config.get('small_world_params', {}).get('k', 4),
                p=self.config.get('small_world_params', {}).get('p', 0.3),
                seed=42
            )
        elif self.topology_type == 'modular':
            topology = ModularTopology(
                size=total_size,  # Remove num_layers
                num_modules=self.config.get('modular_params', {}).get('num_modules', 4),
                inter_module_prob=self.config.get('modular_params', {}).get('inter_module_prob', 0.2),
                intra_module_prob=self.config.get('modular_params', {}).get('intra_module_prob', 0.8),
                seed=42
            )
        elif self.topology_type == 'hybrid':
            topology = HybridTopology(
                size=total_size,  # Remove num_layers
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
        
        # All topologies now return a single graph
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
        
        # Log capacity matching results
        if self.config and self.config.get('use_capacity_matching', False):
            actual_params = self._get_topology_params(network)
            print(f"       • Actual parameters: {actual_params:,}")
            if self.topology_type != reference_topology:
                divergence = abs(actual_params - target_capacity) / target_capacity * 100
                print(f"       • Capacity divergence: {divergence:.2f}%")
        
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
    
    def _debug_network_structure(self):
        """Debug the network structure."""
        print(f"   🔍 Debugging network structure:")
        
        # Check actor network
        print(f"     • Actor network attributes: {dir(self.actor_topology)}")
        if hasattr(self.actor_topology, 'graph'):
            print(f"     • Actor graph edges: {len(self.actor_topology.graph)}")
        if hasattr(self.actor_topology, 'input_nodes'):
            print(f"     • Actor input nodes: {self.actor_topology.input_nodes}")
        if hasattr(self.actor_topology, 'output_nodes'):
            print(f"     • Actor output nodes: {self.actor_topology.output_nodes}")
        
        # Check critic network
        print(f"     • Critic network attributes: {dir(self.critic_topology)}")
        if hasattr(self.critic_topology, 'graph'):
            print(f"     • Critic graph edges: {len(self.critic_topology.graph)}")
        if hasattr(self.critic_topology, 'input_nodes'):
            print(f"     • Critic input nodes: {self.critic_topology.input_nodes}")
        if hasattr(self.critic_topology, 'output_nodes'):
            print(f"     • Critic output nodes: {self.critic_topology.output_nodes}")
    
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

class EnhancedDebugCallback(BaseCallback):
    """Enhanced callback to track detailed training progress and network metrics."""
    
    def __init__(self, verbose=0, wandb_run=None, log_freq=100):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_count = 0
        self.wandb_run = wandb_run
        self.log_freq = log_freq
        self.episode_count = 0
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
            # Get detailed training info from the model
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
                # Extract metrics from the logger's name_to_value dict
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
                
                # Add gradient norm if available
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                    total_norm = 0
                    for p in self.model.policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    metrics["train/gradient_norm"] = total_norm
                
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
        """Log final training summary and create visualizations."""
        try:
            if len(self.training_metrics['episode_rewards']) > 0:
                # Create training curves
                self._create_training_curves()
                
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
    
    def _create_training_curves(self):
        """Create and log training curves."""
        try:
            # Create reward curve
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Reward curve
            rewards = self.training_metrics['episode_rewards']
            ax1.plot(rewards, alpha=0.6, color='blue')
            ax1.set_title('Training Reward Curve')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
            
            # Moving average
            if len(rewards) > 10:
                window = min(10, len(rewards) // 10)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(rewards)), moving_avg, color='red', linewidth=2, label=f'{window}-episode moving average')
                ax1.legend()
            
            # Reward distribution
            ax2.hist(rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Reward Distribution')
            ax2.set_xlabel('Reward')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to wandb image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            
            self.wandb_run.log({"training_curves": wandb.Image(img)})
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️  Error creating training curves: {e}")

def create_network_visualization(topology_network, topology_type, num_layers):
    """Create and return network structure visualization."""
    try:
        # Get the graph from the topology network
        if hasattr(topology_network, 'topology'):
            G = topology_network.topology
        else:
            return None
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Network structure plot
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Color nodes by layer (if multi-layer)
        # Handle num_layers as list or int
        actual_layers = num_layers[0] if isinstance(num_layers, list) else num_layers
        
        if actual_layers > 1:
            node_colors = []
            nodes_per_layer = len(G.nodes()) // actual_layers
            for node in G.nodes():
                layer = node // nodes_per_layer
                node_colors.append(layer)
            cmap = plt.cm.viridis
        else:
            node_colors = 'lightblue'
            cmap = None
        
        # Draw the network
        nx.draw(G, pos, ax=ax1, 
                node_color=node_colors, 
                cmap=cmap,
                node_size=100, 
                with_labels=True, 
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                alpha=0.7)
        
        ax1.set_title(f'{topology_type.replace("_", " ").title()} Network Structure\n({len(G.nodes())} nodes, {len(G.edges())} edges)')
        
        # Network metrics
        metrics = {
            'Nodes': len(G.nodes()),
            'Edges': len(G.edges()),
            'Density': nx.density(G),
            'Avg Clustering': nx.average_clustering(G.to_undirected()) if not G.is_directed() else nx.average_clustering(G),
            'Avg Path Length': nx.average_shortest_path_length(G.to_undirected()) if not G.is_directed() else 'N/A (Directed)',
        }
        
        # Create metrics table
        ax2.axis('off')
        table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
        table = ax2.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax2.set_title('Network Metrics')
        
        plt.tight_layout()
        
        # Convert to wandb image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        plt.close()
        return img
        
    except Exception as e:
        print(f"   ⚠️  Error creating network visualization: {e}")
        return None

def create_connection_heatmap(topology_network, topology_type):
    """Create connection strength heatmap visualization."""
    try:
        # Get the graph from the topology network
        if hasattr(topology_network, 'topology'):
            G = topology_network.topology
        else:
            return None
        
        # Create adjacency matrix
        adj_matrix = nx.adjacency_matrix(G).todense()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap with custom colormap
        sns.heatmap(adj_matrix, 
                   cmap='Blues', 
                   cbar_kws={'label': 'Connection Strength'},
                   ax=ax,
                   square=True)
        
        ax.set_title(f'{topology_type.replace("_", " ").title()} Connection Matrix')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('Node Index')
        
        plt.tight_layout()
        
        # Convert to wandb image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        plt.close()
        return img
        
    except Exception as e:
        print(f"   ⚠️  Error creating connection heatmap: {e}")
        return None

def create_layer_analysis_visualization(topology_network, topology_type, num_layers):
    """Create detailed layer-by-layer analysis visualization."""
    try:
        # Get the graph from the topology network
        if hasattr(topology_network, 'topology'):
            G = topology_network.topology
        else:
            return None
        
        # Calculate layer information
        total_nodes = len(G.nodes())
        # Handle num_layers as list or int
        actual_layers = num_layers[0] if isinstance(num_layers, list) else num_layers
        nodes_per_layer = total_nodes // actual_layers if actual_layers > 1 else total_nodes
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Layer connectivity matrix
        ax1 = plt.subplot(2, 3, 1)
        layer_connectivity = np.zeros((actual_layers, actual_layers))
        
        for i in range(actual_layers):
            for j in range(actual_layers):
                layer_i_start = i * nodes_per_layer
                layer_i_end = layer_i_start + nodes_per_layer
                layer_j_start = j * nodes_per_layer
                layer_j_end = layer_j_start + nodes_per_layer
                
                # Count connections between layers
                connections = 0
                for edge in G.edges():
                    if (layer_i_start <= edge[0] < layer_i_end and 
                        layer_j_start <= edge[1] < layer_j_end):
                        connections += 1
                
                layer_connectivity[i, j] = connections
        
        sns.heatmap(layer_connectivity, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Inter-Layer Connections')
        ax1.set_xlabel('Target Layer')
        ax1.set_ylabel('Source Layer')
        
        # 2. Node degree distribution
        ax2 = plt.subplot(2, 3, 2)
        degrees = [G.degree(node) for node in G.nodes()]
        ax2.hist(degrees, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Node Degree Distribution')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Layer-wise degree analysis
        ax3 = plt.subplot(2, 3, 3)
        layer_degrees = []
        layer_labels = []
        
        for layer in range(actual_layers):
            layer_start = layer * nodes_per_layer
            layer_end = layer_start + nodes_per_layer
            layer_nodes = list(range(layer_start, layer_end))
            layer_node_degrees = [G.degree(node) for node in layer_nodes]
            layer_degrees.append(layer_node_degrees)
            layer_labels.append(f'Layer {layer}')
        
        ax3.boxplot(layer_degrees, labels=layer_labels)
        ax3.set_title('Degree Distribution by Layer')
        ax3.set_ylabel('Degree')
        ax3.grid(True, alpha=0.3)
        
        # 4. Clustering coefficient by layer
        ax4 = plt.subplot(2, 3, 4)
        clustering_by_layer = []
        
        for layer in range(actual_layers):
            layer_start = layer * nodes_per_layer
            layer_end = layer_start + nodes_per_layer
            layer_nodes = list(range(layer_start, layer_end))
            
            # Create subgraph for this layer
            layer_subgraph = G.subgraph(layer_nodes)
            if len(layer_subgraph.nodes()) > 2:
                clustering = nx.average_clustering(layer_subgraph.to_undirected())
            else:
                clustering = 0
            clustering_by_layer.append(clustering)
        
        ax4.bar(range(actual_layers), clustering_by_layer, color='lightgreen')
        ax4.set_title('Average Clustering by Layer')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Clustering Coefficient')
        ax4.set_xticks(range(actual_layers))
        ax4.set_xticklabels([f'Layer {i}' for i in range(actual_layers)])
        ax4.grid(True, alpha=0.3)
        
        # 5. Network metrics summary
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        metrics = {
            'Total Nodes': len(G.nodes()),
            'Total Edges': len(G.edges()),
            'Network Density': f"{nx.density(G):.4f}",
            'Avg Clustering': f"{nx.average_clustering(G.to_undirected()):.4f}",
            'Avg Path Length': f"{nx.average_shortest_path_length(G.to_undirected()):.4f}" if not G.is_directed() else "N/A",
            'Diameter': f"{nx.diameter(G.to_undirected()):.1f}" if not G.is_directed() else "N/A",
        }
        
        table_data = [[k, v] for k, v in metrics.items()]
        table = ax5.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#2196F3')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax5.set_title('Network Metrics Summary', fontsize=12, fontweight='bold')
        
        # 6. Connection strength distribution
        ax6 = plt.subplot(2, 3, 6)
        adj_matrix = nx.adjacency_matrix(G).todense()
        connection_strengths = adj_matrix.flatten()
        ax6.hist(connection_strengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax6.set_title('Connection Strength Distribution')
        ax6.set_xlabel('Connection Strength')
        ax6.set_ylabel('Frequency')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'{topology_type.replace("_", " ").title()} Network Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Convert to wandb image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        
        plt.close()
        return img
        
    except Exception as e:
        print(f"   ⚠️  Error creating layer analysis visualization: {e}")
        return None

def create_connection_list(topology_network, topology_type, network_type, num_layers):
    """Create and return a detailed connection list for the topology network."""
    try:
        # Get the graph from the topology network
        if hasattr(topology_network, 'topology'):
            G = topology_network.topology
        else:
            return None
        
        # Calculate layer information
        total_nodes = len(G.nodes())
        nodes_per_layer = total_nodes // num_layers if num_layers > 1 else total_nodes
        
        # Create connection list
        connection_data = {
            'topology_type': topology_type,
            'network_type': network_type,  # 'actor' or 'critic'
            'num_layers': num_layers,
            'total_nodes': total_nodes,
            'nodes_per_layer': nodes_per_layer,
            'total_edges': len(G.edges()),
            'connections': [],
            'layer_info': {},
            'node_degrees': {},
            'edge_weights': {}
        }
        
        # Add layer information
        for layer in range(num_layers):
            layer_start = layer * nodes_per_layer
            layer_end = layer_start + nodes_per_layer
            layer_nodes = list(range(layer_start, layer_end))
            
            connection_data['layer_info'][f'layer_{layer}'] = {
                'start_node': layer_start,
                'end_node': layer_end - 1,
                'nodes': layer_nodes,
                'num_nodes': len(layer_nodes)
            }
        
        # Add all connections
        for edge in G.edges():
            source, target = edge
            
            # Determine source and target layers
            source_layer = source // nodes_per_layer if num_layers > 1 else 0
            target_layer = target // nodes_per_layer if num_layers > 1 else 0
            
            connection = {
                'source_node': int(source),
                'target_node': int(target),
                'source_layer': source_layer,
                'target_layer': target_layer,
                'connection_type': 'intra_layer' if source_layer == target_layer else 'inter_layer',
                'edge_weight': G[source][target].get('weight', 1.0) if G.is_directed() else 1.0
            }
            connection_data['connections'].append(connection)
        
        # Add node degrees
        for node in G.nodes():
            connection_data['node_degrees'][int(node)] = {
                'in_degree': G.in_degree(node) if G.is_directed() else G.degree(node),
                'out_degree': G.out_degree(node) if G.is_directed() else G.degree(node),
                'total_degree': G.degree(node),
                'layer': node // nodes_per_layer if num_layers > 1 else 0
            }
        
        # Add edge weight information
        if G.is_directed():
            for edge in G.edges():
                source, target = edge
                edge_key = f"{source}_{target}"
                connection_data['edge_weights'][edge_key] = G[source][target].get('weight', 1.0)
        
        return connection_data
        
    except Exception as e:
        print(f"   ⚠️  Error creating connection list: {e}")
        return None

def create_simple_connection_list(topology_network, topology_type, network_type, num_layers):
    """Create a simple list of node-to-node connections for easy analysis."""
    try:
        # Get the graph from the topology network
        if hasattr(topology_network, 'topology'):
            G = topology_network.topology
        else:
            return None
        
        # Calculate layer information
        total_nodes = len(G.nodes())
        # Handle num_layers as list or int
        actual_layers = num_layers[0] if isinstance(num_layers, list) else num_layers
        nodes_per_layer = total_nodes // actual_layers if actual_layers > 1 else total_nodes
        
        # Create simple connection list
        connections = []
        
        for edge in G.edges():
            source, target = edge
            
            # Determine source and target layers
            source_layer = source // nodes_per_layer if actual_layers > 1 else 0
            target_layer = target // nodes_per_layer if actual_layers > 1 else 0
            
            connection = {
                'source_node': int(source),
                'target_node': int(target),
                'source_layer': source_layer,
                'target_layer': target_layer,
                'connection_type': 'intra_layer' if source_layer == target_layer else 'inter_layer'
            }
            connections.append(connection)
        
        # Sort by source node, then target node for easy reading
        connections.sort(key=lambda x: (x['source_node'], x['target_node']))
        
        return {
            'topology_type': topology_type,
            'network_type': network_type,
            'num_layers': actual_layers,
            'total_nodes': total_nodes,
            'nodes_per_layer': nodes_per_layer,
            'total_connections': len(connections),
            'connections': connections
        }
        
    except Exception as e:
        print(f"   ⚠️  Error creating simple connection list: {e}")
        return None

def save_simple_connection_files(actor_connections, critic_connections, topology_type, num_layers, results_dir):
    """Save simple connection lists in multiple formats for easy analysis."""
    try:
        # Create connections directory
        connections_dir = os.path.join(results_dir, 'connections')
        os.makedirs(connections_dir, exist_ok=True)
        
        # Save as JSON
        if actor_connections:
            actor_json_file = os.path.join(connections_dir, f'{topology_type}_actor_connections_simple_layers_{num_layers}.json')
            with open(actor_json_file, 'w') as f:
                json.dump(actor_connections, f, indent=2)
            print(f"   ✅ Actor connections (JSON) saved to: {actor_json_file}")
        
        if critic_connections:
            critic_json_file = os.path.join(connections_dir, f'{topology_type}_critic_connections_simple_layers_{num_layers}.json')
            with open(critic_json_file, 'w') as f:
                json.dump(critic_connections, f, indent=2)
            print(f"   ✅ Critic connections (JSON) saved to: {critic_json_file}")
        
        # Save as CSV for easy analysis
        if actor_connections:
            actor_csv_file = os.path.join(connections_dir, f'{topology_type}_actor_connections_layers_{num_layers}.csv')
            with open(actor_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['source_node', 'target_node', 'source_layer', 'target_layer', 'connection_type'])
                for conn in actor_connections['connections']:
                    writer.writerow([
                        conn['source_node'],
                        conn['target_node'],
                        conn['source_layer'],
                        conn['target_layer'],
                        conn['connection_type']
                    ])
            print(f"   ✅ Actor connections (CSV) saved to: {actor_csv_file}")
        
        if critic_connections:
            critic_csv_file = os.path.join(connections_dir, f'{topology_type}_critic_connections_layers_{num_layers}.csv')
            with open(critic_csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['source_node', 'target_node', 'source_layer', 'target_layer', 'connection_type'])
                for conn in critic_connections['connections']:
                    writer.writerow([
                        conn['source_node'],
                        conn['target_node'],
                        conn['source_layer'],
                        conn['target_layer'],
                        conn['connection_type']
                    ])
            print(f"   ✅ Critic connections (CSV) saved to: {critic_csv_file}")
        
        # Save as simple text file for quick viewing
        if actor_connections:
            actor_txt_file = os.path.join(connections_dir, f'{topology_type}_actor_connections_layers_{num_layers}.txt')
            with open(actor_txt_file, 'w') as f:
                f.write(f"Topology: {topology_type}\n")
                f.write(f"Network: Actor\n")
                f.write(f"Layers: {num_layers}\n")
                f.write(f"Total Nodes: {actor_connections['total_nodes']}\n")
                f.write(f"Total Connections: {actor_connections['total_connections']}\n")
                f.write(f"Nodes per Layer: {actor_connections['nodes_per_layer']}\n")
                f.write("=" * 50 + "\n\n")
                f.write("CONNECTIONS (source -> target [layer]):\n")
                f.write("-" * 30 + "\n")
                
                for conn in actor_connections['connections']:
                    f.write(f"{conn['source_node']:3d} -> {conn['target_node']:3d} [{conn['source_layer']}->{conn['target_layer']}] {conn['connection_type']}\n")
            print(f"   ✅ Actor connections (TXT) saved to: {actor_txt_file}")
        
        if critic_connections:
            critic_txt_file = os.path.join(connections_dir, f'{topology_type}_critic_connections_layers_{num_layers}.txt')
            with open(critic_txt_file, 'w') as f:
                f.write(f"Topology: {topology_type}\n")
                f.write(f"Network: Critic\n")
                f.write(f"Layers: {num_layers}\n")
                f.write(f"Total Nodes: {critic_connections['total_nodes']}\n")
                f.write(f"Total Connections: {critic_connections['total_connections']}\n")
                f.write(f"Nodes per Layer: {critic_connections['nodes_per_layer']}\n")
                f.write("=" * 50 + "\n\n")
                f.write("CONNECTIONS (source -> target [layer]):\n")
                f.write("-" * 30 + "\n")
                
                for conn in critic_connections['connections']:
                    f.write(f"{conn['source_node']:3d} -> {conn['target_node']:3d} [{conn['source_layer']}->{conn['target_layer']}] {conn['connection_type']}\n")
            print(f"   ✅ Critic connections (TXT) saved to: {critic_txt_file}")
        
        return connections_dir
        
    except Exception as e:
        print(f"   ⚠️  Error saving simple connection files: {e}")
        return None

def save_connection_lists(actor_connections, critic_connections, topology_type, num_layers, results_dir):
    """Save connection lists to JSON files."""
    try:
        # Create connections directory
        connections_dir = os.path.join(results_dir, 'connections')
        os.makedirs(connections_dir, exist_ok=True)
        
        # Save actor connections
        if actor_connections:
            actor_file = os.path.join(connections_dir, f'{topology_type}_actor_connections_layers_{num_layers}.json')
            with open(actor_file, 'w') as f:
                json.dump(actor_connections, f, indent=2)
            print(f"   ✅ Actor connections saved to: {actor_file}")
        
        # Save critic connections
        if critic_connections:
            critic_file = os.path.join(connections_dir, f'{topology_type}_critic_connections_layers_{num_layers}.json')
            with open(critic_file, 'w') as f:
                json.dump(critic_connections, f, indent=2)
            print(f"   ✅ Critic connections saved to: {critic_file}")
        
        # Create summary file
        summary_data = {
            'topology_type': topology_type,
            'num_layers': num_layers,
            'actor_summary': {
                'total_nodes': actor_connections['total_nodes'] if actor_connections else 0,
                'total_edges': actor_connections['total_edges'] if actor_connections else 0,
                'intra_layer_connections': len([c for c in actor_connections['connections'] if c['connection_type'] == 'intra_layer']) if actor_connections else 0,
                'inter_layer_connections': len([c for c in actor_connections['connections'] if c['connection_type'] == 'inter_layer']) if actor_connections else 0,
            },
            'critic_summary': {
                'total_nodes': critic_connections['total_nodes'] if critic_connections else 0,
                'total_edges': critic_connections['total_edges'] if critic_connections else 0,
                'intra_layer_connections': len([c for c in critic_connections['connections'] if c['connection_type'] == 'intra_layer']) if critic_connections else 0,
                'inter_layer_connections': len([c for c in critic_connections['connections'] if c['connection_type'] == 'inter_layer']) if critic_connections else 0,
            }
        }
        
        summary_file = os.path.join(connections_dir, f'{topology_type}_connection_summary_layers_{num_layers}.json')
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"   ✅ Connection summary saved to: {summary_file}")
        
        return connections_dir
        
    except Exception as e:
        print(f"   ⚠️  Error saving connection lists: {e}")
        return None

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def create_debug_config():
    """Create configuration for debug test with capacity matching."""
    config = {
        # ============================================================================
        # EXPERIMENT PARAMETERS
        # ============================================================================
        'tasks': ['CartPole-v1'],  # Only one task for debugging
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected_2layers', 'fully_connected_3layers'],
        'total_timesteps': 10000,  # Short for debugging
        'n_eval_episodes': 10,     # Few episodes for debugging
        
        # ============================================================================
        # UNIVERSAL TOPOLOGY PARAMETERS
        # ============================================================================
        'universal_input_dim': 6,
        'universal_output_dim': 3,
        'universal_action_dim': 3,
        'hidden_size': 64,  # Fixed size for debugging
        'num_layers': 2,    # Start with 2 layers, can be modified
        
        # ============================================================================
        # PPO TRAINING PARAMETERS
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
        },
        
        # ============================================================================
        # CAPACITY MATCHING PARAMETERS
        # ============================================================================
        'use_capacity_matching': True,
        'capacity_matching_reference': 'small_world',  # Use small_world as reference
        'network_sizes': [64],  # Base network size
        'network_types': ['ffn'],
        'num_layers': [2],  # List of layer configurations
        'num_io_nodes': 4,  # Number of input/output nodes
        'experiment_types': ['same_size', 'match_small_world', 'match_fully_connected'],  # Capacity matching experiment type
        
        # Topology-specific parameters
        'small_world_params': {
            'k': 4,
            'p': 0.3,
            'inter_layer_prob': 0.5
        },
        'modular_params': {
            'num_modules': 4,
            'inter_module_prob': 0.2,
            'intra_module_prob': 0.8,
            'inter_layer_prob': 0.5
        },
        'hybrid_params': {
            'num_modules': 4,
            'k': 4,
            'p': 0.3,
            'inter_module_prob': 0.2,
            'inter_layer_prob': 0.5
        },
        'fully_connected_params': {
            'inter_layer_prob': 1.0,
            'intra_layer_prob': 1.0
        },
        
        # Network parameters
        'network_params': {
            'ffn': {
                'activation': 'relu',
                'dropout': 0.0
            }
        },
        
        # Parameter budget configuration
        'parameter_budget': {
            'enabled': True,
            'budget_type': 'weights',  # 'weights' or 'edges'
            'padding_strategy': 'random'  # 'random' or 'zero'
        }
    }
    return config

def debug_topology_policy(policy_class, topology_type, config, num_layers=2):
    """
    Debug a single topology policy to verify implementation.
    """
    print(f"\n{'='*80}")
    print(f"🔍 DEBUGGING: {topology_type.upper()} TOPOLOGY")
    print(f"{'='*80}")
    
    # Initialize wandb run for this topology
    run_name = f"{topology_type}_layers_{num_layers}"
    try:
        wandb_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="cross-task",  # Use existing project
            name=run_name,
            config={
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": config['hidden_size'],
                "total_timesteps": config['total_timesteps'],
                "n_eval_episodes": config['n_eval_episodes'],
                "task": config['tasks'][0],
                "ppo_params": config['ppo_params'],
                "universal_input_dim": config['universal_input_dim'],
                "universal_output_dim": config['universal_output_dim'],
                "universal_action_dim": config['universal_action_dim'],
            },
            tags=[topology_type, f"layers_{num_layers}", "debug_test"],
            reinit=True
        )
        topology_wandb_enabled = True
    except Exception as e:
        print(f"   ⚠️  WandB logging disabled for {topology_type}: {e}")
        wandb_run = None
        topology_wandb_enabled = False
    
    # Create environment
    train_task = config['tasks'][0]
    train_env = DummyVecEnv([make_env(train_task)])
    
    print(f"📋 Configuration:")
    print(f"   • Task: {train_task}")
    print(f"   • Topology: {topology_type}")
    print(f"   • Hidden size: {config['hidden_size']}")
    print(f"   • Number of layers: {num_layers}")
    print(f"   • Training timesteps: {config['total_timesteps']:,}")
    print(f"   • WandB Run: {run_name}")
    
    # Create policy with debugging
    SpecificPolicyClass = lambda obs_space, action_space, lr_schedule, **kwargs: policy_class(
        obs_space, action_space, lr_schedule, 
        topology_type=topology_type, 
        hidden_size=config['hidden_size'],
        num_layers=num_layers,
        config=config,
        **kwargs
    )
    
    # Initialize model with centralized PPO parameters
    ppo_params = config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env,
        verbose=1,
        tensorboard_log=f"./logs/debug_{topology_type}/",
        **ppo_params
    )
    
    # Create network visualizations and log to wandb
    if topology_wandb_enabled and wandb_run is not None:
        try:
            print(f"   📊 Creating network visualizations...")
            
            # Get the policy to access topology networks
            policy = model.policy
            
            # Create actor network visualization
            if hasattr(policy, 'actor_topology'):
                actor_viz = create_network_visualization(policy.actor_topology, topology_type, num_layers)
                if actor_viz:
                    wandb_run.log({"actor_network_structure": wandb.Image(actor_viz)})
                    print(f"   ✅ Actor network visualization logged")
                
                actor_heatmap = create_connection_heatmap(policy.actor_topology, topology_type)
                if actor_heatmap:
                    wandb_run.log({"actor_connection_matrix": wandb.Image(actor_heatmap)})
                    print(f"   ✅ Actor connection heatmap logged")
            
            # Create critic network visualization
            if hasattr(policy, 'critic_topology'):
                critic_viz = create_network_visualization(policy.critic_topology, topology_type, num_layers)
                if critic_viz:
                    wandb_run.log({"critic_network_structure": wandb.Image(critic_viz)})
                    print(f"   ✅ Critic network visualization logged")
                
                critic_heatmap = create_connection_heatmap(policy.critic_topology, topology_type)
                if critic_heatmap:
                    wandb_run.log({"critic_connection_matrix": wandb.Image(critic_heatmap)})
                    print(f"   ✅ Critic connection heatmap logged")
                
                # Create detailed layer analysis
                actor_layer_analysis = create_layer_analysis_visualization(policy.actor_topology, topology_type, num_layers)
                if actor_layer_analysis:
                    wandb_run.log({"actor_layer_analysis": wandb.Image(actor_layer_analysis)})
                    print(f"   ✅ Actor layer analysis logged")
                
                critic_layer_analysis = create_layer_analysis_visualization(policy.critic_topology, topology_type, num_layers)
                if critic_layer_analysis:
                    wandb_run.log({"critic_layer_analysis": wandb.Image(critic_layer_analysis)})
                    print(f"   ✅ Critic layer analysis logged")
            
            # Create connection lists
            print(f"   📊 Creating connection lists...")
            actor_connections = create_simple_connection_list(policy.actor_topology, topology_type, 'actor', num_layers)
            critic_connections = create_simple_connection_list(policy.critic_topology, topology_type, 'critic', num_layers)
            
            if actor_connections and critic_connections:
                print(f"   ✅ Connection lists created successfully")
                print(f"      • Actor: {actor_connections['total_connections']} connections, {actor_connections['total_nodes']} nodes")
                print(f"      • Critic: {critic_connections['total_connections']} connections, {critic_connections['total_nodes']} nodes")
                
                # Save connection lists to files
                results_dir = f"results/debug_topology_networks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(results_dir, exist_ok=True)
                
                connections_dir = save_simple_connection_files(actor_connections, critic_connections, topology_type, num_layers, results_dir)
                if connections_dir:
                    print(f"   ✅ Connection files saved to: {connections_dir}")
                    print(f"      • JSON files for programmatic analysis")
                    print(f"      • CSV files for spreadsheet analysis") 
                    print(f"      • TXT files for human reading")
            
            # Log network statistics
            if hasattr(policy, 'actor_topology') and hasattr(policy, 'critic_topology'):
                actor_params = policy._get_topology_params(policy.actor_topology)
                critic_params = policy._get_topology_params(policy.critic_topology)
                
                # Calculate additional network metrics
                actor_G = policy.actor_topology.topology
                critic_G = policy.critic_topology.topology
                
                wandb_run.log({
                    "network/actor_parameters": actor_params,
                    "network/critic_parameters": critic_params,
                    "network/total_parameters": actor_params + critic_params,
                    "network/actor_nodes": len(actor_G.nodes()),
                    "network/critic_nodes": len(critic_G.nodes()),
                    "network/actor_edges": len(actor_G.edges()),
                    "network/critic_edges": len(critic_G.edges()),
                    "network/actor_density": nx.density(actor_G),
                    "network/critic_density": nx.density(critic_G),
                    "network/actor_avg_clustering": nx.average_clustering(actor_G.to_undirected()),
                    "network/critic_avg_clustering": nx.average_clustering(critic_G.to_undirected()),
                    "network/actor_avg_degree": sum(dict(actor_G.degree()).values()) / len(actor_G.nodes()),
                    "network/critic_avg_degree": sum(dict(critic_G.degree()).values()) / len(critic_G.nodes()),
                })
                
                # Log connection list data to wandb
                if actor_connections and critic_connections:
                    # Create connection summary for wandb
                    connection_summary = {
                        "connections/actor_total_connections": actor_connections['total_connections'],
                        "connections/critic_total_connections": critic_connections['total_connections'],
                        "connections/actor_intra_layer": len([c for c in actor_connections['connections'] if c['connection_type'] == 'intra_layer']),
                        "connections/critic_intra_layer": len([c for c in critic_connections['connections'] if c['connection_type'] == 'intra_layer']),
                        "connections/actor_inter_layer": len([c for c in actor_connections['connections'] if c['connection_type'] == 'inter_layer']),
                        "connections/critic_inter_layer": len([c for c in critic_connections['connections'] if c['connection_type'] == 'inter_layer']),
                        "connections/actor_nodes_per_layer": actor_connections['nodes_per_layer'],
                        "connections/critic_nodes_per_layer": critic_connections['nodes_per_layer'],
                    }
                    wandb_run.log(connection_summary)
                    
                    # Create connection tables for wandb
                    if len(actor_connections['connections']) > 0:
                        actor_connection_table = wandb.Table(columns=["Source", "Target", "Source_Layer", "Target_Layer", "Type"])
                        for conn in actor_connections['connections'][:100]:  # Limit to first 100 connections
                            actor_connection_table.add_data(
                                conn['source_node'], 
                                conn['target_node'], 
                                conn['source_layer'], 
                                conn['target_layer'], 
                                conn['connection_type']
                            )
                        wandb_run.log({"actor_connections_table": actor_connection_table})
                    
                    if len(critic_connections['connections']) > 0:
                        critic_connection_table = wandb.Table(columns=["Source", "Target", "Source_Layer", "Target_Layer", "Type"])
                        for conn in critic_connections['connections'][:100]:  # Limit to first 100 connections
                            critic_connection_table.add_data(
                                conn['source_node'], 
                                conn['target_node'], 
                                conn['source_layer'], 
                                conn['target_layer'], 
                                conn['connection_type']
                            )
                        wandb_run.log({"critic_connections_table": critic_connection_table})
                
                print(f"   ✅ Network statistics logged")
                
        except Exception as e:
            print(f"   ⚠️  Error creating network visualizations: {e}")
    
    # Setup enhanced callback with wandb
    callback = EnhancedDebugCallback(wandb_run=wandb_run if topology_wandb_enabled else None, log_freq=500)
    
    # Train the model with progress bar
    print(f"\n🎯 Training Phase:")
    print(f"   • Training for {config['total_timesteps']} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Test the model
    print(f"\n🧪 Testing Phase:")
    print(f"   • Evaluating on {train_task}...")
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
            
            # Create a summary table for wandb
            results_table = wandb.Table(columns=["Metric", "Value"])
            results_table.add_data("Topology Type", topology_type)
            results_table.add_data("Number of Layers", str(num_layers))
            results_table.add_data("Hidden Size", str(config['hidden_size']))
            results_table.add_data("Mean Reward", f"{mean_reward:.2f}")
            results_table.add_data("Std Reward", f"{std_reward:.2f}")
            results_table.add_data("Training Time (s)", f"{training_time:.2f}")
            results_table.add_data("Timesteps/sec", f"{config['total_timesteps'] / training_time:.2f}")
            results_table.add_data("Performance Score", f"{mean_reward / training_time:.4f}")
            results_table.add_data("Total Timesteps", str(config['total_timesteps']))
            
            wandb_run.log({"results_summary": results_table})
            
            # Finish wandb run
            wandb_run.finish()
        except Exception as e:
            print(f"   ⚠️  Error logging to WandB: {e}")
    
    train_env.close()
    
    return {
        'topology_type': topology_type,
        'num_layers': num_layers,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'training_time': training_time
    }

def evaluate_model(model, env, n_eval_episodes=3):
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

def main():
    """Main function to run debug test."""
    print("🔍 Debug Topology Networks: Simplified smoke test for debugging")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize main wandb run for the overall experiment
    experiment_name = f"debug_topology_networks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        main_wandb_run = wandb.init(
            entity="katko",
            project="topology-playground",  # Use existing project
            name=experiment_name,
            config={
                "experiment_type": "debug_topology_networks",
                "description": "Debug test for topology network implementation verification",
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            },
            tags=["debug_test", "topology_verification"],
            reinit=True
        )
        wandb_enabled = True
        print(f"   ✅ WandB logging enabled: {experiment_name}")
    except Exception as e:
        print(f"   ⚠️  WandB logging disabled: {e}")
        main_wandb_run = None
        wandb_enabled = False
    
    # Create configuration
    config = create_debug_config()
    
    # Define topologies
    topologies = config['topologies']
    
    print(f"\n📋 Debug Configuration:")
    print(f"   • Task: {config['tasks'][0]}")
    print(f"   • Topologies: {', '.join(topologies)}")
    print(f"   • Hidden size: {config['hidden_size']}")
    print(f"   • Number of layers: {config['num_layers']}")
    print(f"   • Training timesteps: {config['total_timesteps']:,}")
    print(f"   • Evaluation episodes: {config['n_eval_episodes']}")
    print(f"   • Main WandB Run: {experiment_name}")
    
    # Store all results
    all_results = []
    
    # Debug each topology
    for topology in topologies:
        print(f"\n{'='*80}")
        print(f"🔍 DEBUGGING TOPOLOGY: {topology.upper()}")
        print(f"{'='*80}")
        
        # Determine number of layers based on topology type
        if topology == 'fully_connected_2layers':
            actual_topology = 'fully_connected'
            num_layers = 2
        elif topology == 'fully_connected_3layers':
            actual_topology = 'fully_connected'
            num_layers = 3
        else:
            actual_topology = topology
            num_layers = config['num_layers']  # Default for non-FC topologies
        
        # Debug the topology policy
        result = debug_topology_policy(
            DebugTopologyPolicy,
            actual_topology,
            config,
            num_layers=num_layers
        )
        
        # Update the result to reflect the original topology name for display
        result['topology_type'] = topology
        all_results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/debug_topology_networks_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(f"{results_dir}/debug_results.csv", index=False)
    
    # Save configuration
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Log final results to main wandb run
    if wandb_enabled and main_wandb_run is not None and not df.empty:
        try:
            # Create comparison table
            comparison_table = wandb.Table(columns=["Topology", "Layers", "Mean Reward", "Std Reward", "Training Time"])
            for _, row in df.iterrows():
                comparison_table.add_data(
                    row['topology_type'],
                    row['num_layers'],
                    f"{row['mean_reward']:.2f}",
                    f"{row['std_reward']:.2f}",
                    f"{row['training_time']:.2f}s"
                )
            
            # Find best performing topology
            best_result = df.loc[df['mean_reward'].idxmax()]
            
            # Log summary metrics
            main_wandb_run.log({
                "experiment/best_topology": best_result['topology_type'],
                "experiment/best_mean_reward": best_result['mean_reward'],
                "experiment/best_std_reward": best_result['std_reward'],
                "experiment/total_topologies_tested": len(all_results),
                "experiment/comparison_table": comparison_table,
            })
            
            # Log individual topology results
            for _, row in df.iterrows():
                main_wandb_run.log({
                    f"topology/{row['topology_type']}/mean_reward": row['mean_reward'],
                    f"topology/{row['topology_type']}/std_reward": row['std_reward'],
                    f"topology/{row['topology_type']}/training_time": row['training_time'],
                })
            
            # Finish main wandb run
            main_wandb_run.finish()
        except Exception as e:
            print(f"   ⚠️  Error logging final results to WandB: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🎉 DEBUG TEST COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Results Summary:")
    print(f"   • Total Topologies Tested: {len(all_results)}")
    print(f"   • Results saved to: {results_dir}")
    print(f"   • CSV file: debug_results.csv")
    print(f"   • Config file: config.json")
    print(f"   • WandB Project: topologies-smoke-test")
    print(f"   • Main Run: {experiment_name}")
    
    # Print results
    if not df.empty:
        print(f"\n📈 Results by Topology:")
        for _, row in df.iterrows():
            print(f"   • {row['topology_type']}: {row['mean_reward']:.2f} ± {row['std_reward']:.2f} (layers: {row['num_layers']})")
        
        print(f"\n🔍 Analysis:")
        best_result = df.loc[df['mean_reward'].idxmax()]
        print(f"   • Best performing topology: {best_result['topology_type']} ({best_result['mean_reward']:.2f})")
        print(f"   • All topologies completed training successfully")
        print(f"   • Ready for layer exploration and further debugging")
    
    print(f"\n✅ Debug test completed successfully!")
    print(f"💡 Next steps:")
    print(f"   • Modify num_layers in create_debug_config() to explore different layer counts")
    print(f"   • Check the detailed debug output above for topology network verification")
    print(f"   • Examine the saved results for performance comparison")
    print(f"   • View results in WandB dashboard: https://wandb.ai/katko/topologies-smoke-test")

if __name__ == "__main__":
    main() 