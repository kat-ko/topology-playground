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
        """Create topology network for actor or critic."""
        print(f"   • Creating {network_type} topology network...")
        
        # Use the hidden_size that was passed to the policy (already capacity-matched)
        # This follows the working implementation pattern exactly
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
    
    def _debug_network_structure(self):
        """Debug the network structure."""
        print(f"   🔍 Debugging network structure:")
        
        # Check actor network
        # print(f"     • Actor network attributes: {dir(self.actor_topology)}")
        if hasattr(self.actor_topology, 'graph'):
            print(f"     • Actor graph edges: {len(self.actor_topology.graph)}")
        if hasattr(self.actor_topology, 'input_nodes'):
            print(f"     • Actor input nodes: {self.actor_topology.input_nodes}")
        if hasattr(self.actor_topology, 'output_nodes'):
            print(f"     • Actor output nodes: {self.actor_topology.output_nodes}")
        
        # Check critic network
        # print(f"     • Critic network attributes: {dir(self.critic_topology)}")
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
                    "train/episode_count": self.episode_count,
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
                
                                    # Add network-specific metrics if available
                    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
                        actor_params = self.model.policy._get_topology_params(self.model.policy.actor_topology)
                        critic_params = self.model.policy._get_topology_params(self.model.policy.critic_topology)
                        metrics.update({
                            "network/actor_parameters": actor_params,
                            "network/critic_parameters": critic_params,
                            "network/total_parameters": actor_params + critic_params,
                        })
                    
                    # Only log if this is the current step to avoid step ordering issues in double-task training
                    # This prevents logging historical data from previous training phases
                    if self.step_count % self.log_freq == 0:
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
                    
                    # Calculate additional statistics
                    positive_rewards = episode_rewards[episode_rewards > 0]
                    negative_rewards = episode_rewards[episode_rewards < 0]
                    zero_rewards = episode_rewards[episode_rewards == 0]
                    
                    metrics = {
                        "rollout/mean_reward": np.mean(episode_rewards),
                        "rollout/std_reward": np.std(episode_rewards),
                        "rollout/max_reward": np.max(episode_rewards),
                        "rollout/min_reward": np.min(episode_rewards),
                        "rollout/mean_length": np.mean(episode_lengths),
                        "rollout/episode_count": len(episode_rewards),
                        "rollout/positive_reward_ratio": len(positive_rewards) / len(episode_rewards),
                        "rollout/negative_reward_ratio": len(negative_rewards) / len(episode_rewards),
                        "rollout/zero_reward_ratio": len(zero_rewards) / len(episode_rewards),
                        "rollout/reward_variance": np.var(episode_rewards),
                        "rollout/reward_skewness": self._calculate_skewness(episode_rewards),
                    }
                    
                    # Add percentiles for reward distribution
                    percentiles = [10, 25, 50, 75, 90]
                    for p in percentiles:
                        metrics[f"rollout/reward_p{p}"] = np.percentile(episode_rewards, p)
                    
                    # Log rollout metrics at current step (rollout_end is called at current step)
                    self.wandb_run.log(metrics, step=self.num_timesteps)
                    
                    # Store for final summary
                    self.training_metrics['episode_rewards'].extend(episode_rewards.tolist())
                    self.training_metrics['episode_lengths'].extend(episode_lengths.tolist())
                    
                    # Update episode count
                    self.episode_count += len(episode_rewards)
        except Exception as e:
            print(f"   ⚠️  Error logging rollout metrics: {e}")
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        skewness = np.mean(((data - mean) / std) ** 3)
        return skewness
    
    def _log_final_training_summary(self):
        """Log final training summary and create visualizations."""
        try:
            if len(self.training_metrics['episode_rewards']) > 0:
                # Create training curves
                self._create_training_curves()
                
                # Calculate comprehensive final statistics
                rewards = np.array(self.training_metrics['episode_rewards'])
                lengths = np.array(self.training_metrics['episode_lengths'])
                
                final_metrics = {
                    "final/mean_reward": np.mean(rewards),
                    "final/std_reward": np.std(rewards),
                    "final/max_reward": np.max(rewards),
                    "final/min_reward": np.min(rewards),
                    "final/total_episodes": len(rewards),
                    "final/total_steps": self.step_count,
                    "final/mean_episode_length": np.mean(lengths),
                    "final/reward_variance": np.var(rewards),
                    "final/reward_skewness": self._calculate_skewness(rewards),
                    "final/positive_reward_ratio": len(rewards[rewards > 0]) / len(rewards),
                    "final/negative_reward_ratio": len(rewards[rewards < 0]) / len(rewards),
                    "final/zero_reward_ratio": len(rewards[rewards == 0]) / len(rewards),
                }
                
                # Add percentiles
                percentiles = [10, 25, 50, 75, 90]
                for p in percentiles:
                    final_metrics[f"final/reward_p{p}"] = np.percentile(rewards, p)
                
                self.wandb_run.log(final_metrics)
                
                # Create training summary table
                if self.wandb_run is not None:
                    summary_table = wandb.Table(columns=["Metric", "Value", "Description"])
                    summary_table.add_data("Total Episodes", str(len(rewards)), "Number of episodes completed")
                    summary_table.add_data("Total Steps", str(self.step_count), "Total training steps")
                    summary_table.add_data("Mean Reward", f"{np.mean(rewards):.2f}", "Average reward per episode")
                    summary_table.add_data("Std Reward", f"{np.std(rewards):.2f}", "Standard deviation of rewards")
                    summary_table.add_data("Max Reward", f"{np.max(rewards):.2f}", "Best episode reward")
                    summary_table.add_data("Min Reward", f"{np.min(rewards):.2f}", "Worst episode reward")
                    summary_table.add_data("Mean Episode Length", f"{np.mean(lengths):.1f}", "Average steps per episode")
                    summary_table.add_data("Positive Reward Ratio", f"{len(rewards[rewards > 0]) / len(rewards):.1%}", "Percentage of positive rewards")
                    summary_table.add_data("Training Efficiency", f"{np.mean(rewards) / self.step_count:.4f}", "Reward per step")
                    
                    self.wandb_run.log({"training_summary": summary_table})
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
            nodes_per_layer = len(list(G.nodes())) // actual_layers
            for node in list(G.nodes()):
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
        
        ax1.set_title(f'{topology_type.replace("_", " ").title()} Network Structure\n({len(list(G.nodes()))} nodes, {len(G.edges())} edges)')
        
        # Network metrics
        metrics = {
            'Nodes': len(list(G.nodes())),
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
        'tasks': ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1'],  # All three tasks for sequential testing
        'total_timesteps': 1000,  # Reasonable training time for each task
        'n_eval_episodes': 15,     # Good evaluation episodes
        
        # ============================================================================
        # TOPOLOGY CONFIGURATION
        # ============================================================================
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
        
        # Unified layer configuration - all topologies can have multiple layers
        'layer_configs': {
            'small_world': [1],      # Small world: single layer
            'modular': [1],          # Modular: single layer  
            'hybrid': [1],           # Hybrid: single layer
            'fully_connected': [1, 2]  # FC: test both single and multi-layer
        },
        
        # ============================================================================
        # NETWORK DIMENSIONS
        # ============================================================================
        'universal_input_dim': 6,
        'universal_output_dim': 3,
        'universal_action_dim': 3,
        'hidden_size': 64,  # Fixed size for debugging
        'network_sizes': [64],  # Base network size for capacity matching
        'network_types': ['ffn'],
        'num_io_nodes': 4,  # Number of input/output nodes
        
        # ============================================================================
        # EXPERIMENT TYPES
        # ============================================================================
        'experiment_types': ['same_size', 'match_small_world'],
        'capacity_matching_reference': 'small_world',  # Reference topology for capacity matching
        
        # ============================================================================
        # TOPOLOGY-SPECIFIC PARAMETERS
        # ============================================================================
        'topology_params': {
            'small_world': {
                'k': 4,
                'p': 0.3,
                'inter_layer_prob': 0.5
            },
            'modular': {
                'num_modules': 4,
                'inter_module_prob': 0.2,
                'intra_module_prob': 0.8,
                'inter_layer_prob': 0.5
            },
            'hybrid': {
                'num_modules': 4,
                'k': 4,
                'p': 0.3,
                'inter_module_prob': 0.2,
                'inter_layer_prob': 0.5
            },
            'fully_connected': {
                'inter_layer_prob': 1.0,
                'intra_layer_prob': 1.0
            }
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
        # CAPACITY MATCHING CONFIGURATION
        # ============================================================================
        'capacity_matching': {
            'enabled': True,
            'use_capacity_mapping': False,  # Use incremental adjustment instead
            'min_search_size': 10,
            'max_search_size': 2000,
            'seeds': [42]
        },
        
        # ============================================================================
        # PARAMETER BUDGET CONFIGURATION
        # ============================================================================
        'parameter_budget': {
            'enabled': True,
            'budget_type': 'weights',  # 'weights' or 'edges'
            'padding_strategy': 'random'  # 'random' or 'zero'
        },
        
        # ============================================================================
        # NODE SELECTION CONFIGURATION
        # ============================================================================
        'node_selection_strategies': ['random']
    }
    
    # Add backward compatibility aliases for existing code
    config['use_capacity_matching'] = config['capacity_matching']['enabled']
    config['use_capacity_mapping'] = config['capacity_matching']['use_capacity_mapping']
    config['min_search_size'] = config['capacity_matching']['min_search_size']
    config['max_search_size'] = config['capacity_matching']['max_search_size']
    config['seeds'] = config['capacity_matching']['seeds']
    
    # Add layer configuration aliases for backward compatibility
    # ParameterBudgetCalculator expects these specific keys
    config['num_layers'] = [1]  # Default for non-FC topologies
    config['fc_num_layers'] = config['layer_configs']['fully_connected']  # FC layer configs
    
    # Add topology-specific parameter aliases for backward compatibility
    config['small_world_params'] = config['topology_params']['small_world']
    config['modular_params'] = config['topology_params']['modular']
    config['hybrid_params'] = config['topology_params']['hybrid']
    config['fully_connected_params'] = config['topology_params']['fully_connected']
    
    return config

# UNUSED FUNCTION - REMOVED FOR CLEANUP
# def debug_topology_policy(policy_class, topology_type, config, num_layers=2, hidden_size=None, task=None):
    """
    Debug a single topology policy to verify implementation.
    """
    pass  # Function removed - no longer used

def double_task_training(policy_class, topology_type, config, num_layers=2, hidden_size=None, train_task_1=None, train_task_2=None):
    """
    Double-task training: Train on task_1, then task_2, then evaluate on all tasks.
    
    Args:
        policy_class: The policy class to use
        topology_type: Type of topology (fully_connected, small_world, etc.)
        config: Configuration dictionary
        num_layers: Number of layers for the topology
        hidden_size: Hidden size for the network
        train_task_1: First task to train on (e.g., 'CartPole-v1')
        train_task_2: Second task to train on (e.g., 'Acrobot-v1')
    
    Returns:
        Dictionary with training and cross-task evaluation results
    """
    print(f"\n{'='*80}")
    print(f"🔄 DOUBLE-TASK TRAINING: {topology_type.upper()} TOPOLOGY")
    print(f"{'='*80}")
    
    # Create environment for first training task
    train_env_1 = DummyVecEnv([make_env(train_task_1)])
    
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
    
    # Initialize model with centralized PPO parameters
    ppo_params = config['ppo_params']
    model = PPO(
        SpecificPolicyClass,
        train_env_1,
        verbose=1,
        tensorboard_log=f"./logs/double_task_{topology_type}/",
        **ppo_params
    )
    
    # Get network size and parameter count for run name
    policy = model.policy
    actor_params = policy._get_topology_params(policy.actor_topology)
    critic_params = policy._get_topology_params(policy.critic_topology)
    total_params = actor_params + critic_params
    
    # Create descriptive run name for training
    training_run_name = f"training_{topology_type}_{num_layers}_{actual_hidden_size}_{total_params}_{train_task_1}_then_{train_task_2}"
    
    # Initialize wandb run for training (separate from testing)
    try:
        training_wandb_run = wandb.init(
            entity="katko-it-universitetet-i-k-benhavn",
            project="topologies--double-task-training",  # Double-task training project
            name=training_run_name,
            config={
                "run_type": "training",
                "topology_type": topology_type,
                "num_layers": num_layers,
                "hidden_size": actual_hidden_size,
                "total_params": total_params,
                "actor_params": actor_params,
                "critic_params": critic_params,
                "total_timesteps": config['total_timesteps'] * 2,  # Total for both tasks
                "train_task_1": train_task_1,
                "train_task_2": train_task_2,
                "ppo_params": config['ppo_params'],
                "universal_input_dim": config['universal_input_dim'],
                "universal_output_dim": config['universal_output_dim'],
                "universal_action_dim": config['universal_action_dim'],
                "experiment_type": "double_task_training",
            },
            tags=[topology_type, f"layers_{num_layers}", f"size_{actual_hidden_size}", f"params_{total_params}", train_task_1, train_task_2, "training"],
            reinit=True
        )
        training_wandb_enabled = True
    except Exception as e:
        print(f"   ⚠️  Training WandB logging disabled for {topology_type}: {e}")
        training_wandb_run = None
        training_wandb_enabled = False
    
    print(f"📋 Double-Task Configuration:")
    print(f"   • Training Task 1: {train_task_1}")
    print(f"   • Training Task 2: {train_task_2}")
    print(f"   • Test Tasks: {', '.join(config['tasks'])}")
    print(f"   • Topology: {topology_type}")
    print(f"   • Hidden size: {actual_hidden_size}")
    print(f"   • Number of layers: {num_layers}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Training timesteps per task: {config['total_timesteps']:,}")
    print(f"   • Total training timesteps: {config['total_timesteps'] * 2:,}")
    print(f"   • Training WandB Run: {training_run_name}")
    
    # Phase 1: Train the model on the first task
    print(f"\n🎯 Training Phase 1:")
    print(f"   • Training on {train_task_1}...")
    
    # Create callback for training monitoring
    callback = EnhancedDebugCallback(verbose=1, wandb_run=training_wandb_run, log_freq=100)
    
    # Train the model on first task
    start_time_1 = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    training_time_1 = time.time() - start_time_1
    print(f"✅ Training on {train_task_1} completed in {training_time_1:.2f} seconds")
    
    # Evaluate performance on first task after training
    print(f"\n📊 Evaluating performance on {train_task_1} after training...")
    task1_performance = evaluate_model_enhanced(model, train_env_1, train_task_1, n_eval_episodes=config['n_eval_episodes'])
    print(f"   • {train_task_1}: {task1_performance['mean_reward']:.2f} ± {task1_performance['std_reward']:.2f} (success: {task1_performance['success_rate']:.1%})")
    
    # Phase 2: Continue training on the second task
    print(f"\n🎯 Training Phase 2:")
    print(f"   • Continuing training on {train_task_2}...")
    
    # Create new environment for second task
    train_env_2 = DummyVecEnv([make_env(train_task_2)])
    model.set_env(train_env_2)
    
    # Continue training on second task
    start_time_2 = time.time()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    training_time_2 = time.time() - start_time_2
    print(f"✅ Training on {train_task_2} completed in {training_time_2:.2f} seconds")
    
    total_training_time = training_time_1 + training_time_2
    print(f"✅ Total training completed in {total_training_time:.2f} seconds")
    
    # Log final training metrics and finish training run
    if training_wandb_enabled and training_wandb_run is not None:
        try:
            # Log training summary
            training_summary = {
                "training/final_training_time": total_training_time,
                "training/timesteps_per_second": (config['total_timesteps'] * 2) / total_training_time,
                "training/topology_type": topology_type,
                "training/num_layers": num_layers,
                "training/train_task_1": train_task_1,
                "training/train_task_2": train_task_2,
                "training/network_size": actual_hidden_size,
                "training/total_params": total_params,
                "training/actor_params": actor_params,
                "training/critic_params": critic_params,
                "training/task1_performance_after_task1": task1_performance['mean_reward'],
                "training/task1_success_rate_after_task1": task1_performance['success_rate'],
            }
            training_wandb_run.log(training_summary)
            
            # Create training summary table
            training_summary_table = wandb.Table(columns=["Metric", "Value", "Description"])
            training_summary_table.add_data("Topology Type", topology_type, "Network topology used")
            training_summary_table.add_data("Number of Layers", str(num_layers), "Network depth")
            training_summary_table.add_data("Hidden Size", str(actual_hidden_size), "Hidden units per layer")
            training_summary_table.add_data("Total Parameters", f"{total_params:,}", "Trainable parameters")
            training_summary_table.add_data("Training Task 1", train_task_1, "First task used for training")
            training_summary_table.add_data("Training Task 2", train_task_2, "Second task used for training")
            training_summary_table.add_data("Total Training Time", f"{total_training_time:.2f}s", "Total training duration")
            training_summary_table.add_data("Timesteps/sec", f"{(config['total_timesteps'] * 2) / total_training_time:.2f}", "Training speed")
            training_summary_table.add_data("Task 1 Performance", f"{task1_performance['mean_reward']:.2f}", "Performance on Task 1 after training")
            training_summary_table.add_data("Task 1 Success Rate", f"{task1_performance['success_rate']:.1%}", "Success rate on Task 1 after training")
            
            training_wandb_run.log({"training_summary": training_summary_table})
            
            # Finish training run
            training_wandb_run.finish()
            print(f"   ✅ Training run completed: {training_run_name}")
            
        except Exception as e:
            print(f"   ⚠️  Error finishing training run: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-task evaluation
    print(f"\n🧪 Cross-Task Evaluation Phase:")
    cross_task_results = {}
    
    for test_task in config['tasks']:
        print(f"   • Evaluating on {test_task}...")
        test_env = DummyVecEnv([make_env(test_task)])
        
        try:
            # Use enhanced evaluation function for comprehensive metrics
            eval_results = evaluate_model_enhanced(model, test_env, test_task, n_eval_episodes=config['n_eval_episodes'])
            cross_task_results[test_task] = eval_results
            print(f"     ✅ {test_task}: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f} (success: {eval_results['success_rate']:.1%}, length: {eval_results['mean_length']:.1f} ± {eval_results['std_length']:.1f})")
        except Exception as e:
            print(f"     ❌ {test_task}: Error - {str(e)}")
            cross_task_results[test_task] = {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_length': 0.0,
                'std_length': 0.0,
                'success_rate': 0.0,
                'episode_rewards': [],
                'episode_lengths': []
            }
        finally:
            test_env.close()
    
    # Create separate testing runs for each task
    print(f"\n📊 Creating separate testing runs...")
    
    # Calculate transfer learning metrics for reference
    # Use the third task (not trained on) as reference for transfer metrics
    third_task = None
    for task in config['tasks']:
        if task != train_task_1 and task != train_task_2:
            third_task = task
            break
    
    third_task_performance = cross_task_results.get(third_task, {'mean_reward': 0.0})
    third_task_reward = third_task_performance['mean_reward']
    
    # Create individual testing runs for each task
    for test_task, results in cross_task_results.items():
        # Create descriptive run name for testing
        testing_run_name = f"testing_{topology_type}_{num_layers}_{actual_hidden_size}_{total_params}_train_{train_task_1}_then_{train_task_2}_test_{test_task}"
        
        try:
            # Initialize wandb run for this specific test
            testing_wandb_run = wandb.init(
                entity="katko-it-universitetet-i-k-benhavn",
                project="topologies--double-task-training",  # Double-task training project
                name=testing_run_name,
                config={
                    "run_type": "testing",
                    "topology_type": topology_type,
                    "num_layers": num_layers,
                    "hidden_size": actual_hidden_size,
                    "total_params": total_params,
                    "actor_params": actor_params,
                    "critic_params": critic_params,
                    "train_task_1": train_task_1,
                    "train_task_2": train_task_2,
                    "test_task": test_task,
                    "n_eval_episodes": config['n_eval_episodes'],
                    "ppo_params": config['ppo_params'],
                    "universal_input_dim": config['universal_input_dim'],
                    "universal_output_dim": config['universal_output_dim'],
                    "universal_action_dim": config['universal_action_dim'],
                    "experiment_type": "double_task_training",
                },
                tags=[topology_type, f"layers_{num_layers}", f"size_{actual_hidden_size}", f"params_{total_params}", train_task_1, train_task_2, test_task, "testing"],
                reinit=True
            )
            
            # Calculate transfer metrics for this specific test
            is_training_task_1 = test_task == train_task_1
            is_training_task_2 = test_task == train_task_2
            is_training_task = is_training_task_1 or is_training_task_2
            
            # Calculate forgetting for Task 1 (if testing on Task 1)
            if is_training_task_1:
                forgetting_task1 = task1_performance['mean_reward'] - results['mean_reward']
                transfer_ratio = 1.0  # No transfer ratio for training tasks
            elif is_training_task_2:
                transfer_ratio = 1.0  # No transfer ratio for training tasks
            else:
                # Transfer to third task
                transfer_ratio = results['mean_reward'] / third_task_reward if third_task_reward != 0 else 0
                forgetting_task1 = 0.0  # Not applicable for third task
            
            # Task-specific maximum rewards for normalization
            max_rewards = {
                'CartPole-v1': 200.0,
                'MountainCar-v0': 0.0,  # MountainCar has negative rewards
                'Acrobot-v1': -100.0,   # Acrobot has negative rewards
            }
            max_reward = max_rewards.get(test_task, 200.0)
            
            # Calculate normalized performance (0-1 scale)
            if max_reward != 0:
                normalized_performance = (results['mean_reward'] - max_reward) / abs(max_reward) if max_reward < 0 else results['mean_reward'] / max_reward
            else:
                normalized_performance = 0.0
            
            # Log comprehensive testing metrics
            testing_metrics = {
                # Basic performance metrics
                "testing/mean_reward": results['mean_reward'],
                "testing/std_reward": results['std_reward'],
                "testing/n_eval_episodes": config['n_eval_episodes'],
                
                # Enhanced metrics (Phase 1)
                "testing/mean_length": results['mean_length'],
                "testing/std_length": results['std_length'],
                "testing/success_rate": results['success_rate'],
                
                # Transfer learning metrics
                "transfer/transfer_ratio": transfer_ratio,
                "transfer/relative_performance": transfer_ratio * 100,  # Percentage
                "transfer/is_training_task": is_training_task,
                "transfer/is_training_task_1": is_training_task_1,
                "transfer/is_training_task_2": is_training_task_2,
                "transfer/third_task_performance": third_task_reward,
                
                # Forgetting metrics (for Task 1)
                "forgetting/task1_forgetting": forgetting_task1 if is_training_task_1 else 0.0,
                "forgetting/task1_performance_after_task1": task1_performance['mean_reward'],
                "forgetting/task1_performance_final": results['mean_reward'] if is_training_task_1 else 0.0,
                
                # Task-specific analysis
                "task_analysis/normalized_performance": normalized_performance,
                "task_analysis/raw_performance": results['mean_reward'],
                "task_analysis/max_possible": max_reward,
                "task_analysis/task_difficulty": test_task,
                
                # Network architecture
                "network/topology_type": topology_type,
                "network/layers": num_layers,
                "network/size": actual_hidden_size,
                "network/total_parameters": total_params,
                "network/parameter_efficiency": total_params / actual_hidden_size,
                "network/actor_critic_ratio": actor_params / critic_params if critic_params > 0 else 0,
                
                # Training context
                "context/training_time": total_training_time,
                "context/timesteps_per_second": (config['total_timesteps'] * 2) / total_training_time,
                "context/total_training_timesteps": config['total_timesteps'] * 2,
            }
            
            testing_wandb_run.log(testing_metrics)
            
            # Create detailed testing summary table
            testing_summary_table = wandb.Table(columns=["Metric", "Value", "Description"])
            testing_summary_table.add_data("Topology Type", topology_type, "Network topology used")
            testing_summary_table.add_data("Training Task 1", train_task_1, "First task used for training")
            testing_summary_table.add_data("Training Task 2", train_task_2, "Second task used for training")
            testing_summary_table.add_data("Test Task", test_task, "Task being tested")
            testing_summary_table.add_data("Mean Reward", f"{results['mean_reward']:.2f}", "Average reward on test task")
            testing_summary_table.add_data("Std Reward", f"{results['std_reward']:.2f}", "Standard deviation of rewards")
            testing_summary_table.add_data("Mean Episode Length", f"{results['mean_length']:.1f}", "Average steps per episode")
            testing_summary_table.add_data("Std Episode Length", f"{results['std_length']:.1f}", "Standard deviation of episode lengths")
            testing_summary_table.add_data("Success Rate", f"{results['success_rate']:.1%}", "Percentage of successful episodes")
            testing_summary_table.add_data("Transfer Ratio", f"{transfer_ratio:.3f}", "Performance ratio vs third task")
            testing_summary_table.add_data("Normalized Performance", f"{normalized_performance:.3f}", "Task-relative performance (0-1)")
            testing_summary_table.add_data("Is Training Task", "Yes" if is_training_task else "No", "Whether this is a training task")
            testing_summary_table.add_data("Is Training Task 1", "Yes" if is_training_task_1 else "No", "Whether this is the first training task")
            testing_summary_table.add_data("Is Training Task 2", "Yes" if is_training_task_2 else "No", "Whether this is the second training task")
            if is_training_task_1:
                testing_summary_table.add_data("Task 1 Forgetting", f"{forgetting_task1:.2f}", "Performance drop on Task 1 after Task 2 training")
            testing_summary_table.add_data("Network Size", str(actual_hidden_size), "Hidden units in network")
            testing_summary_table.add_data("Total Parameters", f"{total_params:,}", "Total trainable parameters")
            
            testing_wandb_run.log({"testing_summary": testing_summary_table})
            
            # Finish this testing run
            testing_wandb_run.finish()
            print(f"   ✅ Testing run completed: {testing_run_name}")
            
        except Exception as e:
            print(f"   ⚠️  Error creating testing run for {test_task}: {e}")
            import traceback
            traceback.print_exc()
    
    train_env_1.close()
    train_env_2.close()
    
    # Prepare result dictionary
    result = {
        'topology_type': topology_type,
        'num_layers': num_layers,
        'train_task_1': train_task_1,
        'train_task_2': train_task_2,
        'network_size': actual_hidden_size,
        'total_params': total_params,
        'actor_params': actor_params,
        'critic_params': critic_params,
        'training_time_task_1': training_time_1,
        'training_time_task_2': training_time_2,
        'total_training_time': total_training_time,
        
        # Performance after Task 1 training
        f'{train_task_1}_after_task1_mean_reward': task1_performance['mean_reward'],
        f'{train_task_1}_after_task1_std_reward': task1_performance['std_reward'],
        f'{train_task_1}_after_task1_mean_length': task1_performance['mean_length'],
        f'{train_task_1}_after_task1_std_length': task1_performance['std_length'],
        f'{train_task_1}_after_task1_success_rate': task1_performance['success_rate'],
    }
    
    # Add final cross-task results
    for test_task, results in cross_task_results.items():
        result[f'{test_task}_final_mean_reward'] = results['mean_reward']
        result[f'{test_task}_final_std_reward'] = results['std_reward']
        result[f'{test_task}_final_mean_length'] = results['mean_length']
        result[f'{test_task}_final_std_length'] = results['std_length']
        result[f'{test_task}_final_success_rate'] = results['success_rate']
    
    # Add forgetting metrics
    if train_task_1 in cross_task_results:
        result[f'{train_task_1}_forgetting'] = task1_performance['mean_reward'] - cross_task_results[train_task_1]['mean_reward']
    
    # Add transfer metrics for third task
    if third_task and third_task in cross_task_results:
        result[f'{third_task}_transfer_ratio'] = cross_task_results[third_task]['mean_reward'] / third_task_reward if third_task_reward != 0 else 0
    
    return result

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

def evaluate_model_enhanced(model, env, task_name, n_eval_episodes=3):
    """
    Enhanced evaluation function that tracks episode lengths and success rates.
    
    Args:
        model: The trained model to evaluate
        env: The environment to evaluate on
        task_name: Name of the task (e.g., 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1')
        n_eval_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    rewards = []
    episode_lengths = []
    
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
        episode_lengths.append(step_count)
    
    # Calculate basic statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    # Calculate success rate based on task-specific criteria
    success_rate = calculate_success_rate(rewards, episode_lengths, task_name)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'std_length': std_length,
        'success_rate': success_rate,
        'episode_rewards': rewards,
        'episode_lengths': episode_lengths
    }

def calculate_success_rate(rewards, episode_lengths, task_name):
    """
    Calculate success rate based on task-specific criteria.
    
    Args:
        rewards: List of episode rewards
        episode_lengths: List of episode lengths
        task_name: Name of the task
    
    Returns:
        Success rate as a float between 0 and 1
    """
    if task_name == 'CartPole-v1':
        # Success: episode length >= 200 (CartPole solved)
        return np.mean([length >= 200 for length in episode_lengths])
    elif task_name == 'MountainCar-v0':
        # Success: reached goal position (reward > -200)
        return np.mean([reward > -200 for reward in rewards])
    elif task_name == 'Acrobot-v1':
        # Success: swung up to vertical (reward > -100)
        return np.mean([reward > -100 for reward in rewards])
    else:
        # Default: no success criteria defined
        return 0.0

def verify_capacity_matching_debug(config):
    """Verify capacity matching implementation using the same pattern as working scripts."""
    print("\n" + "="*80)
    print("🔍 CAPACITY MATCHING VERIFICATION")
    print("="*80)
    
    # Create calculator for capacity matching
    calculator = ParameterBudgetCalculator(config)
    
    # Use base topology names like working scripts
    base_topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    
    # Track results for summary
    results_summary = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {},
        'matching_sizes': {}  # Store matching sizes for experiments
    }
    
    print(f"📋 Verification Configuration:")
    print(f"   • Network Sizes: {config['network_sizes']}")
    print(f"   • Base Topologies: {base_topologies}")
    print(f"   • Network Types: {config['network_types']}")
    print(f"   • Layer Configurations:")
    for topology, layers in config['layer_configs'].items():
        print(f"     - {topology}: {layers} layers")
    print(f"   • Experiment Types: {config['experiment_types']}")
    print("="*80)
    
    print(f"\n📊 Capacity Matching Calculations:")
    print("-" * 60)
    
    for experiment_type in config['experiment_types']:
        if experiment_type.startswith('match_'):
            reference_topology = experiment_type[len('match_'):]
            print(f"\n🔧 {experiment_type.upper()} (matching to {reference_topology}):")
            
            for topology in base_topologies:
                if topology == reference_topology:
                    continue  # Skip matching topology to itself
                    
                for size in config['network_sizes']:
                    for network_type in config['network_types']:
                        # Get layer configurations for this topology
                        topology_layers = config['layer_configs'].get(topology, [1])
                        
                        for num_layers in topology_layers:
                            print(f"  📋 {topology} (size {size}, {num_layers} layers) matching to {reference_topology}:")
                            print(f"      • Capacity matching: {size} nodes → target capacity from {reference_topology}")
                            
                            try:
                                # Get target capacity from reference topology (use num_layers=1 for capacity matching like working scripts)
                                target_capacity = calculator._get_reference_capacity(
                                    reference_topology, size, network_type, 1
                                )
                                
                                # Get matching size for current topology (use num_layers=1 for capacity matching like working scripts)
                                matching_size = calculator.calculate_matching_size(topology, target_capacity, network_type, 1)
                                
                                # Store matching size for experiments (use actual layers)
                                key = f"{experiment_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['matching_sizes'][key] = matching_size
                                
                                # Create network using calculator's create_network method (like working scripts)
                                network = calculator.create_network(
                                    topology=topology,
                                    size=matching_size,
                                    experiment_type='same_size',  # Use 'same_size' to avoid recursive matching
                                    network_type=network_type,
                                    num_layers=num_layers,  # Use actual layers
                                    seed=42
                                )
                                
                                # Get actual parameter count
                                metrics = network.get_network_metrics()
                                actual_capacity = sum(
                                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                )
                                
                                # Calculate divergence
                                divergence = abs(actual_capacity - target_capacity) / target_capacity * 100
                                
                                # Check if within threshold
                                threshold = 10.0  # 10% threshold
                                if divergence <= threshold:
                                    status = "✓ PASSED"
                                    results_summary['passed'] += 1
                                else:
                                    status = "✗ FAILED"
                                    results_summary['failed'] += 1
                                
                                print(f"    {status}: {divergence:.1f}% divergence")
                                print(f"      • Target: {target_capacity:,} parameters")
                                print(f"      • Actual: {actual_capacity:,} parameters")
                                print(f"      • Size: {size} → {matching_size} nodes")
                                print(f"      • Network: {topology} with {num_layers} layers")
                                
                                # Store details
                                results_summary['details'][key] = {
                                    'target_capacity': target_capacity,
                                    'actual_capacity': actual_capacity,
                                    'matching_size': matching_size,
                                    'divergence': divergence,
                                    'status': 'passed' if divergence <= threshold else 'failed'
                                }
                                
                            except Exception as e:
                                print(f"    ✗ ERROR: {e}")
                                results_summary['errors'] += 1
                                key = f"{experiment_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'error': str(e),
                                    'status': 'error'
                                }
                                
                                # Get actual parameter count
                                metrics = network.get_network_metrics()
                                actual_capacity = sum(
                                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                )
                                
                                # Calculate divergence
                                divergence = abs(actual_capacity - target_capacity) / target_capacity * 100
                                
                                # Check if within threshold
                                threshold = 10.0  # 10% threshold
                                if divergence <= threshold:
                                    status = "✓ PASSED"
                                    results_summary['passed'] += 1
                                else:
                                    status = "✗ FAILED"
                                    results_summary['failed'] += 1
                                
                                print(f"    {status}: {divergence:.1f}% divergence")
                                print(f"      • Target: {target_capacity:,} parameters")
                                print(f"      • Actual: {actual_capacity:,} parameters")
                                print(f"      • Size: {size} → {matching_size} nodes")
                                print(f"      • Network: {topology} with 1 layer")
                                
                                # Store details
                                results_summary['details'][key] = {
                                    'target_capacity': target_capacity,
                                    'actual_capacity': actual_capacity,
                                    'matching_size': matching_size,
                                    'divergence': divergence,
                                    'status': 'passed' if divergence <= threshold else 'failed'
                                }
                                
                            except Exception as e:
                                print(f"    ✗ ERROR: {e}")
                                results_summary['errors'] += 1
                                key = f"{experiment_type}_{topology}_{size}_{network_type}_1"
                                results_summary['details'][key] = {
                                    'error': str(e),
                                    'status': 'error'
                                }
        else:
            # Same size experiments - just verify baseline capacities (like working scripts)
            print(f"\n🔧 {experiment_type.upper()}: Baseline capacity verification")
            print("All topologies use the same node count (not matched capacities)")
            
            for topology in base_topologies:
                for size in config['network_sizes']:
                    for network_type in config['network_types']:
                        # Get layer configurations for this topology
                        topology_layers = config['layer_configs'].get(topology, [1])
                        
                        for num_layers in topology_layers:
                            try:
                                # For same_size, just use the original size (like working scripts)
                                network = calculator.create_network(
                                    topology=topology,
                                    size=size,
                                    experiment_type='same_size',
                                    network_type=network_type,
                                    num_layers=num_layers,
                                    seed=42
                                )
                                
                                # Get actual parameter count
                                metrics = network.get_network_metrics()
                                actual_capacity = sum(
                                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                )
                                
                                print(f"  📋 {topology} (size {size}, {num_layers} layers): {actual_capacity:,} parameters")
                                print(f"      • Network: {topology} with {num_layers} layers")
                                print(f"      ✅ Same size experiment - no capacity matching required")
                                
                                # Store details for summary
                                key = f"{experiment_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'actual_capacity': actual_capacity,
                                    'status': 'passed',
                                    'type': 'same_size'
                                }
                                results_summary['passed'] += 1
                                
                            except Exception as e:
                                print(f"  ✗ ERROR {topology} (size {size}, {num_layers} layers): {e}")
                                results_summary['errors'] += 1
                                key = f"{experiment_type}_{topology}_{size}_{network_type}_{num_layers}"
                                results_summary['details'][key] = {
                                    'error': str(e),
                                    'status': 'error'
                                }
    
    # Print summary
    print("\n" + "="*80)
    print("📊 CAPACITY MATCHING VERIFICATION SUMMARY")
    print("="*80)
    print(f"   • Passed: {results_summary['passed']}")
    print(f"   • Failed: {results_summary['failed']}")
    print(f"   • Errors: {results_summary['errors']}")
    print(f"   • Total: {results_summary['passed'] + results_summary['failed'] + results_summary['errors']}")
    
    if results_summary['failed'] > 0 or results_summary['errors'] > 0:
        print(f"\n⚠️  WARNING: {results_summary['failed']} capacity matching calculations failed and {results_summary['errors']} had errors!")
        print("   Consider checking the capacity matching implementation.")
    else:
        print(f"\n✅ All capacity matching calculations passed!")
    
    print("="*80)
    
    return results_summary

def main():
    """Main function to run debug test."""
    print("🔄 Double-Task Training: Sequential Task Training with Transfer Evaluation")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Individual wandb runs will be created for each topology
    print(f"   ✅ Individual WandB runs will be created for each topology")
    
    # Create configuration
    config = create_debug_config()
    
    # Verify capacity matching first (like working scripts)
    print("\n🔍 Verifying capacity matching implementation...")
    verification_results = verify_capacity_matching_debug(config)
    
    # Define topologies and experiment types (use base topology names like working scripts)
    base_topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    experiment_types = config.get('experiment_types', ['same_size'])
    
    # Define all possible double-task training combinations
    double_task_combinations = [
        ('CartPole-v1', 'Acrobot-v1'),      # x, y
        ('CartPole-v1', 'MountainCar-v0'),  # x, z
        ('Acrobot-v1', 'MountainCar-v0'),   # y, z
        ('Acrobot-v1', 'CartPole-v1'),      # y, x
        ('MountainCar-v0', 'CartPole-v1'),  # z, x
        ('MountainCar-v0', 'Acrobot-v1'),   # z, y
    ]
    
    print(f"\n📋 Double-Task Training Configuration:")
    print(f"   • Training Task Combinations: {len(double_task_combinations)} combinations")
    for task1, task2 in double_task_combinations:
        print(f"     - {task1} → {task2}")
    print(f"   • Base Topologies: {', '.join(base_topologies)}")
    print(f"   • Experiment Types: {', '.join(experiment_types)}")
    print(f"   • Hidden size: {config['hidden_size']}")
    print(f"   • Layer Configurations:")
    for topology, layers in config['layer_configs'].items():
        print(f"     - {topology}: {layers} layers")
    print(f"   • Training timesteps: {config['total_timesteps']:,} per training task")
    print(f"   • Total training timesteps: {config['total_timesteps'] * 2:,} per experiment")
    print(f"   • Evaluation episodes: {config['n_eval_episodes']} per test task")
    print(f"   • Transfer Testing: Train on X → Y → Test on all {', '.join(config['tasks'])}")
    print(f"   • Capacity Matching: {'Enabled' if config.get('use_capacity_matching', False) else 'Disabled'}")
    print(f"   • Capacity Mapping: {'Enabled' if config.get('use_capacity_mapping', True) else 'Disabled (using incremental adjustment)'}")
    print(f"   • WandB Runs: Individual runs per topology with descriptive names")
    
    # Store all results
    all_results = []
    
    # Debug each topology and experiment type combination
    experiment_types = config.get('experiment_types', ['same_size'])
    
    # Generate experiments using unified layer configuration
    filtered_experiments = []
    for topology in base_topologies:
        for experiment_type in experiment_types:
            # Get layer configurations for this topology
            topology_layers = config['layer_configs'].get(topology, [1])
            
            for num_layers in topology_layers:
                if experiment_type == 'same_size':
                    filtered_experiments.append((topology, experiment_type, num_layers))
                elif experiment_type.startswith('match_'):
                    reference_topology = experiment_type[len('match_'):]
                    if topology != reference_topology:
                        filtered_experiments.append((topology, experiment_type, num_layers))
    
    experiment_count = 0
    total_experiments = len(filtered_experiments) * len(double_task_combinations)
    
    print(f"\n🔬 Double-Task Experiment Plan:")
    print(f"   • Total training experiments: {total_experiments}")
    print(f"   • Transfer testing: Each trained model tested on all {len(config['tasks'])} tasks")
    print(f"   • Total evaluations: {total_experiments * len(config['tasks'])}")
    print(f"   • Topology combinations to test:")
    for topology, experiment_type, num_layers in filtered_experiments:
        print(f"     - {topology} + {experiment_type} ({num_layers} layers)")
    print(f"   • Task combinations to test:")
    for task1, task2 in double_task_combinations:
        print(f"     - {task1} → {task2}")
    
    # Double-task training: Train on task1, then task2, test on all tasks
    for train_task_1, train_task_2 in double_task_combinations:
        print(f"\n{'='*80}")
        print(f"🎯 TRAINING SEQUENCE: {train_task_1.upper()} → {train_task_2.upper()}")
        print(f"{'='*80}")
        
        for topology, experiment_type, num_layers in filtered_experiments:
            experiment_count += 1
            print(f"\n{'='*80}")
            layer_info = f" ({num_layers} layers)" if topology == 'fully_connected' else ""
            print(f"🔍 EXPERIMENT {experiment_count}/{total_experiments}: {topology.upper()} + {experiment_type.upper()}{layer_info} TRAINED ON {train_task_1.upper()} → {train_task_2.upper()}")
            print(f"{'='*80}")
            
            # Use base topology names and handle layer variations
            actual_topology = topology  # Already using base names now
            
            # Update config with current experiment type and tasks
            current_config = config.copy()
            current_config['current_experiment_type'] = experiment_type
            current_config['current_task_1'] = train_task_1
            current_config['current_task_2'] = train_task_2
            
            # Show capacity matching calculations (using pre-calculated results from verification)
            print(f"🔧 Capacity Matching Analysis:")
            base_size = config['network_sizes'][0]
            
            if experiment_type.startswith('match_'):
                # Use pre-calculated matching size from verification results
                key = f"{experiment_type}_{topology}_{base_size}_ffn_{num_layers}"
                if key in verification_results['matching_sizes']:
                    matching_size = verification_results['matching_sizes'][key]
                    reference_topology = experiment_type[len('match_'):]
                    
                    print(f"   • Reference Topology: {reference_topology}")
                    print(f"   • Base Size: {base_size} nodes")
                    print(f"   • Pre-calculated Matching Size: {matching_size} nodes")
                    print(f"   • Capacity Matching: {base_size} → {matching_size} nodes")
                    
                    # Store the matching size to pass to the policy (don't modify config)
                    actual_size = matching_size
                else:
                    print(f"   ⚠️  No pre-calculated matching size found for {key}")
                    print(f"   • Using base size as fallback: {base_size} nodes")
                    actual_size = base_size
            else:
                print(f"   • Same Size Experiment: {base_size} nodes")
                print(f"   • No capacity matching required")
                actual_size = base_size
            
            # Double-task training: Train on task1, then task2, test on all tasks
            result = double_task_training(
                DebugTopologyPolicy,
                actual_topology,
                current_config,
                num_layers=num_layers,
                hidden_size=actual_size,  # Pass the capacity-matched size directly
                train_task_1=train_task_1,  # Pass the first training task
                train_task_2=train_task_2   # Pass the second training task
            )
            
            # Update the result to reflect the original topology name and experiment type
            result['topology_type'] = topology
            result['experiment_type'] = experiment_type
            result['num_layers'] = num_layers
            result['train_task_1'] = train_task_1
            result['train_task_2'] = train_task_2
            all_results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/double_task_training_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save DataFrame
    df.to_csv(f"{results_dir}/double_task_results.csv", index=False)
    
    # Save configuration
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Individual wandb runs are handled within double_task_training function
    # No main run logging needed since each topology has its own run
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🎉 DOUBLE-TASK TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Results Summary:")
    print(f"   • Total Training Experiments: {len(all_results)}")
    print(f"   • Transfer Evaluations: {len(all_results) * len(config['tasks'])}")
    print(f"   • Results saved to: {results_dir}")
    print(f"   • CSV file: double_task_results.csv")
    print(f"   • Config file: config.json")
    print(f"   • WandB Project: topologies--double-task-training")
    print(f"   • Training runs: training_[topology]_[layers]_[size]_[params]_[task1]_then_[task2]")
    print(f"   • Testing runs: testing_[topology]_[layers]_[size]_[params]_train_[task1]_then_[task2]_test_[test_task]")
    
    # Print results
    if not df.empty:
        print(f"\n📈 Double-Task Results by Topology:")
        for _, row in df.iterrows():
            print(f"   • {row['topology_type']} (trained on {row['train_task_1']} → {row['train_task_2']}):")
            print(f"     - CartPole-v1: {row.get('CartPole-v1_final_mean_reward', 'N/A'):.2f} ± {row.get('CartPole-v1_final_std_reward', 'N/A'):.2f} (success: {row.get('CartPole-v1_final_success_rate', 'N/A'):.1%}, length: {row.get('CartPole-v1_final_mean_length', 'N/A'):.1f})")
            print(f"     - MountainCar-v0: {row.get('MountainCar-v0_final_mean_reward', 'N/A'):.2f} ± {row.get('MountainCar-v0_final_std_reward', 'N/A'):.2f} (success: {row.get('MountainCar-v0_final_success_rate', 'N/A'):.1%}, length: {row.get('MountainCar-v0_final_mean_length', 'N/A'):.1f})")
            print(f"     - Acrobot-v1: {row.get('Acrobot-v1_final_mean_reward', 'N/A'):.2f} ± {row.get('Acrobot-v1_final_std_reward', 'N/A'):.2f} (success: {row.get('Acrobot-v1_final_success_rate', 'N/A'):.1%}, length: {row.get('Acrobot-v1_final_mean_length', 'N/A'):.1f})")
            print(f"     - Architecture: {row['num_layers']} layers, {row['network_size']} nodes, {row['total_params']:,} params")
            forgetting_key = f"{row['train_task_1']}_forgetting"
            if row.get(forgetting_key, None) is not None:
                print(f"     - Forgetting on {row['train_task_1']}: {row[forgetting_key]:.2f}")
        
        print(f"\n🔍 Double-Task Analysis:")
        print(f"   • Sequential training evaluation completed")
        print(f"   • Each topology trained on two tasks sequentially")
        print(f"   • Results show transfer capabilities and forgetting patterns")
        print(f"   • Forgetting metrics show performance retention on first task")
    
    print(f"\n✅ Debug test completed successfully!")
    print(f"💡 Next steps:")
    print(f"   • Modify num_layers in create_debug_config() to explore different layer counts")
    print(f"   • Check the detailed debug output above for topology network verification")
    print(f"   • Examine the saved results for performance comparison")
    print(f"   • View results in WandB dashboard: https://wandb.ai/katko-it-universitetet-i-k-benhavn/projects")

if __name__ == "__main__":
    main() 