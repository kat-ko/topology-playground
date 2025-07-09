"""
Baseline PPO Comparison with Standard Networks

This script implements standard PPO training with:
1. Standard MLP (Feed-Forward Network)
2. Standard RNN/LSTM Network

This provides a baseline comparison for our topology-based networks.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Type, Union

class TrainingCallback(BaseCallback):
    """Callback to track training metrics."""
    
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
        # Track episode rewards
        if len(self.training_env.buf_rews) > 0:
            self.current_episode_reward += self.training_env.buf_rews[0]
            self.current_episode_length += 1
            
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        # Track losses (if available)
        if hasattr(self.model, 'logger') and self.model.logger.name_to_value:
            if 'train/value_loss' in self.model.logger.name_to_value:
                self.value_losses.append(self.model.logger.name_to_value['train/value_loss'])
            if 'train/policy_loss' in self.model.logger.name_to_value:
                self.policy_losses.append(self.model.logger.name_to_value['train/policy_loss'])
            if 'train/entropy_loss' in self.model.logger.name_to_value:
                self.entropy_losses.append(self.model.logger.name_to_value['train/entropy_loss'])
        
        return True

class StandardMLPPolicy(ActorCriticPolicy):
    """Standard MLP policy for baseline comparison."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Standard MLP architecture
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)  # CartPole has discrete action space
        )
        
        # Critic head (value)
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

class StandardRNNPolicy(ActorCriticPolicy):
    """Standard RNN policy for baseline comparison."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # RNN parameters
        self.hidden_size = 64
        self.num_layers = 2
        
        # RNN layer
        self.lstm = nn.LSTM(
            input_size=self.features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Actor head (policy)
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, action_space.n)  # CartPole has discrete action space
        )
        
        # Critic head (value)
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Hidden state storage
        self.hidden = None
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        if self.hidden is None:
            batch_size = features.size(0)
            self.hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)
            )
        lstm_out, self.hidden = self.lstm(features, self.hidden)
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out[:, -1, :]
        return self.action_net(lstm_out)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        if self.hidden is None:
            batch_size = features.size(0)
            self.hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=features.device)
            )
        lstm_out, self.hidden = self.lstm(features, self.hidden)
        if len(lstm_out.shape) == 3:
            lstm_out = lstm_out[:, -1, :]
        return self.value_net(lstm_out)
    
    def reset_hidden(self):
        self.hidden = None

def test_standard_mlp():
    """Test standard MLP with PPO."""
    print("=== Testing Standard MLP with PPO ===")
    
    # Create environment
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Create PPO with standard MLP
    model = PPO(
        StandardMLPPolicy,
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
    
    # Setup callback
    callback = TrainingCallback()
    
    # Train
    print("Training standard MLP...")
    model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    
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
    print(f"Standard MLP - Average test reward: {avg_reward:.2f}")
    print(f"Standard MLP - All test rewards: {test_rewards}")
    
    return {
        'type': 'Standard MLP',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'value_losses': callback.value_losses,
        'policy_losses': callback.policy_losses,
        'entropy_losses': callback.entropy_losses
    }

def test_standard_rnn():
    """Test standard RNN with PPO."""
    print("\n=== Testing Standard RNN with PPO ===")
    
    # Create environment
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Create PPO with standard RNN
    model = PPO(
        StandardRNNPolicy,
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
    
    # Setup callback
    callback = TrainingCallback()
    
    # Train
    print("Training standard RNN...")
    model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    
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
    print(f"Standard RNN - Average test reward: {avg_reward:.2f}")
    print(f"Standard RNN - All test rewards: {test_rewards}")
    
    return {
        'type': 'Standard RNN',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'value_losses': callback.value_losses,
        'policy_losses': callback.policy_losses,
        'entropy_losses': callback.entropy_losses
    }

def compare_networks():
    """Compare standard networks with topology networks."""
    print("=== Network Comparison Study ===")
    
    # Test standard networks
    mlp_results = test_standard_mlp()
    rnn_results = test_standard_rnn()
    
    # Test topology networks (using existing quick test)
    print("\n=== Testing Topology Networks ===")
    from quick_test_improvements import quick_test_improvements
    topology_success = quick_test_improvements()
    
    # Summary comparison
    print("\n" + "="*60)
    print("NETWORK COMPARISON SUMMARY")
    print("="*60)
    
    print(f"\nStandard MLP:")
    print(f"  Average reward: {mlp_results['avg_reward']:.2f}")
    print(f"  Test rewards: {mlp_results['test_rewards']}")
    
    print(f"\nStandard RNN:")
    print(f"  Average reward: {rnn_results['avg_reward']:.2f}")
    print(f"  Test rewards: {rnn_results['test_rewards']}")
    
    print(f"\nTopology Networks:")
    print(f"  Success: {topology_success}")
    
    # Performance ranking
    performances = [
        ('Standard MLP', mlp_results['avg_reward']),
        ('Standard RNN', rnn_results['avg_reward']),
    ]
    
    performances.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nPerformance Ranking:")
    for i, (name, reward) in enumerate(performances, 1):
        print(f"  {i}. {name}: {reward:.2f}")
    
    # Create comparison plots
    create_comparison_plots(mlp_results, rnn_results)
    
    return mlp_results, rnn_results

def create_comparison_plots(mlp_results, rnn_results):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards comparison
    if mlp_results['episode_rewards']:
        axes[0, 0].plot(mlp_results['episode_rewards'], label='Standard MLP', alpha=0.7)
    if rnn_results['episode_rewards']:
        axes[0, 0].plot(rnn_results['episode_rewards'], label='Standard RNN', alpha=0.7)
    axes[0, 0].set_title('Episode Rewards Comparison')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test performance comparison
    networks = ['Standard MLP', 'Standard RNN']
    avg_rewards = [mlp_results['avg_reward'], rnn_results['avg_reward']]
    axes[0, 1].bar(networks, avg_rewards, color=['blue', 'orange'])
    axes[0, 1].set_title('Final Test Performance')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Value losses comparison
    if mlp_results['value_losses']:
        axes[1, 0].plot(mlp_results['value_losses'], label='Standard MLP', alpha=0.7)
    if rnn_results['value_losses']:
        axes[1, 0].plot(rnn_results['value_losses'], label='Standard RNN', alpha=0.7)
    axes[1, 0].set_title('Value Loss Comparison')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Policy losses comparison
    if mlp_results['policy_losses']:
        axes[1, 1].plot(mlp_results['policy_losses'], label='Standard MLP', alpha=0.7)
    if rnn_results['policy_losses']:
        axes[1, 1].plot(rnn_results['policy_losses'], label='Standard RNN', alpha=0.7)
    axes[1, 1].set_title('Policy Loss Comparison')
    axes[1, 1].set_xlabel('Update Step')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_network_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plots saved to 'baseline_network_comparison.png'")

if __name__ == "__main__":
    mlp_results, rnn_results = compare_networks()
    print("\n=== Baseline Comparison Complete ===") 