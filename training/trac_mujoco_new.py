#!/usr/bin/env python3
"""
MuJoCo PPO implementation based on the exact working implementation patterns.
Uses the same hyperparameters and architecture as the working Brax implementation.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import argparse
from tqdm import tqdm
import os

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    """Actor-Critic network matching the working implementation architecture"""
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        # Shared layers - matching working implementation
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights properly - matching working implementation
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use orthogonal initialization with gain=1.0 like working implementation
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # Use tanh activation like working implementation
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        # Actor outputs
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std)
        
        # Critic output
        value = self.critic(x)
        
        return mean, std, value
    
    def get_action(self, state, action_space):
        mean, std, _ = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        
        # Clip actions to environment bounds
        action = torch.clamp(action, 
                           torch.tensor(action_space.low, device=action.device, dtype=action.dtype),
                           torch.tensor(action_space.high, device=action.device, dtype=action.dtype))
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).cpu().numpy(), log_prob.item()
    
    def evaluate(self, state, action):
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(), entropy

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 0
            next_val = next_value
        else:
            next_non_terminal = 1
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
    
    returns = advantages + values
    return advantages, returns

def ppo_update(actor_critic, optimizer, states, actions, old_log_probs, advantages, returns, 
               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, epochs=10):
    """PPO update with proper clipping and normalization"""
    
    # Convert to tensors
    states = torch.FloatTensor(states).to(device)
    actions = torch.FloatTensor(actions).to(device)
    old_log_probs = torch.FloatTensor(old_log_probs).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    returns = torch.FloatTensor(returns).to(device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(epochs):
        # Get current policy
        log_probs, values, entropy = actor_critic.evaluate(states, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 0.5)
        optimizer.step()

def train_ppo(env_name, num_episodes=1000, max_timesteps=1000, update_frequency=2048):
    """Train PPO on MuJoCo environment with working implementation parameters"""
    print(f"Training PPO on {env_name}")
    print("=" * 60)
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    
    # Create actor-critic with working implementation architecture
    actor_critic = ActorCritic(state_dim, action_dim, hidden_size=64).to(device)
    # Use same learning rate as working implementation
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
    
    # Storage
    states = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    
    episode_rewards = []
    recent_rewards = deque(maxlen=100)
    
    total_steps = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_timesteps):
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob = actor_critic.get_action(state_tensor, env.action_space)
            
            # Get value
            with torch.no_grad():
                _, _, value = actor_critic.forward(state_tensor)
                value = value.squeeze().item()
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            state = next_state
            
            # Update if buffer is full - use same frequency as working implementation
            if len(states) >= update_frequency:
                # Get next value for GAE
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    _, _, next_value = actor_critic.forward(next_state_tensor)
                    next_value = next_value.squeeze().item()
                
                # Compute advantages and returns
                advantages, returns = compute_gae(rewards, values, next_value)
                
                # PPO update with working implementation parameters
                ppo_update(actor_critic, optimizer, states, actions, log_probs, advantages, returns,
                          clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, epochs=10)
                
                # Clear buffers
                states.clear()
                actions.clear()
                rewards.clear()
                values.clear()
                log_probs.clear()
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        # Print progress
        if episode % 50 == 0 or episode_reward > 0:
            avg_recent = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode {episode:4d}: Reward={episode_reward:8.2f}, "
                  f"Recent Avg={avg_recent:8.2f}, Steps={episode_steps:4d}")
    
    # Final analysis
    episode_rewards = np.array(episode_rewards)
    positive_episodes = np.sum(episode_rewards > 0)
    
    print(f"\nFinal Results:")
    print(f"  Episodes with positive rewards: {positive_episodes}/{len(episode_rewards)}")
    print(f"  Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"  Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Final 100 episodes mean: {np.mean(episode_rewards[-100:]):.2f}")
    
    if positive_episodes > 0:
        print(f"  ✅ SUCCESS: Achieved positive rewards!")
    else:
        print(f"  ❌ No positive rewards achieved")
    
    env.close()
    return episode_rewards

def test_baseline(env_name):
    """Test baseline strategies"""
    print(f"\nBaseline strategies for {env_name}:")
    env = gym.make(env_name)
    
    # Zero actions
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(1000):
            action = np.zeros(env.action_space.shape[0])
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    print(f"  Zero actions: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Small random actions
    rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(1000):
            action = np.random.normal(0, 0.1, env.action_space.shape[0])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        rewards.append(episode_reward)
    print(f"  Small random: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    env.close()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="MuJoCo PPO Training - Working Implementation")
    parser.add_argument("--task", type=str, choices=['halfcheetah', 'ant'], default='halfcheetah',
                        help="Task to train on")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes to train")
    parser.add_argument("--test_baseline", action='store_true',
                        help="Test baseline strategies first")
    
    args = parser.parse_args()
    
    env_name = 'HalfCheetah-v4' if args.task == 'halfcheetah' else 'Ant-v4'
    
    print(f"MuJoCo PPO - Working Implementation for {env_name}")
    print("=" * 60)
    
    # Test baseline first
    if args.test_baseline:
        test_baseline(env_name)
    
    # Train PPO
    episode_rewards = train_ppo(env_name, num_episodes=args.episodes, max_timesteps=1000)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.title(f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{args.task}_ppo_working_results.png')
    plt.show()

if __name__ == "__main__":
    main()
