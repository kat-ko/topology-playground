#!/usr/bin/env python3
"""
Baseline MLP training script that uses the exact PPO implementation from main.ipynb
but with W&B logging like topologies_continual_task_training_sweep.py
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from typing import List, Tuple, Dict, Any
import random

# Set seeds for reproducibility
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Constants from main.ipynb (EXACT VALUES)
levels = 10  # num distribution shifts
lr = 0.01  # Learning rate
max_episodes = 2  # Episodes per iteration
train_epochs = 5  # Training epochs per batch
max_timesteps = 400  # Maximum episode length
state_scale = 1.0  # No observation scaling
reward_scale = 20.0  # Reward scaling factor
batch_size = 32  # Batch size for training
level_switch = 200  # When to introduce distribution shift
max_iterations = levels * level_switch  # Total iterations = levels × level_switch
rp_range = 2  # Perturbation range

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def cumulative_sum(rewards, gamma=0.99):
    """Calculate cumulative discounted rewards"""
    cumulative = []
    running_sum = 0
    for reward in reversed(rewards):
        running_sum = reward + gamma * running_sum
        cumulative.append(running_sum)
    return list(reversed(cumulative))

def normalize_list(values):
    """Normalize a list of values"""
    if not values:
        return values
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]

class Episode:
    def __init__(self, env, policy_model, value_model, state_scale, reward_scale, random_perturbation):
        self.env = env
        self.policy_model = policy_model
        self.value_model = value_model
        self.state_scale = state_scale
        self.reward_scale = reward_scale
        self.random_perturbation = random_perturbation
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probabilities = []
        self.values = []
        
    def run(self):
        observation, _ = self.env.reset()
        # Apply perturbation to initial observation exactly like main.ipynb
        observation += self.random_perturbation
        done = False
        truncated = False
        
        while not done and not truncated:
            # Scale observation for network input
            scaled_obs = observation / self.state_scale
            scaled_obs_tensor = torch.FloatTensor(scaled_obs).unsqueeze(0).to(device)
            
            # Get action probabilities
            action_probs = self.policy_model(scaled_obs_tensor)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Get state value
            value = self.value_model(scaled_obs_tensor)
            
            # Store step data
            self.observations.append(observation)
            self.actions.append(action.item())
            self.log_probabilities.append(log_prob.item())
            self.values.append(value.item())
            
            # Take action
            observation, reward, done, truncated, _ = self.env.step(action.item())
            # Apply perturbation to new observation exactly like main.ipynb
            observation += self.random_perturbation
            
            # Scale reward for training
            scaled_reward = reward / self.reward_scale
            self.rewards.append(scaled_reward)
            
        # Get final value for last observation
        if observation is not None:
            scaled_obs = observation / self.state_scale
            scaled_obs_tensor = torch.FloatTensor(scaled_obs).unsqueeze(0).to(device)
            last_value = self.value_model(scaled_obs_tensor)
            if hasattr(last_value, 'cpu'):
                last_value = last_value.cpu().detach().numpy()
            self.values.append(last_value.item())
        
        return self.observations, self.actions, self.rewards, self.log_probabilities, self.values

class History:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probabilities = []
        self.values = []
        
    def add_episode(self, episode_data):
        obs, actions, rewards, log_probs, values = episode_data
        self.observations.extend(obs)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.log_probabilities.extend(log_probs)
        self.values.extend(values)
        
    def build_dataset(self):
        # Convert to tensors for proper indexing
        self.observations = torch.FloatTensor(self.observations).to(device)
        self.actions = torch.LongTensor(self.actions).to(device)
        self.advantages = torch.FloatTensor(self.advantages).to(device)
        self.log_probabilities = torch.FloatTensor(self.log_probabilities).to(device)
        self.rewards_to_go = torch.FloatTensor(self.rewards_to_go).to(device)
        
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.log_probabilities = []
        self.values = []
        self.advantages = []
        self.rewards_to_go = []

class DataLoader:
    def __init__(self, history, batch_size):
        self.history = history
        self.batch_size = batch_size
        self.n_samples = len(history.observations)
        self.indices = torch.randperm(self.n_samples)
        self.current_idx = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
            
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        batch_obs = self.history.observations[batch_indices]
        batch_actions = self.history.actions[batch_indices]
        batch_advantages = self.history.advantages[batch_indices]
        batch_log_probs = self.history.log_probabilities[batch_indices]
        batch_rewards_to_go = self.history.rewards_to_go[batch_indices]
        
        return batch_obs, batch_actions, batch_advantages, batch_log_probs, batch_rewards_to_go

def train_combined_networks(policy_model, value_model, history, batch_size, train_epochs, lr):
    """Train both policy and value networks using PPO-style updates"""
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    value_optimizer = optim.Adam(value_model.parameters(), lr=lr)
    
    # Calculate advantages and rewards-to-go
    advantages = []
    rewards_to_go = []
    
    for i in range(len(history.rewards)):
        # Calculate advantage using TD(0) error
        if i < len(history.values) - 1:
            advantage = history.rewards[i] + 0.99 * history.values[i + 1] - history.values[i]
        else:
            advantage = history.rewards[i] - history.values[i]
        advantages.append(advantage)
        
        # Calculate rewards-to-go
        reward_to_go = sum(history.rewards[i:] * (0.99 ** np.arange(len(history.rewards) - i)))
        rewards_to_go.append(reward_to_go)
    
    # Normalize advantages
    advantages = normalize_list(advantages)
    
    # Store in history
    history.advantages = advantages
    history.rewards_to_go = rewards_to_go
    
    # Build dataset
    history.build_dataset()
    
    # Training loop
    for epoch in range(train_epochs):
        dataloader = DataLoader(history, batch_size)
        
        for batch_obs, batch_actions, batch_advantages, batch_old_log_probs, batch_rewards_to_go in dataloader:
            # Ensure tensors are on the correct device
            if not isinstance(batch_obs, torch.Tensor):
                batch_obs = torch.FloatTensor(batch_obs).to(device)
            if not isinstance(batch_actions, torch.Tensor):
                batch_actions = torch.LongTensor(batch_actions).to(device)
            if not isinstance(batch_advantages, torch.Tensor):
                batch_advantages = torch.FloatTensor(batch_advantages).to(device)
            if not isinstance(batch_old_log_probs, torch.Tensor):
                batch_old_log_probs = torch.FloatTensor(batch_old_log_probs).to(device)
            if not isinstance(batch_rewards_to_go, torch.Tensor):
                batch_rewards_to_go = torch.FloatTensor(batch_rewards_to_go).to(device)
            
            # Policy update
            action_probs = policy_model(batch_obs)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(batch_actions)
            
            # PPO ratio
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            
            # Value update
            predicted_values = value_model(batch_obs).squeeze()
            value_loss = nn.MSELoss()(predicted_values, batch_rewards_to_go)
            
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

def get_perturbations(env_name, seed):
    """Generate perturbations exactly like main.ipynb"""
    import gymnasium as gym
    env = gym.make(env_name, render_mode="rgb_array")
    observation = env.reset()[0]
    random_perturbations = [
        np.random.normal(0, rp_range, observation.shape) for _ in range(levels)
    ]
    # make the first random perturbation zero
    random_perturbations[0] = np.zeros(observation.shape)
    env.close()
    return random_perturbations

def create_baseline_mlp_run_name(seed, levels, lr, max_episodes, train_epochs, max_timesteps, state_scale, reward_scale, batch_size, level_switch, rp_range):
    """Create a unique run name for baseline MLP training"""
    return f"baseline_mlp_seed{seed}_levels{levels}_lr{lr}_ep{max_episodes}_epochs{train_epochs}_timesteps{max_timesteps}_state{state_scale}_reward{reward_scale}_batch{batch_size}_switch{level_switch}_rp{rp_range}"

def _create_and_log_iteration_plot(iteration, episode_rewards, episode_lengths, wandb_run):
    """Create and log iteration plots to W&B"""
    # Calculate raw rewards (multiply by reward_scale)
    raw_rewards = [reward * reward_scale for reward in episode_rewards]
    
    # Log raw rewards and lengths
    wandb_run.log({
        "iteration": iteration,
        "raw_rewards/mean": np.mean(raw_rewards),
        "raw_rewards/std": np.std(raw_rewards),
        "raw_rewards/min": np.min(raw_rewards),
        "raw_rewards/max": np.max(raw_rewards),
        "episode_lengths/mean": np.mean(episode_lengths),
        "episode_lengths/std": np.std(episode_lengths),
        "episode_lengths/min": np.min(episode_lengths),
        "episode_lengths/max": np.max(episode_lengths),
    })

def main():
    parser = argparse.ArgumentParser(description='Baseline MLP training with main.ipynb PPO implementation')
    parser.add_argument('--seed', type=int, default=seed, help='Random seed')
    parser.add_argument('--levels', type=int, default=levels, help='Number of distribution shifts')
    parser.add_argument('--lr', type=float, default=lr, help='Learning rate')
    parser.add_argument('--max_episodes', type=int, default=max_episodes, help='Episodes per iteration')
    parser.add_argument('--train_epochs', type=int, default=train_epochs, help='Training epochs per batch')
    parser.add_argument('--max_timesteps', type=int, default=max_timesteps, help='Maximum episode length')
    parser.add_argument('--state_scale', type=float, default=state_scale, help='Observation scaling factor')
    parser.add_argument('--reward_scale', type=float, default=reward_scale, help='Reward scaling factor')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size for training')
    parser.add_argument('--level_switch', type=int, default=level_switch, help='When to introduce distribution shift')
    parser.add_argument('--rp_range', type=float, default=rp_range, help='Perturbation range')
    parser.add_argument('--max_iterations', type=int, default=max_iterations, help='Total iterations')
    
    args = parser.parse_args()
    
    # Initialize W&B
    run_name = create_baseline_mlp_run_name(
        args.seed, args.levels, args.lr, args.max_episodes, args.train_epochs,
        args.max_timesteps, args.state_scale, args.reward_scale, args.batch_size,
        args.level_switch, args.rp_range
    )
    
    wandb_run = wandb.init(
        project="topology-playground",
        name=run_name,
        config=vars(args),
        tags=["baseline_mlp", "main_ipynb_ppo"]
    )
    
    print(f"Starting baseline MLP training with run name: {run_name}")
    print(f"Device: {device}")
    print(f"Args: {vars(args)}")
    
    # Initialize environment (CartPole-v1)
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    env.reset(seed=args.seed)
    
    # Generate perturbations exactly like main.ipynb
    print("Generating perturbations...")
    random_perturbations = get_perturbations('CartPole-v1', args.seed)
    print(f"Generated {len(random_perturbations)} perturbation levels")
    print(f"Level 0 (baseline): {random_perturbations[0]}")
    print(f"Level 1: {random_perturbations[1]}")
    
    # Initialize networks
    input_size = env.observation_space.shape[0]
    hidden_size = 64
    output_size = env.action_space.n
    
    policy_model = PolicyNetwork(input_size, hidden_size, output_size).to(device)
    value_model = ValueNetwork(input_size, hidden_size).to(device)
    
    print(f"Policy network: {policy_model}")
    print(f"Value network: {value_model}")
    
    # Training loop
    history = History()
    level = 0  # Start at level 0 (no perturbation)
    
    for iteration in range(args.max_iterations):
        print(f"\nIteration {iteration + 1}/{args.max_iterations}")
        
        # Switch perturbation level exactly like main.ipynb
        if iteration % args.level_switch == 0:
            random_perturbation = random_perturbations[level]
            level += 1
            print(f"  Switching to level {level-1}: perturbation {random_perturbation}")
        
        # Clear history for new iteration
        history.clear()
        
        # Run episodes
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(args.max_episodes):
            print(f"  Episode {episode + 1}/{args.max_episodes}")
            
            # Create episode instance with current perturbation
            episode_instance = Episode(env, policy_model, value_model, args.state_scale, args.reward_scale, random_perturbation)
            
            # Run episode
            obs, actions, rewards, log_probs, values = episode_instance.run()
            
            # Store episode data
            history.add_episode((obs, actions, rewards, log_probs, values))
            
            # Calculate raw rewards for logging
            raw_rewards = [r * args.reward_scale for r in rewards]
            episode_rewards.append(np.sum(raw_rewards))
            episode_lengths.append(len(rewards))
            
            print(f"    Raw episode reward: {np.sum(raw_rewards):.2f}, Length: {len(rewards)}")
        
        # Train networks
        print(f"  Training networks for {args.train_epochs} epochs...")
        train_combined_networks(policy_model, value_model, history, args.batch_size, args.train_epochs, args.lr)
        
        # Log to W&B
        _create_and_log_iteration_plot(iteration + 1, episode_rewards, episode_lengths, wandb_run)
        
        print(f"  Iteration {iteration + 1} complete. Mean raw reward: {np.mean(episode_rewards):.2f}")
    
    print("\nTraining complete!")
    wandb_run.finish()
    env.close()

if __name__ == "__main__":
    main()
