#!/usr/bin/env python3
"""
Baseline MLP Test Script - Their Exact MLP Implementation

This script implements their exact MLP architecture from main.ipynb and integrates it
with our continual learning W&B system for fair comparison with our topology networks.

Key Features:
- Their exact PolicyNetwork and ValueNetwork (3 hidden layers, 128 nodes, LeakyReLU)
- Adam optimizer (no TRAC for now)
- Same continual learning setup as our topologies
- Same W&B project and logging patterns
- Same single plot: iterations vs. mean episode reward
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from collections import deque

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Device setup (will be set dynamically in main())
device = None

# ============================================================================
# THEIR EXACT MLP IMPLEMENTATION FROM MAIN.IPYNB
# ============================================================================

class PolicyNetwork(nn.Module):
    """Their exact PolicyNetwork implementation."""
    
    def __init__(self, n=4, in_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # Their exact architecture: 3 hidden layers of 128 nodes each
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n)
        self.l_relu = nn.LeakyReLU(0.1)
        
        # Calculate total parameters for logging
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 PolicyNetwork: {self.total_params:,} parameters")
        
        # Verify parameter count matches their implementation
        expected_params = in_dim * 128 + 128 + 128 * 128 + 128 + 128 * 128 + 128 + 128 * n + n
        print(f"📊 Expected parameters: {expected_params:,}")
        assert self.total_params == expected_params, f"Parameter count mismatch: {self.total_params} vs {expected_params}"

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)
        y = F.softmax(y, dim=-1)
        return y

    def sample_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        
        y = self(state)
        dist = Categorical(y)
        action = dist.sample()
        log_probability = dist.log_prob(action)
        
        return action.item(), log_probability.item()

    def best_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        
        y = self(state).squeeze()
        action = torch.argmax(y)
        
        return action.item()

    def evaluate_actions(self, states, actions):
        y = self(states)
        dist = Categorical(y)
        entropy = dist.entropy()
        log_probabilities = dist.log_prob(actions)
        
        return log_probabilities, entropy


class ValueNetwork(nn.Module):
    """Their exact ValueNetwork implementation."""
    
    def __init__(self, in_dim=128):
        super(ValueNetwork, self).__init__()
        
        # Their exact architecture: 3 hidden layers of 128 nodes each
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.l_relu = nn.LeakyReLU(0.1)
        
        # Calculate total parameters for logging
        self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 ValueNetwork: {self.total_params:,} parameters")
        
        # Verify parameter count matches their implementation
        expected_params = in_dim * 128 + 128 + 128 * 128 + 128 + 128 * 128 + 128 + 128 * 1 + 1
        print(f"📊 Expected parameters: {expected_params:,}")
        assert self.total_params == expected_params, f"Parameter count mismatch: {self.total_params} vs {expected_params}"

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))
        y = self.fc4(x)
        return y.squeeze(1)

    def state_value(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
        
        y = self(state)
        return y.item()


# ============================================================================
# CONTINUAL LEARNING WRAPPER (REUSED FROM OUR SYSTEM)
# ============================================================================

class ContinualLearningWrapper(gym.Wrapper):
    """
    Wrapper for continual learning with piecewise-constant observation shifts.
    
    PAPER-ACCURATE IMPLEMENTATION:
    - Iteration-based training (not step-based)
    - Pre-generated perturbations for all levels
    - Level 0 (iterations 0-199): NO NOISE - Clean baseline learning
    - Level 1+ (iterations 200+): Perturbations applied every 200 iterations
    - Reward scaling: Division by 20 (creates small gradients)
    - Episode capping at 400 steps maximum
    """
    
    def __init__(self, env, task_name, max_iterations=3000, level_switch=200, shift_range=[0, 2], seed=None, reward_scale=20.0, episode_cap=400, num_levels=15, no_noise=False):
        super().__init__(env)
        self.task_name = task_name
        self.max_iterations = max_iterations
        self.level_switch = level_switch
        self.shift_range = shift_range
        self.seed = seed
        self.reward_scale = reward_scale  # This will be used for division, not multiplication
        self.episode_cap = episode_cap
        self.num_levels = num_levels  # Number of distribution shift levels
        self.no_noise = no_noise
        
        # Initialize iteration and level tracking
        self.current_iteration = 0
        self.current_level = 0
        self.episodes_in_current_iteration = 0
        self.max_episodes_per_iteration = 2
        
        # Episode tracking
        self.episode_steps = 0
        
        # Pre-generate all perturbation levels using seed
        if seed is not None:
            self.perturbation_rng = np.random.RandomState(seed)
        else:
            self.perturbation_rng = np.random.RandomState(42)
        
        # Generate perturbations for all levels
        obs_dim = self.observation_space.shape[0]
        self.perturbations = []
        
        for level in range(self.num_levels):
            if level == 0:
                # Level 0: NO NOISE (clean baseline learning)
                self.perturbations.append(np.zeros(obs_dim))
            else:
                # Level 1+: Random perturbations (or no noise if ablation study)
                if self.no_noise:
                    perturbation = np.zeros(obs_dim)
                else:
                    perturbation = self.perturbation_rng.normal(
                        loc=0,  # Mean = 0 (centered around zero)
                        scale=shift_range[1],  # Standard deviation = shift_range[1] = 2
                        size=obs_dim
                    )
                self.perturbations.append(perturbation)
        print(f"🎯 Continual Learning Setup:")
        print(f"   • Task: {task_name}")
        print(f"   • Max Iterations: {max_iterations}")
        print(f"   • Level Switch: Every {level_switch} iterations")
        print(f"   • Total Levels: {num_levels}")
        print(f"   • Shift Range: {shift_range} (Gaussian std={shift_range[1]})")
        print(f"   • Reward Scale: ÷{reward_scale}")
        print(f"   • Episode Cap: {episode_cap} steps")
        print(f"   • Episodes per Iteration: {self.max_episodes_per_iteration}")
        if self.no_noise:
            print(f"   🚫 NOISE DISABLED: Running no-noise ablation study")
        else:
            print(f"   🔧 Noise enabled: Gaussian perturbations applied")
    
    def set_iteration(self, iteration):
        """Set current iteration and update perturbation level."""
        self.current_iteration = iteration
        new_level = iteration // self.level_switch
        
        # Only update and log if the level actually changed
        if new_level != self.current_level:
            self.current_level = new_level
            
            # Ensure we don't exceed the number of pre-generated perturbations
            if self.current_level < len(self.perturbations):
                self.current_perturbation = self.perturbations[self.current_level]
            else:
                # If we exceed, use the last perturbation
                self.current_perturbation = self.perturbations[-1]
                self.current_level = len(self.perturbations) - 1
            
            # Log the level activation (only when it changes)
            if self.current_level == 0:
                print(f"\n🎯 NEW NOISE LEVEL ACTIVATED:")
                print(f"   🧹 Level {self.current_level}: Clean Baseline (NO NOISE)")
                print(f"   📍 Iteration: {iteration}")
                print(f"   📊 Environment Steps: ~{iteration * 800:,}")
            else:
                print(f"\n🎯 NEW NOISE LEVEL ACTIVATED:")
                if self.no_noise:
                    # No-noise ablation study - show that no noise is applied
                    print(f"   🚫 Level {self.current_level}: No-Noise Ablation (ZERO PERTURBATION)")
                    print(f"   📍 Iteration: {iteration}")
                    print(f"   📊 Environment Steps: ~{iteration * 800:,}")
                    print(f"   🔧 Perturbation: ZERO (Noise disabled for ablation study)")
                else:
                    # Normal noise study - show actual perturbation
                    print(f"   📊 Level {self.current_level}: Noise Vector Applied")
                    print(f"   📍 Iteration: {iteration}")
                    print(f"   📊 Environment Steps: ~{iteration * 800:,}")
                    print(f"   🔧 Perturbation: {self.current_perturbation}")
        else:
            # Just update the perturbation without logging
            if self.current_level < len(self.perturbations):
                self.current_perturbation = self.perturbations[self.current_level]
            else:
                self.current_perturbation = self.perturbations[-1]
        
        self.episodes_in_current_iteration = 0
        return self.current_perturbation
    
    def reset(self, **kwargs):
        """Reset environment and apply current perturbation."""
        obs, info = super().reset(**kwargs)
        
        # Reset episode step counter
        self.episode_steps = 0
        
        # Apply current perturbation ONLY if noise is enabled
        if not self.no_noise and self.current_level < len(self.perturbations):
            obs += self.perturbations[self.current_level]
        
        return obs, info
    
    def step(self, action):
        """Take step and apply current perturbation."""
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Increment episode step counter
        self.episode_steps += 1
        
        # Apply current perturbation ONLY if noise is enabled
        if not self.no_noise and self.current_level < len(self.perturbations):
            obs += self.perturbations[self.current_level]
        
        # Scale reward (division by reward_scale)
        reward = reward / self.reward_scale
        
        # Cap episode length
        if self.episode_cap > 0:
            truncated = truncated or (self.episode_cap > 0 and self.episode_steps >= self.episode_cap)
        
        return obs, reward, terminated, truncated, info
    
    def _show_progress(self):
        """Show progress using tqdm for smooth, continuous tracking (matching our topology system)."""
        # This method is called by set_iteration but progress bars are handled in the main training loop
        # to avoid conflicts and ensure proper display
        pass


# ============================================================================
# TRAINING LOOP (THEIR METHODOLOGY + OUR W&B INTEGRATION)
# ============================================================================

def train_baseline_mlp(env_name, seed=42, num_levels=15, use_wandb=True, no_noise=False):
    """
    Train their exact MLP using their methodology but with our W&B integration.
    
    Args:
        env_name: Environment name (CartPole-v1, Acrobot-v1, LunarLander-v2)
        seed: Random seed for reproducibility
        num_levels: Number of distribution shift levels
        use_wandb: Whether to use W&B logging
    """
    
    # Configuration (matching their ACTUAL implementation from main.ipynb)
    max_iterations = num_levels * 200  # Total iterations = num_levels × 200
    level_switch = 200                 # Switch perturbation every 200 iterations
    shift_range = [0, 2]              # Uniform[0, 2] per dimension
    episode_cap = 400                  # Max episode length
    reward_scale = 20.0                # Division factor (creates small gradients)
    max_episodes_per_iteration = 2     # Episodes per iteration
    
    # Hyperparameters (matching their ACTUAL implementation)
    lr = 0.01                          # Learning rate
    max_epochs = 40                    # Training epochs per batch (THEIR ACTUAL VALUE)
    clip_epsilon = 0.2                 # PPO clip range
    c1 = 0.01                          # Entropy coefficient
    c2 = 0.5                           # Value loss coefficient
    gamma = 0.99                       # Discount factor
    gae_lambda = 0.95                  # GAE lambda
    batch_size = 32                    # Batch size for training
    
    print(f"🚀 Starting Baseline MLP Training (THEIR ACTUAL CONFIGURATION)")
    print(f"   Environment: {env_name}")
    print(f"   Seed: {seed}")
    print(f"   Number of Levels: {num_levels}")
    print(f"   Max Iterations: {max_iterations}")
    print(f"   Level Switch: Every {level_switch} iterations")
    print(f"   Max Episodes per Iteration: {max_episodes_per_iteration}")
    print(f"   Training Epochs per Batch: {max_epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   W&B Logging: {'Enabled' if use_wandb else 'Disabled'}")
    print("=" * 80)
    
    # Initialize W&B if enabled
    if use_wandb:
        # Create run name following our naming convention
        run_name = create_baseline_mlp_run_name(env_name, seed, num_levels, max_iterations, level_switch, shift_range, no_noise)
        
        wandb.init(
            project="topologies--continual-learning-training",
            name=run_name,
            config={
                'task_name': env_name,
                'topology_type': 'baseline_mlp',
                'seed': seed,
                'max_iterations': max_iterations,
                'level_switch': level_switch,
                'shift_range': shift_range,
                'reward_scale': reward_scale,
                'episode_cap': episode_cap,
                'num_levels': num_levels,
                'learning_rate': lr,
                'max_epochs': max_epochs,
                'clip_epsilon': clip_epsilon,
                'entropy_coef': c1,
                'value_coef': c2,
                'gamma': gamma,
                'gae_lambda': gae_lambda,
                'max_episodes_per_iteration': max_episodes_per_iteration,
                'batch_size': batch_size
            },
            tags=["baseline_mlp", "continual_learning", "paper_accurate", "iteration_based", "their_implementation"]
        )
        
        print(f"✅ W&B initialized: {run_name}")
    
    # Create environment with continual learning wrapper
    env = gym.make(env_name)
    env = ContinualLearningWrapper(
        env, 
        env_name, 
        max_iterations, 
        level_switch, 
        shift_range, 
        seed, 
        reward_scale, 
        episode_cap, 
        num_levels,
        no_noise
    )
    
    # Get environment dimensions
    observation = env.reset()[0]
    n_actions = env.action_space.n
    feature_dim = observation.size
    
    print(f"📊 Environment Details:")
    print(f"   • Observation Dimension: {feature_dim}")
    print(f"   • Action Space: {n_actions} actions")
    print(f"   • Feature Dimension: {feature_dim}")
    
    # Create networks (their exact implementation)
    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    
    # Create optimizer (Adam, matching their implementation)
    combined_optimizer = optim.Adam([
        {'params': policy_model.parameters(), 'lr': lr},
        {'params': value_model.parameters(), 'lr': lr},
    ])
    
    print(f"📊 Network Details:")
    print(f"   • Policy Network: {policy_model.total_params:,} parameters")
    print(f"   • Value Network: {value_model.total_params:,} parameters")
    print(f"   • Total Parameters: {policy_model.total_params + value_model.total_params:,}")
    print(f"   • Optimizer: Adam (lr={lr})")
    
    # Initialize tracking variables
    iteration_rewards = []
    episodes_rewards = []  # Track individual episode rewards
    total_env_steps = 0
    current_pbar = None
    training_start_time = time.time()  # Track total training time
    
    # Main training loop
    for current_iteration in range(max_iterations):
        # Set current iteration in environment wrapper (this will show progress and level changes)
        current_perturbation = env.set_iteration(current_iteration)
        
        # Create/update tqdm progress bar for this level
        if current_iteration % level_switch == 0:
            level = current_iteration // level_switch
            if current_pbar:
                current_pbar.close()
            current_pbar = tqdm(
                total=level_switch,
                desc=f"Level {level:2d} ({current_iteration:4d}-{min(current_iteration + level_switch - 1, max_iterations - 1):4d})",
                unit="iter",
                position=0,
                leave=True
            )
        
        # Collect episodes for this iteration (matching their methodology)
        episodes_data = []
        iteration_steps = 0
        iteration_episode_rewards = []  # Track rewards for this iteration
        
        for episode_idx in range(max_episodes_per_iteration):
            # Reset environment and apply perturbation
            observation = env.reset()[0]
            observation += current_perturbation
            episode_reward = 0
            episode_steps = 0
            
            # Run one episode
            for step in range(episode_cap):
                # Get action from policy
                action, log_prob = policy_model.sample_action(observation / reward_scale)
                value = value_model.state_value(observation / reward_scale)
                
                # Take step in environment
                next_observation, reward, done, truncated, info = env.step(action)
                next_observation += current_perturbation
                
                # Store transition (matching their data structure)
                episodes_data.append({
                    'observation': observation / reward_scale,
                    'action': action,
                    'reward': reward,
                    'value': value,
                    'log_probability': log_prob,
                    'done': done
                })
                
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                iteration_steps += 1
                
                if done or truncated:
                    break
            
            # Store episode summary
            iteration_episode_rewards.append(episode_reward)
            episodes_rewards.append(episode_reward)
        
        # Calculate mean reward for this iteration
        mean_iteration_reward = np.mean(iteration_episode_rewards)
        iteration_rewards.append(mean_iteration_reward)
        
        # Update progress bar
        if current_pbar:
            current_pbar.set_postfix({
                'Level': current_iteration // level_switch,
                'Reward': f"{mean_iteration_reward:.1f}",
                'Steps': iteration_steps
            })
            current_pbar.update(1)
        
        # Train on the collected batch (matching their methodology)
        if len(episodes_data) > 0:
            # Convert to tensors for batch training
            observations = torch.tensor([d['observation'] for d in episodes_data], dtype=torch.float32).to(device)
            actions = torch.tensor([d['action'] for d in episodes_data], dtype=torch.long).to(device)
            rewards = torch.tensor([d['reward'] for d in episodes_data], dtype=torch.float32).to(device)
            values = torch.tensor([d['value'] for d in episodes_data], dtype=torch.float32).to(device)
            old_log_probs = torch.tensor([d['log_probability'] for d in episodes_data], dtype=torch.float32).to(device)
            
            # Calculate advantages and returns (simplified GAE)
            advantages = rewards - values.detach()
            returns = rewards
            
            # Train for multiple epochs on this batch (matching their 40 epochs)
            for epoch in range(max_epochs):
                # Policy update
                new_log_probs, entropy = policy_model.evaluate_actions(observations, actions)
                policy_loss = -torch.mean(new_log_probs * advantages.detach()) - 0.01 * entropy.mean()
                
                # Value update
                new_values = value_model(observations)
                value_loss = 0.5 * torch.mean((new_values - returns) ** 2)
                
                # Combined update
                combined_optimizer.zero_grad()
                total_loss = policy_loss + 0.5 * value_loss
                total_loss.backward()
                combined_optimizer.step()
        
        # Update total environment steps
        total_env_steps += iteration_steps
        
        # Log iteration results to W&B
        if use_wandb and wandb.run:
            wandb.log({
                'training/iteration': current_iteration,
                'training/level': current_iteration // level_switch,
                'training/mean_episode_reward': mean_iteration_reward,
                'training/iteration_steps': iteration_steps,
                'training/total_env_steps': total_env_steps,
                'training/mean_reward_last_episodes': mean_iteration_reward,
                'training/episodes_in_iteration': len(iteration_episode_rewards)
            }, step=current_iteration)
        
        # Print progress only at the end of each level to avoid interfering with progress bars
        if current_iteration == max_iterations - 1:
            elapsed_time = time.time() - training_start_time
            print(f"\n📊 Final Iteration {current_iteration:4d}: Level {current_iteration // level_switch:2d}, Mean Reward: {mean_iteration_reward:8.3f}, Time: {elapsed_time:.1f}s")
        
        # Create and log plot every 200 iterations (level changes) - matching our system
        if current_iteration % level_switch == 0 and current_iteration > 0:
            _create_and_log_iteration_plot(iteration_rewards, env_name, seed, use_wandb)
    
    # Close final progress bar
    if current_pbar:
        current_pbar.close()
    
    # Final plot
    _create_and_log_iteration_plot(iteration_rewards, env_name, seed, use_wandb)
    
    # Final training summary (matching our topology system)
    total_training_time = time.time() - training_start_time
    
    print(f"\n🎯 Training completed! Total iterations: {max_iterations}")
    print(f"   Total environment steps: {total_env_steps:,}")
    print(f"   Total perturbation levels: {max_iterations // level_switch}")
    print(f"   Training time: {total_training_time:.1f}s")
    print(f"   Final Mean Reward: {mean_iteration_reward:.3f}")
    
    # Log final metrics to W&B
    if use_wandb and wandb.run:
        wandb.log({
            'training/final_mean_reward': mean_iteration_reward,
            'training/total_time': total_training_time,
            'training/total_iterations': max_iterations,
            'training/total_env_steps': total_env_steps,
            'training/total_levels': max_iterations // level_switch
        })
        wandb.finish()
    
    return policy_model, value_model, iteration_rewards


def _create_and_log_iteration_plot(iteration_rewards, task_name, seed, use_wandb):
    """Create and log the iteration vs mean episode reward plot (matching our system)."""
    if not use_wandb or not wandb.run:
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot iterations vs mean episode reward
    iterations = list(range(len(iteration_rewards)))
    ax.plot(iterations, iteration_rewards, 'b-', linewidth=2, alpha=0.8)
    
    # Add level boundaries
    level_switch = 200
    for level in range(0, len(iteration_rewards), level_switch):
        if level < len(iteration_rewards):
            ax.axvline(x=level, color='r', linestyle='--', alpha=0.5, label=f'Level {level//level_switch}' if level == 0 else "")
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title(f'{task_name} - Iterations vs Mean Episode Reward (Seed: {seed})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Log to W&B
    wandb.log({
        'plots/iterations_vs_mean_episode_reward': wandb.Image(fig)
    })
    
    plt.close(fig)


def create_baseline_mlp_run_name(env_name, seed, num_levels, max_iterations, level_switch, shift_range, no_noise=False):
    """
    Create run name for baseline MLP experiments.
    
    Format: BASELINE_MLP_{network_details}_{task_abbrev}_{seed}_{experiment_details}
    Example: BASELINE_MLP_L3_S128_P33922_CP_seed42_L5_I1000_LS200_N00_LReLU
    """
    # Task abbreviation
    task_abbrev = {
        'LunarLander-v2': 'LL',
        'Acrobot-v1': 'AC', 
        'CartPole-v1': 'CP',
        'MountainCar-v0': 'MC'
    }.get(env_name, env_name[:2].upper())
    
    # Noise interval from shift_range (or no noise if ablation study)
    if no_noise:
        noise_interval = 'N00'  # No noise ablation study
    elif shift_range and len(shift_range) == 2:
        noise_interval = f"N{int(shift_range[0]):02d}{int(shift_range[1]):02d}"
    else:
        noise_interval = 'N00'  # Default fallback
    
    # Build name parts
    name_parts = [
        'BASELINE_MLP',
        f"L3",  # 3 hidden layers (their actual implementation)
        f"S128",  # 128 nodes per layer
        f"P33922",  # Parameter count placeholder
        task_abbrev,
        f"seed{seed}",
        f"L{num_levels}",
        f"I{max_iterations}",
        f"LS{level_switch}",
        noise_interval,
        "LReLU"  # LeakyReLU activation function identifier
    ]
    
    return "_".join(name_parts)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Baseline MLP Training with Continual Learning")
    parser.add_argument("--task", type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v2"],
                       help="Environment to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_levels", type=int, default=15, 
                       help="Number of distribution shift levels (default: 15)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA for training")
    parser.add_argument("--no_noise", action="store_true", help="Disable all perturbation noise for ablation study")
    
    args = parser.parse_args()
    
    print("🚀 Baseline MLP Training - Their Exact Implementation")
    print("=" * 80)
    print(f"🎯 Configuration:")
    print(f"   Task: {args.task}")
    print(f"   Seed: {args.seed}")
    print(f"   Number of Levels: {args.num_levels}")
    print(f"   W&B: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"   CUDA: {'Disabled' if args.no_cuda else 'Enabled'}")
    print("=" * 80)
    
    global device
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA is available. Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️ CUDA is not available or disabled. Using CPU: {device}")
    
    try:
        # Train the baseline MLP
        policy_model, value_model, rewards = train_baseline_mlp(
            env_name=args.task,
            seed=args.seed,
            num_levels=args.num_levels,
            use_wandb=not args.no_wandb,
            no_noise=args.no_noise
        )
        
        print("✅ Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"   • Policy Model: {policy_model.total_params:,} parameters")
        print(f"   • Value Model: {value_model.total_params:,} parameters")
        print(f"   • Total Parameters: {policy_model.total_params + value_model.total_params:,}")
        print(f"   • Final Mean Reward: {rewards[-1]:.3f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
