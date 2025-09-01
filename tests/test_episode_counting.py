#!/usr/bin/env python3
"""
Test file to investigate episode counting mismatch between main training loop and environment wrapper.
This will help us understand why we're seeing 188,771 episodes instead of 6,000.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any, Optional
import time

# Import the ContinualLearningWrapper from the main script
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the imports that the wrapper needs
class MockLoggingCallback:
    def _log_episode_completion(self, episode_info):
        pass
    
    def _log_perturbation_level_change(self, iteration, level, perturbation):
        pass

class MockEnv:
    def __init__(self, max_steps=400):
        self.max_steps = max_steps
        self.step_count = 0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)
    
    def step(self, action):
        self.step_count += 1
        
        # Simulate CartPole behavior - episodes end early
        if self.step_count >= self.max_steps:
            done = True
            truncated = False
        elif np.random.random() < 0.1:  # 10% chance of early termination
            done = True
            truncated = False
        else:
            done = False
            truncated = False
        
        # Simulate observation and reward
        obs = np.random.random(4)
        reward = 1.0 if not done else 0.0
        
        return obs, reward, done, truncated, {}
    
    def reset(self, **kwargs):
        self.step_count = 0
        return np.random.random(4), {}

# Copy the ContinualLearningWrapper class for testing
class ContinualLearningWrapper:
    """
    Simplified version of the ContinualLearningWrapper for testing.
    """
    
    def __init__(self, env, task_name, max_iterations=3000, level_switch=200, shift_range=[0, 1], seed=None, reward_scale=20.0, episode_cap=400, logging_callback=None, num_levels=15, no_noise=False):
        self.env = env
        self.task_name = task_name
        self.max_iterations = max_iterations
        self.level_switch = level_switch
        self.shift_range = shift_range
        self.no_noise = no_noise
        self.seed = seed
        self.reward_scale = reward_scale
        self.episode_cap = episode_cap
        self.logging_callback = logging_callback
        self.num_levels = num_levels
        
        # Initialize iteration and level tracking
        self.current_iteration = 0
        self.current_level = 0
        self.episodes_in_current_iteration = 0
        self.max_episodes_per_iteration = 2
        
        # Pre-generate all perturbation levels using seed
        if seed is not None:
            self.perturbation_rng = np.random.RandomState(seed)
        else:
            self.perturbation_rng = np.random.RandomState(42)
        
        # Generate perturbations for all levels
        obs_dim = self.env.observation_space.shape[0]
        self.perturbations = []
        
        # Ensure we have at least one level (even if num_levels is 0)
        effective_levels = max(1, self.num_levels)
        
        for level in range(effective_levels):
            if level == 0:
                # Level 0: NO NOISE - Clean baseline
                perturbation = np.zeros(obs_dim)
            else:
                # Levels 1+: Random perturbations
                if self.no_noise:
                    perturbation = np.zeros(obs_dim)
                else:
                    perturbation = self.perturbation_rng.normal(
                        self.shift_range[0], 
                        self.shift_range[1], 
                        obs_dim
                    )
            self.perturbations.append(perturbation)
        
        # Set initial perturbation (Level 0 = no noise)
        self.current_perturbation = self.perturbations[0]
        
        # Episode tracking for capping
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_returns = []
        
        # Environment step counting (for logging)
        self.total_env_steps = 0
        
        print(f"🎲 Test Continual Learning Wrapper initialized:")
        print(f"   Task: {task_name}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Level switch: {level_switch} iterations")
        print(f"   Number of levels: {self.num_levels}")
        print(f"   Shift range: {shift_range}")
        print(f"   Episode cap: {episode_cap} steps")
        print(f"   Max episodes per iteration: {self.max_episodes_per_iteration}")
    
    def set_iteration(self, iteration):
        """Set the current iteration and update perturbation level accordingly."""
        self.current_iteration = iteration
        
        # Reset episode counter for new iteration to prevent accumulation
        self.episodes_in_current_iteration = 0
        
        # Calculate which perturbation level we're in
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
            
            print(f"\n🎯 NEW NOISE LEVEL ACTIVATED:")
            if self.current_level == 0:
                print(f"   🧹 Level {self.current_level}: Clean Baseline (NO NOISE)")
            else:
                print(f"   📊 Level {self.current_level}: Noise Vector Applied")
            print(f"   📍 Iteration: {iteration}")
            print(f"   📊 Environment Steps: ~{iteration * 800:,}")
    
    def step(self, action):
        """Step environment and apply current observation shift with reward scaling and episode capping."""
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Apply current perturbation to observation ONLY if noise is enabled
        if self.no_noise:
            shifted_obs = obs
        else:
            shifted_obs = obs + self.current_perturbation
        
        # Apply reward scaling (divide by 20 for training)
        scaled_reward = reward / self.reward_scale
        
        # Store raw reward for logging
        self.episode_reward += reward
        
        # Update episode tracking
        self.episode_step += 1
        self.total_env_steps += 1
        
        # Check episode termination (cap at episode_cap steps)
        episode_ended = done or truncated or self.episode_step >= self.episode_cap
        
        if episode_ended:
            print(f"🔍 Episode ended: step={self.episode_step}, reward={self.episode_reward:.2f}, iteration={self.current_iteration}")
            self._log_episode()
            self._reset_episode()
            
            # Increment episode counter for current iteration
            self.episodes_in_current_iteration += 1
        
        return shifted_obs, scaled_reward, episode_ended, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment and maintain perturbation state."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self._reset_episode()
        
        # Apply current perturbation to reset observation ONLY if noise is enabled
        if self.no_noise:
            shifted_obs = obs
        else:
            shifted_obs = obs + self.current_perturbation
        
        return shifted_obs, info
    
    def _log_episode(self):
        """Log episode completion with raw returns and iteration information."""
        episode_info = {
            'global_step_end': self.total_env_steps,
            'episode_length': self.episode_step,
            'episode_return_raw': self.episode_reward,
            'episode_return_scaled': self.episode_reward * self.reward_scale,
            'shift_id': self.current_level,
            'iteration': self.current_iteration,
            'level': self.current_level,
            'perturbation_applied': self.current_perturbation.tolist(),
            'shift_boundary': (self.current_iteration % self.level_switch == 0)
        }
        
        # Store episode data
        self.episode_returns.append(episode_info)
        
        # Use enhanced logging callback if available
        if self.logging_callback and hasattr(self.logging_callback, '_log_episode_completion'):
            try:
                self.logging_callback._log_episode_completion(episode_info)
            except Exception as e:
                print(f"⚠️  Episode logging callback failed: {e}")
    
    def _reset_episode(self):
        """Reset episode tracking."""
        self.episode_step = 0
        self.episode_reward = 0.0
    
    def get_current_info(self):
        """Get current wrapper state information."""
        return {
            'current_iteration': self.current_iteration,
            'current_level': self.current_level,
            'current_perturbation': self.current_perturbation.copy(),
            'episodes_in_iteration': self.episodes_in_current_iteration,
            'total_env_steps': self.total_env_steps,
            'total_episodes': len(self.episode_returns)
        }

def test_episode_counting():
    """Test the episode counting logic to understand the mismatch."""
    print("🧪 Testing Episode Counting Logic")
    print("=" * 60)
    
    # Create mock environment
    env = MockEnv(max_steps=400)
    
    # Create wrapper
    wrapper = ContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=10,  # Test with 10 iterations
        level_switch=5,     # Switch levels every 5 iterations
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=2,       # Only 2 levels for testing
        no_noise=False
    )
    
    print(f"\n📊 Starting test with 10 iterations, 2 episodes per iteration")
    print(f"   Expected episodes: 10 iterations × 2 episodes = 20 episodes")
    print(f"   Expected timesteps: ~10 iterations × 800 steps = ~8,000 steps")
    
    # Main iteration loop (simplified version of the training loop)
    current_iteration = 0
    max_iterations = 10
    total_training_episodes = 0
    
    while current_iteration < max_iterations:
        print(f"\n🔄 Iteration {current_iteration + 1}/{max_iterations}")
        
        # Set current iteration in environment wrapper
        wrapper.set_iteration(current_iteration)
        
        # Phase 1: Collect 2 episodes per iteration
        episodes_data = []
        iteration_episode_rewards = []
        
        print(f"   📝 Collecting 2 episodes...")
        
        # Collect exactly 2 episodes per iteration
        for episode_idx in range(2):
            print(f"      📝 Episode {episode_idx + 1}/2")
            
            # Reset environment for new episode
            observation = wrapper.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            episode_transitions = []
            
            # Run one episode
            for step in range(wrapper.episode_cap):
                # Simulate action
                action = np.random.randint(0, 2)
                
                # Take environment step
                next_observation, reward, done, truncated, info = wrapper.step(action)
                
                # Store transition data
                transition = {
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'done': done,
                    'truncated': truncated,
                    'step': step
                }
                episode_transitions.append(transition)
                
                # Update tracking
                episode_reward += reward
                episode_steps += 1
                observation = next_observation
                
                if done or truncated:
                    print(f"         ✅ Episode ended at step {episode_steps}, reward: {episode_reward:.2f}")
                    break
            
            # Store episode data
            episode_data = {
                'episode_idx': episode_idx,
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'transitions': episode_transitions,
                'iteration': current_iteration
            }
            episodes_data.append(episode_data)
            iteration_episode_rewards.append(episode_reward)
            total_training_episodes += 1
        
        print(f"   📊 Iteration complete:")
        print(f"      • Episodes collected: {len(episodes_data)}")
        print(f"      • Mean episode reward: {np.mean(iteration_episode_rewards):.2f}")
        print(f"      • Total transitions: {sum(len(ep['transitions']) for ep in episodes_data)}")
        
        # Get wrapper info
        wrapper_info = wrapper.get_current_info()
        print(f"      • Wrapper episodes in iteration: {wrapper_info['episodes_in_iteration']}")
        print(f"      • Wrapper total episodes: {wrapper_info['total_episodes']}")
        
        # Move to next iteration
        current_iteration += 1
    
    print(f"\n🎯 Test completed!")
    print(f"   Total training episodes: {total_training_episodes}")
    print(f"   Wrapper total episodes: {len(wrapper.episode_returns)}")
    print(f"   Total environment steps: {wrapper.total_env_steps}")
    
    # Analyze the mismatch
    print(f"\n🔍 Analysis:")
    print(f"   Expected episodes: {max_iterations * 2}")
    print(f"   Actual training episodes: {total_training_episodes}")
    print(f"   Actual wrapper episodes: {len(wrapper.episode_returns)}")
    print(f"   Mismatch: {len(wrapper.episode_returns) - total_training_episodes} extra episodes")
    
    if len(wrapper.episode_returns) != total_training_episodes:
        print(f"\n❌ EPISODE COUNT MISMATCH DETECTED!")
        print(f"   The wrapper is logging {len(wrapper.episode_returns)} episodes")
        print(f"   But the training loop only collected {total_training_episodes} episodes")
        print(f"   This explains the 188,771 vs 6,000 discrepancy!")
        
        # Show some wrapper episodes to understand what's being logged
        print(f"\n📋 Sample wrapper episodes:")
        for i, ep in enumerate(wrapper.episode_returns[:5]):
            print(f"   Episode {i+1}: {ep['episode_length']} steps, reward {ep['episode_return_raw']:.2f}, iteration {ep['iteration']}")
    else:
        print(f"\n✅ Episode counts match! No issue detected.")
    
    return wrapper, total_training_episodes

def test_episode_step_counter():
    """Test specifically the episode_step counter behavior."""
    print(f"\n🧪 Testing Episode Step Counter Behavior")
    print("=" * 60)
    
    # Create mock environment
    env = MockEnv(max_steps=400)
    
    # Create wrapper
    wrapper = ContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=2,
        level_switch=1,
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=2,
        no_noise=False
    )
    
    print(f"📊 Testing episode step counter with 2 iterations")
    
    for iteration in range(2):
        print(f"\n🔄 Iteration {iteration}")
        wrapper.set_iteration(iteration)
        
        for episode_idx in range(2):
            print(f"   📝 Episode {episode_idx + 1}")
            
            # Reset environment
            observation = wrapper.reset()[0]
            print(f"      🔄 env.reset() called, wrapper.episode_step = {wrapper.episode_step}")
            
            # Run a few steps
            for step in range(5):  # Just 5 steps for testing
                action = np.random.randint(0, 2)
                next_observation, reward, done, truncated, info = wrapper.step(action)
                
                print(f"         Step {step + 1}: wrapper.episode_step = {wrapper.episode_step}")
                
                if done or truncated:
                    print(f"         ✅ Episode ended at step {step + 1}")
                    break
            
            print(f"      📊 After episode: wrapper.episodes_in_current_iteration = {wrapper.episodes_in_current_iteration}")
    
    print(f"\n📋 Final wrapper state:")
    wrapper_info = wrapper.get_current_info()
    for key, value in wrapper_info.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    print("🚀 Episode Counting Investigation Test")
    print("=" * 60)
    
    # Test 1: Full episode counting logic
    wrapper, training_episodes = test_episode_counting()
    
    # Test 2: Episode step counter behavior
    test_episode_step_counter()
    
    print(f"\n🎯 Investigation complete!")
    print(f"   Check the output above to understand the episode counting mismatch.")

