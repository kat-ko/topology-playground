#!/usr/bin/env python3
"""
Simple test to investigate episode counting mismatch.
"""

import numpy as np
import gymnasium as gym

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

class ContinualLearningWrapper:
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
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_returns = []
        self.total_env_steps = 0
        
        print(f"🎲 Test wrapper initialized")
    
    def set_iteration(self, iteration):
        self.current_iteration = iteration
        self.episodes_in_current_iteration = 0
        
        new_level = iteration // self.level_switch
        if new_level != self.current_level:
            self.current_level = new_level
            print(f"🎯 Level {self.current_level} activated at iteration {iteration}")
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_step += 1
        self.total_env_steps += 1
        self.episode_reward += reward
        
        # Check episode termination
        episode_ended = done or truncated or self.episode_step >= self.episode_cap
        
        if episode_ended:
            print(f"🔍 Episode ended: step={self.episode_step}, reward={self.episode_reward:.2f}, iteration={self.current_iteration}")
            self._log_episode()
            self._reset_episode()
            self.episodes_in_current_iteration += 1
        
        return obs, reward, episode_ended, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode()
        return obs, info
    
    def _log_episode(self):
        episode_info = {
            'global_step_end': self.total_env_steps,
            'episode_length': self.episode_step,
            'episode_return_raw': self.episode_reward,
            'shift_id': self.current_level,
            'iteration': self.current_iteration,
        }
        self.episode_returns.append(episode_info)
    
    def _reset_episode(self):
        self.episode_step = 0
        self.episode_reward = 0.0
    
    def get_current_info(self):
        return {
            'current_iteration': self.current_iteration,
            'current_level': self.current_level,
            'episodes_in_iteration': self.episodes_in_current_iteration,
            'total_env_steps': self.total_env_steps,
            'total_episodes': len(self.episode_returns)
        }

def test_episode_counting():
    print("🧪 Testing Episode Counting Logic")
    print("=" * 50)
    
    # Create mock environment
    env = MockEnv(max_steps=400)
    
    # Create wrapper
    wrapper = ContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=10,
        level_switch=5,
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=2,
        no_noise=False
    )
    
    print(f"📊 Starting test with 10 iterations, 2 episodes per iteration")
    print(f"   Expected episodes: 10 iterations × 2 episodes = 20 episodes")
    
    # Main iteration loop
    current_iteration = 0
    max_iterations = 10
    total_training_episodes = 0
    
    while current_iteration < max_iterations:
        print(f"\n🔄 Iteration {current_iteration + 1}/{max_iterations}")
        
        # Set current iteration in environment wrapper
        wrapper.set_iteration(current_iteration)
        
        # Collect exactly 2 episodes per iteration
        episodes_data = []
        iteration_episode_rewards = []
        
        print(f"   📝 Collecting 2 episodes...")
        
        for episode_idx in range(2):
            print(f"      📝 Episode {episode_idx + 1}/2")
            
            # Reset environment for new episode
            observation = wrapper.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            
            # Run one episode
            for step in range(wrapper.episode_cap):
                action = np.random.randint(0, 2)
                next_observation, reward, done, truncated, info = wrapper.step(action)
                
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
                'iteration': current_iteration
            }
            episodes_data.append(episode_data)
            iteration_episode_rewards.append(episode_reward)
            total_training_episodes += 1
        
        print(f"   📊 Iteration complete:")
        print(f"      • Episodes collected: {len(episodes_data)}")
        print(f"      • Mean episode reward: {np.mean(iteration_episode_rewards):.2f}")
        
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
        print(f"   This explains the 188,771 vs 6,000 discrepancy!")
    else:
        print(f"\n✅ Episode counts match! No issue detected.")
    
    return wrapper, total_training_episodes

if __name__ == "__main__":
    print("🚀 Episode Counting Investigation Test")
    print("=" * 50)
    
    wrapper, training_episodes = test_episode_counting()
    
    print(f"\n🎯 Investigation complete!")
    print(f"   Check the output above to understand the episode counting mismatch.")
