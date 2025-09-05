#!/usr/bin/env python3
"""
Fix for the episode counting mismatch.
This modifies the ContinualLearningWrapper to only log training episodes, not PPO internal episodes.
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
        self.total_resets = 0
    
    def step(self, action):
        self.step_count += 1
        
        # Simulate CartPole behavior - episodes end early
        if self.step_count >= self.max_steps:
            done = True
            truncated = False
        elif np.random.random() < 0.15:  # 15% chance of early termination
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
        self.total_resets += 1
        return np.random.random(4), {}

class FixedContinualLearningWrapper:
    """
    Fixed version of ContinualLearningWrapper that only logs training episodes.
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
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_returns = []
        self.total_env_steps = 0
        
        # NEW: Track whether we're in training mode or PPO internal mode
        self.training_mode = False
        self.training_episode_count = 0
        
        print(f"🎲 Fixed wrapper initialized")
    
    def set_iteration(self, iteration):
        self.current_iteration = iteration
        self.episodes_in_current_iteration = 0
        
        new_level = iteration // self.level_switch
        if new_level != self.current_level:
            self.current_level = new_level
            print(f"🎯 Level {self.current_level} activated at iteration {iteration}")
    
    def start_training_episode(self):
        """Call this when starting a training episode to enable logging."""
        self.training_mode = True
        self.training_episode_count += 1
        print(f"🚀 Starting training episode {self.training_episode_count}")
    
    def end_training_episode(self):
        """Call this when ending a training episode to disable logging."""
        self.training_mode = False
        print(f"🏁 Ending training episode {self.training_episode_count}")
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_step += 1
        self.total_env_steps += 1
        self.episode_reward += reward
        
        # Check episode termination
        episode_ended = done or truncated or self.episode_step >= self.episode_cap
        
        if episode_ended:
            # Only log episodes if we're in training mode
            if self.training_mode:
                print(f"🔍 Training episode ended: step={self.episode_step}, reward={self.episode_reward:.2f}, iteration={self.current_iteration}")
                self._log_episode()
                self.episodes_in_current_iteration += 1
            else:
                print(f"⚡ PPO internal episode ended: step={self.episode_step}, reward={self.episode_reward:.2f} (not logged)")
            
            self._reset_episode()
        
        return obs, reward, episode_ended, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_episode()
        return obs, info
    
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
            'perturbation_applied': [0.0] * 4,  # Simplified for test
            'shift_boundary': (self.current_iteration % self.level_switch == 0),
            'episode_type': 'training'  # NEW: Mark as training episode
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
            'episodes_in_iteration': self.episodes_in_current_iteration,
            'total_env_steps': self.total_env_steps,
            'total_episodes': len(self.episode_returns),
            'training_episode_count': self.training_episode_count
        }

def test_fixed_wrapper():
    """
    Test the fixed wrapper to ensure it only logs training episodes.
    """
    print("🧪 Testing Fixed Wrapper")
    print("=" * 60)
    
    # Create mock environment
    env = MockEnv(max_steps=400)
    
    # Create fixed wrapper
    wrapper = FixedContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=5,
        level_switch=3,
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=2,
        no_noise=False
    )
    
    print(f"📊 Testing fixed wrapper with 5 iterations")
    
    for iteration in range(5):
        print(f"\n🔄 Iteration {iteration}")
        wrapper.set_iteration(iteration)
        
        # Phase 1: Collect 2 training episodes
        for episode_idx in range(2):
            print(f"   📝 Training Episode {episode_idx + 1}")
            
            # Enable training mode
            wrapper.start_training_episode()
            
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
                    print(f"         ✅ Training episode ended at step {episode_steps}, reward: {episode_reward:.2f}")
                    break
            
            # Disable training mode
            wrapper.end_training_episode()
        
        # Phase 2: Simulate PPO internal steps (should not log episodes)
        print(f"   🔄 Simulating PPO internal steps...")
        
        ppo_steps = 50  # Simulate 50 PPO internal steps
        for ppo_step in range(ppo_steps):
            action = np.random.randint(0, 2)
            next_observation, reward, done, truncated, info = wrapper.step(action)
            
            if ppo_step % 10 == 0:  # Show progress every 10 steps
                print(f"      PPO step {ppo_step + 1}/{ppo_steps}")
        
        print(f"   📊 End of iteration:")
        wrapper_info = wrapper.get_current_info()
        print(f"      • Training episodes: 2")
        print(f"      • Wrapper episodes in iteration: {wrapper_info['episodes_in_iteration']}")
        print(f"      • Wrapper total episodes: {wrapper_info['total_episodes']}")
        print(f"      • Training episode count: {wrapper_info['training_episode_count']}")
    
    print(f"\n🎯 Test completed!")
    print(f"   Total training episodes: {wrapper.training_episode_count}")
    print(f"   Wrapper total episodes: {len(wrapper.episode_returns)}")
    print(f"   Total environment steps: {wrapper.total_env_steps}")
    
    # Analyze the results
    print(f"\n🔍 Analysis:")
    print(f"   Expected episodes: 5 iterations × 2 episodes = 10")
    print(f"   Actual training episodes: {wrapper.training_episode_count}")
    print(f"   Actual wrapper episodes: {len(wrapper.episode_returns)}")
    print(f"   Mismatch: {len(wrapper.episode_returns) - wrapper.training_episode_count} extra episodes")
    
    if len(wrapper.episode_returns) == wrapper.training_episode_count:
        print(f"\n✅ SUCCESS! Episode counts now match!")
        print(f"   Only training episodes are logged, PPO internal episodes are ignored.")
    else:
        print(f"\n❌ Still have a mismatch!")
    
    return wrapper

if __name__ == "__main__":
    print("🚀 Episode Counting Fix Test")
    print("=" * 60)
    
    wrapper = test_fixed_wrapper()
    
    print(f"\n🎯 Fix test complete!")
    print(f"   The fixed wrapper should now only log training episodes.")
