#!/usr/bin/env python3
"""
Simple test to debug episode logging during training mode.
"""

import numpy as np
import gymnasium as gym
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_simple_episode_logging():
    """Test simple episode logging with training mode."""
    print("🧪 Simple Episode Logging Test")
    print("=" * 50)
    
    try:
        from topologies_continual_task_training_normal_0_1 import ContinualLearningWrapper
        print("✅ Successfully imported ContinualLearningWrapper")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Create a simple environment that always ends episodes quickly
    class SimpleEnv:
        def __init__(self):
            self.step_count = 0
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)
        
        def step(self, action):
            self.step_count += 1
            
            # Always end episode after 3 steps
            done = self.step_count >= 3
            truncated = False
            
            obs = np.random.random(4)
            reward = 1.0
            
            return obs, reward, done, truncated, {}
        
        def reset(self, **kwargs):
            self.step_count = 0
            return np.random.random(4), {}
    
    class MockLoggingCallback:
        def _log_episode_completion(self, episode_info):
            pass
        
        def _log_perturbation_level_change(self, iteration, level, perturbation):
            pass
    
    # Create wrapper
    env = SimpleEnv()
    wrapper = ContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=1,
        level_switch=1,
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=1,
        no_noise=False
    )
    
    print(f"🎲 Simple wrapper initialized")
    print(f"   Initial training mode: {wrapper.training_mode}")
    print(f"   Initial episode count: {len(wrapper.episode_returns)}")
    
    # Test 1: No training mode - should not log episodes
    print(f"\n📋 Test 1: No Training Mode")
    print(f"   Taking 10 steps without training mode...")
    
    for step in range(10):
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        
        if step % 3 == 0:
            print(f"      Step {step}: episode_ended={done or truncated}, episodes_logged={len(wrapper.episode_returns)}")
        
        if done or truncated:
            break
    
    print(f"   Episodes logged without training mode: {len(wrapper.episode_returns)}")
    
    # Test 2: With training mode - should log episodes
    print(f"\n📋 Test 2: With Training Mode")
    wrapper.start_training_episode()
    print(f"   Training mode enabled: {wrapper.training_mode}")
    
    print(f"   Taking 10 steps with training mode...")
    
    for step in range(10):
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        
        if step % 3 == 0:
            print(f"      Step {step}: episode_ended={done or truncated}, episodes_logged={len(wrapper.episode_returns)}")
        
        if done or truncated:
            break
    
    print(f"   Episodes logged with training mode: {len(wrapper.episode_returns)}")
    
    # Test 3: Disable training mode - should not log more episodes
    print(f"\n📋 Test 3: Disable Training Mode")
    wrapper.end_training_episode()
    print(f"   Training mode disabled: {wrapper.training_mode}")
    
    initial_count = len(wrapper.episode_returns)
    print(f"   Taking 10 more steps...")
    
    for step in range(10):
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        
        if step % 3 == 0:
            print(f"      Step {step}: episode_ended={done or truncated}, episodes_logged={len(wrapper.episode_returns)}")
        
        if done or truncated:
            break
    
    final_count = len(wrapper.episode_returns)
    print(f"   Additional episodes logged: {final_count - initial_count}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total episodes logged: {len(wrapper.episode_returns)}")
    print(f"   Training episodes completed: {wrapper.training_episode_count}")
    
    if len(wrapper.episode_returns) > 0:
        print(f"   ✅ SUCCESS: Episodes are being logged")
        return True
    else:
        print(f"   ❌ FAILURE: No episodes were logged")
        return False

if __name__ == "__main__":
    success = test_simple_episode_logging()
    sys.exit(0 if success else 1)
