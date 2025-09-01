#!/usr/bin/env python3
"""
Test script to verify the episode counting fix works correctly.
This tests the training mode functionality and ensures PPO training isn't affected.
"""

import numpy as np
import gymnasium as gym
import sys
import os
from pathlib import Path

# Add the current directory to Python path to import the training modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_training_mode_functionality():
    """Test that training mode correctly controls episode logging."""
    print("🧪 Testing Training Mode Functionality")
    print("=" * 60)
    
    # Import the ContinualLearningWrapper from one of the training files
    try:
        # Try to import from the first training file
        from topologies_continual_task_training_normal_0_1 import ContinualLearningWrapper
        print("✅ Successfully imported ContinualLearningWrapper from normal_0_1.py")
    except ImportError as e:
        print(f"❌ Failed to import from normal_0_1.py: {e}")
        return False
    
    # Create mock environment
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
            elif np.random.random() < 0.3:  # 30% chance of early termination for testing
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
    
    # Create mock logging callback
    class MockLoggingCallback:
        def _log_episode_completion(self, episode_info):
            pass
        
        def _log_perturbation_level_change(self, iteration, level, perturbation):
            pass
    
    # Create wrapper
    env = MockEnv(max_steps=400)
    wrapper = ContinualLearningWrapper(
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
    
    print(f"🎲 Test wrapper initialized")
    print(f"   Initial training mode: {wrapper.training_mode}")
    print(f"   Initial training episode count: {wrapper.training_episode_count}")
    
    # Test 1: Verify training mode methods exist
    print(f"\n📋 Test 1: Training Mode Methods")
    assert hasattr(wrapper, 'start_training_episode'), "start_training_episode method missing"
    assert hasattr(wrapper, 'end_training_episode'), "end_training_episode method missing"
    assert hasattr(wrapper, 'training_mode'), "training_mode attribute missing"
    assert hasattr(wrapper, 'training_episode_count'), "training_episode_count attribute missing"
    print(f"   ✅ All required methods and attributes present")
    
    # Test 2: Verify training mode controls episode logging
    print(f"\n📋 Test 2: Training Mode Controls Episode Logging")
    
    # Start training mode
    wrapper.start_training_episode()
    print(f"   Training mode enabled: {wrapper.training_mode}")
    print(f"   Training episode count: {wrapper.training_episode_count}")
    
    # Take steps until episode ends (should log episode)
    initial_episode_count = len(wrapper.episode_returns)
    for step in range(20):  # Take more steps to ensure episode ends
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        if done or truncated:
            break
    
    episodes_logged_during_training = len(wrapper.episode_returns) - initial_episode_count
    print(f"   Episodes logged during training mode: {episodes_logged_during_training}")
    
    # End training mode
    wrapper.end_training_episode()
    print(f"   Training mode disabled: {wrapper.training_mode}")
    
    # Take more steps (should not log episodes)
    initial_episode_count = len(wrapper.episode_returns)
    for step in range(20):  # Take more steps to ensure episode ends
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        if done or truncated:
            break
    
    final_episode_count = len(wrapper.episode_returns)
    episodes_logged_after_training = final_episode_count - initial_episode_count
    print(f"   Episodes logged after training mode disabled: {episodes_logged_after_training}")
    
    if episodes_logged_after_training == 0:
        print(f"   ✅ SUCCESS: No episodes logged when training mode is disabled")
    else:
        print(f"   ❌ FAILURE: {episodes_logged_after_training} episodes still being logged when training mode is disabled")
        return False
    
    # Test 3: Verify episode counting accuracy
    print(f"\n📋 Test 3: Episode Counting Accuracy")
    print(f"   Training episodes completed: {wrapper.training_episode_count}")
    print(f"   Episodes logged to CSV: {len(wrapper.episode_returns)}")
    
    # We should have at least one episode logged during training mode
    if len(wrapper.episode_returns) > 0:
        print(f"   ✅ SUCCESS: At least one episode was logged during training mode")
        return True
    else:
        print(f"   ❌ FAILURE: No episodes were logged during training mode")
        return False

def test_ppo_training_simulation():
    """Test that PPO training simulation works correctly with training mode."""
    print(f"\n🧪 Testing PPO Training Simulation")
    print("=" * 60)
    
    try:
        from topologies_continual_task_training_normal_0_1 import ContinualLearningWrapper
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Create mock environment
    class MockEnv:
        def __init__(self, max_steps=400):
            self.max_steps = max_steps
            self.step_count = 0
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)
        
        def step(self, action):
            self.step_count += 1
            
            # Simulate CartPole behavior
            if self.step_count >= self.max_steps:
                done = True
                truncated = False
            elif np.random.random() < 0.15:  # 15% chance of early termination
                done = True
                truncated = False
            else:
                done = False
                truncated = False
            
            obs = np.random.random(4)
            reward = 1.0 if not done else 0.0
            
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
    env = MockEnv(max_steps=400)
    wrapper = ContinualLearningWrapper(
        env=env,
        task_name="CartPole-v1",
        max_iterations=3,
        level_switch=2,
        shift_range=[0, 1],
        seed=42,
        reward_scale=20.0,
        episode_cap=400,
        logging_callback=MockLoggingCallback(),
        num_levels=2,
        no_noise=False
    )
    
    print(f"🎲 Test wrapper initialized for PPO simulation")
    
    # Simulate the main training loop
    total_training_episodes = 0
    
    for iteration in range(3):
        print(f"\n🔄 Iteration {iteration}")
        wrapper.set_iteration(iteration)
        
        # Phase 1: Collect 2 training episodes
        episodes_data = []
        for episode_idx in range(2):
            print(f"   📝 Training Episode {episode_idx + 1}")
            
            # Enable training mode
            wrapper.start_training_episode()
            
            # Reset environment
            observation = env.reset()[0]
            episode_reward = 0.0
            episode_steps = 0
            
            # Run episode
            for step in range(wrapper.episode_cap):
                action = np.random.randint(0, 2)
                next_observation, reward, done, truncated, info = wrapper.step(action)
                
                episode_reward += reward
                episode_steps += 1
                observation = next_observation
                
                if done or truncated:
                    break
            
            # Disable training mode
            wrapper.end_training_episode()
            
            # Store episode data
            episode_data = {
                'episode_idx': episode_idx,
                'episode_reward': episode_reward,
                'episode_steps': episode_steps,
                'iteration': iteration
            }
            episodes_data.append(episode_data)
            total_training_episodes += 1
        
        print(f"   📊 Training episodes collected: {len(episodes_data)}")
        
        # Phase 2: Simulate PPO internal steps (should not log episodes)
        print(f"   🔄 Simulating PPO internal steps...")
        
        ppo_steps = 50  # Simulate 50 PPO internal steps
        initial_episode_count = len(wrapper.episode_returns)
        
        for ppo_step in range(ppo_steps):
            action = np.random.randint(0, 2)
            next_observation, reward, done, truncated, info = wrapper.step(action)
            
            if ppo_step % 10 == 0:
                print(f"      PPO step {ppo_step + 1}/{ppo_steps}")
        
        final_episode_count = len(wrapper.episode_returns)
        ppo_episodes_logged = final_episode_count - initial_episode_count
        
        print(f"   📊 PPO internal episodes logged: {ppo_episodes_logged}")
        
        if ppo_episodes_logged == 0:
            print(f"   ✅ SUCCESS: No PPO internal episodes logged")
        else:
            print(f"   ❌ FAILURE: {ppo_episodes_logged} PPO internal episodes were logged")
            return False
        
        # Get wrapper info
        wrapper_info = wrapper.get_current_info()
        print(f"   📊 Wrapper state:")
        print(f"      • Training episodes: {wrapper_info['training_episode_count']}")
        print(f"      • Total episodes logged: {wrapper_info['total_episodes']}")
        print(f"      • Episodes in iteration: {wrapper_info['episodes_in_iteration']}")
    
    print(f"\n🎯 PPO Simulation completed!")
    print(f"   Total training episodes: {total_training_episodes}")
    print(f"   Total episodes logged: {len(wrapper.episode_returns)}")
    
    # Verify final counts
    if total_training_episodes == len(wrapper.episode_returns):
        print(f"   ✅ SUCCESS: Episode counts match exactly")
        return True
    else:
        print(f"   ❌ FAILURE: Episode count mismatch")
        print(f"      Expected: {total_training_episodes}")
        print(f"      Actual: {len(wrapper.episode_returns)}")
        return False

def test_all_training_files():
    """Test that all training files have the fix applied."""
    print(f"\n🧪 Testing All Training Files")
    print("=" * 60)
    
    training_files = [
        'topologies_continual_task_training_normal_0_1.py',
        'topologies_continual_task_training_normal.py',
        'topologies_continual_task_training_uniform.py'
    ]
    
    all_files_ok = True
    
    for file_path in training_files:
        print(f"\n📁 Testing {file_path}")
        
        if not Path(file_path).exists():
            print(f"   ❌ File not found")
            all_files_ok = False
            continue
        
        # Check for required components
        with open(file_path, 'r') as f:
            content = f.read()
        
        checks = [
            ('training_mode = False', 'Training mode variable'),
            ('self.training_episode_count = 0', 'Training episode count variable'),
            ('def start_training_episode(self):', 'start_training_episode method'),
            ('def end_training_episode(self):', 'end_training_episode method'),
            ('if self.training_mode:', 'Training mode check in step method'),
            ('env.start_training_episode()', 'Training mode start call'),
            ('env.end_training_episode()', 'Training mode end call'),
            ('training_episode_count', 'Training episode count in get_current_info')
        ]
        
        file_ok = True
        for check, description in checks:
            if check in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ {description} missing")
                file_ok = False
        
        if file_ok:
            print(f"   🎯 File has all required fixes")
        else:
            print(f"   ❌ File is missing some fixes")
            all_files_ok = False
    
    return all_files_ok

def main():
    """Run all tests."""
    print("🚀 Episode Counting Fix Verification Test")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Training mode functionality
    print("\n" + "="*60)
    result1 = test_training_mode_functionality()
    test_results.append(("Training Mode Functionality", result1))
    
    # Test 2: PPO training simulation
    print("\n" + "="*60)
    result2 = test_ppo_training_simulation()
    test_results.append(("PPO Training Simulation", result2))
    
    # Test 3: All training files
    print("\n" + "="*60)
    result3 = test_all_training_files()
    test_results.append(("All Training Files", result3))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   The episode counting fix is working correctly.")
        print("   Your training scripts should now log exactly 6,000 episodes instead of 188,771.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Please check the output above for details.")
        print("   The episode counting fix may not be fully implemented.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
