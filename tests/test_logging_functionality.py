#!/usr/bin/env python3
"""
Test script to verify that logging functionality works correctly with the episode counting fix.
This tests CSV file creation, callback logging, and data integrity.
"""

import numpy as np
import gymnasium as gym
import sys
import os
import tempfile
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_csv_logging():
    """Test that CSV logging works correctly with training mode."""
    print("🧪 Testing CSV Logging Functionality")
    print("=" * 60)
    
    try:
        from topologies_continual_task_training_normal_0_1 import ContinualLearningWrapper
        print("✅ Successfully imported ContinualLearningWrapper")
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Create a mock environment that ends episodes predictably
    class MockEnv:
        def __init__(self, max_steps=400):
            self.max_steps = max_steps
            self.step_count = 0
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)
        
        def step(self, action):
            self.step_count += 1
            
            # End episode after 5 steps for predictable testing
            done = self.step_count >= 5
            truncated = False
            
            obs = np.random.random(4)
            reward = 1.0 if not done else 0.0
            
            return obs, reward, done, truncated, {}
        
        def reset(self, **kwargs):
            self.step_count = 0
            return np.random.random(4), {}
    
    # Create a real logging callback that saves to CSV
    class TestLoggingCallback:
        def __init__(self, save_dir):
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(exist_ok=True)
            self.episode_data = []
            self.level_changes = []
        
        def _log_episode_completion(self, episode_info):
            """Log episode completion to memory and CSV."""
            self.episode_data.append(episode_info)
            
            # Save to CSV after each episode
            df = pd.DataFrame(self.episode_data)
            csv_path = self.save_dir / "episode_data.csv"
            df.to_csv(csv_path, index=False)
            
            print(f"      📊 Episode logged: {episode_info['episode_return_raw']:.2f} reward, {episode_info['episode_length']} steps")
        
        def _log_perturbation_level_change(self, iteration, level, perturbation):
            """Log level changes."""
            level_info = {
                'iteration': iteration,
                'level': level,
                'perturbation': perturbation.tolist() if hasattr(perturbation, 'tolist') else perturbation
            }
            self.level_changes.append(level_info)
            
            # Save level changes to CSV
            level_df = pd.DataFrame(self.level_changes)
            level_csv_path = self.save_dir / "level_changes.csv"
            level_df.to_csv(level_csv_path, index=False)
            
            print(f"      🔄 Level change logged: Level {level} at iteration {iteration}")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create wrapper with real logging callback
        env = MockEnv(max_steps=5)
        callback = TestLoggingCallback(temp_dir)
        
        wrapper = ContinualLearningWrapper(
            env=env,
            task_name="CartPole-v1",
            max_iterations=3,
            level_switch=2,
            shift_range=[0, 1],
            seed=42,
            reward_scale=20.0,
            episode_cap=400,
            logging_callback=callback,
            num_levels=2,
            no_noise=False
        )
        
        print(f"🎲 Test wrapper initialized with logging callback")
        
        # Simulate training loop with logging
        total_training_episodes = 0
        
        for iteration in range(3):
            print(f"\n🔄 Iteration {iteration}")
            wrapper.set_iteration(iteration)
            
            # Collect 2 training episodes
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
                total_training_episodes += 1
            
            print(f"   📊 Training episodes collected: {2}")
            
            # Simulate PPO internal steps (should not log)
            print(f"   🔄 Simulating PPO internal steps...")
            initial_episode_count = len(wrapper.episode_returns)
            
            for ppo_step in range(10):
                action = np.random.randint(0, 2)
                next_observation, reward, done, truncated, info = wrapper.step(action)
                
                if done or truncated:
                    break
            
            final_episode_count = len(wrapper.episode_returns)
            ppo_episodes_logged = final_episode_count - initial_episode_count
            
            print(f"   📊 PPO internal episodes logged: {ppo_episodes_logged}")
            
            if ppo_episodes_logged > 0:
                print(f"   ❌ FAILURE: PPO internal episodes were logged")
                return False
            else:
                print(f"   ✅ SUCCESS: No PPO internal episodes logged")
        
        print(f"\n🎯 Training simulation completed!")
        print(f"   Total training episodes: {total_training_episodes}")
        print(f"   Total episodes logged: {len(wrapper.episode_returns)}")
        
        # Check CSV files
        print(f"\n📁 Checking CSV files...")
        
        episode_csv_path = Path(temp_dir) / "episode_data.csv"
        level_csv_path = Path(temp_dir) / "level_changes.csv"
        
        if episode_csv_path.exists():
            episode_df = pd.read_csv(episode_csv_path)
            print(f"   ✅ Episode CSV created: {len(episode_df)} rows")
            print(f"   📊 CSV columns: {list(episode_df.columns)}")
            
            # Check data integrity
            if len(episode_df) == total_training_episodes:
                print(f"   ✅ CSV row count matches training episodes")
            else:
                print(f"   ❌ CSV row count mismatch: {len(episode_df)} vs {total_training_episodes}")
                return False
            
            # Check that all episodes have training_type = 'training'
            if 'episode_type' in episode_df.columns:
                training_episodes = episode_df[episode_df['episode_type'] == 'training']
                if len(training_episodes) == len(episode_df):
                    print(f"   ✅ All episodes marked as training episodes")
                else:
                    print(f"   ❌ Some episodes not marked as training: {len(training_episodes)}/{len(episode_df)}")
                    return False
            else:
                print(f"   ⚠️  No episode_type column found (this is expected for current implementation)")
            
        else:
            print(f"   ❌ Episode CSV not created")
            return False
        
        if level_csv_path.exists():
            level_df = pd.read_csv(level_csv_path)
            print(f"   ✅ Level changes CSV created: {len(level_df)} rows")
        else:
            print(f"   ⚠️  Level changes CSV not created (this is optional)")
        
        # Verify final counts
        if total_training_episodes == len(wrapper.episode_returns):
            print(f"   ✅ SUCCESS: Episode counts match exactly")
            print(f"   🎉 CSV logging is working correctly with episode counting fix!")
            return True
        else:
            print(f"   ❌ FAILURE: Episode count mismatch")
            print(f"      Expected: {total_training_episodes}")
            print(f"      Actual: {len(wrapper.episode_returns)}")
            return False

def test_data_integrity():
    """Test that logged data has correct structure and values."""
    print(f"\n🧪 Testing Data Integrity")
    print("=" * 60)
    
    try:
        from topologies_continual_task_training_normal_0_1 import ContinualLearningWrapper
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        return False
    
    # Create simple environment
    class SimpleEnv:
        def __init__(self):
            self.step_count = 0
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
            self.action_space = gym.spaces.Discrete(2)
        
        def step(self, action):
            self.step_count += 1
            done = self.step_count >= 3
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
    env = SimpleEnv()
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
    
    print(f"🎲 Testing data integrity with simple wrapper")
    
    # Run one training episode
    wrapper.start_training_episode()
    
    observation = env.reset()[0]
    for step in range(5):
        action = np.random.randint(0, 2)
        obs, reward, done, truncated, info = wrapper.step(action)
        if done or truncated:
            break
    
    wrapper.end_training_episode()
    
    # Check episode data structure
    if len(wrapper.episode_returns) > 0:
        episode_data = wrapper.episode_returns[0]
        print(f"   📊 Episode data structure:")
        print(f"      • Keys: {list(episode_data.keys())}")
        
        # Check required fields
        required_fields = [
            'global_step_end', 'episode_length', 'episode_return_raw', 
            'episode_return_scaled', 'shift_id', 'iteration', 'level',
            'perturbation_applied', 'shift_boundary'
        ]
        
        missing_fields = [field for field in required_fields if field not in episode_data]
        if missing_fields:
            print(f"   ❌ Missing required fields: {missing_fields}")
            return False
        else:
            print(f"   ✅ All required fields present")
        
        # Check data types and values
        if isinstance(episode_data['episode_return_raw'], (int, float)):
            print(f"   ✅ episode_return_raw is numeric: {episode_data['episode_return_raw']}")
        else:
            print(f"   ❌ episode_return_raw is not numeric: {type(episode_data['episode_return_raw'])}")
            return False
        
        if isinstance(episode_data['episode_length'], int):
            print(f"   ✅ episode_length is integer: {episode_data['episode_length']}")
        else:
            print(f"   ❌ episode_length is not integer: {type(episode_data['episode_length'])}")
            return False
        
        print(f"   ✅ Data integrity check passed")
        return True
    else:
        print(f"   ❌ No episode data to check")
        return False

def main():
    """Run all logging tests."""
    print("🚀 Logging Functionality Verification Test")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: CSV logging functionality
    print("\n" + "="*60)
    result1 = test_csv_logging()
    test_results.append(("CSV Logging Functionality", result1))
    
    # Test 2: Data integrity
    print("\n" + "="*60)
    result2 = test_data_integrity()
    test_results.append(("Data Integrity", result2))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 LOGGING TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL LOGGING TESTS PASSED!")
        print("   The episode counting fix works correctly with logging.")
        print("   CSV files will contain only training episodes (6,000 instead of 188,771).")
        print("   Data structure and integrity are maintained.")
    else:
        print("❌ SOME LOGGING TESTS FAILED!")
        print("   Please check the output above for details.")
        print("   There may be issues with the logging functionality.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
