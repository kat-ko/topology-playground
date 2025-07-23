#!/usr/bin/env python3
"""
Debug script to test UniversalActionWrapper and identify observation space issues.
"""

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

# Copy the UniversalActionWrapper from the main script
class UniversalActionWrapper(gym.Wrapper):
    """
    Wrapper to create universal action space (3 actions) and universal observation space (6 dimensions) for all tasks.
    Maps universal actions to task-specific actions using action masking.
    Pads observations to universal dimensions.
    """
    
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        
        # Universal action space: 3 actions for all tasks
        self.action_space = gym.spaces.Discrete(3)
        
        # Universal observation space: 6 dimensions for all tasks
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,),  # Universal 6-dimensional observation space
            dtype=np.float32
        )
        
        # Task-specific action masks and mappings
        self.action_masks = {
            'CartPole-v1': [True, True, False],    # Actions 0,1 valid, 2 invalid
            'MountainCar-v0': [True, True, True],  # All 3 actions valid
            'Acrobot-v1': [True, True, False]      # Actions 0,1 valid, 2 invalid
        }
        
        # Action mappings for invalid actions (fallback to valid action)
        self.action_mappings = {
            'CartPole-v1': {2: 0},      # Map action 2 to action 0
            'MountainCar-v0': {},       # No mapping needed (all valid)
            'Acrobot-v1': {2: 0}        # Map action 2 to action 0
        }
        
        self.current_mask = self.action_masks.get(task_name, [True, True, True])
        self.current_mapping = self.action_mappings.get(task_name, {})
    
    def step(self, action):
        """
        Map universal action to task-specific action and step the environment.
        Pad observations to universal dimensions.
        """
        # Map universal action to task-specific action
        if action in self.current_mapping:
            mapped_action = self.current_mapping[action]
        else:
            mapped_action = action
        
        # Step the environment with mapped action
        obs, reward, done, truncated, info = self.env.step(mapped_action)
        
        # Pad observation to universal dimensions (6)
        obs = self._pad_observation(obs)
        
        # Add action masking info to info dict
        info['universal_action'] = action
        info['mapped_action'] = mapped_action
        info['action_mask'] = self.current_mask
        
        return obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        """Pad observation to universal 6-dimensional space."""
        obs = np.array(obs, dtype=np.float32)
        
        if len(obs.shape) == 1:
            # Single observation
            if obs.shape[0] < 6:
                # Pad with zeros
                padded_obs = np.zeros(6, dtype=np.float32)
                padded_obs[:obs.shape[0]] = obs
                return padded_obs
            elif obs.shape[0] > 6:
                # Truncate
                return obs[:6]
            else:
                return obs
        else:
            # Vectorized observation
            batch_size = obs.shape[0]
            if obs.shape[1] < 6:
                # Pad with zeros
                padded_obs = np.zeros((batch_size, 6), dtype=np.float32)
                padded_obs[:, :obs.shape[1]] = obs
                return padded_obs
            elif obs.shape[1] > 6:
                # Truncate
                return obs[:, :6]
            else:
                return obs
    
    def reset(self, **kwargs):
        """Reset the environment and pad observation."""
        result = self.env.reset(**kwargs)
        
        # Handle different reset return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Pad the observation
        padded_obs = self._pad_observation(obs)
        
        # Return in the same format as received
        if isinstance(result, tuple):
            return padded_obs, info
        else:
            return padded_obs
    
    def get_action_mask(self):
        """Get the action mask for the current task."""
        return self.current_mask

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def test_single_env():
    """Test single environment wrapper."""
    print("Testing single environment wrapper...")
    
    for task in ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']:
        print(f"\n--- Testing {task} ---")
        try:
            env = make_env(task)()
            print(f"  Original obs space: {env.env.observation_space}")
            print(f"  Wrapped obs space: {env.observation_space}")
            print(f"  Original action space: {env.env.action_space}")
            print(f"  Wrapped action space: {env.action_space}")
            
            # Test reset
            obs = env.reset()
            print(f"  Reset obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"  Reset obs: {obs}")
            
            # Test step
            action = 0
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"  Step obs: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Info: {info}")
            
            env.close()
            print(f"  ✅ {task} single env test passed")
            
        except Exception as e:
            print(f"  ❌ {task} single env test failed: {e}")
            import traceback
            traceback.print_exc()

def test_vectorized_env():
    """Test vectorized environment wrapper."""
    print("\nTesting vectorized environment wrapper...")
    
    for task in ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']:
        print(f"\n--- Testing {task} vectorized ---")
        try:
            vec_env = DummyVecEnv([make_env(task)])
            print(f"  Vectorized obs space: {vec_env.observation_space}")
            print(f"  Vectorized action space: {vec_env.action_space}")
            
            # Test reset
            obs = vec_env.reset()
            print(f"  Reset obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"  Reset obs: {obs}")
            
            # Test step
            action = np.array([0])
            obs, reward, done, truncated, info = vec_env.step(action)
            print(f"  Step obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"  Step obs: {obs}")
            print(f"  Reward: {reward}")
            print(f"  Info: {info}")
            
            vec_env.close()
            print(f"  ✅ {task} vectorized env test passed")
            
        except Exception as e:
            print(f"  ❌ {task} vectorized env test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("🔍 UniversalActionWrapper Debug Test")
    print("=" * 50)
    
    test_single_env()
    test_vectorized_env()
    
    print("\n✅ Debug test completed!") 