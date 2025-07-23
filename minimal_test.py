#!/usr/bin/env python3
"""
Minimal test to reproduce the Acrobot-v1 evaluation error.
"""

import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

# Copy the UniversalActionWrapper from the main script
class UniversalActionWrapper(gym.Wrapper):
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_masks = {
            'CartPole-v1': [True, True, False],
            'MountainCar-v0': [True, True, True],
            'Acrobot-v1': [True, True, False]
        }
        self.action_mappings = {
            'CartPole-v1': {2: 0},
            'MountainCar-v0': {},
            'Acrobot-v1': {2: 0}
        }
        self.current_mask = self.action_masks.get(task_name, [True, True, True])
        self.current_mapping = self.action_mappings.get(task_name, {})
    
    def step(self, action):
        if action in self.current_mapping:
            mapped_action = self.current_mapping[action]
        else:
            mapped_action = action
        
        obs, reward, done, truncated, info = self.env.step(mapped_action)
        obs = self._pad_observation(obs)
        info['universal_action'] = action
        info['mapped_action'] = mapped_action
        info['action_mask'] = self.current_mask
        return obs, reward, done, truncated, info
    
    def _pad_observation(self, obs):
        obs = np.array(obs, dtype=np.float32)
        if len(obs.shape) == 1:
            if obs.shape[0] < 6:
                padded_obs = np.zeros(6, dtype=np.float32)
                padded_obs[:obs.shape[0]] = obs
                return padded_obs
            elif obs.shape[0] > 6:
                return obs[:6]
            else:
                return obs
        else:
            batch_size = obs.shape[0]
            if obs.shape[1] < 6:
                padded_obs = np.zeros((batch_size, 6), dtype=np.float32)
                padded_obs[:, :obs.shape[1]] = obs
                return padded_obs
            elif obs.shape[1] > 6:
                return obs[:, :6]
            else:
                return obs
    
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        padded_obs = self._pad_observation(obs)
        if isinstance(result, tuple):
            return padded_obs, info
        else:
            return padded_obs

def make_env(env_name):
    def _make_env():
        env = gym.make(env_name)
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

def test_acrobot_evaluation():
    """Test Acrobot-v1 evaluation specifically."""
    print("Testing Acrobot-v1 evaluation...")
    
    try:
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env('Acrobot-v1')])
        print(f"  Vectorized obs space: {vec_env.observation_space}")
        print(f"  Vectorized action space: {vec_env.action_space}")
        
        # Test reset
        obs = vec_env.reset()
        print(f"  Reset obs shape: {obs.shape}")
        print(f"  Reset obs type: {type(obs)}")
        print(f"  Reset obs: {obs}")
        
        # Test multiple steps to see if the error occurs
        for step in range(10):
            action = np.array([0])
            obs, reward, done, truncated, info = vec_env.step(action)
            print(f"  Step {step}: obs shape={obs.shape}, reward={reward}, done={done}")
            
            if done[0]:
                print(f"  Episode ended at step {step}")
                break
        
        vec_env.close()
        print("  ✅ Acrobot-v1 evaluation test passed")
        
    except Exception as e:
        print(f"  ❌ Acrobot-v1 evaluation test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_acrobot_evaluation() 