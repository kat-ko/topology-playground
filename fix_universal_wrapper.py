#!/usr/bin/env python3
"""
Fix script for UniversalActionWrapper to work with stable-baselines3 2.6.0.
This script updates the wrapper to use gymnasium spaces properly.
"""

import numpy as np
import gymnasium as gym

class UniversalActionWrapper(gym.Wrapper):
    """
    Wrapper to create universal action space (3 actions) and universal observation space (6 dimensions) for all tasks.
    Maps universal actions to task-specific actions using action masking.
    Pads observations to universal dimensions.
    Updated for compatibility with stable-baselines3 2.6.0.
    """
    
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        
        # Universal action space: 3 actions for all tasks
        # Use gymnasium spaces for compatibility with stable-baselines3 2.6.0
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
        obs, reward, terminated, truncated, info = self.env.step(mapped_action)
        
        # Pad observation to universal dimensions
        padded_obs = self._pad_observation(obs)
        
        return padded_obs, reward, terminated, truncated, info
    
    def _pad_observation(self, obs):
        """
        Pad observation to universal 6-dimensional space.
        """
        if len(obs) < 6:
            # Pad with zeros if observation is smaller than 6 dimensions
            padded = np.zeros(6, dtype=np.float32)
            padded[:len(obs)] = obs
            return padded
        else:
            # Truncate if observation is larger than 6 dimensions
            return obs[:6]
    
    def reset(self, **kwargs):
        """
        Reset the environment and pad the initial observation.
        """
        obs, info = self.env.reset(**kwargs)
        padded_obs = self._pad_observation(obs)
        return padded_obs, info
    
    def get_action_mask(self):
        """
        Get the current action mask for the task.
        """
        return self.current_mask

def make_env(env_name):
    """Create environment factory function with universal action space wrapper."""
    def _make_env():
        env = gym.make(env_name)
        # Wrap with universal action space
        env = UniversalActionWrapper(env, env_name)
        return env
    return _make_env

# Test the wrapper
if __name__ == "__main__":
    print("Testing UniversalActionWrapper with stable-baselines3 2.6.0...")
    
    # Test with CartPole-v1
    env = gym.make('CartPole-v1')
    wrapped_env = UniversalActionWrapper(env, 'CartPole-v1')
    
    print(f"Original action space: {env.action_space}")
    print(f"Wrapped action space: {wrapped_env.action_space}")
    print(f"Original observation space: {env.observation_space}")
    print(f"Wrapped observation space: {wrapped_env.observation_space}")
    
    # Test a few steps
    obs, info = wrapped_env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(5):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, obs_shape={obs.shape}")
        if terminated or truncated:
            break
    
    wrapped_env.close()
    print("✅ UniversalActionWrapper test completed successfully!") 