import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from stable_baselines3 import PPO, A2C, SAC
import gymnasium as gym

@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    buffer_size: int = 10000
    batch_size: int = 64
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_update_freq: int = 1000
    tau: float = 0.005

class PPOAgent:
    """PPO agent using Stable Baselines3."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = PPO(
            "MlpPolicy",
            env=None,  # Will be set in run_curriculum
            learning_rate=config['rl_params']['ppo']['learning_rate'],
            n_steps=config['rl_params']['ppo']['n_steps'],
            batch_size=config['rl_params']['ppo']['batch_size'],
            n_epochs=config['rl_params']['ppo']['n_epochs'],
            gamma=config['rl_params']['ppo']['gamma'],
            gae_lambda=config['rl_params']['ppo']['gae_lambda'],
            clip_range=config['rl_params']['ppo']['clip_ratio'],
            ent_coef=config['rl_params']['ppo']['entropy_coef'],
            verbose=0
        )
    
    def select_action(self, state):
        """Select action using the trained model."""
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update is handled by SB3's learn method."""
        pass

class A2CAgent:
    """A2C agent using Stable Baselines3."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = A2C(
            "MlpPolicy",
            env=None,  # Will be set in run_curriculum
            learning_rate=config['rl_params']['a2c']['learning_rate'],
            n_steps=5,
            gamma=config['rl_params']['a2c']['gamma'],
            verbose=0
        )
    
    def select_action(self, state):
        """Select action using the trained model."""
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update is handled by SB3's learn method."""
        pass

class SACAgent:
    """SAC agent using Stable Baselines3."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = SAC(
            "MlpPolicy",
            env=None,  # Will be set in run_curriculum
            learning_rate=config['rl_params']['sac']['learning_rate'],
            buffer_size=config['rl_params']['sac']['buffer_size'],
            batch_size=config['rl_params']['sac']['batch_size'],
            tau=config['rl_params']['sac']['tau'],
            gamma=config['rl_params']['sac']['gamma'],
            verbose=0
        )
    
    def select_action(self, state):
        """Select action using the trained model."""
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update is handled by SB3's learn method."""
        pass 