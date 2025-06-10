import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from gymnasium import spaces

@dataclass
class RLTaskConfig:
    """Configuration for RL tasks."""
    env_name: str
    max_episode_steps: int
    reward_threshold: float
    state_dim: int
    action_dim: int
    action_space_type: str  # 'discrete' or 'continuous'

class RLTaskGenerator:
    """Generator for reinforcement learning tasks."""
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.task_configs = {
            'cartpole': RLTaskConfig(
                env_name='CartPole-v1',
                max_episode_steps=200,
                reward_threshold=475.0,
                state_dim=4,
                action_dim=2,
                action_space_type='discrete'
            ),
            'mountain_car': RLTaskConfig(
                env_name='MountainCar-v0',
                max_episode_steps=200,
                reward_threshold=-110.0,
                state_dim=2,
                action_dim=3,
                action_space_type='discrete'
            ),
            'acrobot': RLTaskConfig(
                env_name='Acrobot-v1',
                max_episode_steps=200,
                reward_threshold=-100.0,
                state_dim=6,
                action_dim=3,
                action_space_type='discrete'
            )
        }
    
    def generate_cartpole_task(self) -> Tuple[gym.Env, RLTaskConfig]:
        """Generate CartPole-v1 environment and config."""
        env = gym.make('CartPole-v1')
        return env, self.task_configs['cartpole']
    
    def generate_mountain_car_task(self) -> Tuple[gym.Env, RLTaskConfig]:
        """Generate MountainCar-v0 environment and config."""
        env = gym.make('MountainCar-v0')
        return env, self.task_configs['mountain_car']
    
    def generate_acrobot_task(self) -> Tuple[gym.Env, RLTaskConfig]:
        """Generate Acrobot-v1 environment and config."""
        env = gym.make('Acrobot-v1')
        return env, self.task_configs['acrobot']

class RLTaskEvaluator:
    """Evaluator for reinforcement learning tasks."""
    
    @staticmethod
    def evaluate_episode(env: gym.Env, agent, config: RLTaskConfig) -> Dict[str, float]:
        """Evaluate a single episode."""
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_length < config.max_episode_steps:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'solved': episode_reward >= config.reward_threshold
        }
    
    @staticmethod
    def evaluate_episodes(env: gym.Env, agent, config: RLTaskConfig, 
                         num_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate multiple episodes."""
        episode_rewards = []
        episode_lengths = []
        solved_count = 0
        
        for _ in range(num_episodes):
            metrics = RLTaskEvaluator.evaluate_episode(env, agent, config)
            episode_rewards.append(metrics['episode_reward'])
            episode_lengths.append(metrics['episode_length'])
            if metrics['solved']:
                solved_count += 1
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'solved_rate': solved_count / num_episodes
        }

def get_task_config(task_name: str) -> Dict[str, Any]:
    """Get configuration for a specific task."""
    configs = {
        'cartpole': {
            'env_name': 'CartPole-v1',
            'action_space_type': 'discrete',
            'state_dim': 6,  # Max dimension
            'action_dim': 3,  # Max dimension
            'actual_state_dim': 4,  # CartPole actual dimension
            'actual_action_dim': 2,  # CartPole actual dimension
            'max_steps': 500,
            'reward_threshold': 475
        },
        'mountain_car': {
            'env_name': 'MountainCar-v0',
            'action_space_type': 'discrete',
            'state_dim': 6,  # Max dimension
            'action_dim': 3,  # Max dimension
            'actual_state_dim': 2,  # MountainCar actual dimension
            'actual_action_dim': 3,  # MountainCar actual dimension
            'max_steps': 200,
            'reward_threshold': -110
        },
        'acrobot': {
            'env_name': 'Acrobot-v1',
            'action_space_type': 'discrete',
            'state_dim': 6,  # Max dimension
            'action_dim': 3,  # Max dimension
            'actual_state_dim': 6,  # Acrobot actual dimension
            'actual_action_dim': 3,  # Acrobot actual dimension
            'max_steps': 500,
            'reward_threshold': -100
        }
    }
    return configs.get(task_name, {}) 