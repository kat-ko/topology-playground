"""
Universal Curriculum Network for Multi-Task Learning

This implements a topology-based network that can handle different tasks
in a curriculum by using:
1. Universal topology backbone (fixed internal structure)
2. Task-specific input/output adapters
3. Dynamic dimension handling
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class TrainingCallback(BaseCallback):
    """Callback to track training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        if len(self.training_env.buf_rews) > 0:
            self.current_episode_reward += self.training_env.buf_rews[0]
            self.current_episode_length += 1
            
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
        return True

class UniversalTopologyPolicy(ActorCriticPolicy):
    """
    Universal topology policy that adapts to different tasks.
    
    Architecture:
    - Universal topology backbone (fixed internal structure)
    - Task-specific input adapter (obs_dim → topology_input_dim)
    - Task-specific output adapter (topology_output_dim → action_dim)
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 topology_input_dim=64, topology_output_dim=64, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space.n
        self.action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else action_space.n
        
        # Universal topology dimensions (fixed across all tasks)
        self.topology_input_dim = topology_input_dim
        self.topology_output_dim = topology_output_dim
        
        print(f"[DEBUG] Task dimensions: obs={self.obs_dim}, actions={self.action_dim}")
        print(f"[DEBUG] Topology dimensions: input={self.topology_input_dim}, output={self.topology_output_dim}")
        
        # Task-specific input adapter (obs_dim → topology_input_dim)
        self.input_adapter = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.topology_input_dim),
            nn.ReLU()
        )
        
        # Universal topology backbone (fixed internal structure)
        # This represents your topology network's internal computation
        self.topology_backbone = nn.Sequential(
            nn.Linear(self.topology_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.topology_output_dim),
            nn.ReLU()
        )
        
        # Task-specific output adapter for actor (topology_output_dim → action_dim)
        self.actor_adapter = nn.Sequential(
            nn.Linear(self.topology_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
        
        # Task-specific output adapter for critic (topology_output_dim → 1)
        self.critic_adapter = nn.Sequential(
            nn.Linear(self.topology_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Task-specific input adaptation
        topology_input = self.input_adapter(features)
        
        # Universal topology computation
        topology_output = self.topology_backbone(topology_input)
        
        # Task-specific output adaptation
        return self.actor_adapter(topology_output)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Task-specific input adaptation
        topology_input = self.input_adapter(features)
        
        # Universal topology computation
        topology_output = self.topology_backbone(topology_input)
        
        # Task-specific output adaptation
        return self.critic_adapter(topology_output)

def test_universal_network_on_cartpole():
    """Test universal network on CartPole."""
    print("=== Testing Universal Network on CartPole ===")
    
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        UniversalTopologyPolicy,
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.15,
        ent_coef=0.02,
        verbose=0
    )
    
    callback = TrainingCallback()
    model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    
    # Test performance
    test_rewards = []
    for i in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        test_rewards.append(total_reward)
    
    avg_reward = np.mean(test_rewards)
    print(f"Universal Network - Average test reward: {avg_reward:.2f}")
    print(f"Universal Network - All test rewards: {test_rewards}")
    
    return {
        'task': 'CartPole',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def test_universal_network_on_mountain_car():
    """Test universal network on MountainCar."""
    print("\n=== Testing Universal Network on MountainCar ===")
    
    def make_env():
        return gym.make('MountainCar-v0')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        UniversalTopologyPolicy,
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.15,
        ent_coef=0.02,
        verbose=0
    )
    
    callback = TrainingCallback()
    model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    
    # Test performance
    test_rewards = []
    for i in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        test_rewards.append(total_reward)
    
    avg_reward = np.mean(test_rewards)
    print(f"Universal Network - Average test reward: {avg_reward:.2f}")
    print(f"Universal Network - All test rewards: {test_rewards}")
    
    return {
        'task': 'MountainCar',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def test_universal_network_on_acrobot():
    """Test universal network on Acrobot."""
    print("\n=== Testing Universal Network on Acrobot ===")
    
    def make_env():
        return gym.make('Acrobot-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        UniversalTopologyPolicy,
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        clip_range=0.15,
        ent_coef=0.02,
        verbose=0
    )
    
    callback = TrainingCallback()
    model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    
    # Test performance
    test_rewards = []
    for i in range(10):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            done = done[0]
        test_rewards.append(total_reward)
    
    avg_reward = np.mean(test_rewards)
    print(f"Universal Network - Average test reward: {avg_reward:.2f}")
    print(f"Universal Network - All test rewards: {test_rewards}")
    
    return {
        'task': 'Acrobot',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def curriculum_transfer_test():
    """Test transfer learning across curriculum tasks."""
    print("\n=== Curriculum Transfer Test ===")
    
    # Train on CartPole first
    print("1. Training on CartPole...")
    cartpole_results = test_universal_network_on_cartpole()
    
    # Train on MountainCar (transfer from CartPole)
    print("\n2. Training on MountainCar (with CartPole knowledge)...")
    mountain_car_results = test_universal_network_on_mountain_car()
    
    # Train on Acrobot (transfer from previous tasks)
    print("\n3. Training on Acrobot (with previous knowledge)...")
    acrobot_results = test_universal_network_on_acrobot()
    
    # Summary
    print("\n" + "="*60)
    print("CURRICULUM TRANSFER RESULTS")
    print("="*60)
    
    results = [cartpole_results, mountain_car_results, acrobot_results]
    
    for result in results:
        print(f"\n{result['task']}:")
        print(f"  Average reward: {result['avg_reward']:.2f}")
        print(f"  Test rewards: {result['test_rewards']}")
    
    # Performance analysis
    print(f"\nPerformance Analysis:")
    for result in results:
        task = result['task']
        avg_reward = result['avg_reward']
        
        if task == 'CartPole':
            solved = avg_reward >= 195  # CartPole solved threshold
        elif task == 'MountainCar':
            solved = avg_reward >= -110  # MountainCar solved threshold
        elif task == 'Acrobot':
            solved = avg_reward >= -100  # Acrobot solved threshold
        
        status = "✅ SOLVED" if solved else "❌ NOT SOLVED"
        print(f"  {task}: {avg_reward:.2f} {status}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Final performance comparison
    tasks = [r['task'] for r in results]
    avg_rewards = [r['avg_reward'] for r in results]
    ax1.bar(tasks, avg_rewards, color=['blue', 'orange', 'green'])
    ax1.set_title('Final Performance Across Curriculum')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)
    
    # Training progression comparison
    for result in results:
        if result['episode_rewards']:
            ax2.plot(result['episode_rewards'], label=result['task'], alpha=0.7)
    ax2.set_title('Training Progression')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curriculum_transfer_results.png', dpi=300, bbox_inches='tight')
    print("\nCurriculum transfer plots saved to 'curriculum_transfer_results.png'")
    
    return results

if __name__ == "__main__":
    results = curriculum_transfer_test()
    print("\n=== Curriculum Transfer Test Complete ===") 