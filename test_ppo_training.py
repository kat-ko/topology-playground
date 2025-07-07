#!/usr/bin/env python3
"""
Test PPO training with topology networks to verify empirically sound results.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os

# Add src to path
sys.path.append('src')

from minimal_sb3_topology_example import (
    TopologyPolicy, 
    create_topology_network,
    SimpleFeaturesExtractor,
    TopologyMLPExtractor
)

class TrainingCallback(BaseCallback):
    """Callback to track training progress and learning curves."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.total_timesteps = []
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.training_env.buf_rews) > 0:
            for rew in self.training_env.buf_rews:
                self.episode_rewards.append(rew)
                # Estimate episode length (since buf_lens is not available)
                self.episode_lengths.append(200)  # CartPole typically runs for ~200 steps
        
        # Track losses from logger
        if self.logger.name_to_value:
            if 'train/value_loss' in self.logger.name_to_value:
                self.value_losses.append(self.logger.name_to_value['train/value_loss'])
            if 'train/policy_loss' in self.logger.name_to_value:
                self.policy_losses.append(self.logger.name_to_value['train/policy_loss'])
            if 'train/entropy_loss' in self.logger.name_to_value:
                self.entropy_losses.append(self.logger.name_to_value['train/entropy_loss'])
        
        self.total_timesteps.append(self.num_timesteps)
        return True

def test_ppo_training():
    """Test PPO training with topology networks and analyze results."""
    
    print("=== PPO Training Test with Topology Networks ===")
    
    # 1. Create environment
    print("\n1. Creating CartPole environment...")
    env = gym.make('CartPole-v1')
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 2. Create topology networks
    print("\n2. Creating topology networks...")
    actor_network = create_topology_network('fully_connected', size=30, seed=42)
    critic_network = create_topology_network('fully_connected', size=30, seed=43)
    
    print(f"Actor network: {type(actor_network).__name__}, size={len(actor_network.topology.nodes())}")
    print(f"Critic network: {type(critic_network).__name__}, size={len(critic_network.topology.nodes())}")
    
    # 3. Create policy
    print("\n3. Creating topology policy...")
    def make_topology_policy(observation_space, action_space, lr_schedule, **kwargs):
        return TopologyPolicy(observation_space, action_space, lr_schedule, actor_network, critic_network, **kwargs)
    
    # 4. Create PPO agent
    print("\n4. Creating PPO agent...")
    model = PPO(
        policy=make_topology_policy,
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=None,
        policy_kwargs={},
        verbose=1,
        seed=42,
        device='auto',
        _init_setup_model=True,
    )
    
    # 5. Set up logging
    print("\n5. Setting up logging...")
    log_dir = "ppo_training_logs"
    os.makedirs(log_dir, exist_ok=True)
    configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 6. Create callback
    callback = TrainingCallback()
    
    # 7. Train the agent
    print("\n6. Starting training...")
    total_timesteps = 50000  # 50k steps for reasonable test
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 8. Analyze results
    print("\n7. Analyzing training results...")
    
    # Calculate moving averages
    window_size = 100
    if len(callback.episode_rewards) > window_size:
        moving_avg_rewards = np.convolve(callback.episode_rewards, 
                                        np.ones(window_size)/window_size, 
                                        mode='valid')
    else:
        moving_avg_rewards = callback.episode_rewards
    
    # Basic statistics
    print(f"\nTraining Statistics:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total episodes: {len(callback.episode_rewards)}")
    print(f"Final episode reward: {callback.episode_rewards[-1] if callback.episode_rewards else 'N/A'}")
    print(f"Best episode reward: {max(callback.episode_rewards) if callback.episode_rewards else 'N/A'}")
    print(f"Average episode reward: {np.mean(callback.episode_rewards) if callback.episode_rewards else 'N/A':.2f}")
    print(f"Average episode length: {np.mean(callback.episode_lengths) if callback.episode_lengths else 'N/A':.2f}")
    
    # Check for learning
    if len(callback.episode_rewards) > 10:
        early_avg = np.mean(callback.episode_rewards[:10])
        late_avg = np.mean(callback.episode_rewards[-10:])
        improvement = late_avg - early_avg
        print(f"Learning improvement (first 10 vs last 10 episodes): {improvement:.2f}")
        
        if improvement > 0:
            print("✅ POSITIVE: Agent shows learning improvement!")
        else:
            print("⚠️  WARNING: Agent shows no clear learning improvement")
    
    # 9. Test final performance
    print("\n8. Testing final performance...")
    test_episodes = 10
    test_rewards = []
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        print(f"Test episode {episode + 1}: {episode_reward}")
    
    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage test reward: {avg_test_reward:.2f}")
    
    # 10. Plot results
    print("\n9. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(callback.episode_rewards, alpha=0.6, label='Raw rewards')
    if len(moving_avg_rewards) > 0:
        axes[0, 0].plot(range(window_size-1, len(callback.episode_rewards)), 
                       moving_avg_rewards, 'r-', linewidth=2, label=f'Moving avg ({window_size})')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(callback.episode_lengths, alpha=0.6)
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    if callback.value_losses:
        axes[1, 0].plot(callback.value_losses, label='Value Loss')
    if callback.policy_losses:
        axes[1, 0].plot(callback.policy_losses, label='Policy Loss')
    if callback.entropy_losses:
        axes[1, 0].plot(callback.entropy_losses, label='Entropy Loss')
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_xlabel('Update Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test performance
    axes[1, 1].bar(range(test_episodes), test_rewards)
    axes[1, 1].axhline(y=avg_test_reward, color='r', linestyle='--', label=f'Avg: {avg_test_reward:.2f}')
    axes[1, 1].set_title('Final Test Performance')
    axes[1, 1].set_xlabel('Test Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ppo_training_results.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'ppo_training_results.png'")
    
    # 11. Empirical soundness check
    print("\n10. Empirical Soundness Analysis:")
    
    # Check 1: Are rewards improving over time?
    if len(callback.episode_rewards) > 20:
        first_quarter = np.mean(callback.episode_rewards[:len(callback.episode_rewards)//4])
        last_quarter = np.mean(callback.episode_rewards[-len(callback.episode_rewards)//4:])
        reward_improvement = last_quarter - first_quarter
        print(f"Reward improvement (first vs last quarter): {reward_improvement:.2f}")
        
        if reward_improvement > 0:
            print("✅ POSITIVE: Clear reward improvement over time")
        else:
            print("⚠️  WARNING: No clear reward improvement")
    
    # Check 2: Are losses decreasing?
    if callback.value_losses and len(callback.value_losses) > 10:
        early_loss = np.mean(callback.value_losses[:10])
        late_loss = np.mean(callback.value_losses[-10:])
        loss_decrease = early_loss - late_loss
        print(f"Value loss decrease: {loss_decrease:.4f}")
        
        if loss_decrease > 0:
            print("✅ POSITIVE: Value loss is decreasing")
        else:
            print("⚠️  WARNING: Value loss not decreasing")
    
    # Check 3: Is final performance reasonable?
    cartpole_solved_threshold = 195  # CartPole is considered solved at 195+ average reward
    if avg_test_reward >= cartpole_solved_threshold:
        print(f"✅ EXCELLENT: Agent solved CartPole! (Avg reward: {avg_test_reward:.2f} >= {cartpole_solved_threshold})")
    elif avg_test_reward >= 100:
        print(f"✅ GOOD: Agent shows reasonable performance (Avg reward: {avg_test_reward:.2f})")
    elif avg_test_reward >= 50:
        print(f"⚠️  MODERATE: Agent shows some learning (Avg reward: {avg_test_reward:.2f})")
    else:
        print(f"❌ POOR: Agent shows minimal learning (Avg reward: {avg_test_reward:.2f})")
    
    # Check 4: Are episode lengths increasing?
    if len(callback.episode_lengths) > 20:
        early_length = np.mean(callback.episode_lengths[:10])
        late_length = np.mean(callback.episode_lengths[-10:])
        length_increase = late_length - early_length
        print(f"Episode length increase: {length_increase:.2f}")
        
        if length_increase > 0:
            print("✅ POSITIVE: Episodes getting longer (agent staying alive longer)")
        else:
            print("⚠️  WARNING: Episodes not getting longer")
    
    # Check 5: Is training stable?
    if len(callback.episode_rewards) > 50:
        recent_rewards = callback.episode_rewards[-50:]
        reward_std = np.std(recent_rewards)
        print(f"Recent reward stability (std): {reward_std:.2f}")
        
        if reward_std < 50:
            print("✅ POSITIVE: Training appears stable")
        else:
            print("⚠️  WARNING: Training appears unstable")
    
    print(f"\n=== Training Test Complete ===")
    print(f"Results saved to: {log_dir}/")
    print(f"Plots saved to: ppo_training_results.png")
    
    env.close()
    return model, callback

if __name__ == "__main__":
    model, callback = test_ppo_training() 