#!/usr/bin/env python3
"""
Quick test to verify network improvements work without running full training.
"""

import sys
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add src to path
sys.path.append('src')

from minimal_sb3_topology_example import TopologyPolicy, create_topology_network

def quick_test_improvements():
    """Quick test to verify the network improvements work."""
    print("=== Quick Test of Network Improvements ===")
    
    # 1. Create environment
    print("1. Creating environment...")
    def make_env():
        return gym.make('CartPole-v1')
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # 2. Create topology networks
    print("2. Creating topology networks...")
    actor_network = create_topology_network('fully_connected', size=30, seed=42)
    critic_network = create_topology_network('fully_connected', size=30, seed=43)
    
    # 3. Create policy
    print("3. Creating policy...")
    policy_kwargs = {
        'actor_network': actor_network,
        'critic_network': critic_network,
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {
            'eps': 1e-7,
            'betas': (0.9, 0.999),
        }
    }
    
    # 4. Test different training durations
    print("4. Testing different training durations...")
    training_durations = [2000, 5000,10000]  # Increased for proper CartPole learning
    results = {}
    
    for timesteps in training_durations:
        print(f"\n--- Testing {timesteps} timesteps ---")
        
        # Create fresh model for each test
        model = PPO(
            TopologyPolicy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=2e-4,  # Increased from 1e-4 for better learning
            n_steps=1024,  # Keep for frequent updates
            batch_size=64,
            n_epochs=4,  # Keep reduced epochs
            gamma=0.99,
            clip_range=0.15,  # Increased from 0.1 for more learning
            ent_coef=0.02,  # Increased from 0.01 to maintain exploration
            verbose=0  # Reduce output
        )
        
        # Train
        model.learn(total_timesteps=timesteps)
        
        # Test performance
        test_rewards = []
        for i in range(5):
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
        best_reward = max(test_rewards)
        results[timesteps] = {
            'avg_reward': avg_reward,
            'best_reward': best_reward,
            'all_rewards': test_rewards
        }
        
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Best reward: {best_reward:.2f}")
        print(f"  All rewards: {test_rewards}")
    
    # 5. Analyze learning progression
    print("\n5. Learning progression analysis...")
    timesteps_list = sorted(results.keys())
    avg_rewards = [results[ts]['avg_reward'] for ts in timesteps_list]
    
    print(f"Training progression:")
    for i, ts in enumerate(timesteps_list):
        print(f"  {ts} timesteps: {avg_rewards[i]:.2f} avg reward")
    
    # Check if learning is improving
    if len(avg_rewards) >= 2:
        improvement = avg_rewards[-1] - avg_rewards[0]
        print(f"Overall improvement: {improvement:.2f}")
        
        if improvement > 5:
            print("✅ STRONG: Clear learning progression detected")
        elif improvement > 0:
            print("⚠️  MODERATE: Some learning progression")
        else:
            print("❌ POOR: No learning progression")
    
    # 6. Compare with random baseline
    print("\n6. Random baseline test...")
    env_random = DummyVecEnv([make_env])
    env_random = VecNormalize(env_random, norm_obs=True, norm_reward=False)
    random_rewards = []
    
    for i in range(10):  # More random episodes for better baseline
        obs = env_random.reset()
        total_reward = 0
        done = False
        while not done:
            action = [env_random.action_space.sample()]
            obs, reward, done, info = env_random.step(action)
            total_reward += reward[0]
            done = done[0]
        random_rewards.append(total_reward)
    
    avg_random = np.mean(random_rewards)
    random_std = np.std(random_rewards)
    print(f"Random baseline: {avg_random:.2f} ± {random_std:.2f}")
    print(f"Random rewards: {random_rewards}")
    
    # 7. Statistical significance test
    print("\n7. Statistical significance analysis...")
    best_policy_rewards = results[timesteps_list[-1]]['all_rewards']
    
    # Simple t-test approximation
    policy_mean = np.mean(best_policy_rewards)
    policy_std = np.std(best_policy_rewards)
    random_mean = avg_random
    random_std = random_std
    
    # Calculate effect size
    pooled_std = np.sqrt((policy_std**2 + random_std**2) / 2)
    effect_size = (policy_mean - random_mean) / pooled_std
    
    print(f"Policy performance: {policy_mean:.2f} ± {policy_std:.2f}")
    print(f"Effect size: {effect_size:.2f}")
    
    if effect_size > 0.5:
        print("✅ SIGNIFICANT: Policy clearly outperforms random")
    elif effect_size > 0.2:
        print("⚠️  MODERATE: Policy shows some advantage")
    else:
        print("❌ WEAK: Policy advantage is minimal")
    
    # 8. Policy behavior analysis
    print("\n8. Policy behavior analysis...")
    best_model = PPO(
        TopologyPolicy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-4,  # Updated to match new configuration
        n_steps=1024,  # Updated to match new configuration
        batch_size=64,  # Updated to match new configuration
        n_epochs=4,
        gamma=0.99,
        clip_range=0.15,  # Updated for consistency
        ent_coef=0.02,  # Updated for consistency
        verbose=0
    )
    best_model.learn(total_timesteps=timesteps_list[-1])
    
    # Test different states
    test_states = [
        np.array([0.0, 0.0, 0.0, 0.0]),      # Center
        np.array([1.0, 0.0, 0.0, 0.0]),      # Right
        np.array([-1.0, 0.0, 0.0, 0.0]),     # Left
        np.array([0.0, 1.0, 0.0, 0.0]),      # Moving right
        np.array([0.0, -1.0, 0.0, 0.0]),     # Moving left
    ]
    
    print("Policy responses to different states:")
    for i, state in enumerate(test_states):
        obs_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            actions, values, log_probs = best_model.policy.forward(obs_tensor)
            distribution = best_model.policy.get_distribution(
                best_model.policy.mlp_extractor.forward_actor(
                    best_model.policy.features_extractor(obs_tensor)
                )
            )
            probs = distribution.distribution.probs
            entropy = distribution.entropy()
        
        print(f"  State {i+1} {state}: action={actions[0]}, probs={probs[0].tolist()}, entropy={entropy[0]:.3f}")
    
    # 9. Final assessment
    print("\n9. Final assessment...")
    success_criteria = 0
    
    # Criterion 1: Outperform random
    if policy_mean > avg_random:
        success_criteria += 1
        print("✅ Criterion 1: Policy outperforms random")
    else:
        print("❌ Criterion 1: Policy does not outperform random")
    
    # Criterion 2: Learning progression
    if len(avg_rewards) >= 2 and avg_rewards[-1] > avg_rewards[0]:
        success_criteria += 1
        print("✅ Criterion 2: Learning progression detected")
    else:
        print("❌ Criterion 2: No learning progression")
    
    # Criterion 3: Reasonable performance
    if policy_mean > 15:
        success_criteria += 1
        print("✅ Criterion 3: Reasonable performance level")
    else:
        print("❌ Criterion 3: Performance too low")
    
    # Criterion 4: Policy shows variation
    if policy_std > 1.0:
        success_criteria += 1
        print("✅ Criterion 4: Policy shows appropriate variation")
    else:
        print("❌ Criterion 4: Policy too deterministic")
    
    print(f"\nSuccess criteria met: {success_criteria}/4")
    
    if success_criteria >= 3:
        print("🎉 EXCELLENT: System is working well!")
    elif success_criteria >= 2:
        print("✅ GOOD: System shows promise")
    else:
        print("⚠️  NEEDS WORK: System needs further tuning")
    
    print("\n=== Enhanced Test Complete ===")
    
    return success_criteria >= 3

if __name__ == "__main__":
    success = quick_test_improvements()
    if success:
        print("\n🎉 SUCCESS: Improvements are working! You can now run the full test.")
    else:
        print("\n⚠️  WARNING: May need further tuning before full test.") 