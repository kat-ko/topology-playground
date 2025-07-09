#!/usr/bin/env python3
"""
Debug script to investigate policy learning issues.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Add src to path
sys.path.append('src')

from minimal_sb3_topology_example import TopologyPolicy, create_topology_network

def debug_policy_learning():
    """Debug why the policy is not learning."""
    print("=== Debugging Policy Learning Issues ===")
    
    # 1. Create environment
    print("1. Creating environment...")
    env = gym.make('CartPole-v1')
    print(f"Environment: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 2. Create topology networks
    print("\n2. Creating topology networks...")
    actor_network = create_topology_network('fully_connected', size=30, seed=42)
    critic_network = create_topology_network('fully_connected', size=30, seed=43)
    
    # 3. Create policy
    print("\n3. Creating policy...")
    policy = TopologyPolicy(
        env.observation_space, 
        env.action_space, 
        lambda _: 3e-4,  # Higher learning rate
        actor_network, 
        critic_network
    )
    
    # 4. Test policy behavior
    print("\n4. Testing policy behavior...")
    test_obs = env.reset()[0]
    print(f"Test observation: {test_obs}")
    
    # Test multiple observations
    test_observations = [
        test_obs,
        np.array([0.0, 0.0, 0.0, 0.0]),  # Center position
        np.array([1.0, 1.0, 1.0, 1.0]),  # Extreme position
        np.array([-1.0, -1.0, -1.0, -1.0]),  # Opposite extreme
    ]
    
    for i, obs in enumerate(test_observations):
        obs_tensor = torch.tensor([obs], dtype=torch.float32)
        
        # Get policy outputs
        with torch.no_grad():
            actions, values, log_probs = policy.forward(obs_tensor)
            distribution = policy.get_distribution(policy.mlp_extractor.forward_actor(
                policy.features_extractor(obs_tensor)
            ))
            probs = distribution.distribution.probs
        
        print(f"\nTest observation {i+1}: {obs}")
        print(f"  Actions: {actions}")
        print(f"  Values: {values}")
        print(f"  Log probs: {log_probs}")
        print(f"  Action probabilities: {probs}")
        print(f"  Probability entropy: {distribution.entropy()}")
    
    # 5. Test gradient flow with loss
    print("\n5. Testing gradient flow with loss...")
    obs_tensor = torch.tensor([test_obs], dtype=torch.float32)
    actions_tensor = torch.tensor([0], dtype=torch.long)  # Action 0
    
    # Forward pass
    actions, values, log_probs = policy.forward(obs_tensor)
    
    # Create a simple loss
    value_loss = nn.MSELoss()(values, torch.tensor([[0.0]]))  # Target value 0
    policy_loss = -log_probs[0]  # Negative log likelihood
    
    total_loss = value_loss + policy_loss
    
    print(f"Value loss: {value_loss.item():.6f}")
    print(f"Policy loss: {policy_loss.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    param_count = 0
    
    for name, param in policy.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            if grad_norm > 0.01:  # Only show significant gradients
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
    
    avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0
    print(f"Average gradient norm: {avg_grad_norm:.6f}")
    print(f"Parameters with gradients: {param_count}")
    
    # 6. Test with different learning rates
    print("\n6. Testing with different learning rates...")
    learning_rates = [1e-3, 3e-4, 1e-4, 1e-5]
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        # Create new policy with this learning rate
        policy_lr = TopologyPolicy(
            env.observation_space, 
            env.action_space, 
            lambda _: lr,
            actor_network, 
            critic_network
        )
        
        # Test forward pass
        obs_tensor = torch.tensor([test_obs], dtype=torch.float32)
        actions, values, log_probs = policy_lr.forward(obs_tensor)
        
        print(f"  Actions: {actions}")
        print(f"  Values: {values}")
        print(f"  Log probs: {log_probs}")
    
    # 7. Test network outputs directly
    print("\n7. Testing network outputs directly...")
    obs_tensor = torch.tensor([test_obs], dtype=torch.float32)
    
    # Pad to 6 dimensions
    if obs_tensor.shape[1] < 6:
        padding = torch.zeros(obs_tensor.shape[0], 6 - obs_tensor.shape[1])
        obs_tensor = torch.cat([obs_tensor, padding], dim=1)
    
    with torch.no_grad():
        actor_output = actor_network(obs_tensor)
        critic_output = critic_network(obs_tensor)
        
        print(f"Actor network output: {actor_output}")
        print(f"Critic network output: {critic_output}")
        
        # Test through action head
        actor_head = policy.mlp_extractor.actor_head
        critic_head = policy.mlp_extractor.critic_head
        
        logits = actor_head(actor_output)
        value = critic_head(critic_output)
        
        print(f"Final logits: {logits}")
        print(f"Final value: {value}")
        
        # Check probabilities
        probs = torch.softmax(logits, dim=1)
        print(f"Action probabilities: {probs}")
        print(f"Probability entropy: {-(probs * torch.log(probs + 1e-8)).sum(dim=1)}")
    
    # 8. Test with random actions
    print("\n8. Testing with random actions...")
    for episode in range(5):
        obs = env.reset()[0]
        total_reward = 0
        done = False
        
        while not done:
            # Use random actions instead of policy
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if truncated:
                break
        
        print(f"Random episode {episode+1}: {total_reward}")
    
    env.close()
    print("\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_policy_learning() 