#!/usr/bin/env python3
"""
Test gradient flow through topology networks.
"""

import sys
import torch
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append('src')

from src.networks.ffn_custom_sb3 import PyTorchFeedForwardNetwork
from src.topologies.fully_connected import FullyConnectedTopology

def test_gradient_flow():
    """Test if gradients flow through the topology network."""
    print("=== Testing Gradient Flow ===")
    
    # 1. Create a simple topology network
    topology = FullyConnectedTopology(size=10, num_layers=1, seed=42)
    graph = topology.generate(1)
    
    network = PyTorchFeedForwardNetwork(
        graph, 
        input_nodes=list(range(6)), 
        output_nodes=list(range(6, 9)),
        network_params={'activation': 'relu'}
    )
    
    print(f"Network created: {type(network).__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in network.parameters())}")
    
    # 2. Create a simple input
    input_tensor = torch.randn(2, 6, requires_grad=True)  # Batch size 2, 6 features
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input requires grad: {input_tensor.requires_grad}")
    
    # 3. Forward pass
    output = network(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output requires grad: {output.requires_grad}")
    
    # 4. Create a simple loss
    target = torch.randn_like(output)
    loss = torch.nn.functional.mse_loss(output, target)
    print(f"Loss: {loss.item():.6f}")
    
    # 5. Backward pass
    loss.backward()
    
    # 6. Check gradients
    print("\n=== Gradient Analysis ===")
    
    # Check input gradients
    if input_tensor.grad is not None:
        print(f"✅ Input gradients exist: {input_tensor.grad.shape}")
        print(f"   Input grad norm: {input_tensor.grad.norm().item():.6f}")
    else:
        print("❌ No input gradients!")
    
    # Check network parameter gradients
    total_grad_norm = 0
    num_params_with_grad = 0
    
    for name, param in network.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            num_params_with_grad += 1
            print(f"✅ {name}: grad norm = {grad_norm:.6f}")
        else:
            print(f"❌ {name}: no gradient!")
    
    print(f"\nTotal parameters with gradients: {num_params_with_grad}")
    print(f"Average gradient norm: {total_grad_norm / max(num_params_with_grad, 1):.6f}")
    
    if num_params_with_grad > 0:
        print("✅ SUCCESS: Gradients are flowing through the network!")
        return True
    else:
        print("❌ FAILURE: No gradients detected!")
        return False

def test_ppo_integration():
    """Test if the topology network works with PPO-style forward pass."""
    print("\n=== Testing PPO Integration ===")
    
    # Create network
    topology = FullyConnectedTopology(size=10, num_layers=1, seed=42)
    graph = topology.generate(1)
    
    network = PyTorchFeedForwardNetwork(
        graph, 
        input_nodes=list(range(6)), 
        output_nodes=list(range(6, 9)),
        network_params={'activation': 'relu'}
    )
    
    # Simulate PPO forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 4, requires_grad=True)  # CartPole observation
    
    # Pad to 6 dimensions (like in the policy)
    padding = torch.zeros(batch_size, 2, device=obs.device)
    padded_obs = torch.cat([obs, padding], dim=1)
    
    print(f"Original obs shape: {obs.shape}")
    print(f"Padded obs shape: {padded_obs.shape}")
    
    # Forward pass
    output = network(padded_obs)
    print(f"Network output shape: {output.shape}")
    
    # Simulate actor head (linear layer)
    actor_head = torch.nn.Linear(3, 2)  # 3 features -> 2 actions
    logits = actor_head(output)
    print(f"Actor logits shape: {logits.shape}")
    
    # Simulate critic head (linear layer)
    critic_head = torch.nn.Linear(3, 1)  # 3 features -> 1 value
    value = critic_head(output)
    print(f"Critic value shape: {value.shape}")
    
    # Create loss
    target_logits = torch.randn_like(logits)
    target_value = torch.randn_like(value)
    
    policy_loss = torch.nn.functional.mse_loss(logits, target_logits)
    value_loss = torch.nn.functional.mse_loss(value, target_value)
    total_loss = policy_loss + value_loss
    
    print(f"Policy loss: {policy_loss.item():.6f}")
    print(f"Value loss: {value_loss.item():.6f}")
    print(f"Total loss: {total_loss.item():.6f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    print("\n=== PPO Integration Gradient Check ===")
    
    if obs.grad is not None:
        print(f"✅ Observation gradients: {obs.grad.norm().item():.6f}")
    else:
        print("❌ No observation gradients!")
    
    if network.node_weights['0_to_6'].grad is not None:
        print(f"✅ Network weight gradients: {network.node_weights['0_to_6'].grad.norm().item():.6f}")
    else:
        print("❌ No network weight gradients!")
    
    if actor_head.weight.grad is not None:
        print(f"✅ Actor head gradients: {actor_head.weight.grad.norm().item():.6f}")
    else:
        print("❌ No actor head gradients!")
    
    if critic_head.weight.grad is not None:
        print(f"✅ Critic head gradients: {critic_head.weight.grad.norm().item():.6f}")
    else:
        print("❌ No critic head gradients!")
    
    return True

if __name__ == "__main__":
    print("Testing topology network gradient flow...")
    
    # Test 1: Basic gradient flow
    grad_success = test_gradient_flow()
    
    # Test 2: PPO integration
    ppo_success = test_ppo_integration()
    
    print(f"\n=== Summary ===")
    print(f"Gradient flow test: {'✅ PASSED' if grad_success else '❌ FAILED'}")
    print(f"PPO integration test: {'✅ PASSED' if ppo_success else '❌ FAILED'}")
    
    if grad_success and ppo_success:
        print("🎉 All tests passed! The topology networks should work with PPO.")
    else:
        print("⚠️  Some tests failed. There may be issues with the implementation.") 