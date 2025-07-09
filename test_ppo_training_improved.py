#!/usr/bin/env python3
"""
Improved PPO training test with ReLU activations, better initialization, and proper gradient flow.
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.training_env.buf_rews) > 0:
            for rew in self.training_env.buf_rews:
                self.episode_rewards.append(rew)
                self.episode_lengths.append(200)  # CartPole typically runs for ~200 steps
        
        # Track losses from logger
        if self.logger.name_to_value:
            if 'train/value_loss' in self.logger.name_to_value:
                self.value_losses.append(self.logger.name_to_value['train/value_loss'])
            if 'train/policy_loss' in self.logger.name_to_value:
                self.policy_losses.append(self.logger.name_to_value['train/policy_loss'])
            if 'train/entropy_loss' in self.logger.name_to_value:
                self.entropy_losses.append(self.logger.name_to_value['train/entropy_loss'])
        
        return True

def create_improved_topology_network(topology_type, size, seed, num_layers=2):
    """Create topology network with ReLU activations and better initialization."""
    from src.topologies.small_world import SmallWorldTopology
    from src.topologies.modular import ModularTopology
    from src.topologies.hybrid import HybridTopology
    from src.topologies.fully_connected import FullyConnectedTopology
    from src.networks.ffn_custom_sb3 import PyTorchFeedForwardNetwork
    import numpy as np
    
    num_input_nodes = 6
    num_output_nodes = 3
    
    # Create topology with multiple layers
    if topology_type == 'small_world':
        topology = SmallWorldTopology(
            size=size,
            k=4,
            p=0.1,
            num_layers=num_layers,
            inter_layer_prob=0.3,  # Higher for multi-layer
            seed=seed
        )
    elif topology_type == 'modular':
        topology = ModularTopology(
            size=size,
            num_modules=4,
            inter_module_prob=0.1,
            intra_module_prob=0.3,
            num_layers=num_layers,
            inter_layer_prob=0.3,  # Higher for multi-layer
            seed=seed
        )
    elif topology_type == 'hybrid':
        topology = HybridTopology(
            size=size,
            num_modules=4,
            k=4,
            p=0.1,
            inter_module_prob=0.1,
            num_layers=num_layers,
            inter_layer_prob=0.3,  # Higher for multi-layer
            seed=seed
        )
    elif topology_type == 'fully_connected':
        topology = FullyConnectedTopology(
            size=size,
            num_layers=num_layers,
            inter_layer_prob=0.5,  # Moderate for multi-layer
            intra_layer_prob=0.8,  # High within layers
            seed=seed
        )
    else:
        raise ValueError(f"Unknown topology: {topology_type}")
    
    # Generate multi-layer graph
    graph = topology.generate(num_layers)
    
    # For multi-layer networks, we need to handle the graph structure properly
    if num_layers == 1:
        # Single layer: use the graph directly
        final_graph = graph
    else:
        # Multi-layer: combine all layers into one graph
        if isinstance(graph, list):
            # Combine multiple graphs into one
            final_graph = graph[0].copy()
            for layer_graph in graph[1:]:
                # Add nodes and edges from subsequent layers
                for node in layer_graph.nodes():
                    if node not in final_graph:
                        final_graph.add_node(node)
                for edge in layer_graph.edges():
                    final_graph.add_edge(*edge)
        else:
            final_graph = graph
    
    # Select input/output nodes (always 6/3 for compatibility)
    input_nodes = list(range(num_input_nodes))
    output_nodes = list(range(size - num_output_nodes, size))  # Last nodes as outputs
    
    network_params = {
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    
    network = PyTorchFeedForwardNetwork(final_graph, input_nodes, output_nodes, network_params)
    return network

def test_multi_layer_networks():
    """Test multi-layer topology networks to ensure they work correctly."""
    print("=== Testing Multi-Layer Topology Networks ===")
    
    # Test different layer configurations
    layer_configs = [1, 2, 3]
    topology_types = ['fully_connected', 'small_world', 'modular']
    
    for num_layers in layer_configs:
        print(f"\n--- Testing {num_layers}-layer networks ---")
        
        for topology_type in topology_types:
            print(f"\nTesting {topology_type} topology with {num_layers} layers...")
            
            try:
                # Create network
                network = create_improved_topology_network(
                    topology_type=topology_type,
                    size=30,  # Smaller size for testing
                    seed=42,
                    num_layers=num_layers
                )
                
                print(f"  ✅ Network created: {type(network).__name__}")
                print(f"  Network size: {len(network.topology.nodes())} nodes")
                print(f"  Network edges: {len(network.topology.edges())} edges")
                print(f"  Parameters: {sum(p.numel() for p in network.parameters())}")
                
                # Test forward pass
                test_input = torch.randn(4, 6, requires_grad=True)
                output = network(test_input)
                
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output requires grad: {output.requires_grad}")
                
                # Test gradient flow
                loss = output.sum()
                loss.backward()
                
                grad_count = sum(1 for p in network.parameters() if p.grad is not None)
                total_params = sum(p.numel() for p in network.parameters())
                
                print(f"  Gradients: {grad_count}/{total_params} parameters have gradients")
                
                if grad_count > 0:
                    print(f"  ✅ Gradient flow working")
                else:
                    print(f"  ❌ No gradients detected!")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()

def test_ppo_training_improved():
    """Test PPO training with improved topology networks."""
    print("=== Improved PPO Training Test with Multi-Layer Topology Networks ===")
    
    # First test multi-layer networks
    test_multi_layer_networks()
    
    # 1. Create environment
    print("1. Creating CartPole environment...")
    def make_env():
        return gym.make('CartPole-v1')
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 2. Create improved topology networks (multi-layer)
    print("\n2. Creating improved multi-layer topology networks...")
    actor_network = create_improved_topology_network('fully_connected', size=48, seed=42, num_layers=2)  # 2-layer network
    critic_network = create_improved_topology_network('fully_connected', size=48, seed=43, num_layers=2)  # 2-layer network
    
    print(f"Actor network: {type(actor_network).__name__}, size={len(actor_network.topology.nodes())}")
    print(f"Critic network: {type(critic_network).__name__}, size={len(critic_network.topology.nodes())}")
    
    # 3. Create policy
    print("\n3. Creating topology policy...")
    policy_kwargs = {
        'actor_network': actor_network,
        'critic_network': critic_network,
        # Optimizer configuration
        'optimizer_class': torch.optim.Adam,  # Default in SB3
        'optimizer_kwargs': {
            'eps': 1e-7,  # Adam epsilon
            'betas': (0.9, 0.999),  # Adam betas
        }
        # Alternative: Use SGD instead
        # 'optimizer_class': torch.optim.SGD,
        # 'optimizer_kwargs': {
        #     'momentum': 0.9,
        #     'weight_decay': 1e-4,
        # }
    }
    
    # 4. Create PPO agent with better hyperparameters for multi-layer networks
    print("\n4. Creating PPO agent...")
    model = PPO(
        TopologyPolicy,  # Pass the class, not a string
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,  # Lowered from 3e-4 for stability
        n_steps=1024,  # Reduced from 2048 for more frequent updates
        batch_size=64,
        n_epochs=4,  # Reduced from 10 to prevent overfitting
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Reduced from 0.2 for more conservative updates
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Reduced from 0.05 for better balance
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=None,
        verbose=1,
        seed=42,
        device='auto',
        _init_setup_model=True,
    )
    
    # 5. Setup logging
    print("\n5. Setting up logging...")
    log_dir = "ppo_training_logs_improved"
    os.makedirs(log_dir, exist_ok=True)
    configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 6. Setup callback
    callback = TrainingCallback()
    
    # 7. Train for longer (multi-layer networks need more time)
    print("\n6. Starting improved training...")
    total_timesteps = 100000  # More timesteps for multi-layer networks
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 8. Analyze results
    print("\n7. Analyzing training results...")
    
    # Calculate statistics
    total_episodes = len(callback.episode_rewards)
    final_reward = callback.episode_rewards[-1] if callback.episode_rewards else 0
    best_reward = max(callback.episode_rewards) if callback.episode_rewards else 0
    avg_reward = np.mean(callback.episode_rewards) if callback.episode_rewards else 0
    avg_length = np.mean(callback.episode_lengths) if callback.episode_lengths else 0
    
    # Calculate learning improvement
    if len(callback.episode_rewards) >= 20:
        first_10_avg = np.mean(callback.episode_rewards[:10])
        last_10_avg = np.mean(callback.episode_rewards[-10:])
        learning_improvement = last_10_avg - first_10_avg
    else:
        learning_improvement = 0
    
    print(f"\nTraining Statistics:")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total episodes: {total_episodes}")
    print(f"Final episode reward: {final_reward:.1f}")
    print(f"Best episode reward: {best_reward:.1f}")
    print(f"Average episode reward: {avg_reward:.2f}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Learning improvement (first 10 vs last 10 episodes): {learning_improvement:.2f}")
    
    if learning_improvement > 10:
        print("✅ POSITIVE: Clear learning improvement detected")
    elif learning_improvement > 0:
        print("⚠️  WARNING: Minimal learning improvement")
    else:
        print("❌ POOR: No learning improvement")
    
    # 9. Test final performance
    print("\n8. Testing final performance...")
    def make_test_env():
        return gym.make('CartPole-v1')
    test_env = DummyVecEnv([make_test_env])
    test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False)
    test_rewards = []
    
    for i in range(10):
        obs = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward[0]
            done = done[0]
        test_rewards.append(total_reward)
        print(f"Test episode {i+1}: {total_reward:.1f}")
    
    test_env.close()
    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage test reward: {avg_test_reward:.2f}")
    
    # 10. Create plots
    print("\n9. Creating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    if callback.episode_rewards:
        axes[0, 0].plot(callback.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
    
    # Value losses
    if callback.value_losses:
        axes[0, 1].plot(callback.value_losses)
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    # Policy losses
    if callback.policy_losses:
        axes[1, 0].plot(callback.policy_losses)
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Entropy losses
    if callback.entropy_losses:
        axes[1, 1].plot(callback.entropy_losses)
        axes[1, 1].set_title('Entropy Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training_results_improved.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'ppo_training_results_improved.png'")
    
    # 11. Empirical soundness analysis
    print("\n10. Empirical Soundness Analysis:")
    
    # Check if rewards improved over time
    if len(callback.episode_rewards) >= 100:
        first_quarter = np.mean(callback.episode_rewards[:len(callback.episode_rewards)//4])
        last_quarter = np.mean(callback.episode_rewards[-len(callback.episode_rewards)//4:])
        reward_improvement = last_quarter - first_quarter
        print(f"Reward improvement (first vs last quarter): {reward_improvement:.2f}")
    else:
        reward_improvement = 0
        print("Reward improvement (first vs last quarter): N/A (not enough data)")
    
    # Check value loss decrease
    if len(callback.value_losses) >= 10:
        value_loss_decrease = callback.value_losses[0] - callback.value_losses[-1]
        print(f"Value loss decrease: {value_loss_decrease:.4f}")
    else:
        value_loss_decrease = 0
        print("Value loss decrease: N/A (not enough data)")
    
    # Check episode length increase
    if len(callback.episode_lengths) >= 20:
        first_10_length = np.mean(callback.episode_lengths[:10])
        last_10_length = np.mean(callback.episode_lengths[-10:])
        length_increase = last_10_length - first_10_length
        print(f"Episode length increase: {length_increase:.2f}")
    else:
        length_increase = 0
        print("Episode length increase: N/A (not enough data)")
    
    # Check recent reward stability
    if len(callback.episode_rewards) >= 20:
        recent_rewards = callback.episode_rewards[-20:]
        reward_std = np.std(recent_rewards)
        print(f"Recent reward stability (std): {reward_std:.2f}")
    else:
        reward_std = 0
        print("Recent reward stability (std): N/A (not enough data)")
    
    # Overall assessment
    print(f"\nOverall Assessment:")
    if avg_test_reward > 150:
        print("✅ EXCELLENT: Agent learned to solve CartPole")
    elif avg_test_reward > 100:
        print("✅ GOOD: Agent shows significant learning")
    elif avg_test_reward > 50:
        print("⚠️  MODERATE: Agent shows some learning")
    else:
        print("❌ POOR: Agent shows minimal learning")
    
    if value_loss_decrease > 10:
        print("✅ POSITIVE: Value function is learning")
    else:
        print("⚠️  WARNING: Value function not improving significantly")
    
    if reward_improvement > 20:
        print("✅ POSITIVE: Clear reward improvement over time")
    else:
        print("⚠️  WARNING: Limited reward improvement")
    
    print(f"\n=== Improved Training Test Complete ===")
    print(f"Results saved to: {log_dir}/")
    print(f"Plots saved to: ppo_training_results_improved.png")

if __name__ == "__main__":
    test_ppo_training_improved() 