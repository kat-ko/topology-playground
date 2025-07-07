#!/usr/bin/env python3
"""
Test script to use custom topology networks with real RL training.
This will help understand what's needed to integrate real RL into the existing system.
"""

import sys
import torch
import numpy as np
import gymnasium as gym
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.topologies.small_world import SmallWorldTopology
from src.agents.network_agent import NetworkAgent
from src.tasks.rl_tasks import RLTaskGenerator, RLTaskEvaluator
from src.utils.parameter_budget import ParameterBudgetCalculator

def test_small_world_ppo_training():
    """Test PPO training with a small-world network topology."""
    print("="*80)
    print("TESTING SMALL-WORLD NETWORK WITH REAL PPO TRAINING")
    print("="*80)
    
    # Configuration
    network_size = 20
    seed = 42
    task_name = 'cartpole'
    max_episodes = 100
    max_steps_per_episode = 200
    
    print(f"Network size: {network_size}")
    print(f"Task: {task_name}")
    print(f"Max episodes: {max_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    # 1. Create small-world network topology
    print("\n1. Creating small-world network topology...")
    topology = SmallWorldTopology(
        size=network_size,
        k=4,
        p=0.1,
        inter_layer_prob=0.1,
        seed=seed
    )
    
    # Create network using parameter budget calculator
    config = {
        'network_sizes': [network_size],
        'topologies': ['small_world'],
        'network_types': ['ffn'],
        'num_layers': [1],
        'seeds': [seed],
        'experiment_types': ['same_size'],
        'small_world_params': {'k': 4, 'p': 0.1, 'inter_layer_prob': 0.1},
        'modular_params': {'num_modules': 4, 'inter_module_prob': 0.1, 'intra_module_prob': 0.3, 'inter_layer_prob': 0.1},
        'hybrid_params': {'k': 4, 'p': 0.1, 'num_modules': 4, 'inter_module_prob': 0.1, 'intra_module_prob': 0.3, 'inter_layer_prob': 0.1},
        'fully_connected_params': {'inter_layer_prob': 1.0, 'intra_layer_prob': 1.0},
        'network_params': {
            'ffn': {
                'activation': 'relu',
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'rnn': {
                'hidden_size': 32,
                'sequence_length': 10,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        },
        'num_io_nodes': 5
    }
    calculator = ParameterBudgetCalculator(config)
    
    network = calculator.create_network(
        topology='small_world',
        size=network_size,
        experiment_type='same_size',
        network_type='ffn',
        num_layers=1,
        seed=seed
    )
    
    print(f"Network created successfully")
    print(f"Input nodes: {len(network.input_nodes)}")
    print(f"Output nodes: {len(network.output_nodes)}")
    
    # 2. Create RL task environment
    print("\n2. Creating RL task environment...")
    task_generator = RLTaskGenerator(seed=seed)
    env, task_config = task_generator.generate_cartpole_task()
    
    print(f"Environment: {task_config.env_name}")
    print(f"State dimension: {task_config.state_dim}")
    print(f"Action dimension: {task_config.action_dim}")
    
    # 3. Create network agent
    print("\n3. Creating network agent...")
    agent = NetworkAgent(network, task_config)
    
    print(f"Agent created successfully")
    print(f"Active outputs: {len(agent.active_outputs)}")
    
    # 4. Test basic agent functionality
    print("\n4. Testing basic agent functionality...")
    test_state = env.reset()[0]
    print(f"Test state shape: {test_state.shape}")
    print(f"Test state: {test_state}")
    
    action = agent.select_action(test_state)
    print(f"Selected action: {action}")
    
    # 5. Run real RL training
    print("\n5. Running real RL training...")
    
    # Training parameters
    learning_rate = 0.001
    gamma = 0.99
    
    # Simple policy gradient training
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Store episode data for training
        states = []
        actions = []
        rewards = []
        
        while not (done or truncated) and episode_length < max_steps_per_episode:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Simple policy gradient update (very basic)
        if len(states) > 0:
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            
            # Normalize returns
            returns = np.array(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Update network (simplified - would need proper PPO implementation)
            # This is just a placeholder for the actual training logic
            pass
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}: Reward = {episode_reward:.1f}, Avg (last 10) = {avg_reward:.1f}")
    
    # 6. Evaluate final performance
    print("\n6. Evaluating final performance...")
    evaluator = RLTaskEvaluator()
    final_metrics = evaluator.evaluate_episodes(env, agent, task_config, num_episodes=20)
    
    print(f"Final evaluation results:")
    print(f"  Mean reward: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}")
    print(f"  Mean length: {final_metrics['mean_length']:.2f} ± {final_metrics['std_length']:.2f}")
    print(f"  Solved rate: {final_metrics['solved_rate']:.2f}")
    
    # 7. Analysis of what's needed for full integration
    print("\n7. Analysis for full integration...")
    print("What's working:")
    print("  ✅ Network topology creation")
    print("  ✅ Agent action selection")
    print("  ✅ Environment interaction")
    print("  ✅ Basic episode execution")
    
    print("\nWhat's missing for full PPO:")
    print("  ❌ Proper PPO algorithm implementation")
    print("  ❌ Policy gradient updates")
    print("  ❌ Value function estimation")
    print("  ❌ Advantage calculation")
    print("  ❌ Clipping mechanism")
    print("  ❌ Proper loss functions")
    
    print("\nIntegration options:")
    print("  Option 1: Implement full PPO from scratch")
    print("  Option 2: Use Stable Baselines3 with custom policy networks")
    print("  Option 3: Use existing NetworkAgent with simple policy gradient")
    
    env.close()
    return {
        'episode_rewards': episode_rewards,
        'final_metrics': final_metrics,
        'network': network,
        'agent': agent
    }

if __name__ == "__main__":
    results = test_small_world_ppo_training()
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80) 