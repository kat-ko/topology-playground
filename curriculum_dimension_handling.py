"""
Curriculum Learning: Handling Different Task Dimensions

This file demonstrates different approaches to handle varying observation
and action spaces across curriculum tasks while maintaining topology networks.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ============================================================================
# APPROACH 1: Universal Architecture with Task-Specific Adapters
# ============================================================================

class UniversalTopologyPolicy(ActorCriticPolicy):
    """
    Universal topology policy that adapts to different tasks.
    
    Key idea: Keep topology backbone fixed, adapt input/output dimensions.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 topology_input_dim=64, topology_output_dim=64, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space.n
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Universal topology dimensions (fixed across all tasks)
        self.topology_input_dim = topology_input_dim
        self.topology_output_dim = topology_output_dim
        
        # Task-specific input adapter (obs_dim → topology_input_dim)
        self.input_adapter = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.topology_input_dim),
            nn.ReLU()
        )
        
        # Universal topology backbone (fixed internal structure)
        self.topology_backbone = nn.Sequential(
            nn.Linear(self.topology_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.topology_output_dim),
            nn.ReLU()
        )
        
        # Task-specific output adapters
        self.actor_adapter = nn.Sequential(
            nn.Linear(self.topology_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
        
        self.critic_adapter = nn.Sequential(
            nn.Linear(self.topology_output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        topology_input = self.input_adapter(features)
        topology_output = self.topology_backbone(topology_input)
        return self.actor_adapter(topology_output)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        topology_input = self.input_adapter(features)
        topology_output = self.topology_backbone(topology_input)
        return self.critic_adapter(topology_output)

# ============================================================================
# APPROACH 2: Dynamic Topology with Maximum Dimensions
# ============================================================================

class DynamicTopologyPolicy(ActorCriticPolicy):
    """
    Dynamic topology policy that uses maximum dimensions across all tasks.
    
    Key idea: Use largest possible dimensions, mask unused parts per task.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 max_obs_dim=6, max_action_dim=3, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Current task dimensions
        self.obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space.n
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Maximum dimensions across all curriculum tasks
        self.max_obs_dim = max_obs_dim
        self.max_action_dim = max_action_dim
        
        print(f"[DEBUG] Current task: obs={self.obs_dim}, actions={self.action_dim}")
        print(f"[DEBUG] Max dimensions: obs={self.max_obs_dim}, actions={self.max_action_dim}")
        
        # Create topology for maximum dimensions
        self.topology_backbone = nn.Sequential(
            nn.Linear(self.max_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.max_action_dim),
            nn.ReLU()
        )
        
        # Actor and critic heads
        self.actor_head = nn.Linear(self.max_action_dim, self.max_action_dim)
        self.critic_head = nn.Linear(self.max_action_dim, 1)
        
        # Create masks for current task
        self.obs_mask = torch.zeros(self.max_obs_dim)
        self.obs_mask[:self.obs_dim] = 1.0
        
        self.action_mask = torch.zeros(self.max_action_dim)
        self.action_mask[:self.action_dim] = 1.0
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        
        # Pad observations to max dimension
        if features.shape[1] < self.max_obs_dim:
            padding = torch.zeros(features.shape[0], self.max_obs_dim - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        # Apply observation mask
        features = features * self.obs_mask.to(features.device)
        
        # Forward through topology
        topology_output = self.topology_backbone(features)
        
        # Apply action mask
        topology_output = topology_output * self.action_mask.to(topology_output.device)
        
        # Get final logits
        logits = self.actor_head(topology_output)
        
        # Return only valid actions
        return logits[:, :self.action_dim]
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        
        # Pad observations to max dimension
        if features.shape[1] < self.max_obs_dim:
            padding = torch.zeros(features.shape[0], self.max_obs_dim - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        
        # Apply observation mask
        features = features * self.obs_mask.to(features.device)
        
        # Forward through topology
        topology_output = self.topology_backbone(features)
        
        # Get value
        return self.critic_head(topology_output)

# ============================================================================
# APPROACH 3: Task-Specific Topology with Shared Components
# ============================================================================

class SharedComponentTopologyPolicy(ActorCriticPolicy):
    """
    Topology policy with shared components across tasks.
    
    Key idea: Share some topology components, adapt others per task.
    """
    
    def __init__(self, observation_space, action_space, lr_schedule, 
                 shared_hidden_dim=64, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Task-specific dimensions
        self.obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else observation_space.n
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Shared hidden dimension
        self.shared_hidden_dim = shared_hidden_dim
        
        # Task-specific input processing
        self.input_processor = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.shared_hidden_dim),
            nn.ReLU()
        )
        
        # Shared topology components (these would be your actual topology networks)
        self.shared_layer1 = nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim)
        self.shared_layer2 = nn.Linear(self.shared_hidden_dim, self.shared_hidden_dim)
        
        # Task-specific output processing
        self.actor_processor = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )
        
        self.critic_processor = nn.Sequential(
            nn.Linear(self.shared_hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        
        # Task-specific input processing
        hidden = self.input_processor(features)
        
        # Shared topology computation
        hidden = torch.relu(self.shared_layer1(hidden))
        hidden = torch.relu(self.shared_layer2(hidden))
        
        # Task-specific output processing
        return self.actor_processor(hidden)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        
        # Task-specific input processing
        hidden = self.input_processor(features)
        
        # Shared topology computation
        hidden = torch.relu(self.shared_layer1(hidden))
        hidden = torch.relu(self.shared_layer2(hidden))
        
        # Task-specific output processing
        return self.critic_processor(hidden)

# ============================================================================
# TASK DEFINITIONS AND TESTING
# ============================================================================

TASK_CONFIGS = {
    'CartPole-v1': {
        'obs_dim': 4,
        'action_dim': 2,
        'solved_threshold': 195,
        'description': 'Balance pole on cart'
    },
    'MountainCar-v0': {
        'obs_dim': 2,
        'action_dim': 3,
        'solved_threshold': -110,
        'description': 'Drive car to top of mountain'
    },
    'Acrobot-v1': {
        'obs_dim': 6,
        'action_dim': 3,
        'solved_threshold': -100,
        'description': 'Swing up double pendulum'
    }
}

def test_approach_on_task(approach_class, task_name, timesteps=50000):
    """Test a specific approach on a given task."""
    print(f"\n=== Testing {approach_class.__name__} on {task_name} ===")
    
    def make_env():
        return gym.make(task_name)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # Create model with appropriate parameters
    if approach_class == DynamicTopologyPolicy:
        model = PPO(
            approach_class,
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
    else:
        model = PPO(
            approach_class,
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
    
    # Train
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
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
    solved_threshold = TASK_CONFIGS[task_name]['solved_threshold']
    solved = avg_reward >= solved_threshold
    
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Solved: {'✅ YES' if solved else '❌ NO'}")
    print(f"Test rewards: {test_rewards}")
    
    return {
        'approach': approach_class.__name__,
        'task': task_name,
        'avg_reward': avg_reward,
        'solved': solved,
        'test_rewards': test_rewards
    }

def compare_approaches():
    """Compare all approaches across all tasks."""
    print("=== Curriculum Dimension Handling Comparison ===")
    
    approaches = [
        UniversalTopologyPolicy,
        DynamicTopologyPolicy,
        SharedComponentTopologyPolicy
    ]
    
    tasks = list(TASK_CONFIGS.keys())
    
    results = []
    
    for approach in approaches:
        for task in tasks:
            result = test_approach_on_task(approach, task, timesteps=30000)  # Shorter for demo
            results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for approach in approaches:
        approach_name = approach.__name__
        print(f"\n{approach_name}:")
        
        for task in tasks:
            result = next(r for r in results if r['approach'] == approach_name and r['task'] == task)
            status = "✅ SOLVED" if result['solved'] else "❌ NOT SOLVED"
            print(f"  {task}: {result['avg_reward']:.2f} {status}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, approach in enumerate(approaches):
        approach_name = approach.__name__
        approach_results = [r for r in results if r['approach'] == approach_name]
        
        tasks = [r['task'] for r in approach_results]
        avg_rewards = [r['avg_reward'] for r in approach_results]
        solved = [r['solved'] for r in approach_results]
        
        colors = ['green' if s else 'red' for s in solved]
        
        axes[i].bar(tasks, avg_rewards, color=colors, alpha=0.7)
        axes[i].set_title(f'{approach_name}')
        axes[i].set_ylabel('Average Reward')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add solved thresholds
        for j, task in enumerate(tasks):
            threshold = TASK_CONFIGS[task]['solved_threshold']
            axes[i].axhline(y=threshold, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('curriculum_approaches_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plots saved to 'curriculum_approaches_comparison.png'")
    
    return results

if __name__ == "__main__":
    results = compare_approaches()
    print("\n=== Curriculum Dimension Handling Complete ===") 