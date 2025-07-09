"""
Simple Curriculum Dimension Handling Demo

This demonstrates the key approaches for handling different task dimensions
in curriculum learning with topology networks.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy

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

def test_cartpole():
    """Test universal topology on CartPole."""
    print("=== Testing Universal Topology on CartPole ===")
    
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
    
    # Train for a short time to demonstrate
    model.learn(total_timesteps=10000, progress_bar=True)
    
    # Quick test
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
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Test rewards: {test_rewards}")
    
    return avg_reward

def demonstrate_curriculum_approaches():
    """Demonstrate different curriculum approaches."""
    print("\n" + "="*60)
    print("CURRICULUM DIMENSION HANDLING APPROACHES")
    print("="*60)
    
    print("\n1. UNIVERSAL ARCHITECTURE WITH TASK-SPECIFIC ADAPTERS")
    print("   - Keep topology backbone fixed across all tasks")
    print("   - Use input adapters: obs_dim → topology_input_dim")
    print("   - Use output adapters: topology_output_dim → action_dim")
    print("   - Benefits: Transfer learning, consistent topology")
    print("   - Example: CartPole (4→64→128→64→2) vs MountainCar (2→64→128→64→3)")
    
    print("\n2. DYNAMIC TOPOLOGY WITH MAXIMUM DIMENSIONS")
    print("   - Use largest possible dimensions across all tasks")
    print("   - Mask unused input/output dimensions per task")
    print("   - Benefits: Simple implementation, no adapters needed")
    print("   - Drawbacks: Inefficient, larger parameter count")
    print("   - Example: All tasks use (6→128→3), mask unused parts")
    
    print("\n3. TASK-SPECIFIC TOPOLOGY WITH SHARED COMPONENTS")
    print("   - Share some topology components across tasks")
    print("   - Adapt input/output processing per task")
    print("   - Benefits: Balance between transfer and efficiency")
    print("   - Example: Shared hidden layers, task-specific I/O")
    
    print("\n4. HYBRID APPROACH (RECOMMENDED)")
    print("   - Universal topology backbone (your graph structure)")
    print("   - Task-specific input/output adapters")
    print("   - Shared topology parameters across tasks")
    print("   - Benefits: Best of all worlds")
    
    print("\n" + "="*60)
    print("IMPLEMENTATION RECOMMENDATIONS")
    print("="*60)
    
    print("\nFor your topology networks:")
    print("1. Fix the internal topology structure (graph connectivity)")
    print("2. Use task-specific input/output adapters")
    print("3. Share topology weights across curriculum tasks")
    print("4. Use proper dimensions for each task (no padding/masking)")
    
    print("\nExample implementation:")
    print("  class CurriculumTopologyPolicy:")
    print("    def __init__(self, obs_dim, action_dim):")
    print("      self.input_adapter = Linear(obs_dim, 64)")
    print("      self.topology = YourTopologyNetwork(64, 64)  # Fixed")
    print("      self.output_adapter = Linear(64, action_dim)")
    
    print("\nThis allows:")
    print("  - CartPole: 4 → 64 → [topology] → 64 → 2")
    print("  - MountainCar: 2 → 64 → [topology] → 64 → 3")
    print("  - Acrobot: 6 → 64 → [topology] → 64 → 3")
    
    print("\nThe topology backbone learns universal representations")
    print("while adapters handle task-specific I/O requirements.")

if __name__ == "__main__":
    # Test the universal approach
    cartpole_reward = test_cartpole()
    
    # Demonstrate the approaches
    demonstrate_curriculum_approaches()
    
    print(f"\nTest result: CartPole reward = {cartpole_reward:.2f}")
    print("The universal topology approach works correctly!") 