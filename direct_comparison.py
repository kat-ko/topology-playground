"""
Direct Comparison: Fully Connected Topology vs Standard MLP

This script creates a fully connected topology that should theoretically
be equivalent to a standard MLP, but with proper CartPole dimensions.
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

class TrainingCallback(BaseCallback):
    """Callback to track training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.value_losses = []
        self.policy_losses = []
        self.entropy_losses = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track episode rewards
        if len(self.training_env.buf_rews) > 0:
            self.current_episode_reward += self.training_env.buf_rews[0]
            self.current_episode_length += 1
            
            if self.training_env.buf_dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        return True

class SimpleFullyConnectedPolicy(ActorCriticPolicy):
    """Simple fully connected policy that should match standard MLP performance."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Simple architecture matching the standard MLP
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),  # 4 → 64
            nn.ReLU(),
            nn.Linear(64, 64),                 # 64 → 64
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.action_net = nn.Sequential(
            nn.Linear(64, 32),                 # 64 → 32
            nn.ReLU(),
            nn.Linear(32, action_space.n)      # 32 → 2
        )
        
        # Critic head (value)
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),                 # 64 → 32
            nn.ReLU(),
            nn.Linear(32, 1)                   # 32 → 1
        )
    
    def forward_actor(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.action_net(shared_features)
    
    def forward_critic(self, obs):
        features = self.extract_features(obs)
        shared_features = self.shared_net(features)
        return self.value_net(shared_features)

class CartPoleTopologyPolicy(ActorCriticPolicy):
    """Topology-based policy with proper CartPole dimensions (4 inputs, 2 outputs)."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Create a simple fully connected topology equivalent to MLP
        # 4 input nodes + 64 hidden nodes + 2 output nodes = 70 total nodes
        self.num_input_nodes = 4
        self.num_hidden_nodes = 64
        self.num_output_nodes = 2
        self.total_nodes = self.num_input_nodes + self.num_hidden_nodes + self.num_output_nodes
        
        # Create weight matrices (equivalent to MLP layers)
        # Layer 1: 4 → 64
        self.layer1_weights = nn.Parameter(torch.randn(4, 64) * 0.1)
        self.layer1_bias = nn.Parameter(torch.randn(64) * 0.1)
        
        # Layer 2: 64 → 64  
        self.layer2_weights = nn.Parameter(torch.randn(64, 64) * 0.1)
        self.layer2_bias = nn.Parameter(torch.randn(64) * 0.1)
        
        # Actor head: 64 → 32 → 2
        self.actor_layer1 = nn.Linear(64, 32)
        self.actor_layer2 = nn.Linear(32, action_space.n)
        
        # Critic head: 64 → 32 → 1
        self.critic_layer1 = nn.Linear(64, 32)
        self.critic_layer2 = nn.Linear(32, 1)
    
    def forward_actor(self, obs):
        """Forward pass for actor (policy)."""
        features = self.extract_features(obs)
        
        # Simulate topology computation (equivalent to MLP)
        hidden1 = torch.relu(features @ self.layer1_weights + self.layer1_bias)
        hidden2 = torch.relu(hidden1 @ self.layer2_weights + self.layer2_bias)
        
        # Actor head
        actor_hidden = torch.relu(self.actor_layer1(hidden2))
        return self.actor_layer2(actor_hidden)
    
    def forward_critic(self, obs):
        """Forward pass for critic (value)."""
        features = self.extract_features(obs)
        
        # Simulate topology computation (equivalent to MLP)
        hidden1 = torch.relu(features @ self.layer1_weights + self.layer1_bias)
        hidden2 = torch.relu(hidden1 @ self.layer2_weights + self.layer2_bias)
        
        # Critic head
        critic_hidden = torch.relu(self.critic_layer1(hidden2))
        return self.critic_layer2(critic_hidden)

def test_standard_mlp():
    """Test standard MLP with PPO."""
    print("=== Testing Standard MLP ===")
    
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        SimpleFullyConnectedPolicy,
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
    print(f"Standard MLP - Average test reward: {avg_reward:.2f}")
    print(f"Standard MLP - All test rewards: {test_rewards}")
    
    return {
        'type': 'Standard MLP',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def test_cartpole_topology():
    """Test topology-based policy with proper CartPole dimensions."""
    print("\n=== Testing CartPole Topology ===")
    
    def make_env():
        return gym.make('CartPole-v1')
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    model = PPO(
        CartPoleTopologyPolicy,
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
    print(f"CartPole Topology - Average test reward: {avg_reward:.2f}")
    print(f"CartPole Topology - All test rewards: {test_rewards}")
    
    return {
        'type': 'CartPole Topology',
        'avg_reward': avg_reward,
        'test_rewards': test_rewards,
        'episode_rewards': callback.episode_rewards
    }

def compare_results():
    """Compare the results of both approaches."""
    print("\n=== Direct Comparison Study ===")
    
    mlp_results = test_standard_mlp()
    topology_results = test_cartpole_topology()
    
    print("\n" + "="*60)
    print("DIRECT COMPARISON RESULTS")
    print("="*60)
    
    print(f"\nStandard MLP:")
    print(f"  Average reward: {mlp_results['avg_reward']:.2f}")
    print(f"  Test rewards: {mlp_results['test_rewards']}")
    
    print(f"\nCartPole Topology:")
    print(f"  Average reward: {topology_results['avg_reward']:.2f}")
    print(f"  Test rewards: {topology_results['test_rewards']}")
    
    # Performance comparison
    mlp_avg = mlp_results['avg_reward']
    topology_avg = topology_results['avg_reward']
    
    print(f"\nPerformance Analysis:")
    print(f"  MLP performance: {mlp_avg:.2f}")
    print(f"  Topology performance: {topology_avg:.2f}")
    print(f"  Performance ratio: {topology_avg/mlp_avg:.3f}")
    
    if topology_avg >= mlp_avg * 0.9:
        print("✅ SUCCESS: Topology matches MLP performance!")
    elif topology_avg >= mlp_avg * 0.5:
        print("⚠️  MODERATE: Topology shows reasonable performance")
    else:
        print("❌ POOR: Topology significantly underperforms MLP")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Test performance comparison
    networks = ['Standard MLP', 'CartPole Topology']
    avg_rewards = [mlp_avg, topology_avg]
    ax1.bar(networks, avg_rewards, color=['blue', 'orange'])
    ax1.set_title('Final Test Performance')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, alpha=0.3)
    
    # Training progression comparison
    if mlp_results['episode_rewards']:
        ax2.plot(mlp_results['episode_rewards'], label='Standard MLP', alpha=0.7)
    if topology_results['episode_rewards']:
        ax2.plot(topology_results['episode_rewards'], label='CartPole Topology', alpha=0.7)
    ax2.set_title('Training Progression')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('direct_comparison_results.png', dpi=300, bbox_inches='tight')
    print("\nComparison plots saved to 'direct_comparison_results.png'")
    
    return mlp_results, topology_results

if __name__ == "__main__":
    mlp_results, topology_results = compare_results()
    print("\n=== Direct Comparison Complete ===") 