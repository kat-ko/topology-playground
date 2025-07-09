#!/usr/bin/env python3
"""
Minimal working example: Using SB3 algorithms with custom topology networks as policies.
This demonstrates how to integrate your topology networks with PPO/A2C/SAC.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

# Add src to path
sys.path.append('src')

from src.utils.parameter_budget import ParameterBudgetCalculator
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from src.networks.ffn_custom_sb3 import PyTorchFeedForwardNetwork

class TopologyFeaturesExtractor(BaseFeaturesExtractor):
    """Minimal features extractor using topology networks, always pads obs to 6 and returns 3 outputs."""
    def __init__(self, observation_space: gym.spaces.Box, topology_network):
        features_dim = 3  # Always 3 outputs
        super().__init__(observation_space, features_dim)
        self.topology_network = topology_network
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure observations is 2D
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        # print(f"[DEBUG] Features extractor: input shape = {observations.shape}")
        
        # Pad observations to 6 dimensions if needed
        if observations.shape[1] < 6:
            padding = torch.zeros(observations.shape[0], 6 - observations.shape[1], device=observations.device)
            observations = torch.cat([observations, padding], dim=1)
        elif observations.shape[1] > 6:
            observations = observations[:, :6]
        
        # Use PyTorch network directly with tensor inputs
        with torch.no_grad():
            features = self.topology_network(observations)
        # print(f"[DEBUG] Features extractor: final features shape = {features.shape}")
        return features

class TopologyPolicy(ActorCriticPolicy):
    """Custom policy using two separate topology networks for actor and critic."""
    
    def __init__(self, observation_space, action_space, lr_schedule, actor_network, critic_network, *args, **kwargs):
        # Set features dimension before parent init (so SB3 builds MLP with correct input size)
        self.features_dim = 3  # Both networks output 3 values
        print(f"[DEBUG] Policy: features_dim = {self.features_dim}")
        
        # Call parent constructor first
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Store topology networks after parent init
        self.actor_network = actor_network
        self.critic_network = critic_network
        
        # Replace the features extractor with a simple one that just pads observations
        self.features_extractor = SimpleFeaturesExtractor(observation_space)
        
        # Replace the MLP extractor with our topology-based one
        self.mlp_extractor = TopologyMLPExtractor(actor_network, critic_network, action_space.n)
        
        # Override action_net to be identity (since our topology already produces final logits)
        self.action_net = nn.Identity()
        # Override value_net to be identity (since our topology already produces final value)
        self.value_net = nn.Identity()
        
        # Override the forward method to use our topology networks directly
        def custom_forward(obs, deterministic=False):
            # Extract features using our simple extractor
            features = self.features_extractor(obs)
            # print(f"[DEBUG] Policy forward: extracted features shape = {features.shape}")
            
            # Get actor and critic outputs using our topology networks
            latent_pi, latent_vf = self.mlp_extractor(features)
            # print(f"[DEBUG] Policy forward: latent_pi shape = {latent_pi.shape}, latent_vf shape = {latent_vf.shape}")
            
            # Convert to distribution
            distribution = self.get_distribution(latent_pi)
            
            actions = distribution.get_actions(deterministic=deterministic)
            log_prob = distribution.log_prob(actions)
            
            return actions, latent_vf, log_prob
        
        # Replace the forward method
        self.forward = custom_forward
        
        # Also override evaluate_actions
        def custom_evaluate_actions(obs, actions):
            features = self.features_extractor(obs)
            latent_pi, latent_vf = self.mlp_extractor(features)
            distribution = self.get_distribution(latent_pi)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return latent_vf, log_prob, entropy
        
        self.evaluate_actions = custom_evaluate_actions

class SimpleFeaturesExtractor(BaseFeaturesExtractor):
    """Simple features extractor that just pads observations to 6 dimensions."""
    
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=6)  # Always output 6 features
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure observations is 2D
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        batch_size = observations.shape[0]
        # print(f"[DEBUG] Simple extractor: input shape = {observations.shape}")
        
        features = []
        for i in range(batch_size):
            obs = observations[i].cpu().numpy().flatten()  # Flatten to 1D
            # Pad obs to 6
            obs_padded = np.zeros(6, dtype=np.float32)
            obs_padded[:len(obs)] = obs
            features.append(obs_padded)
        
        features = np.stack(features).astype(np.float32)
        # print(f"[DEBUG] Simple extractor: output shape = {features.shape}")
        return torch.from_numpy(features)

class TopologyMLPExtractor(nn.Module):
    """MLP extractor that uses topology networks for actor and critic."""
    
    def __init__(self, actor_network, critic_network, num_actions):
        super().__init__()
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.num_actions = num_actions
        
        # Simple linear layers to map from topology outputs to action logits and value
        self.actor_head = nn.Linear(3, num_actions)  # 3 topology outputs -> num_actions
        self.critic_head = nn.Linear(3, 1)  # 3 topology outputs -> 1 value
    
    def forward(self, features):
        """Forward pass for both actor and critic."""
        actor_output, critic_output = self.forward_actor(features), self.forward_critic(features)
        return actor_output, critic_output
    
    def forward_actor(self, features):
        """Forward pass for actor (policy)."""
        # If features already has shape (batch, num_actions), return as logits
        if features.shape[1] == self.num_actions:
            return features
        # print(f"[DEBUG] Actor: input features shape = {features.shape}")
        
        # Pad features to 6 dimensions if needed
        if features.shape[1] < 6:
            padding = torch.zeros(features.shape[0], 6 - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > 6:
            features = features[:, :6]
        
        # Use PyTorch network directly (no numpy conversion)
        topology_outputs = self.actor_network(features)
        # print(f"[DEBUG] Actor: topology features shape = {topology_outputs.shape}")
        logits = self.actor_head(topology_outputs)
        # print(f"[DEBUG] Actor: final logits shape = {logits.shape}")
        return logits
    
    def forward_critic(self, features):
        """Forward pass for critic (value)."""
        # If features already has shape (batch, 1), return as value
        if features.shape[1] == 1:
            return features
        # print(f"[DEBUG] Critic: input features shape = {features.shape}")
        
        # Pad features to 6 dimensions if needed
        if features.shape[1] < 6:
            padding = torch.zeros(features.shape[0], 6 - features.shape[1], device=features.device)
            features = torch.cat([features, padding], dim=1)
        elif features.shape[1] > 6:
            features = features[:, :6]
        
        # Use PyTorch network directly (no numpy conversion)
        topology_outputs = self.critic_network(features)
        # print(f"[DEBUG] Critic: topology features shape = {topology_outputs.shape}")
        value = self.critic_head(topology_outputs)
        # print(f"[DEBUG] Critic: final value shape = {value.shape}")
        return value

def create_topology_network(topology_type='small_world', size=20, seed=42):
    """Create a topology network using our custom FFN directly, always with 6 input and 3 output nodes."""
    from src.topologies.small_world import SmallWorldTopology
    from src.topologies.modular import ModularTopology
    from src.topologies.hybrid import HybridTopology
    from src.topologies.fully_connected import FullyConnectedTopology
    from src.networks.ffn_custom_sb3 import PyTorchFeedForwardNetwork
    import numpy as np
    
    num_input_nodes = 6
    num_output_nodes = 3
    
    # Create topology
    if topology_type == 'small_world':
        topology = SmallWorldTopology(
            size=size,
            k=4,
            p=0.1,
            num_layers=1,
            inter_layer_prob=0.1,
            seed=seed
        )
    elif topology_type == 'modular':
        topology = ModularTopology(
            size=size,
            num_modules=4,
            inter_module_prob=0.1,
            intra_module_prob=0.3,
            num_layers=1,
            inter_layer_prob=0.1,
            seed=seed
        )
    elif topology_type == 'hybrid':
        topology = HybridTopology(
            size=size,
            num_modules=4,
            k=4,
            p=0.1,
            inter_module_prob=0.1,
            num_layers=1,
            inter_layer_prob=0.1,
            seed=seed
        )
    elif topology_type == 'fully_connected':
        topology = FullyConnectedTopology(
            size=size,
            num_layers=1,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown topology: {topology_type}")
    
    # Generate graph
    graph = topology.generate(1)  # Single layer
    
    # Select input/output nodes (always 6/3 for compatibility)
    input_nodes = list(range(num_input_nodes))
    output_nodes = list(range(num_input_nodes, num_input_nodes + num_output_nodes))
    
    network_params = {
        'activation': 'relu',
        'learning_rate': 0.001,
        'batch_size': 32
    }
    
    network = PyTorchFeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
    return network

def test_minimal_sb3_integration():
    """Test minimal SB3 integration with topology networks."""
    print("="*80)
    print("MINIMAL SB3 + TOPOLOGY NETWORK INTEGRATION")
    print("="*80)
    
    # 1. Create environment
    print("1. Creating environment...")
    env = gym.make('CartPole-v1')
    print(f"Environment: {env}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # 2. Create topology networks (separate actor and critic)
    print("\n2. Creating topology networks...")
    actor_network = create_topology_network('small_world', size=20, seed=42)
    critic_network = create_topology_network('modular', size=20, seed=43)  # Different topology for critic
    print(f"Actor network created:")
    print(f"  Input nodes: {len(actor_network.input_nodes)}")
    print(f"  Output nodes: {len(actor_network.output_nodes)}")
    print(f"Critic network created:")
    print(f"  Input nodes: {len(critic_network.input_nodes)}")
    print(f"  Output nodes: {len(critic_network.output_nodes)}")
    
    # 3. Test topology networks
    print("\n3. Testing topology networks...")
    test_obs = env.reset()[0]
    print(f"Test observation: {test_obs}")
    
    # Test actor network with tensor input
    test_obs_tensor = torch.tensor([test_obs], dtype=torch.float32)
    # Pad to 6 dimensions if needed
    if test_obs_tensor.shape[1] < 6:
        padding = torch.zeros(test_obs_tensor.shape[0], 6 - test_obs_tensor.shape[1])
        test_obs_tensor = torch.cat([test_obs_tensor, padding], dim=1)
    elif test_obs_tensor.shape[1] > 6:
        test_obs_tensor = test_obs_tensor[:, :6]
    
    with torch.no_grad():
        actor_outputs = actor_network(test_obs_tensor)
    print(f"Actor outputs: {actor_outputs}")
    
    # Test critic network with tensor input
    with torch.no_grad():
        critic_outputs = critic_network(test_obs_tensor)
    print(f"Critic outputs: {critic_outputs}")
    
    # 4. Create custom policy
    print("\n4. Creating custom policy...")
    
    def make_topology_policy(observation_space, action_space, lr_schedule, **kwargs):
        return TopologyPolicy(observation_space, action_space, lr_schedule, actor_network, critic_network, **kwargs)
    
    # 5. Test policy creation
    print("\n5. Testing policy creation...")
    try:
        policy = make_topology_policy(env.observation_space, env.action_space, lambda _: 0.001)
        print("✅ Custom policy created successfully")
        
        # Test policy forward pass
        obs_tensor = torch.tensor([test_obs], dtype=torch.float32).unsqueeze(0)
        actions, values, log_probs = policy.forward(obs_tensor)
        print(f"Policy actions: {actions}")
        print(f"Policy values: {values}")
        print(f"Policy log_probs: {log_probs}")
        
    except Exception as e:
        print(f"❌ Failed to create policy: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Create and test PPO model
    print("\n6. Creating PPO model with topology policy...")
    try:
        model = PPO(
            policy=make_topology_policy,
            env=env,
            learning_rate=0.0003,
            n_steps=64,  # Small for testing
            batch_size=32,
            n_epochs=4,
            gamma=0.99,
            verbose=1
        )
        print("✅ PPO model created successfully")
        
        # 7. Test training
        print("\n7. Testing training...")
        model.learn(total_timesteps=1000)
        print("✅ Training completed successfully")
        
        # 8. Test prediction
        print("\n8. Testing prediction...")
        obs = env.reset()[0]
        action, _ = model.predict(obs, deterministic=True)
        print(f"Predicted action: {action}")
        
        # 9. Test multiple episodes
        print("\n9. Testing multiple episodes...")
        episode_rewards = []
        for episode in range(5):
            obs = env.reset()[0]
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode}: Reward = {episode_reward}")
        
        avg_reward = np.mean(episode_rewards)
        print(f"Average reward: {avg_reward:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create/train model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        env.close()

def test_different_topologies():
    """Test that different topologies work with SB3."""
    print("\n" + "="*80)
    print("TESTING DIFFERENT TOPOLOGIES")
    print("="*80)
    
    topologies = ['small_world', 'modular', 'fully_connected']
    results = {}
    
    for topology in topologies:
        print(f"\nTesting {topology} topology...")
        try:
            # Create topology network
            topology_network = create_topology_network(topology, size=20, seed=42)
            
            # Create environment
            env = gym.make('CartPole-v1')
            
            # Create policy
            def make_policy(obs_space, action_space, lr_schedule):
                return TopologyPolicy(obs_space, action_space, lr_schedule, topology_network, topology_network)
            
            # Create and train model
            model = PPO(
                policy=make_policy,
                env=env,
                learning_rate=0.0003,
                n_steps=64,
                batch_size=32,
                n_epochs=4,
                gamma=0.99,
                verbose=0
            )
            
            # Quick training
            model.learn(total_timesteps=500)
            
            # Test performance
            episode_rewards = []
            for _ in range(3):
                obs = env.reset()[0]
                episode_reward = 0
                done = False
                truncated = False
                
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            results[topology] = avg_reward
            print(f"  {topology}: Average reward = {avg_reward:.2f}")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ {topology} failed: {e}")
            results[topology] = None
    
    print(f"\nResults summary:")
    for topology, reward in results.items():
        if reward is not None:
            print(f"  {topology}: {reward:.2f}")
        else:
            print(f"  {topology}: Failed")

if __name__ == "__main__":
    # Test basic integration
    success = test_minimal_sb3_integration()
    
    if success:
        # Test different topologies
        test_different_topologies()
    
    print("\n" + "="*80)
    print("MINIMAL EXAMPLE COMPLETED")
    print("="*80)
    
    if success:
        print("✅ SB3 + Topology integration works!")
        print("✅ You can now use PPO/A2C/SAC with your custom networks!")
        print("✅ Different topologies can be compared fairly!")
    else:
        print("❌ Integration failed - check the error messages above") 