"""
Universal Topology Policy for Stable Baselines3
Integrates your topology networks with PPO training.
Enhanced with transfer learning controls and minimal adapters.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import Dict, Any, List, Tuple, Type, Optional
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# Import your topology modules
from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..topologies.hybrid import HybridTopology
from ..topologies.fully_connected import FullyConnectedTopology
from ..networks.ffn import FeedForwardNetwork

class MinimalAdapter(nn.Module):
    """Minimal adapter with configurable complexity to prevent overfitting."""
    
    def __init__(self, input_dim: int, output_dim: int, adapter_type: str = 'linear', hidden_dim: int = 8):
        super().__init__()
        self.adapter_type = adapter_type
        
        if adapter_type == 'linear':
            # Simple linear projection
            self.projection = nn.Linear(input_dim, output_dim)
        elif adapter_type == 'tiny_mlp':
            # Tiny MLP with minimal hidden layer
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        elif adapter_type == 'identity':
            # Identity mapping (for dimension matching)
            if input_dim == output_dim:
                self.projection = nn.Identity()
            else:
                # Pad or truncate to match dimensions
                self.projection = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class UniversalTopologyFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using universal topology networks with minimal adapters.
    Enhanced with transfer learning controls and gradient tracking.
    """
    
    def __init__(self, 
                 observation_space: spaces.Box,
                 topology_name: str,
                 universal_input_dim: int = 6,
                 universal_output_dim: int = 3,
                 hidden_size: int = 100,
                 topology_params: Dict[str, Any] = None,
                 adapter_type: str = 'linear',
                 adapter_hidden_dim: int = 8,
                 freeze_adapters: bool = False):
        """
        Initialize the universal topology feature extractor.
        
        Args:
            observation_space: Gym observation space
            topology_name: Name of topology ('small_world', 'modular', etc.)
            universal_input_dim: Fixed input dimension for universal topology
            universal_output_dim: Fixed output dimension for universal topology
            hidden_size: Number of hidden nodes
            topology_params: Parameters for topology generation
            adapter_type: Type of adapter ('linear', 'tiny_mlp', 'identity')
            adapter_hidden_dim: Hidden dimension for tiny_mlp adapters
            freeze_adapters: Whether to freeze adapter weights
        """
        super().__init__(observation_space, features_dim=universal_output_dim)
        
        self.topology_name = topology_name
        self.universal_input_dim = universal_input_dim
        self.universal_output_dim = universal_output_dim
        self.hidden_size = hidden_size
        self.task_input_dim = observation_space.shape[0]
        self.adapter_type = adapter_type
        self.freeze_adapters = freeze_adapters
        
        # Generate universal topology
        self.universal_topology = self._generate_universal_topology(topology_params)
        
        # Create minimal input adapter
        self.input_adapter = MinimalAdapter(
            self.task_input_dim, 
            universal_input_dim, 
            adapter_type, 
            adapter_hidden_dim
        )
        
        # Initialize topology network
        self.topology_network = self._create_topology_network()
        
        # Track if topology weights are frozen (for transfer learning)
        self.topology_frozen = False
        
        # Gradient tracking for analysis
        self.gradient_norms = {
            'topology': [],
            'input_adapter': [],
            'output_adapter': []
        }
        
        # Freeze adapters if requested
        if freeze_adapters:
            self.freeze_adapter_weights()
    
    def _generate_universal_topology(self, topology_params: Dict[str, Any]) -> nx.Graph:
        """Generate the universal topology with maximum dimensions."""
        if topology_params is None:
            topology_params = {}
        
        # Calculate total size for topology (input + hidden + output)
        total_size = self.universal_input_dim + self.hidden_size + self.universal_output_dim
        
        if self.topology_name == 'small_world':
            topology = SmallWorldTopology(
                size=total_size,
                k=topology_params.get('k', 4),
                p=topology_params.get('p', 0.3),
                num_layers=topology_params.get('num_layers', 1),
                inter_layer_prob=topology_params.get('inter_layer_prob', 0.1),
                seed=topology_params.get('seed', None)
            )
        elif self.topology_name == 'modular':
            topology = ModularTopology(
                size=total_size,
                num_modules=topology_params.get('num_modules', 4),
                inter_module_prob=topology_params.get('inter_module_prob', 0.1),
                intra_module_prob=topology_params.get('intra_module_prob', 0.8),
                num_layers=topology_params.get('num_layers', 1),
                inter_layer_prob=topology_params.get('inter_layer_prob', 0.1),
                seed=topology_params.get('seed', None)
            )
        elif self.topology_name == 'hybrid':
            topology = HybridTopology(
                size=total_size,
                num_modules=topology_params.get('num_modules', 4),
                k=topology_params.get('k', 4),
                p=topology_params.get('p', 0.3),
                inter_module_prob=topology_params.get('inter_module_prob', 0.1),
                num_layers=topology_params.get('num_layers', 1),
                inter_layer_prob=topology_params.get('inter_layer_prob', 0.1),
                seed=topology_params.get('seed', None)
            )
        elif self.topology_name == 'fully_connected':
            topology = FullyConnectedTopology(
                size=total_size,
                num_layers=topology_params.get('num_layers', 1),
                inter_layer_prob=topology_params.get('inter_layer_prob', 0.5),
                intra_layer_prob=topology_params.get('intra_layer_prob', 0.8),
                seed=topology_params.get('seed', None)
            )
        else:
            raise ValueError(f"Unknown topology: {self.topology_name}")
        
        return topology.generate()
    
    def _create_topology_network(self) -> FeedForwardNetwork:
        """Create the topology network with universal dimensions."""
        # Define input/output nodes for universal topology
        input_nodes = list(range(self.universal_input_dim))
        output_nodes = list(range(self.universal_input_dim + self.hidden_size, 
                                 self.universal_input_dim + self.hidden_size + self.universal_output_dim))
        
        # Create network parameters
        network_params = {
            'learning_rate': 0.001,
            'activation': 'tanh'
        }
        
        return FeedForwardNetwork(
            topology=self.universal_topology,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
            network_params=network_params
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through universal topology with minimal adapters.
        
        Args:
            observations: Input tensor of shape (batch_size, task_input_dim)
            
        Returns:
            Output tensor of shape (batch_size, universal_output_dim)
        """
        # Input adapter: project to universal input space
        universal_input = self.input_adapter(observations)  # (batch_size, universal_input_dim)
        
        # Convert to dictionary format for topology network
        input_dict = {i: universal_input[:, i] for i in range(self.universal_input_dim)}
        
        # Forward through topology network
        topology_output = self.topology_network.forward(input_dict)
        
        # Convert topology output to tensor
        output_values = []
        batch_size = observations.shape[0]
        for i in range(self.universal_output_dim):
            output_node_idx = self.universal_input_dim + self.hidden_size + i
            val = topology_output.get(output_node_idx, 0.0)
            # Ensure val is a tensor of shape (batch_size,)
            if not torch.is_tensor(val):
                val = torch.tensor(val, dtype=torch.float32, device=observations.device)
                if val.dim() == 0:
                    val = val.repeat(batch_size)
            output_values.append(val)
        universal_output = torch.stack(output_values, dim=1)  # (batch_size, universal_output_dim)
        return universal_output
    
    def freeze_topology_weights(self):
        """Freeze the topology network weights for transfer learning."""
        for param in self.topology_network.parameters():
            param.requires_grad = False
        self.topology_frozen = True
    
    def unfreeze_topology_weights(self):
        """Unfreeze the topology network weights."""
        for param in self.topology_network.parameters():
            param.requires_grad = True
        self.topology_frozen = False
    
    def freeze_adapter_weights(self):
        """Freeze the adapter weights to ensure topology does the work."""
        for param in self.input_adapter.parameters():
            param.requires_grad = False
        self.freeze_adapters = True
    
    def unfreeze_adapter_weights(self):
        """Unfreeze the adapter weights."""
        for param in self.input_adapter.parameters():
            param.requires_grad = True
        self.freeze_adapters = False
    
    def track_gradient_norms(self):
        """Track gradient norms for topology vs adapter analysis."""
        topology_norm = 0.0
        adapter_norm = 0.0
        
        # Calculate topology gradient norm
        for param in self.topology_network.parameters():
            if param.grad is not None:
                topology_norm += param.grad.norm().item() ** 2
        
        # Calculate adapter gradient norm
        for param in self.input_adapter.parameters():
            if param.grad is not None:
                adapter_norm += param.grad.norm().item() ** 2
        
        self.gradient_norms['topology'].append(np.sqrt(topology_norm))
        self.gradient_norms['input_adapter'].append(np.sqrt(adapter_norm))
    
    def get_gradient_analysis(self) -> Dict[str, Any]:
        """Get gradient analysis for topology vs adapter contribution."""
        if not self.gradient_norms['topology']:
            return {'topology_ratio': 0.0, 'adapter_ratio': 0.0}
        
        avg_topology = np.mean(self.gradient_norms['topology'])
        avg_adapter = np.mean(self.gradient_norms['input_adapter'])
        total = avg_topology + avg_adapter
        
        if total == 0:
            return {'topology_ratio': 0.0, 'adapter_ratio': 0.0}
        
        return {
            'topology_ratio': avg_topology / total,
            'adapter_ratio': avg_adapter / total,
            'topology_norm': avg_topology,
            'adapter_norm': avg_adapter
        }
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for different components."""
        topology_params = sum(p.numel() for p in self.topology_network.parameters())
        input_adapter_params = sum(p.numel() for p in self.input_adapter.parameters())
        
        return {
            'topology': topology_params,
            'input_adapter': input_adapter_params,
            'total': topology_params + input_adapter_params,
            'adapter_type': self.adapter_type,
            'adapter_frozen': self.freeze_adapters
        }

class UniversalTopologyActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy using universal topology networks with minimal adapters.
    Enhanced with transfer learning controls and gradient tracking.
    Properly handles discrete action spaces.
    """
    
    def __init__(self, 
                 observation_space: spaces.Space,
                 action_space: spaces.Space,
                 lr_schedule: callable,
                 topology_name: str,
                 universal_input_dim: int = 6,
                 universal_output_dim: int = 3,
                 hidden_size: int = 100,
                 topology_params: Dict[str, Any] = None,
                 adapter_type: str = 'linear',
                 adapter_hidden_dim: int = 8,
                 freeze_adapters: bool = False,
                 freeze_output_adapters: bool = False,
                 *args, **kwargs):
        """
        Initialize the universal topology actor-critic policy.
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space (must be Discrete)
            lr_schedule: Learning rate schedule function
            topology_name: Name of topology ('small_world', 'modular', etc.)
            universal_input_dim: Fixed input dimension for universal topology
            universal_output_dim: Fixed output dimension for universal topology
            hidden_size: Number of hidden nodes
            topology_params: Parameters for topology generation
            adapter_type: Type of adapter ('linear', 'tiny_mlp', 'identity')
            adapter_hidden_dim: Hidden dimension for tiny_mlp adapters
            freeze_adapters: Whether to freeze input adapter weights
            freeze_output_adapters: Whether to freeze output adapter weights
        """
        # Validate action space
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError("Universal topology policy only supports discrete action spaces")
        
        # Create features extractor with minimal adapters
        features_extractor_class = UniversalTopologyFeaturesExtractor
        features_extractor_kwargs = {
            'topology_name': topology_name,
            'universal_input_dim': universal_input_dim,
            'universal_output_dim': universal_output_dim,
            'hidden_size': hidden_size,
            'topology_params': topology_params,
            'adapter_type': adapter_type,
            'adapter_hidden_dim': adapter_hidden_dim,
            'freeze_adapters': freeze_adapters
        }
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args, **kwargs
        )
        
        # Create minimal output adapters
        self.output_adapter_type = adapter_type
        self.freeze_output_adapters = freeze_output_adapters
        
        # Actor head (universal_output_dim -> action_space.n) - outputs logits for discrete actions
        self.action_net = MinimalAdapter(
            universal_output_dim,
            action_space.n,
            adapter_type,
            adapter_hidden_dim
        )
        
        # Critic head (universal_output_dim -> 1)
        self.value_net = MinimalAdapter(
            universal_output_dim,
            1,
            adapter_type,
            adapter_hidden_dim
        )
        
        # Freeze output adapters if requested
        if freeze_output_adapters:
            self.freeze_output_adapter_weights()
    
    def freeze_output_adapter_weights(self):
        """Freeze the output adapter weights."""
        for param in self.action_net.parameters():
            param.requires_grad = False
        for param in self.value_net.parameters():
            param.requires_grad = False
        self.freeze_output_adapters = True
    
    def unfreeze_output_adapter_weights(self):
        """Unfreeze the output adapter weights."""
        for param in self.action_net.parameters():
            param.requires_grad = True
        for param in self.value_net.parameters():
            param.requires_grad = True
        self.freeze_output_adapters = False
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient tracking.
        Returns logits for discrete actions, not continuous actions.
        """
        # Get features from topology network
        features = self.extract_features(obs)
        
        # Track gradients if training
        if self.training:
            self.features_extractor.track_gradient_norms()
        
        # Actor and critic heads
        latent_pi = self.action_net(features)  # Logits for discrete actions
        latent_vf = self.value_net(features)   # Value function
        
        # For discrete actions, we return logits directly (no mean/std)
        # The parent class will handle the conversion to probabilities
        return latent_pi, None, latent_vf
    
    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for actor only."""
        features = self.extract_features(obs)
        return self.action_net(features)  # Returns logits
    
    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for critic only."""
        features = self.extract_features(obs)
        return self.value_net(features)
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for all components."""
        base_counts = self.features_extractor.get_parameter_count()
        action_params = sum(p.numel() for p in self.action_net.parameters())
        value_params = sum(p.numel() for p in self.value_net.parameters())
        
        return {
            **base_counts,
            'action_adapter': action_params,
            'value_adapter': value_params,
            'total': base_counts['total'] + action_params + value_params,
            'output_adapter_frozen': self.freeze_output_adapters
        }
    
    def get_gradient_analysis(self) -> Dict[str, Any]:
        """Get gradient analysis for all components."""
        base_analysis = self.features_extractor.get_gradient_analysis()
        
        # Calculate output adapter gradient norms
        action_norm = 0.0
        value_norm = 0.0
        
        for param in self.action_net.parameters():
            if param.grad is not None:
                action_norm += param.grad.norm().item() ** 2
        
        for param in self.value_net.parameters():
            if param.grad is not None:
                value_norm += param.grad.norm().item() ** 2
        
        return {
            **base_analysis,
            'action_adapter_norm': np.sqrt(action_norm),
            'value_adapter_norm': np.sqrt(value_norm)
        }

def create_universal_topology_policy(topology_name: str,
                                   universal_input_dim: int = 6,
                                   universal_output_dim: int = 3,
                                   hidden_size: int = 100,
                                   topology_params: Dict[str, Any] = None,
                                   adapter_type: str = 'linear',
                                   adapter_hidden_dim: int = 8,
                                   freeze_adapters: bool = False,
                                   freeze_output_adapters: bool = False) -> Type[ActorCriticPolicy]:
    """
    Create a universal topology policy with minimal adapters and transfer controls.
    
    Args:
        topology_name: Name of topology ('small_world', 'modular', etc.)
        universal_input_dim: Fixed input dimension for universal topology
        universal_output_dim: Fixed output dimension for universal topology
        hidden_size: Number of hidden nodes
        topology_params: Parameters for topology generation
        adapter_type: Type of adapter ('linear', 'tiny_mlp', 'identity')
        adapter_hidden_dim: Hidden dimension for tiny_mlp adapters
        freeze_adapters: Whether to freeze input adapter weights
        freeze_output_adapters: Whether to freeze output adapter weights
    """
    
    class SpecificTopologyPolicy(UniversalTopologyActorCriticPolicy):
        def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                topology_name=topology_name,
                universal_input_dim=universal_input_dim,
                universal_output_dim=universal_output_dim,
                hidden_size=hidden_size,
                topology_params=topology_params,
                adapter_type=adapter_type,
                adapter_hidden_dim=adapter_hidden_dim,
                freeze_adapters=freeze_adapters,
                freeze_output_adapters=freeze_output_adapters,
                *args, **kwargs
            )
    
    return SpecificTopologyPolicy 