from typing import Dict, Any, List
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from .base import BaseNetwork

class PyTorchFeedForwardNetwork(nn.Module):
    """PyTorch-compatible feed-forward network that can receive gradients."""
    
    def __init__(self, topology: nx.Graph, input_nodes: List[int], output_nodes: List[int],
                 network_params: Dict[str, Any]):
        super().__init__()
        
        self.topology = topology
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.network_params = network_params
        
        # Ensure topology is a DAG
        if not nx.is_directed_acyclic_graph(self.topology):
            raise ValueError("FFN requires a Directed Acyclic Graph (DAG) topology")
        
        # Store node ordering for forward pass
        self._node_order = list(nx.topological_sort(self.topology))
        
        # Get activation function from network_params
        self.activation = network_params.get('activation', 'relu')
        
        # Create PyTorch parameters for weights and biases
        self._create_parameters()
    
    def _create_parameters(self):
        """Create PyTorch parameters for weights and biases with Xavier initialization."""
        self.node_biases = nn.ParameterDict()
        self.node_weights = nn.ParameterDict()
        
        for node in self.topology.nodes():
            # Count incoming connections for Xavier initialization
            num_incoming = len(list(self.topology.predecessors(node)))
            
            # Xavier initialization: std = sqrt(2.0 / (fan_in + fan_out))
            if num_incoming > 0:
                weight_std = np.sqrt(2.0 / num_incoming) * 2.0  # Scale up by 2x
            else:
                weight_std = 0.2  # Default for nodes with no incoming edges
            
            # Create bias parameter with larger initialization
            self.node_biases[str(node)] = nn.Parameter(torch.randn(1) * 0.1)
            
            # Create weight parameters for incoming edges
            for neighbor in self.topology.predecessors(node):
                weight_name = f"{neighbor}_to_{node}"
                self.node_weights[weight_name] = nn.Parameter(torch.randn(1) * weight_std)
    
    def _apply_activation(self, x: torch.Tensor, is_output: bool = False) -> torch.Tensor:
        """Apply activation function."""
        if is_output:
            # For output nodes, use linear activation (no tanh)
            return x
        else:
            # For hidden nodes, use ReLU
            return torch.relu(x)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process inputs through the network using PyTorch tensors."""
        batch_size = inputs.shape[0]
        
        # Initialize activations for all nodes
        activations = {node: torch.zeros(batch_size, 1, device=inputs.device) 
                      for node in self.topology.nodes()}
        
        # Set input node activations
        for i, node in enumerate(self.input_nodes):
            if i < inputs.shape[1]:
                activations[node] = inputs[:, i:i+1]
        
        # Process through network in topological order
        for layer in self._node_order:
            if layer not in self.input_nodes:
                weighted_sum = self.node_biases[str(layer)].expand(batch_size, 1)
                
                for neighbor in self.topology.predecessors(layer):
                    weight_name = f"{neighbor}_to_{layer}"
                    weight = self.node_weights[weight_name]
                    weighted_sum = weighted_sum + activations[neighbor] * weight
                
                # Apply activation function
                is_output = layer in self.output_nodes
                activations[layer] = self._apply_activation(weighted_sum, is_output)
        
        # Return output node activations
        output_nodes = self.output_nodes[:3]  # Always return 3 outputs
        outputs = []
        for node in output_nodes:
            outputs.append(activations[node])
        
        # Scale up outputs for better RL performance
        output_tensor = torch.cat(outputs, dim=1)
        scaled_output = output_tensor * 5.0  # Scale up by 5x
        
        return scaled_output

class FeedForwardNetwork(BaseNetwork):
    """Feed-forward neural network implementation (custom for SB3 integration)."""
    
    def __init__(self, topology: nx.Graph, input_nodes: List[int], output_nodes: List[int],
                 network_params: Dict[str, Any]):
        """Initialize the FFN."""
        super().__init__(topology, input_nodes, output_nodes, network_params)
        
        # Ensure topology is a DAG
        if not nx.is_directed_acyclic_graph(self.topology):
            raise ValueError("FFN requires a Directed Acyclic Graph (DAG) topology")
        
        # Store node ordering for forward pass
        self._node_order = list(nx.topological_sort(self.topology))
        
        # Get activation function from network_params
        self.activation = network_params.get('activation', 'relu')
    
    def _initialize_node_states(self) -> Dict[str, Any]:
        """Initialize node states for FFN with Xavier initialization."""
        states = {}
        for node in self.topology.nodes():
            # Count incoming connections for Xavier initialization
            num_incoming = len(list(self.topology.predecessors(node)))
            
            # Xavier initialization: std = sqrt(2.0 / (fan_in + fan_out))
            # For simplicity, use sqrt(2.0 / fan_in) for weights
            if num_incoming > 0:
                weight_std = np.sqrt(2.0 / num_incoming)
            else:
                weight_std = 0.1  # Default for nodes with no incoming edges
            
            states[node] = {
                'activation': 0.0,
                'bias': np.random.normal(0, 0.01),  # Small bias initialization
                'weights': {
                    neighbor: np.random.normal(0, weight_std)
                    for neighbor in self.topology.predecessors(node)
                }
            }
        return states
    
    def _apply_activation(self, x: float, is_output: bool = False) -> float:
        """Apply activation function."""
        if is_output:
            # For output nodes, use linear activation (no tanh)
            return x
        else:
            # For hidden nodes, use ReLU
            return max(0.0, x)
    
    def forward(self, inputs: Dict[int, Any]) -> Dict[int, Any]:
        """Process inputs through the FFN (safe for SB3 integration).
        Ensures all activations are scalars, not arrays.
        Output nodes use linear activation (no tanh) for proper logits.
        """
        self._clear_active_edges()
        activations = {node: 0.0 for node in self.topology.nodes()}
        
        # Set input node activations
        for node, value in inputs.items():
            if node in self.input_nodes:
                # If value is array-like, take scalar
                if isinstance(value, (np.ndarray, list)):
                    activations[node] = float(np.asarray(value).flatten()[0])
                else:
                    activations[node] = float(value)
        
        # Process through network in topological order
        for layer in self._node_order:
            if layer not in self.input_nodes:
                active_predecessors = [
                    neighbor for neighbor in self.topology.predecessors(layer)
                    if float(activations[neighbor]) != 0.0
                ]
                self._update_active_edges(layer, active_predecessors)
                is_valid, error_msg = self._validate_runtime_edges()
                if not is_valid:
                    raise ValueError(f"Runtime topology violation: {error_msg}")
                
                weighted_sum = self.node_states[layer]['bias']
                for neighbor in self.topology.predecessors(layer):
                    weighted_sum += (
                        float(activations[neighbor]) * 
                        self.node_states[layer]['weights'][neighbor]
                    )
                
                # Apply activation function
                is_output = layer in self.output_nodes
                activations[layer] = self._apply_activation(weighted_sum, is_output)
        
        # Return output node activations (linear, no tanh)
        output_nodes = self.output_nodes[:3]  # Always return 3 outputs
        output_vec = {}
        for i, node in enumerate(output_nodes):
            output_vec[node] = activations[node]
        return output_vec
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get FFN-specific metrics."""
        return {
            'num_weights': sum(len(node['weights']) for node in self.node_states.values()),
            'num_biases': len(self.node_states),
            'avg_weight_magnitude': np.mean([
                abs(w) for node in self.node_states.values()
                for w in node['weights'].values()
            ]),
            'avg_bias_magnitude': np.mean([
                abs(node['bias']) for node in self.node_states.values()
            ]),
            'activation_function': self.activation
        } 