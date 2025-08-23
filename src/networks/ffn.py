from typing import Dict, Any, List
import networkx as nx
import numpy as np
from .base import BaseNetwork
import torch

class FeedForwardNetwork(BaseNetwork):
    """Feed-forward neural network implementation."""
    
    def __init__(self, topology: nx.Graph, input_nodes: List[int], output_nodes: List[int],
                 network_params: Dict[str, Any]):
        """Initialize the FFN."""
        super().__init__(topology, input_nodes, output_nodes, network_params)
        
        # Ensure topology is a DAG
        if not nx.is_directed_acyclic_graph(self.topology):
            raise ValueError("FFN requires a Directed Acyclic Graph (DAG) topology")
        
        # Store node ordering for forward pass
        self._node_order = list(nx.topological_sort(self.topology))
    
    def parameters(self):
        """Return empty list since this network doesn't use PyTorch parameters."""
        return []
    
    def _initialize_node_states(self) -> Dict[str, Any]:
        """Initialize node states for FFN."""
        states = {}
        for node in list(self.topology.nodes()):
            # Only initialize weights for incoming edges (predecessors)
            states[node] = {
                'activation': 0.0,
                'bias': np.random.normal(0, 0.1),
                'weights': {
                    neighbor: np.random.normal(0, 0.1)
                    for neighbor in self.topology.predecessors(node)
                }
            }
        return states
    
    def forward(self, inputs: Dict[int, Any]) -> Dict[int, Any]:
        """Process inputs through the FFN.
        
        Args:
            inputs: Dictionary mapping input node indices to their input values
            
        Returns:
            Dictionary mapping output node indices to their output values
        """
        # Clear active edges at start of forward pass
        self._clear_active_edges()
        
        # Initialize activations with tensors
        # Get batch size from first input tensor
        first_input = next(iter(inputs.values()))
        if torch.is_tensor(first_input):
            batch_size = first_input.shape[0]
            device = first_input.device
        else:
            batch_size = 1
            device = torch.device('cpu')
        
        activations = {node: torch.zeros(batch_size, device=device) for node in list(self.topology.nodes())}
        
        # Set input node activations
        for node, value in inputs.items():
            if node in self.input_nodes:
                activations[node] = value
        
        # Process through network in topological order
        for layer in self._node_order:
            if layer not in self.input_nodes:
                # Get active predecessors
                active_predecessors = [
                    neighbor for neighbor in self.topology.predecessors(layer)
                    if torch.any(activations[neighbor] != 0)
                ]
                
                # Update active edges
                self._update_active_edges(layer, active_predecessors)
                
                # Validate runtime edges
                is_valid, error_msg = self._validate_runtime_edges()
                if not is_valid:
                    raise ValueError(f"Runtime topology violation: {error_msg}")
                
                # Sum weighted inputs from predecessors
                bias = self.node_states[layer]['bias']
                weighted_sum = torch.full((batch_size,), bias, dtype=torch.float32, device=device)
                for neighbor in self.topology.predecessors(layer):
                    weight = torch.tensor(self.node_states[layer]['weights'][neighbor], dtype=torch.float32, device=activations[neighbor].device)
                    weighted_sum += activations[neighbor] * weight
                # Apply activation function (LeakyReLU)
                activations[layer] = torch.nn.LeakyReLU(0.1)(weighted_sum)
        
        # Return output node activations
        return {node: activations[node] for node in self.output_nodes}
    
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
            ])
        } 