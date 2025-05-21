from typing import Dict, Any, List
import networkx as nx
import numpy as np
from .base import BaseNetwork

class RecurrentNetwork(BaseNetwork):
    """Recurrent Neural Network implementation."""
    
    def _initialize_node_states(self) -> Dict[str, Any]:
        """Initialize node states for RNN."""
        states = {}
        for node in self.topology.nodes():
            states[node] = {
                'activation': 0.0,
                'hidden_state': np.zeros(self.network_params.get('hidden_size', 32)),
                'bias': np.random.normal(0, 0.1),
                'input_weights': {
                    neighbor: np.random.normal(0, 0.1)
                    for neighbor in self.topology.predecessors(node)
                },
                'recurrent_weights': {
                    neighbor: np.random.normal(0, 0.1)
                    for neighbor in self.topology.neighbors(node)
                },
                'hidden_weights': np.random.normal(0, 0.1, size=self.network_params.get('hidden_size', 32))
            }
        return states
    
    def forward(self, inputs: Dict[int, Any]) -> Dict[int, Any]:
        """Process inputs through the RNN.
        
        Args:
            inputs: Dictionary mapping input node indices to their input values
            
        Returns:
            Dictionary mapping output node indices to their output values
        """
        # Initialize activations and hidden states
        activations = {node: 0.0 for node in self.topology.nodes()}
        hidden_states = {
            node: self.node_states[node]['hidden_state'].copy()
            for node in self.topology.nodes()
        }
        
        # Set input node activations
        for node, value in inputs.items():
            if node in self.input_nodes:
                activations[node] = value
        
        # Process through network for sequence_length steps
        sequence_length = self.network_params.get('sequence_length', 10)
        for _ in range(sequence_length):
            # Update each node's state
            for node in self.topology.nodes():
                if node not in self.input_nodes:
                    # Sum weighted inputs
                    weighted_sum = self.node_states[node]['bias']
                    
                    # Input connections
                    for neighbor in self.topology.predecessors(node):
                        weighted_sum += (
                            activations[neighbor] * 
                            self.node_states[node]['input_weights'][neighbor]
                        )
                    
                    # Recurrent connections
                    for neighbor in self.topology.neighbors(node):
                        weighted_sum += (
                            hidden_states[neighbor] * 
                            self.node_states[node]['recurrent_weights'][neighbor]
                        )
                    
                    # Update hidden state
                    hidden_states[node] = np.tanh(
                        weighted_sum + 
                        np.dot(hidden_states[node], self.node_states[node]['hidden_weights'])
                    )
                    
                    # Update activation
                    activations[node] = np.tanh(weighted_sum)
        
        # Return output node activations
        return {node: activations[node] for node in self.output_nodes}
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get RNN-specific metrics."""
        hidden_size = self.network_params.get('hidden_size', 32)
        return {
            'num_input_weights': sum(len(node['input_weights']) for node in self.node_states.values()),
            'num_recurrent_weights': sum(len(node['recurrent_weights']) for node in self.node_states.values()),
            'num_hidden_weights': len(self.node_states) * hidden_size,
            'num_biases': len(self.node_states),
            'avg_input_weight_magnitude': np.mean([
                abs(w) for node in self.node_states.values()
                for w in node['input_weights'].values()
            ]),
            'avg_recurrent_weight_magnitude': np.mean([
                abs(w) for node in self.node_states.values()
                for w in node['recurrent_weights'].values()
            ]),
            'avg_hidden_weight_magnitude': np.mean([
                abs(w) for node in self.node_states.values()
                for w in node['hidden_weights']
            ]),
            'avg_bias_magnitude': np.mean([
                abs(node['bias']) for node in self.node_states.values()
            ])
        } 