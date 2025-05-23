from typing import Dict, Any, List
import networkx as nx
import numpy as np
from .base import BaseNetwork

class FeedForwardNetwork(BaseNetwork):
    """FeedForward Network implementation."""
    
    def _violates_feed_forward(self, u: int, v: int) -> bool:
        """Check if an edge violates feed-forward constraints.
        
        Args:
            u: Source node
            v: Target node
            
        Returns:
            bool: True if the edge violates feed-forward constraints
        """
        # Input nodes can only have outgoing edges
        if u in self.input_nodes and v in self.input_nodes:
            return True
        
        # Output nodes can only have incoming edges
        if u in self.output_nodes and v in self.output_nodes:
            return True
        
        # Hidden nodes: ensure acyclic structure
        if u not in self.input_nodes and v not in self.output_nodes:
            return u > v  # Only allow edges from lower to higher indices
        
        return False
    
    def _create_processing_graph(self) -> nx.DiGraph:
        """Create a directed processing graph for FFN that respects topology constraints."""
        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from(self.topology.nodes())
        
        # Phase 1: Create all edges from topology (both directions)
        for u, v in self.topology.edges():
            G.add_edge(u, v)
            G.add_edge(v, u)
        
        # Phase 2: Remove edges that violate feed-forward constraints
        for u, v in list(G.edges()):
            if self._violates_feed_forward(u, v):
                G.remove_edge(u, v)
        
        return G
    
    def _initialize_node_states(self) -> Dict[str, Any]:
        """Initialize node states for FFN."""
        states = {}
        for node in self.topology.nodes():
            states[node] = {
                'activation': 0.0,
                'bias': np.random.normal(0, 0.1),
                'weights': {
                    neighbor: np.random.normal(0, 0.1)
                    for neighbor in self.processing_graph.predecessors(node)
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
        # Initialize activations
        activations = {node: 0.0 for node in self.topology.nodes()}
        
        # Set input node activations
        for node, value in inputs.items():
            if node in self.input_nodes:
                activations[node] = value
        
        # Process through network layers using topological sort
        for layer in nx.topological_sort(self.processing_graph):
            if layer not in self.input_nodes:
                # Sum weighted inputs
                weighted_sum = self.node_states[layer]['bias']
                for neighbor in self.processing_graph.predecessors(layer):
                    weighted_sum += (
                        activations[neighbor] * 
                        self.node_states[layer]['weights'][neighbor]
                    )
                # Apply activation function (ReLU)
                activations[layer] = max(0, weighted_sum)
        
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