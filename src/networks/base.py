from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import networkx as nx
import numpy as np

class BaseNetwork(ABC):
    """Abstract base class for network types (FFN, RNN)."""
    
    def __init__(self, topology: nx.Graph, input_nodes: List[int], output_nodes: List[int],
                 network_params: Dict[str, Any]):
        """
        Initialize the network.
        
        Args:
            topology: NetworkX graph representing the network topology
            input_nodes: List of input node indices
            output_nodes: List of output node indices
            network_params: Dictionary of network-specific parameters
        """
        self.topology = topology
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.network_params = network_params
        self.num_nodes = len(topology.nodes())
        
        # Initialize node states
        self.node_states = self._initialize_node_states()
    
    @abstractmethod
    def _initialize_node_states(self) -> Dict[str, Any]:
        """Initialize the states of all nodes in the network."""
        pass
    
    @abstractmethod
    def forward(self, inputs: Dict[int, Any]) -> Dict[int, Any]:
        """Process inputs through the network.
        
        Args:
            inputs: Dictionary mapping input node indices to their input values
            
        Returns:
            Dictionary mapping output node indices to their output values
        """
        pass
    
    @abstractmethod
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network-specific metrics."""
        pass
    
    def get_topology_metrics(self) -> Dict[str, Any]:
        """Get metrics about the network topology."""
        return {
            'num_nodes': self.num_nodes,
            'num_edges': self.topology.number_of_edges(),
            'density': nx.density(self.topology),
            'avg_degree': sum(dict(self.topology.degree()).values()) / self.num_nodes,
            'diameter': nx.diameter(self.topology),
            'avg_shortest_path': nx.average_shortest_path_length(self.topology)
        }
    
    def get_node_metrics(self, node_idx: int) -> Dict[str, Any]:
        """Get metrics for a specific node."""
        metrics = {
            'degree': self.topology.degree(node_idx),
            'betweenness_centrality': nx.betweenness_centrality(self.topology)[node_idx],
            'closeness_centrality': nx.closeness_centrality(self.topology)[node_idx],
            'pagerank': nx.pagerank(self.topology)[node_idx]
        }
        
        # Try to calculate eigenvector centrality with increased max_iter
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
                self.topology,
                max_iter=1000,  # Increase max iterations
                tol=1e-6  # Adjust tolerance
            )[node_idx]
        except nx.PowerIterationFailedConvergence:
            # If still fails, use a fallback metric
            # Calculate a simple centrality based on degree and neighbors' degrees
            neighbors = list(self.topology.neighbors(node_idx))
            if neighbors:
                metrics['eigenvector_centrality'] = np.mean([
                    self.topology.degree(n) for n in neighbors
                ]) / self.num_nodes
            else:
                metrics['eigenvector_centrality'] = 0.0
        
        return metrics 