from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import networkx as nx
import numpy as np
from ..topologies.utils import prune_output_edges

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
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.network_params = network_params
        self.num_nodes = len(topology.nodes())
        
        # Store original topology for metrics
        self.original_topology = topology
        
        # Prune output-output edges before initializing the network
        self.topology = prune_output_edges(topology, output_nodes)
        
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
        # Calculate basic metrics that don't require connectivity
        metrics = {
            'num_nodes': self.num_nodes,
            'num_edges': self.topology.number_of_edges(),
            'density': nx.density(self.topology),
            'avg_degree': sum(dict(self.topology.degree()).values()) / self.num_nodes,
            'num_output_edges': sum(1 for edge in self.topology.edges() 
                                  if edge[0] in self.output_nodes or edge[1] in self.output_nodes)
        }
        
        # Add pruning metrics
        original_edges = self.original_topology.number_of_edges()
        pruned_edges = original_edges - sum(1 for edge in self.original_topology.edges()
                                          if edge[0] in self.output_nodes and edge[1] in self.output_nodes)
        metrics.update({
            'original_edges': original_edges,
            'pruned_edges': pruned_edges,
            'edges_removed': original_edges - pruned_edges
        })
        
        # Calculate connectivity-dependent metrics only if graph is connected
        if nx.is_connected(self.topology):
            metrics.update({
                'diameter': nx.diameter(self.topology),
                'avg_shortest_path': nx.average_shortest_path_length(self.topology)
            })
        else:
            # Calculate metrics for largest connected component
            largest_cc = max(nx.connected_components(self.topology), key=len)
            largest_cc_graph = self.topology.subgraph(largest_cc)
            metrics.update({
                'diameter': nx.diameter(largest_cc_graph),
                'avg_shortest_path': nx.average_shortest_path_length(largest_cc_graph),
                'num_connected_components': nx.number_connected_components(self.topology),
                'largest_component_size': len(largest_cc)
            })
        
        return metrics
    
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