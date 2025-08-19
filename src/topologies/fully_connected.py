import networkx as nx
import numpy as np
import torch
from typing import List, Union, Dict, Any, Optional
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'fully_connected')
class FullyConnectedTopology(BaseTopology, BasePlugin):
    """Fully connected network topology generator."""
    
    def __init__(self, size: int, seed: int = 42):
        """
        Initialize fully connected topology generator.
        
        Args:
            size: Total number of nodes in the network
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """
        Generate fully connected network topology as a single complete graph.
        
        Args:
            num_layers: Ignored for fully connected topology (always creates single graph)
            
        Returns:
            Single complete graph where every node connects to every other node
        """
        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add all nodes
        G.add_nodes_from(range(self.size))
        
        # Add connections: every node connects to every other node
        for i in range(self.size):
            for j in range(self.size):
                if i != j:  # Don't connect node to itself
                    G.add_edge(i, j)
        
        return G
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'seed': self.seed
        }
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about layer structure."""
        return {
            'num_layers': 1,  # Always single layer for fully connected
            'total_nodes': self.size,
            'connection_type': 'complete'
        }
    
    def get_parameter_count(self) -> int:
        """Calculate the number of parameters in the fully connected network."""
        # This is a placeholder - actual parameter counting will be done
        # by FeedForwardNetwork when it creates the real network
        return 0
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Calculate network metrics."""
        G = self.generate()
        
        # Convert to undirected for metrics that don't support directed graphs
        G_undirected = G.to_undirected() if G.is_directed() else G
        
        return {
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G_undirected),
            'average_shortest_path_length': nx.average_shortest_path_length(G_undirected),
            'diameter': nx.diameter(G_undirected),
            'parameter_count': self.get_parameter_count()
        }
    
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.Graph]:
        """Get the inter-layer connections between two layers.
        
        Args:
            layer1: Index of first layer (ignored for fully connected)
            layer2: Index of second layer (ignored for fully connected)
            
        Returns:
            networkx.Graph: Complete graph (fully connected has no layer concept)
        """
        # Fully connected topology has no layers - return the complete graph
        return self.generate()
    
    def get_layer_metrics(self, layer: int) -> Dict[str, Any]:
        """Get metrics specific to a particular layer.
        
        Args:
            layer: Index of the layer (ignored for fully connected)
            
        Returns:
            Dict[str, Any]: Dictionary of layer-specific metrics
        """
        # Fully connected topology has no layers
        return {
            'layer_index': 0,  # Always single layer
            'start_node': 0,
            'end_node': self.size,
            'node_count': self.size,
            'is_input_layer': True,  # All nodes are both input and output
            'is_output_layer': True
        }
    
    def generate_adjacency_mask(self) -> torch.Tensor:
        """
        Generate the adjacency mask for the network.
        
        Returns:
            Binary adjacency mask tensor
        """
        # Generate the network
        G = self.generate()
        
        # Create adjacency matrix
        adj_matrix = nx.to_numpy_array(G)
        
        # Convert to PyTorch tensor
        mask = torch.from_numpy(adj_matrix).float()
        
        # Validate the mask
        is_valid, error_msg = self.validate_mask(mask)
        if not is_valid:
            raise ValueError(error_msg)
        
        return mask 