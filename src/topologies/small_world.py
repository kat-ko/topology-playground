import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'small_world')
class SmallWorldTopology(BaseTopology, BasePlugin):
    def __init__(self, size: int, k: int, p: float, seed: int = None):
        """
        Initialize a small-world network topology.
        
        Args:
            size: Number of nodes in the network
            k: Number of nearest neighbors for each node
            p: Probability of rewiring
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.k = k
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """Generate the small-world network topology as a single connected graph."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Create initial ring lattice structure (directed, acyclic)
        for i in range(self.size):
            # Only add edges to higher-indexed nodes to maintain acyclicity
            for j in range(1, self.k // 2 + 1):
                target = (i + j) % self.size
                if target > i:  # Only add forward edges
                    G.add_edge(i, target)
        
        # Rewire edges with probability p (maintaining acyclicity)
        for edge in list(G.edges()):
            if self.rng.random() < self.p:
                # Remove the edge
                G.remove_edge(*edge)
                # Add a new random edge (only to higher-indexed nodes)
                new_node = self.rng.randint(edge[0] + 1, self.size)
                while G.has_edge(edge[0], new_node):
                    new_node = self.rng.randint(edge[0] + 1, self.size)
                G.add_edge(edge[0], new_node)
        
        return G
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'k': self.k,
            'p': self.p,
            'seed': self.seed
        }
    
    def get_network_metrics(self) -> Dict[str, Any]:
        """Get metrics for the entire network."""
        G = self.generate()
        
        # Convert to undirected for metrics that don't support directed graphs
        G_undirected = G.to_undirected() if G.is_directed() else G
        
        return {
            'clustering_coefficient': nx.average_clustering(G_undirected),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'diameter': nx.diameter(G_undirected),
            'avg_shortest_path': nx.average_shortest_path_length(G_undirected)
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
    
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.Graph]:
        """Get the inter-layer connections between two layers.
        
        Note: This topology doesn't use layers, so this method returns None.
        """
        return None
    
    def get_layer_metrics(self, layer: int) -> Dict[str, Any]:
        """Get metrics specific to a particular layer.
        
        Note: This topology doesn't use layers, so this method returns network-wide metrics.
        """
        return self.get_network_metrics() 