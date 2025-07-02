import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'small_world')
class SmallWorldTopology(BaseTopology, BasePlugin):
    def __init__(self, size: int, k: int, p: float, num_layers: int = 1,
                 inter_layer_prob: float = 0.1, seed: int = None):
        """
        Initialize a small-world network topology.
        
        Args:
            size: Number of nodes in the network
            k: Number of nearest neighbors for each node
            p: Probability of rewiring
            num_layers: Number of layers in the network (default: 1)
            inter_layer_prob: Probability of connections between layers (default: 0.1)
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.k = k
        self.p = p
        self.num_layers = num_layers
        self.inter_layer_prob = inter_layer_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Store layer graphs and inter-layer connections
        self.layers: List[nx.Graph] = []
        self.inter_layer_connections: Dict[tuple, nx.Graph] = {}
    
    def _create_layer(self, layer_idx: int) -> nx.DiGraph:
        """Create a single layer of the network as a directed acyclic graph."""
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
    
    def _create_inter_layer_connections(self, layer1: int, layer2: int) -> nx.DiGraph:
        """Create directed acyclic connections between two layers."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # For inter-layer connections, we can allow more flexibility
        # since they're between different layers
        for node1 in range(self.size):
            for node2 in range(self.size):
                if self.rng.random() < self.inter_layer_prob:
                    # In a feedforward network, layer1 nodes should only connect to layer2 nodes
                    if layer1 < layer2:
                        G.add_edge(node1, node2)
                    else:
                        G.add_edge(node2, node1)
        
        return G
    
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """Generate the small-world network topology."""
        self.num_layers = num_layers
        self.layers = []
        self.inter_layer_connections = {}
        
        # Generate each layer
        for i in range(num_layers):
            self.layers.append(self._create_layer(i))
        
        # Generate inter-layer connections
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                self.inter_layer_connections[(i, j)] = self._create_inter_layer_connections(i, j)
        
        # Return single graph or list of graphs based on num_layers
        if num_layers == 1:
            return self.layers[0]
        return self.layers
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'k': self.k,
            'p': self.p,
            'num_layers': self.num_layers,
            'inter_layer_prob': self.inter_layer_prob,
            'seed': self.seed
        }
    
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.Graph]:
        """Get the inter-layer connections between two layers."""
        if layer1 > layer2:
            layer1, layer2 = layer2, layer1
        return self.inter_layer_connections.get((layer1, layer2))
    
    def get_layer_metrics(self, layer: int) -> Dict[str, Any]:
        """Get metrics specific to a particular layer."""
        if layer >= len(self.layers):
            raise ValueError(f"Layer {layer} does not exist")
        
        G = self.layers[layer]
        
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
        # Generate the network if not already generated
        if not self.layers:
            self.generate(self.num_layers)
        
        # Create adjacency matrix for the first layer
        adj_matrix = nx.to_numpy_array(self.layers[0])
        
        # Add inter-layer connections if multiple layers
        if self.num_layers > 1:
            for i in range(self.num_layers):
                for j in range(i + 1, self.num_layers):
                    inter_layer_adj = nx.to_numpy_array(self.inter_layer_connections[(i, j)])
                    # Add inter-layer connections to the adjacency matrix
                    adj_matrix = np.block([
                        [adj_matrix, inter_layer_adj],
                        [inter_layer_adj.T, np.zeros((self.size, self.size))]
                    ])
        
        # Convert to PyTorch tensor
        mask = torch.from_numpy(adj_matrix).float()
        
        # Validate the mask
        is_valid, error_msg = self.validate_mask(mask)
        if not is_valid:
            raise ValueError(error_msg)
        
        return mask 