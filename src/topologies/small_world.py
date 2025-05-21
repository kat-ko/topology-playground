import networkx as nx
import numpy as np
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
    
    def _create_layer(self, layer_idx: int) -> nx.Graph:
        """Create a single layer of the network."""
        # Create a ring lattice
        G = nx.Graph()
        G.add_nodes_from(range(self.size))
        
        # Add edges to create a ring lattice
        for i in range(self.size):
            for j in range(1, self.k // 2 + 1):
                # Add edge to the right
                G.add_edge(i, (i + j) % self.size)
                # Add edge to the left
                G.add_edge(i, (i - j) % self.size)
        
        # Rewire edges with probability p
        for edge in list(G.edges()):
            if self.rng.random() < self.p:
                # Remove the edge
                G.remove_edge(*edge)
                # Add a new random edge
                new_node = self.rng.randint(0, self.size)
                while new_node == edge[0] or G.has_edge(edge[0], new_node):
                    new_node = self.rng.randint(0, self.size)
                G.add_edge(edge[0], new_node)
        
        return G
    
    def _create_inter_layer_connections(self, layer1: int, layer2: int) -> nx.Graph:
        """Create connections between two layers."""
        G = nx.Graph()
        
        # Add nodes from both layers
        G.add_nodes_from(range(self.size))
        
        # Add random inter-layer connections
        for node1 in range(self.size):
            for node2 in range(self.size):
                if self.rng.random() < self.inter_layer_prob:
                    G.add_edge(node1, node2)
        
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
        return {
            'clustering_coefficient': nx.average_clustering(G),
            'density': nx.density(G),
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'diameter': nx.diameter(G),
            'avg_shortest_path': nx.average_shortest_path_length(G)
        } 