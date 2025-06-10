import networkx as nx
import numpy as np
from typing import List, Union, Dict, Any

class FullyConnectedTopology:
    """Fully connected network topology generator."""
    
    def __init__(self, size: int, num_layers: int = 1,
                 inter_layer_prob: float = 1.0,
                 intra_layer_prob: float = 1.0,
                 seed: int = 42):
        """
        Initialize fully connected topology generator.
        
        Args:
            size: Number of nodes in each layer
            num_layers: Number of layers in the network
            inter_layer_prob: Probability of connections between layers (1.0 for fully connected)
            intra_layer_prob: Probability of connections within layers (1.0 for fully connected)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.num_layers = num_layers
        self.inter_layer_prob = inter_layer_prob
        self.intra_layer_prob = intra_layer_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def generate(self, num_layers: int = None) -> Union[nx.Graph, List[nx.Graph]]:
        """
        Generate fully connected network topology.
        
        Args:
            num_layers: Number of layers to generate (overrides initialization)
            
        Returns:
            Single graph for single layer, list of graphs for multiple layers
        """
        if num_layers is None:
            num_layers = self.num_layers
        
        if num_layers == 1:
            return self._generate_single_layer()
        else:
            return [self._generate_single_layer() for _ in range(num_layers)]
    
    def _generate_single_layer(self) -> nx.DiGraph:
        """Generate a single fully connected layer."""
        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(range(self.size))
        
        # Add edges (fully connected, directed, acyclic)
        for i in range(self.size):
            for j in range(i + 1, self.size):  # Only connect to higher indices
                if self.rng.random() < self.intra_layer_prob:
                    G.add_edge(i, j)  # Forward edge only
        
        return G
    
    def get_parameter_count(self) -> int:
        """Calculate the number of parameters in the fully connected network."""
        # For a single layer, it's the number of edges
        single_layer_params = (self.size * (self.size - 1)) // 2
        
        # For multiple layers, add inter-layer connections
        if self.num_layers > 1:
            inter_layer_params = self.size * self.size * (self.num_layers - 1)
            return single_layer_params * self.num_layers + inter_layer_params
        else:
            return single_layer_params
    
    def get_network_metrics(self) -> Dict[str, float]:
        """Calculate network metrics."""
        G = self._generate_single_layer()
        
        return {
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'average_shortest_path_length': nx.average_shortest_path_length(G),
            'parameter_count': self.get_parameter_count()
        } 