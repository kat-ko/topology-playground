import networkx as nx
import numpy as np
from typing import List, Union, Dict, Any

class FullyConnectedTopology:
    """Fully connected network topology generator."""
    
    def __init__(self, size: int, num_layers: int = 1, seed: int = 42):
        """
        Initialize fully connected topology generator.
        
        Args:
            size: Total number of nodes in the network
            num_layers: Number of layers in the network
            seed: Random seed for reproducibility
        """
        self.size = size
        self.num_layers = num_layers
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate node distribution across layers
        self.layer_sizes = self._calculate_layer_sizes()
        self.layer_start_indices = self._calculate_layer_start_indices()
    
    def _calculate_layer_sizes(self) -> List[int]:
        """Calculate how many nodes to put in each layer."""
        if self.num_layers == 1:
            return [self.size]
        
        # Distribute nodes as evenly as possible
        base_size = self.size // self.num_layers
        remainder = self.size % self.num_layers
        
        layer_sizes = []
        for i in range(self.num_layers):
            # Add extra node to first few layers if needed
            layer_size = base_size + (1 if i < remainder else 0)
            layer_sizes.append(layer_size)
        
        return layer_sizes
    
    def _calculate_layer_start_indices(self) -> List[int]:
        """Calculate the starting index for each layer."""
        start_indices = [0]
        for i in range(len(self.layer_sizes) - 1):
            start_indices.append(start_indices[-1] + self.layer_sizes[i])
        return start_indices
    
    def generate(self) -> nx.Graph:
        """
        Generate fully connected network topology as a unified multi-layer graph.
        
        Returns:
            Single unified graph with proper layer structure
        """
        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add all nodes
        G.add_nodes_from(range(self.size))
        
        # Add intra-layer connections (fully connected within each layer)
        for layer_idx in range(self.num_layers):
            start_idx = self.layer_start_indices[layer_idx]
            end_idx = start_idx + self.layer_sizes[layer_idx]
            
            # Connect all nodes within this layer (directed, acyclic)
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    G.add_edge(i, j)  # Forward edge only
        
        # Add inter-layer connections (fully connected between adjacent layers)
        for layer_idx in range(self.num_layers - 1):
            current_layer_start = self.layer_start_indices[layer_idx]
            current_layer_end = current_layer_start + self.layer_sizes[layer_idx]
            next_layer_start = self.layer_start_indices[layer_idx + 1]
            next_layer_end = next_layer_start + self.layer_sizes[layer_idx + 1]
            
            # Connect all nodes from current layer to next layer
            for i in range(current_layer_start, current_layer_end):
                for j in range(next_layer_start, next_layer_end):
                    G.add_edge(i, j)  # Forward edge only
        
        return G
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about layer structure."""
        return {
            'num_layers': self.num_layers,
            'layer_sizes': self.layer_sizes,
            'layer_start_indices': self.layer_start_indices,
            'total_nodes': self.size
        }
    
    def get_parameter_count(self) -> int:
        """Calculate the number of parameters in the fully connected network."""
        total_params = 0
        
        # Intra-layer parameters
        for layer_size in self.layer_sizes:
            # Fully connected within layer: n * (n-1) / 2 edges
            intra_layer_params = (layer_size * (layer_size - 1)) // 2
            total_params += intra_layer_params
        
        # Inter-layer parameters
        for i in range(self.num_layers - 1):
            current_layer_size = self.layer_sizes[i]
            next_layer_size = self.layer_sizes[i + 1]
            # Fully connected between layers: n1 * n2 edges
            inter_layer_params = current_layer_size * next_layer_size
            total_params += inter_layer_params
        
        return total_params
    
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