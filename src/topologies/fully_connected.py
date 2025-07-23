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
    
    def __init__(self, size: int, num_layers: int = 1, seed: int = 42):
        """
        Initialize fully connected topology generator.
        
        Args:
            size: Total number of nodes in the network
            num_layers: Number of layers in the network
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
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
    
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """
        Generate fully connected network topology as a unified multi-layer graph.
        
        Args:
            num_layers: Number of layers to generate (default: 1)
            
        Returns:
            Single unified graph with proper layer structure
        """
        # Use the provided num_layers parameter instead of self.num_layers
        # This makes it compatible with the BaseTopology interface
        if num_layers != self.num_layers:
            # Recalculate layer sizes for the new num_layers
            if num_layers == 1:
                layer_sizes = [self.size]
            else:
                base_size = self.size // num_layers
                remainder = self.size % num_layers
                layer_sizes = []
                for i in range(num_layers):
                    layer_size = base_size + (1 if i < remainder else 0)
                    layer_sizes.append(layer_size)
            
            # Recalculate layer start indices
            start_indices = [0]
            for i in range(len(layer_sizes) - 1):
                start_indices.append(start_indices[-1] + layer_sizes[i])
        else:
            # Use pre-calculated values
            layer_sizes = self.layer_sizes
            start_indices = self.layer_start_indices
        
        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add all nodes
        G.add_nodes_from(range(self.size))
        
        # Add intra-layer connections (fully connected within each layer)
        for layer_idx in range(num_layers):
            start_idx = start_indices[layer_idx]
            end_idx = start_idx + layer_sizes[layer_idx]
            
            # Connect all nodes within this layer (directed, acyclic)
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    G.add_edge(i, j)  # Forward edge only
        
        # Add inter-layer connections (fully connected between adjacent layers)
        for layer_idx in range(num_layers - 1):
            current_layer_start = start_indices[layer_idx]
            current_layer_end = current_layer_start + layer_sizes[layer_idx]
            next_layer_start = start_indices[layer_idx + 1]
            next_layer_end = next_layer_start + layer_sizes[layer_idx + 1]
            
            # Connect all nodes from current layer to next layer
            for i in range(current_layer_start, current_layer_end):
                for j in range(next_layer_start, next_layer_end):
                    G.add_edge(i, j)  # Forward edge only
        
        return G
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'num_layers': self.num_layers,
            'seed': self.seed
        }
    
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
    
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.Graph]:
        """Get the inter-layer connections between two layers.
        
        Args:
            layer1: Index of first layer
            layer2: Index of second layer
            
        Returns:
            networkx.Graph or None: Graph representing inter-layer connections,
                                  or None if layers are not connected
        """
        if layer1 < 0 or layer2 < 0 or layer1 >= self.num_layers or layer2 >= self.num_layers:
            return None
        
        # Create a subgraph with only the connections between the two layers
        G = nx.DiGraph()
        
        # Add nodes from both layers
        layer1_start = self.layer_start_indices[layer1]
        layer1_end = layer1_start + self.layer_sizes[layer1]
        layer2_start = self.layer_start_indices[layer2]
        layer2_end = layer2_start + self.layer_sizes[layer2]
        
        layer1_nodes = list(range(layer1_start, layer1_end))
        layer2_nodes = list(range(layer2_start, layer2_end))
        
        G.add_nodes_from(layer1_nodes)
        G.add_nodes_from(layer2_nodes)
        
        # Add edges between layers (only forward connections)
        if layer1 < layer2:
            for i in layer1_nodes:
                for j in layer2_nodes:
                    G.add_edge(i, j)
        
        return G
    
    def get_layer_metrics(self, layer: int) -> Dict[str, Any]:
        """Get metrics specific to a particular layer.
        
        Args:
            layer: Index of the layer
            
        Returns:
            Dict[str, Any]: Dictionary of layer-specific metrics
        """
        if layer < 0 or layer >= self.num_layers:
            return {}
        
        layer_start = self.layer_start_indices[layer]
        layer_end = layer_start + self.layer_sizes[layer]
        layer_nodes = list(range(layer_start, layer_end))
        
        # Create subgraph for this layer
        G = nx.DiGraph()
        G.add_nodes_from(layer_nodes)
        
        # Add intra-layer connections
        for i in range(len(layer_nodes)):
            for j in range(i + 1, len(layer_nodes)):
                G.add_edge(layer_nodes[i], layer_nodes[j])
        
        # Convert to undirected for metrics
        G_undirected = G.to_undirected()
        
        return {
            'layer_size': self.layer_sizes[layer],
            'layer_start_index': layer_start,
            'layer_end_index': layer_end,
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / len(layer_nodes),
            'clustering_coefficient': nx.average_clustering(G_undirected)
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