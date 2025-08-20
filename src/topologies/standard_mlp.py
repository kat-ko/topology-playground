import networkx as nx
import numpy as np
import torch
from typing import List, Union, Dict, Any, Optional
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'standard_mlp')
class StandardMLPTopology(BaseTopology, BasePlugin):
    """Standard MLP topology generator for baseline comparison."""
    
    def __init__(self, size: int, num_layers: int = 1, activation: str = 'relu', seed: int = 42):
        """
        Initialize standard MLP topology generator.
        
        Args:
            size: Number of hidden nodes per layer
            num_layers: Number of hidden layers (MLP supports multiple layers)
            activation: Activation function to use
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.num_layers = num_layers
        self.activation = activation
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate total network size
        self.total_nodes = self._calculate_total_nodes()
    
    def _calculate_total_nodes(self) -> int:
        """Calculate total number of nodes in the MLP."""
        # Input layer: observation space dimension (will be set dynamically)
        # Hidden layers: size * num_layers
        # Output layer: action space dimension (will be set dynamically)
        hidden_nodes = self.size * self.num_layers
        return hidden_nodes
    
    def generate(self, num_layers: int = 1, input_dim: int = None, output_dim: int = None) -> Union[nx.Graph, List[nx.Graph]]:
        """
        Generate standard MLP network topology as a unified graph.
        
        Args:
            num_layers: Number of layers to generate (default: 1)
            input_dim: Number of input nodes (if provided, extends graph)
            output_dim: Number of output nodes (if provided, extends graph)
            
        Returns:
            Single unified graph representing MLP architecture
        """
        # Always use the provided num_layers parameter for consistency
        hidden_nodes = self.size * num_layers
        
        # Calculate total nodes needed
        if input_dim is not None and output_dim is not None:
            total_nodes = input_dim + hidden_nodes + output_dim
        else:
            total_nodes = hidden_nodes
        
        # Create empty directed graph
        G = nx.DiGraph()
        
        # Add all nodes
        G.add_nodes_from(range(total_nodes))
        
        # Add connections representing MLP architecture
        # Each layer is fully connected to the next layer
        
        # Calculate layer boundaries
        input_start = 0
        input_end = input_dim if input_dim is not None else 0
        hidden_start = input_end
        hidden_end = hidden_start + hidden_nodes
        output_start = hidden_end
        output_end = total_nodes
        
        # Connect input layer to first hidden layer
        if input_dim is not None:
            for i in range(input_start, input_end):
                for j in range(hidden_start, hidden_start + self.size):
                    G.add_edge(i, j)
        
        # Connect hidden layers to each other
        for layer_idx in range(num_layers):
            # Calculate start and end indices for this hidden layer
            layer_start = hidden_start + (layer_idx * self.size)
            layer_end = layer_start + self.size
            
            # Add connections to next hidden layer (if not the last hidden layer)
            if layer_idx < num_layers - 1:
                next_layer_start = hidden_start + ((layer_idx + 1) * self.size)
                next_layer_end = next_layer_start + self.size
                
                # Connect all nodes from current layer to next layer
                for i in range(layer_start, layer_end):
                    for j in range(next_layer_start, next_layer_end):
                        G.add_edge(i, j)  # Forward edge only
        
        # Connect last hidden layer to output layer
        if output_dim is not None:
            last_hidden_start = hidden_end - self.size
            for i in range(last_hidden_start, hidden_end):
                for j in range(output_start, output_end):
                    G.add_edge(i, j)
        
        return G
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'seed': self.seed
        }
    
    def get_layer_info(self) -> Dict[str, Any]:
        """Get information about layer structure."""
        return {
            'num_layers': self.num_layers,
            'hidden_size': self.size,
            'total_hidden_nodes': self.total_nodes,
            'activation': self.activation
        }
    
    def get_parameter_count(self) -> int:
        """Calculate the number of parameters in the standard MLP."""
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
            'total_nodes': len(G.nodes()),
            'total_edges': len(G.edges())
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
        layer1_start = layer1 * self.size
        layer1_end = layer1_start + self.size
        layer2_start = layer2 * self.size
        layer2_end = layer2_start + self.size
        
        layer1_nodes = list(range(layer1_start, layer1_end))
        layer2_nodes = list(range(layer2_start, layer2_end))
        
        G.add_nodes_from(layer1_nodes)
        G.add_nodes_from(layer2_nodes)
        
        # Add connections between layers (if they are consecutive)
        if abs(layer2 - layer1) == 1:
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
        
        layer_start = layer * self.size
        layer_end = layer_start + self.size
        
        return {
            'layer_index': layer,
            'start_node': layer_start,
            'end_node': layer_end,
            'node_count': self.size,
            'is_input_layer': layer == 0,
            'is_output_layer': layer == self.num_layers - 1
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
