import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'hybrid')
class HybridTopology(BaseTopology, BasePlugin):
    def __init__(self, size: int, num_modules: int, k: int, p: float, 
                 inter_module_prob: float, seed: int = None):
        """
        Initialize a hybrid topology that combines small-world and modular properties.
        
        Args:
            size: Total number of nodes in the network
            num_modules: Number of modules in the network
            k: Number of nearest neighbors for small-world connections within modules
            p: Probability of rewiring for small-world connections
            inter_module_prob: Probability of connections between modules
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.num_modules = num_modules
        self.k = k
        self.p = p
        self.inter_module_prob = inter_module_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate module sizes
        self.module_size = size // num_modules
        self.extra_nodes = size % num_modules
        
        # Initialize module assignments
        self.module_assignments = self._assign_modules()
    
    def _assign_modules(self) -> Dict[int, int]:
        """Assign nodes to modules."""
        assignments = {}
        current_node = 0
        
        for module in range(self.num_modules):
            # Add extra node to first few modules if needed
            module_size = self.module_size + (1 if module < self.extra_nodes else 0)
            
            for _ in range(module_size):
                assignments[current_node] = module
                current_node += 1
        
        return assignments
    
    def _create_module_graph(self, module_nodes: List[int]) -> nx.DiGraph:
        """Create a small-world graph for a single module."""
        G = nx.DiGraph()
        G.add_nodes_from(module_nodes)
        
        # Sort nodes to ensure acyclicity
        module_nodes.sort()
        
        # Add edges to create a ring lattice (directed, acyclic)
        for i in range(len(module_nodes)):
            for j in range(1, self.k // 2 + 1):
                target_idx = (i + j) % len(module_nodes)
                if target_idx > i:  # Only add forward edges
                    G.add_edge(module_nodes[i], module_nodes[target_idx])
        
        # Rewire edges with probability p (maintaining acyclicity)
        for edge in list(G.edges()):
            if self.rng.random() < self.p:
                # Remove the edge
                G.remove_edge(*edge)
                # Add a new random edge (only to higher-indexed nodes)
                new_node = self.rng.choice([n for n in module_nodes if n > edge[0]])
                while G.has_edge(edge[0], new_node):
                    new_node = self.rng.choice([n for n in module_nodes if n > edge[0]])
                G.add_edge(edge[0], new_node)
        
        return G
    
    def generate(self, num_layers: int = 1, input_dim: int = None, output_dim: int = None) -> Union[nx.Graph, List[nx.Graph]]:
        """
        Generate the hybrid network topology as a single connected graph.
        
        Args:
            num_layers: Number of layers (ignored for hybrid)
            input_dim: Number of input nodes (if provided, extends graph)
            output_dim: Number of output nodes (if provided, extends graph)
        """
        # Calculate total nodes needed
        if input_dim is not None and output_dim is not None:
            total_nodes = input_dim + self.size + output_dim
        else:
            total_nodes = self.size
        
        G = nx.DiGraph()
        G.add_nodes_from(range(total_nodes))
        
        # Create hybrid connections only among hidden nodes
        hidden_start = input_dim if input_dim is not None else 0
        hidden_end = hidden_start + self.size
        
        # Create small-world graphs for each module (directed, acyclic)
        for module in range(self.num_modules):
            module_nodes = [node for node, mod in self.module_assignments.items() if mod == module]
            # Map module nodes to actual graph indices
            actual_module_nodes = [hidden_start + node for node in module_nodes]
            module_graph = self._create_module_graph(actual_module_nodes)
            G.add_edges_from(module_graph.edges())
        
        # Add inter-module connections (directed, acyclic)
        for module1 in range(self.num_modules):
            for module2 in range(self.num_modules):
                if module1 != module2:
                    module1_nodes = [node for node, mod in self.module_assignments.items() if mod == module1]
                    module2_nodes = [node for node, mod in self.module_assignments.items() if mod == module2]
                    # Map to actual graph indices
                    actual_module1_nodes = [hidden_start + node for node in module1_nodes]
                    actual_module2_nodes = [hidden_start + node for node in module2_nodes]
                    # Sort nodes to ensure acyclicity
                    actual_module1_nodes.sort()
                    actual_module2_nodes.sort()
                    for node1 in actual_module1_nodes:
                        for node2 in actual_module2_nodes:
                            if node1 < node2 and self.rng.random() < self.inter_module_prob:
                                G.add_edge(node1, node2)
        
        # Add connections from input nodes to hidden nodes
        if input_dim is not None:
            for input_node in range(input_dim):
                for hidden_node in range(hidden_start, hidden_start + min(4, self.size)):  # Connect to first few hidden nodes
                    G.add_edge(input_node, hidden_node)
        
        # Add connections from hidden nodes to output nodes
        if output_dim is not None:
            for output_node in range(hidden_end, total_nodes):
                for hidden_node in range(hidden_end - min(4, self.size), hidden_end):  # Connect from last few hidden nodes
                    G.add_edge(hidden_node, output_node)
        
        return G
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'num_modules': self.num_modules,
            'k': self.k,
            'p': self.p,
            'inter_module_prob': self.inter_module_prob,
            'seed': self.seed
        }
    
    def get_module_assignments(self) -> Dict[int, int]:
        """Get the module assignments for each node."""
        return self.module_assignments
    
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