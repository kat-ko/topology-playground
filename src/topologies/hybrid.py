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
                 inter_module_prob: float, num_layers: int = 1,
                 inter_layer_prob: float = 0.1, seed: int = None):
        """
        Initialize a hybrid topology that combines small-world and modular properties.
        
        Args:
            size: Total number of nodes in the network
            num_modules: Number of modules in the network
            k: Number of nearest neighbors for small-world connections within modules
            p: Probability of rewiring for small-world connections
            inter_module_prob: Probability of connections between modules
            num_layers: Number of layers in the network (default: 1)
            inter_layer_prob: Probability of connections between layers (default: 0.1)
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.num_modules = num_modules
        self.k = k
        self.p = p
        self.inter_module_prob = inter_module_prob
        self.num_layers = num_layers
        self.inter_layer_prob = inter_layer_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Calculate module sizes
        self.module_size = size // num_modules
        self.extra_nodes = size % num_modules
        
        # Initialize module assignments
        self.module_assignments = self._assign_modules()
        
        # Store layer graphs and inter-layer connections
        self.layers: List[nx.DiGraph] = []
        self.inter_layer_connections: Dict[tuple, nx.DiGraph] = {}
    
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
    
    def _create_layer(self, layer_idx: int) -> nx.DiGraph:
        """Create a single layer of the network as a directed acyclic graph."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Create small-world graphs for each module (directed, acyclic)
        for module in range(self.num_modules):
            module_nodes = [node for node, mod in self.module_assignments.items() if mod == module]
            module_graph = self._create_module_graph(module_nodes)
            G.add_edges_from(module_graph.edges())
        
        # Add inter-module connections (directed, acyclic)
        for module1 in range(self.num_modules):
            for module2 in range(self.num_modules):
                if module1 != module2:
                    module1_nodes = [node for node, mod in self.module_assignments.items() if mod == module1]
                    module2_nodes = [node for node, mod in self.module_assignments.items() if mod == module2]
                    # Sort nodes to ensure acyclicity
                    module1_nodes.sort()
                    module2_nodes.sort()
                    for node1 in module1_nodes:
                        for node2 in module2_nodes:
                            if node1 < node2 and self.rng.random() < self.inter_module_prob:
                                G.add_edge(node1, node2)
        
        return G
    
    def _create_inter_layer_connections(self, layer1: int, layer2: int) -> nx.DiGraph:
        """Create directed acyclic connections between two layers."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # For inter-layer connections in a feedforward network,
        # layer1 nodes should only connect to layer2 nodes
        if layer1 < layer2:
            for node1 in range(self.size):
                for node2 in range(self.size):
                    if self.rng.random() < self.inter_layer_prob:
                        G.add_edge(node1, node2)
        else:
            for node2 in range(self.size):
                for node1 in range(self.size):
                    if self.rng.random() < self.inter_layer_prob:
                        G.add_edge(node2, node1)
        
        return G
    
    def generate(self, num_layers: int = 1) -> Union[nx.DiGraph, List[nx.DiGraph]]:
        """Generate the hybrid network topology."""
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
            'num_modules': self.num_modules,
            'k': self.k,
            'p': self.p,
            'inter_module_prob': self.inter_module_prob,
            'num_layers': self.num_layers,
            'inter_layer_prob': self.inter_layer_prob,
            'seed': self.seed
        }
    
    def get_module_assignments(self) -> Dict[int, int]:
        """Get the module assignments for each node."""
        return self.module_assignments
    
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.DiGraph]:
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