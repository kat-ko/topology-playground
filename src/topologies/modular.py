import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'modular')
class ModularTopology(BaseTopology, BasePlugin):
    def __init__(self, size: int, num_modules: int, inter_module_prob: float,
                 intra_module_prob: float, seed: int = None):
        """
        Initialize a modular network topology.
        
        Args:
            size: Total number of nodes in the network
            num_modules: Number of modules in the network
            inter_module_prob: Probability of connections between modules
            intra_module_prob: Probability of connections within modules
            seed: Random seed for reproducibility
        """
        super().__init__(n_in=0, n_hidden=size, n_out=0)  # Initialize base class
        self.size = size
        self.num_modules = num_modules
        self.inter_module_prob = inter_module_prob
        self.intra_module_prob = intra_module_prob
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
    
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """Generate the modular network topology as a single connected graph."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Add intra-module connections (directed, acyclic)
        for module in range(self.num_modules):
            module_nodes = [node for node, mod in self.module_assignments.items() if mod == module]
            # Sort nodes to ensure acyclicity
            module_nodes.sort()
            for i in range(len(module_nodes)):
                for j in range(i + 1, len(module_nodes)):
                    if self.rng.random() < self.intra_module_prob:
                        G.add_edge(module_nodes[i], module_nodes[j])
        
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
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters."""
        return {
            'size': self.size,
            'num_modules': self.num_modules,
            'inter_module_prob': self.inter_module_prob,
            'intra_module_prob': self.intra_module_prob,
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
    
    def get_node_distances(self, graph: nx.Graph) -> np.ndarray:
        """Calculate pairwise shortest path distances between all nodes."""
        return nx.floyd_warshall_numpy(graph)
    
    def get_centrality_measures(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate degree and betweenness centrality for all nodes."""
        degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
        betweenness_centrality = np.array(list(nx.betweenness_centrality(graph).values()))
        return degree_centrality, betweenness_centrality 
    
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