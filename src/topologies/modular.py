import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from .base import BaseTopology
from ..core.plugin_registry import PluginRegistry
from ..core.base import BasePlugin

@PluginRegistry.register('topologies', 'modular')
class ModularTopology(BaseTopology, BasePlugin):
    def __init__(self, size: int, num_modules: int, inter_module_prob: float,
                 intra_module_prob: float, num_layers: int = 1,
                 inter_layer_prob: float = 0.1, seed: int = None):
        """
        Initialize a modular network topology.
        
        Args:
            size: Total number of nodes in the network
            num_modules: Number of modules in the network
            inter_module_prob: Probability of connections between modules
            intra_module_prob: Probability of connections within modules
            num_layers: Number of layers in the network (default: 1)
            inter_layer_prob: Probability of connections between layers (default: 0.1)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.num_modules = num_modules
        self.inter_module_prob = inter_module_prob
        self.intra_module_prob = intra_module_prob
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
        self.layers: List[nx.Graph] = []
        self.inter_layer_connections: Dict[tuple, nx.Graph] = {}
    
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
    
    def _create_layer(self, layer_idx: int) -> nx.Graph:
        """Create a single layer of the network."""
        G = nx.Graph()
        G.add_nodes_from(range(self.size))
        
        # Add intra-module connections
        for module in range(self.num_modules):
            module_nodes = [node for node, mod in self.module_assignments.items() if mod == module]
            for i in range(len(module_nodes)):
                for j in range(i + 1, len(module_nodes)):
                    if self.rng.random() < self.intra_module_prob:
                        G.add_edge(module_nodes[i], module_nodes[j])
        
        # Add inter-module connections
        for module1 in range(self.num_modules):
            for module2 in range(module1 + 1, self.num_modules):
                module1_nodes = [node for node, mod in self.module_assignments.items() if mod == module1]
                module2_nodes = [node for node, mod in self.module_assignments.items() if mod == module2]
                
                for node1 in module1_nodes:
                    for node2 in module2_nodes:
                        if self.rng.random() < self.inter_module_prob:
                            G.add_edge(node1, node2)
        
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
        """Generate the modular network topology."""
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
            'inter_module_prob': self.inter_module_prob,
            'intra_module_prob': self.intra_module_prob,
            'num_layers': self.num_layers,
            'inter_layer_prob': self.inter_layer_prob,
            'seed': self.seed
        }
    
    def get_module_assignments(self) -> Dict[int, int]:
        """Get the module assignments for each node."""
        return self.module_assignments
    
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
    
    def get_node_distances(self, graph: nx.Graph) -> np.ndarray:
        """Calculate pairwise shortest path distances between all nodes."""
        return nx.floyd_warshall_numpy(graph)
    
    def get_centrality_measures(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate degree and betweenness centrality for all nodes."""
        degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
        betweenness_centrality = np.array(list(nx.betweenness_centrality(graph).values()))
        return degree_centrality, betweenness_centrality 