import networkx as nx
import numpy as np
from typing import Tuple, List, Dict

class ModularTopology:
    def __init__(self, size: int, num_modules: int, 
                 inter_module_prob: float = 0.1,
                 intra_module_prob: float = 0.3,
                 seed: int = None):
        self.size = size
        self.num_modules = num_modules
        self.module_size = size // num_modules
        self.inter_module_prob = inter_module_prob
        self.intra_module_prob = intra_module_prob
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def generate(self) -> nx.Graph:
        """Generate a modular network with specified parameters."""
        graph = nx.Graph()
        
        # Add all nodes
        graph.add_nodes_from(range(self.size))
        
        # Assign nodes to modules
        module_assignments = {}
        for i in range(self.size):
            module_assignments[i] = i // self.module_size
        
        # Create edges within modules
        for module in range(self.num_modules):
            module_nodes = [n for n, m in module_assignments.items() if m == module]
            for i in module_nodes:
                for j in module_nodes:
                    if i < j and self.rng.random() < self.intra_module_prob:
                        graph.add_edge(i, j)
        
        # Create edges between modules
        for i in range(self.size):
            for j in range(i + 1, self.size):
                if module_assignments[i] != module_assignments[j]:
                    if self.rng.random() < self.inter_module_prob:
                        graph.add_edge(i, j)
        
        return graph
    
    def get_module_assignments(self) -> Dict[int, int]:
        """Get the module assignment for each node."""
        return {i: i // self.module_size for i in range(self.size)}
    
    def get_node_distances(self, graph: nx.Graph) -> np.ndarray:
        """Calculate pairwise shortest path distances between all nodes."""
        return nx.floyd_warshall_numpy(graph)
    
    def get_centrality_measures(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate degree and betweenness centrality for all nodes."""
        degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
        betweenness_centrality = np.array(list(nx.betweenness_centrality(graph).values()))
        return degree_centrality, betweenness_centrality 