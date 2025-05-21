import networkx as nx
import numpy as np
from typing import Tuple, List

class SmallWorldTopology:
    def __init__(self, size: int, k: int = 4, p: float = 0.1, seed: int = None):
        self.size = size
        self.k = k
        self.p = p
        self.seed = seed
        
    def generate(self) -> nx.Graph:
        """Generate a small-world network using Watts-Strogatz model."""
        return nx.watts_strogatz_graph(
            n=self.size,
            k=self.k,
            p=self.p,
            seed=self.seed
        )
    
    def get_node_distances(self, graph: nx.Graph) -> np.ndarray:
        """Calculate pairwise shortest path distances between all nodes."""
        return nx.floyd_warshall_numpy(graph)
    
    def get_centrality_measures(self, graph: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate degree and betweenness centrality for all nodes."""
        degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
        betweenness_centrality = np.array(list(nx.betweenness_centrality(graph).values()))
        return degree_centrality, betweenness_centrality 