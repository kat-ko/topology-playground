import numpy as np
from typing import List, Tuple, Dict
import networkx as nx

class NodeSelector:
    def __init__(self, graph: nx.Graph, num_nodes: int, seed: int = None):
        self.graph = graph
        self.num_nodes = num_nodes
        self.rng = np.random.RandomState(seed)
        
    def random_selection(self) -> Tuple[List[int], List[int]]:
        """Select input and output nodes randomly."""
        all_nodes = list(self.graph.nodes())
        self.rng.shuffle(all_nodes)
        return all_nodes[:self.num_nodes], all_nodes[self.num_nodes:2*self.num_nodes]
    
    def centrality_based_selection(self) -> Tuple[List[int], List[int]]:
        """Select input and output nodes based on centrality measures."""
        degree_centrality = np.array(list(nx.degree_centrality(self.graph).values()))
        betweenness_centrality = np.array(list(nx.betweenness_centrality(self.graph).values()))
        
        # Combine centrality measures
        combined_centrality = degree_centrality + betweenness_centrality
        
        # Select top nodes for input
        input_nodes = np.argsort(combined_centrality)[-self.num_nodes:]
        
        # Select nodes with medium centrality for output
        remaining_nodes = list(set(range(len(self.graph))) - set(input_nodes))
        remaining_centrality = combined_centrality[remaining_nodes]
        output_nodes = [remaining_nodes[i] for i in np.argsort(remaining_centrality)[-self.num_nodes:]]
        
        return list(input_nodes), output_nodes
    
    def distance_based_selection(self) -> Tuple[List[int], List[int]]:
        """Select input and output nodes based on their distance in the network."""
        distances = nx.floyd_warshall_numpy(self.graph)
        
        # Find nodes that are far apart
        max_dist = np.max(distances)
        far_pairs = np.where(distances == max_dist)
        
        # Select input nodes around the first far node
        input_center = far_pairs[0][0]
        input_distances = distances[input_center]
        input_nodes = np.argsort(input_distances)[:self.num_nodes]
        
        # Select output nodes around the second far node
        output_center = far_pairs[1][0]
        output_distances = distances[output_center]
        output_nodes = np.argsort(output_distances)[:self.num_nodes]
        
        return list(input_nodes), list(output_nodes)
    
    def module_based_selection(self, module_assignments: Dict[int, int]) -> Tuple[List[int], List[int]]:
        """Select input and output nodes from different modules."""
        # Group nodes by module
        module_nodes = {}
        for node, module in module_assignments.items():
            if module not in module_nodes:
                module_nodes[module] = []
            module_nodes[module].append(node)
        
        # Select input nodes from first module
        input_module = min(module_nodes.keys())
        input_nodes = self.rng.choice(module_nodes[input_module], 
                                    size=min(self.num_nodes, len(module_nodes[input_module])),
                                    replace=False)
        
        # Select output nodes from last module
        output_module = max(module_nodes.keys())
        output_nodes = self.rng.choice(module_nodes[output_module],
                                     size=min(self.num_nodes, len(module_nodes[output_module])),
                                     replace=False)
        
        return list(input_nodes), list(output_nodes) 