"""
Direct Network Generator for Topology Visualizations

This module creates network topologies directly using the same logic as the
original topology classes, but without the complex dependencies.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import torch


class DirectNetworkGenerator:
    """Generates network topologies directly using the same logic as training classes."""
    
    def __init__(self, size: int = 64, seed: int = 42):
        """
        Initialize the network generator.
        
        Args:
            size: Network size (default: 64)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Topology configurations matching training parameters
        self.topology_configs = {
            'fully_connected': {
                'k': None, 'p': None, 'num_modules': None,
                'inter_module_prob': None, 'intra_module_prob': None,
                'num_layers': 1
            },
            'small_world': {
                'k': 4, 'p': 0.2, 'num_modules': None,
                'inter_module_prob': None, 'intra_module_prob': None,
                'num_layers': 1
            },
            'modular': {
                'k': None, 'p': None, 'num_modules': 4,
                'inter_module_prob': 0.1, 'intra_module_prob': 0.8,
                'num_layers': 1
            },
            'hybrid': {
                'k': 4, 'p': 0.2, 'num_modules': 4,
                'inter_module_prob': 0.1, 'intra_module_prob': 0.8,
                'num_layers': 1
            },
            'standard_mlp_1layer': {
                'k': None, 'p': None, 'num_modules': None,
                'inter_module_prob': None, 'intra_module_prob': None,
                'num_layers': 1
            },
            'standard_mlp_3layers': {
                'k': None, 'p': None, 'num_modules': None,
                'inter_module_prob': None, 'intra_module_prob': None,
                'num_layers': 3
            }
        }
    
    def generate_all_topologies(self) -> Dict[str, nx.Graph]:
        """
        Generate all network topologies.
        
        Returns:
            Dictionary mapping topology names to NetworkX graphs
        """
        networks = {}
        
        for name, config in self.topology_configs.items():
            try:
                # Generate the network graph
                graph = self._generate_topology(name, config)
                
                # Store with metadata
                networks[name] = {
                    'graph': graph,
                    'params': config,
                    'metrics': self._calculate_graph_metrics(graph)
                }
                
                print(f"✓ Generated {name} topology: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
                
            except Exception as e:
                print(f"✗ Failed to generate {name} topology: {e}")
                networks[name] = None
        
        return networks
    
    def _generate_topology(self, topology_name: str, config: Dict[str, Any]) -> nx.Graph:
        """Generate a specific topology based on the configuration."""
        if topology_name == 'fully_connected':
            return self._generate_fully_connected()
        elif topology_name == 'small_world':
            return self._generate_small_world(config['k'], config['p'])
        elif topology_name == 'modular':
            return self._generate_modular(config['num_modules'], 
                                        config['inter_module_prob'], 
                                        config['intra_module_prob'])
        elif topology_name == 'hybrid':
            return self._generate_hybrid(config['num_modules'], config['k'], config['p'],
                                       config['inter_module_prob'])
        elif 'mlp' in topology_name:
            return self._generate_mlp(config['num_layers'])
        else:
            raise ValueError(f"Unknown topology: {topology_name}")
    
    def _generate_fully_connected(self) -> nx.Graph:
        """Generate fully connected topology (complete DAG)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Add connections: every node connects to every higher-indexed node (ensures DAG)
        for i in range(self.size):
            for j in range(i + 1, self.size):
                G.add_edge(i, j)
        
        return G
    
    def _generate_small_world(self, k: int, p: float) -> nx.Graph:
        """Generate small-world topology."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Create initial ring lattice structure (directed, acyclic)
        for i in range(self.size):
            for j in range(1, k // 2 + 1):
                target = (i + j) % self.size
                if target > i:  # Only add forward edges
                    G.add_edge(i, target)
        
        # Rewire edges with probability p (maintaining acyclicity)
        for edge in list(G.edges()):
            if self.rng.random() < p:
                # Remove the edge
                G.remove_edge(*edge)
                # Add a new random edge (only to higher-indexed nodes)
                new_node = self.rng.randint(edge[0] + 1, self.size)
                while G.has_edge(edge[0], new_node):
                    new_node = self.rng.randint(edge[0] + 1, self.size)
                G.add_edge(edge[0], new_node)
        
        return G
    
    def _generate_modular(self, num_modules: int, inter_module_prob: float, 
                         intra_module_prob: float) -> nx.Graph:
        """Generate modular topology."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Calculate module sizes
        module_size = self.size // num_modules
        extra_nodes = self.size % num_modules
        
        # Assign nodes to modules
        module_assignments = {}
        current_node = 0
        
        for module in range(num_modules):
            # Add extra node to first few modules if needed
            current_module_size = module_size + (1 if module < extra_nodes else 0)
            
            for _ in range(current_module_size):
                module_assignments[current_node] = module
                current_node += 1
        
        # Add intra-module connections (directed, acyclic)
        for module in range(num_modules):
            module_nodes = [node for node, mod in module_assignments.items() if mod == module]
            module_nodes.sort()  # Ensure acyclicity
            
            for i in range(len(module_nodes)):
                for j in range(i + 1, len(module_nodes)):
                    if self.rng.random() < intra_module_prob:
                        G.add_edge(module_nodes[i], module_nodes[j])
        
        # Add inter-module connections (directed, acyclic)
        for module1 in range(num_modules):
            for module2 in range(num_modules):
                if module1 != module2:
                    module1_nodes = [node for node, mod in module_assignments.items() if mod == module1]
                    module2_nodes = [node for node, mod in module_assignments.items() if mod == module2]
                    module1_nodes.sort()
                    module2_nodes.sort()
                    
                    for node1 in module1_nodes:
                        for node2 in module2_nodes:
                            if node1 < node2 and self.rng.random() < inter_module_prob:
                                G.add_edge(node1, node2)
        
        return G
    
    def _generate_hybrid(self, num_modules: int, k: int, p: float, 
                        inter_module_prob: float) -> nx.Graph:
        """Generate hybrid topology (small-world within modules + modular structure)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.size))
        
        # Calculate module sizes
        module_size = self.size // num_modules
        extra_nodes = self.size % num_modules
        
        # Assign nodes to modules
        module_assignments = {}
        current_node = 0
        
        for module in range(num_modules):
            current_module_size = module_size + (1 if module < extra_nodes else 0)
            
            for _ in range(current_module_size):
                module_assignments[current_node] = module
                current_node += 1
        
        # Create small-world graphs for each module
        for module in range(num_modules):
            module_nodes = [node for node, mod in module_assignments.items() if mod == module]
            module_nodes.sort()
            
            # Create ring lattice within module
            for i in range(len(module_nodes)):
                for j in range(1, k // 2 + 1):
                    target_idx = (i + j) % len(module_nodes)
                    if target_idx > i:  # Only add forward edges
                        G.add_edge(module_nodes[i], module_nodes[target_idx])
            
            # Rewire edges within module
            module_edges = [(u, v) for u, v in G.edges() 
                           if u in module_nodes and v in module_nodes]
            
            for edge in module_edges:
                if self.rng.random() < p:
                    G.remove_edge(*edge)
                    # Add new random edge within module
                    new_node = self.rng.choice([n for n in module_nodes if n > edge[0]])
                    while G.has_edge(edge[0], new_node):
                        new_node = self.rng.choice([n for n in module_nodes if n > edge[0]])
                    G.add_edge(edge[0], new_node)
        
        # Add inter-module connections
        for module1 in range(num_modules):
            for module2 in range(num_modules):
                if module1 != module2:
                    module1_nodes = [node for node, mod in module_assignments.items() if mod == module1]
                    module2_nodes = [node for node, mod in module_assignments.items() if mod == module2]
                    module1_nodes.sort()
                    module2_nodes.sort()
                    
                    for node1 in module1_nodes:
                        for node2 in module2_nodes:
                            if node1 < node2 and self.rng.random() < inter_module_prob:
                                G.add_edge(node1, node2)
        
        return G
    
    def _generate_mlp(self, num_layers: int) -> nx.Graph:
        """Generate MLP topology with specified number of layers."""
        G = nx.DiGraph()
        
        # Calculate layer sizes
        if num_layers == 1:
            # Single hidden layer
            layer_sizes = [4, self.size, 4]  # Input, Hidden, Output
        else:
            # Multiple hidden layers
            hidden_size = self.size // num_layers
            layer_sizes = [4] + [hidden_size] * num_layers + [4]
        
        # Add all nodes
        total_nodes = sum(layer_sizes)
        G.add_nodes_from(range(total_nodes))
        
        # Calculate layer boundaries
        layer_boundaries = []
        current_node = 0
        for size in layer_sizes:
            layer_boundaries.append((current_node, current_node + size))
            current_node += size
        
        # Connect layers sequentially
        for i in range(len(layer_boundaries) - 1):
            start1, end1 = layer_boundaries[i]
            start2, end2 = layer_boundaries[i + 1]
            
            # Connect all nodes from current layer to next layer
            for node1 in range(start1, end1):
                for node2 in range(start2, end2):
                    G.add_edge(node1, node2)
        
        return G
    
    def _calculate_graph_metrics(self, graph: nx.Graph) -> Dict[str, Any]:
        """Calculate metrics for the generated graph."""
        try:
            # Convert to undirected for some metrics
            undirected_graph = graph.to_undirected() if graph.is_directed() else graph
            
            metrics = {
                'num_nodes': len(graph.nodes()),
                'num_edges': len(graph.edges()),
                'density': nx.density(graph),
                'is_directed': graph.is_directed(),
                'is_connected': nx.is_connected(undirected_graph) if len(graph.nodes()) > 0 else True,
                'average_degree': np.mean([d for n, d in graph.degree()]) if len(graph.nodes()) > 0 else 0
            }
            
            # Add connectivity metrics for undirected graph
            if metrics['is_connected'] and len(graph.nodes()) > 1:
                try:
                    metrics['average_clustering'] = nx.average_clustering(undirected_graph)
                    metrics['diameter'] = nx.diameter(undirected_graph)
                    metrics['average_shortest_path'] = nx.average_shortest_path_length(undirected_graph)
                except:
                    metrics['average_clustering'] = 0.0
                    metrics['diameter'] = 0
                    metrics['average_shortest_path'] = 0.0
            else:
                metrics['average_clustering'] = 0.0
                metrics['diameter'] = 0
                metrics['average_shortest_path'] = 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Could not calculate some metrics: {e}")
            return {
                'num_nodes': len(graph.nodes()),
                'num_edges': len(graph.edges()),
                'density': 0.0,
                'is_directed': graph.is_directed(),
                'is_connected': False,
                'average_degree': 0.0,
                'average_clustering': 0.0,
                'diameter': 0,
                'average_shortest_path': 0.0
            }
    
    def get_topology_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available topologies."""
        info = {}
        
        descriptions = {
            'fully_connected': 'Complete DAG where every node connects to every higher-indexed node',
            'small_world': 'Ring lattice with some rewired connections (k=4, p=0.2)',
            'modular': 'Nodes grouped into 4 modules with high intra-module and low inter-module connectivity',
            'hybrid': 'Combines small-world within modules and modular structure between modules',
            'standard_mlp_1layer': 'Traditional feedforward network with 1 hidden layer',
            'standard_mlp_3layers': 'Multi-layer feedforward network with 3 hidden layers'
        }
        
        for name, config in self.topology_configs.items():
            info[name] = {
                'parameters': config,
                'description': descriptions.get(name, 'Unknown topology')
            }
        
        return info


if __name__ == "__main__":
    # Test the direct network generator
    print("Testing Direct Network Generator...")
    generator = DirectNetworkGenerator(size=64, seed=42)
    
    # Generate all topologies
    networks = generator.generate_all_topologies()
    
    # Print summary
    print("\n" + "="*60)
    print("NETWORK TOPOLOGY SUMMARY")
    print("="*60)
    
    for name, network_data in networks.items():
        if network_data is None:
            print(f"{name:20s}: FAILED")
            continue
            
        metrics = network_data['metrics']
        print(f"{name:20s}: {metrics['num_nodes']:3d} nodes, {metrics['num_edges']:4d} edges, "
              f"density={metrics['density']:.3f}, clustering={metrics['average_clustering']:.3f}")
    
    print("✓ Direct network generator test completed!")
