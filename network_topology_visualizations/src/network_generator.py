"""
Network Generator for Topology Visualizations

This module creates actual network topologies using the existing topology classes
to ensure consistency with training code.
"""

import sys
import os
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import torch

# Add the main project root to the path to import topology classes
# Get absolute path to the project root (topology-playground directory)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

# Add both the project root and src directory to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import topology classes using the proper module path
try:
    from topologies.fully_connected import FullyConnectedTopology
    from topologies.small_world import SmallWorldTopology
    from topologies.modular import ModularTopology
    from topologies.hybrid import HybridTopology
    from topologies.standard_mlp import StandardMLPTopology
except ImportError as e:
    print(f"Error importing topology classes: {e}")
    print(f"Project root: {project_root}")
    print(f"Src directory: {src_dir}")
    print(f"Python path: {sys.path[:5]}...")  # Show first 5 entries
    print(f"Topology directory exists: {os.path.exists(os.path.join(src_dir, 'topologies'))}")
    if os.path.exists(os.path.join(src_dir, 'topologies')):
        print(f"Topology files: {os.listdir(os.path.join(src_dir, 'topologies'))}")
    raise


class NetworkGenerator:
    """Generates network topologies for visualization using existing topology classes."""
    
    def __init__(self, size: int = 64, seed: int = 42):
        """
        Initialize the network generator.
        
        Args:
            size: Network size (default: 64)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.seed = seed
        
        # Topology configurations matching training parameters
        self.topology_configs = {
            'fully_connected': {
                'class': FullyConnectedTopology,
                'params': {'size': size, 'seed': seed}
            },
            'small_world': {
                'class': SmallWorldTopology,
                'params': {'size': size, 'k': 4, 'p': 0.2, 'seed': seed}
            },
            'modular': {
                'class': ModularTopology,
                'params': {
                    'size': size,
                    'num_modules': 4,
                    'inter_module_prob': 0.1,
                    'intra_module_prob': 0.8,
                    'seed': seed
                }
            },
            'hybrid': {
                'class': HybridTopology,
                'params': {
                    'size': size,
                    'num_modules': 4,
                    'k': 4,
                    'p': 0.2,
                    'inter_module_prob': 0.1,
                    'seed': seed
                }
            },
            'standard_mlp_1layer': {
                'class': StandardMLPTopology,
                'params': {'size': size, 'num_layers': 1, 'activation': 'relu', 'seed': seed}
            },
            'standard_mlp_3layers': {
                'class': StandardMLPTopology,
                'params': {'size': size, 'num_layers': 3, 'activation': 'relu', 'seed': seed}
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
                # Create topology instance
                topology = config['class'](**config['params'])
                
                # Generate the network graph
                graph = topology.generate()
                
                # Store with metadata
                networks[name] = {
                    'graph': graph,
                    'topology': topology,
                    'params': config['params'],
                    'metrics': self._calculate_graph_metrics(graph, topology)
                }
                
                print(f"✓ Generated {name} topology: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
                
            except Exception as e:
                print(f"✗ Failed to generate {name} topology: {e}")
                networks[name] = None
        
        return networks
    
    def generate_single_topology(self, topology_name: str) -> Optional[Dict[str, Any]]:
        """
        Generate a single topology.
        
        Args:
            topology_name: Name of the topology to generate
            
        Returns:
            Dictionary with graph and metadata, or None if failed
        """
        if topology_name not in self.topology_configs:
            raise ValueError(f"Unknown topology: {topology_name}")
        
        config = self.topology_configs[topology_name]
        
        try:
            # Create topology instance
            topology = config['class'](**config['params'])
            
            # Generate the network graph
            graph = topology.generate()
            
            return {
                'graph': graph,
                'topology': topology,
                'params': config['params'],
                'metrics': self._calculate_graph_metrics(graph, topology)
            }
            
        except Exception as e:
            print(f"Failed to generate {topology_name} topology: {e}")
            return None
    
    def _calculate_graph_metrics(self, graph: nx.Graph, topology: Any) -> Dict[str, Any]:
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
            
            # Add topology-specific metrics if available
            try:
                if hasattr(topology, 'get_network_metrics'):
                    topology_metrics = topology.get_network_metrics()
                    metrics.update(topology_metrics)
            except:
                pass
            
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
        
        for name, config in self.topology_configs.items():
            info[name] = {
                'class_name': config['class'].__name__,
                'parameters': config['params'],
                'description': self._get_topology_description(name)
            }
        
        return info
    
    def _get_topology_description(self, topology_name: str) -> str:
        """Get a description of each topology."""
        descriptions = {
            'fully_connected': 'Complete DAG where every node connects to every higher-indexed node',
            'small_world': 'Ring lattice with some rewired connections (k=4, p=0.2)',
            'modular': 'Nodes grouped into 4 modules with high intra-module and low inter-module connectivity',
            'hybrid': 'Combines small-world within modules and modular structure between modules',
            'standard_mlp_1layer': 'Traditional feedforward network with 1 hidden layer',
            'standard_mlp_3layers': 'Multi-layer feedforward network with 3 hidden layers'
        }
        return descriptions.get(topology_name, 'Unknown topology')
    
    def export_graphs(self, networks: Dict[str, Dict], output_dir: str = "outputs"):
        """
        Export generated graphs to files.
        
        Args:
            networks: Dictionary of generated networks
            output_dir: Output directory for files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, network_data in networks.items():
            if network_data is None:
                continue
                
            graph = network_data['graph']
            
            # Export as GraphML
            graphml_path = os.path.join(output_dir, f"{name}.graphml")
            nx.write_graphml(graph, graphml_path)
            
            # Export as edge list
            edgelist_path = os.path.join(output_dir, f"{name}.edgelist")
            nx.write_edgelist(graph, edgelist_path)
            
            # Export metrics as JSON
            import json
            metrics_path = os.path.join(output_dir, f"{name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(network_data['metrics'], f, indent=2, default=str)
            
            print(f"✓ Exported {name} to {output_dir}/")


if __name__ == "__main__":
    # Test the network generator
    print("Testing Network Generator...")
    generator = NetworkGenerator(size=64, seed=42)
    
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
    
    # Export graphs
    generator.export_graphs(networks)
