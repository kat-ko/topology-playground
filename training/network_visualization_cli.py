#!/usr/bin/env python3
"""
Network Topology Visualization Script

This script creates visualizations of network topologies that would be used in training,
without actually training them. It accepts the same command line arguments as the
continual task training script.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import argparse
import sys
import os
from pathlib import Path

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.standard_mlp import StandardMLPTopology

# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def get_task_dimensions(task_name: str) -> Tuple[int, int]:
    """Get input and output dimensions for a given task."""
    # Create environment to get dimensions
    env = gym.make(task_name)
    
    # Get observation dimension
    if hasattr(env.observation_space, 'shape'):
        if len(env.observation_space.shape) == 1:
            input_dim = env.observation_space.shape[0]
        else:
            # Flatten multi-dimensional observations
            input_dim = np.prod(env.observation_space.shape)
    else:
        input_dim = env.observation_space.n
    
    # Get action dimension
    if hasattr(env.action_space, 'n'):
        output_dim = env.action_space.n
    elif hasattr(env.action_space, 'shape'):
        output_dim = env.action_space.shape[0]
    else:
        output_dim = 1
    
    env.close()
    return int(input_dim), int(output_dim)

def create_topology_network(topology_type: str, hidden_size: int, num_layers: int, 
                           activation: str, dropout: float, seed: int, 
                           topology_params: Dict) -> Any:
    """Create topology network based on type and parameters."""
    
    if topology_type == 'fully_connected':
        topology = FullyConnectedTopology(
            size=hidden_size,
            seed=seed
        )
    elif topology_type == 'standard_mlp':
        topology = StandardMLPTopology(
            size=hidden_size,
            num_layers=num_layers,
            activation=activation
        )
    elif topology_type == 'small_world':
        topology = SmallWorldTopology(
            size=hidden_size,
            k=topology_params.get('small_world_k', 4),
            p=topology_params.get('small_world_p', 0.2)
        )
    elif topology_type == 'modular':
        topology = ModularTopology(
            size=hidden_size,
            num_modules=topology_params.get('modular_num_modules', 4),
            inter_module_prob=topology_params.get('modular_inter_module_prob', 0.1),
            intra_module_prob=topology_params.get('modular_intra_module_prob', 0.8)
        )
    elif topology_type == 'hybrid':
        topology = HybridTopology(
            size=hidden_size,
            num_modules=topology_params.get('modular_num_modules', 4),
            k=topology_params.get('hybrid_k', 4),
            p=topology_params.get('hybrid_p', 0.2),
            inter_module_prob=topology_params.get('hybrid_inter_module_prob', 0.1)
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return topology

def visualize_network(graph: nx.Graph, config: Dict, input_dim: int, output_dim: int, 
                     output_path: str = None):
    """Create a comprehensive visualization of the network topology."""
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (20, 12)
    plt.rcParams['font.size'] = 12
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot 1: Network graph
    ax1.set_title(f"Network Topology: {config['topology_type'].upper()}\n{config['task']} - {config['hidden_size']} hidden nodes", 
                  fontsize=16, fontweight='bold')
    
    # Determine node positions based on topology type
    if config['topology_type'] == 'standard_mlp':
        # Layered layout for MLP
        pos = nx.multipartite_layout(graph, subset_key='layer')
    else:
        # Spring layout for other topologies
        pos = nx.spring_layout(graph, k=3, iterations=50, seed=config['seed'])
    
    # Color nodes by type
    node_colors = []
    node_sizes = []
    
    for node in graph.nodes():
        if node < input_dim:
            # Input nodes
            node_colors.append('lightblue')
            node_sizes.append(300)
        elif node < input_dim + config['hidden_size']:
            # Hidden nodes
            node_colors.append('lightgreen')
            node_sizes.append(200)
        else:
            # Output nodes
            node_colors.append('lightcoral')
            node_sizes.append(300)
    
    # Draw the network
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray', width=0.5, ax=ax1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label=f'Input Nodes ({input_dim})'),
        Patch(facecolor='lightgreen', label=f'Hidden Nodes ({config["hidden_size"]})'),
        Patch(facecolor='lightcoral', label=f'Output Nodes ({output_dim})')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Network statistics
    ax2.set_title("Network Statistics", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Calculate network statistics
    total_nodes = len(graph.nodes())
    total_edges = len(graph.edges())
    density = nx.density(graph)
    
    # Calculate average degree
    degrees = [d for n, d in graph.degree()]
    avg_degree = np.mean(degrees)
    
    # Calculate clustering coefficient if possible
    try:
        clustering = nx.average_clustering(graph)
    except:
        clustering = "N/A"
    
    # Display statistics
    stats_text = f"""
Network Statistics:

Topology Type: {config['topology_type'].upper()}
Task: {config['task']}
Hidden Size: {config['hidden_size']}
Number of Layers: {config['num_layers']}

Total Nodes: {total_nodes}
Total Edges: {total_edges}
Network Density: {density:.4f}
Average Degree: {avg_degree:.2f}
Clustering Coefficient: {clustering}

Node Distribution:
• Input: {input_dim}
• Hidden: {config['hidden_size']}
• Output: {output_dim}
"""
    
    ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📁 Network visualization saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def analyze_topology_specific_features(graph: nx.Graph, config: Dict):
    """Analyze topology-specific features and display them."""
    
    topology_type = config['topology_type']
    
    print(f"\n🔍 Topology-Specific Analysis:")
    print(f"=" * 50)
    
    if topology_type == 'small_world':
        # Small-world characteristics
        try:
            # Calculate average path length
            avg_path_length = nx.average_shortest_path_length(graph)
            clustering = nx.average_clustering(graph)
            
            print(f"Small-World Analysis:")
            print(f"   k (nearest neighbors): {config.get('small_world_k', 4)}")
            print(f"   p (rewiring probability): {config.get('small_world_p', 0.2)}")
            print(f"   Average path length: {avg_path_length:.3f}")
            print(f"   Clustering coefficient: {clustering:.3f}")
            
            # Small-world property: high clustering + low path length
            if clustering > 0.3 and avg_path_length < 3:
                print(f"   ✅ Small-world property confirmed!")
            else:
                print(f"   ⚠️  Small-world property not clearly established")
                
        except Exception as e:
            print(f"   ⚠️  Could not calculate small-world metrics: {e}")
    
    elif topology_type == 'modular':
        # Modular characteristics
        print(f"Modular Analysis:")
        print(f"   Number of modules: {config.get('modular_num_modules', 4)}")
        print(f"   Inter-module probability: {config.get('modular_inter_module_prob', 0.1)}")
        print(f"   Intra-module probability: {config.get('modular_intra_module_prob', 0.8)}")
        
        # Calculate modularity
        try:
            from community import best_partition
            partition = best_partition(graph)
            modularity = nx.community.modularity(graph, [set(n for n in partition if partition[n] == i) 
                                                       for i in set(partition.values())])
            print(f"   Modularity score: {modularity:.3f}")
        except ImportError:
            print(f"   ⚠️  Community detection library not available for modularity calculation")
        except Exception as e:
            print(f"   ⚠️  Could not calculate modularity: {e}")
    
    elif topology_type == 'hybrid':
        # Hybrid characteristics
        print(f"Hybrid Analysis:")
        print(f"   Small-world k: {config.get('hybrid_k', 4)}")
        print(f"   Small-world p: {config.get('hybrid_p', 0.2)}")
        print(f"   Number of modules: {config.get('modular_num_modules', 4)}")
        print(f"   Inter-module probability: {config.get('hybrid_inter_module_prob', 0.1)}")
        
        # Calculate both small-world and modular properties
        try:
            avg_path_length = nx.average_shortest_path_length(graph)
            clustering = nx.average_clustering(graph)
            print(f"   Average path length: {avg_path_length:.3f}")
            print(f"   Clustering coefficient: {clustering:.3f}")
        except Exception as e:
            print(f"   ⚠️  Could not calculate hybrid metrics: {e}")
    
    elif topology_type == 'standard_mlp':
        # MLP characteristics
        print(f"MLP Analysis:")
        print(f"   Number of layers: {config['num_layers']}")
        print(f"   Activation function: {config.get('activation', 'leaky_relu')}")
        print(f"   Dropout rate: {config.get('dropout', 0.0)}")
        
        # Calculate layer-wise statistics
        try:
            # Group nodes by layer
            layer_nodes = {}
            for node in graph.nodes():
                if 'layer' in graph.nodes[node]:
                    layer = graph.nodes[node]['layer']
                    if layer not in layer_nodes:
                        layer_nodes[layer] = []
                    layer_nodes[layer].append(node)
            
            print(f"   Layer distribution:")
            for layer in sorted(layer_nodes.keys()):
                print(f"     Layer {layer}: {len(layer_nodes[layer])} nodes")
        except Exception as e:
            print(f"   ⚠️  Could not analyze layer structure: {e}")
    
    elif topology_type == 'fully_connected':
        # Fully connected characteristics
        print(f"Fully Connected Analysis:")
        print(f"   Total connections: {len(graph.edges())}")
        print(f"   Network density: {nx.density(graph):.4f}")
        print(f"   Average degree: {np.mean([d for n, d in graph.degree()]):.2f}")
        
        # Check if it's truly fully connected
        expected_edges = len(graph.nodes()) * (len(graph.nodes()) - 1) // 2
        if len(graph.edges()) == expected_edges:
            print(f"   ✅ Fully connected topology confirmed!")
        else:
            print(f"   ⚠️  Not fully connected (expected {expected_edges}, got {len(graph.edges())})")

def main():
    """Main function to create and visualize network topologies."""
    
    # Parse command line arguments (same as training script)
    parser = argparse.ArgumentParser(description="Network Topology Visualization")
    parser.add_argument("--topology", type=str, default="small_world", 
                       choices=["small_world", "modular", "hybrid", "fully_connected", "standard_mlp"],
                       help="Network topology type")
    parser.add_argument("--task", type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v2"],
                       help="Environment to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_layers", type=int, default=1, 
                       help="Number of layers for topology networks (default: 1)")
    parser.add_argument("--hidden_size", type=int, default=128, 
                       help="Hidden layer size (default: 128)")
    parser.add_argument("--output", type=str, help="Output file path for visualization (optional)")
    
    args = parser.parse_args()
    
    print("🚀 Network Topology Visualization")
    print("=" * 60)
    print(f"🎯 Configuration:")
    print(f"   Topology: {args.topology}")
    print(f"   Task: {args.task}")
    print(f"   Seed: {args.seed}")
    print(f"   Hidden Size: {args.hidden_size}")
    print(f"   Number of Layers: {args.num_layers}")
    print(f"   Output: {args.output or 'Display only (no save)'}")
    print("=" * 60)
    
    # Create configuration dictionary
    config = {
        'topology_type': args.topology,
        'task': args.task,
        'seed': args.seed,
        'num_layers': args.num_layers,
        'hidden_size': args.hidden_size,
        'activation': 'leaky_relu',
        'dropout': 0.0,
        
        # Topology-specific parameters
        'small_world_k': 4,
        'small_world_p': 0.2,
        'modular_num_modules': 4,
        'modular_inter_module_prob': 0.1,
        'modular_intra_module_prob': 0.8,
        'hybrid_k': 4,
        'hybrid_p': 0.2,
        'hybrid_inter_module_prob': 0.1
    }
    
    try:
        # Get task dimensions
        input_dim, output_dim = get_task_dimensions(args.task)
        print(f"📊 Task dimensions: {input_dim} input, {output_dim} output")
        
        # Create topology network
        topology = create_topology_network(
            topology_type=args.topology,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            activation=config['activation'],
            dropout=config['dropout'],
            seed=args.seed,
            topology_params=config
        )
        
        # Generate graph
        graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
        
        print(f"✅ Network created successfully")
        print(f"   Total nodes: {len(graph.nodes())}")
        print(f"   Total edges: {len(graph.edges())}")
        
        # Create visualization
        fig = visualize_network(graph, config, input_dim, output_dim, args.output)
        
        # Perform topology-specific analysis
        analyze_topology_specific_features(graph, config)
        
        print(f"\n🎉 Network visualization completed successfully!")
        
        if not args.output:
            print(f"💡 Tip: Use --output <filename> to save the visualization to a file")
        
    except Exception as e:
        print(f"❌ Error creating network visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
