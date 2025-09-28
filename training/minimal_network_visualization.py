#!/usr/bin/env python3
"""
Minimal Network Topology Visualization for Clear Comparison

Creates small, clean visualizations that clearly show the differences between
network topologies - perfect for presentations and papers.
"""

import torch
import torch.nn as nn
import numpy as np
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

def create_minimal_topology(topology_type: str, hidden_size: int, seed: int) -> Any:
    """Create minimal topology network for clear visualization."""
    
    if topology_type == 'small_world':
        topology = SmallWorldTopology(
            size=hidden_size,
            k=min(4, hidden_size//2),  # Ensure k doesn't exceed half the network size
            p=0.3  # Higher rewiring for clearer small-world effect
        )
    elif topology_type == 'modular':
        topology = ModularTopology(
            size=hidden_size,
            num_modules=min(4, hidden_size//4),  # Ensure reasonable module count
            inter_module_prob=0.05,  # Very low inter-module connections
            intra_module_prob=0.8   # High intra-module connections
        )
    elif topology_type == 'hybrid':
        topology = HybridTopology(
            size=hidden_size,
            num_modules=min(3, hidden_size//5),
            k=min(3, hidden_size//3),
            p=0.25,
            inter_module_prob=0.1
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    return topology

def create_minimal_visualization(graph: nx.Graph, topology_type: str, 
                               input_dim: int, output_dim: int, 
                               output_path: str = None):
    """Create a minimal, clean visualization focused on topology structure."""
    
    # Set up clean plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['grid.linewidth'] = 0.3
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Determine layout based on topology
    if topology_type == 'modular':
        # Use circular layout for modular networks to show modules clearly
        pos = nx.circular_layout(graph)
    else:
        # Use spring layout for others
        pos = nx.spring_layout(graph, k=2, iterations=100, seed=42)
    
    # Color nodes by type with clear, distinct colors
    node_colors = []
    node_sizes = []
    
    for node in graph.nodes():
        if node < input_dim:
            # Input nodes - blue
            node_colors.append('#2E86AB')
            node_sizes.append(150)
        elif node < input_dim + (len(graph.nodes()) - input_dim - output_dim):
            # Hidden nodes - green
            node_colors.append('#A23B72')
            node_sizes.append(100)
        else:
            # Output nodes - red
            node_colors.append('#F18F01')
            node_sizes.append(150)
    
    # Draw the network with clean styling
    nx.draw_networkx_nodes(graph, pos, 
                          node_color=node_colors, 
                          node_size=node_sizes,
                          alpha=0.9, 
                          linewidths=0.5,
                          edgecolors='black',
                          ax=ax)
    
    # Draw edges with appropriate styling
    if topology_type == 'modular':
        # For modular networks, use different colors for intra vs inter module edges
        intra_edges = []
        inter_edges = []
        
        # Simple heuristic: edges between nodes close in index are likely intra-module
        for edge in graph.edges():
            node1, node2 = edge
            if abs(node1 - node2) <= len(graph.nodes()) // 4:  # Rough module size estimate
                intra_edges.append(edge)
            else:
                inter_edges.append(edge)
        
        if intra_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=intra_edges, 
                                 alpha=0.6, edge_color='#A23B72', width=1.5, ax=ax)
        if inter_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=inter_edges, 
                                 alpha=0.4, edge_color='gray', width=0.8, ax=ax)
    else:
        # For other topologies, use uniform edge styling
        nx.draw_networkx_edges(graph, pos, 
                              alpha=0.5, 
                              edge_color='gray', 
                              width=0.8, 
                              ax=ax)
    
    # Clean up the plot
    ax.set_title(f'{topology_type.replace("_", " ").title()} Network\n{len(graph.nodes())} nodes, {len(graph.edges())} edges', 
                fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add minimal legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E86AB', label=f'Input ({input_dim})'),
        Patch(facecolor='#A23B72', label=f'Hidden ({len(graph.nodes()) - input_dim - output_dim})'),
        Patch(facecolor='#F18F01', label=f'Output ({output_dim})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📁 Minimal visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig

def create_comparison_visualization(topologies: List[str], hidden_size: int, 
                                  input_dim: int, output_dim: int, 
                                  output_path: str = None):
    """Create side-by-side comparison of different topologies."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams['font.size'] = 9
    
    for i, topology_type in enumerate(topologies):
        ax = axes[i]
        
        # Create topology
        topology = create_minimal_topology(topology_type, hidden_size, 42)
        graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
        
        # Determine layout
        if topology_type == 'modular':
            pos = nx.circular_layout(graph)
        else:
            pos = nx.spring_layout(graph, k=1.5, iterations=50, seed=42)
        
        # Color nodes
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            if node < input_dim:
                node_colors.append('#2E86AB')
                node_sizes.append(120)
            elif node < input_dim + (len(graph.nodes()) - input_dim - output_dim):
                node_colors.append('#A23B72')
                node_sizes.append(80)
            else:
                node_colors.append('#F18F01')
                node_sizes.append(120)
        
        # Draw network
        nx.draw_networkx_nodes(graph, pos, 
                              node_color=node_colors, 
                              node_size=node_sizes,
                              alpha=0.9, 
                              linewidths=0.3,
                              edgecolors='black',
                              ax=ax)
        
        # Draw edges
        if topology_type == 'modular':
            # Different colors for intra/inter module edges
            intra_edges = []
            inter_edges = []
            for edge in graph.edges():
                node1, node2 = edge
                if abs(node1 - node2) <= len(graph.nodes()) // 4:
                    intra_edges.append(edge)
                else:
                    inter_edges.append(edge)
            
            if intra_edges:
                nx.draw_networkx_edges(graph, pos, edgelist=intra_edges, 
                                     alpha=0.7, edge_color='#A23B72', width=1.2, ax=ax)
            if inter_edges:
                nx.draw_networkx_edges(graph, pos, edgelist=inter_edges, 
                                     alpha=0.4, edge_color='gray', width=0.6, ax=ax)
        else:
            nx.draw_networkx_edges(graph, pos, 
                                  alpha=0.5, 
                                  edge_color='gray', 
                                  width=0.6, 
                                  ax=ax)
        
        # Clean styling
        ax.set_title(f'{topology_type.replace("_", " ").title()}\n{len(graph.nodes())} nodes, {len(graph.edges())} edges', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Network Topology Comparison', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"📁 Comparison visualization saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    return fig

def main():
    """Main function to create minimal network visualizations."""
    
    parser = argparse.ArgumentParser(description="Minimal Network Topology Visualization")
    parser.add_argument("--topology", type=str, default="small_world", 
                       choices=["small_world", "modular", "hybrid"],
                       help="Network topology type")
    parser.add_argument("--task", type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v2"],
                       help="Environment to visualize")
    parser.add_argument("--hidden_size", type=int, default=12, 
                       help="Hidden layer size (default: 12 for minimal visualization)")
    parser.add_argument("--output", type=str, help="Output file path for visualization")
    parser.add_argument("--comparison", action="store_true", 
                       help="Create side-by-side comparison of all topologies")
    
    args = parser.parse_args()
    
    print("🎯 Minimal Network Topology Visualization")
    print("=" * 50)
    print(f"Configuration:")
    print(f"   Topology: {args.topology}")
    print(f"   Task: {args.task}")
    print(f"   Hidden Size: {args.hidden_size}")
    print(f"   Output: {args.output or 'Display only'}")
    print("=" * 50)
    
    # Get task dimensions
    import gymnasium as gym
    env = gym.make(args.task)
    
    if hasattr(env.observation_space, 'shape'):
        if len(env.observation_space.shape) == 1:
            input_dim = env.observation_space.shape[0]
        else:
            input_dim = np.prod(env.observation_space.shape)
    else:
        input_dim = env.observation_space.n
    
    if hasattr(env.action_space, 'n'):
        output_dim = env.action_space.n
    elif hasattr(env.action_space, 'shape'):
        output_dim = env.action_space.shape[0]
    else:
        output_dim = 1
    
    env.close()
    
    print(f"📊 Task dimensions: {input_dim} input, {output_dim} output")
    
    try:
        if args.comparison:
            # Create comparison visualization
            topologies = ['small_world', 'modular', 'hybrid']
            output_path = args.output or 'minimal_network_comparison.png'
            create_comparison_visualization(topologies, args.hidden_size, 
                                          input_dim, output_dim, output_path)
        else:
            # Create single topology visualization
            topology = create_minimal_topology(args.topology, args.hidden_size, 42)
            graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
            
            output_path = args.output or f'minimal_{args.topology}_network.png'
            create_minimal_visualization(graph, args.topology, 
                                       input_dim, output_dim, output_path)
        
        print(f"✅ Visualization completed successfully!")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
