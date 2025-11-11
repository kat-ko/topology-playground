#!/usr/bin/env python3
"""
Single Topology Analysis Example

This example demonstrates how to analyze and visualize
a single network topology in detail.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.network_generator import NetworkGenerator
from src.visualization_engine import VisualizationEngine


def analyze_single_topology(topology_name: str, size: int = 64):
    """
    Analyze a single topology in detail.
    
    Args:
        topology_name: Name of the topology to analyze
        size: Network size
    """
    print(f"🔍 Analyzing {topology_name} topology (size={size})")
    print("=" * 50)
    
    # Generate the network
    generator = NetworkGenerator(size=size, seed=42)
    network_data = generator.generate_single_topology(topology_name)
    
    if network_data is None:
        print(f"❌ Failed to generate {topology_name} topology")
        return
    
    graph = network_data['graph']
    metrics = network_data['metrics']
    
    # Print detailed metrics
    print("📊 Network Metrics:")
    print(f"  Nodes: {metrics['num_nodes']}")
    print(f"  Edges: {metrics['num_edges']}")
    print(f"  Density: {metrics['density']:.4f}")
    print(f"  Average Degree: {metrics['average_degree']:.2f}")
    print(f"  Directed: {metrics['is_directed']}")
    print(f"  Connected: {metrics['is_connected']}")
    
    if 'average_clustering' in metrics:
        print(f"  Average Clustering: {metrics['average_clustering']:.4f}")
    if 'diameter' in metrics:
        print(f"  Diameter: {metrics['diameter']}")
    if 'average_shortest_path' in metrics:
        print(f"  Average Shortest Path: {metrics['average_shortest_path']:.4f}")
    
    # Create visualizations
    visualizer = VisualizationEngine(figsize=(12, 10))
    
    print("\\n🎨 Creating visualizations...")
    
    # Connection density
    fig1 = visualizer.visualize_connection_density(
        graph, topology_name,
        title=f"{topology_name.replace('_', ' ').title()} - Connection Density\\n"
              f"Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}, "
              f"Density: {metrics['density']:.3f}"
    )
    
    # Structural patterns
    fig2 = visualizer.visualize_structural_patterns(
        graph, topology_name,
        title=f"{topology_name.replace('_', ' ').title()} - Structural Patterns"
    )
    
    # Information flow
    fig3 = visualizer.visualize_information_flow(
        graph, topology_name,
        title=f"{topology_name.replace('_', ' ').title()} - Information Flow"
    )
    
    # Save visualizations
    output_dir = f"single_analysis_{topology_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    fig1.savefig(os.path.join(output_dir, "connection_density.png"), 
                dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, "structural_patterns.png"), 
                dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, "information_flow.png"), 
                dpi=300, bbox_inches='tight')
    
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    
    print(f"✅ Analysis complete! Results saved to: {output_dir}/")
    
    return network_data


def compare_two_topologies(topology1: str, topology2: str, size: int = 64):
    """
    Compare two specific topologies side by side.
    
    Args:
        topology1: First topology name
        topology2: Second topology name
        size: Network size
    """
    print(f"⚖️  Comparing {topology1} vs {topology2} (size={size})")
    print("=" * 60)
    
    # Generate both networks
    generator = NetworkGenerator(size=size, seed=42)
    
    network1 = generator.generate_single_topology(topology1)
    network2 = generator.generate_single_topology(topology2)
    
    if network1 is None or network2 is None:
        print("❌ Failed to generate one or both topologies")
        return
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.suptitle(f'Topology Comparison: {topology1.replace("_", " ").title()} vs {topology2.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    visualizer = VisualizationEngine(figsize=(8, 6))
    
    # Row 1: Topology 1
    fig1 = visualizer.visualize_connection_density(network1['graph'], topology1)
    fig2 = visualizer.visualize_structural_patterns(network1['graph'], topology1)
    fig3 = visualizer.visualize_information_flow(network1['graph'], topology1)
    
    # Copy to first row
    for i, fig in enumerate([fig1, fig2, fig3]):
        _copy_figure_to_subplot(fig, axes[0, i])
        plt.close(fig)
    
    # Row 2: Topology 2
    fig1 = visualizer.visualize_connection_density(network2['graph'], topology2)
    fig2 = visualizer.visualize_structural_patterns(network2['graph'], topology2)
    fig3 = visualizer.visualize_information_flow(network2['graph'], topology2)
    
    # Copy to second row
    for i, fig in enumerate([fig1, fig2, fig3]):
        _copy_figure_to_subplot(fig, axes[1, i])
        plt.close(fig)
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, topology1.replace('_', ' ').title(), 
                   rotation=90, ha='center', va='center', 
                   transform=axes[0, 0].transAxes, fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, topology2.replace('_', ' ').title(), 
                   rotation=90, ha='center', va='center', 
                   transform=axes[1, 0].transAxes, fontsize=12, fontweight='bold')
    
    # Add column labels
    col_labels = ['Connection Density', 'Structural Patterns', 'Information Flow']
    for i, label in enumerate(col_labels):
        axes[0, i].text(0.5, -0.15, label, ha='center', va='top', 
                       transform=axes[0, i].transAxes, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison
    output_file = f"comparison_{topology1}_vs_{topology2}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison saved to: {output_file}")
    
    # Print metric comparison
    print("\\n📊 Metric Comparison:")
    print(f"{'Metric':<25} {topology1:<20} {topology2}")
    print("-" * 70)
    
    metrics1 = network1['metrics']
    metrics2 = network2['metrics']
    
    for metric in ['num_nodes', 'num_edges', 'density', 'average_degree', 'average_clustering']:
        if metric in metrics1 and metric in metrics2:
            val1 = f"{metrics1[metric]:.4f}" if isinstance(metrics1[metric], float) else str(metrics1[metric])
            val2 = f"{metrics2[metric]:.4f}" if isinstance(metrics2[metric], float) else str(metrics2[metric])
            print(f"{metric:<25} {val1:<20} {val2}")


def _copy_figure_to_subplot(source_fig, target_ax):
    """Helper function to copy figure content to subplot."""
    source_ax = source_fig.get_axes()[0]
    
    # Clear target axis
    target_ax.clear()
    
    # Copy all elements from source to target
    for element in source_ax.get_children():
        if hasattr(element, 'get_data'):
            # Line plots
            x_data, y_data = element.get_data()
            target_ax.plot(x_data, y_data, 
                         color=element.get_color(),
                         linewidth=element.get_linewidth(),
                         alpha=element.get_alpha())
        elif hasattr(element, 'get_offsets'):
            # Scatter plots
            offsets = element.get_offsets()
            if len(offsets) > 0:
                target_ax.scatter(offsets[:, 0], offsets[:, 1],
                                c=element.get_facecolors(),
                                s=element.get_sizes(),
                                edgecolors=element.get_edgecolors(),
                                linewidths=element.get_linewidths())
        elif hasattr(element, 'get_position'):
            # Text elements
            pos = element.get_position()
            text = element.get_text()
            target_ax.text(pos[0], pos[1], text,
                         fontsize=element.get_fontsize(),
                         fontweight=element.get_fontweight(),
                         ha=element.get_ha(),
                         va=element.get_va())
    
    # Copy axis properties
    target_ax.set_xlim(source_ax.get_xlim())
    target_ax.set_ylim(source_ax.get_ylim())
    target_ax.set_aspect('equal')
    target_ax.axis('off')
    
    # Copy title (smaller font)
    title = source_ax.get_title()
    if title:
        target_ax.set_title(title, fontsize=8, fontweight='bold', pad=5)


def main():
    """Main example function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Single topology analysis example")
    parser.add_argument('--topology', type=str, 
                       choices=['fully_connected', 'small_world', 'modular', 'hybrid', 
                               'standard_mlp_1layer', 'standard_mlp_3layers'],
                       default='small_world',
                       help='Topology to analyze')
    parser.add_argument('--size', type=int, default=64,
                       help='Network size')
    parser.add_argument('--compare', type=str,
                       help='Compare with another topology')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_two_topologies(args.topology, args.compare, args.size)
    else:
        analyze_single_topology(args.topology, args.size)


if __name__ == "__main__":
    main()
