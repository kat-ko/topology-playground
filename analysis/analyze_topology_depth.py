#!/usr/bin/env python3
"""
Analyze Topology Depth: Compute longest path length and average path length for different topologies.
This helps understand the actual depth characteristics of each topology type.
"""

import networkx as nx
import numpy as np
import json
import os
from datetime import datetime

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology

def analyze_topology_depth(topology_type, size=73, num_layers=2, **topology_params):
    """Analyze depth characteristics of a topology."""
    print(f"🔍 Analyzing {topology_type} topology...")
    
    # Create topology
    if topology_type == 'fully_connected':
        topology = FullyConnectedTopology(
            size=size,
            num_layers=num_layers,
            inter_layer_prob=1.0,
            intra_layer_prob=1.0,
            seed=42
        )
    elif topology_type == 'small_world':
        topology = SmallWorldTopology(
            size=size,
            k=4,
            p=0.1,
            num_layers=num_layers,
            inter_layer_prob=0.1,
            seed=42
        )
    elif topology_type == 'modular':
        topology = ModularTopology(
            size=size,
            num_modules=4,
            inter_module_prob=0.1,
            intra_module_prob=0.3,
            num_layers=num_layers,
            inter_layer_prob=0.1,
            seed=42
        )
    elif topology_type == 'hybrid':
        topology = HybridTopology(
            size=size,
            num_modules=4,
            k=4,
            p=0.1,
            inter_module_prob=0.1,
            num_layers=num_layers,
            inter_layer_prob=0.1,
            seed=42
        )
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")
    
    # Generate graph
    graph = topology.generate()
    
    # Handle case where generate() returns a list of graphs
    if isinstance(graph, list):
        print(f"   • Generated {len(graph)} graphs (multi-layer)")
        # For now, use the first graph - in a real implementation you'd want to combine them
        graph = graph[0]
        print(f"   • Using first graph with {len(graph.edges())} edges")
    else:
        print(f"   • Graph generated with {len(graph.edges())} edges")
    
    # Convert to undirected for path analysis (if directed)
    if graph.is_directed():
        undirected_graph = graph.to_undirected()
        print(f"   • Converted directed graph to undirected for path analysis")
    else:
        undirected_graph = graph
    
    # Check if graph is connected
    if not nx.is_connected(undirected_graph):
        print(f"   ⚠️  Graph is not connected! Analyzing largest connected component.")
        largest_cc = max(nx.connected_components(undirected_graph), key=len)
        undirected_graph = undirected_graph.subgraph(largest_cc)
        print(f"   • Largest connected component has {len(largest_cc)} nodes")
    
    # Compute depth metrics
    print(f"   • Computing path lengths...")
    
    # Longest shortest path (diameter) - depth proxy
    try:
        diameter = nx.diameter(undirected_graph)
        print(f"   • Diameter (longest shortest path): {diameter}")
    except nx.NetworkXError as e:
        print(f"   ⚠️  Error computing diameter: {e}")
        diameter = None
    
    # Average shortest path length - efficiency proxy
    try:
        avg_path_length = nx.average_shortest_path_length(undirected_graph)
        print(f"   • Average shortest path length: {avg_path_length:.3f}")
    except nx.NetworkXError as e:
        print(f"   ⚠️  Error computing average path length: {e}")
        avg_path_length = None
    
    # Additional metrics
    try:
        # Eccentricity (longest shortest path from each node)
        eccentricity = nx.eccentricity(undirected_graph)
        max_eccentricity = max(eccentricity.values())
        min_eccentricity = min(eccentricity.values())
        avg_eccentricity = np.mean(list(eccentricity.values()))
        print(f"   • Eccentricity - Max: {max_eccentricity}, Min: {min_eccentricity}, Avg: {avg_eccentricity:.3f}")
    except Exception as e:
        print(f"   ⚠️  Error computing eccentricity: {e}")
        max_eccentricity = min_eccentricity = avg_eccentricity = None
    
    # Network density
    density = nx.density(undirected_graph)
    print(f"   • Network density: {density:.4f}")
    
    # Clustering coefficient
    try:
        clustering = nx.average_clustering(undirected_graph)
        print(f"   • Average clustering coefficient: {clustering:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error computing clustering: {e}")
        clustering = None
    
    # Node degree statistics
    degrees = [undirected_graph.degree(node) for node in undirected_graph.nodes()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)
    print(f"   • Node degrees - Avg: {avg_degree:.2f}, Max: {max_degree}, Min: {min_degree}")
    
    return {
        'topology_type': topology_type,
        'size': size,
        'num_layers': num_layers,
        'total_nodes': len(undirected_graph.nodes()),
        'total_edges': len(undirected_graph.edges()),
        'diameter': diameter,
        'avg_path_length': avg_path_length,
        'max_eccentricity': max_eccentricity,
        'min_eccentricity': min_eccentricity,
        'avg_eccentricity': avg_eccentricity,
        'density': density,
        'clustering': clustering,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree,
        'is_connected': nx.is_connected(undirected_graph) if undirected_graph == graph else False,
        'largest_cc_size': len(undirected_graph.nodes()) if undirected_graph != graph else size
    }

def analyze_all_topologies():
    """Analyze depth characteristics for all topology types."""
    print("=" * 80)
    print("🔍 TOPOLOGY DEPTH ANALYSIS")
    print("=" * 80)
    
    # Configuration
    size = 73  # Same as in debug script (6 input + 64 hidden + 3 output)
    num_layers = 2  # Default number of layers
    
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    results = {}
    
    for topology_type in topologies:
        print(f"\n{'='*60}")
        print(f"Analyzing: {topology_type.upper()}")
        print(f"{'='*60}")
        
        try:
            result = analyze_topology_depth(topology_type, size, num_layers)
            results[topology_type] = result
            print(f"✅ {topology_type} analysis completed")
        except Exception as e:
            print(f"❌ Error analyzing {topology_type}: {e}")
            results[topology_type] = {'error': str(e)}
    
    # Create summary
    print(f"\n{'='*80}")
    print("📊 DEPTH ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Create comparison table
    print(f"\n{'Topology':<15} {'Diameter':<10} {'Avg Path':<10} {'Density':<10} {'Clustering':<12} {'Avg Degree':<12}")
    print("-" * 80)
    
    for topology_type in topologies:
        if topology_type in results and 'error' not in results[topology_type]:
            r = results[topology_type]
            diameter = r.get('diameter', 'N/A')
            avg_path = f"{r.get('avg_path_length', 0):.3f}" if r.get('avg_path_length') is not None else 'N/A'
            density = f"{r.get('density', 0):.4f}"
            clustering = f"{r.get('clustering', 0):.4f}" if r.get('clustering') is not None else 'N/A'
            avg_degree = f"{r.get('avg_degree', 0):.2f}"
            
            print(f"{topology_type:<15} {diameter:<10} {avg_path:<10} {density:<10} {clustering:<12} {avg_degree:<12}")
        else:
            print(f"{topology_type:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/topology_depth_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(results_dir, 'depth_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to: {results_file}")
    
    # Create summary file
    summary_file = os.path.join(results_dir, 'depth_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("TOPOLOGY DEPTH ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network Size: {size} nodes\n")
        f.write(f"Number of Layers: {num_layers}\n\n")
        
        f.write("KEY METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write("• Diameter: Longest shortest path (depth proxy)\n")
        f.write("• Avg Path Length: Average shortest path (efficiency proxy)\n")
        f.write("• Density: Ratio of actual to possible connections\n")
        f.write("• Clustering: Local connectivity measure\n")
        f.write("• Avg Degree: Average number of connections per node\n\n")
        
        f.write("COMPARISON TABLE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Topology':<15} {'Diameter':<10} {'Avg Path':<10} {'Density':<10} {'Clustering':<12} {'Avg Degree':<12}\n")
        f.write("-" * 80 + "\n")
        
        for topology_type in topologies:
            if topology_type in results and 'error' not in results[topology_type]:
                r = results[topology_type]
                diameter = r.get('diameter', 'N/A')
                avg_path = f"{r.get('avg_path_length', 0):.3f}" if r.get('avg_path_length') is not None else 'N/A'
                density = f"{r.get('density', 0):.4f}"
                clustering = f"{r.get('clustering', 0):.4f}" if r.get('clustering') is not None else 'N/A'
                avg_degree = f"{r.get('avg_degree', 0):.2f}"
                
                f.write(f"{topology_type:<15} {diameter:<10} {avg_path:<10} {density:<10} {clustering:<12} {avg_degree:<12}\n")
            else:
                f.write(f"{topology_type:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<12}\n")
        
        f.write("\n\nRECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("• For non-fully-connected topologies, 'layers' concept is artificial\n")
        f.write("• Use diameter as depth proxy for comparison\n")
        f.write("• Use average path length as efficiency measure\n")
        f.write("• Consider only manipulating layers in fully-connected topology\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return results

if __name__ == "__main__":
    results = analyze_all_topologies()
    print(f"\n🎉 Analysis completed! Check the results directory for detailed files.") 