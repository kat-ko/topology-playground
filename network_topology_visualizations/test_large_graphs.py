#!/usr/bin/env python3
"""
Test script for large graph visualizations
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from direct_network_generator import DirectNetworkGenerator
from visualization_engine import VisualizationEngine

def test_large_graphs():
    """Test visualization with large graphs."""
    print("Testing large graph visualizations...")
    
    # Generate networks
    generator = DirectNetworkGenerator(size=64, seed=42)
    networks = generator.generate_all_topologies()
    
    visualizer = VisualizationEngine()
    
    # Test each topology individually
    for name, network_data in networks.items():
        if network_data is None:
            continue
            
        graph = network_data['graph']
        print(f"\nTesting {name}: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        try:
            # Test connection density
            fig = visualizer.visualize_connection_density(graph, name)
            fig.savefig(f"test_{name}_density.png", dpi=100, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Density visualization saved")
            
        except Exception as e:
            print(f"  ✗ Error with {name}: {e}")
            import traceback
            traceback.print_exc()

def test_comparison_grid():
    """Test comparison grid creation."""
    print("\nTesting comparison grid creation...")
    
    try:
        from comparison_interface import ComparisonInterface
        
        interface = ComparisonInterface(size=64, seed=42, output_dir="test_large_output")
        networks = interface.generator.generate_all_topologies()
        
        print("Creating density comparison...")
        interface._generate_density_comparison(networks)
        print("✓ Density comparison created successfully")
        
    except Exception as e:
        print(f"✗ Error creating comparison grid: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_large_graphs()
    test_comparison_grid()
    print("\nTest completed!")

