#!/usr/bin/env python3
"""
Debug script to isolate visualization issues
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.direct_network_generator import DirectNetworkGenerator
from src.visualization_engine import VisualizationEngine

def test_single_visualization():
    """Test a single visualization."""
    print("Testing single visualization...")
    
    # Generate a small network
    generator = DirectNetworkGenerator(size=16, seed=42)
    networks = generator.generate_all_topologies()
    
    # Test each visualization type
    visualizer = VisualizationEngine()
    
    for name, network_data in networks.items():
        if network_data is None:
            continue
            
        print(f"Testing {name}...")
        graph = network_data['graph']
        
        try:
            # Test connection density
            fig1 = visualizer.visualize_connection_density(graph, name)
            fig1.savefig(f"debug_{name}_density.png", dpi=100, bbox_inches='tight')
            plt.close(fig1)
            print(f"  ✓ Density visualization saved")
            
            # Test structural patterns
            fig2 = visualizer.visualize_structural_patterns(graph, name)
            fig2.savefig(f"debug_{name}_structural.png", dpi=100, bbox_inches='tight')
            plt.close(fig2)
            print(f"  ✓ Structural visualization saved")
            
            # Test information flow
            fig3 = visualizer.visualize_information_flow(graph, name)
            fig3.savefig(f"debug_{name}_flow.png", dpi=100, bbox_inches='tight')
            plt.close(fig3)
            print(f"  ✓ Flow visualization saved")
            
        except Exception as e:
            print(f"  ✗ Error with {name}: {e}")
            import traceback
            traceback.print_exc()

def test_comparison_grid():
    """Test comparison grid creation."""
    print("\nTesting comparison grid...")
    
    try:
        from src.comparison_interface import ComparisonInterface
        
        interface = ComparisonInterface(size=16, seed=42, output_dir="debug_output")
        interface.generate_comparison_grid("density")
        print("✓ Comparison grid created successfully")
        
    except Exception as e:
        print(f"✗ Error creating comparison grid: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_visualization()
    test_comparison_grid()
    print("\nDebug test completed!")
