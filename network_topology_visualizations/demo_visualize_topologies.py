#!/usr/bin/env python3
"""
Network Topology Visualization Demo

This script demonstrates the complete visualization system for comparing
all network topologies at size 64. It generates:

1. Side-by-side comparison grids for all three visualization approaches
2. Individual high-resolution visualizations for each topology
3. Summary report with metrics and statistics
4. Raw graph data exports

Usage:
    python demo_visualize_topologies.py [--size SIZE] [--seed SEED] [--output-dir DIR]
"""

import argparse
import sys
import os
from datetime import datetime

# Add the visualization src directory to the path
viz_src_dir = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, viz_src_dir)

from src.comparison_interface import ComparisonInterface


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive network topology visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_visualize_topologies.py
  python demo_visualize_topologies.py --size 128 --seed 123
  python demo_visualize_topologies.py --output-dir my_results
        """
    )
    
    parser.add_argument(
        '--size', 
        type=int, 
        default=64,
        help='Network size (default: 64)'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='topology_visualization_results',
        help='Output directory for results (default: topology_visualization_results)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Generate only comparison grids (skip individual visualizations)'
    )
    
    parser.add_argument(
        '--type',
        choices=['density', 'structural', 'flow', 'all'],
        default='all',
        help='Type of visualization to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    print("🌐 Network Topology Visualization System")
    print("=" * 50)
    print(f"Network Size: {args.size}")
    print(f"Random Seed: {args.seed}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Visualization Type: {args.type}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # Initialize comparison interface
        interface = ComparisonInterface(
            size=args.size,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        if args.quick:
            # Quick mode: only comparison grids
            print("🚀 Running in QUICK mode (comparison grids only)...")
            interface.generate_comparison_grid(args.type)
        else:
            # Full mode: complete analysis
            print("🚀 Running FULL analysis...")
            interface.run_full_comparison()
        
        print("\\n✅ SUCCESS! All visualizations completed successfully.")
        print(f"📁 Results saved to: {args.output_dir}/")
        
        # List generated files
        print("\\n📋 Generated Files:")
        if os.path.exists(args.output_dir):
            for root, dirs, files in os.walk(args.output_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), args.output_dir)
                    print(f"  • {rel_path}")
        
        print("\\n🎯 Next Steps:")
        print("  1. Open the comparison PNG files to see side-by-side comparisons")
        print("  2. Check individual topology folders for detailed visualizations")
        print("  3. Read the summary report for quantitative analysis")
        print("  4. Use the exported graph files for further analysis")
        
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
        print("\\n🔧 Troubleshooting:")
        print("  1. Make sure you're running from the correct directory")
        print("  2. Check that all dependencies are installed")
        print("  3. Verify the topology classes are accessible")
        sys.exit(1)


def test_individual_components():
    """Test individual components of the visualization system."""
    print("\\n🧪 Testing individual components...")
    
    try:
        # Test network generator
        print("  Testing network generator...")
        from src.direct_network_generator import DirectNetworkGenerator as NetworkGenerator
        generator = NetworkGenerator(size=16, seed=42)  # Smaller size for testing
        networks = generator.generate_all_topologies()
        print(f"    ✓ Generated {len([n for n in networks.values() if n is not None])} networks")
        
        # Test visualization engine
        print("  Testing visualization engine...")
        from src.visualization_engine import VisualizationEngine
        import networkx as nx
        
        engine = VisualizationEngine()
        test_graph = nx.erdos_renyi_graph(10, 0.3)
        
        fig1 = engine.visualize_connection_density(test_graph, "test")
        fig2 = engine.visualize_structural_patterns(test_graph, "test")
        fig3 = engine.visualize_information_flow(test_graph, "test")
        
        print("    ✓ All visualization methods working")
        
        # Test comparison interface
        print("  Testing comparison interface...")
        interface = ComparisonInterface(size=16, seed=42, output_dir="test_output")
        interface.generate_comparison_grid("density")
        print("    ✓ Comparison interface working")
        
        print("\\n✅ All components tested successfully!")
        
    except Exception as e:
        print(f"\\n❌ Component test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    # Check if user wants to run tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_individual_components()
    else:
        main()
