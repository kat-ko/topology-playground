#!/usr/bin/env python3
"""
Quick Comparison Example

This example demonstrates how to quickly generate and compare
network topologies using the visualization system.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.comparison_interface import ComparisonInterface


def main():
    """Generate a quick comparison of all topologies."""
    print("🚀 Quick Network Topology Comparison")
    print("=" * 40)
    
    # Create interface with smaller size for quick demo
    interface = ComparisonInterface(
        size=32,  # Smaller size for faster generation
        seed=42,
        output_dir="quick_demo_results"
    )
    
    # Generate only comparison grids (faster)
    print("Generating comparison grids...")
    interface.generate_comparison_grid("all")
    
    print("✅ Quick comparison complete!")
    print("📁 Results saved to: quick_demo_results/")
    print("\\n🎯 Check the following files:")
    print("  • connection_density_comparison.png")
    print("  • structural_patterns_comparison.png") 
    print("  • information_flow_comparison.png")


if __name__ == "__main__":
    main()
