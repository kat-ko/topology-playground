"""
Network Topology Visualization System

This package provides comprehensive visualization tools for comparing different
neural network topologies at size 64, including:
- Fully Connected
- Small World
- Modular
- Hybrid
- Standard MLP (1 layer)
- Standard MLP (3 layers)

The visualization system provides three different approaches:
1. Connection Density Visualization
2. Structural Pattern Visualization  
3. Information Flow Visualization
"""

__version__ = "1.0.0"
__author__ = "Topology Playground Team"

from .src.network_generator import NetworkGenerator
from .src.visualization_engine import VisualizationEngine
from .src.comparison_interface import ComparisonInterface

__all__ = [
    'NetworkGenerator',
    'VisualizationEngine', 
    'ComparisonInterface'
]
