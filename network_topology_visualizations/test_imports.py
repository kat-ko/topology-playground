#!/usr/bin/env python3
"""
Test script to debug import issues
"""

import sys
import os

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the main project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..')
print(f"Project root: {project_root}")
print(f"Project root exists: {os.path.exists(project_root)}")

sys.path.insert(0, project_root)
print(f"Updated Python path: {sys.path}")

# Check if topology directory exists
topology_dir = os.path.join(project_root, 'src', 'topologies')
print(f"Topology directory: {topology_dir}")
print(f"Topology directory exists: {os.path.exists(topology_dir)}")

if os.path.exists(topology_dir):
    print(f"Files in topology directory: {os.listdir(topology_dir)}")

try:
    from src.topologies.fully_connected import FullyConnectedTopology
    print("✓ Successfully imported FullyConnectedTopology")
except ImportError as e:
    print(f"✗ Failed to import FullyConnectedTopology: {e}")

try:
    from src.topologies.small_world import SmallWorldTopology
    print("✓ Successfully imported SmallWorldTopology")
except ImportError as e:
    print(f"✗ Failed to import SmallWorldTopology: {e}")

try:
    from src.topologies.modular import ModularTopology
    print("✓ Successfully imported ModularTopology")
except ImportError as e:
    print(f"✗ Failed to import ModularTopology: {e}")

try:
    from src.topologies.hybrid import HybridTopology
    print("✓ Successfully imported HybridTopology")
except ImportError as e:
    print(f"✗ Failed to import HybridTopology: {e}")

try:
    from src.topologies.standard_mlp import StandardMLPTopology
    print("✓ Successfully imported StandardMLPTopology")
except ImportError as e:
    print(f"✗ Failed to import StandardMLPTopology: {e}")

print("\nTesting network generation...")
try:
    # Test creating a topology
    topology = FullyConnectedTopology(size=16, seed=42)
    graph = topology.generate()
    print(f"✓ Successfully generated fully connected graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
except Exception as e:
    print(f"✗ Failed to generate graph: {e}")
