#!/usr/bin/env python3
"""
Wrapper script to run the visualization system from the correct directory
"""

import os
import sys
import subprocess

def main():
    """Run the visualization system from the project root."""
    # Get the project root directory (topology-playground)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Change to project root
    os.chdir(project_root)
    
    # Add the visualization directory to Python path
    viz_dir = os.path.join(project_root, 'network_topology_visualizations')
    if viz_dir not in sys.path:
        sys.path.insert(0, viz_dir)
    
    # Import and run the demo
    try:
        from network_topology_visualizations.demo_visualize_topologies import main as demo_main
        demo_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying alternative approach...")
        
        # Alternative: run as subprocess
        demo_script = os.path.join(viz_dir, 'demo_visualize_topologies.py')
        cmd = [sys.executable, demo_script] + sys.argv[1:]
        
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
