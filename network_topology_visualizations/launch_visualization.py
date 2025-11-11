#!/usr/bin/env python3
"""
Launch script for network topology visualizations.
This script runs from the project root to ensure proper imports.
"""

import os
import sys
import subprocess

def main():
    """Launch the visualization system from the project root."""
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Change to project root
    os.chdir(project_root)
    
    # Get the demo script path
    demo_script = os.path.join(current_dir, 'demo_visualize_topologies.py')
    
    # Run the demo script with all arguments
    cmd = [sys.executable, demo_script] + sys.argv[1:]
    
    print(f"Running from project root: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Execute the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
