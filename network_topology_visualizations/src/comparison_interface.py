"""
Comparison Interface for Network Topology Visualizations

This module provides a unified interface for comparing all network topologies
side-by-side using the three visualization approaches.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Any
import os
import warnings
warnings.filterwarnings('ignore')

from .direct_network_generator import DirectNetworkGenerator as NetworkGenerator
from .visualization_engine import VisualizationEngine


class ComparisonInterface:
    """Interface for comparing network topologies."""
    
    def __init__(self, size: int = 64, seed: int = 42, output_dir: str = "outputs"):
        """
        Initialize the comparison interface.
        
        Args:
            size: Network size
            seed: Random seed for reproducibility
            output_dir: Directory for output files
        """
        self.size = size
        self.seed = seed
        self.output_dir = output_dir
        
        # Initialize components
        self.generator = NetworkGenerator(size=size, seed=seed)
        self.visualizer = VisualizationEngine(figsize=(10, 8))
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Topology display names
        self.display_names = {
            'fully_connected': 'Fully Connected',
            'small_world': 'Small World',
            'modular': 'Modular',
            'hybrid': 'Hybrid',
            'standard_mlp_1layer': 'Standard MLP (1 Layer)',
            'standard_mlp_3layers': 'Standard MLP (3 Layers)'
        }
    
    def generate_comparison_grid(self, visualization_type: str = "all") -> None:
        """
        Generate comparison grids for all topologies.
        
        Args:
            visualization_type: Type of visualization ("density", "structural", "flow", or "all")
        """
        print(f"Generating {visualization_type} comparison for size {self.size}...")
        
        # Generate all networks
        networks = self.generator.generate_all_topologies()
        
        if visualization_type == "all":
            self._generate_all_comparisons(networks)
        elif visualization_type == "density":
            self._generate_density_comparison(networks)
        elif visualization_type == "structural":
            self._generate_structural_comparison(networks)
        elif visualization_type == "flow":
            self._generate_flow_comparison(networks)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        print(f"✓ Comparison visualizations saved to {self.output_dir}/")
    
    def _generate_all_comparisons(self, networks: Dict[str, Any]) -> None:
        """Generate all three types of comparisons."""
        self._generate_density_comparison(networks)
        self._generate_structural_comparison(networks)
        self._generate_flow_comparison(networks)
    
    def _generate_density_comparison(self, networks: Dict[str, Any]) -> None:
        """Generate connection density comparison."""
        print("  Creating connection density comparison...")
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.suptitle('Network Topology Comparison - Connection Density', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        topology_names = list(self.display_names.keys())
        
        for i, topology_name in enumerate(topology_names):
            row = i // 3
            col = i % 3
            
            if networks[topology_name] is None:
                axes[row, col].text(0.5, 0.5, f"{self.display_names[topology_name]}\\n(Failed to generate)",
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=12, color='red')
                axes[row, col].axis('off')
                continue
            
            graph = networks[topology_name]['graph']
            metrics = networks[topology_name]['metrics']
            
            # Create individual visualization
            fig_single = self.visualizer.visualize_connection_density(
                graph, topology_name, 
                title=f"{self.display_names[topology_name]}\\n"
                      f"Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}\\n"
                      f"Density: {metrics['density']:.3f}"
            )
            
            # Copy to subplot
            self._copy_figure_to_subplot(fig_single, axes[row, col])
            plt.close(fig_single)
        
        # Hide unused subplots
        if len(topology_names) < 6:
            for i in range(len(topology_names), 6):
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'connection_density_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_structural_comparison(self, networks: Dict[str, Any]) -> None:
        """Generate structural pattern comparison."""
        print("  Creating structural pattern comparison...")
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.suptitle('Network Topology Comparison - Structural Patterns', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        topology_names = list(self.display_names.keys())
        
        for i, topology_name in enumerate(topology_names):
            row = i // 3
            col = i % 3
            
            if networks[topology_name] is None:
                axes[row, col].text(0.5, 0.5, f"{self.display_names[topology_name]}\\n(Failed to generate)",
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=12, color='red')
                axes[row, col].axis('off')
                continue
            
            graph = networks[topology_name]['graph']
            metrics = networks[topology_name]['metrics']
            
            # Create individual visualization
            fig_single = self.visualizer.visualize_structural_patterns(
                graph, topology_name,
                title=f"{self.display_names[topology_name]}\\n"
                      f"Clustering: {metrics.get('average_clustering', 0):.3f}\\n"
                      f"Diameter: {metrics.get('diameter', 0)}"
            )
            
            # Copy to subplot
            self._copy_figure_to_subplot(fig_single, axes[row, col])
            plt.close(fig_single)
        
        # Hide unused subplots
        if len(topology_names) < 6:
            for i in range(len(topology_names), 6):
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'structural_patterns_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_flow_comparison(self, networks: Dict[str, Any]) -> None:
        """Generate information flow comparison."""
        print("  Creating information flow comparison...")
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.suptitle('Network Topology Comparison - Information Flow', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        topology_names = list(self.display_names.keys())
        
        for i, topology_name in enumerate(topology_names):
            row = i // 3
            col = i % 3
            
            if networks[topology_name] is None:
                axes[row, col].text(0.5, 0.5, f"{self.display_names[topology_name]}\\n(Failed to generate)",
                                  ha='center', va='center', transform=axes[row, col].transAxes,
                                  fontsize=12, color='red')
                axes[row, col].axis('off')
                continue
            
            graph = networks[topology_name]['graph']
            metrics = networks[topology_name]['metrics']
            
            # Create individual visualization
            fig_single = self.visualizer.visualize_information_flow(
                graph, topology_name,
                title=f"{self.display_names[topology_name]}\\n"
                      f"Directed: {metrics['is_directed']}\\n"
                      f"Connected: {metrics['is_connected']}"
            )
            
            # Copy to subplot
            self._copy_figure_to_subplot(fig_single, axes[row, col])
            plt.close(fig_single)
        
        # Hide unused subplots
        if len(topology_names) < 6:
            for i in range(len(topology_names), 6):
                row = i // 3
                col = i % 3
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'information_flow_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _copy_figure_to_subplot(self, source_fig, target_ax):
        """Copy content from a figure to a subplot axis."""
        # Get the source axes
        source_ax = source_fig.get_axes()[0]
        
        # Clear target axis
        target_ax.clear()
        
        # Copy all elements from source to target
        for element in source_ax.get_children():
            if hasattr(element, 'get_data'):
                # Line plots
                try:
                    x_data, y_data = element.get_data()
                    target_ax.plot(x_data, y_data, 
                                 color=element.get_color(),
                                 linewidth=element.get_linewidth(),
                                 alpha=element.get_alpha())
                except:
                    pass  # Skip elements that can't be copied
            elif hasattr(element, 'get_offsets'):
                # Scatter plots
                try:
                    offsets = element.get_offsets()
                    if len(offsets) > 0:
                        target_ax.scatter(offsets[:, 0], offsets[:, 1],
                                        c=element.get_facecolors(),
                                        s=element.get_sizes(),
                                        edgecolors=element.get_edgecolors(),
                                        linewidths=element.get_linewidths())
                except:
                    pass  # Skip elements that can't be copied
            elif hasattr(element, 'get_position') and hasattr(element, 'get_text'):
                # Text elements
                try:
                    pos = element.get_position()
                    text = element.get_text()
                    target_ax.text(pos[0], pos[1], text,
                                 fontsize=element.get_fontsize(),
                                 fontweight=element.get_fontweight(),
                                 ha=element.get_ha(),
                                 va=element.get_va())
                except:
                    pass  # Skip elements that don't have text
        
        # Copy axis properties
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())
        target_ax.set_aspect('equal')
        target_ax.axis('off')
        
        # Copy title
        title = source_ax.get_title()
        target_ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    
    def generate_individual_visualizations(self, networks: Dict[str, Any]) -> None:
        """Generate individual high-resolution visualizations for each topology."""
        print("Creating individual topology visualizations...")
        
        for topology_name, network_data in networks.items():
            if network_data is None:
                continue
                
            print(f"  Processing {topology_name}...")
            
            graph = network_data['graph']
            metrics = network_data['metrics']
            
            # Create individual directory for this topology
            topology_dir = os.path.join(self.output_dir, topology_name)
            os.makedirs(topology_dir, exist_ok=True)
            
            # Generate all three visualization types
            fig1 = self.visualizer.visualize_connection_density(
                graph, topology_name,
                title=f"{self.display_names[topology_name]} - Connection Density\\n"
                      f"Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}, "
                      f"Density: {metrics['density']:.3f}, Clustering: {metrics.get('average_clustering', 0):.3f}"
            )
            fig1.savefig(os.path.join(topology_dir, 'connection_density.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            fig2 = self.visualizer.visualize_structural_patterns(
                graph, topology_name,
                title=f"{self.display_names[topology_name]} - Structural Patterns\\n"
                      f"Modules/Layers highlighted with structural connections"
            )
            fig2.savefig(os.path.join(topology_dir, 'structural_patterns.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            fig3 = self.visualizer.visualize_information_flow(
                graph, topology_name,
                title=f"{self.display_names[topology_name]} - Information Flow\\n"
                      f"Directed: {metrics['is_directed']}, Connected: {metrics['is_connected']}"
            )
            fig3.savefig(os.path.join(topology_dir, 'information_flow.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close(fig3)
    
    def generate_summary_report(self, networks: Dict[str, Any]) -> None:
        """Generate a summary report with metrics and statistics."""
        print("Generating summary report...")
        
        report_path = os.path.join(self.output_dir, 'topology_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("NETWORK TOPOLOGY COMPARISON SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Network Size: {self.size}\\n")
            f.write(f"Random Seed: {self.seed}\\n")
            f.write(f"Generated: {np.datetime64('now')}\\n\\n")
            
            # Topology information
            topology_info = self.generator.get_topology_info()
            
            for topology_name, network_data in networks.items():
                f.write(f"{self.display_names[topology_name].upper()}\\n")
                f.write("-" * len(self.display_names[topology_name]) + "\\n")
                
                if network_data is None:
                    f.write("Status: FAILED TO GENERATE\\n\\n")
                    continue
                
                # Get topology description
                description = topology_info[topology_name]['description']
                f.write(f"Description: {description}\\n")
                
                # Get metrics
                metrics = network_data['metrics']
                f.write(f"Nodes: {metrics['num_nodes']}\\n")
                f.write(f"Edges: {metrics['num_edges']}\\n")
                f.write(f"Density: {metrics['density']:.4f}\\n")
                f.write(f"Average Degree: {metrics['average_degree']:.2f}\\n")
                f.write(f"Directed: {metrics['is_directed']}\\n")
                f.write(f"Connected: {metrics['is_connected']}\\n")
                
                if 'average_clustering' in metrics:
                    f.write(f"Average Clustering: {metrics['average_clustering']:.4f}\\n")
                if 'diameter' in metrics:
                    f.write(f"Diameter: {metrics['diameter']}\\n")
                if 'average_shortest_path' in metrics:
                    f.write(f"Average Shortest Path: {metrics['average_shortest_path']:.4f}\\n")
                
                f.write("\\n")
            
            # Comparative analysis
            f.write("COMPARATIVE ANALYSIS\\n")
            f.write("=" * 20 + "\\n\\n")
            
            # Find topology with highest/lowest metrics
            valid_networks = {k: v for k, v in networks.items() if v is not None}
            
            if valid_networks:
                densities = {k: v['metrics']['density'] for k, v in valid_networks.items()}
                clustering = {k: v['metrics'].get('average_clustering', 0) for k, v in valid_networks.items()}
                
                max_density_topology = max(densities, key=densities.get)
                min_density_topology = min(densities, key=densities.get)
                max_clustering_topology = max(clustering, key=clustering.get)
                
                f.write(f"Highest Density: {self.display_names[max_density_topology]} "
                       f"({densities[max_density_topology]:.4f})\\n")
                f.write(f"Lowest Density: {self.display_names[min_density_topology]} "
                       f"({densities[min_density_topology]:.4f})\\n")
                f.write(f"Highest Clustering: {self.display_names[max_clustering_topology]} "
                       f"({clustering[max_clustering_topology]:.4f})\\n")
        
        print(f"✓ Summary report saved to {report_path}")
    
    def run_full_comparison(self) -> None:
        """Run the complete comparison analysis."""
        print("Starting full network topology comparison...")
        print(f"Network size: {self.size}, Seed: {self.seed}")
        print("=" * 60)
        
        # Generate all networks
        networks = self.generator.generate_all_topologies()
        
        # Generate comparison grids
        self.generate_comparison_grid("all")
        
        # Generate individual visualizations
        self.generate_individual_visualizations(networks)
        
        # Generate summary report
        self.generate_summary_report(networks)
        
        # Export raw data
        self.generator.export_graphs(networks, self.output_dir)
        
        print("\\n" + "=" * 60)
        print("COMPARISON COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}/")
        print("=" * 60)


if __name__ == "__main__":
    # Test the comparison interface
    print("Testing Comparison Interface...")
    
    interface = ComparisonInterface(size=64, seed=42, output_dir="test_outputs")
    interface.run_full_comparison()
    
    print("✓ Comparison interface test completed!")
