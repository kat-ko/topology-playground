#!/usr/bin/env python3
"""
Modular Network Structure Analysis Script

This script reuses the training infrastructure to create and analyze modular networks
with different module counts (m=4 vs m=8) without actually training them.
Focuses on structural analysis and visualization of network differences.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.standard_mlp import StandardMLPTopology
from src.networks.ffn import FeedForwardNetwork
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.device_manager import get_device_manager, get_device_info

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

class ModularNetworkAnalyzer:
    """
    Analyzes and compares modular networks with different module counts.
    Reuses training infrastructure but skips actual training.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = get_device_manager().get_device()
        self.results = {}
        
        # Create output directory
        self.output_dir = f"modular_analysis_{args.environment}_{args.size}_{int(time.time())}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🔬 Modular Network Analyzer initialized")
        print(f"   Environment: {args.environment}")
        print(f"   Network size: {args.size}")
        print(f"   Output directory: {self.output_dir}")
    
    def create_networks(self):
        """Create modular networks with m=4 and m=8."""
        print("\n🏗️  Creating modular networks...")
        
        # Get task dimensions
        env = gym.make(self.args.environment)
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.close()
        
        # Network configurations to compare
        configs = {
            'm4': {'num_modules': 4, 'inter_module_prob': 0.2, 'intra_module_prob': 0.8},
            'm8': {'num_modules': 8, 'inter_module_prob': 0.2, 'intra_module_prob': 0.8}
        }
        
        networks = {}
        
        for config_name, config in configs.items():
            print(f"   Creating {config_name} network...")
            
            # Create modular topology
            topology = ModularTopology(
                size=self.args.size,
                num_modules=config['num_modules'],
                inter_module_prob=config['inter_module_prob'],
                intra_module_prob=config['intra_module_prob'],
                seed=self.args.seed
            )
            
            # Create network using the same approach as training scripts
            # Generate the network graph
            graph = topology.generate(input_dim=obs_dim, output_dim=action_dim)
            
            # Define input and output nodes
            input_nodes = list(range(obs_dim))
            output_nodes = list(range(obs_dim + self.args.size, obs_dim + self.args.size + action_dim))
            
            # Create network parameters
            network_params = {
                'learning_rate': 0.001, 
                'activation': self.args.activation,
                'seed': self.args.seed
            }
            
            # Create FeedForwardNetwork
            network = FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
            
            networks[config_name] = {
                'network': network,
                'topology': topology,
                'config': config
            }
        
        self.networks = networks
        print(f"   ✅ Created {len(networks)} network configurations")
        
        return networks
    
    def analyze_network_structure(self):
        """Analyze structural properties of the networks."""
        print("\n📊 Analyzing network structures...")
        
        for config_name, network_data in self.networks.items():
            print(f"   Analyzing {config_name}...")
            
            topology = network_data['topology']
            network = network_data['network']
            
            # Get network graph
            G = topology.generate()
            
            # Calculate structural metrics
            metrics = self._calculate_network_metrics(G, topology)
            
            # Store results
            self.results[config_name] = {
                'metrics': metrics,
                'topology': topology,
                'network': network,
                'graph': G
            }
            
            print(f"      Nodes: {metrics['num_nodes']}")
            print(f"      Edges: {metrics['num_edges']}")
            print(f"      Density: {metrics['density']:.4f}")
            print(f"      Modularity: {metrics['modularity']:.4f}")
    
    def _calculate_network_metrics(self, G, topology):
        """Calculate comprehensive network metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # Modularity metrics
        if hasattr(topology, 'module_assignments'):
            # Calculate modularity using the actual module assignments
            communities = {}
            for node, module in topology.module_assignments.items():
                if module not in communities:
                    communities[module] = []
                communities[module].append(node)
            
            community_list = list(communities.values())
            metrics['modularity'] = nx.community.modularity(G, community_list)
            metrics['num_modules'] = len(communities)
            metrics['module_sizes'] = [len(module) for module in community_list]
        else:
            metrics['modularity'] = 0.0
            metrics['num_modules'] = 0
            metrics['module_sizes'] = []
        
        # Connectivity metrics
        # Convert to undirected for metrics that don't support directed graphs
        G_undirected = G.to_undirected() if G.is_directed() else G
        metrics['avg_clustering'] = nx.average_clustering(G_undirected)
        metrics['avg_path_length'] = nx.average_shortest_path_length(G_undirected) if nx.is_connected(G_undirected) else float('inf')
        
        # Degree metrics
        degrees = [G.degree(n) for n in G.nodes()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['degree_std'] = np.std(degrees)
        metrics['max_degree'] = max(degrees)
        metrics['min_degree'] = min(degrees)
        
        # Module-specific metrics
        if hasattr(topology, 'module_assignments'):
            metrics['intra_module_connections'] = self._count_intra_module_connections(G, topology)
            metrics['inter_module_connections'] = self._count_inter_module_connections(G, topology)
            metrics['module_connectivity'] = self._calculate_module_connectivity(G, topology)
        
        return metrics
    
    def _count_intra_module_connections(self, G, topology):
        """Count connections within modules."""
        intra_connections = 0
        for edge in G.edges():
            node1, node2 = edge
            if topology.module_assignments.get(node1) == topology.module_assignments.get(node2):
                intra_connections += 1
        return intra_connections
    
    def _count_inter_module_connections(self, G, topology):
        """Count connections between modules."""
        inter_connections = 0
        for edge in G.edges():
            node1, node2 = edge
            if topology.module_assignments.get(node1) != topology.module_assignments.get(node2):
                inter_connections += 1
        return inter_connections
    
    def _calculate_module_connectivity(self, G, topology):
        """Calculate connectivity between modules."""
        modules = {}
        for node, module in topology.module_assignments.items():
            if module not in modules:
                modules[module] = []
            modules[module].append(node)
        
        connectivity_matrix = np.zeros((len(modules), len(modules)))
        
        for i, module1 in enumerate(modules.values()):
            for j, module2 in enumerate(modules.values()):
                if i != j:
                    connections = 0
                    for node1 in module1:
                        for node2 in module2:
                            if G.has_edge(node1, node2):
                                connections += 1
                    connectivity_matrix[i, j] = connections
        
        return connectivity_matrix
    
    def visualize_networks(self):
        """Create visualizations comparing the networks."""
        print("\n🎨 Creating visualizations...")
        
        # 1. Network structure comparison
        self._plot_network_structures()
        
        # 2. Metrics comparison
        self._plot_metrics_comparison()
        
        # 3. Module connectivity matrices
        self._plot_module_connectivity()
        
        # 4. Degree distribution comparison
        self._plot_degree_distributions()
        
        print(f"   ✅ Visualizations saved to {self.output_dir}/")
    
    def _plot_network_structures(self):
        """Plot side-by-side network structures."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        for i, (config_name, data) in enumerate(self.results.items()):
            G = data['graph']
            topology = data['topology']
            
            # Create layout
            pos = nx.spring_layout(G, seed=42)
            
            # Color nodes by module
            if hasattr(topology, 'module_assignments'):
                node_colors = [topology.module_assignments[node] for node in G.nodes()]
                cmap = plt.cm.tab10
            else:
                node_colors = 'lightblue'
                cmap = None
            
            # Draw network
            nx.draw(G, pos, 
                   node_color=node_colors,
                   cmap=cmap,
                   node_size=50,
                   edge_color='gray',
                   alpha=0.7,
                   ax=axes[i])
            
            axes[i].set_title(f'Modular Network (m={data["metrics"]["num_modules"]})', 
                            fontsize=16, fontweight='bold')
            axes[i].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/network_structures.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self):
        """Plot comparison of network metrics."""
        metrics_to_plot = [
            'density', 'modularity', 'avg_clustering', 'avg_degree',
            'intra_module_connections', 'inter_module_connections'
        ]
        
        # Prepare data
        configs = list(self.results.keys())
        values = {metric: [] for metric in metrics_to_plot}
        
        for config_name in configs:
            for metric in metrics_to_plot:
                if metric in self.results[config_name]['metrics']:
                    values[metric].append(self.results[config_name]['metrics'][metric])
                else:
                    values[metric].append(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            axes[i].bar(configs, values[metric], color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
            axes[i].set_ylabel('Value')
            
            # Add value labels on bars
            for j, v in enumerate(values[metric]):
                axes[i].text(j, v + max(values[metric]) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_module_connectivity(self):
        """Plot module connectivity matrices."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, (config_name, data) in enumerate(self.results.items()):
            if 'module_connectivity' in data['metrics']:
                connectivity_matrix = data['metrics']['module_connectivity']
                
                im = axes[i].imshow(connectivity_matrix, cmap='Blues', aspect='auto')
                axes[i].set_title(f'Module Connectivity (m={data["metrics"]["num_modules"]})', 
                                fontweight='bold')
                axes[i].set_xlabel('Target Module')
                axes[i].set_ylabel('Source Module')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[i])
                
                # Add text annotations
                for row in range(connectivity_matrix.shape[0]):
                    for col in range(connectivity_matrix.shape[1]):
                        text = axes[i].text(col, row, int(connectivity_matrix[row, col]),
                                          ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/module_connectivity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_degree_distributions(self):
        """Plot degree distributions for both networks."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for i, (config_name, data) in enumerate(self.results.items()):
            G = data['graph']
            degrees = [G.degree(n) for n in G.nodes()]
            
            axes[i].hist(degrees, bins=20, alpha=0.7, color=['skyblue', 'lightcoral'][i])
            axes[i].set_title(f'Degree Distribution (m={data["metrics"]["num_modules"]})', 
                            fontweight='bold')
            axes[i].set_xlabel('Degree')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/degree_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self):
        """Export analysis results to files."""
        print("\n💾 Exporting results...")
        
        # Export metrics to CSV
        metrics_data = []
        for config_name, data in self.results.items():
            row = {'configuration': config_name}
            row.update(data['metrics'])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{self.output_dir}/network_metrics.csv", index=False)
        
        # Export detailed results to JSON
        detailed_results = {}
        for config_name, data in self.results.items():
            detailed_results[config_name] = {
                'metrics': data['metrics'],
                'config': data['topology'].__dict__ if hasattr(data['topology'], '__dict__') else {}
            }
        
        with open(f"{self.output_dir}/detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"   ✅ Results exported to {self.output_dir}/")
    
    def print_summary(self):
        """Print a summary of the analysis."""
        print("\n" + "="*80)
        print("📋 MODULAR NETWORK ANALYSIS SUMMARY")
        print("="*80)
        
        for config_name, data in self.results.items():
            metrics = data['metrics']
            print(f"\n🔧 {config_name.upper()} Configuration:")
            print(f"   Modules: {metrics['num_modules']}")
            print(f"   Module sizes: {metrics['module_sizes']}")
            print(f"   Total nodes: {metrics['num_nodes']}")
            print(f"   Total edges: {metrics['num_edges']}")
            print(f"   Density: {metrics['density']:.4f}")
            print(f"   Modularity: {metrics['modularity']:.4f}")
            print(f"   Avg clustering: {metrics['avg_clustering']:.4f}")
            print(f"   Intra-module connections: {metrics['intra_module_connections']}")
            print(f"   Inter-module connections: {metrics['inter_module_connections']}")
        
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print("="*80)

def parse_arguments():
    """Parse command line arguments, reusing training script structure."""
    parser = argparse.ArgumentParser(description='Modular Network Structure Analysis')
    
    # Environment arguments
    parser.add_argument('--environment', type=str, default='CartPole-v1',
                       help='Gymnasium environment name')
    
    # Network arguments
    parser.add_argument('--size', type=int, default=256,
                       help='Network size (number of hidden units)')
    parser.add_argument('--activation', type=str, default='ReLU',
                       help='Activation function')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Analysis arguments
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    print("🔬 Modular Network Structure Analysis")
    print("="*50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Create analyzer
    analyzer = ModularNetworkAnalyzer(args)
    
    # Run analysis pipeline
    analyzer.create_networks()
    analyzer.analyze_network_structure()
    analyzer.visualize_networks()
    analyzer.export_results()
    analyzer.print_summary()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    import time
    main()
