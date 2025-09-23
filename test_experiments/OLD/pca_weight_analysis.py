#!/usr/bin/env python3
"""
PCA Weight Analysis for Modular and Hybrid Networks

This script analyzes the weights of trained modular and hybrid networks to investigate
whether functional modularity emerges beyond the imposed structural modularity.

Usage:
    python pca_weight_analysis.py --task cartpole --noise N0002 --size S256
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import topology classes
import sys
sys.path.append('..')
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.networks.ffn import FeedForwardNetwork

def load_experiment_metadata(base_path: str) -> Dict:
    """Load experiment metadata to understand network configurations."""
    metadata = {}
    
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue
            
        metadata_file = os.path.join(item_path, 'run_metadata.json')
        if not os.path.exists(metadata_file):
            continue
            
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            config = data.get('training_config', {})
            topology_type = config.get('topology_type', 'unknown')
            seed = config.get('seed', 'unknown')
            
            if topology_type in ['modular', 'hybrid']:
                metadata[f"{topology_type}_{seed}"] = {
                    'topology_type': topology_type,
                    'seed': seed,
                    'config': config,
                    'path': item_path
                }
                
        except Exception as e:
            print(f"⚠️  Error loading {item}: {e}")
            continue
    
    return metadata

def create_network_from_config(config: Dict, topology_type: str, seed: int) -> FeedForwardNetwork:
    """Create a network instance from configuration."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Extract network parameters
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 1)
    
    # Task-specific input/output dimensions
    task_name = config.get('task_name', 'cartpole')
    if task_name == 'cartpole':
        input_dim, output_dim = 4, 2
    elif task_name == 'acrobot':
        input_dim, output_dim = 6, 3
    elif task_name == 'lunarlander':
        input_dim, output_dim = 8, 4
    else:
        input_dim, output_dim = 4, 2  # Default
    
    # Create topology
    if topology_type == 'modular':
        num_modules = config.get('num_modules', 4)
        topology = ModularTopology(
            size=hidden_size,
            num_modules=num_modules,
            inter_module_prob=config.get('inter_module_prob', 0.1),
            intra_module_prob=config.get('intra_module_prob', 0.8)
        )
    elif topology_type == 'hybrid':
        num_modules = config.get('num_modules', 4)
        topology = HybridTopology(
            size=hidden_size,
            num_modules=num_modules,
            k=config.get('k', 4),
            p=config.get('p', 0.2),
            inter_module_prob=config.get('inter_module_prob', 0.1),
            intra_module_prob=config.get('intra_module_prob', 0.8)
        )
    else:
        raise ValueError(f"Unsupported topology type: {topology_type}")
    
    # Generate network graph
    graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
    
    # Create input and output nodes
    input_nodes = list(range(input_dim))
    output_nodes = list(range(input_dim + hidden_size, input_dim + hidden_size + output_dim))
    
    # Create network
    network_params = {
        'learning_rate': 0.001,
        'activation': 'leaky_relu'
    }
    
    network = FeedForwardNetwork(graph, input_nodes, output_nodes, network_params)
    
    return network, topology

def extract_weights_by_structure(network: FeedForwardNetwork, topology_type: str, topology) -> Dict:
    """Extract weights grouped by structural role."""
    weights_by_structure = {}
    
    if topology_type == 'modular':
        # For modular networks, group by module assignment
        num_modules = topology.num_modules
        module_size = network.topology.number_of_nodes() // num_modules
        
        for module_id in range(num_modules):
            module_weights = []
            module_biases = []
            
            # Get nodes in this module
            start_node = module_id * module_size
            end_node = min((module_id + 1) * module_size, network.topology.number_of_nodes())
            module_nodes = list(range(start_node, end_node))
            
            for node in module_nodes:
                if node in network.node_states:
                    state = network.node_states[node]
                    
                    # Collect weights
                    if 'weights' in state:
                        for neighbor, weight in state['weights'].items():
                            module_weights.append(weight)
                    
                    # Collect biases
                    if 'bias' in state:
                        module_biases.append(state['bias'])
            
            weights_by_structure[f'module_{module_id}'] = {
                'weights': np.array(module_weights),
                'biases': np.array(module_biases),
                'node_count': len(module_nodes)
            }
    
    elif topology_type == 'hybrid':
        # For hybrid networks, group by modular vs small-world components
        modular_weights = []
        small_world_weights = []
        modular_biases = []
        small_world_biases = []
        
        for node, state in network.node_states.items():
            if 'weights' in state:
                for neighbor, weight in state['weights'].items():
                    # Simple heuristic: if both nodes are in same module, it's modular
                    # This is a simplified approach - could be more sophisticated
                    if abs(node - neighbor) < 50:  # Rough module boundary
                        modular_weights.append(weight)
                    else:
                        small_world_weights.append(weight)
            
            if 'bias' in state:
                if node < 100:  # Rough module boundary
                    modular_biases.append(state['bias'])
                else:
                    small_world_biases.append(state['bias'])
        
        weights_by_structure['modular_component'] = {
            'weights': np.array(modular_weights),
            'biases': np.array(modular_biases),
            'node_count': len(modular_weights)
        }
        
        weights_by_structure['small_world_component'] = {
            'weights': np.array(small_world_weights),
            'biases': np.array(small_world_biases),
            'node_count': len(small_world_weights)
        }
    
    return weights_by_structure

def perform_pca_analysis(weights_by_structure: Dict, topology_type: str) -> Dict:
    """Perform PCA analysis on weight groups."""
    pca_results = {}
    
    for group_name, weight_data in weights_by_structure.items():
        if len(weight_data['weights']) < 3:  # Need minimum samples for PCA
            continue
        
        # Prepare data
        weights = weight_data['weights'].reshape(-1, 1)  # Reshape for sklearn
        biases = weight_data['biases'].reshape(-1, 1)
        
        # Combine weights and biases
        combined_data = np.hstack([weights, biases])
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)
        
        # Perform PCA
        pca = PCA(n_components=min(2, len(scaled_data)-1))
        pca_result = pca.fit_transform(scaled_data)
        
        pca_results[group_name] = {
            'pca_result': pca_result,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'mean': pca.mean_,
            'n_samples': len(scaled_data),
            'weights': weights.flatten(),
            'biases': biases.flatten()
        }
    
    return pca_results

def visualize_pca_results(pca_results: Dict, topology_type: str, seed: int, output_dir: str):
    """Visualize PCA results."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PCA scatter plot
    ax1 = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(pca_results)))
    
    for i, (group_name, result) in enumerate(pca_results.items()):
        if result['pca_result'].shape[1] >= 2:
            ax1.scatter(result['pca_result'][:, 0], result['pca_result'][:, 1], 
                       label=group_name, alpha=0.7, color=colors[i])
        else:
            # If only 1 component, plot against index
            ax1.scatter(range(len(result['pca_result'])), result['pca_result'][:, 0], 
                       label=group_name, alpha=0.7, color=colors[i])
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(f'PCA Analysis - {topology_type.title()} (Seed {seed})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance
    ax2 = axes[1]
    group_names = list(pca_results.keys())
    explained_var = [result['explained_variance_ratio'][0] for result in pca_results.values()]
    
    ax2.bar(group_names, explained_var, alpha=0.7)
    ax2.set_xlabel('Weight Groups')
    ax2.set_ylabel('Explained Variance (PC1)')
    ax2.set_title('PCA Explained Variance by Group')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_analysis_{topology_type}_seed{seed}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ PCA visualization saved: {output_dir}/pca_analysis_{topology_type}_seed{seed}.png")

def analyze_functional_modularity(pca_results: Dict, topology_type: str) -> Dict:
    """Analyze functional modularity patterns."""
    analysis = {
        'topology_type': topology_type,
        'n_groups': len(pca_results),
        'group_analysis': {}
    }
    
    for group_name, result in pca_results.items():
        # Calculate basic statistics
        weights = result['weights']
        biases = result['biases']
        
        group_analysis = {
            'n_weights': len(weights),
            'n_biases': len(biases),
            'weight_mean': np.mean(weights),
            'weight_std': np.std(weights),
            'bias_mean': np.mean(biases),
            'bias_std': np.std(biases),
            'explained_variance_pc1': result['explained_variance_ratio'][0],
            'explained_variance_pc2': result['explained_variance_ratio'][1] if len(result['explained_variance_ratio']) > 1 else 0
        }
        
        analysis['group_analysis'][group_name] = group_analysis
    
    return analysis

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='PCA Weight Analysis for Modular and Hybrid Networks')
    parser.add_argument('--task', type=str, choices=['cartpole', 'acrobot', 'lunarlander'], 
                       default='cartpole', help='Task to analyze')
    parser.add_argument('--noise', type=str, default='N0002', help='Noise level to analyze')
    parser.add_argument('--size', type=str, default='S256', help='Network size to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], 
                       help='Seeds to analyze')
    
    args = parser.parse_args()
    
    print(f"🎯 PCA Weight Analysis")
    print(f"📊 Task: {args.task.upper()}")
    print(f"📊 Noise: {args.noise}")
    print(f"📊 Size: {args.size}")
    print("=" * 60)
    
    # Set up paths
    base_path = f"{args.task}/{args.noise}/{args.size}"
    output_dir = f"pca_analysis_{args.task}_{args.noise}_{args.size}"
    
    if not os.path.exists(base_path):
        print(f"❌ Directory {base_path} not found")
        return
    
    # Load experiment metadata
    print("🔄 Loading experiment metadata...")
    metadata = load_experiment_metadata(base_path)
    
    if not metadata:
        print("❌ No modular or hybrid experiments found")
        return
    
    print(f"✅ Found {len(metadata)} experiments")
    
    # Analyze each experiment
    all_analyses = {}
    
    for exp_key, exp_data in metadata.items():
        topology_type = exp_data['topology_type']
        seed = exp_data['seed']
        config = exp_data['config']
        
        print(f"\n🔄 Analyzing {topology_type} (seed {seed})...")
        
        try:
            # Create network
            network, topology = create_network_from_config(config, topology_type, seed)
            
            # Extract weights by structure
            weights_by_structure = extract_weights_by_structure(network, topology_type, topology)
            
            # Perform PCA analysis
            pca_results = perform_pca_analysis(weights_by_structure, topology_type)
            
            # Visualize results
            visualize_pca_results(pca_results, topology_type, seed, output_dir)
            
            # Analyze functional modularity
            analysis = analyze_functional_modularity(pca_results, topology_type)
            all_analyses[exp_key] = analysis
            
            print(f"✅ Analysis complete for {topology_type} (seed {seed})")
            
        except Exception as e:
            print(f"❌ Error analyzing {exp_key}: {e}")
            continue
    
    # Save analysis results
    results_file = f"{output_dir}/pca_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_analyses, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Results saved to: {output_dir}/")
    print(f"📊 Analysis file: {results_file}")

if __name__ == "__main__":
    main()

