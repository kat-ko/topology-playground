#!/usr/bin/env python3
"""
Functional Modularity Analysis for Network Topologies

This script extends the base continual learning training to analyze functional modularity
in different network topologies following the methodology from:
- Tanner et al. 2023: Functional connectivity from activation correlations  
- Ellefsen 2015: Community detection for functional modules

Workflow:
1. Train models using continual_learning_training() and save checkpoints
2. Record hidden activations from frozen models across different difficulty levels
3. Build functional connectivity matrix using Pearson correlations
4. Apply Louvain community detection to identify functional modules
5. Compare functional vs structural modularity across topologies
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Import base training functionality
from topologies_continual_task_training_normal_modularity import (
    continual_learning_training,
    create_continual_learning_run_name,
    DebugTopologyPolicy
)

# Suppress gymnasium warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

# Community detection
try:
    import community as community_louvain
except ImportError:
    print("Installing python-louvain...")
    os.system("pip install python-louvain")
    import community as community_louvain

# ============================================================================
# ACTIVATION RECORDING SYSTEM
# ============================================================================

class ActivationRecorder:
    """Records hidden layer activations from frozen PPO models."""
    
    def __init__(self):
        self.recorded_activations = {}
    
    def record_activations(self, model, env, level, num_episodes=50):
        """
        Record hidden activations from a frozen model on a specific difficulty level.
        
        Args:
            model: Trained PPO model
            env: Gymnasium environment configured for specific level
            level: Difficulty level identifier  
            num_episodes: Number of episodes to record
            
        Returns:
            activations: numpy array of shape (timesteps, num_neurons)
        """
        print(f"🎯 Recording activations for level {level} ({num_episodes} episodes)...")
        
        activations_list = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_activations = []
            
            while not done:
                # Get model prediction and extract hidden activations
                with torch.no_grad():
                    # Convert observation to tensor
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    
                    # Get features from the policy network
                    features = model.policy.extract_features(obs_tensor)
                    
                    # Record the hidden layer activations
                    if hasattr(features, 'numpy'):
                        hidden_activations = features.numpy().flatten()
                    else:
                        hidden_activations = features.detach().cpu().numpy().flatten()
                    
                    episode_activations.append(hidden_activations)
                
                # Take action and step environment
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # Add episode activations to collection
            if episode_activations:
                activations_list.extend(episode_activations)
        
        # Convert to numpy array
        activations = np.array(activations_list)
        print(f"   Recorded {activations.shape[0]} timesteps × {activations.shape[1]} neurons")
        
        return activations

# ============================================================================
# FUNCTIONAL CONNECTIVITY & COMMUNITY DETECTION
# ============================================================================

class FunctionalModularityAnalyzer:
    """Analyzes functional modularity using correlation-based connectivity."""
    
    def __init__(self, correlation_threshold=0.1):
        self.correlation_threshold = correlation_threshold
        self.results = {}
    
    def compute_functional_connectivity_matrix(self, activations):
        """
        Compute Pearson correlation matrix between all neuron pairs.
        
        Args:
            activations: Shape (timesteps, num_neurons)
            
        Returns:
            fc_matrix: Shape (num_neurons, num_neurons) correlation matrix
        """
        print("🔗 Computing functional connectivity matrix...")
        
        # Pearson correlation between columns (neurons)
        fc_matrix = np.corrcoef(activations.T)
        
        # Handle NaN values (constant activations)
        fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
        
        print(f"   FC matrix shape: {fc_matrix.shape}")
        print(f"   Correlation range: [{fc_matrix.min():.3f}, {fc_matrix.max():.3f}]")
        
        return fc_matrix
    
    def build_functional_graph(self, fc_matrix):
        """
        Build NetworkX graph from functional connectivity matrix.
        
        Args:
            fc_matrix: Correlation matrix
            
        Returns:
            graph: NetworkX graph with edges above threshold
        """
        print(f"🕸️ Building functional graph (threshold={self.correlation_threshold})...")
        
        # Use absolute correlations and apply threshold
        abs_fc = np.abs(fc_matrix)
        
        # Create graph from adjacency matrix
        graph = nx.from_numpy_array(abs_fc)
        
        # Remove edges below threshold
        edges_to_remove = [(u, v) for u, v, d in graph.edges(data=True) 
                          if d['weight'] < self.correlation_threshold]
        graph.remove_edges_from(edges_to_remove)
        
        print(f"   Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        
        return graph
    
    def detect_communities(self, graph):
        """
        Apply Louvain algorithm to detect functional communities.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            communities: Dict mapping node_id -> community_id
            modularity_score: Modularity Q value
        """
        print("🔍 Detecting functional communities (Louvain algorithm)...")
        
        if graph.number_of_edges() == 0:
            # No connections - all nodes in single community
            communities = {node: 0 for node in graph.nodes()}
            modularity_score = 0.0
        else:
            # Apply Louvain community detection
            communities = community_louvain.best_partition(graph)
            modularity_score = community_louvain.modularity(communities, graph)
        
        num_communities = len(set(communities.values()))
        print(f"   Found {num_communities} communities")
        print(f"   Modularity Q = {modularity_score:.4f}")
        
        return communities, modularity_score
    
    def analyze_level_modularity(self, activations, level):
        """
        Complete modularity analysis pipeline for one difficulty level.
        
        Args:
            activations: Recorded neural activations
            level: Difficulty level identifier
            
        Returns:
            results: Dict with all analysis results
        """
        print(f"\n🧠 Analyzing functional modularity for level {level}")
        
        # Step 1: Functional connectivity matrix
        fc_matrix = self.compute_functional_connectivity_matrix(activations)
        
        # Step 2: Build functional graph  
        graph = self.build_functional_graph(fc_matrix)
        
        # Step 3: Community detection
        communities, modularity_score = self.detect_communities(graph)
        
        # Compile results
        results = {
            'level': level,
            'modularity_score': modularity_score,
            'num_communities': len(set(communities.values())),
            'communities': communities,
            'fc_matrix': fc_matrix,
            'graph_stats': {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
                'density': nx.density(graph)
            }
        }
        
        return results

# ============================================================================
# ENVIRONMENT CREATION FOR DIFFERENT LEVELS
# ============================================================================

def create_level_specific_env(task_name, level, max_level=15):
    """
    Create environment with specific difficulty level (noise magnitude).
    
    Args:
        task_name: Environment name (e.g., 'CartPole-v1')
        level: Difficulty level (0 = no noise, higher = more noise)
        max_level: Maximum level for noise scaling
        
    Returns:
        env: Configured gymnasium environment
    """
    env = gym.make(task_name)
    
    # Calculate noise magnitude based on level
    if level == 0:
        noise_magnitude = 0.0  # Clean baseline
    else:
        # Scale noise from 0.0 to 1.0 across levels
        noise_magnitude = level / max_level
    
    # Wrap environment with noise if needed
    if noise_magnitude > 0:
        env = NoisyObservationWrapper(env, noise_magnitude)
    
    return env

class NoisyObservationWrapper(gym.ObservationWrapper):
    """Add Gaussian noise to observations."""
    
    def __init__(self, env, noise_magnitude):
        super().__init__(env)
        self.noise_magnitude = noise_magnitude
    
    def observation(self, obs):
        noise = np.random.normal(0, self.noise_magnitude, obs.shape)
        return obs + noise

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def train_and_analyze_topology(topology_type, task_name, seed, num_levels=3, 
                             level_switch=200, hidden_size=128, num_episodes=50):
    """
    Complete pipeline: Train model → Record activations → Analyze modularity.
    
    Args:
        topology_type: Network topology ('standard_mlp', 'modular', 'hybrid')
        task_name: Environment name
        seed: Random seed
        num_levels: Number of difficulty levels to test
        level_switch: Iterations per level during training
        hidden_size: Network hidden layer size
        num_episodes: Episodes for activation recording
        
    Returns:
        results: Complete analysis results
    """
    print(f"\n🚀 Starting analysis for {topology_type} on {task_name}")
    print(f"   Seed: {seed}, Levels: {num_levels}, Hidden size: {hidden_size}")
    
    # Step 1: Train model with checkpoint saving
    print("\n📚 Phase 1: Training model...")
    
    config = {
        'max_iterations': num_levels * level_switch,
        'level_switch': level_switch,
        'shift_range': [0, 1],
        'episode_cap': 500,
        'num_levels': num_levels,
        'num_layers': 1,
        'hidden_size': hidden_size,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_epochs': 10,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95,
        'gamma': 0.99
    }
    
    checkpoint_dir = f"modularity_checkpoints/{topology_type}_{task_name}_seed{seed}"
    
    model, env = continual_learning_training(
        config=config,
        task_name=task_name, 
        topology_type=topology_type,
        seed=seed,
        use_wandb=False,
        save_final_model=True,
        checkpoint_dir=checkpoint_dir
    )
    
    env.close()
    
    # Step 2: Record activations across levels
    print("\n🎯 Phase 2: Recording activations...")
    
    recorder = ActivationRecorder()
    analyzer = FunctionalModularityAnalyzer()
    
    level_results = []
    
    for level in range(1, num_levels + 1):
        # Create level-specific environment
        level_env = create_level_specific_env(task_name, level, num_levels)
        
        # Record activations
        activations = recorder.record_activations(model, level_env, level, num_episodes)
        
        # Analyze modularity
        level_result = analyzer.analyze_level_modularity(activations, level)
        level_results.append(level_result)
        
        level_env.close()
    
    # Step 3: Compile and save results
    print("\n💾 Phase 3: Saving results...")
    
    final_results = {
        'metadata': {
            'topology_type': topology_type,
            'task_name': task_name,
            'seed': seed,
            'num_levels': num_levels,
            'hidden_size': hidden_size,
            'num_episodes': num_episodes,
            'analysis_completed': datetime.now().isoformat()
        },
        'level_results': level_results,
        'summary': {
            'modularity_scores': [r['modularity_score'] for r in level_results],
            'num_communities': [r['num_communities'] for r in level_results],
            'avg_modularity': np.mean([r['modularity_score'] for r in level_results]),
            'max_modularity': np.max([r['modularity_score'] for r in level_results])
        }
    }
    
    # Save results
    results_path = f"{checkpoint_dir}/functional_modularity_analysis.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = final_results.copy()
        for level_result in json_results['level_results']:
            if 'fc_matrix' in level_result:
                level_result['fc_matrix'] = level_result['fc_matrix'].tolist()
        
        json.dump(json_results, f, indent=2)
    
    print(f"📊 Results saved: {results_path}")
    
    return final_results

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================

def compare_topology_modularity(results_list):
    """Compare modularity results across different topologies."""
    print("\n📊 Comparing topology modularity...")
    
    comparison_data = []
    for results in results_list:
        topology = results['metadata']['topology_type']
        avg_modularity = results['summary']['avg_modularity']
        max_modularity = results['summary']['max_modularity']
        
        comparison_data.append({
            'topology': topology,
            'avg_modularity': avg_modularity,
            'max_modularity': max_modularity,
            'modularity_scores': results['summary']['modularity_scores']
        })
    
    # Print comparison table
    print("\n🏆 Topology Modularity Ranking:")
    print("Topology        | Avg Q  | Max Q  | Scores")
    print("-" * 45)
    
    for data in sorted(comparison_data, key=lambda x: x['avg_modularity'], reverse=True):
        scores_str = ', '.join([f"{s:.3f}" for s in data['modularity_scores']])
        print(f"{data['topology']:<14} | {data['avg_modularity']:.3f} | {data['max_modularity']:.3f} | [{scores_str}]")
    
    return comparison_data

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Functional Modularity Analysis")
    
    # Basic parameters
    parser.add_argument("--task", type=str, default="CartPole-v1", 
                       help="Environment name")
    parser.add_argument("--topology", type=str, default="standard_mlp",
                       choices=['standard_mlp', 'modular', 'hybrid', 'small_world', 'fully_connected'],
                       help="Network topology")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    
    # Training parameters
    parser.add_argument("--num_levels", type=int, default=3,
                       help="Number of difficulty levels")
    parser.add_argument("--level_switch", type=int, default=200,
                       help="Iterations per level during training")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden layer size")
    
    # Analysis parameters  
    parser.add_argument("--num_episodes", type=int, default=50,
                       help="Episodes for activation recording")
    
    # Modes
    parser.add_argument("--single", action="store_true",
                       help="Run single topology analysis")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple topologies")
    
    args = parser.parse_args()
    
    if args.single:
        # Single topology analysis
        results = train_and_analyze_topology(
            topology_type=args.topology,
            task_name=args.task,
            seed=args.seed,
            num_levels=args.num_levels,
            level_switch=args.level_switch,
            hidden_size=args.hidden_size,
            num_episodes=args.num_episodes
        )
        
        print(f"\n✅ Analysis completed for {args.topology}")
        print(f"   Average modularity: {results['summary']['avg_modularity']:.4f}")
        print(f"   Max modularity: {results['summary']['max_modularity']:.4f}")
        
    elif args.compare:
        # Compare multiple topologies
        topologies = ['standard_mlp', 'modular', 'hybrid']
        results_list = []
        
        for topology in topologies:
            print(f"\n{'='*60}")
            results = train_and_analyze_topology(
                topology_type=topology,
                task_name=args.task,
                seed=args.seed,
                num_levels=args.num_levels,
                level_switch=args.level_switch,
                hidden_size=args.hidden_size,
                num_episodes=args.num_episodes
            )
            results_list.append(results)
        
        # Compare results
        compare_topology_modularity(results_list)
        
    else:
        print("Please specify --single or --compare mode")

if __name__ == "__main__":
    main()
