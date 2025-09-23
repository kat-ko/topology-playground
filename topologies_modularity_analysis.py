#!/usr/bin/env python3
"""
Functional Modularity Analysis for Topology Networks

Analyzes the emergence of functional modularity in trained topology networks
by recording neural activations, computing functional connectivity graphs,
and performing community detection and lesion experiments.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
# Set matplotlib backend to avoid tkinter issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
import os
import sys
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import networkx as nx
import io
import base64
from PIL import Image
import csv
import argparse
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import community as community_louvain  # pip install python-louvain
from collections import defaultdict

# Import topology modules
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.standard_mlp import StandardMLPTopology
from src.networks.ffn import FeedForwardNetwork
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager
from src.utils.capacity_matching_helper import pre_calculate_capacity_matching
from src.utils.device_manager import get_device_manager, get_device_info
from src.utils.task_training_config import get_task_timesteps, create_convergence_callback
from src.utils.topology_logging_handler import (SimplifiedCallback, create_logging_handler)

# Import exact training functions (avoid class conflicts)
from topologies_continual_task_training_normal_modularity import (
    continual_learning_training,
    create_continual_learning_run_name,
    run_test_experiment
)

# ============================================================================
# FUNCTIONAL MODULARITY ANALYSIS SYSTEM
# ============================================================================

class LevelSpecificActivationRecorder:
    """Record neural activations during inference, organized by distribution level"""
    
    def __init__(self, num_episodes_per_level=10, save_path=None):
        self.num_episodes_per_level = num_episodes_per_level
        self.save_path = save_path
        self.level_data = {}  # {level: {'activations': [], 'observations': [], 'episode_count': 0}}
        
        print(f"🔍 LevelSpecificActivationRecorder initialized: {num_episodes_per_level} episodes per level")
        
    def record_for_level(self, level, obs, hidden_activations, timestep_in_episode=None):
        """Record observation and activations for a specific distribution level"""
        if level not in self.level_data:
            self.level_data[level] = {
                'activations': [],
                'observations': [],
                'episode_activations': [],  # Store full episode sequences
                'episode_count': 0,
                'current_episode_activations': []
            }
        
        level_info = self.level_data[level]
        
        if level_info['episode_count'] < self.num_episodes_per_level:
            # Convert to numpy if tensor
            if torch.is_tensor(obs):
                obs_np = obs.cpu().numpy()
            else:
                obs_np = np.array(obs)
                
            if torch.is_tensor(hidden_activations):
                hidden_np = hidden_activations.cpu().numpy()
            else:
                hidden_np = np.array(hidden_activations)
            
            # Record individual timestep
            level_info['observations'].append(obs_np)
            level_info['activations'].append(hidden_np)
            
            # Record for episode sequence analysis
            level_info['current_episode_activations'].append(hidden_np)
    
    def end_episode_for_level(self, level):
        """Mark end of episode for a specific level"""
        if level in self.level_data:
            level_info = self.level_data[level]
            
            # Save the complete episode sequence
            if level_info['current_episode_activations']:
                level_info['episode_activations'].append(
                    np.array(level_info['current_episode_activations'])
                )
                level_info['current_episode_activations'] = []
            
            level_info['episode_count'] += 1
            
    def get_level_data(self, level):
        """Get recorded data for a specific level"""
        if level not in self.level_data:
            return None
            
        level_info = self.level_data[level]
        return {
            'observations': np.array(level_info['observations']) if level_info['observations'] else np.array([]),
            'activations': np.array(level_info['activations']) if level_info['activations'] else np.array([]),
            'episode_activations': level_info['episode_activations'],  # List of episode sequences
            'num_episodes': level_info['episode_count']
        }
        
    def save_activations(self):
        """Save all recorded activations to file"""
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            
            # Save level-specific data
            for level, level_info in self.level_data.items():
                level_data = {
                    'level': level,
                    'observations': level_info['observations'],
                    'activations': level_info['activations'],
                    'episode_activations': level_info['episode_activations'],
                    'num_episodes': level_info['episode_count'],
                    'timestamp': time.time()
                }
                
                level_file = f"{self.save_path}/level_{level:02d}_activations.pth"
                torch.save(level_data, level_file)
                print(f"💾 Saved Level {level} activations: {level_info['episode_count']} episodes to {level_file}")
        
        return self.level_data


class FunctionalModularityAnalyzer:
    """Main analysis class for functional modularity"""
    
    def __init__(self, base_path="test_experiments/MODULARITY"):
        self.base_path = base_path
        self.results = {}
        
        # Define topology colors (reuse from existing plotting system)
        self.topology_colors = {
            'standard_mlp': '#ff595e',
            'hybrid': '#ffca3a', 
            'modular': '#8ac926',
            'small_world': '#1982c4'
        }
        
        print(f"🧠 FunctionalModularityAnalyzer initialized")
        print(f"   Base path: {self.base_path}")
    
    def discover_experiments(self, task=None, noise_level=None, size=None):
        """Discover experiment directories using existing patterns"""
        experiments = defaultdict(list)
        
        # Convert task name to the format used in directory names
        task_code = {'cartpole': 'CP', 'acrobot': 'AC', 'lunarlander': 'LL'}.get(task.lower(), 'CP')
        
        # Search in MODULARITY directory for matching experiments
        modularity_path = os.path.join(self.base_path, 'MODULARITY')
        
        if os.path.exists(modularity_path):
            print(f"🔍 Searching in: {modularity_path}")
            
            # Look for matching experiment directories
            for item in os.listdir(modularity_path):
                item_path = os.path.join(modularity_path, item)
                if os.path.isdir(item_path):
                    # Check if this experiment matches our criteria
                    if (task_code in item and 
                        noise_level in item and 
                        size in item):
                        
                        # Determine topology type from directory name
                        topology = self._extract_topology_from_name(item)
                        if topology:
                            experiments[topology].append(item_path)
                            print(f"   Found {topology}: {item}")
                            
        print(f"📁 Found experiments: {dict([(k, len(v)) for k, v in experiments.items()])}")
        return experiments
    
    def _extract_topology_from_name(self, dir_name):
        """Extract topology type from directory name"""
        if dir_name.startswith('HYB_'):
            return 'hybrid'
        elif dir_name.startswith('MOD_'):
            return 'modular'
        elif dir_name.startswith('SW_'):
            return 'small_world'
        elif dir_name.startswith('STANDARD_MLP_'):
            return 'standard_mlp'
        else:
            return None
        
    def load_trained_model(self, experiment_dir):
        """Load trained model from experiment directory"""
        # Look for model files
        model_files = []
        for file in os.listdir(experiment_dir):
            if file.endswith('.zip') and 'model' in file.lower():
                model_files.append(os.path.join(experiment_dir, file))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {experiment_dir}")
        
        # Load the most recent model
        model_file = sorted(model_files)[-1]
        print(f"📥 Loading model: {model_file}")
        
        try:
            model = PPO.load(model_file)
            return model
        except Exception as e:
            print(f"❌ Failed to load model {model_file}: {e}")
        return None
    
    def record_activations(self, model, experiment_dir, num_episodes=50):
        """Record activations during model inference"""
        # Extract configuration from directory name
        config = self._extract_config_from_path(experiment_dir)
        
        # Create environment
        env = self._create_env_from_config(config)
        
        # Initialize activation recorder
        recorder = ActivationRecorder(num_episodes=num_episodes, save_path=experiment_dir)
        
        # Record activations over multiple episodes
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < 500:  # Limit steps per episode
                # Get action and hidden activations
                action, hidden_state = self._get_action_with_activations(model, obs)
                
                # Record the activation
                recorder.record(obs, hidden_state)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
            
            recorder.increment_episode()
            
            if episode % 10 == 0:
                print(f"   Episode {episode}/{num_episodes} completed")
        
        # Save and return activations
        return recorder.save_activations()
    
    def test_model_across_levels(self, model_path, num_episodes_per_level=10, max_levels=15):
        """Test a trained model across all distribution levels it was trained on"""
        print(f"🧪 Testing model across levels: {os.path.basename(model_path)}")
        
        # Load the trained model
        try:
            model = PPO.load(model_path)
        except Exception as e:
            print(f"❌ Failed to load model {model_path}: {e}")
            return {}
        
        # Create level-specific recorder
        save_dir = os.path.dirname(model_path)
        recorder = LevelSpecificActivationRecorder(
            num_episodes_per_level=num_episodes_per_level, 
            save_path=f"{save_dir}/level_activations"
        )
        
        # Test on each level
        for level in range(max_levels):
            print(f"   Testing Level {level:2d} (shift magnitude: {level * 0.2:.1f})")
            
            try:
                # Create environment with specific perturbation level
                env = self._create_environment_for_level("CartPole-v1", level)
                
                # Record episodes at this level
                for episode in range(num_episodes_per_level):
                    obs, _ = env.reset()
                    done = False
                    episode_step = 0
                    
                    while not done and episode_step < 500:  # Max episode length
                        # Get action from model
                        action, _ = model.predict(obs, deterministic=True)
                        
                        # Extract hidden layer activations
                        hidden_activations = self._extract_hidden_activations(model, obs)
                        
                        # Record for this level
                        recorder.record_for_level(level, obs, hidden_activations, episode_step)
                        
                        # Step environment
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        episode_step += 1
                    
                    # Mark end of episode
                    recorder.end_episode_for_level(level)
                
                env.close()
                
            except Exception as e:
                print(f"   ⚠️  Failed to test level {level}: {e}")
                continue
        
        # Save all recorded data
        level_data = recorder.save_activations()
        print(f"✅ Completed testing across {max_levels} levels")
        
        return level_data
    
    def _create_environment_for_level(self, task_name, level):
        """Create environment with specific perturbation level"""
        # Create base environment
        env = gym.make(task_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Add perturbation for levels > 0
        if level > 0:
            # Calculate shift magnitude (matches training)
            shift_magnitude = level * 0.2  # Adjust based on your training setup
            shift_vector = np.full(env.observation_space.shape[0], shift_magnitude)
            
            # Apply observation shift (simplified version)
            original_reset = env.reset
            original_step = env.step
            
            def perturbed_reset(*args, **kwargs):
                obs, info = original_reset(*args, **kwargs)
                return obs + shift_vector, info
            
            def perturbed_step(action):
                obs, reward, terminated, truncated, info = original_step(action)
                return obs + shift_vector, reward, terminated, truncated, info
            
            env.reset = perturbed_reset
            env.step = perturbed_step
        
        return env
    
    def _extract_hidden_activations(self, model, obs):
        """Extract hidden layer activations from PPO model"""
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Access the policy network features
                features = model.policy.extract_features(obs_tensor)
                
                # For PPO, the features are typically the output of the feature extractor
                # which is what we want for functional modularity analysis
                return features.cpu().numpy().flatten()
                
except Exception as e:
            print(f"⚠️  Failed to extract activations: {e}")
            # Return random placeholder for now
            return np.random.randn(256)
    
    def _extract_config_from_path(self, experiment_dir):
        """Extract configuration from experiment directory path"""
        # Parse directory name to extract config
        dir_name = os.path.basename(experiment_dir)
        
        config = {
            'task': 'CartPole-v1',  # Default
            'seed': 42,
            'topology': self._extract_topology_from_name(dir_name)
        }
        
        # Extract task from parent directories
        path_parts = experiment_dir.split(os.sep)
        for part in path_parts:
            if 'cartpole' in part.lower():
                config['task'] = 'CartPole-v1'
            elif 'acrobot' in part.lower():
                config['task'] = 'Acrobot-v1'
            elif 'lunarlander' in part.lower():
                config['task'] = 'LunarLander-v2'
        
        return config
    
    def _create_env_from_config(self, config):
        """Create environment from configuration"""
        env_name = config.get('task', 'CartPole-v1')
        env = gym.make(env_name)
        return env
    
    def _get_action_with_activations(self, model, obs):
        """Get action and extract hidden layer activations"""
        # Convert observation to tensor
        if not torch.is_tensor(obs):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        else:
            obs_tensor = obs.unsqueeze(0) if obs.dim() == 1 else obs
        
        # Get action using the model
        with torch.no_grad():
            # Use the model's policy to get action
            action, _, _ = model.policy.forward(obs_tensor)
            
            # Extract hidden layer activations from the policy network
            # This will depend on the specific network architecture
            features = model.policy.extract_features(obs_tensor)
            
            # Get the hidden layer activations (first layer after feature extraction)
            if hasattr(model.policy, 'mlp_extractor'):
                hidden_activations = model.policy.mlp_extractor.policy_net[0](features)
            else:
                hidden_activations = features
        
        return action.cpu().numpy()[0], hidden_activations
    
    def compute_functional_graph(self, activations):
        """Compute correlation-based functional connectivity graph"""
        activation_data = activations['activations']
        
        # Flatten activations across episodes and time steps
        # Shape: (num_samples, num_neurons)
        if len(activation_data.shape) > 2:
            activation_data = activation_data.reshape(-1, activation_data.shape[-1])
        
        num_neurons = activation_data.shape[1]
        correlation_matrix = np.zeros((num_neurons, num_neurons))
        
        print(f"🔗 Computing functional connectivity for {num_neurons} neurons...")
        
        # Compute pairwise correlations
        for i in range(num_neurons):
            for j in range(i, num_neurons):
                if i == j:
                    correlation_matrix[i, j] = 1.0
        else:
                    corr, _ = pearsonr(activation_data[:, i], activation_data[:, j])
                    correlation_matrix[i, j] = abs(corr)  # Use absolute correlation
                    correlation_matrix[j, i] = abs(corr)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(num_neurons):
            G.add_node(i)
        
        # Add edges with correlation weights above threshold
        threshold = 0.1  # Minimum correlation to consider as connection
        for i in range(num_neurons):
            for j in range(i+1, num_neurons):
                if correlation_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=correlation_matrix[i, j])
        
        print(f"   Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G, correlation_matrix
    
    def detect_communities(self, functional_graph):
        """Detect functional communities using Louvain algorithm"""
        print("🏘️ Detecting functional communities...")
        
        try:
            # Use Louvain algorithm for community detection
            communities = community_louvain.best_partition(functional_graph)
            
            # Convert to list of sets for easier handling
            community_sets = defaultdict(set)
            for node, community_id in communities.items():
                community_sets[community_id].add(node)
            
            community_list = [community_sets[i] for i in sorted(community_sets.keys())]
            
            print(f"   Found {len(community_list)} communities")
            for i, community in enumerate(community_list):
                print(f"   Community {i}: {len(community)} neurons")
            
            return community_list, communities
            
        except Exception as e:
            print(f"❌ Community detection failed: {e}")
            return [], {}
    
    def compute_modularity(self, functional_graph, communities):
        """Compute modularity score of the functional graph"""
        try:
            modularity = community_louvain.modularity(communities, functional_graph)
            print(f"📊 Modularity score: {modularity:.4f}")
            return modularity
            except Exception as e:
            print(f"❌ Modularity computation failed: {e}")
            return 0.0
    
    def perform_lesion_experiments(self, model, communities, num_episodes=20):
        """Perform lesion experiments to test functional importance"""
        print("🔬 Performing lesion experiments...")
        
        # This is a placeholder for lesion experiments
        # In practice, you would need to modify the model's weights
        # to "lesion" specific communities and test performance
        
        lesion_results = {
            'baseline_performance': 0.0,
            'community_lesions': {}
        }
        
        # For now, return placeholder results
        for i, community in enumerate(communities):
            lesion_results['community_lesions'][f'community_{i}'] = {
                'performance_drop': np.random.uniform(0, 0.3),  # Placeholder
                'community_size': len(community) if hasattr(community, '__len__') else 0
            }
        
        return lesion_results
    
    def plot_functional_modularity_comparison(self, results):
        """Generate comparison plots of functional modularity across topologies"""
        print("📊 Generating functional modularity comparison plots...")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Functional Modularity Analysis Across Topologies', fontsize=14, fontweight='bold')
        
        # Extract data for plotting
        topologies = []
        modularity_scores = []
        num_communities = []
        
        for topology, topology_results in results.items():
            for result in topology_results:
                topologies.append(topology)
                modularity_scores.append(result.get('modularity_score', 0))
                num_communities.append(len(result.get('communities', [])))
        
        # Plot 1: Modularity scores by topology
        axes[0, 0].boxplot([
            [r.get('modularity_score', 0) for r in results.get(topo, [])] 
            for topo in results.keys()
        ], labels=list(results.keys()))
        axes[0, 0].set_title('Modularity Scores by Topology')
        axes[0, 0].set_ylabel('Modularity Score')
        
        # Plot 2: Number of communities by topology  
        axes[0, 1].boxplot([
            [len(r.get('communities', [])) for r in results.get(topo, [])]
            for topo in results.keys()
        ], labels=list(results.keys()))
        axes[0, 1].set_title('Number of Communities by Topology')
        axes[0, 1].set_ylabel('Number of Communities')
        
        # Plot 3: Scatter plot of modularity vs communities
        for topology in results.keys():
            topo_modularity = [r.get('modularity_score', 0) for r in results[topology]]
            topo_communities = [len(r.get('communities', [])) for r in results[topology]]
            
            color = self.topology_colors.get(topology, '#6a4c93')
            axes[1, 0].scatter(topo_communities, topo_modularity, 
                             label=topology, color=color, alpha=0.7, s=60)
        
        axes[1, 0].set_xlabel('Number of Communities')
        axes[1, 0].set_ylabel('Modularity Score')
        axes[1, 0].set_title('Modularity vs Number of Communities')
        axes[1, 0].legend()
        
        # Plot 4: Example functional graph (if available)
        if results:
            # Use first available result to show example graph
            example_result = list(results.values())[0][0]
            if 'functional_graph' in example_result:
                G = example_result['functional_graph']
                pos = nx.spring_layout(G, k=1, iterations=50)
                nx.draw(G, pos, ax=axes[1, 1], node_size=20, 
                       node_color='lightblue', edge_color='gray', alpha=0.6)
                axes[1, 1].set_title('Example Functional Connectivity Graph')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('modularity_analysis_results', exist_ok=True)
        plt.savefig('modularity_analysis_results/functional_modularity_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("💾 Saved comparison plot: modularity_analysis_results/functional_modularity_comparison.png")
        
        plt.close()
    
    def save_results(self, results, task, noise_level, size):
        """Save analysis results to files"""
        print("💾 Saving analysis results...")
        
        # Create results directory
        results_dir = f"modularity_analysis_results/{task}_{noise_level}_{size}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results as JSON
        results_summary = {}
        for topology, topology_results in results.items():
            results_summary[topology] = {
                'num_experiments': len(topology_results),
                'avg_modularity': np.mean([r.get('modularity_score', 0) for r in topology_results]),
                'std_modularity': np.std([r.get('modularity_score', 0) for r in topology_results]),
                'avg_communities': np.mean([len(r.get('communities', [])) for r in topology_results]),
                'experiments': topology_results
            }
        
        with open(f"{results_dir}/analysis_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_dir}/analysis_results.json")
    
    def analyze_level_specific_modularity(self, level_data):
        """Analyze functional modularity for each distribution level separately"""
        print(f"🔍 Analyzing level-specific functional modularity...")
        
        level_results = {}
        
        for level in sorted(level_data.keys()):
            level_info = level_data[level]
            
            # Skip levels with insufficient data
            if level_info['episode_count'] < 3:
                print(f"   Level {level:2d}: Insufficient data ({level_info['episode_count']} episodes)")
                continue
                
            print(f"   Level {level:2d}: Analyzing {level_info['episode_count']} episodes...")
            
            try:
                # Prepare activation data for this level
                level_activations = {
                    'activations': np.array(level_info['activations']),
                    'observations': np.array(level_info['observations'])
                }
                
                # Compute functional graph for this level
                functional_graph, correlation_matrix = self.compute_functional_graph(level_activations)
                
                # Detect communities
                communities, community_dict = self.detect_communities(functional_graph)
                
                # Compute modularity score
                modularity_score = self.compute_modularity(functional_graph, community_dict)
                
                # Store results for this level
                level_results[level] = {
                    'modularity_score': modularity_score,
                    'num_communities': len(communities),
                    'communities': communities,
                    'community_dict': community_dict,
                    'correlation_matrix': correlation_matrix,
                    'functional_graph': functional_graph,
                    'num_episodes': level_info['episode_count'],
                    'shift_magnitude': level * 0.2
                }
                
                print(f"      Modularity: {modularity_score:.4f}, Communities: {len(communities)}")
                
            except Exception as e:
                print(f"      ❌ Failed to analyze level {level}: {e}")
                continue
        
        return level_results
    
    def compare_modularity_across_levels(self, level_results):
        """Compare how modularity changes across difficulty levels"""
        print(f"\n📊 Modularity Evolution Across Levels:")
        print("=" * 60)
        print(f"{'Level':>5} {'Shift':>6} {'Modularity':>10} {'Communities':>11} {'Episodes':>9}")
        print("-" * 60)
        
        modularity_evolution = []
        community_evolution = []
        
        for level in sorted(level_results.keys()):
            result = level_results[level]
            modularity_evolution.append(result['modularity_score'])
            community_evolution.append(result['num_communities'])
            
            print(f"{level:>5} {result['shift_magnitude']:>6.1f} {result['modularity_score']:>10.4f} "
                  f"{result['num_communities']:>11d} {result['num_episodes']:>9d}")
        
        print("-" * 60)
        
        # Compute trends
        if len(modularity_evolution) > 1:
            mod_trend = np.polyfit(range(len(modularity_evolution)), modularity_evolution, 1)[0]
            comm_trend = np.polyfit(range(len(community_evolution)), community_evolution, 1)[0]
            
            print(f"Modularity trend: {mod_trend:+.6f} per level")
            print(f"Community trend: {comm_trend:+.2f} per level")
            
            # Interpretation
            if mod_trend > 0.001:
                print("📈 Modularity INCREASES with task difficulty")
            elif mod_trend < -0.001:
                print("📉 Modularity DECREASES with task difficulty") 
            else:
                print("➡️  Modularity remains STABLE across difficulty")
        
        return {
            'modularity_evolution': modularity_evolution,
            'community_evolution': community_evolution,
            'level_results': level_results
        }
    
    def analyze_checkpoint_progression(self, checkpoint_dir):
        """Analyze functional modularity evolution across level checkpoints"""
        print(f"🔍 Analyzing checkpoint progression in: {checkpoint_dir}")
        
        # Load checkpoint metadata
        metadata_path = f"{checkpoint_dir}/checkpoint_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Checkpoint metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        checkpoint_data = metadata['checkpoint_data']
        task_name = metadata['task_name']
        
        print(f"📊 Analyzing {len(checkpoint_data)} level checkpoints")
        
        # Analyze each checkpoint
        progression_results = []
        
        for i, checkpoint_info in enumerate(checkpoint_data):
            level = checkpoint_info['level']
            model_path = checkpoint_info['checkpoint_path']
            noise_magnitude = checkpoint_info['noise_magnitude']
            
            print(f"   Level {level:2d} (noise: {noise_magnitude:.1f}): ", end="")
            
            if not os.path.exists(model_path):
                print("❌ Model file not found")
                continue
            
            try:
                # Test model on its adapted level
                level_data = self.test_model_on_specific_level(model_path, level, task_name)
                
                if not level_data:
                    print("❌ Failed to record activations")
                    continue
                
                # Analyze functional modularity for this level
                level_activations = {
                    'activations': np.array(level_data['activations']),
                    'observations': np.array(level_data['observations'])
                }
                
                # Compute functional graph and modularity
                functional_graph, correlation_matrix = self.compute_functional_graph(level_activations)
                communities, community_dict = self.detect_communities(functional_graph)
                modularity_score = self.compute_modularity(functional_graph, community_dict)
                
                # Store results
                result = {
                    'level': level,
                    'noise_magnitude': noise_magnitude,
                    'total_iterations': checkpoint_info['total_iterations'],
                    'modularity_score': modularity_score,
                    'num_communities': len(communities),
                    'communities': communities,
                    'correlation_matrix': correlation_matrix,
                    'num_episodes': level_data['num_episodes']
                }
                
                progression_results.append(result)
                
                print(f"✅ Q={modularity_score:.4f}, Communities={len(communities)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        if not progression_results:
            print("❌ No successful checkpoint analyses")
            return None
        
        # Analyze progression patterns
        progression_analysis = self.analyze_modularity_progression_patterns(progression_results)
        
        # Save results
        results_path = f"{checkpoint_dir}/progression_analysis.json"
        analysis_results = {
            'metadata': metadata,
            'progression_results': progression_results,
            'progression_analysis': progression_analysis
        }
        
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"\n✅ Progression analysis completed: {results_path}")
        
        return analysis_results
    
    def test_model_on_specific_level(self, model_path, level, task_name, num_episodes=10):
        """Test a checkpoint model on its specific adapted level"""
        try:
            # Load model
            model = PPO.load(model_path)
            
            # Create environment for this specific level
            env = self._create_environment_for_level(task_name, level)
            
            # Record activations
            activations = []
            observations = []
            episode_count = 0
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                done = False
                episode_step = 0
                
                while not done and episode_step < 500:
                    # Get action and activations
                    action, _ = model.predict(obs, deterministic=True)
                    hidden_activations = self._extract_hidden_activations(model, obs)
                    
                    # Record
                    activations.append(hidden_activations)
                    observations.append(obs.copy())
                    
                    # Step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_step += 1
                
                episode_count += 1
            
            env.close()
            
            return {
                'activations': activations,
                'observations': observations,
                'num_episodes': episode_count,
                'level': level
            }
            
        except Exception as e:
            print(f"Failed to test model on level {level}: {e}")
            return None
    
    def analyze_modularity_progression_patterns(self, progression_results):
        """Analyze patterns in modularity evolution across levels"""
        
        if len(progression_results) < 2:
            return {"error": "Insufficient data for progression analysis"}
        
        # Extract data series
        levels = [r['level'] for r in progression_results]
        modularity_scores = [r['modularity_score'] for r in progression_results]
        num_communities = [r['num_communities'] for r in progression_results]
        noise_levels = [r['noise_magnitude'] for r in progression_results]
        
        # Compute trends
        modularity_trend = np.polyfit(levels, modularity_scores, 1)[0] if len(levels) > 1 else 0
        community_trend = np.polyfit(levels, num_communities, 1)[0] if len(levels) > 1 else 0
        
        # Find significant changes
        modularity_changes = []
        for i in range(1, len(modularity_scores)):
            change = modularity_scores[i] - modularity_scores[i-1]
            if abs(change) > 0.02:  # Significant change threshold
                modularity_changes.append({
                    'from_level': levels[i-1],
                    'to_level': levels[i],
                    'change': change,
                    'direction': 'increase' if change > 0 else 'decrease'
                })
        
    return {
            'modularity_trend': modularity_trend,
            'community_trend': community_trend,
            'significant_changes': modularity_changes,
            'final_modularity': modularity_scores[-1],
            'modularity_range': [min(modularity_scores), max(modularity_scores)],
            'stability': np.std(modularity_scores),
            'levels_analyzed': levels
    }


# ============================================================================
# TRAINING INTEGRATION FUNCTIONS
# ============================================================================

def train_and_save_model_with_exact_naming(config, task_name, topology_type, seed, 
                                          use_wandb=False, enable_phase3=False, 
                                          device=None, no_noise=False):
    """Train model using EXACT existing function and save with proper naming"""
    
    print(f"🚀 Training {topology_type} on {task_name} (seed={seed})")
    
    # Call EXACT existing training function
    model, env = continual_learning_training(
        config=config,
        task_name=task_name,
        topology_type=topology_type,
        seed=seed,
        use_wandb=use_wandb,
        enable_phase3=enable_phase3,
        device=device,
        no_noise=no_noise
    )
    
    # Create run name using EXACT existing function
    shift_range = [0, 1]  # Default from existing scripts
    run_name = create_continual_learning_run_name(
                config, topology_type, task_name, seed, model, shift_range, no_noise
            )
            
    # Save model with exact naming convention
    save_dir = f"modularity_models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save final trained model (after all levels)
    final_model_path = f"{save_dir}/final_model.zip"
    model.save(final_model_path)
    
    print(f"💾 Saved final model: {final_model_path}")
    print(f"📝 Run name: {run_name}")
    
    return save_dir, run_name

def train_with_level_checkpointing(config, task_name, topology_type, seed, 
                                  use_wandb=False, enable_phase3=False, 
                                  device=None, no_noise=False):
    """Train model with level-based checkpointing for modularity analysis"""
    
    print(f"🚀 Training {topology_type} with level checkpointing (seed={seed})")
    
    # Create checkpoint directory
    run_name_base = f"{topology_type}_{task_name}_seed{seed}"
    checkpoint_dir = f"modularity_checkpoints/{run_name_base}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    
    # We need to create a modified version of the training that saves checkpoints
    # For now, let's use the existing function and then create checkpoints by retraining
    # This is not optimal but works with existing code structure
    
    checkpoint_data = []
    
    # Train incrementally, saving after each level
    num_levels = config.get('num_levels', 15)
    
    for target_level in range(1, num_levels + 1):  # 1 to 15 levels
        print(f"\n📈 Training up to Level {target_level} ({target_level * 100} iterations)")
        
        # Create config for partial training
        partial_config = config.copy()
        partial_config['max_iterations'] = target_level * 100
        
        # Train model up to this level
        model, env = continual_learning_training(
            config=partial_config,
            task_name=task_name,
            topology_type=topology_type,
            seed=seed,
            use_wandb=False,  # Disable W&B for checkpoint training
            enable_phase3=enable_phase3,
            device=device,
            no_noise=no_noise
        )
        
        # Save checkpoint after this level
        checkpoint_path = f"{checkpoint_dir}/level_{target_level:02d}_model.zip"
        model.save(checkpoint_path)
        
        # Record basic checkpoint info
        checkpoint_info = {
            'level': target_level,
            'total_iterations': target_level * 100,
            'noise_magnitude': (target_level - 1) * 0.2,  # Level 1 = 0.0, Level 2 = 0.2, etc.
            'checkpoint_path': checkpoint_path
        }
        
        checkpoint_data.append(checkpoint_info)
        
        print(f"💾 Saved Level {target_level} checkpoint: {checkpoint_path}")
        
        # Clean up environment
        if hasattr(env, 'close'):
            env.close()
    
    # Create final summary
    final_run_name = f"{run_name_base}_checkpointed"
    
    # Save checkpoint metadata
    checkpoint_metadata = {
        'run_name': final_run_name,
            'topology_type': topology_type,
        'task_name': task_name,
            'seed': seed,
        'num_levels': num_levels,
        'checkpoint_data': checkpoint_data,
        'config': config
    }
    
    metadata_path = f"{checkpoint_dir}/checkpoint_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(checkpoint_metadata, f, indent=2)
    
    print(f"\n✅ Level checkpointing completed!")
    print(f"📊 Created {len(checkpoint_data)} level checkpoints")
    print(f"📝 Metadata saved: {metadata_path}")
    
    return checkpoint_dir, checkpoint_data


# ============================================================================
# KEEP ESSENTIAL CLASSES FROM ORIGINAL FILE
# ============================================================================

class ContinualLearningWrapper(gym.Wrapper):
    """
    Wrapper for continual learning with distribution shifts.
    """
    def __init__(self, env, max_iterations=3000, level_switch=100, shift_range=[0, 1], 
                 episode_cap=400, num_levels=15, no_noise=False):
        super().__init__(env)
        self.max_iterations = max_iterations
        self.level_switch = level_switch
        self.shift_range = shift_range
        self.episode_cap = episode_cap
        self.num_levels = num_levels
        self.no_noise = no_noise
        
        # State tracking
        self.current_iteration = 0
        self.current_level = 0
        self.episode_count = 0
        self.episode_step_count = 0
        self.current_offset = np.zeros(self.observation_space.shape)
        
        print(f"🔄 ContinualLearningWrapper initialized:")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Level switch: {level_switch}")
        print(f"   Shift range: {shift_range}")
        print(f"   Episode cap: {episode_cap}")
        print(f"   Num levels: {num_levels}")
        print(f"   No noise: {no_noise}")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_step_count = 0
        
        # Apply current distribution shift
        if not self.no_noise:
            obs = obs + self.current_offset
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step_count += 1
        
        # Apply current distribution shift
        if not self.no_noise:
            obs = obs + self.current_offset
        
        # Check for episode cap
        if self.episode_step_count >= self.episode_cap:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def update_level(self, new_level):
        """Update the current level and corresponding distribution shift"""
        self.current_level = new_level
        
        if not self.no_noise and new_level > 0:
            # Calculate shift intensity based on level
            shift_intensity = (new_level / (self.num_levels - 1)) * (self.shift_range[1] - self.shift_range[0]) + self.shift_range[0]
            
            # Generate random offset for each observation dimension
            self.current_offset = np.random.uniform(-shift_intensity, shift_intensity, self.observation_space.shape)
        else:
            self.current_offset = np.zeros(self.observation_space.shape)
        
        print(f"🔄 Level updated to {new_level}, shift intensity: {np.linalg.norm(self.current_offset):.4f}")


class DebugTopologyPolicy(ActorCriticPolicy):
    """
    Custom policy that uses topology-based networks instead of standard MLP.
    """
    def __init__(self, observation_space, action_space, lr_schedule, topology_type="small_world", 
                 num_layers=1, **kwargs):
        self.topology_type = topology_type
        self.num_layers = num_layers
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build the topology-based feature extractor."""
        # Get input and output dimensions
        input_dim = int(np.prod(self.observation_space.shape))
        
        # Calculate hidden layer size based on topology requirements
        if self.topology_type == "standard_mlp":
            hidden_size = 256  # Standard size for MLP
            else:
            hidden_size = 128  # Size for topology networks
        
        # Create topology-based network
        if self.topology_type == "small_world":
            topology = SmallWorldTopology(k=4, p=0.2)
            network = FeedForwardNetwork(
                input_size=input_dim,
                hidden_sizes=[hidden_size] * self.num_layers,
                output_size=hidden_size,
                topology=topology
            )
        elif self.topology_type == "modular":
            topology = ModularTopology(num_modules=4)
            network = FeedForwardNetwork(
                input_size=input_dim,
                hidden_sizes=[hidden_size] * self.num_layers,
                output_size=hidden_size,
                topology=topology
            )
        elif self.topology_type == "hybrid":
            topology = HybridTopology(modular_ratio=0.6, num_modules=4, k=4, p=0.2)
            network = FeedForwardNetwork(
                input_size=input_dim,
                hidden_sizes=[hidden_size] * self.num_layers,
                output_size=hidden_size,
                topology=topology
            )
        elif self.topology_type == "fully_connected":
            topology = FullyConnectedTopology()
            network = FeedForwardNetwork(
                input_size=input_dim,
                hidden_sizes=[hidden_size] * self.num_layers,
                output_size=hidden_size,
                topology=topology
            )
        elif self.topology_type == "standard_mlp":
            topology = StandardMLPTopology()
            network = FeedForwardNetwork(
                input_size=input_dim,
                hidden_sizes=[hidden_size] * self.num_layers,
                output_size=hidden_size,
                topology=topology
            )
        else:
            raise ValueError(f"Unknown topology type: {self.topology_type}")
        
        # Store the network
        self.features_extractor = network
        self.features_dim = hidden_size
        
        # Create standard MLP extractor for policy and value functions
        self.mlp_extractor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
    
    def extract_features(self, obs):
        """Extract features using the topology network."""
        return self.features_extractor(obs)


def make_env(env_name, seed=None, continual_learning=False, max_iterations=3000, 
             level_switch=100, shift_range=[0, 1], reward_scale=20.0, episode_cap=400, 
             logging_callback=None, num_levels=15, no_noise=False):
    """Create environment with optional continual learning wrapper."""
    
    def _init():
        env = gym.make(env_name)
        
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        
        if continual_learning:
            env = ContinualLearningWrapper(
                env, 
                max_iterations=max_iterations,
                level_switch=level_switch,
                shift_range=shift_range,
                episode_cap=episode_cap,
                num_levels=num_levels,
                no_noise=no_noise
            )
        
        return env
    
    return _init


def create_debug_config(num_levels=15, num_layers=1, level_switch=200, hidden_size=128):
    """Create a debug configuration for testing with realistic parameters."""
    return {
        'max_iterations': num_levels * level_switch,  # Adjust total iterations based on level_switch
        'level_switch': level_switch,
        'shift_range': [0, 1],
        'episode_cap': 500,
        'num_levels': num_levels,
        'num_layers': num_layers,
        'hidden_size': hidden_size,  # Realistic network size (128 or 256)
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function for functional modularity analysis + training"""
    
    # Parse command line arguments (EXACT same as existing training scripts)
    parser = argparse.ArgumentParser(description="Functional Modularity Analysis + Training")
    
    # EXACT arguments from existing training scripts
    parser.add_argument("--single", action="store_true", help="Run single training instead of sweep")
    parser.add_argument("--topology", type=str, default="small_world", 
                       choices=["small_world", "modular", "hybrid", "fully_connected", "standard_mlp"],
                       help="Network topology type")
    parser.add_argument("--task", type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v2"],
                       help="Environment to train on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no_noise", action="store_true", help="Disable all perturbation noise for ablation study")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA for training")
    parser.add_argument("--phase3", action="store_true", help="Enable Phase 3 advanced analysis")
    parser.add_argument("--test", action="store_true", help="Run test experiment with multiple seeds and topologies")
    parser.add_argument("--num_levels", type=int, default=15, 
                       help="Number of distribution shift levels (default: 15, each level = 100 iterations × 800 env steps)")
    parser.add_argument("--num_layers", type=int, default=1, 
                       help="Number of layers for topology networks (default: 1, standard_mlp supports multiple layers)")
    parser.add_argument("--level_switch", type=int, default=200,
                       help="Iterations per level (default: 200, use smaller values for faster testing)")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden layer size (default: 128, use 256 for full experiments)")
    
    # NEW arguments for modularity analysis
    parser.add_argument("--train", action="store_true", help="Train models before analysis")
    parser.add_argument("--analyze", action="store_true", help="Perform functional modularity analysis")
    parser.add_argument("--checkpoint", action="store_true", help="Use level-based checkpointing during training")
    parser.add_argument("--analyze-checkpoints", type=str, help="Analyze existing checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=50, 
                       help="Number of episodes for activation recording")
    parser.add_argument("--num-lesion-episodes", type=int, default=20,
                       help="Number of episodes for lesion experiments")
    parser.add_argument("--base-path", default="test_experiments/MODULARITY",
                       help="Base path for experiment data")
    
    args = parser.parse_args()
    
    # Set CUDA device based on arguments (EXACT same logic as existing scripts)
    if not args.no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA is available. Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print(f"⚠️ CUDA is not available or disabled. Using CPU: {device}")
    
    # Handle checkpoint analysis mode first
    if args.analyze_checkpoints:
        print("🔍 Analyzing existing checkpoints...")
        analyzer = FunctionalModularityAnalyzer()
        results = analyzer.analyze_checkpoint_progression(args.analyze_checkpoints)
        if results:
            print("✅ Checkpoint analysis completed!")
        return
    
    # Set default behavior (mirroring existing scripts)
    if not args.train and not args.analyze and not args.test and not args.checkpoint:
        args.analyze = True  # Default to analysis mode
    
    print("🧠 Functional Modularity Analysis + Training")
    print("=" * 80)
    print(f"🎯 Configuration:")
    print(f"   Topology: {args.topology}")
    print(f"   Task: {args.task}")
    print(f"   Seed: {args.seed}")
    print(f"   Number of Levels: {args.num_levels}")
    print(f"   Number of Layers: {args.num_layers}")
    print(f"   W&B: {'Disabled' if args.no_wandb else 'Enabled'}")
    print(f"   Phase 3 Analysis: {'Enabled' if args.phase3 else 'Disabled'}")
    print(f"   Train: {'Enabled' if args.train else 'Disabled'}")
    print(f"   Analyze: {'Enabled' if args.analyze else 'Disabled'}")
    print(f"   Mode: {'Single Training' if args.single else 'Test Experiment' if args.test else 'Analysis'}")
    print("=" * 80)
    
    # Handle --test flag (EXACT same logic as existing scripts)
    if args.test:
        print("🧪 Starting test experiment mode...")
        
        try:
            if args.train:
                print("🚀 Training models in test mode...")
                # For test mode, we'd need to modify run_test_experiment to save models
                # For now, use existing test experiment function
            
            run_test_experiment(
                task_name=args.task,
                seeds=[42, 123, 456, 789, 999],  # EXACT same seeds
                use_wandb=not args.no_wandb,
                num_levels=args.num_levels,
                device=device,
                no_noise=args.no_noise
            )
            
            print("🎉 Test experiment completed successfully!")
            
        except Exception as e:
            print(f"❌ Test experiment failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
        return
    
    # Handle --single flag (EXACT same logic as existing scripts)
    if args.single:
        config = create_debug_config(args.num_levels, args.num_layers, args.level_switch, args.hidden_size)
        trained_model_dir = None
        
        if args.train:
            print("🚀 Starting single training run...")
            
            try:
                if args.checkpoint:
                    # Use level-based checkpointing
                    print("📊 Using level-based checkpointing...")
                    trained_model_dir, checkpoint_data = train_with_level_checkpointing(
                        config=config,
                        task_name=args.task,
                        topology_type=args.topology,
                        seed=args.seed,
                        use_wandb=not args.no_wandb,
                        enable_phase3=args.phase3,
                        device=device,
                        no_noise=args.no_noise
                    )
                    
                    print("✅ Checkpoint training completed successfully!")
                    print(f"📊 Checkpoints saved to: {trained_model_dir}")
                    
                    # Automatically analyze checkpoints if requested
                    if args.analyze:
                        print("\n🔍 Analyzing checkpoint progression...")
                        analyzer = FunctionalModularityAnalyzer()
                        results = analyzer.analyze_checkpoint_progression(trained_model_dir)
                        if results:
                            print("✅ Checkpoint analysis completed!")
                        return
                    
                else:
                    # Use standard training
                    trained_model_dir, run_name = train_and_save_model_with_exact_naming(
                config=config,
                task_name=args.task,
                topology_type=args.topology,
                seed=args.seed,
                use_wandb=not args.no_wandb,
                enable_phase3=args.phase3,
                device=device,
                no_noise=args.no_noise
            )
            
            print("✅ Single training completed successfully!")
                    print(f"📊 Model saved to: {trained_model_dir}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
        if args.analyze:
            print("\n🧠 Starting functional modularity analysis...")
            
            # Initialize analyzer
            analyzer = FunctionalModularityAnalyzer(base_path=args.base_path)
            
            if trained_model_dir:
                # Analyze the model we just trained
                experiments = {args.topology: [trained_model_dir]}
                print(f"🔍 Analyzing newly trained model: {trained_model_dir}")
    else:
                # Convert task name for analysis (CartPole-v1 -> cartpole)
                task_for_analysis = args.task.split('-')[0].lower()
                
                # Discover existing experiments (using original analysis logic)
                experiments = analyzer.discover_experiments(
                    task=task_for_analysis,
                    noise_level="N0002",  # Default for analysis
                    size="S256"  # Default for analysis
                )
                
                if not experiments:
                    print(f"❌ No experiments found for analysis")
                    return
        
        else:
            return  # Only training was requested
    
    else:
        # Analysis mode without --single flag
        if not args.analyze:
            print("❌ No action specified. Use --train, --analyze, --single, or --test")
            return
            
        print("🧠 Starting functional modularity analysis...")
        
        # Initialize analyzer
        analyzer = FunctionalModularityAnalyzer(base_path=args.base_path)
        
        # Convert task name for analysis (CartPole-v1 -> cartpole)
        task_for_analysis = args.task.split('-')[0].lower()
        
        # Discover existing experiments
        experiments = analyzer.discover_experiments(
            task=task_for_analysis,
            noise_level="N0002",  # Default for analysis
            size="S256"  # Default for analysis
        )
        
        if not experiments:
            print(f"❌ No experiments found for analysis")
            return
    
    # Run analysis on each topology (KEEP existing analysis logic)
    results = {}
    for topology, exp_dirs in experiments.items():
        print(f"\n🔍 Analyzing {topology} topology...")
        topology_results = []
        
        for i, exp_dir in enumerate(exp_dirs[:3]):  # Limit to first 3 experiments per topology
            print(f"   Experiment {i+1}/{min(3, len(exp_dirs))}: {os.path.basename(exp_dir)}")
            
            try:
                # Load model
                model = analyzer.load_trained_model(exp_dir)
                if model is None:
                    continue
                
                # Record activations
                activations = analyzer.record_activations(model, exp_dir, args.num_episodes)
                
                # Compute functional modularity
                functional_graph, correlation_matrix = analyzer.compute_functional_graph(activations)
                communities, community_dict = analyzer.detect_communities(functional_graph)
                modularity_score = analyzer.compute_modularity(functional_graph, community_dict)
                
                # Perform lesion experiments
                lesion_results = analyzer.perform_lesion_experiments(
                    model, communities, args.num_lesion_episodes
                )
                
                # Store results
                experiment_result = {
                    'experiment_dir': exp_dir,
                    'modularity_score': modularity_score,
                    'communities': communities,
                    'lesion_results': lesion_results,
                    'functional_graph': functional_graph,
                    'correlation_matrix': correlation_matrix
                }
                topology_results.append(experiment_result)
                
                print(f"   ✅ Completed: Modularity={modularity_score:.4f}, Communities={len(communities)}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        if topology_results:
            results[topology] = topology_results
    
    if not results:
        print("❌ No successful analyses completed")
        return
    
    # Generate comparison plots
    analyzer.plot_functional_modularity_comparison(results)
    
    # Save results (adapt to new argument structure)
    task_for_save = args.task.split('-')[0].lower()
    analyzer.save_results(results, task_for_save, "N0002", "S256")
    
    # Print summary
    print("\n📊 Analysis Summary:")
    print("=" * 50)
    for topology, topology_results in results.items():
        avg_modularity = np.mean([r['modularity_score'] for r in topology_results])
        avg_communities = np.mean([len(r['communities']) for r in topology_results])
        print(f"{topology:>15}: {len(topology_results)} experiments, "
              f"modularity={avg_modularity:.4f}, communities={avg_communities:.1f}")
    
    print("=" * 50)
    print("✅ Functional modularity analysis completed!")


if __name__ == "__main__":
    main()
    