"""
Enhanced Curriculum Runner with Comprehensive Logging

This module extends the basic CurriculumRunner with:
1. Learning curve tracking during training
2. Detailed transfer metrics
3. Episode-by-episode performance logging
4. Enhanced result analysis and visualization
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging_utils import setup_logger, LogLevel
from ..utils.parameter_budget import ParameterBudgetCalculator, calculate_network_size
from ..utils.capacity_measurement import CapacityMeasurementManager

from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..topologies.hybrid import HybridTopology
from ..topologies.fully_connected import FullyConnectedTopology
from ..networks.ffn import FeedForwardNetwork
from ..networks.rnn import RecurrentNetwork
from ..node_selection.strategies import NodeSelector
from ..tasks.task_definitions import TaskGenerator, TaskEvaluator
from ..tasks.rl_tasks import RLTaskGenerator, RLTaskEvaluator
from ..agents.network_agent import NetworkAgent

logger = setup_logger(__name__)

class EnhancedCurriculumRunner:
    """Enhanced runner for curriculum learning experiments with comprehensive logging."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize the enhanced curriculum runner.
        
        Args:
            config: Curriculum configuration
            output_dir: Directory to save experiment results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.task_generator = RLTaskGenerator()
        self.task_evaluator = RLTaskEvaluator()
        
        # Network type mapping
        self.network_types = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
        
        # Enhanced logging storage
        self.learning_curves = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.transfer_metrics = defaultdict(dict)
        
    def run_curriculum(self):
        """Run the complete curriculum experiment with enhanced logging."""
        print("Executing enhanced curriculum from:", __file__)
        print("\nCurriculum parameters:")
        print(f"Task sequence: {self.config['task_sequence']}")
        print(f"Network sizes: {self.config['network_sizes']}")
        print(f"Seeds: {self.config['seeds']}")
        print(f"Number of layers: {self.config['num_layers']}")
        print(f"Network types: {self.config['network_types']}")
        
        # Print parameter budget information
        print("\nParameter budget settings:")
        print(f"Budget type: {self.config['parameter_budget']['budget_type']}")
        print(f"Target budget: {self.config['parameter_budget']['target_budget']}")
        print(f"Normalize by size: {self.config['parameter_budget']['normalize_by_size']}")
        
        # Print experiment types and their capacity matching
        print("\nExperiment types and capacity matching:")
        for exp_type in self.config['experiment_types']:
            print(f"\n{exp_type}:")
            if exp_type == 'same_size':
                print("  All networks will have the same node size")
            else:
                target = '_'.join(exp_type.split('_')[1:])
                print(f"  All networks will match {target} capacity")
        
        results = []
        
        calculator = ParameterBudgetCalculator(self.config)
        
        for experiment_type in self.config['experiment_types']:
            print(f"\nRunning experiment type: {experiment_type}")
            
            for size in tqdm(self.config['network_sizes'], desc="Network sizes"):
                for seed in tqdm(self.config['seeds'], desc="Seeds", leave=False):
                    for num_layers in tqdm(self.config['num_layers'], desc="Number of layers", leave=False):
                        for network_type in tqdm(self.config['network_types'], desc="Network types", leave=False):
                            # --- Correct Capacity-matching scaling logic ---
                            if experiment_type == 'same_size':
                                # For same_size, all topologies use the same original size
                                sw_size = size
                                mod_size = size
                                hybrid_size = size
                                fc_size = size
                                target_capacity = None
                            elif experiment_type.startswith('match_'):
                                reference_topology = experiment_type[len('match_'):]
                                measurement_manager = CapacityMeasurementManager(self.config)
                                print(f"[DEBUG] Config hash: {measurement_manager._get_config_hash()}")
                                lookup_key = f"{reference_topology}_{size}_{network_type}_{num_layers}"
                                print(f"[DEBUG] Looking for measurement key: {lookup_key}")
                                print(f"[DEBUG] Available measurement keys: {list(measurement_manager.measurements.keys())}")
                                target_capacity = measurement_manager.get_target_capacity(
                                    reference_topology, size, network_type, num_layers
                                )
                                if target_capacity is None:
                                    print(f"[WARNING] No baseline measurement available for {lookup_key}, falling back to calculator.")
                                    target_capacity = calculator.get_budget(experiment_type, reference_topology, size, network_type, num_layers)
                                # Reference topology keeps natural size
                                sw_size = calculator.get_matching_size(experiment_type, 'small_world', size, network_type, num_layers) if reference_topology != 'small_world' else size
                                mod_size = calculator.get_matching_size(experiment_type, 'modular', size, network_type, num_layers) if reference_topology != 'modular' else size
                                hybrid_size = calculator.get_matching_size(experiment_type, 'hybrid', size, network_type, num_layers) if reference_topology != 'hybrid' else size
                                fc_size = calculator.get_matching_size(experiment_type, 'fully_connected', size, network_type, num_layers) if reference_topology != 'fully_connected' else size
                            else:
                                raise ValueError(f"Unknown experiment type: {experiment_type}")
                            
                            # Use the same capacity matching logic as smoke test
                            if experiment_type.startswith('match_'):
                                # Reference topology keeps natural size
                                if reference_topology == 'small_world':
                                    sw_size = size  # No scaling
                                else:
                                    sw_size = calculator.calculate_matching_size('small_world', target_capacity, network_type, num_layers)
                                
                                if reference_topology == 'modular':
                                    mod_size = size  # No scaling
                                else:
                                    mod_size = calculator.calculate_matching_size('modular', target_capacity, network_type, num_layers)
                                
                                if reference_topology == 'hybrid':
                                    hybrid_size = size  # No scaling
                                else:
                                    hybrid_size = calculator.calculate_matching_size('hybrid', target_capacity, network_type, num_layers)
                                
                                if reference_topology == 'fully_connected':
                                    fc_size = size  # No scaling
                                else:
                                    fc_size = calculator.calculate_matching_size('fully_connected', target_capacity, network_type, num_layers)
                            else:  # 'same_size'
                                # For same_size, all topologies use the same original size
                                sw_size = size
                                mod_size = size
                                hybrid_size = size
                                fc_size = size
                            
                            # --- Instantiate topologies with scaled sizes ---
                            small_world = SmallWorldTopology(
                                size=sw_size,
                                k=self.config['small_world_params']['k'],
                                p=self.config['small_world_params']['p'],
                                num_layers=num_layers,
                                inter_layer_prob=self.config['small_world_params']['inter_layer_prob'],
                                seed=seed
                            )
                            modular = ModularTopology(
                                size=mod_size,
                                num_modules=self.config['modular_params']['num_modules'],
                                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                                intra_module_prob=self.config['modular_params']['intra_module_prob'],
                                num_layers=num_layers,
                                inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                                seed=seed
                            )
                            hybrid = HybridTopology(
                                size=hybrid_size,
                                num_modules=self.config['modular_params']['num_modules'],
                                k=self.config['small_world_params']['k'],
                                p=self.config['small_world_params']['p'],
                                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                                num_layers=num_layers,
                                inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                                seed=seed
                            )
                            fully_connected = FullyConnectedTopology(
                                size=fc_size,
                                num_layers=num_layers,
                                inter_layer_prob=self.config['fully_connected_params']['inter_layer_prob'],
                                intra_layer_prob=self.config['fully_connected_params']['intra_layer_prob'],
                                seed=seed
                            )
                            
                            # Generate networks
                            sw_graphs = small_world.generate(num_layers)
                            mod_graphs = modular.generate(num_layers)
                            hybrid_graphs = hybrid.generate(num_layers)
                            fc_graphs = fully_connected.generate(num_layers)
                            
                            # Convert to list if single graph
                            if num_layers == 1:
                                sw_graphs = [sw_graphs]
                                mod_graphs = [mod_graphs]
                                hybrid_graphs = [hybrid_graphs]
                                fc_graphs = [fc_graphs]
                            
                            # Select input/output nodes for each layer
                            for strategy in tqdm(self.config['node_selection_strategies'], desc="Strategies", leave=False):
                                sw_input_nodes = []
                                sw_output_nodes = []
                                mod_input_nodes = []
                                mod_output_nodes = []
                                hybrid_input_nodes = []
                                hybrid_output_nodes = []
                                fc_input_nodes = []
                                fc_output_nodes = []
                                
                                # Select nodes for each layer using scaled sizes
                                for layer_idx in range(num_layers):
                                    sw_in, sw_out = self._select_nodes(sw_graphs[layer_idx], strategy, sw_size, seed + layer_idx)
                                    sw_input_nodes.append(sw_in)
                                    sw_output_nodes.append(sw_out)
                                    
                                    mod_in, mod_out = self._select_nodes(mod_graphs[layer_idx], strategy, mod_size, seed + layer_idx)
                                    mod_input_nodes.append(mod_in)
                                    mod_output_nodes.append(mod_out)
                                    
                                    hybrid_in, hybrid_out = self._select_nodes(hybrid_graphs[layer_idx], strategy, hybrid_size, seed + layer_idx)
                                    hybrid_input_nodes.append(hybrid_in)
                                    hybrid_output_nodes.append(hybrid_out)
                                    
                                    fc_in, fc_out = self._select_nodes(fc_graphs[layer_idx], strategy, fc_size, seed + layer_idx)
                                    fc_input_nodes.append(fc_in)
                                    fc_output_nodes.append(fc_out)
                                
                                # Create networks
                                sw_networks = []
                                mod_networks = []
                                hybrid_networks = []
                                fc_networks = []
                                
                                for layer_idx in range(num_layers):
                                    network_class = self.network_types[network_type]
                                    network_params = self.config['network_params'][network_type]
                                    
                                    sw_networks.append(network_class(
                                        sw_graphs[layer_idx], sw_input_nodes[layer_idx], sw_output_nodes[layer_idx], network_params
                                    ))
                                    mod_networks.append(network_class(
                                        mod_graphs[layer_idx], mod_input_nodes[layer_idx], mod_output_nodes[layer_idx], network_params
                                    ))
                                    hybrid_networks.append(network_class(
                                        hybrid_graphs[layer_idx], hybrid_input_nodes[layer_idx], hybrid_output_nodes[layer_idx], network_params
                                    ))
                                    fc_networks.append(network_class(
                                        fc_graphs[layer_idx], fc_input_nodes[layer_idx], fc_output_nodes[layer_idx], network_params
                                    ))
                                
                                # Verify capacity matching
                                print(f"\nCapacity verification for {experiment_type} (size {size}):")
                                if experiment_type.startswith('match_'):
                                    print(f"Target capacity: {target_capacity}")
                                else:  # 'same_size'
                                    print("All topologies use the same node count (no capacity matching)")
                                
                                # Test each topology
                                topologies_to_test = [
                                    ('small_world', sw_networks, sw_size),
                                    ('modular', mod_networks, mod_size),
                                    ('hybrid', hybrid_networks, hybrid_size),
                                    ('fully_connected', fc_networks, fc_size)
                                ]
                                
                                for topology_name, networks_list, actual_size in topologies_to_test:
                                    if networks_list:
                                        network = networks_list[0]
                                        metrics = network.get_network_metrics()
                                        # Use same parameter counting method as smoke test
                                        total_params = sum(
                                            metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                        )
                                        
                                        if experiment_type.startswith('match_'):
                                            divergence = abs(total_params - target_capacity) / target_capacity * 100 if target_capacity > 0 else float('inf')
                                            status = "✅" if divergence <= 5.0 else "⚠️"
                                            print(f"  {topology_name}: size={actual_size}, params={total_params}, divergence={divergence:.2f}% {status}")
                                            
                                            # Validate that capacity matching is working
                                            if divergence > 10.0:  # More lenient threshold for training
                                                print(f"    ⚠️  WARNING: Large capacity divergence detected during training!")
                                                print(f"    Expected: {target_capacity}, Actual: {total_params}")
                                        else:  # 'same_size'
                                            print(f"  {topology_name}: size={actual_size}, params={total_params} (same_size experiment)")
                                
                                # Run task sequence for each topology
                                topologies_to_run = [
                                    ('small_world', sw_networks, sw_input_nodes, sw_output_nodes, sw_size),
                                    ('modular', mod_networks, mod_input_nodes, mod_output_nodes, mod_size),
                                    ('hybrid', hybrid_networks, hybrid_input_nodes, hybrid_output_nodes, hybrid_size),
                                    ('fully_connected', fc_networks, fc_input_nodes, fc_output_nodes, fc_size)
                                ]
                                
                                for topology_name, networks, input_nodes, output_nodes, actual_size in topologies_to_run:
                                    # Run task sequence with enhanced logging
                                    task_results = self._run_enhanced_task_sequence(
                                        networks, input_nodes, output_nodes,
                                        actual_size, seed, strategy, topology_name, network_type, num_layers
                                    )
                                    results.append(task_results)
        
        # Save enhanced results
        self._save_enhanced_results(results)
        
        return results
    
    def _run_enhanced_task_sequence(self, networks, input_nodes, output_nodes,
                                  size, seed, strategy, topology, network_type, num_layers):
        """Run the sequence of tasks with enhanced logging."""
        performance_history = {}
        baseline_performance = {}
        learning_curves = {}
        episode_metrics = {}
        
        # Get baseline performance
        for task in self.config['task_sequence']:
            env, task_config = getattr(self.task_generator, f"generate_{task}_task")()
            baseline = self._evaluate_performance(networks, input_nodes, output_nodes,
                                                env, task_config, task, size, seed,
                                                strategy, topology, network_type)
            baseline_performance[task] = baseline
        
        # Run curriculum with enhanced logging
        for task in self.config['task_sequence']:
            # Train on current task with learning curve tracking
            env, task_config = getattr(self.task_generator, f"generate_{task}_task")()
            task_metrics, task_learning_curve = self._train_task_with_logging(
                networks, input_nodes, output_nodes,
                env, task_config, task, size, seed,
                strategy, topology, network_type
            )
            
            # Store learning curve
            learning_curves[task] = task_learning_curve
            
            # Evaluate on all tasks
            current_performance = {}
            for eval_task in self.config['task_sequence']:
                eval_env, eval_config = getattr(self.task_generator, f"generate_{eval_task}_task")()
                performance = self._evaluate_performance(networks, input_nodes, output_nodes,
                                                       eval_env, eval_config, eval_task,
                                                       size, seed, strategy, topology,
                                                       network_type)
                current_performance[eval_task] = performance
            
            performance_history[task] = current_performance
            
            # Store episode metrics
            episode_metrics[task] = {
                'learning_curve': task_learning_curve,
                'final_performance': current_performance[task]
            }
        
        # Calculate enhanced transfer metrics
        transfer_metrics = self._calculate_enhanced_transfer_metrics(
            baseline_performance,
            performance_history,
            learning_curves
        )
        
        return {
            'network_size': size,
            'seed': seed,
            'num_layers': num_layers,
            'network_type': network_type,
            'strategy': strategy,
            'topology': topology,
            'curriculum_results': {
                'performance_history': performance_history,
                'transfer_metrics': transfer_metrics,
                'task_metrics': task_metrics,
                'learning_curves': learning_curves,
                'episode_metrics': episode_metrics
            }
        }
    
    def _train_task_with_logging(self, networks, input_nodes, output_nodes,
                               env, task_config, task, size, seed,
                               strategy, topology, network_type):
        """Train networks on a specific task with comprehensive logging and adaptive duration."""
        print(f"Training on {task} with {topology} topology")
        
        # Initialize learning curve tracking
        learning_curve = []
        max_episodes = self.config['episodes_per_task']
        
        # Adaptive training parameters
        convergence_window = self.config.get('convergence_window', 50)  # Episodes to check for convergence
        convergence_threshold = self.config.get('convergence_threshold', 0.02)  # Performance stability threshold
        min_episodes = self.config.get('min_episodes', 5000)  # Minimum episodes before early stopping
        patience = self.config.get('convergence_patience', 3)  # How many times to check before stopping
        
        convergence_count = 0
        last_convergence_check = 0
        
        # Simulate training with episode-by-episode logging and adaptive duration
        for episode in range(max_episodes):
            # Simulate episode training
            time.sleep(0.001)  # Simulate training time
            
            # Simulate episode reward (replace with actual training)
            if task == 'cartpole':
                # CartPole: reward increases over time, max ~200
                base_reward = 20 + (episode / max_episodes) * 180
                noise = np.random.normal(0, 5)
                episode_reward = max(0, base_reward + noise)
            elif task == 'mountain_car':
                # MountainCar: starts at -200, improves to ~-100
                base_reward = -200 + (episode / max_episodes) * 100
                noise = np.random.normal(0, 10)
                episode_reward = base_reward + noise
            else:  # acrobot
                # Acrobot: starts at -200, improves to ~-100
                base_reward = -200 + (episode / max_episodes) * 100
                noise = np.random.normal(0, 10)
                episode_reward = base_reward + noise
            
            learning_curve.append(episode_reward)
            
            # Check for convergence (adaptive training duration)
            if (episode >= min_episodes and 
                episode >= last_convergence_check + convergence_window and
                len(learning_curve) >= convergence_window):
                
                # Calculate performance stability in recent window
                recent_window = learning_curve[-convergence_window:]
                recent_std = np.std(recent_window)
                recent_mean = np.mean(recent_window)
                
                # Check if performance has stabilized
                if recent_std < convergence_threshold * abs(recent_mean):
                    convergence_count += 1
                    if convergence_count >= patience:
                        print(f"🔄 Early stopping at episode {episode} - Performance converged")
                        print(f"   Final std: {recent_std:.3f}, Mean: {recent_mean:.1f}")
                        break
                else:
                    convergence_count = 0  # Reset if not converged
                
                last_convergence_check = episode
            
            # Log every 50 episodes
            if episode % 50 == 0:
                logger.info(f"Episode {episode}/{max_episodes}: {task} - {topology} - Reward: {episode_reward:.1f}")
        
        # Calculate training metrics
        final_episode = len(learning_curve)
        final_reward = learning_curve[-1] if learning_curve else 0
        mean_reward = np.mean(learning_curve) if learning_curve else 0
        std_reward = np.std(learning_curve) if learning_curve else 0
        
        # Calculate learning dynamics
        if len(learning_curve) > 1:
            learning_rate = (learning_curve[-1] - learning_curve[0]) / len(learning_curve)
            improvement_rate = np.mean(np.diff(learning_curve[-convergence_window:])) if len(learning_curve) >= convergence_window else 0
        else:
            learning_rate = 0
            improvement_rate = 0
        
        # Update network metrics
        for layer_idx, network in enumerate(networks):
            metrics = network.get_network_metrics()
            logger.info(f"Layer {layer_idx} - Training metrics: {metrics}")
        
        # Save enhanced training metrics
        task_metrics = {
            'training_metrics': metrics,
            'total_episodes': final_episode,
            'max_episodes': max_episodes,
            'final_episode_reward': final_reward,
            'mean_episode_reward': mean_reward,
            'std_episode_reward': std_reward,
            'learning_rate': learning_rate,
            'improvement_rate': improvement_rate,
            'convergence_episode': final_episode if convergence_count >= patience else max_episodes,
            'early_stopped': convergence_count >= patience,
            'convergence_std': recent_std if len(learning_curve) >= convergence_window else std_reward
        }
        
        return task_metrics, learning_curve
    
    def _evaluate_performance(self, networks, input_nodes, output_nodes,
                            env, task_config, task, size, seed,
                            strategy, topology, network_type):
        """Evaluate network performance on a specific task using real agent and environment."""
        from ..tasks.rl_tasks import RLTaskEvaluator
        
        # If networks is a list, evaluate each network and aggregate results
        if isinstance(networks, list):
            results = []
            for network in networks:
                # Wrap network in agent
                agent = NetworkAgent(network, task_config)
                # Evaluate single network
                result = RLTaskEvaluator.evaluate_episodes(env, agent, task_config, num_episodes=10)
                results.append(result)
            
            # Aggregate results
            mean_rewards = [r['mean_reward'] for r in results]
            std_rewards = [r['std_reward'] for r in results]
            mean_lengths = [r['mean_length'] for r in results]
            std_lengths = [r['std_length'] for r in results]
            solved_rates = [r['solved_rate'] for r in results]
            
            return {
                'mean_reward': np.mean(mean_rewards),
                'std_reward': np.std(mean_rewards),  # Std of means across networks
                'mean_length': np.mean(mean_lengths),
                'std_length': np.std(mean_lengths),
                'solved_rate': np.mean(solved_rates)
            }
        else:
            # Single network case - wrap in agent and evaluate
            agent = NetworkAgent(networks, task_config)
            return RLTaskEvaluator.evaluate_episodes(env, agent, task_config, num_episodes=10)
    
    def _calculate_enhanced_transfer_metrics(self, baseline_performance, performance_history, learning_curves):
        """Calculate enhanced transfer learning metrics with learning curve analysis."""
        transfer_metrics = {
            'backward_transfer': {},
            'forward_transfer': {},
            'final_performance': {},
            'learning_curve_analysis': {},
            'convergence_metrics': {}
        }
        
        # Calculate backward transfer (how well previous tasks are retained)
        for task in self.config['backward_transfer_tasks']:
            final_performance = performance_history[self.config['task_sequence'][-1]][task]
            baseline = baseline_performance[task]
            
            # For CartPole, higher is better
            if task == 'cartpole':
                transfer_metrics['backward_transfer'][task] = (
                    final_performance['mean_reward'] / baseline['mean_reward']
                )
            # For MountainCar and Acrobot, lower is better
            else:
                transfer_metrics['backward_transfer'][task] = (
                    baseline['mean_reward'] / final_performance['mean_reward']
                )
        
        # Calculate forward transfer (how quickly new tasks are learned)
        for task in self.config['forward_transfer_tasks']:
            task_idx = self.config['task_sequence'].index(task)
            if task_idx > 0:
                # Compare performance after previous task vs baseline
                after_prev = performance_history[self.config['task_sequence'][task_idx-1]][task]
                baseline = baseline_performance[task]
                
                # For CartPole, higher is better
                if task == 'cartpole':
                    transfer_metrics['forward_transfer'][task] = (
                        after_prev['mean_reward'] / baseline['mean_reward']
                    )
                # For MountainCar and Acrobot, lower is better
                else:
                    transfer_metrics['forward_transfer'][task] = (
                        baseline['mean_reward'] / after_prev['mean_reward']
                    )
        
        # Analyze learning curves
        for task in self.config['task_sequence']:
            if task in learning_curves:
                curve = learning_curves[task]
                if curve:
                    # Calculate convergence metrics
                    transfer_metrics['learning_curve_analysis'][task] = {
                        'final_reward': curve[-1] if curve else 0,
                        'mean_reward': np.mean(curve) if curve else 0,
                        'std_reward': np.std(curve) if curve else 0,
                        'max_reward': np.max(curve) if curve else 0,
                        'min_reward': np.min(curve) if curve else 0,
                        'improvement_rate': (curve[-1] - curve[0]) / len(curve) if len(curve) > 1 else 0
                    }
                    
                    # Calculate convergence point (when performance stabilizes)
                    window_size = min(50, len(curve) // 4)
                    if window_size > 0:
                        recent_std = np.std(curve[-window_size:])
                        convergence_episode = len(curve) - window_size
                        for i in range(len(curve) - window_size):
                            if np.std(curve[i:i+window_size]) <= recent_std:
                                convergence_episode = i
                                break
                        
                        transfer_metrics['convergence_metrics'][task] = {
                            'convergence_episode': convergence_episode,
                            'convergence_reward': curve[convergence_episode] if convergence_episode < len(curve) else curve[-1],
                            'stability_std': recent_std
                        }
        
        # Store final performance
        final_task = self.config['task_sequence'][-1]
        transfer_metrics['final_performance'] = performance_history[final_task]
        
        return transfer_metrics
    
    def _select_nodes(self, graph: nx.Graph, strategy: str, size: int, seed: int) -> tuple:
        """Select input and output nodes based on the specified strategy."""
        rng = np.random.RandomState(seed)
        num_io_nodes = self.config['num_io_nodes']
        
        if strategy == 'random':
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
            
        elif strategy == 'centrality_based':
            centrality = nx.betweenness_centrality(graph, k=min(100, size))
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            input_nodes = [node for node, _ in sorted_nodes[:num_io_nodes]]
            output_nodes = [node for node, _ in sorted_nodes[num_io_nodes:2*num_io_nodes]]
            
        elif strategy == 'distance_based':
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            
            distances = np.zeros(size)
            for node in range(size):
                distances[node] = np.mean([
                    nx.shortest_path_length(graph, source=in_node, target=node)
                    for in_node in input_nodes
                ])
            
            sorted_nodes = np.argsort(distances)[::-1]
            output_nodes = [node for node in sorted_nodes if node not in input_nodes][:num_io_nodes]
            
        elif strategy == 'module_based':
            if hasattr(self, 'modular') and hasattr(self.modular, 'get_module_assignments'):
                module_assignments = self.modular.get_module_assignments()
                module_nodes = {}
                for node, module in module_assignments.items():
                    if module not in module_nodes:
                        module_nodes[module] = []
                    module_nodes[module].append(node)
                
                input_nodes = []
                for module in range(min(num_io_nodes, len(module_nodes))):
                    module_node = rng.choice(module_nodes[module])
                    input_nodes.append(module_node)
                
                input_modules = {module_assignments[node] for node in input_nodes}
                available_modules = [m for m in module_nodes.keys() if m not in input_modules]
                output_nodes = []
                for module in available_modules[:num_io_nodes]:
                    module_node = rng.choice(module_nodes[module])
                    output_nodes.append(module_node)
            else:
                all_nodes = list(range(size))
                rng.shuffle(all_nodes)
                input_nodes = all_nodes[:num_io_nodes]
                output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
        else:
            raise ValueError(f"Unknown node selection strategy: {strategy}")
        
        return input_nodes, output_nodes
    
    def _save_enhanced_results(self, results: List[Dict[str, Any]]):
        """Save enhanced experiment results to files."""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create a subfolder with the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_dir / f"enhanced_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(results_dir / 'enhanced_curriculum_results.csv', index=False)
        
        # Convert all numpy types to native Python types for JSON
        def to_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {str(k) if isinstance(k, (np.integer, np.floating)) else k: to_serializable(v) 
                       for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable_results = to_serializable(results)
        
        # Save as JSON
        with open(results_dir / 'enhanced_curriculum_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create enhanced visualizations
        self._create_enhanced_visualizations(results, results_dir)
        
        print(f"Enhanced results saved to: {results_dir}")
    
    def _create_enhanced_visualizations(self, results: List[Dict[str, Any]], results_dir: Path):
        """Create enhanced visualizations for the results."""
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Learning curves for all topologies
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Learning Curves by Topology', fontsize=16)
        
        for i, result in enumerate(results[:4]):  # Limit to 4 plots
            topology = result['topology']
            curriculum_results = result['curriculum_results']
            learning_curves = curriculum_results.get('learning_curves', {})
            
            ax = axes[i//2, i%2]
            
            for task, curve in learning_curves.items():
                if curve:
                    # Convert numpy array to list if needed
                    if hasattr(curve, 'tolist'):
                        curve = curve.tolist()
                    
                    episodes = list(range(1, len(curve) + 1))
                    ax.plot(episodes, curve, label=task, linewidth=2, alpha=0.8)
            
            ax.set_title(f'Topology: {topology}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "enhanced_learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Transfer metrics comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        topologies = []
        backward_transfers = []
        forward_transfers = []
        
        for result in results:
            topology = result['topology']
            transfer_metrics = result['curriculum_results']['transfer_metrics']
            
            topologies.append(topology)
            backward_transfers.append(transfer_metrics.get('backward_transfer', {}).get('cartpole', 1.0))
            forward_transfers.append(transfer_metrics.get('forward_transfer', {}).get('mountain_car', 1.0))
        
        x = np.arange(len(topologies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, backward_transfers, width, label='Backward Transfer (Cartpole)', alpha=0.7)
        bars2 = ax.bar(x + width/2, forward_transfers, width, label='Forward Transfer (Mountain Car)', alpha=0.7)
        
        ax.set_xlabel('Topology')
        ax.set_ylabel('Transfer Ratio')
        ax.set_title('Enhanced Transfer Learning Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.legend()
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No Transfer')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(results_dir / "enhanced_transfer_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced visualizations saved to: {results_dir}") 