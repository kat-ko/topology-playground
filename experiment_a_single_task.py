"""
EXPERIMENT A: Single Task, Single Topology Comparison
Objective: Evaluate if topology alone impacts learning for a single task.

Factor	Setting
Task	CartPole-v1
Network type	ffn
Layers	1
Size	50
Topologies	small_world, modular, fully_connected, hybrid
Seed	42
Node selection	random
Episodes	200
Env steps per task	5,000
Evaluation episodes	10

Why useful:
- Reveals whether certain topologies start with higher/better reward curves.
- Easy to visualize differences in learning speed or stability.
"""

import logging
import torch
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from src.curriculum.enhanced_runner import EnhancedCurriculumRunner
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager
from config.curriculum_config import CurriculumConfig

class ExperimentAConfig:
    """Configuration for Experiment A: Single Task, Single Topology Comparison."""
    
    def __init__(self):
        # Core experiment settings
        self.network_sizes = [25]
        self.num_layers = [1]
        self.network_types = ['ffn']
        self.experiment_types = ['match_small_world']
        self.task_sequence = ['cartpole']
        self.seeds = [42]
        self.node_selection_strategies = ['random']
        
        # Training parameters
        self.episodes_per_task = 200
        self.evaluation_episodes = 10
        self.max_env_steps_per_task = 5000
        
        # Transfer learning (minimal for single task)
        self.backward_transfer_tasks = []
        self.forward_transfer_tasks = []
        
        # Forgetting test (disabled for single task)
        self.forgetting_test = {
            'enabled': False,
            'retention_interval': 1,
            'retention_episodes': 2,
            'forgetting_threshold': 0.8,
            'retention_threshold': 0.9
        }
        
        # Parameter budget
        self.parameter_budget = {
            'budget_type': 'edges',
            'target_budget': 10000,
            'normalize_by_size': True
        }
        
        # Network parameters
        self.network_params = {
            'ffn': {
                'learning_rate': 0.01,
                'activation': 'tanh'
            },
            'rnn': {
                'learning_rate': 0.01,
                'hidden_size': 32,
                'activation': 'tanh'
            }
        }
        
        # Topology parameters (required for capacity measurement)
        self.fully_connected_params = {
            'intra_layer_prob': 0.8,
            'inter_layer_prob': 0.5
        }
        self.small_world_params = {
            'k': 4,
            'p': 0.3,
            'inter_layer_prob': 0.5
        }
        self.modular_params = {
            'num_modules': 4,
            'intra_module_prob': 0.8,
            'inter_module_prob': 0.1,
            'inter_layer_prob': 0.5
        }
        
        # IO configuration
        self.num_io_nodes = 4
        
        # RL parameters
        self.rl_params = {
            'learning_rate': 0.01,
            'gamma': 0.99,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01
        }
        
        # Capacity mapping - DISABLED to match smoke test
        self.use_capacity_mapping = False
        
        # Topologies list (all topologies for Experiment A)
        self.topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
        
    def to_dict(self):
        """Convert to dictionary format for compatibility."""
        return {
            'network_sizes': self.network_sizes,
            'num_layers': self.num_layers,
            'network_types': self.network_types,
            'experiment_types': self.experiment_types,
            'task_sequence': self.task_sequence,
            'seeds': self.seeds,
            'node_selection_strategies': self.node_selection_strategies,
            'episodes_per_task': self.episodes_per_task,
            'evaluation_episodes': self.evaluation_episodes,
            'max_env_steps_per_task': self.max_env_steps_per_task,
            'backward_transfer_tasks': self.backward_transfer_tasks,
            'forward_transfer_tasks': self.forward_transfer_tasks,
            'forgetting_test': self.forgetting_test,
            'parameter_budget': self.parameter_budget,
            'network_params': self.network_params,
            'fully_connected_params': self.fully_connected_params,
            'small_world_params': self.small_world_params,
            'modular_params': self.modular_params,
            'num_io_nodes': self.num_io_nodes,
            'rl_params': self.rl_params,
            'use_capacity_mapping': self.use_capacity_mapping,
            'topologies': self.topologies
        }

def verify_capacity_matching(config):
    """Verify capacity matching for Experiment A using the same logic as smoke test."""
    print("="*80)
    print("EXPERIMENT A: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    
    # Initialize measurement manager
    measurement_manager = CapacityMeasurementManager(config)
    
    # Extract parameters
    sizes = config['network_sizes']
    topologies = config['topologies']
    network_types = config['network_types']
    num_layers_list = config['num_layers']
    seeds = config['seeds']
    experiment_types = config['experiment_types']
    node_selection_strategies = config['node_selection_strategies']

    # Baseline measurement phase
    print("\nBaseline measurement phase...")
    for topology in topologies:
        for size in sizes:
            for network_type in network_types:
                for num_layers in num_layers_list:
                    for seed in seeds:
                        if measurement_manager.get_measurement(topology, size, network_type, num_layers) is None:
                            actual_capacity = measurement_manager.measure_capacity(topology, size, network_type, num_layers, seed)
                            measurement_manager.store_measurement(topology, size, network_type, num_layers, actual_capacity, seed)
    measurement_manager._save_measurements()
    print("Baseline measurement complete.\n")

    # Create calculator AFTER disabling capacity mapping
    calculator = ParameterBudgetCalculator(config)
    
    # Track results
    results_summary = {'passed': 0, 'failed': 0, 'errors': 0, 'details': {}}
    
    for exp_type in experiment_types:
        reference_topology = exp_type[len('match_'):]
        print(f"\n{'='*20} EXPERIMENT TYPE: {exp_type.upper()} {'='*20}")
        print(f"All topologies matched to {reference_topology} capacity")
        
        for size in sizes:
            print(f"\n--- Network Size: {size} ---")
            
            # Get target capacities from baseline
            target_capacities = {}
            for network_type in network_types:
                for num_layers in num_layers_list:
                    target_capacity = measurement_manager.get_target_capacity(
                        reference_topology, size, network_type, num_layers
                    )
                    target_capacities[f"{network_type}_{num_layers}"] = target_capacity
            
            print(f"Target capacities from baseline: {target_capacities}")
            
            # Test each topology
            for topology in topologies:
                print(f"\n  Topology: {topology.upper()}")
                print(f"  {'-' * (len(topology) + 10)}")
                
                for network_type in network_types:
                    for num_layers in num_layers_list:
                        for seed in seeds:
                            for strategy in node_selection_strategies:
                                torch.manual_seed(seed)
                                np.random.seed(seed)
                                
                                config_key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}_{strategy}"
                                target_capacity = target_capacities[f"{network_type}_{num_layers}"]
                                
                                if target_capacity is None:
                                    print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                    print(f"      ❌ NO BASELINE MEASUREMENT AVAILABLE")
                                    results_summary['errors'] += 1
                                    results_summary['details'][config_key] = {
                                        'status': 'error',
                                        'error': 'No baseline measurement available',
                                        'type': 'error'
                                    }
                                    continue
                                
                                try:
                                    # For the reference topology, use the baseline size
                                    if topology == reference_topology:
                                        matching_size = size
                                        print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                        print(f"      Target: {target_capacity:,} parameters (from {reference_topology})")
                                        print(f"      Size: {matching_size} nodes (reference, no adjustment)")
                                    else:
                                        # Use incremental adjustment to find matching size
                                        matching_size = calculator.calculate_matching_size(
                                            topology, target_capacity, network_type, num_layers
                                        )
                                        print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                        print(f"      Target: {target_capacity:,} parameters (from {reference_topology})")
                                        print(f"      Size adjustment: {size} → {matching_size} nodes (incremental adjustment)")
                                    
                                    # Create network using the matching size
                                    network = calculator.create_network(
                                        topology=topology,
                                        size=matching_size,
                                        experiment_type='same_size',
                                        network_type=network_type,
                                        num_layers=num_layers,
                                        seed=seed
                                    )
                                    
                                    metrics = network.get_network_metrics()
                                    actual_capacity = sum(
                                        metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                    )
                                    
                                    # Calculate divergence
                                    divergence = abs(actual_capacity - target_capacity) / target_capacity * 100 if target_capacity > 0 else float('inf')
                                    
                                    print(f"      Actual: {actual_capacity:,} parameters")
                                    print(f"      Divergence: {divergence:.2f}%")
                                    
                                    if divergence <= 5.0:
                                        print(f"      ✅ Within threshold (5.0%)")
                                        results_summary['passed'] += 1
                                        status = 'passed'
                                    else:
                                        print(f"      ⚠️  Exceeds threshold (5.0%)")
                                        results_summary['failed'] += 1
                                        status = 'failed'
                                    
                                    results_summary['details'][config_key] = {
                                        'status': status,
                                        'target_capacity': target_capacity,
                                        'actual_capacity': actual_capacity,
                                        'matching_size': matching_size,
                                        'divergence': divergence
                                    }
                                except Exception as e:
                                    print(f"      ❌ ERROR: {e}")
                                    results_summary['errors'] += 1
                                    results_summary['details'][config_key] = {
                                        'status': 'error',
                                        'error': str(e),
                                        'type': 'error'
                                    }
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("EXPERIMENT A: CAPACITY MATCHING SUMMARY")
    print("="*80)
    
    total_tests = results_summary['passed'] + results_summary['failed'] + results_summary['errors']
    print(f"Total configurations tested: {total_tests}")
    print(f"✅ Passed: {results_summary['passed']} ({results_summary['passed']/total_tests*100:.1f}%)")
    print(f"⚠️  Failed: {results_summary['failed']} ({results_summary['failed']/total_tests*100:.1f}%)")
    print(f"❌ Errors: {results_summary['errors']} ({results_summary['errors']/total_tests*100:.1f}%)")
    
    if results_summary['failed'] == 0 and results_summary['errors'] == 0:
        print("\n🎉 ALL CONFIGURATIONS PASSED! Ready for training.")
    else:
        print(f"\n⚠️  {results_summary['failed'] + results_summary['errors']} CONFIGURATIONS NEED FIXING.")
        sys.exit(1)
    
    return results_summary

def run_experiment_a_training(config):
    """Run Experiment A training."""
    print("\n" + "="*80)
    print("EXPERIMENT A: SINGLE TASK TRAINING")
    print("="*80)
    
    print("Starting single task topology comparison...")
    
    # Create runner and execute
    runner = EnhancedCurriculumRunner(config)
    
    # Print experiment parameters
    print("\nExperiment A Parameters:")
    print(f"Task: {config['task_sequence'][0]}")
    print(f"Network sizes: {config['network_sizes']}")
    print(f"Network type: {config['network_types'][0]}")
    print(f"Topologies: small_world, modular, hybrid, fully_connected")
    print(f"Episodes per task: {config['episodes_per_task']}")
    print(f"Evaluation episodes: {config['evaluation_episodes']}")
    print(f"Max env steps: {config['max_env_steps_per_task']}")
    
    # Execute training
    try:
        print("\nExecuting Experiment A training...")
        training_results = runner.run_curriculum()
        print(f"\n✅ Experiment A completed successfully")
        print(f"📊 Results: {len(training_results)} topology configurations")
        return training_results
    except Exception as e:
        print(f"\n❌ Experiment A failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_experiment_a_results(results):
    """Analyze Experiment A results for topology comparison with enhanced metrics."""
    print("\n" + "="*80)
    print("EXPERIMENT A: ENHANCED RESULTS ANALYSIS")
    print("="*80)
    
    if not results:
        print("❌ No results to analyze")
        return
    
    print(f"Analyzing {len(results)} topology configurations...")
    
    # Group results by topology
    topology_results = {}
    for result in results:
        topology = result['topology']
        if topology not in topology_results:
            topology_results[topology] = []
        topology_results[topology].append(result)
    
    # Analyze each topology with enhanced metrics
    print("\nTopology Performance Summary:")
    print("-" * 60)
    
    for topology, results_list in topology_results.items():
        if not results_list:
            continue
            
        result = results_list[0]  # Single result per topology
        curriculum_results = result['curriculum_results']
        
        # Extract performance metrics
        if 'performance_history' in curriculum_results:
            performance = curriculum_results['performance_history']
            if 'cartpole' in performance:
                cartpole_perf = performance['cartpole']['cartpole']
                mean_reward = cartpole_perf['mean_reward']
                std_reward = cartpole_perf['std_reward']
                solved_rate = cartpole_perf['solved_rate']
                
                print(f"\n{topology.upper()}:")
                print(f"  Mean Reward: {mean_reward:.1f} ± {std_reward:.1f}")
                print(f"  Solved Rate: {solved_rate:.1%}")
                
                # Enhanced learning curve analysis
                if 'learning_curves' in curriculum_results:
                    learning_curves = curriculum_results['learning_curves']
                    if 'cartpole' in learning_curves:
                        curve = learning_curves['cartpole']
                        if curve:
                            initial_reward = curve[0]
                            final_reward = curve[-1]
                            max_reward = max(curve)
                            improvement = final_reward - initial_reward
                            learning_rate = (final_reward - initial_reward) / len(curve) if len(curve) > 1 else 0
                            
                            print(f"  Learning Curve Analysis:")
                            print(f"    Initial: {initial_reward:.1f} → Final: {final_reward:.1f} (Δ{improvement:+.1f})")
                            print(f"    Max Reward: {max_reward:.1f}")
                            print(f"    Learning Rate: {learning_rate:.3f} reward/episode")
                            
                            # Convergence analysis
                            if 'convergence_metrics' in curriculum_results.get('transfer_metrics', {}):
                                conv_metrics = curriculum_results['transfer_metrics']['convergence_metrics']
                                if 'cartpole' in conv_metrics:
                                    conv_episode = conv_metrics['cartpole']['convergence_episode']
                                    conv_reward = conv_metrics['cartpole']['convergence_reward']
                                    stability_std = conv_metrics['cartpole']['stability_std']
                                    print(f"    Convergence: Episode {conv_episode} (Reward: {conv_reward:.1f})")
                                    print(f"    Stability: σ={stability_std:.2f}")
    
    print(f"\n📊 Enhanced analysis complete for {len(topology_results)} topologies")

def main():
    """Run Experiment A: Single Task, Single Topology Comparison."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Experiment A: Single Task Topology Comparison")
    
    # Create experiment configuration
    config = ExperimentAConfig()
    config_dict = config.to_dict()
    
    # Print experiment details
    print("\nExperiment A Configuration:")
    print("="*50)
    print("Single Task Topology Comparison")
    print(f"Task: {config.task_sequence[0]}")
    print(f"Network: {config.network_types[0].upper()}, {config.num_layers[0]} layer")
    print(f"Size: {config.network_sizes[0]} nodes")
    print(f"Topologies: small_world, modular, hybrid, fully_connected")
    print(f"Episodes: {config.episodes_per_task}")
    print(f"Seed: {config.seeds[0]}")
    
    # Run capacity matching verification
    print("\n" + "="*80)
    print("PHASE 1: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    capacity_results = verify_capacity_matching(config_dict)
    
    # Run training
    print("\n" + "="*80)
    print("PHASE 2: SINGLE TASK TRAINING")
    print("="*80)
    training_results = run_experiment_a_training(config_dict)
    
    # Analyze results
    print("\n" + "="*80)
    print("PHASE 3: RESULTS ANALYSIS")
    print("="*80)
    analyze_experiment_a_results(training_results)
    
    logger.info("Experiment A completed successfully")
    
    return training_results

if __name__ == "__main__":
    main() 