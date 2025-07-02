"""
EXPERIMENT B: Forgetting Baseline on Two Tasks
Objective: Observe basic retention vs forgetting when switching tasks.

| Task Sequence | ['cartpole', 'mountain_car'] |
| Topologies | modular, fully_connected |
| Network | ffn, 1 layer |
| Use retention test: | retention_interval = 1, retention_episodes = 2 |

Why useful:
- Highlights how well each topology retains previous task knowledge.
- May indicate early trends in transfer/interference tradeoffs.
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

class ExperimentBConfig:
    """Configuration for Experiment B: Forgetting Baseline on Two Tasks."""
    
    def __init__(self):
        # Core experiment settings
        self.network_sizes = [25]
        self.num_layers = [1]
        self.network_types = ['ffn']
        self.experiment_types = ['match_small_world']
        self.task_sequence = ['cartpole', 'mountain_car']
        self.seeds = [42]
        self.node_selection_strategies = ['random']
        
        # Training parameters
        self.episodes_per_task = 200
        self.evaluation_episodes = 10
        self.max_env_steps_per_task = 5000
        
        # Transfer learning (minimal for two tasks)
        self.backward_transfer_tasks = ['cartpole']
        self.forward_transfer_tasks = ['mountain_car']
        
        # Forgetting test (enabled for retention analysis)
        self.forgetting_test = {
            'enabled': True,
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
        
        # Topologies list (only modular and fully_connected for Experiment B)
        self.topologies = ['modular', 'fully_connected']
        
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
    """Verify capacity matching for Experiment B using the same logic as smoke test."""
    print("="*80)
    print("EXPERIMENT B: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    
    # Initialize measurement manager
    measurement_manager = CapacityMeasurementManager(config)
    
    # Extract parameters (only test modular and fully_connected)
    sizes = config['network_sizes']
    topologies = ['modular', 'fully_connected']  # Only these two for Experiment B
    network_types = config['network_types']
    num_layers_list = config['num_layers']
    seeds = config['seeds']
    experiment_types = config['experiment_types']
    node_selection_strategies = config['node_selection_strategies']

    # Baseline measurement phase - need to measure reference topology too
    print("\nBaseline measurement phase...")
    baseline_topologies = topologies + ['small_world']  # Add reference topology
    for topology in baseline_topologies:
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
    print("EXPERIMENT B: CAPACITY MATCHING SUMMARY")
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

def run_experiment_b_training(config):
    """Run Experiment B training."""
    print("\n" + "="*80)
    print("EXPERIMENT B: FORGETTING BASELINE TRAINING")
    print("="*80)
    
    print("Starting forgetting baseline experiment...")
    
    # Create runner and execute
    runner = EnhancedCurriculumRunner(config)
    
    # Print experiment parameters
    print("\nExperiment B Parameters:")
    print(f"Task sequence: {config['task_sequence']}")
    print(f"Network sizes: {config['network_sizes']}")
    print(f"Network type: {config['network_types'][0]}")
    print(f"Topologies: modular, fully_connected")
    print(f"Episodes per task: {config['episodes_per_task']}")
    print(f"Evaluation episodes: {config['evaluation_episodes']}")
    print(f"Max env steps: {config['max_env_steps_per_task']}")
    print(f"Forgetting test: {config['forgetting_test']['enabled']}")
    if config['forgetting_test']['enabled']:
        print(f"  Retention interval: {config['forgetting_test']['retention_interval']}")
        print(f"  Retention episodes: {config['forgetting_test']['retention_episodes']}")
    
    # Execute training
    try:
        print("\nExecuting Experiment B training...")
        training_results = runner.run_curriculum()
        print(f"\n✅ Experiment B completed successfully")
        print(f"📊 Results: {len(training_results)} topology configurations")
        return training_results
    except Exception as e:
        print(f"\n❌ Experiment B failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_experiment_b_results(results):
    """Analyze Experiment B results for forgetting patterns."""
    print("\n" + "="*80)
    print("EXPERIMENT B: FORGETTING ANALYSIS")
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
    
    # Analyze each topology
    print("\nTopology Forgetting Patterns:")
    print("-" * 60)
    
    for topology, results_list in topology_results.items():
        if not results_list:
            continue
            
        result = results_list[0]  # Single result per topology
        curriculum_results = result['curriculum_results']
        
        print(f"\n{topology.upper()}:")
        
        # Extract performance metrics
        if 'performance_history' in curriculum_results:
            performance = curriculum_results['performance_history']
            
            # Cartpole performance
            if 'cartpole' in performance:
                cartpole_perf = performance['cartpole']['cartpole']
                cartpole_reward = cartpole_perf['mean_reward']
                cartpole_solved = cartpole_perf['solved_rate']
                print(f"  Cartpole: {cartpole_reward:.1f} reward, {cartpole_solved:.1%} solved")
            
            # Mountain Car performance
            if 'mountain_car' in performance:
                mountain_car_perf = performance['mountain_car']['mountain_car']
                mountain_car_reward = mountain_car_perf['mean_reward']
                mountain_car_solved = mountain_car_perf['solved_rate']
                print(f"  Mountain Car: {mountain_car_reward:.1f} reward, {mountain_car_solved:.1%} solved")
        
        # Extract transfer metrics
        if 'transfer_metrics' in curriculum_results:
            transfer = curriculum_results['transfer_metrics']
            
            # Backward transfer (retention of cartpole after learning mountain_car)
            if 'backward_transfer' in transfer:
                backward_transfer = transfer['backward_transfer']
                if 'cartpole' in backward_transfer:
                    bt_score = backward_transfer['cartpole']
                    print(f"  Backward Transfer (Cartpole retention): {bt_score:.3f}")
                    if bt_score > 1.0:
                        print(f"    ✅ Positive transfer (better than baseline)")
                    elif bt_score < 0.8:
                        print(f"    ⚠️  Forgetting detected (below threshold)")
                    else:
                        print(f"    ➖ Neutral retention")
            
            # Forward transfer (mountain_car learning with cartpole knowledge)
            if 'forward_transfer' in transfer:
                forward_transfer = transfer['forward_transfer']
                if 'mountain_car' in forward_transfer:
                    ft_score = forward_transfer['mountain_car']
                    print(f"  Forward Transfer (Mountain Car benefit): {ft_score:.3f}")
                    if ft_score > 1.0:
                        print(f"    ✅ Positive transfer")
                    elif ft_score < 0.8:
                        print(f"    ⚠️  Negative transfer")
                    else:
                        print(f"    ➖ Neutral transfer")
        
        # Extract forgetting test results if available
        if 'forgetting_test_results' in curriculum_results:
            forgetting_results = curriculum_results['forgetting_test_results']
            print(f"  Forgetting Test Results:")
            for task, task_results in forgetting_results.items():
                if 'retention_scores' in task_results:
                    retention_scores = task_results['retention_scores']
                    avg_retention = sum(retention_scores) / len(retention_scores)
                    print(f"    {task}: {avg_retention:.3f} avg retention")
    
    print(f"\n📊 Forgetting analysis complete for {len(topology_results)} topologies")

def main():
    """Run Experiment B: Forgetting Baseline on Two Tasks."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Experiment B: Forgetting Baseline")
    
    # Create experiment configuration
    config = ExperimentBConfig()
    config_dict = config.to_dict()
    
    # Print experiment details
    print("\nExperiment B Configuration:")
    print("="*50)
    print("Forgetting Baseline on Two Tasks")
    print(f"Task sequence: {' → '.join(config.task_sequence)}")
    print(f"Network: {config.network_types[0].upper()}, {config.num_layers[0]} layer")
    print(f"Size: {config.network_sizes[0]} nodes")
    print(f"Topologies: modular, fully_connected")
    print(f"Episodes: {config.episodes_per_task}")
    print(f"Seed: {config.seeds[0]}")
    print(f"Forgetting test: {config.forgetting_test['enabled']}")
    
    # Run capacity matching verification
    print("\n" + "="*80)
    print("PHASE 1: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    capacity_results = verify_capacity_matching(config_dict)
    
    # Run training
    print("\n" + "="*80)
    print("PHASE 2: FORGETTING BASELINE TRAINING")
    print("="*80)
    training_results = run_experiment_b_training(config_dict)
    
    # Analyze results
    print("\n" + "="*80)
    print("PHASE 3: FORGETTING ANALYSIS")
    print("="*80)
    analyze_experiment_b_results(training_results)
    
    logger.info("Experiment B completed successfully")
    
    return training_results

if __name__ == "__main__":
    main() 