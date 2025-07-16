"""
EXPERIMENT B SCALED: Forgetting Baseline on Two Tasks (Larger Networks)
Objective: Observe basic retention vs forgetting when switching tasks with larger networks.

| Task Sequence | ['cartpole', 'mountain_car'] |
| Topologies | modular, fully_connected |
| Network | ffn, 1 layer |
| Size | 50, 100 (scaled up from 25) |
| Use retention test: | retention_interval = 1, retention_episodes = 5 (increased from 2) |

Why useful:
- Tests if forgetting patterns scale with network size
- Longer training reveals transfer learning dynamics
- More comprehensive retention testing
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

# GPU Support: Initialize device manager early
try:
    from src.utils.device_manager import get_device_manager, get_device_info
    DEVICE_MANAGER = get_device_manager()
    DEVICE_INFO = get_device_info()
    GPU_SUPPORT_ENABLED = True
except ImportError as e:
    print(f"Warning: GPU support not available: {e}")
    DEVICE_MANAGER = None
    DEVICE_INFO = {'device': 'cpu', 'is_cuda': False, 'is_gpu_available': False}
    GPU_SUPPORT_ENABLED = False
except Exception as e:
    print(f"Warning: Failed to initialize GPU support: {e}")
    DEVICE_MANAGER = None
    DEVICE_INFO = {'device': 'cpu', 'is_cuda': False, 'is_gpu_available': False}
    GPU_SUPPORT_ENABLED = False

class ExperimentBScaledConfig:
    """Configuration for Experiment B Scaled: Forgetting Baseline with larger networks."""
    
    def __init__(self):
        # Core experiment settings - DRAMATICALLY SCALED UP FOR MEANINGFUL LEARNING
        self.network_sizes = [20,200,2000]  # Increased 10x from [150, 300] for meaningful capacity
        self.num_layers = [1]
        self.network_types = ['ffn']
        self.experiment_types = ['same_size', 'match_small_world']
        self.task_sequence = ['mountain_car', 'acrobot']  # Keep available tasks
        self.seeds = [42]
        self.node_selection_strategies = ['random']
        
        # Training parameters - DRAMATICALLY INCREASED FOR LEARNING
        self.episodes_per_task = 20000  # Increased 5x from 150 for convergence
        self.evaluation_episodes = 500  # Increased for better evaluation
        self.max_env_steps = 5000  # Increased from 500 for more exploration
        self.learning_rate = 0.001
        self.batch_size = 64  # Increased for better training
        
        # Adaptive training parameters
        self.convergence_window = 100  # Episodes to check for convergence
        self.convergence_threshold = 0.02  # Performance stability threshold
        self.min_episodes = 200  # Minimum episodes before early stopping
        self.convergence_patience = 3  # How many times to check before stopping
        
        # Parameter budget - MUCH LARGER FOR MEANINGFUL NETWORKS
        self.parameter_budget = {
            'budget_type': 'edges',
            'target_budget': 1000000,  # Increased 10x for larger networks
            'normalize_by_size': True,
            'threshold': 0.05
        }
        
        # Forgetting test settings - ENABLED FOR EXPERIMENT B
        self.forgetting_test = {
            'enabled': True,
            'retention_interval': 1,
            'retention_episodes': 10,  # Increased for better forgetting measurement
            'forgetting_threshold': 0.8,
            'retention_threshold': 0.9
        }
        
        # Transfer learning (minimal for forgetting baseline)
        self.backward_transfer_tasks = []
        self.forward_transfer_tasks = []
        
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
        
        # Topologies list (subset for Experiment B)
        self.topologies = ['modular', 'fully_connected']  # Reduced for speed
        
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
            'max_env_steps': self.max_env_steps,
            'convergence_window': self.convergence_window,
            'convergence_threshold': self.convergence_threshold,
            'min_episodes': self.min_episodes,
            'convergence_patience': self.convergence_patience,
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
    """Verify capacity matching for Experiment B Scaled using the same logic as smoke test."""
    print("="*80)
    print("EXPERIMENT B SCALED: CAPACITY MATCHING VERIFICATION")
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
        print(f"\n{'='*20} EXPERIMENT TYPE: {exp_type.upper()} {'='*20}")
        
        if exp_type == 'same_size':
            print("All topologies use the same node count (not matched capacities)")
        else:
            reference_topology = exp_type[len('match_'):]
            print(f"All topologies matched to {reference_topology} capacity")
        
        for size in sizes:
            print(f"\n--- Network Size: {size} ---")
            
            if exp_type == 'same_size':
                print("All topologies will use this exact node count")
            else:
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
                                
                                if exp_type == 'same_size':
                                    # For same_size experiments, use the original size directly
                                    matching_size = size
                                    print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                    print(f"      Size: {matching_size} nodes (same_size experiment)")
                                    
                                    # Create network using the original size
                                    network = calculator.create_network(
                                        topology=topology,
                                        size=matching_size,
                                        experiment_type=exp_type,
                                        network_type=network_type,
                                        num_layers=num_layers,
                                        seed=seed
                                    )
                                    
                                    metrics = network.get_network_metrics()
                                    actual_capacity = sum(
                                        metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                    )
                                    
                                    print(f"      Actual: {actual_capacity:,} parameters")
                                    print(f"      ✅ Same size experiment - no capacity matching required")
                                    
                                    results_summary['passed'] += 1
                                    results_summary['details'][config_key] = {
                                        'status': 'passed',
                                        'actual_capacity': actual_capacity,
                                        'matching_size': matching_size,
                                        'type': 'same_size'
                                    }
                                else:
                                    # For match_* experiments, use capacity matching logic
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
                                            # Use incremental adjustment to find matching size directly
                                            matching_size = calculator.calculate_matching_size(
                                                topology, target_capacity, network_type, num_layers
                                            )
                                            print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                            print(f"      Target: {target_capacity:,} parameters (from {reference_topology})")
                                            print(f"      Size adjustment: {size} → {matching_size} nodes (incremental adjustment)")
                                        
                                        # Create network using the matching size with 'same_size' to avoid recursive matching
                                        network = calculator.create_network(
                                            topology=topology,
                                            size=matching_size,
                                            experiment_type='same_size',  # Use same_size to avoid recursive matching
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
    print("EXPERIMENT B SCALED: CAPACITY MATCHING SUMMARY")
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

def run_experiment_b_scaled_training(config):
    """Run Experiment B Scaled training with comprehensive logging."""
    print("\n" + "="*80)
    print("EXPERIMENT B SCALED: FORGETTING BASELINE TRAINING")
    print("="*80)
    
    print("Starting scaled forgetting baseline experiment...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/experiment_b_scaled_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Create runner and execute
    runner = EnhancedCurriculumRunner(config)
    
    # Print experiment parameters
    print("\nExperiment B Scaled Parameters:")
    print(f"Task sequence: {config['task_sequence']}")
    print(f"Network sizes: {config['network_sizes']}")
    print(f"Network type: {config['network_types'][0]}")
    print(f"Topologies: {', '.join(config['topologies'])}")
    print(f"Episodes per task: {config['episodes_per_task']}")
    print(f"Evaluation episodes: {config['evaluation_episodes']}")
    print(f"Max env steps: {config['max_env_steps']}")
    print(f"Forgetting test: {config['forgetting_test']['enabled']}")
    if config['forgetting_test']['enabled']:
        print(f"  Retention interval: {config['forgetting_test']['retention_interval']}")
        print(f"  Retention episodes: {config['forgetting_test']['retention_episodes']}")
    
    # Execute training
    try:
        print("\nExecuting Experiment B Scaled training...")
        training_results = runner.run_curriculum()
        print(f"\n✅ Experiment B Scaled completed successfully")
        print(f"📊 Results: {len(training_results)} topology configurations")
        
        # Save raw results
        with open(results_dir / "training_results.json", "w") as f:
            json.dump(training_results, f, indent=2, default=str)
        
        return training_results, results_dir
    except Exception as e:
        print(f"\n❌ Experiment B Scaled failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_experiment_b_scaled_results(results, results_dir):
    """Analyze Experiment B Scaled results for forgetting patterns with comprehensive metrics."""
    print("\n" + "="*80)
    print("EXPERIMENT B SCALED: COMPREHENSIVE FORGETTING ANALYSIS")
    print("="*80)
    
    print(f"Analyzing {len(results)} topology configurations...")
    
    # Extract forgetting and transfer data
    forgetting_data = []
    transfer_metrics = {}
    learning_curves = {}
    
    # Get the actual task names from the experiment (should be ['acrobot', 'mountain_car'])
    task_names = ['acrobot', 'mountain_car']  # From ExperimentBScaledConfig.task_sequence
    
    for result in results:
        topology = result['topology']
        size = result['network_size']
        curriculum_results = result['curriculum_results']
        
        # Extract final performance for each task - handle enhanced runner format
        if 'performance_history' in curriculum_results:
            # Enhanced runner format
            performance_history = curriculum_results['performance_history']
            transfer_metrics_data = curriculum_results.get('transfer_metrics', {})
            
            # Extract final performance from the last task in sequence
            final_task = task_names[-1]  # 'mountain_car'
            if final_task in performance_history:
                final_performance = performance_history[final_task]
            else:
                final_performance = {}
        elif 'final_performance' in curriculum_results:
            # Alternative enhanced format
            final_performance = curriculum_results['final_performance']
            transfer_metrics_data = curriculum_results.get('transfer_metrics', {})
        else:
            # No performance data available
            final_performance = {}
            transfer_metrics_data = {}
        
        # Extract task-specific performance with correct task names
        acrobot_perf = final_performance.get('acrobot', {})
        mountain_car_perf = final_performance.get('mountain_car', {})
        
        forgetting_data.append({
            'topology': topology,
            'size': size,
            'acrobot_reward': acrobot_perf.get('mean_reward', 0),
            'acrobot_solved': acrobot_perf.get('solved_rate', 0),
            'mountain_car_reward': mountain_car_perf.get('mean_reward', 0),
            'mountain_car_solved': mountain_car_perf.get('solved_rate', 0),
            'backward_transfer': transfer_metrics_data.get('backward_transfer', {}).get('acrobot', 1.0),
            'forward_transfer': transfer_metrics_data.get('forward_transfer', {}).get('mountain_car', 1.0)
        })
        
        # Store transfer metrics
        transfer_metrics[f"{topology}_{size}"] = {
            'backward_transfer': transfer_metrics_data.get('backward_transfer', {}),
            'forward_transfer': transfer_metrics_data.get('forward_transfer', {})
        }
        
        # Extract learning curves if available
        if 'learning_curves' in curriculum_results:
            learning_curves_data = curriculum_results['learning_curves']
            for task_name in task_names:
                if task_name in learning_curves_data:
                    learning_curves[f"{topology}_{size}_{task_name}"] = learning_curves_data[task_name]
        elif 'performance_history' in curriculum_results:
            # Try to extract from performance history
            performance_history = curriculum_results['performance_history']
            for task_name in task_names:
                if task_name in performance_history:
                    task_history = performance_history[task_name]
                    if task_name in task_history:  # Nested structure
                        episode_data = task_history[task_name]
                        if 'learning_curve' in episode_data:
                            learning_curves[f"{topology}_{size}_{task_name}"] = episode_data['learning_curve']
    
    # Create sophisticated forgetting analysis
    print("\n" + "="*80)
    print("SOPHISTICATED FORGETTING ANALYSIS - DETAILED METRICS")
    print("="*80)
    
    # Group by size
    for size in sorted(set(d['size'] for d in forgetting_data)):
        print(f"\nNetwork Size: {size}")
        print("-" * 60)
        
        size_data = [d for d in forgetting_data if d['size'] == size]
        size_data.sort(key=lambda x: x['backward_transfer'], reverse=True)
        
        for data in size_data:
            topology_key = f"{data['topology']}_{size}"
            acrobot_curve = learning_curves.get(f"{topology_key}_acrobot", [])
            mountain_car_curve = learning_curves.get(f"{topology_key}_mountain_car", [])
            
            print(f"\n{data['topology'].upper():15} | "
                  f"Acrobot: {data['acrobot_reward']:6.1f} reward, {data['acrobot_solved']:5.1f}% solved | "
                  f"Mountain Car: {data['mountain_car_reward']:6.1f} reward, {data['mountain_car_solved']:5.1f}% solved")
            
            # Basic transfer metrics
            print(f"{'':15} | Transfer Metrics:")
            print(f"{'':15} |   Backward Transfer: {data['backward_transfer']:6.3f} | "
                  f"Forward Transfer: {data['forward_transfer']:6.3f}")
            
            # Sophisticated forgetting analysis
            if acrobot_curve and mountain_car_curve:
                # Convert numpy arrays to lists if needed
                if hasattr(acrobot_curve, 'tolist'):
                    acrobot_curve = acrobot_curve.tolist()
                if hasattr(mountain_car_curve, 'tolist'):
                    mountain_car_curve = mountain_car_curve.tolist()
                
                # Calculate forgetting metrics
                acrobot_initial = acrobot_curve[0] if acrobot_curve else 0
                acrobot_final = acrobot_curve[-1] if acrobot_curve else 0
                mountain_car_initial = mountain_car_curve[0] if mountain_car_curve else 0
                mountain_car_final = mountain_car_curve[-1] if mountain_car_curve else 0
                
                # Retention decay analysis
                retention_decay_acrobot = (acrobot_final - acrobot_initial) / max(abs(acrobot_initial), 1)
                retention_decay_mountain_car = (mountain_car_final - mountain_car_initial) / max(abs(mountain_car_initial), 1)
                
                # Forgetting rate (negative values indicate forgetting)
                forgetting_rate_acrobot = retention_decay_acrobot if retention_decay_acrobot < 0 else 0
                forgetting_rate_mountain_car = retention_decay_mountain_car if retention_decay_mountain_car < 0 else 0
                
                # Interference analysis (how much learning second task affects first task)
                interference_score = abs(forgetting_rate_acrobot)  # Higher = more interference
                
                # Recovery analysis (ability to recover performance)
                acrobot_variance = np.std(acrobot_curve) if len(acrobot_curve) > 1 else 0
                mountain_car_variance = np.std(mountain_car_curve) if len(mountain_car_curve) > 1 else 0
                recovery_stability = min(acrobot_variance, mountain_car_variance)  # Lower = more stable recovery
                
                # Learning efficiency (how quickly each task was learned)
                acrobot_learning_efficiency = (acrobot_final - acrobot_initial) / len(acrobot_curve) if len(acrobot_curve) > 1 else 0
                mountain_car_learning_efficiency = (mountain_car_final - mountain_car_initial) / len(mountain_car_curve) if len(mountain_car_curve) > 1 else 0
                
                print(f"{'':15} | Sophisticated Forgetting Analysis:")
                print(f"{'':15} |   Retention Decay:")
                print(f"{'':15} |     Acrobot: {retention_decay_acrobot:6.3f} | Mountain Car: {retention_decay_mountain_car:6.3f}")
                print(f"{'':15} |   Forgetting Rate:")
                print(f"{'':15} |     Acrobot: {forgetting_rate_acrobot:6.3f} | Mountain Car: {forgetting_rate_mountain_car:6.3f}")
                print(f"{'':15} |   Interference Score: {interference_score:6.3f}")
                print(f"{'':15} |   Recovery Stability: {recovery_stability:6.3f}")
                print(f"{'':15} |   Learning Efficiency:")
                print(f"{'':15} |     Acrobot: {acrobot_learning_efficiency:6.3f} | Mountain Car: {mountain_car_learning_efficiency:6.3f}")
                
                # Forgetting pattern classification
                if interference_score < 0.1:
                    pattern = "Low Interference"
                elif interference_score < 0.3:
                    pattern = "Moderate Interference"
                else:
                    pattern = "High Interference"
                
                if recovery_stability < 5:
                    stability = "Very Stable"
                elif recovery_stability < 10:
                    stability = "Stable"
                else:
                    stability = "Unstable"
                
                print(f"{'':15} |   Pattern: {pattern} | Stability: {stability}")
                
                # Store sophisticated metrics
                data.update({
                    'retention_decay_acrobot': retention_decay_acrobot,
                    'retention_decay_mountain_car': retention_decay_mountain_car,
                    'forgetting_rate_acrobot': forgetting_rate_acrobot,
                    'forgetting_rate_mountain_car': forgetting_rate_mountain_car,
                    'interference_score': interference_score,
                    'recovery_stability': recovery_stability,
                    'acrobot_learning_efficiency': acrobot_learning_efficiency,
                    'mountain_car_learning_efficiency': mountain_car_learning_efficiency,
                    'forgetting_pattern': pattern,
                    'recovery_stability_level': stability
                })
            else:
                print(f"{'':15} | No learning curve data available for sophisticated analysis")
                print()
    
    # Create plots
    create_forgetting_plots(forgetting_data, learning_curves, results_dir)
    
    # Save analysis results
    analysis_results = {
        'forgetting_data': forgetting_data,
        'transfer_metrics': transfer_metrics,
        'learning_curves': learning_curves,
        'summary': {
            'total_configurations': len(results),
            'network_sizes': sorted(set(d['size'] for d in forgetting_data)),
            'topologies': sorted(set(d['topology'] for d in forgetting_data))
        }
    }
    
    with open(results_dir / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n📊 Forgetting analysis complete for {len(results)} topologies")
    print(f"📁 Results saved to: {results_dir}")
    
    return analysis_results

def get_ax(axes, row, col):
    if isinstance(axes, list):
        return axes[col]
    if isinstance(axes, np.ndarray):
        if axes.ndim == 1:
            return axes[col]
        return axes[row, col]
    return axes  # single Axes object

def create_forgetting_plots(forgetting_data, learning_curves, results_dir):
    """Create comprehensive forgetting and transfer plots."""
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Transfer metrics comparison
    sizes = sorted(set(d['size'] for d in forgetting_data))
    num_sizes = len(sizes)
    
    # Calculate grid dimensions
    cols = min(2, num_sizes)
    rows = (num_sizes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
    fig.suptitle('Experiment B Scaled: Transfer Learning Analysis', fontsize=16)
    
    # Handle single subplot case
    if num_sizes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, size in enumerate(sizes):
        size_data = [d for d in forgetting_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        backward_transfer = [d['backward_transfer'] for d in size_data]
        forward_transfer = [d['forward_transfer'] for d in size_data]
        
        row = i // cols
        col = i % cols
        ax = get_ax(axes, row, col)
        
        x = np.arange(len(topologies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, backward_transfer, width, label='Backward Transfer', alpha=0.7)
        bars2 = ax.bar(x + width/2, forward_transfer, width, label='Forward Transfer', alpha=0.7)
        
        ax.set_title(f'Network Size: {size}')
        ax.set_ylabel('Transfer Ratio')
        ax.set_xlabel('Topology')
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
    
    # Hide empty subplots
    for i in range(num_sizes, rows * cols):
        row = i // cols
        col = i % cols
        ax = get_ax(axes, row, col)
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(results_dir / "transfer_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Task performance comparison
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
    fig.suptitle('Experiment B Scaled: Task Performance Comparison', fontsize=16)
    
    # Handle single subplot case
    if num_sizes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, size in enumerate(sizes):
        size_data = [d for d in forgetting_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        acrobot_rewards = [d['acrobot_reward'] for d in size_data]
        mountain_car_rewards = [d['mountain_car_reward'] for d in size_data]
        
        row = i // cols
        col = i % cols
        ax = get_ax(axes, row, col)
        
        x = np.arange(len(topologies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, acrobot_rewards, width, label='Acrobot', alpha=0.7)
        bars2 = ax.bar(x + width/2, mountain_car_rewards, width, label='Mountain Car', alpha=0.7)
        
        ax.set_title(f'Network Size: {size}')
        ax.set_ylabel('Mean Reward')
        ax.set_xlabel('Topology')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Hide empty subplots
    for i in range(num_sizes, rows * cols):
        row = i // cols
        col = i % cols
        ax = get_ax(axes, row, col)
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(results_dir / "task_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Learning curves (if available)
    if learning_curves:
        curve_keys = list(learning_curves.keys())
        num_curves = min(4, len(curve_keys))  # Limit to 4 plots
        
        if num_curves > 0:
            cols_curves = min(2, num_curves)
            rows_curves = (num_curves + cols_curves - 1) // cols_curves
            
            fig, axes = plt.subplots(rows_curves, cols_curves, figsize=(15, 6*rows_curves))
            fig.suptitle('Experiment B Scaled: Learning Curves', fontsize=16)
            
            # Handle single subplot case
            if num_curves == 1:
                axes = [axes]
            elif rows_curves == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_curves):
                key = curve_keys[i]
                curve = learning_curves[key]
                
                row = i // cols_curves
                col = i % cols_curves
                ax = get_ax(axes, row, col)
                
                # Convert numpy array to list if needed
                if hasattr(curve, 'tolist'):
                    curve = curve.tolist()
                
                episodes = list(range(1, len(curve) + 1))
                ax.plot(episodes, curve, linewidth=2, alpha=0.8)
                ax.set_title(f'Learning Curve: {key}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(num_curves, rows_curves * cols_curves):
                row = i // cols_curves
                col = i % cols_curves
                ax = get_ax(axes, row, col)
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(results_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Solved rate comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i, size in enumerate(sizes):
        size_data = [d for d in forgetting_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        acrobot_solved = [d['acrobot_solved'] for d in size_data]
        mountain_car_solved = [d['mountain_car_solved'] for d in size_data]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        # Handle single size case
        if len(sizes) == 1:
            offset = 0
        else:
            offset = width * (i == 1)  # Offset for second size
        
        ax.bar(x - width/2 + offset, acrobot_solved, width, 
               label=f'Acrobot (Size {size})', alpha=0.7)
        ax.bar(x + width/2 + offset, mountain_car_solved, width, 
               label=f'Mountain Car (Size {size})', alpha=0.7)
    
    ax.set_xlabel('Topology')
    ax.set_ylabel('Solved Rate (%)')
    ax.set_title('Solved Rate Comparison by Task, Topology and Size')
    ax.set_xticks(x + width/2 if len(sizes) > 1 else x)
    ax.set_xticklabels([d['topology'] for d in forgetting_data if d['size'] == sizes[0]])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "solved_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {results_dir}")

def main():
    """Run Experiment B Scaled: Forgetting Baseline on Two Tasks with larger networks."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Experiment B Scaled: Forgetting Baseline")
    
    # GPU Support: Log device information
    if GPU_SUPPORT_ENABLED:
        print(f"\n🔧 GPU Support: {DEVICE_INFO['device']}")
        if DEVICE_INFO['is_cuda']:
            print(f"   GPU: {DEVICE_INFO.get('cuda_device_name', 'Unknown')}")
            print(f"   Memory: {DEVICE_INFO.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB allocated")
        else:
            print(f"   Using CPU (GPU not available or disabled)")
    else:
        print(f"\n🔧 GPU Support: Disabled (fallback to CPU)")
    
    # Create configuration
    config_obj = ExperimentBScaledConfig()
    config = config_obj.to_dict()
    
    # Print configuration
    print("\nExperiment B Scaled Configuration:")
    print("="*50)
    print("Forgetting Baseline on Two Tasks (SCALED)")
    print(f"Task sequence: {' → '.join(config['task_sequence'])}")
    print(f"Network: {config['network_types'][0].upper()}, {config['num_layers'][0]} layer")
    print(f"Size: {config['network_sizes']} nodes")
    print(f"Topologies: {', '.join(config['topologies'])}")
    print(f"Episodes: {config['episodes_per_task']}")
    print(f"Seed: {config['seeds'][0]}")
    print(f"Forgetting test: {config['forgetting_test']['enabled']}")
    
    # Phase 1: Capacity matching verification
    print("\n" + "="*80)
    print("PHASE 1: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    capacity_results = verify_capacity_matching(config)
    
    # Check if capacity matching passed
    if capacity_results['failed'] > 0 or capacity_results['errors'] > 0:
        print(f"\n⚠️  {capacity_results['failed'] + capacity_results['errors']} CONFIGURATIONS NEED FIXING.")
        print("   Skipping training due to capacity matching failures.")
        sys.exit(1)
    
    # Phase 2: Training execution
    print("\n" + "="*80)
    print("PHASE 2: FORGETTING BASELINE TRAINING")
    print("="*80)
    try:
        training_results, results_dir = run_experiment_b_scaled_training(config)
        
        # Phase 3: Results analysis
        print("\n" + "="*80)
        print("PHASE 3: FORGETTING ANALYSIS")
        print("="*80)
        analysis_results = analyze_experiment_b_scaled_results(training_results, results_dir)
        
        print(f"\n🎉 EXPERIMENT B SCALED COMPLETE!")
        print(f"📁 All results saved to: {results_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Experiment B Scaled completed successfully")

if __name__ == "__main__":
    main() 