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

class ExperimentBScaledConfig:
    """Configuration for Experiment B Scaled: Forgetting Baseline with larger networks."""
    
    def __init__(self):
        # Core experiment settings - OPTIMIZED FOR SPEED & LEARNING
        self.network_sizes = [150, 300]  # Increased for better learning capacity
        self.num_layers = [1]
        self.network_types = ['ffn']
        self.experiment_types = ['match_small_world']
        self.task_sequence = ['acrobot', 'mountain_car']  # Changed to available tasks
        self.seeds = [42]
        self.node_selection_strategies = ['random']
        
        # Training parameters - OPTIMIZED FOR SPEED
        self.episodes_per_task = 150  # Reduced from 500 for speed
        self.evaluation_episodes = 10  # Reduced for speed
        self.max_env_steps_per_task = 500  # Reduced from 10000 for speed
        
        # Transfer learning (minimal for forgetting baseline)
        self.backward_transfer_tasks = []
        self.forward_transfer_tasks = []
        
        # Forgetting test - ENABLED for Experiment B
        self.forgetting_test = {
            'enabled': True,
            'retention_interval': 1,
            'retention_episodes': 5,  # Reduced from 10 for speed
            'forgetting_threshold': 0.8,
            'retention_threshold': 0.9
        }
        
        # Parameter budget - INCREASED FOR MEANINGFUL LEARNING
        self.parameter_budget = {
            'budget_type': 'edges',
            'target_budget': 100000,  # Increased from 10000
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
    print(f"Max env steps: {config['max_env_steps_per_task']}")
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
    
    for result in results:
        topology = result['topology']
        size = result['network_size']
        curriculum_results = result['curriculum_results']
        
        # Extract final performance for each task - handle both enhanced and regular formats
        if 'final_performance' in curriculum_results:
            # Enhanced format
            final_performance = curriculum_results['final_performance']
            transfer_metrics_data = curriculum_results.get('transfer_metrics', {})
        elif 'performance_history' in curriculum_results:
            # Regular format - extract from performance history
            performance_history = curriculum_results['performance_history']
            final_performance = {}
            transfer_metrics_data = {}
            
            # Extract final performance from the last task
            if 'mountain_car' in performance_history:
                final_performance = performance_history['mountain_car']
            elif 'cartpole' in performance_history:
                final_performance = performance_history['cartpole']
        else:
            # No performance data available
            final_performance = {}
            transfer_metrics_data = {}
        
        # Extract task-specific performance with fallbacks
        cartpole_perf = final_performance.get('cartpole', {})
        mountain_car_perf = final_performance.get('mountain_car', {})
        
        forgetting_data.append({
            'topology': topology,
            'size': size,
            'cartpole_reward': cartpole_perf.get('mean_reward', 0),
            'cartpole_solved': cartpole_perf.get('solved_rate', 0),
            'mountain_car_reward': mountain_car_perf.get('mean_reward', 0),
            'mountain_car_solved': mountain_car_perf.get('solved_rate', 0),
            'backward_transfer': transfer_metrics_data.get('backward_transfer', {}).get('cartpole', 1.0),
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
            for task_name in ['cartpole', 'mountain_car']:
                if task_name in learning_curves_data:
                    learning_curves[f"{topology}_{size}_{task_name}"] = learning_curves_data[task_name]
        elif 'performance_history' in curriculum_results:
            # Try to extract from performance history
            performance_history = curriculum_results['performance_history']
            for task_name in ['cartpole', 'mountain_car']:
                if task_name in performance_history:
                    task_history = performance_history[task_name]
                    if task_name in task_history:  # Nested structure
                        episode_data = task_history[task_name]
                        if 'learning_curve' in episode_data:
                            learning_curves[f"{topology}_{size}_{task_name}"] = episode_data['learning_curve']
    
    # Create forgetting analysis
    print("\nTopology Forgetting Patterns:")
    print("-" * 80)
    
    # Group by size
    for size in sorted(set(d['size'] for d in forgetting_data)):
        print(f"\nNetwork Size: {size}")
        print("-" * 40)
        
        size_data = [d for d in forgetting_data if d['size'] == size]
        size_data.sort(key=lambda x: x['backward_transfer'], reverse=True)
        
        for data in size_data:
            print(f"{data['topology'].upper():15} | "
                  f"Cartpole: {data['cartpole_reward']:6.1f} reward, {data['cartpole_solved']:5.1f}% solved | "
                  f"Mountain Car: {data['mountain_car_reward']:6.1f} reward, {data['mountain_car_solved']:5.1f}% solved")
            print(f"{'':15} | "
                  f"Backward Transfer: {data['backward_transfer']:6.3f} | "
                  f"Forward Transfer: {data['forward_transfer']:6.3f}")
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
        ax = axes[row, col] if rows > 1 else axes[col]
        
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
        ax = axes[row, col] if rows > 1 else axes[col]
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
        cartpole_rewards = [d['cartpole_reward'] for d in size_data]
        mountain_car_rewards = [d['mountain_car_reward'] for d in size_data]
        
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, cartpole_rewards, width, label='Cartpole', alpha=0.7)
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
        ax = axes[row, col] if rows > 1 else axes[col]
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
                ax = axes[row, col] if rows_curves > 1 else axes[col]
                
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
                ax = axes[row, col] if rows_curves > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(results_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 4. Solved rate comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for size in sizes:
        size_data = [d for d in forgetting_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        cartpole_solved = [d['cartpole_solved'] for d in size_data]
        mountain_car_solved = [d['mountain_car_solved'] for d in size_data]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        ax.bar(x - width/2 + width * (size == sizes[1]), cartpole_solved, width, 
               label=f'Cartpole (Size {size})', alpha=0.7)
        ax.bar(x + width/2 + width * (size == sizes[1]), mountain_car_solved, width, 
               label=f'Mountain Car (Size {size})', alpha=0.7)
    
    ax.set_xlabel('Topology')
    ax.set_ylabel('Solved Rate (%)')
    ax.set_title('Solved Rate Comparison by Task, Topology and Size')
    ax.set_xticks(x + width/2)
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