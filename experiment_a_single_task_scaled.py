"""
EXPERIMENT A SCALED: Single Task, Single Topology Comparison (Larger Networks)
Objective: Evaluate if topology alone impacts learning for a single task with larger networks.

Factor	Setting
Task	CartPole-v1
Network type	ffn
Layers	1
Size	50, 100 (scaled up from 25)
Topologies	small_world, modular, fully_connected, hybrid
Seed	42
Node selection	random
Episodes	500 (increased from 200)
Env steps per task	10,000 (increased from 5,000)
Evaluation episodes	20 (increased from 10)

Why useful:
- Tests if topology effects scale with network size
- Longer training reveals convergence patterns
- More comprehensive evaluation
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

# Experiment A Scaled Configuration
EXPERIMENT_CONFIG = {
    'name': 'Experiment A Scaled: Single Task Topology Comparison',
    'description': 'Single Task Topology Comparison (SCALED) - Optimized for Speed & Learning',
    
    # Task configuration
    'task': 'acrobot',  # Simpler than CartPole, faster to train
    'episodes': 150,    # Reduced for speed while maintaining learning
    'evaluation_episodes': 10,  # Reduced for speed
    'max_env_steps': 500,  # Reduced from 10000 for speed
    
    # Network configuration - larger sizes for meaningful learning
    'network_sizes': [150, 300],  # Increased for better learning capacity
    'network_type': 'ffn',
    'num_layers': 1,
    
    # Topologies to compare
    'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
    
    # Training parameters
    'seed': 42,
    'learning_rate': 0.001,
    'batch_size': 32,
    
    # Parameter budget - much larger for meaningful learning
    'parameter_budget': {
        'type': 'edges',
        'target_budget': 100000,  # Increased for meaningful learning
        'normalize_by_size': True,
        'threshold': 0.05
    },
    
    # Experiment type
    'experiment_type': 'match_small_world',
    
    # Enhanced logging
    'enhanced_logging': True,
    'log_interval': 25,  # Log every 25 episodes instead of 50
    'save_learning_curves': True,
    'save_network_metrics': True
}

class ExperimentAScaledConfig:
    """Configuration for Experiment A Scaled: Single Task, Single Topology Comparison with larger networks."""
    
    def __init__(self):
        # Core experiment settings - OPTIMIZED FOR SPEED & LEARNING
        self.network_sizes = [150, 300]  # Increased for better learning capacity
        self.num_layers = [1]
        self.network_types = ['ffn']
        self.experiment_types = ['match_small_world']
        self.task_sequence = ['acrobot']  # Changed to acrobot for faster training
        self.seeds = [42]
        self.node_selection_strategies = ['random']
        
        # Training parameters - OPTIMIZED FOR SPEED
        self.episodes_per_task = 150  # Reduced from 500 for speed
        self.evaluation_episodes = 10  # Reduced for speed
        self.max_env_steps_per_task = 500  # Reduced from 10000 for speed
        
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
    """Verify capacity matching for Experiment A Scaled using the same logic as smoke test."""
    print("="*80)
    print("EXPERIMENT A SCALED: CAPACITY MATCHING VERIFICATION")
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
    print("EXPERIMENT A SCALED: CAPACITY MATCHING SUMMARY")
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

def run_experiment_a_scaled_training(config):
    """Run Experiment A Scaled training with comprehensive logging."""
    print("\n" + "="*80)
    print("EXPERIMENT A SCALED: SINGLE TASK TRAINING")
    print("="*80)
    
    print("Starting scaled single task topology comparison...")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/experiment_a_scaled_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Create runner and execute
    runner = EnhancedCurriculumRunner(config)
    
    # Print experiment parameters
    print("\nExperiment A Scaled Parameters:")
    print(f"Task: {config['task_sequence'][0]}")
    print(f"Network sizes: {config['network_sizes']}")
    print(f"Network type: {config['network_types'][0]}")
    print(f"Topologies: {', '.join(config['topologies'])}")
    print(f"Episodes per task: {config['episodes_per_task']}")
    print(f"Evaluation episodes: {config['evaluation_episodes']}")
    print(f"Max env steps: {config['max_env_steps_per_task']}")
    
    # Execute training
    try:
        print("\nExecuting Experiment A Scaled training...")
        training_results = runner.run_curriculum()
        print(f"\n✅ Experiment A Scaled completed successfully")
        print(f"📊 Results: {len(training_results)} topology configurations")
        
        # Save raw results
        with open(results_dir / "training_results.json", "w") as f:
            json.dump(training_results, f, indent=2, default=str)
        
        return training_results, results_dir
    except Exception as e:
        print(f"\n❌ Experiment A Scaled failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_experiment_a_scaled_results(results, results_dir):
    """Analyze Experiment A Scaled results with comprehensive metrics and plots."""
    print("\n" + "="*80)
    print("EXPERIMENT A SCALED: COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    print(f"Analyzing {len(results)} topology configurations...")
    
    # Extract performance data
    performance_data = []
    learning_curves = {}
    
    for result in results:
        topology = result['topology']
        size = result['network_size']
        curriculum_results = result['curriculum_results']
        
        # Extract final performance - handle both enhanced and regular formats
        if 'final_performance' in curriculum_results:
            # Enhanced format
            final_performance = curriculum_results['final_performance']
            task_name = list(final_performance.keys())[0]  # Should be 'cartpole'
            task_performance = final_performance[task_name]
        elif 'performance_history' in curriculum_results:
            # Regular format - extract from performance history
            performance_history = curriculum_results['performance_history']
            if 'cartpole' in performance_history:
                task_performance = performance_history['cartpole']['cartpole']
            else:
                # Fallback - create basic performance data
                task_performance = {
                    'mean_reward': 0.0,
                    'std_reward': 0.0,
                    'solved_rate': 0.0,
                    'mean_length': 0.0
                }
        else:
            # No performance data available
            task_performance = {
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'solved_rate': 0.0,
                'mean_length': 0.0
            }
        
        performance_data.append({
            'topology': topology,
            'size': size,
            'mean_reward': task_performance.get('mean_reward', 0.0),
            'std_reward': task_performance.get('std_reward', 0.0),
            'solved_rate': task_performance.get('solved_rate', 0.0),
            'mean_length': task_performance.get('mean_length', 0.0)
        })
        
        # Extract learning curves if available
        if 'learning_curves' in curriculum_results:
            learning_curves_data = curriculum_results['learning_curves']
            if 'cartpole' in learning_curves_data:
                learning_curves[f"{topology}_{size}"] = learning_curves_data['cartpole']
        elif 'performance_history' in curriculum_results:
            # Try to extract from performance history
            performance_history = curriculum_results['performance_history']
            if 'cartpole' in performance_history:
                task_history = performance_history['cartpole']
                if 'cartpole' in task_history:  # Nested structure
                    episode_data = task_history['cartpole']
                    if 'learning_curve' in episode_data:
                        learning_curves[f"{topology}_{size}"] = episode_data['learning_curve']
    
    # Create performance summary
    print("\nTopology Performance Summary:")
    print("-" * 60)
    
    # Group by size
    for size in sorted(set(d['size'] for d in performance_data)):
        print(f"\nNetwork Size: {size}")
        print("-" * 30)
        
        size_data = [d for d in performance_data if d['size'] == size]
        size_data.sort(key=lambda x: x['mean_reward'], reverse=True)
        
        for data in size_data:
            print(f"{data['topology'].upper():15} | "
                  f"Reward: {data['mean_reward']:6.1f} ± {data['std_reward']:4.1f} | "
                  f"Solved: {data['solved_rate']:5.1f}% | "
                  f"Length: {data['mean_length']:5.1f}")
    
    # Create plots
    create_performance_plots(performance_data, learning_curves, results_dir)
    
    # Save analysis results
    analysis_results = {
        'performance_data': performance_data,
        'learning_curves': learning_curves,
        'summary': {
            'total_configurations': len(results),
            'network_sizes': sorted(set(d['size'] for d in performance_data)),
            'topologies': sorted(set(d['topology'] for d in performance_data))
        }
    }
    
    with open(results_dir / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n📊 Analysis complete for {len(results)} topologies")
    print(f"📁 Results saved to: {results_dir}")
    
    return analysis_results

def create_performance_plots(performance_data, learning_curves, results_dir):
    """Create comprehensive performance plots."""
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Performance comparison by topology and size
    sizes = sorted(set(d['size'] for d in performance_data))
    num_sizes = len(sizes)
    
    # Calculate grid dimensions
    cols = min(2, num_sizes)
    rows = (num_sizes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
    fig.suptitle('Experiment A Scaled: Topology Performance Comparison', fontsize=16)
    
    # Handle single subplot case
    if num_sizes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, size in enumerate(sizes):
        size_data = [d for d in performance_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        rewards = [d['mean_reward'] for d in size_data]
        stds = [d['std_reward'] for d in size_data]
        
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        bars = ax.bar(topologies, rewards, yerr=stds, capsize=5, alpha=0.7)
        ax.set_title(f'Network Size: {size}')
        ax.set_ylabel('Mean Reward')
        ax.set_xlabel('Topology')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{reward:.1f}', ha='center', va='bottom')
    
    # Hide empty subplots
    for i in range(num_sizes, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(results_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Learning curves (if available)
    if learning_curves:
        curve_keys = list(learning_curves.keys())
        num_curves = min(4, len(curve_keys))  # Limit to 4 plots
        
        if num_curves > 0:
            cols = min(2, num_curves)
            rows = (num_curves + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6*rows))
            fig.suptitle('Experiment A Scaled: Learning Curves', fontsize=16)
            
            # Handle single subplot case
            if num_curves == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_curves):
                key = curve_keys[i]
                curve = learning_curves[key]
                
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                episodes = list(range(1, len(curve) + 1))
                ax.plot(episodes, curve, linewidth=2, alpha=0.8)
                ax.set_title(f'Learning Curve: {key}')
                ax.set_xlabel('Episode')
                ax.set_ylabel('Reward')
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(num_curves, rows * cols):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            plt.savefig(results_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. Solved rate comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for size in sizes:
        size_data = [d for d in performance_data if d['size'] == size]
        topologies = [d['topology'] for d in size_data]
        solved_rates = [d['solved_rate'] for d in size_data]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        ax.bar(x + width * (size == sizes[1]), solved_rates, width, 
               label=f'Size {size}', alpha=0.7)
    
    ax.set_xlabel('Topology')
    ax.set_ylabel('Solved Rate (%)')
    ax.set_title('Solved Rate Comparison by Topology and Size')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([d['topology'] for d in performance_data if d['size'] == sizes[0]])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "solved_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to: {results_dir}")

def main():
    """Run Experiment A Scaled: Single Task Topology Comparison with larger networks."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Experiment A Scaled: Single Task Topology Comparison")
    
    # Create configuration
    config_obj = ExperimentAScaledConfig()
    config = config_obj.to_dict()
    
    # Print configuration
    print("\nExperiment A Scaled Configuration:")
    print("="*50)
    print("Single Task Topology Comparison (SCALED)")
    print(f"Task: {config['task_sequence'][0]}")
    print(f"Network: {config['network_types'][0].upper()}, {config['num_layers'][0]} layer")
    print(f"Size: {config['network_sizes']} nodes")
    print(f"Topologies: {', '.join(config['topologies'])}")
    print(f"Episodes: {config['episodes_per_task']}")
    print(f"Seed: {config['seeds'][0]}")
    
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
    print("PHASE 2: SINGLE TASK TRAINING")
    print("="*80)
    try:
        training_results, results_dir = run_experiment_a_scaled_training(config)
        
        # Phase 3: Results analysis
        print("\n" + "="*80)
        print("PHASE 3: RESULTS ANALYSIS")
        print("="*80)
        analysis_results = analyze_experiment_a_scaled_results(training_results, results_dir)
        
        print(f"\n🎉 EXPERIMENT A SCALED COMPLETE!")
        print(f"📁 All results saved to: {results_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("Experiment A Scaled completed successfully")

if __name__ == "__main__":
    main() 