#!/usr/bin/env python3
"""
Launch Weights & Biases Sweep for Topology Network Hyperparameter Optimization

This script launches wandb sweeps to optimize hyperparameters for the topology training scripts.
"""

import wandb
from wandb_sweep_config import (
    create_sweep_config, create_baseline_sweep_config, create_baseline_focused_sweep_config,
    create_focused_sweep_config, create_task_specific_sweep_config,
    create_double_task_sweep_config, create_triple_task_sweep_config,
    create_focused_double_task_sweep_config, create_focused_triple_task_sweep_config,
    create_sweep_agent_config, create_baseline_sweep_agent_config,
    # Individual topology optimization functions
    create_small_world_optimization_sweep_config,
    create_modular_optimization_sweep_config,
    create_hybrid_optimization_sweep_config,
    create_fully_connected_optimization_sweep_config,
)

def launch_comprehensive_sweep():
    """Launch a comprehensive sweep over all hyperparameters."""
    print("🚀 Launching comprehensive hyperparameter sweep...")
    
    # Create sweep configuration
    sweep_config = create_sweep_config()
    sweep_config['name'] = 'comprehensive_topology_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Comprehensive sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_focused_sweep(focus_area='ppo'):
    """Launch a focused sweep on specific hyperparameter areas."""
    print(f"🚀 Launching {focus_area}-focused hyperparameter sweep...")
    
    # Create focused sweep configuration
    sweep_config = create_focused_sweep_config(focus_area)
    sweep_config['name'] = f'{focus_area}_focused_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {focus_area}-focused sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_task_specific_sweep(task='CartPole-v1'):
    """Launch a task-specific sweep optimized for particular environments."""
    print(f"🚀 Launching {task}-specific hyperparameter sweep...")
    
    # Create task-specific sweep configuration
    sweep_config = create_task_specific_sweep_config(task)
    sweep_config['name'] = f'{task}_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {task}-specific sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_custom_sweep(sweep_config, name="custom_optimization"):
    """Launch a custom sweep with user-defined configuration."""
    print(f"🚀 Launching custom hyperparameter sweep: {name}...")
    
    # Add name to sweep configuration
    sweep_config['name'] = name
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Custom sweep '{name}' created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

# ============================================================================
# BASELINE SWEEP FUNCTIONS
# ============================================================================

def launch_baseline_comprehensive_sweep():
    """Launch a comprehensive baseline sweep over all hyperparameters."""
    print("🚀 Launching comprehensive baseline hyperparameter sweep...")
    
    # Create baseline sweep configuration
    sweep_config = create_baseline_sweep_config()
    sweep_config['name'] = 'comprehensive_baseline_optimization'
    
    # Create baseline sweep agent configuration
    agent_config = create_baseline_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Comprehensive baseline sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_baseline_focused_sweep(focus_area='ppo'):
    """Launch a focused baseline sweep on specific hyperparameter areas."""
    print(f"🚀 Launching {focus_area}-focused baseline hyperparameter sweep...")
    
    # Create focused baseline sweep configuration
    sweep_config = create_baseline_focused_sweep_config(focus_area)
    sweep_config['name'] = f'{focus_area}_focused_baseline_optimization'
    
    # Create baseline sweep agent configuration
    agent_config = create_baseline_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {focus_area}-focused baseline sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_baseline_custom_sweep(sweep_config, name="custom_baseline_optimization"):
    """Launch a custom baseline sweep with user-defined configuration."""
    print(f"🚀 Launching custom baseline hyperparameter sweep: {name}...")
    
    # Add name to sweep configuration
    sweep_config['name'] = name
    
    # Create baseline sweep agent configuration
    agent_config = create_baseline_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Custom baseline sweep '{name}' created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

# ============================================================================
# DOUBLE-TASK SWEEP FUNCTIONS
# ============================================================================

def launch_double_task_comprehensive_sweep():
    """Launch a comprehensive double-task sweep over all hyperparameters."""
    print("🚀 Launching comprehensive double-task hyperparameter sweep...")
    
    # Create double-task sweep configuration
    sweep_config = create_double_task_sweep_config()
    sweep_config['name'] = 'comprehensive_double_task_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Comprehensive double-task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_double_task_focused_sweep(focus_area='ppo'):
    """Launch a focused double-task sweep on specific hyperparameter areas."""
    print(f"🚀 Launching {focus_area}-focused double-task hyperparameter sweep...")
    
    # Create focused double-task sweep configuration
    sweep_config = create_focused_double_task_sweep_config(focus_area)
    sweep_config['name'] = f'{focus_area}_focused_double_task_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {focus_area}-focused double-task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_double_task_custom_sweep(sweep_config, name="custom_double_task_optimization"):
    """Launch a custom double-task sweep with user-defined configuration."""
    print(f"🚀 Launching custom double-task hyperparameter sweep: {name}...")
    
    # Add name to sweep configuration
    sweep_config['name'] = name
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Custom double-task sweep '{name}' created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

# ============================================================================
# TRIPLE-TASK SWEEP FUNCTIONS
# ============================================================================

def launch_triple_task_comprehensive_sweep():
    """Launch a comprehensive triple-task sweep over all hyperparameters."""
    print("🚀 Launching comprehensive triple-task hyperparameter sweep...")
    
    # Create triple-task sweep configuration
    sweep_config = create_triple_task_sweep_config()
    sweep_config['name'] = 'comprehensive_triple_task_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Comprehensive triple-task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_triple_task_focused_sweep(focus_area='ppo'):
    """Launch a focused triple-task sweep on specific hyperparameter areas."""
    print(f"🚀 Launching {focus_area}-focused triple-task hyperparameter sweep...")
    
    # Create focused triple-task sweep configuration
    sweep_config = create_focused_triple_task_sweep_config(focus_area)
    sweep_config['name'] = f'{focus_area}_focused_triple_task_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {focus_area}-focused triple-task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_triple_task_custom_sweep(sweep_config, name="custom_triple_task_optimization"):
    """Launch a custom triple-task sweep with user-defined configuration."""
    print(f"🚀 Launching custom triple-task hyperparameter sweep: {name}...")
    
    # Add name to sweep configuration
    sweep_config['name'] = name
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Custom triple-task sweep '{name}' created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

# ============================================================================
# TOPOLOGY COMPARISON AND OPTIMIZATION LAUNCHERS
# ============================================================================

def launch_topology_comparison_sweep():
    """Launch a topology comparison sweep for fair head-to-head comparison."""
    print("🔬 Launching topology comparison sweep...")
    sweep_config = create_topology_comparison_sweep_config()
    sweep_config['name'] = 'topology_comparison_fair_comparison'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Combinations: 4 topologies × 3 tasks = 12 total")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Topology comparison sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_topology_optimization_sweep(topology_type='small_world'):
    """Launch a topology-specific optimization sweep."""
    print(f"🎯 Launching {topology_type} optimization sweep...")
    sweep_config = create_topology_optimization_sweep_config(topology_type)
    sweep_config['name'] = f'{topology_type}_optimization'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Topology: {topology_type}")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ {topology_type} optimization sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_all_topology_optimizations():
    """Launch optimization sweeps for all topology types."""
    print("🚀 Launching optimization sweeps for all topology types...")
    topology_types = ['small_world', 'modular', 'hybrid', 'fully_connected']
    sweep_ids = {}
    
    for topology_type in topology_types:
        print(f"\n📊 Optimizing {topology_type} topology...")
        sweep_id = launch_topology_optimization_sweep(topology_type)
        sweep_ids[topology_type] = sweep_id
    
    print(f"\n✅ All topology optimization sweeps launched!")
    print("📋 Sweep IDs:")
    for topology_type, sweep_id in sweep_ids.items():
        print(f"   • {topology_type}: {sweep_id}")
    
    return sweep_ids


def launch_meta_analysis_sweep():
    """Launch a meta-analysis sweep to compare optimized topologies."""
    print("🔍 Launching meta-analysis sweep...")
    sweep_config = create_meta_analysis_sweep_config()
    sweep_config['name'] = 'meta_analysis_optimized_comparison'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Combinations: 4 topologies × 3 presets × 3 tasks = 36 total")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Meta-analysis sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_capacity_matched_comparison_sweep():
    """Launch a capacity-matched comparison sweep."""
    print("⚖️ Launching capacity-matched comparison sweep...")
    sweep_config = create_capacity_matched_comparison_sweep()
    sweep_config['name'] = 'capacity_matched_comparison'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Combinations: 4 topologies × 4 capacities × 3 tasks = 48 total")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Capacity-matched comparison sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_topology_analysis_suite():
    """Launch the comprehensive topology analysis suite."""
    print("🎯 Topology Analysis Suite")
    print("==========================")
    print("1. Topology Comparison (Fair head-to-head)")
    print("2. Small World Optimization")
    print("3. Modular Optimization") 
    print("4. Hybrid Optimization")
    print("5. Fully Connected Optimization")
    print("6. All Topology Optimizations (Sequential)")
    print("7. Meta-Analysis (Optimized vs Optimized)")
    print("8. Capacity-Matched Comparison")
    print("9. Back to main menu")
    
    choice = input("\nEnter your choice (1-9): ")
    
    if choice == "1":
        return launch_topology_comparison_sweep()
    elif choice == "2":
        return launch_topology_optimization_sweep('small_world')
    elif choice == "3":
        return launch_topology_optimization_sweep('modular')
    elif choice == "4":
        return launch_topology_optimization_sweep('hybrid')
    elif choice == "5":
        return launch_topology_optimization_sweep('fully_connected')
    elif choice == "6":
        return launch_all_topology_optimizations()
    elif choice == "7":
        return launch_meta_analysis_sweep()
    elif choice == "8":
        return launch_capacity_matched_comparison_sweep()
    elif choice == "9":
        return main()
    else:
        print("❌ Invalid choice. Please try again.")
        return launch_topology_analysis_suite()

def main():
    """Main function to launch sweeps based on user input."""
    print("🎯 Weights & Biases Sweep Launcher for Topology Networks")
    print("============================================================")
    print("\nTraining Types:")
    print("1. Single-task training (cross-task evaluation)")
    print("2. Baseline training (same-task evaluation only)")
    print("3. Double-task training (sequential)")
    print("4. Triple-task training (sequential)")
    
    training_type = input("\nSelect training type (1-4): ").strip()
    
    print("\nAnalysis Types:")
    print("1. Topology Comparison (Fair head-to-head)")
    print("2. Topology Optimization (Individual tuning)")
    print("3. Meta-Analysis (Optimized vs optimized)")
    print("4. Capacity-Matched Comparison")
    print("5. Individual Topology Optimization (Choose specific topology)")
    print("6. Comprehensive (All parameters)")
    
    analysis_type = input("\nSelect analysis type (1-6): ").strip()
    
    # Map selections to actual values
    training_map = {'1': 'single_task', '2': 'baseline', '3': 'double_task', '4': 'triple_task'}
    analysis_map = {'1': 'topology_comparison', '2': 'topology_optimization', '3': 'meta_analysis', '4': 'capacity_matched', '5': 'individual_topology_optimization', '6': 'comprehensive'}
    
    training_type = training_map.get(training_type, 'single_task')
    analysis_type = analysis_map.get(analysis_type, 'comprehensive')
    
    print(f"\n🎯 Launching {training_type} training with {analysis_type} analysis...")
    
    # Handle individual topology optimization
    if analysis_type == 'individual_topology_optimization':
        return launch_individual_topology_optimization_for_training_type(training_type)
    
    # Launch appropriate sweep based on selections
    if training_type == 'single_task':
        if analysis_type == 'comprehensive':
            launch_comprehensive_sweep()
        else:
            launch_focused_sweep(analysis_type)
    elif training_type == 'baseline':
        if analysis_type == 'comprehensive':
            launch_baseline_comprehensive_sweep()
        else:
            launch_baseline_focused_sweep(analysis_type)
    elif training_type == 'double_task':
        if analysis_type == 'comprehensive':
            launch_double_task_comprehensive_sweep()
        else:
            launch_double_task_focused_sweep(analysis_type)
    elif training_type == 'triple_task':
        if analysis_type == 'comprehensive':
            launch_triple_task_comprehensive_sweep()
        else:
            launch_triple_task_focused_sweep(analysis_type)
    else:
        print("❌ Invalid training type selection.")
        return main()


def launch_focused_sweep(analysis_type='topology_comparison'):
    """Launch focused sweep for single-task training."""
    print(f"🔬 Launching single-task {analysis_type} sweep...")
    sweep_config = create_focused_sweep_config(analysis_type)
    sweep_config['name'] = f'single_task_{analysis_type}'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Single-task {analysis_type} sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_baseline_focused_sweep(analysis_type='topology_comparison'):
    """Launch focused sweep for baseline training."""
    print(f"🔬 Launching baseline {analysis_type} sweep...")
    sweep_config = create_baseline_focused_sweep_config(analysis_type)
    sweep_config['name'] = f'baseline_{analysis_type}'
    agent_config = create_baseline_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Baseline {analysis_type} sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_double_task_focused_sweep(analysis_type='topology_comparison'):
    """Launch focused sweep for double-task training."""
    print(f"🔬 Launching double-task {analysis_type} sweep...")
    sweep_config = create_focused_double_task_sweep_config(analysis_type)
    sweep_config['name'] = f'double_task_{analysis_type}'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Double-task {analysis_type} sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_triple_task_focused_sweep(analysis_type='topology_comparison'):
    """Launch focused sweep for triple-task training."""
    print(f"🔬 Launching triple-task {analysis_type} sweep...")
    sweep_config = create_focused_triple_task_sweep_config(analysis_type)
    sweep_config['name'] = f'triple_task_{analysis_type}'
    agent_config = create_sweep_agent_config()
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Triple-task {analysis_type} sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_individual_topology_optimization_for_training_type(training_type):
    """Launch individual topology optimization for a specific training type."""
    print(f"\n🎯 Individual Topology Optimization for {training_type} training")
    print("=" * 60)
    print("1. Small World Optimization")
    print("2. Modular Optimization")
    print("3. Hybrid Optimization")
    print("4. Fully Connected Optimization")
    print("5. All Topologies (Sequential)")
    
    choice = input("\nSelect topology to optimize (1-5): ").strip()
    
    if choice == "1":
        return launch_small_world_optimization_for_training_type(training_type)
    elif choice == "2":
        return launch_modular_optimization_for_training_type(training_type)
    elif choice == "3":
        return launch_hybrid_optimization_for_training_type(training_type)
    elif choice == "4":
        return launch_fully_connected_optimization_for_training_type(training_type)
    elif choice == "5":
        return launch_all_topology_optimizations_for_training_type(training_type)
    else:
        print("❌ Invalid choice.")
        return launch_individual_topology_optimization_for_training_type(training_type)


def launch_small_world_optimization_for_training_type(training_type):
    """Launch Small World optimization for a specific training type."""
    print(f"🔬 Launching Small World optimization for {training_type} training...")
    
    if training_type == 'single_task':
        sweep_config = create_small_world_optimization_sweep_config()
        agent_config = create_sweep_agent_config()
    elif training_type == 'baseline':
        sweep_config = create_small_world_optimization_sweep_config('topologies--baseline-training-sweep.py')
        agent_config = create_baseline_sweep_agent_config()
    elif training_type == 'double_task':
        sweep_config = create_small_world_optimization_sweep_config('topologies--double-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    elif training_type == 'triple_task':
        sweep_config = create_small_world_optimization_sweep_config('topologies--triple-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    else:
        print("❌ Invalid training type.")
        return None
    
    sweep_config['name'] = f'{training_type}_small_world_optimization'
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Training Type: {training_type}")
    print(f"   • Topology: Small World (fixed)")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Small World optimization sweep for {training_type} created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_modular_optimization_for_training_type(training_type):
    """Launch Modular optimization for a specific training type."""
    print(f"🔬 Launching Modular optimization for {training_type} training...")
    
    if training_type == 'single_task':
        sweep_config = create_modular_optimization_sweep_config()
        agent_config = create_sweep_agent_config()
    elif training_type == 'baseline':
        sweep_config = create_modular_optimization_sweep_config('topologies--baseline-training-sweep.py')
        agent_config = create_baseline_sweep_agent_config()
    elif training_type == 'double_task':
        sweep_config = create_modular_optimization_sweep_config('topologies--double-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    elif training_type == 'triple_task':
        sweep_config = create_modular_optimization_sweep_config('topologies--triple-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    else:
        print("❌ Invalid training type.")
        return None
    
    sweep_config['name'] = f'{training_type}_modular_optimization'
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Training Type: {training_type}")
    print(f"   • Topology: Modular (fixed)")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Modular optimization sweep for {training_type} created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_hybrid_optimization_for_training_type(training_type):
    """Launch Hybrid optimization for a specific training type."""
    print(f"🔬 Launching Hybrid optimization for {training_type} training...")
    
    if training_type == 'single_task':
        sweep_config = create_hybrid_optimization_sweep_config()
        agent_config = create_sweep_agent_config()
    elif training_type == 'baseline':
        sweep_config = create_hybrid_optimization_sweep_config('topologies--baseline-training-sweep.py')
        agent_config = create_baseline_sweep_agent_config()
    elif training_type == 'double_task':
        sweep_config = create_hybrid_optimization_sweep_config('topologies--double-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    elif training_type == 'triple_task':
        sweep_config = create_hybrid_optimization_sweep_config('topologies--triple-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    else:
        print("❌ Invalid training type.")
        return None
    
    sweep_config['name'] = f'{training_type}_hybrid_optimization'
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Training Type: {training_type}")
    print(f"   • Topology: Hybrid (fixed)")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Hybrid optimization sweep for {training_type} created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_fully_connected_optimization_for_training_type(training_type):
    """Launch Fully Connected optimization for a specific training type."""
    print(f"🔬 Launching Fully Connected optimization for {training_type} training...")
    
    if training_type == 'single_task':
        sweep_config = create_fully_connected_optimization_sweep_config()
        agent_config = create_sweep_agent_config()
    elif training_type == 'baseline':
        sweep_config = create_fully_connected_optimization_sweep_config('topologies--baseline-training-sweep.py')
        agent_config = create_baseline_sweep_agent_config()
    elif training_type == 'double_task':
        sweep_config = create_fully_connected_optimization_sweep_config('topologies--double-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    elif training_type == 'triple_task':
        sweep_config = create_fully_connected_optimization_sweep_config('topologies--triple-task-training-sweep.py')
        agent_config = create_sweep_agent_config()
    else:
        print("❌ Invalid training type.")
        return None
    
    sweep_config['name'] = f'{training_type}_fully_connected_optimization'
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Parameters: {len(sweep_config['parameters'])}")
    print(f"   • Training Type: {training_type}")
    print(f"   • Topology: Fully Connected (fixed)")
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    print(f"✅ Fully Connected optimization sweep for {training_type} created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    return sweep_id


def launch_all_topology_optimizations_for_training_type(training_type):
    """Launch all topology optimization sweeps for a specific training type."""
    print(f"🚀 Launching all topology optimization sweeps for {training_type} training...")
    sweep_ids = []
    
    # Launch Small World optimization
    print(f"\n1️⃣ Small World Optimization for {training_type}:")
    sweep_ids.append(launch_small_world_optimization_for_training_type(training_type))
    
    # Launch Modular optimization
    print(f"\n2️⃣ Modular Optimization for {training_type}:")
    sweep_ids.append(launch_modular_optimization_for_training_type(training_type))
    
    # Launch Hybrid optimization
    print(f"\n3️⃣ Hybrid Optimization for {training_type}:")
    sweep_ids.append(launch_hybrid_optimization_for_training_type(training_type))
    
    # Launch Fully Connected optimization
    print(f"\n4️⃣ Fully Connected Optimization for {training_type}:")
    sweep_ids.append(launch_fully_connected_optimization_for_training_type(training_type))
    
    print(f"\n✅ All topology optimization sweeps for {training_type} launched!")
    print(f"📊 Sweep IDs: {sweep_ids}")
    return sweep_ids


if __name__ == "__main__":
    main() 