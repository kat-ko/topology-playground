#!/usr/bin/env python3
"""
Test script to verify sweep configurations work as intended.
This script shows what each analysis type would do without launching actual sweeps.
"""

import json
from wandb_sweep_config import (
    create_focused_sweep_config,
    create_baseline_focused_sweep_config,
    create_small_world_optimization_sweep_config,
    create_modular_optimization_sweep_config,
    create_hybrid_optimization_sweep_config,
    create_fully_connected_optimization_sweep_config,
)


def print_sweep_config_summary(name, config):
    """Print a summary of a sweep configuration."""
    print(f"\n{'='*60}")
    print(f"📋 {name}")
    print(f"{'='*60}")
    print(f"Program: {config['program']}")
    print(f"Method: {config['method']}")
    print(f"Metric: {config['metric']['name']}")
    
    # Count parameters by type
    fixed_params = []
    variable_params = []
    topology_params = []
    
    for param_name, param_config in config['parameters'].items():
        if 'value' in param_config:
            fixed_params.append(f"{param_name}={param_config['value']}")
        elif 'values' in param_config:
            variable_params.append(f"{param_name}={param_config['values']}")
        elif 'distribution' in param_config:
            variable_params.append(f"{param_name}={param_config['distribution']}({param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')})")
        
        # Check if it's topology-specific
        if any(topology in param_name for topology in ['small_world', 'modular', 'hybrid']):
            topology_params.append(param_name)
    
    print(f"\n📊 Parameter Summary:")
    print(f"  Fixed parameters: {len(fixed_params)}")
    print(f"  Variable parameters: {len(variable_params)}")
    print(f"  Topology-specific parameters: {len(topology_params)}")
    
    if fixed_params:
        print(f"\n🔒 Fixed Parameters:")
        for param in fixed_params[:5]:  # Show first 5
            print(f"  • {param}")
        if len(fixed_params) > 5:
            print(f"  • ... and {len(fixed_params)-5} more")
    
    if variable_params:
        print(f"\n🔄 Variable Parameters:")
        for param in variable_params[:5]:  # Show first 5
            print(f"  • {param}")
        if len(variable_params) > 5:
            print(f"  • ... and {len(variable_params)-5} more")
    
    if topology_params:
        print(f"\n🏗️  Topology-Specific Parameters:")
        for param in topology_params:
            print(f"  • {param}")


def test_analysis_types():
    """Test all analysis types for single-task training."""
    print("🧪 Testing Analysis Types for Single-Task Training")
    print("="*80)
    
    # Test 1: Topology Comparison
    config = create_focused_sweep_config('topology_comparison')
    print_sweep_config_summary("1. Topology Comparison", config)
    
    # Test 2: Topology Optimization (old approach)
    config = create_focused_sweep_config('topology_optimization')
    print_sweep_config_summary("2. Topology Optimization (All Topologies)", config)
    
    # Test 3: Individual Topology Optimizations
    config = create_small_world_optimization_sweep_config()
    print_sweep_config_summary("3. Small World Optimization (Individual)", config)
    
    config = create_modular_optimization_sweep_config()
    print_sweep_config_summary("4. Modular Optimization (Individual)", config)
    
    config = create_hybrid_optimization_sweep_config()
    print_sweep_config_summary("5. Hybrid Optimization (Individual)", config)
    
    config = create_fully_connected_optimization_sweep_config()
    print_sweep_config_summary("6. Fully Connected Optimization (Individual)", config)


def test_training_types():
    """Test that different training types work correctly."""
    print("\n\n🧪 Testing Different Training Types")
    print("="*80)
    
    # Test baseline training
    config = create_baseline_focused_sweep_config('topology_comparison')
    print_sweep_config_summary("Baseline Training - Topology Comparison", config)
    
    # Test individual topology for baseline
    config = create_small_world_optimization_sweep_config('topologies--baseline-training-sweep.py')
    print_sweep_config_summary("Baseline Training - Small World Optimization", config)


def verify_parameter_filtering():
    """Verify that individual topology optimizations only include relevant parameters."""
    print("\n\n🔍 Verifying Parameter Filtering")
    print("="*80)
    
    # Test Small World
    sw_config = create_small_world_optimization_sweep_config()
    sw_params = set(sw_config['parameters'].keys())
    
    # Test Modular
    mod_config = create_modular_optimization_sweep_config()
    mod_params = set(mod_config['parameters'].keys())
    
    # Test Hybrid
    hyb_config = create_hybrid_optimization_sweep_config()
    hyb_params = set(hyb_config['parameters'].keys())
    
    # Test Fully Connected
    fc_config = create_fully_connected_optimization_sweep_config()
    fc_params = set(fc_config['parameters'].keys())
    
    print("📊 Parameter Count Comparison:")
    print(f"  Small World: {len(sw_params)} parameters")
    print(f"  Modular: {len(mod_params)} parameters")
    print(f"  Hybrid: {len(hyb_params)} parameters")
    print(f"  Fully Connected: {len(fc_params)} parameters")
    
    # Check for topology-specific parameters
    sw_specific = [p for p in sw_params if 'small_world' in p]
    mod_specific = [p for p in mod_params if 'modular' in p]
    hyb_specific = [p for p in hyb_params if 'hybrid' in p]
    fc_specific = [p for p in fc_params if any(t in p for t in ['small_world', 'modular', 'hybrid'])]
    
    print(f"\n🏗️  Topology-Specific Parameters:")
    print(f"  Small World: {sw_specific}")
    print(f"  Modular: {mod_specific}")
    print(f"  Hybrid: {hyb_specific}")
    print(f"  Fully Connected: {fc_specific} (should be empty)")
    
    # Verify no cross-contamination
    print(f"\n✅ Cross-Contamination Check:")
    sw_has_modular = any('modular' in p for p in sw_params)
    mod_has_small_world = any('small_world' in p for p in mod_params)
    hyb_has_others = any(p in hyb_params for p in ['small_world_k', 'modular_num_modules'])
    
    print(f"  Small World has Modular params: {sw_has_modular} (should be False)")
    print(f"  Modular has Small World params: {mod_has_small_world} (should be False)")
    print(f"  Hybrid has other topology params: {hyb_has_others} (should be False)")


def test_metric_naming():
    """Verify that metric names are correctly set for different analysis types."""
    print("\n\n📈 Verifying Metric Naming")
    print("="*80)
    
    # Test different analysis types
    configs = [
        ("Topology Comparison", create_focused_sweep_config('topology_comparison')),
        ("Topology Optimization", create_focused_sweep_config('topology_optimization')),
        ("Meta Analysis", create_focused_sweep_config('meta_analysis')),
        ("Capacity Matched", create_focused_sweep_config('capacity_matched')),
        ("Small World Individual", create_small_world_optimization_sweep_config()),
    ]
    
    for name, config in configs:
        metric = config['metric']['name']
        print(f"  {name}: {metric}")


if __name__ == "__main__":
    print("🚀 Starting Sweep Configuration Verification")
    print("This script will test all analysis types without launching actual sweeps.")
    
    test_analysis_types()
    test_training_types()
    verify_parameter_filtering()
    test_metric_naming()
    
    print("\n\n✅ Verification Complete!")
    print("Check the output above to ensure each analysis type works as intended.")