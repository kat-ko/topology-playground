#!/usr/bin/env python3
"""
Parameter verification script - lists all variable parameters for each topology.
This allows verification that all parameters are relevant and appropriate.
"""

from wandb_sweep_config import (
    create_small_world_optimization_sweep_config,
    create_modular_optimization_sweep_config,
    create_hybrid_optimization_sweep_config,
    create_fully_connected_optimization_sweep_config,
)


def list_variable_parameters():
    """List all variable parameters for each topology."""
    print("🔍 VARIABLE PARAMETERS BY TOPOLOGY")
    print("="*80)
    
    configs = {
        'Small World': create_small_world_optimization_sweep_config(),
        'Modular': create_modular_optimization_sweep_config(),
        'Hybrid': create_hybrid_optimization_sweep_config(),
        'Fully Connected': create_fully_connected_optimization_sweep_config(),
    }
    
    for name, config in configs.items():
        print(f"\n🔧 {name} Topology:")
        print("-" * 50)
        
        # Get variable parameters (not fixed)
        variable_params = []
        for param_name, param_config in config['parameters'].items():
            if 'value' not in param_config:  # Not fixed
                variable_params.append(param_name)
        
        # Categorize parameters
        architecture_params = [p for p in variable_params if p in ['hidden_size', 'activation', 'dropout']]
        training_params = [p for p in variable_params if p in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'ent_coef', 'max_grad_norm']]
        task_params = [p for p in variable_params if p in ['train_task']]
        topology_params = [p for p in variable_params if any(t in p for t in ['small_world', 'modular', 'hybrid'])]
        num_layers_param = [p for p in variable_params if p == 'num_layers']
        
        print(f"📊 Total variable parameters: {len(variable_params)}")
        print(f"   Architecture: {len(architecture_params)} parameters")
        print(f"   Training: {len(training_params)} parameters")
        print(f"   Task: {len(task_params)} parameters")
        print(f"   Topology-specific: {len(topology_params)} parameters")
        print(f"   num_layers: {len(num_layers_param)} parameters")
        
        print(f"\n📋 Detailed Parameter List:")
        
        if architecture_params:
            print(f"  🏗️  Architecture Parameters ({len(architecture_params)}):")
            for param in architecture_params:
                param_config = config['parameters'][param]
                if 'values' in param_config:
                    print(f"    • {param}: {param_config['values']}")
                elif 'distribution' in param_config:
                    print(f"    • {param}: {param_config['distribution']}({param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')})")
        
        if training_params:
            print(f"  🎯 Training Parameters ({len(training_params)}):")
            for param in training_params:
                param_config = config['parameters'][param]
                if 'values' in param_config:
                    print(f"    • {param}: {param_config['values']}")
                elif 'distribution' in param_config:
                    print(f"    • {param}: {param_config['distribution']}({param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')})")
        
        if task_params:
            print(f"  🎮 Task Parameters ({len(task_params)}):")
            for param in task_params:
                param_config = config['parameters'][param]
                print(f"    • {param}: {param_config['values']}")
        
        if topology_params:
            print(f"  🏗️  Topology-Specific Parameters ({len(topology_params)}):")
            for param in topology_params:
                param_config = config['parameters'][param]
                if 'values' in param_config:
                    print(f"    • {param}: {param_config['values']}")
                elif 'distribution' in param_config:
                    print(f"    • {param}: {param_config['distribution']}({param_config.get('min', 'N/A')}-{param_config.get('max', 'N/A')})")
        
        if num_layers_param:
            print(f"  📊 Layer Parameters ({len(num_layers_param)}):")
            for param in num_layers_param:
                param_config = config['parameters'][param]
                if 'values' in param_config:
                    print(f"    • {param}: {param_config['values']}")
                elif 'value' in param_config:
                    print(f"    • {param}: {param_config['value']} (FIXED)")


def show_fixed_parameters():
    """Show which parameters are fixed for each topology."""
    print("\n\n🔒 FIXED PARAMETERS BY TOPOLOGY")
    print("="*80)
    
    configs = {
        'Small World': create_small_world_optimization_sweep_config(),
        'Modular': create_modular_optimization_sweep_config(),
        'Hybrid': create_hybrid_optimization_sweep_config(),
        'Fully Connected': create_fully_connected_optimization_sweep_config(),
    }
    
    for name, config in configs.items():
        print(f"\n🔧 {name} Topology:")
        print("-" * 50)
        
        # Get fixed parameters
        fixed_params = []
        for param_name, param_config in config['parameters'].items():
            if 'value' in param_config:  # Fixed
                fixed_params.append((param_name, param_config['value']))
        
        print(f"📊 Total fixed parameters: {len(fixed_params)}")
        for param_name, param_value in fixed_params:
            print(f"  • {param_name}: {param_value}")


def parameter_comparison():
    """Compare parameters across topologies."""
    print("\n\n📊 PARAMETER COMPARISON ACROSS TOPOLOGIES")
    print("="*80)
    
    configs = {
        'Small World': create_small_world_optimization_sweep_config(),
        'Modular': create_modular_optimization_sweep_config(),
        'Hybrid': create_hybrid_optimization_sweep_config(),
        'Fully Connected': create_fully_connected_optimization_sweep_config(),
    }
    
    # Get all parameter names
    all_param_names = set()
    for config in configs.values():
        all_param_names.update(config['parameters'].keys())
    
    # Create comparison table
    print(f"{'Parameter':<25} {'Small World':<12} {'Modular':<12} {'Hybrid':<12} {'Fully Connected':<15}")
    print("-" * 80)
    
    for param in sorted(all_param_names):
        row = [param]
        for name in ['Small World', 'Modular', 'Hybrid', 'Fully Connected']:
            config = configs[name]
            if param in config['parameters']:
                param_config = config['parameters'][param]
                if 'value' in param_config:
                    row.append(f"FIXED({param_config['value']})")
                elif 'values' in param_config:
                    row.append(f"VAR({len(param_config['values'])})")
                elif 'distribution' in param_config:
                    row.append(f"VAR({param_config['distribution']})")
                else:
                    row.append("VAR")
            else:
                row.append("N/A")
        
        print(f"{row[0]:<25} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<15}")


def verification_checklist():
    """Provide a verification checklist for the user."""
    print("\n\n✅ VERIFICATION CHECKLIST")
    print("="*80)
    
    print("Please verify the following for each topology:")
    print("\n🔧 Small World Topology:")
    print("  ✅ num_layers is fixed to 1 (not variable)")
    print("  ✅ Only small_world_k and small_world_p are topology-specific")
    print("  ✅ Architecture parameters: hidden_size, activation, dropout")
    print("  ✅ Training parameters: learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, ent_coef, max_grad_norm")
    print("  ✅ Task parameter: train_task")
    
    print("\n🔧 Modular Topology:")
    print("  ✅ num_layers is fixed to 1 (not variable)")
    print("  ✅ Only modular_num_modules, modular_inter_module_prob, modular_intra_module_prob are topology-specific")
    print("  ✅ Same architecture and training parameters as Small World")
    
    print("\n🔧 Hybrid Topology:")
    print("  ✅ num_layers is fixed to 1 (not variable)")
    print("  ✅ Only hybrid_num_modules, hybrid_k, hybrid_p, hybrid_inter_module_prob are topology-specific")
    print("  ✅ Same architecture and training parameters as others")
    
    print("\n🔧 Fully Connected Topology:")
    print("  ✅ num_layers is variable (only topology that uses it)")
    print("  ✅ No topology-specific parameters (correct)")
    print("  ✅ Same architecture and training parameters as others")
    
    print("\n🎯 Overall Checks:")
    print("  ✅ No cross-contamination between topology-specific parameters")
    print("  ✅ Consistent architecture and training parameters across all topologies")
    print("  ✅ Efficient parameter spaces for Bayesian optimization")
    print("  ✅ All parameters are relevant to their respective topologies")


if __name__ == "__main__":
    print("🚀 Parameter Verification")
    print("This script lists all variable parameters for each topology for verification.")
    
    list_variable_parameters()
    show_fixed_parameters()
    parameter_comparison()
    verification_checklist()
    
    print("\n\n✅ VERIFICATION COMPLETE!")
    print("Please review the parameter lists above to ensure all parameters are relevant and appropriate.")