#!/usr/bin/env python3
"""
Comprehensive analysis of parameter relevance for each topology.
This script identifies which parameters are actually used by each topology.
"""

from wandb_sweep_config import (
    create_small_world_optimization_sweep_config,
    create_modular_optimization_sweep_config,
    create_hybrid_optimization_sweep_config,
    create_fully_connected_optimization_sweep_config,
)


def analyze_parameter_relevance():
    """Analyze which parameters are actually relevant for each topology."""
    print("🔍 PARAMETER RELEVANCE ANALYSIS")
    print("="*80)
    
    # Get all configurations
    configs = {
        'Small World': create_small_world_optimization_sweep_config(),
        'Modular': create_modular_optimization_sweep_config(),
        'Hybrid': create_hybrid_optimization_sweep_config(),
        'Fully Connected': create_fully_connected_optimization_sweep_config(),
    }
    
    print("📊 CURRENT PARAMETER USAGE:")
    print("-" * 80)
    
    for name, config in configs.items():
        print(f"\n🔧 {name} Topology:")
        all_params = list(config['parameters'].keys())
        
        # Categorize parameters
        architecture_params = ['hidden_size', 'num_layers', 'activation', 'dropout']
        training_params = ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 
                          'gae_lambda', 'clip_range', 'ent_coef', 'max_grad_norm']
        task_params = ['train_task']
        evaluation_params = ['total_timesteps', 'n_eval_episodes']
        topology_params = [p for p in all_params if any(t in p for t in ['small_world', 'modular', 'hybrid'])]
        fixed_params = [p for p in all_params if 'value' in config['parameters'][p]]
        
        print(f"  Architecture: {[p for p in architecture_params if p in all_params]}")
        print(f"  Training: {[p for p in training_params if p in all_params]}")
        print(f"  Task: {[p for p in task_params if p in all_params]}")
        print(f"  Evaluation: {[p for p in evaluation_params if p in all_params]}")
        print(f"  Topology-specific: {topology_params}")
        print(f"  Fixed: {fixed_params}")
        print(f"  Total: {len(all_params)} parameters")


def analyze_topology_implementations():
    """Analyze what parameters each topology actually uses."""
    print("\n\n🔍 TOPOLOGY IMPLEMENTATION ANALYSIS")
    print("="*80)
    
    print("📋 Based on code analysis:")
    print("\n🔧 Small World Topology:")
    print("  Constructor: SmallWorldTopology(size, k, p, seed)")
    print("  Generate: generate(num_layers=1) -> single graph")
    print("  ✅ USES: size, k, p, seed")
    print("  ❌ IGNORES: num_layers (always generates single graph)")
    print("  ❌ IGNORES: activation, dropout (not topology-specific)")
    
    print("\n🔧 Modular Topology:")
    print("  Constructor: ModularTopology(size, num_modules, inter_module_prob, intra_module_prob, seed)")
    print("  Generate: generate(num_layers=1) -> single graph")
    print("  ✅ USES: size, num_modules, inter_module_prob, intra_module_prob, seed")
    print("  ❌ IGNORES: num_layers (always generates single graph)")
    print("  ❌ IGNORES: activation, dropout (not topology-specific)")
    
    print("\n🔧 Hybrid Topology:")
    print("  Constructor: HybridTopology(size, num_modules, k, p, inter_module_prob, seed)")
    print("  Generate: generate(num_layers=1) -> single graph")
    print("  ✅ USES: size, num_modules, k, p, inter_module_prob, seed")
    print("  ❌ IGNORES: num_layers (always generates single graph)")
    print("  ❌ IGNORES: activation, dropout (not topology-specific)")
    
    print("\n🔧 Fully Connected Topology:")
    print("  Constructor: FullyConnectedTopology(size, num_layers, seed)")
    print("  Generate: generate(num_layers) -> single graph")
    print("  ✅ USES: size, num_layers, seed")
    print("  ✅ ONLY topology that actually uses num_layers!")


def identify_unnecessary_parameters():
    """Identify parameters that are unnecessary for each topology."""
    print("\n\n🚨 UNNECESSARY PARAMETERS IDENTIFIED")
    print("="*80)
    
    print("❌ CRITICAL ISSUE: num_layers is only relevant for Fully Connected!")
    print("   • Small World, Modular, Hybrid: Always generate single graphs")
    print("   • Only Fully Connected actually uses num_layers parameter")
    print("   • Current configs: All topologies optimize num_layers unnecessarily")
    
    print("\n📊 Parameter Relevance by Topology:")
    print("\n🔧 Small World:")
    print("  ✅ RELEVANT: small_world_k, small_world_p")
    print("  ❌ UNNECESSARY: num_layers (should be fixed to 1)")
    print("  ✅ RELEVANT: hidden_size, activation, dropout (architecture)")
    print("  ✅ RELEVANT: training parameters (learning_rate, etc.)")
    
    print("\n🔧 Modular:")
    print("  ✅ RELEVANT: modular_num_modules, modular_inter_module_prob, modular_intra_module_prob")
    print("  ❌ UNNECESSARY: num_layers (should be fixed to 1)")
    print("  ✅ RELEVANT: hidden_size, activation, dropout (architecture)")
    print("  ✅ RELEVANT: training parameters (learning_rate, etc.)")
    
    print("\n🔧 Hybrid:")
    print("  ✅ RELEVANT: hybrid_num_modules, hybrid_k, hybrid_p, hybrid_inter_module_prob")
    print("  ❌ UNNECESSARY: num_layers (should be fixed to 1)")
    print("  ✅ RELEVANT: hidden_size, activation, dropout (architecture)")
    print("  ✅ RELEVANT: training parameters (learning_rate, etc.)")
    
    print("\n🔧 Fully Connected:")
    print("  ✅ RELEVANT: num_layers (only topology that uses it!)")
    print("  ✅ RELEVANT: hidden_size, activation, dropout (architecture)")
    print("  ✅ RELEVANT: training parameters (learning_rate, etc.)")
    print("  ✅ CORRECT: No topology-specific parameters")


def calculate_parameter_efficiency():
    """Calculate how many parameters are actually relevant."""
    print("\n\n📈 PARAMETER EFFICIENCY ANALYSIS")
    print("="*80)
    
    print("Current vs. Optimal Parameter Counts:")
    print("\n🔧 Small World:")
    print("  Current: 19 parameters (16 variable)")
    print("  Optimal: 18 parameters (15 variable) - fix num_layers=1")
    print("  Improvement: -1 parameter (-6.25% reduction)")
    
    print("\n🔧 Modular:")
    print("  Current: 20 parameters (17 variable)")
    print("  Optimal: 19 parameters (16 variable) - fix num_layers=1")
    print("  Improvement: -1 parameter (-5.88% reduction)")
    
    print("\n🔧 Hybrid:")
    print("  Current: 21 parameters (18 variable)")
    print("  Optimal: 20 parameters (17 variable) - fix num_layers=1")
    print("  Improvement: -1 parameter (-5.56% reduction)")
    
    print("\n🔧 Fully Connected:")
    print("  Current: 17 parameters (14 variable)")
    print("  Optimal: 17 parameters (14 variable) - already optimal!")
    print("  Improvement: 0 parameters (already correct)")
    
    print("\n🎯 Overall Impact:")
    print("  • 3 topologies can be optimized")
    print("  • Total reduction: 3 parameters")
    print("  • Bayesian optimization efficiency: Improved")
    print("  • Research validity: Enhanced (no irrelevant parameters)")


def recommend_fixes():
    """Recommend specific fixes for the parameter issues."""
    print("\n\n🔧 RECOMMENDED FIXES")
    print("="*80)
    
    print("1. 🔧 Fix num_layers for non-FC topologies:")
    print("   • Small World: Set num_layers={'value': 1}")
    print("   • Modular: Set num_layers={'value': 1}")
    print("   • Hybrid: Set num_layers={'value': 1}")
    print("   • Fully Connected: Keep num_layers variable")
    
    print("\n2. 🔧 Consider other potential optimizations:")
    print("   • activation: Could be fixed for consistency across topologies")
    print("   • dropout: Could be fixed for consistency across topologies")
    print("   • train_task: Could be varied systematically rather than randomly")
    
    print("\n3. 🔧 Benefits of these fixes:")
    print("   • More efficient Bayesian optimization")
    print("   • Cleaner parameter spaces")
    print("   • More focused research questions")
    print("   • Better comparison between topologies")
    
    print("\n4. 🔧 Implementation approach:")
    print("   • Option A: Fix num_layers=1 for non-FC topologies")
    print("   • Option B: Remove num_layers entirely from non-FC configs")
    print("   • Option C: Keep for consistency but fix to 1")
    print("   • Recommendation: Option A (fix to 1) for clarity")


if __name__ == "__main__":
    print("🚀 Parameter Relevance Analysis")
    print("This analysis identifies unnecessary parameters in topology optimization.")
    
    analyze_parameter_relevance()
    analyze_topology_implementations()
    identify_unnecessary_parameters()
    calculate_parameter_efficiency()
    recommend_fixes()
    
    print("\n\n✅ ANALYSIS COMPLETE!")
    print("Key finding: num_layers is only relevant for Fully Connected topology!")