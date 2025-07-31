#!/usr/bin/env python3
"""
Capacity Matching Helper for Training Scripts

This module provides a unified way to handle capacity matching logic
that can be reused across all training scripts (baseline, single-task, double-task, triple-task).
"""

import argparse
from src.utils.parameter_budget import ParameterBudgetCalculator


def pre_calculate_capacity_matching():
    """
    Pre-calculate capacity matching before wandb.init() to avoid config modification errors.
    
    Returns:
        tuple: (effective_hidden_size, target_capacity, args)
    """
    # Parse command line arguments to get sweep parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_capacity', type=int, default=None)
    parser.add_argument('--topology_type', type=str, default='small_world')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--small_world_k', type=int, default=4)
    parser.add_argument('--small_world_p', type=float, default=0.3)
    parser.add_argument('--modular_num_modules', type=int, default=4)
    parser.add_argument('--modular_inter_module_prob', type=float, default=0.2)
    parser.add_argument('--modular_intra_module_prob', type=float, default=0.8)
    parser.add_argument('--hybrid_num_modules', type=int, default=4)
    parser.add_argument('--hybrid_k', type=int, default=4)
    parser.add_argument('--hybrid_p', type=float, default=0.3)
    parser.add_argument('--hybrid_inter_module_prob', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # Parse known args (ignore unknown ones that will be handled by wandb)
    args, unknown = parser.parse_known_args()
    
    # Determine effective hidden_size based on capacity matching
    effective_hidden_size = args.hidden_size
    target_capacity = args.target_capacity
    
    if target_capacity is not None:
        print(f"🔧 Pre-calculating capacity matching...")
        print(f"   • Target: {target_capacity:,} parameters")
        print(f"   • Topology: {args.topology_type}")
        print(f"   • Original hidden_size: {args.hidden_size}")
        
        try:
            # Create ParameterBudgetCalculator with sweep configuration
            calculator_config = {
                'universal_input_dim': 6,
                'universal_output_dim': 3,
                'universal_action_dim': 3,
                'network_sizes': [64, 128, 256],
                'network_types': ['ffn'],
                'num_layers': [1, 2, 3],
                'num_io_nodes': 4,
                'experiment_types': ['same_size', 'match_small_world', 'match_modular', 'match_hybrid', 'match_fully_connected'],
                'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
                'use_capacity_mapping': False,
                'min_search_size': 10,
                'max_search_size': 35000,  # Large enough for all target capacities
                'seeds': [42],
                'small_world_params': {
                    'k': args.small_world_k,
                    'p': args.small_world_p,
                    'inter_layer_prob': 0.5
                },
                'modular_params': {
                    'num_modules': args.modular_num_modules,
                    'inter_module_prob': args.modular_inter_module_prob,
                    'intra_module_prob': args.modular_intra_module_prob,
                    'inter_layer_prob': 0.5
                },
                'hybrid_params': {
                    'num_modules': args.hybrid_num_modules,
                    'k': args.hybrid_k,
                    'p': args.hybrid_p,
                    'inter_module_prob': args.hybrid_inter_module_prob,
                    'inter_layer_prob': 0.5
                },
                'fully_connected_params': {
                    'inter_layer_prob': 1.0,
                    'intra_layer_prob': 1.0
                },
                'network_params': {
                    'ffn': {
                        'activation': args.activation,
                        'dropout': args.dropout
                    }
                }
            }
            
            calculator = ParameterBudgetCalculator(calculator_config)
            
            # Use the improved multi-stage search strategy for Small World
            if args.topology_type == 'small_world':
                # Stage 1: Better initial estimation based on empirical data
                if target_capacity >= 50000:
                    estimated_size = int(target_capacity / 3.3)
                elif target_capacity >= 10000:
                    estimated_size = int(target_capacity / 3.1)
                elif target_capacity >= 5000:
                    estimated_size = int(target_capacity / 3.05)
                else:
                    estimated_size = int(target_capacity / 3.0)
                
                best_size = estimated_size
                best_divergence = float('inf')
                best_params = 0
                
                # Stage 2: Coarse search with adaptive parameters
                if target_capacity >= 50000:
                    search_range = 8000
                    step_size = 200
                elif target_capacity >= 10000:
                    search_range = 4000
                    step_size = 100
                elif target_capacity >= 5000:
                    search_range = 2000
                    step_size = 50
                else:
                    search_range = 1000
                    step_size = 25
                
                # Coarse search
                coarse_sizes = range(
                    max(100, estimated_size - search_range), 
                    min(calculator_config['max_search_size'], estimated_size + search_range), 
                    step_size
                )
                
                for test_size in coarse_sizes:
                    try:
                        test_network = calculator._create_test_network(args.topology_type, test_size, 'ffn', args.num_layers)
                        if test_network is not None:
                            metrics = test_network.get_network_metrics()
                            actual_params = sum(metrics.get(k, 0) for k in metrics if k.startswith('num_'))
                            divergence = abs(actual_params - target_capacity) / target_capacity
                            
                            if divergence < best_divergence:
                                best_divergence = divergence
                                best_size = test_size
                                best_params = actual_params
                                
                            if divergence < 0.02:  # Excellent match
                                break
                    except:
                        continue
                
                # Stage 3: Medium-fine search around best result
                if best_divergence > 0.05:
                    medium_range = min(1000, search_range // 2)
                    medium_step = max(10, step_size // 5)
                    medium_sizes = range(
                        max(100, best_size - medium_range),
                        min(calculator_config['max_search_size'], best_size + medium_range),
                        medium_step
                    )
                    
                    for test_size in medium_sizes:
                        try:
                            test_network = calculator._create_test_network(args.topology_type, test_size, 'ffn', args.num_layers)
                            if test_network is not None:
                                metrics = test_network.get_network_metrics()
                                actual_params = sum(metrics.get(k, 0) for k in metrics if k.startswith('num_'))
                                divergence = abs(actual_params - target_capacity) / target_capacity
                                
                                if divergence < best_divergence:
                                    best_divergence = divergence
                                    best_size = test_size
                                    best_params = actual_params
                                    
                                if divergence < 0.02:  # Excellent match
                                    break
                        except:
                            continue
                
                # Stage 4: Fine-tuning with very small steps
                if best_divergence > 0.02:
                    fine_range = min(500, medium_range // 2)
                    fine_step = max(5, medium_step // 2)
                    fine_sizes = range(
                        max(100, best_size - fine_range),
                        min(calculator_config['max_search_size'], best_size + fine_range),
                        fine_step
                    )
                    
                    for test_size in fine_sizes:
                        try:
                            test_network = calculator._create_test_network(args.topology_type, test_size, 'ffn', args.num_layers)
                            if test_network is not None:
                                metrics = test_network.get_network_metrics()
                                actual_params = sum(metrics.get(k, 0) for k in metrics if k.startswith('num_'))
                                divergence = abs(actual_params - target_capacity) / target_capacity
                                
                                if divergence < best_divergence:
                                    best_divergence = divergence
                                    best_size = test_size
                                    best_params = actual_params
                                    
                                if divergence < 0.01:  # Perfect match
                                    break
                        except:
                            continue
                
                effective_hidden_size = best_size
                
            else:
                # Use standard calculator for other topologies
                effective_hidden_size = calculator.calculate_matching_size(
                    args.topology_type, target_capacity, 'ffn', args.num_layers
                )
                # Get actual parameter count for logging
                test_network = calculator._create_test_network(args.topology_type, effective_hidden_size, 'ffn', args.num_layers)
                if test_network is not None:
                    metrics = test_network.get_network_metrics()
                    best_params = sum(metrics.get(k, 0) for k in metrics if k.startswith('num_'))
                    best_divergence = abs(best_params - target_capacity) / target_capacity
                else:
                    best_params = 0
                    best_divergence = float('inf')
            
            print(f"   ✅ Capacity matching completed!")
            print(f"   • Calculated size: {effective_hidden_size}")
            print(f"   • Actual parameters: {best_params:,}")
            print(f"   • Divergence: {best_divergence:.3f} ({best_divergence*100:.1f}%)")
            
        except Exception as e:
            print(f"   ❌ Capacity matching failed: {e}")
            print(f"   • Using original hidden_size: {args.hidden_size}")
            effective_hidden_size = args.hidden_size
    
    return effective_hidden_size, target_capacity, args 