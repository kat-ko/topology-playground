"""
# first activate the conda environment
.\venv\Scripts\Activate.ps1


# then run main.py
python main.py

# then run experiment analysis
python src/analysis/evaluate_results.py results/20240315_123456 --output-dir analysis_results

"""

import logging
from pathlib import Path
from config.test_curriculum_config import TestCurriculumConfig
from src.curriculum.runner import CurriculumRunner
from src.utils.logging_utils import setup_logger, LogLevel
from src.utils.parameter_budget import ParameterBudgetCalculator, calculate_network_size
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.fully_connected import FullyConnectedTopology
from src.networks.ffn import FeedForwardNetwork
from src.networks.rnn import RecurrentNetwork
import torch
import sys
import numpy as np
from tabulate import tabulate
from src.utils.capacity_measurement import CapacityMeasurementManager

def verify_capacity_matching(config, divergence_threshold=10.0):
    """
    Verify capacity matching for all experiment types and configurations.
    Uses a two-phase approach: baseline measurement and capacity matching verification.
    """
    print("="*80)
    print("CAPACITY MATCHING VERIFICATION (TEST MODE)")
    print("="*80)
    
    # Disable capacity mapping to force incremental adjustment
    config['use_capacity_mapping'] = False
    
    # Initialize measurement manager for baseline measurements
    measurement_manager = CapacityMeasurementManager(config)
    
    # Extract configuration parameters
    sizes = config['network_sizes']
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    network_types = config['network_types']
    num_layers_list = config['num_layers']
    seeds = config['seeds']
    experiment_types = [et for et in config['experiment_types'] if et.startswith('match_')]
    node_selection_strategies = config['node_selection_strategies']

    # --- Baseline measurement phase ---
    print("\nBaseline measurement phase: measuring all required capacities...")
    for topology in topologies:
        for size in sizes:
            for network_type in network_types:
                for num_layers in num_layers_list:
                    for seed in seeds:
                        if measurement_manager.get_measurement(topology, size, network_type, num_layers) is None:
                            actual_capacity = measurement_manager.measure_capacity(topology, size, network_type, num_layers, seed)
                            measurement_manager.store_measurement(topology, size, network_type, num_layers, actual_capacity, seed)
    measurement_manager._save_measurements()
    print("Baseline measurement phase complete.\n")

    # Track results for summary
    results_summary = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {}
    }
    
    # Create calculator AFTER disabling capacity mapping
    calculator = ParameterBudgetCalculator(config)
    
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
    print("CAPACITY MATCHING SUMMARY (TEST MODE)")
    print("="*80)
    
    total_tests = results_summary['passed'] + results_summary['failed'] + results_summary['errors']
    print(f"Total configurations tested: {total_tests}")
    print(f"✅ Passed: {results_summary['passed']} ({results_summary['passed']/total_tests*100:.1f}%)")
    print(f"⚠️  Failed: {results_summary['failed']} ({results_summary['failed']/total_tests*100:.1f}%)")
    print(f"❌ Errors: {results_summary['errors']} ({results_summary['errors']/total_tests*100:.1f}%)")
    
    # Show problematic configurations
    if results_summary['failed'] > 0 or results_summary['errors'] > 0:
        print(f"\n🔧 CONFIGURATIONS NEEDING ATTENTION:")
        print("-" * 50)
        
        for config_key, details in results_summary['details'].items():
            if details['status'] in ['failed', 'error']:
                print(f"  {config_key}: {details['status'].upper()}")
                if details['status'] == 'failed':
                    print(f"    Target: {details['target_capacity']} | Actual: {details['actual_capacity']} | Divergence: {details['divergence']:.2f}%")
                elif details['status'] == 'error':
                    print(f"    Error: {details['error']}")
                print()
        
        print(f"⚠️  {results_summary['failed'] + results_summary['errors']} CONFIGURATIONS NEED FIXING BEFORE TRAINING.")
        print("   Review the problematic configurations above and adjust:")
        print("   - Empirical scaling formulas in EMPIRICAL_SCALING_MODELS")
        print("   - Dynamic multipliers for capacity ranges")
        print("   - Network constructor parameters")
    else:
        print("\n🎉 ALL CONFIGURATIONS PASSED! Ready for training.")
    
    return results_summary

def main():
    """Run a smoke test of the curriculum learning experiments."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting curriculum learning smoke test")
    
    # Create test configuration
    config = TestCurriculumConfig()
    logger.info("Test configuration created with reduced parameters")
    
    # Print test configuration details
    print("\nTest Configuration Details:")
    print("="*50)
    print("This is a smoke test with reduced parameters:")
    print(f"- Network size: {config.network_sizes[0]} (reduced from full experiment)")
    print(f"- Single seed: {config.seeds[0]}")
    print(f"- Single layer: {config.num_layers[0]}")
    print(f"- Single network type: {config.network_types[0]}")
    print(f"- Full task curriculum: {', '.join(config.task_sequence)}")
    print(f"- Single node selection strategy: {config.node_selection_strategies[0]}")
    print(f"- Reduced experiment types: {config.experiment_types}")
    print(f"- Transfer learning tasks:")
    print(f"  * Backward transfer: {config.backward_transfer_tasks}")
    print(f"  * Forward transfer: {config.forward_transfer_tasks}")
    print(f"- Reduced retention testing: {config.forgetting_test['retention_episodes']} episodes")
    
    # Convert to dict for compatibility
    config_dict = config.to_dict()
    
    # Run capacity matching verification
    results = verify_capacity_matching(config_dict, divergence_threshold=10.0)
    
    # Final decision based on results
    if results['failed'] == 0 and results['errors'] == 0:
        print(f"\n🎉 ALL CONFIGURATIONS PASSED! Ready for training.")
    else:
        print(f"\n⚠️  {results['failed'] + results['errors']} CONFIGURATIONS NEED FIXING BEFORE TRAINING.")
        sys.exit(1)
    
    logger.info("Smoke test completed")

if __name__ == "__main__":
    main() 