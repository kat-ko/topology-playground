"""
# first activate the conda environment
.\venv\Scripts\Activate.ps1

# then run main.py
python main.py

# then run experiment analysis
python src/analysis/evaluate_results.py results/20240315_123456 --output-dir analysis_results

"""



import logging
from src.curriculum.runner import CurriculumRunner
from config.curriculum_config import CurriculumConfig
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

def verify_capacity_matching(config, divergence_threshold=5.0):
    """Comprehensive verification of capacity matching using the two-phase approach with incremental adjustment."""
    print("\n" + "="*80)
    print("CAPACITY MATCHING VERIFICATION")
    print("="*80)
    
    calculator = ParameterBudgetCalculator(config)
    sizes = config['network_sizes']
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    experiment_types = config['experiment_types']
    network_types = config['network_types']
    node_selection_strategies = config['node_selection_strategies']
    num_layers_list = config['num_layers']
    seeds = config['seeds']
    
    # Track results for summary
    results_summary = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {}
    }
    
    # Organize by experiment type for better comprehension
    for exp_type in experiment_types:
        print(f"\n{'='*20} EXPERIMENT TYPE: {exp_type.upper()} {'='*20}")
        
        if exp_type == 'same_size':
            print("All topologies use their natural size and capacity")
        else:
            reference_topology = exp_type[len('match_'):]
            print(f"All topologies matched to {reference_topology} capacity")
        
        for size in sizes:
            print(f"\n--- Network Size: {size} ---")
            
            # Show reference capacities for match_* experiments
            if exp_type.startswith('match_'):
                print(f"Reference topology ({reference_topology}) capacities:")
                for network_type in network_types:
                    for num_layers in num_layers_list:
                        ref_capacity = calculator.get_budget(exp_type, reference_topology, size, network_type, num_layers)
                        print(f"  {network_type.upper()} | {num_layers} layer(s): {ref_capacity} parameters")
                print()
            
            # Test each topology
            for topology in topologies:
                print(f"\n  Topology: {topology.upper()}")
                print(f"  {'-' * (len(topology) + 10)}")
                
                for network_type in network_types:
                    for num_layers in num_layers_list:
                        for seed in seeds:
                            for strategy in node_selection_strategies:
                                # Set random seed for consistent results
                                torch.manual_seed(seed)
                                np.random.seed(seed)
                                
                                config_key = f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}_{strategy}"
                                
                                try:
                                    if exp_type == 'same_size':
                                        # For same_size experiments, show natural capacity
                                        network = calculator.create_network(
                                            topology=topology,
                                            size=size,
                                            experiment_type=exp_type,
                                            network_type=network_type,
                                            num_layers=num_layers,
                                            seed=seed
                                        )
                                        metrics = network.get_network_metrics()
                                        natural_capacity = sum(
                                            metrics.get(k, 0) for k in metrics if k.startswith('num_')
                                        )
                                        
                                        print(f"    {network_type.upper()} | {strategy} | {num_layers}L | seed={seed}:")
                                        print(f"      Natural capacity: {natural_capacity:,} parameters")
                                        print(f"      ✅ Same size experiment - no matching required")
                                        
                                        results_summary['passed'] += 1
                                        results_summary['details'][config_key] = {
                                            'status': 'passed',
                                            'capacity': natural_capacity,
                                            'type': 'natural'
                                        }
                                        
                                    else:
                                        # For match_* experiments, use incremental adjustment logic
                                        reference_topology = exp_type[len('match_'):]
                                        target_capacity = calculator.get_budget(exp_type, topology, size, network_type, num_layers)
                                        
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
                                        # Use 'same_size' to avoid recursive matching lookup
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
                                        
                                        if divergence <= divergence_threshold:
                                            print(f"      ✅ Within threshold ({divergence_threshold}%)")
                                            results_summary['passed'] += 1
                                            status = 'passed'
                                        else:
                                            print(f"      ⚠️  Exceeds threshold ({divergence_threshold}%)")
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
                                
                                print()  # Spacing between configurations
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("CAPACITY MATCHING SUMMARY")
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
                    print(f"    Target: {details['target_capacity']:,} | Actual: {details['actual_capacity']:,} | Divergence: {details['divergence']:.2f}%")
                elif details['status'] == 'error':
                    print(f"    Error: {details['error']}")
    
    # Final decision
    if results_summary['failed'] == 0 and results_summary['errors'] == 0:
        print(f"\n🎉 ALL CONFIGURATIONS PASSED! Ready for training.")
    else:
        print(f"\n⚠️  {results_summary['failed'] + results_summary['errors']} CONFIGURATIONS NEED FIXING BEFORE TRAINING.")
        print("   Review the problematic configurations above and adjust:")
        print("   - Empirical scaling formulas in EMPIRICAL_SCALING_MODELS")
        print("   - Dynamic multipliers for capacity ranges")
        print("   - Network constructor parameters")
        sys.exit(1)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create curriculum configuration
    config = CurriculumConfig().to_dict()
    
    # Disable capacity mapping to use the optimized incremental adjustment logic
    config['use_capacity_mapping'] = False
    
    # Verify capacity matching before starting (using actual training logic)
    verify_capacity_matching(config)
    
    # Create and run curriculum experiment
    runner = CurriculumRunner(config)
    logger.info("Starting curriculum experiment...")
    runner.run_curriculum()
    logger.info("Curriculum experiment completed. Results saved in 'results' directory.")

if __name__ == "__main__":
    main() 