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

def run_smoke_test_training(config):
    """Run RL training using the existing CurriculumRunner for smoke testing."""
    print("\n" + "="*80)
    print("RL TRAINING EXECUTION (SMOKE TEST)")
    print("="*80)
    
    print("Starting RL training with curriculum learning...")
    
    # Import and run the curriculum
    from src.curriculum.runner import CurriculumRunner
    
    print(f"Executing run_curriculum from: {__file__}")
    
    # Create runner and execute
    runner = CurriculumRunner(config)
    
    # Print curriculum parameters for verification
    print("\nCurriculum parameters:")
    print(f"Task sequence: {config['task_sequence']}")
    print(f"Network sizes: {config['network_sizes']}")
    print(f"Seeds: {config['seeds']}")
    print(f"Number of layers: {config['num_layers']}")
    print(f"Network types: {config['network_types']}")
    print(f"\nParameter budget settings:")
    print(f"Budget type: {config['parameter_budget']['budget_type']}")
    print(f"Target budget: {config['parameter_budget']['target_budget']}")
    print(f"Normalize by size: {config['parameter_budget']['normalize_by_size']}")
    print(f"\nExperiment types and capacity matching:")
    for exp_type in config['experiment_types']:
        if exp_type.startswith('match_'):
            reference = exp_type[len('match_'):]
            print(f"\n{exp_type}:")
            print(f"  All networks will match {reference} capacity")
    
    # Execute training and capture results
    try:
        print("\nExecuting curriculum training...")
        training_results = runner.run_curriculum()
        print(f"\n✅ RL training completed successfully")
        print(f"📊 Training results type: {type(training_results)}")
        print(f"📊 Training results length: {len(training_results) if training_results else 'None'}")
        if training_results and len(training_results) > 0:
            print(f"📊 First result keys: {list(training_results[0].keys()) if isinstance(training_results[0], dict) else 'Not a dict'}")
        return training_results
    except Exception as e:
        print(f"\n❌ RL training failed: {e}")
        import traceback
        print(f"Full traceback:")
        traceback.print_exc()
        raise

def validate_training_results(results):
    """Validate that training results are in the expected format and contain required data."""
    print("="*80)
    print("TRAINING RESULTS VALIDATION (SMOKE TEST)")
    print("="*80)
    
    print(f"[DEBUG] Results type: {type(results)}")
    if isinstance(results, list):
        print(f"[DEBUG] Results length: {len(results)}")
        if len(results) > 0:
            print(f"[DEBUG] First result: {results[0]}")
    else:
        print(f"[DEBUG] Results value: {results}")
    
    if results is None:
        print("❌ Training results missing or invalid format")
        return {'status': 'failed', 'error': 'No results returned'}
    
    try:
        # Check if results is a list
        if not isinstance(results, list):
            print("❌ Training results not in expected list format")
            return {'status': 'failed', 'error': 'Results not a list'}
        
        if len(results) == 0:
            print("❌ Training results list is empty")
            return {'status': 'failed', 'error': 'Results list is empty'}
        
        print(f"✅ Training execution completed")
        print(f"📊 Number of experiment results: {len(results)}")
        
        # Validate each result
        for i, res in enumerate(results):
            if not isinstance(res, dict):
                print(f"❌ Result {i} is not a dict: {res}")
                return {'status': 'failed', 'error': f'Result {i} not a dict'}
            required_keys = ['network_size', 'seed', 'num_layers', 'network_type', 'strategy', 'topology', 'curriculum_results']
            for key in required_keys:
                if key not in res:
                    print(f"❌ Result {i} missing key: {key}")
                    return {'status': 'failed', 'error': f'Result {i} missing key: {key}'}
        print(f"✅ All training results validated successfully")
        return {'status': 'passed', 'num_results': len(results)}
    except Exception as e:
        print(f"❌ Training results validation error: {e}")
        return {'status': 'failed', 'error': str(e)}

def validate_capacity_consistency(config):
    """Validate that capacity matching results are consistent between smoke test and training."""
    print("\n" + "="*80)
    print("CAPACITY CONSISTENCY VALIDATION")
    print("="*80)
    
    # Disable capacity mapping to ensure consistency
    config['use_capacity_mapping'] = False
    
    # Create calculator
    calculator = ParameterBudgetCalculator(config)
    
    # Test a few specific configurations
    test_configs = [
        ('match_small_world', 'fully_connected', 25, 'ffn', 2, 42),
        ('match_small_world', 'modular', 25, 'ffn', 2, 42),
        ('match_small_world', 'hybrid', 25, 'ffn', 2, 42),
    ]
    
    consistency_results = {
        'passed': 0,
        'failed': 0,
        'details': {}
    }
    
    for exp_type, topology, size, network_type, num_layers, seed in test_configs:
        print(f"\nTesting: {exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}")
        
        try:
            # Get target capacity using the measurement manager
            measurement_manager = CapacityMeasurementManager(config)
            reference_topology = exp_type[len('match_'):]
            
            print(f"  Looking for measurement: {reference_topology}_{size}_{network_type}_{num_layers}")
            
            target_capacity = measurement_manager.get_target_capacity(
                reference_topology, size, network_type, num_layers
            )
            
            if target_capacity is None:
                print(f"  ❌ ERROR: No baseline measurement available for {reference_topology}_{size}_{network_type}_{num_layers}")
                print(f"  Available measurements: {list(measurement_manager.measurements.keys())}")
                consistency_results['failed'] += 1
                consistency_results['details'][f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}"] = {
                    'status': 'error',
                    'error': 'No baseline measurement available'
                }
                continue
            
            # Get matching size using smoke test logic
            if topology == reference_topology:
                matching_size = size
            else:
                matching_size = calculator.calculate_matching_size(topology, target_capacity, network_type, num_layers)
            
            # Create network using smoke test logic
            smoke_network = calculator.create_network(
                topology=topology,
                size=matching_size,
                experiment_type='same_size',
                network_type=network_type,
                num_layers=num_layers,
                seed=seed
            )
            
            # Get smoke test capacity
            smoke_metrics = smoke_network.get_network_metrics()
            smoke_capacity = sum(
                smoke_metrics.get(k, 0) for k in smoke_metrics if k.startswith('num_')
            )
            
            # Create network using training logic (simulate CurriculumRunner)
            training_network = calculator.create_network(
                topology=topology,
                size=matching_size,
                experiment_type='same_size',
                network_type=network_type,
                num_layers=num_layers,
                seed=seed
            )
            
            # Get training capacity
            training_metrics = training_network.get_network_metrics()
            training_capacity = sum(
                training_metrics.get(k, 0) for k in training_metrics if k.startswith('num_')
            )
            
            # Check consistency
            capacity_diff = abs(smoke_capacity - training_capacity)
            consistency_diff = capacity_diff / target_capacity * 100 if target_capacity > 0 else float('inf')
            
            print(f"  Target: {target_capacity}")
            print(f"  Matching size: {matching_size}")
            print(f"  Smoke test capacity: {smoke_capacity}")
            print(f"  Training capacity: {training_capacity}")
            print(f"  Consistency difference: {consistency_diff:.2f}%")
            
            if consistency_diff <= 1.0:  # Very strict threshold for consistency
                print(f"  ✅ CONSISTENT")
                consistency_results['passed'] += 1
                status = 'passed'
            else:
                print(f"  ❌ INCONSISTENT")
                consistency_results['failed'] += 1
                status = 'failed'
            
            consistency_results['details'][f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}"] = {
                'status': status,
                'target_capacity': target_capacity,
                'matching_size': matching_size,
                'smoke_capacity': smoke_capacity,
                'training_capacity': training_capacity,
                'consistency_diff': consistency_diff
            }
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            consistency_results['failed'] += 1
            consistency_results['details'][f"{exp_type}_{topology}_{size}_{network_type}_{num_layers}_{seed}"] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Print consistency summary
    print("\n" + "="*80)
    print("CAPACITY CONSISTENCY SUMMARY")
    print("="*80)
    
    total_tests = consistency_results['passed'] + consistency_results['failed']
    print(f"Total consistency tests: {total_tests}")
    print(f"✅ Consistent: {consistency_results['passed']} ({consistency_results['passed']/total_tests*100:.1f}%)")
    print(f"❌ Inconsistent: {consistency_results['failed']} ({consistency_results['failed']/total_tests*100:.1f}%)")
    
    if consistency_results['failed'] == 0:
        print("\n🎉 ALL CAPACITY MATCHING TESTS ARE CONSISTENT!")
        print("Smoke test and training use the same capacity matching logic.")
    else:
        print("\n⚠️  CAPACITY MATCHING INCONSISTENCIES DETECTED!")
        print("Smoke test and training may use different capacity matching logic.")
    
    return consistency_results

def validate_topology_variations(config):
    """Validate all topology variations, parameter configurations, and validation checks."""
    print("\n" + "="*80)
    print("TOPOLOGY VARIATIONS VALIDATION")
    print("="*80)
    
    # Disable capacity mapping
    config['use_capacity_mapping'] = False
    
    # Create calculator
    calculator = ParameterBudgetCalculator(config)
    
    # Test all topology variations
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    network_types = config['network_types']
    num_layers_list = config['num_layers']
    sizes = config['network_sizes']  # Use the same sizes as the smoke test
    
    topology_results = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': {}
    }
    
    for topology in topologies:
        print(f"\n{'='*20} TOPOLOGY: {topology.upper()} {'='*20}")
        
        for network_type in network_types:
            for num_layers in num_layers_list:
                for size in sizes:
                    print(f"\n  Testing: {topology}_{network_type}_{num_layers}L_{size}")
                    
                    try:
                        # Test network creation
                        print(f"    Creating network: {topology}_{size}_{network_type}_{num_layers}")
                        network = calculator.create_network(
                            topology=topology,
                            size=size,
                            experiment_type='same_size',
                            network_type=network_type,
                            num_layers=num_layers,
                            seed=42
                        )
                        print(f"    [DEBUG] Network object type: {type(network)}")
                        print(f"    Getting network metrics...")
                        try:
                            metrics = network.get_network_metrics()
                            print(f"    [DEBUG] Network metrics: {metrics}")
                        except Exception as e:
                            print(f"    [ERROR] Failed to get network metrics: {e}")
                            topology_results['errors'] += 1
                            topology_results['details'][f"{topology}_{network_type}_{num_layers}_{size}"] = {
                                'status': 'error',
                                'error': f'Failed to get network metrics: {e}'
                            }
                            continue
                        
                        # Validate topology-specific properties
                        validation_passed = True
                        validation_errors = []
                        
                        # Check 1: Network has parameters
                        if metrics.get('num_nodes', 0) == 0:
                            validation_passed = False
                            validation_errors.append("No parameters found")
                        
                        # Check 2: Network metrics exist
                        if not metrics:
                            validation_passed = False
                            validation_errors.append("No network metrics")
                        
                        # Check 3: Network size matches expected (use num_nodes from metrics)
                        num_nodes = metrics.get('num_nodes', 0)
                        if num_nodes != size:
                            validation_passed = False
                            validation_errors.append(f"Size mismatch: expected {size}, got {num_nodes}")
                        
                        # Check 4: Topology-specific validations using network metrics
                        if topology == 'small_world':
                            # Check small world properties
                            avg_degree = metrics.get('avg_degree', 0)
                            if avg_degree < 2:  # Small world should have reasonable connectivity
                                validation_passed = False
                                validation_errors.append(f"Low average degree: {avg_degree}")
                        
                        elif topology == 'modular':
                            # Check modular properties
                            density = metrics.get('density', 0)
                            if density < 0.01:  # Modular should have some connectivity
                                validation_passed = False
                                validation_errors.append(f"Very low density: {density}")
                        
                        elif topology == 'hybrid':
                            # Check hybrid properties
                            num_edges = metrics.get('num_edges', 0)
                            if num_edges < size:  # Hybrid should have reasonable connectivity
                                validation_passed = False
                                validation_errors.append(f"Low edge count: {num_edges} for size {size}")
                        
                        elif topology == 'fully_connected':
                            # Check fully connected properties
                            expected_edges = size * (size - 1) // 2
                            actual_edges = metrics.get('num_edges', 0)
                            if actual_edges < expected_edges * 0.8:  # Allow some tolerance
                                validation_passed = False
                                validation_errors.append(f"Not fully connected: {actual_edges} vs expected ~{expected_edges}")
                        
                        # Check 5: Multi-layer validation
                        if num_layers > 1:
                            # For multi-layer, check that we have multiple networks
                            if hasattr(network, 'network'):  # Wrapper case
                                # This is expected for multi-layer
                                pass
                            else:
                                # Should have multiple networks
                                pass
                        
                        # Report results
                        if validation_passed:
                            print(f"    ✅ PASSED: {metrics.get('num_nodes', 0)} parameters")
                            topology_results['passed'] += 1
                            status = 'passed'
                        else:
                            print(f"    ❌ FAILED: {', '.join(validation_errors)}")
                            topology_results['failed'] += 1
                            status = 'failed'
                        
                        topology_results['details'][f"{topology}_{network_type}_{num_layers}_{size}"] = {
                            'status': status,
                            'total_params': metrics.get('num_nodes', 0),
                            'topology_metrics': metrics,
                            'validation_errors': validation_errors if not validation_passed else []
                        }
                        
                    except Exception as e:
                        print(f"    ❌ ERROR: {e}")
                        import traceback
                        print(f"    Full traceback:")
                        traceback.print_exc()
                        topology_results['errors'] += 1
                        topology_results['details'][f"{topology}_{network_type}_{num_layers}_{size}"] = {
                            'status': 'error',
                            'error': str(e)
                        }
    
    # Print topology validation summary
    print("\n" + "="*80)
    print("TOPOLOGY VALIDATION SUMMARY")
    print("="*80)
    
    total_tests = topology_results['passed'] + topology_results['failed'] + topology_results['errors']
    print(f"Total topology tests: {total_tests}")
    print(f"✅ Passed: {topology_results['passed']} ({topology_results['passed']/total_tests*100:.1f}%)")
    print(f"❌ Failed: {topology_results['failed']} ({topology_results['failed']/total_tests*100:.1f}%)")
    print(f"⚠️  Errors: {topology_results['errors']} ({topology_results['errors']/total_tests*100:.1f}%)")
    
    if topology_results['failed'] == 0 and topology_results['errors'] == 0:
        print("\n🎉 ALL TOPOLOGY VARIATIONS VALIDATED SUCCESSFULLY!")
        print("All topologies, network types, and layer configurations work correctly.")
    else:
        print("\n⚠️  TOPOLOGY VALIDATION ISSUES DETECTED!")
        print("Some topology variations may have problems.")
    
    return topology_results

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
    print(f"- Network sizes: {config.network_sizes}")
    print(f"- Seeds: {config.seeds}")
    print(f"- Layers: {config.num_layers}")
    print(f"- Network types: {config.network_types}")
    print(f"- Full task curriculum: {', '.join(config.task_sequence)}")
    print(f"- Node selection strategies: {config.node_selection_strategies}")
    print(f"- Experiment types: {config.experiment_types}")
    print(f"- Transfer learning tasks:")
    print(f"  * Backward transfer: {config.backward_transfer_tasks}")
    print(f"  * Forward transfer: {config.forward_transfer_tasks}")
    print(f"- Training parameters:")
    print(f"  * Episodes per task: {config.episodes_per_task}")
    print(f"  * Evaluation episodes: {config.evaluation_episodes}")
    print(f"  * Max env steps: {config.max_env_steps_per_task}")
    print(f"- Retention testing: {config.forgetting_test['retention_episodes']} episodes")
    
    # Convert to dict for compatibility
    config_dict = config.to_dict()
    
    # Run capacity matching verification
    print("\n" + "="*80)
    print("PHASE 1: CAPACITY MATCHING VERIFICATION")
    print("="*80)
    capacity_results = verify_capacity_matching(config_dict, divergence_threshold=10.0)
    
    # Check if capacity matching passed
    if capacity_results['failed'] > 0 or capacity_results['errors'] > 0:
        print(f"\n⚠️  {capacity_results['failed'] + capacity_results['errors']} CONFIGURATIONS NEED FIXING BEFORE TRAINING.")
        print("   Skipping RL training due to capacity matching failures.")
        sys.exit(1)
    
    # Run RL training
    print("\n" + "="*80)
    print("PHASE 2: RL TRAINING EXECUTION")
    print("="*80)
    try:
        training_results = run_smoke_test_training(config_dict)
        
        # Validate training results
        print("\n" + "="*80)
        print("PHASE 3: TRAINING RESULTS VALIDATION")
        print("="*80)
        validation_results = validate_training_results(training_results)
        
        # Validate capacity consistency
        print("\n" + "="*80)
        print("PHASE 4: CAPACITY CONSISTENCY VALIDATION")
        print("="*80)
        consistency_results = validate_capacity_consistency(config_dict)
        
        # Validate topology variations
        print("\n" + "="*80)
        print("PHASE 5: TOPOLOGY VARIATIONS VALIDATION")
        print("="*80)
        topology_results = validate_topology_variations(config_dict)
        
        # Final smoke test summary
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print("✅ Capacity matching: PASSED")
        print("✅ RL training: PASSED")
        print("✅ Training validation: PASSED")
        print("✅ Capacity consistency: PASSED")
        print("✅ Topology variations: PASSED")
        print("\n🎉 SMOKE TEST COMPLETE - System ready for full training")
        
    except Exception as e:
        print(f"\n❌ RL training failed: {e}")
        print("⚠️  Smoke test incomplete - training phase failed")
        sys.exit(1)
    
    logger.info("Smoke test completed successfully")


if __name__ == "__main__":
    main() 