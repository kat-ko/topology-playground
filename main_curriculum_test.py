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
from src.utils.parameter_budget import ParameterBudgetCalculator
import torch

def verify_capacity_matching(config):
    """Quick verification of capacity matching for all configurations."""
    print("\nVerifying Capacity Matching")
    print("==========================")
    
    calculator = ParameterBudgetCalculator(config)
    sizes = config['network_sizes']
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    experiment_types = config['experiment_types']
    
    for size in sizes:
        print(f"\nSize: {size}")
        for topology in topologies:
            print(f"\nTopology: {topology}")
            for exp_type in experiment_types:
                # Create network
                network = calculator.create_network(
                    topology=topology,
                    size=size,
                    experiment_type=exp_type
                )
                
                # Get target and actual capacity
                target_capacity = calculator._compute_budget(exp_type, topology, size)
                actual_capacity = sum(p.numel() for p in network.parameters() if p.requires_grad)
                divergence = abs(actual_capacity - target_capacity) / target_capacity * 100
                
                # Print results
                print(f"\n{exp_type}:")
                print(f"Target: {target_capacity}, Actual: {actual_capacity}")
                print(f"Divergence: {divergence:.2f}%")
                if divergence <= 5.0:
                    print("✓ Within 5% threshold")
                else:
                    print("⚠ Exceeds 5% threshold")

def main():
    """Run a smoke test of the curriculum learning experiments."""
    # Setup logging
    logger = setup_logger(__name__)
    logger.setLevel(LogLevel.DEBUG.value)
    logger.info("Starting curriculum learning smoke test")
    
    # Create test configuration
    config = TestCurriculumConfig().to_dict()
    logger.info("Test configuration created with reduced parameters")
    
    # Print test-specific configuration details
    print("\nTest Configuration Details:")
    print("=" * 50)
    print("This is a smoke test with reduced parameters:")
    print(f"- Network size: {config['network_sizes'][0]} (reduced from full experiment)")
    print(f"- Single seed: {config['seeds'][0]}")
    print(f"- Single layer: {config['num_layers'][0]}")
    print(f"- Single network type: {config['network_types'][0]}")
    print(f"- Full task curriculum: {', '.join(config['task_sequence'])}")
    print(f"- Single node selection strategy: {config['node_selection_strategies'][0]}")
    print(f"- Reduced experiment types: {config['experiment_types']}")
    print(f"- Transfer learning tasks:")
    print(f"  * Backward transfer: {config['backward_transfer_tasks']}")
    print(f"  * Forward transfer: {config['forward_transfer_tasks']}")
    print(f"- Reduced retention testing: {config['forgetting_test']['retention_episodes']} episodes")
    
    # Create output directory for test results
    output_dir = Path("results/test_curriculum")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify capacity matching before starting
    verify_capacity_matching(config)
    
    try:
        # Initialize and run curriculum
        runner = CurriculumRunner(config)
        logger.info("Starting curriculum runner")
        runner.run_curriculum()
        logger.info("Curriculum runner completed successfully")
        
    except Exception as e:
        logger.error(f"Error during curriculum test: {str(e)}", exc_info=True)
        raise
    
    logger.info("Smoke test completed")

if __name__ == "__main__":
    main() 