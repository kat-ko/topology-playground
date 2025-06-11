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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create curriculum configuration
    config = CurriculumConfig().to_dict()
    
    # Verify capacity matching before starting
    verify_capacity_matching(config)
    
    # Create and run curriculum experiment
    runner = CurriculumRunner(config)
    logger.info("Starting curriculum experiment...")
    runner.run_curriculum()
    logger.info("Curriculum experiment completed. Results saved in 'results' directory.")

if __name__ == "__main__":
    main() 