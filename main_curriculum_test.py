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

def main():
    """Run a smoke test of the curriculum learning experiments."""
    # Setup logging
    logger = setup_logger(__name__)
    logger.setLevel(LogLevel.DEBUG.value)
    logger.info("Starting curriculum learning smoke test")
    
    # Create test configuration
    config = TestCurriculumConfig()
    logger.info("Test configuration created with reduced parameters")
    
    # Print test-specific configuration details
    print("\nTest Configuration Details:")
    print("=" * 50)
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
    
    # Create output directory for test results
    output_dir = Path("results/test_curriculum")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize and run curriculum
        runner = CurriculumRunner(config.to_dict(), str(output_dir))
        logger.info("Starting curriculum runner")
        runner.run_curriculum()
        logger.info("Curriculum runner completed successfully")
        
    except Exception as e:
        logger.error(f"Error during curriculum test: {str(e)}", exc_info=True)
        raise
    
    logger.info("Smoke test completed")

if __name__ == "__main__":
    main() 