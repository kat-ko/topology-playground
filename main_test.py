"""
# first activate the conda environment
.\venv\Scripts\Activate.ps1

# then run main.py
python main.py

# then run experiment analysis
python src/analysis/evaluate_results.py results/20240315_123456 --output-dir analysis_results

"""

from config.test_config import TestConfig
from src.experiment.runner import ExperimentRunner
import logging
from src.utils.logging_utils import setup_logger, LogLevel

logger = setup_logger(__name__)

def main():
    """Run the test configuration using the same structure as main.py."""
    # Create test configuration
    config = TestConfig()
    
    # Set up logging for test run
    logger.setLevel(LogLevel.DEBUG.value)
    logger.info("Running in TEST mode")
    
    # Create and run experiment
    runner = ExperimentRunner(config.to_dict(), output_dir="results")
    runner.run_experiment()
    
    logger.info("Test run completed. Results saved in results directory.")

if __name__ == "__main__":
    main() 