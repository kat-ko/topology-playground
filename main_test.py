import sys
import logging
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from experiment.runner import ExperimentRunner
from config.test_config import TestConfig

def setup_logging():
    """Configure logging for the test run."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_run.log')
        ]
    )

def validate_results(results):
    """Basic validation of experiment results."""
    logging.info("Validating results...")
    
    # Check if results are not empty
    if not results:
        raise ValueError("No results were generated")
    
    # Check each result entry
    for result in results:
        # Validate required fields
        required_fields = [
            'topology_type', 'network_type', 'size', 'seed',
            'strategy', 'task', 'num_layers', 'metrics'
        ]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate metrics
        metrics = result['metrics']
        if not isinstance(metrics, dict):
            raise ValueError("Metrics should be a dictionary")
        
        # Log the result
        logging.info(f"Valid result: {result['topology_type']} - "
                    f"{result['network_type']} - {result['task']} - "
                    f"Layers: {result['num_layers']}")

def main():
    """Run the test experiment."""
    setup_logging()
    logging.info("Starting test experiment...")
    
    try:
        # Initialize test configuration
        config = TestConfig()
        
        # Create and run experiment
        runner = ExperimentRunner(config.to_dict())
        results = runner.run_experiment()
        
        # Validate results
        validate_results(results)
        
        logging.info("Test experiment completed successfully!")
        return 0
        
    except Exception as e:
        logging.error(f"Test experiment failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 