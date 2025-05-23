from config.test_config import TestConfig
from src.experiment.runner import ExperimentRunner
from tests.test_topology_integrity import TestTopologyIntegrity
import unittest

def main():
    """Run the test configuration using the same structure as main.py."""
    # Create test configuration
    config = TestConfig()
    
    # Create and run experiment
    runner = ExperimentRunner(config.to_dict())
    runner.run_experiment()
    
    # Run topology integrity tests
    print("\nRunning topology integrity tests...")
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    main() 