"""
# first activate the conda environment
.\venv\Scripts\Activate.ps1

# then run main.py
python main.py

# then run experiment analysis
python src/analysis/evaluate_results.py results/20240315_123456 --output-dir analysis_results

"""



from config.experiment_config import ExperimentConfig
from src.experiment.runner import ExperimentRunner
from tests.test_topology_integrity import TestTopologyIntegrity
import unittest

def main():
    """Run the experiment with the specified configuration."""
    # Create configuration
    config = ExperimentConfig()
    
    # Create and run experiment
    runner = ExperimentRunner(config.to_dict())
    runner.run_experiment()
    
    # Run topology integrity tests
    print("\nRunning topology integrity tests...")
    unittest.main(argv=[''], exit=False)

if __name__ == "__main__":
    main() 