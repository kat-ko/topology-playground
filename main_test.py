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

def main():
    """Run the test configuration using the same structure as main.py."""
    # Create test configuration
    config = TestConfig()
    
    # Create and run experiment
    runner = ExperimentRunner(config.to_dict())
    runner.run_experiment()

if __name__ == "__main__":
    main() 