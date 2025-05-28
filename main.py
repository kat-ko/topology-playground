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

def main():
    # Create configuration
    config = ExperimentConfig()
    
    # Run experiment
    runner = ExperimentRunner(config.__dict__)
    runner.run_experiment()

if __name__ == "__main__":
    main() 