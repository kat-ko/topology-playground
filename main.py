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