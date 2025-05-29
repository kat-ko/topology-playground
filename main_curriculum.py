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

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create curriculum configuration
    config = CurriculumConfig()
    
    # Create and run curriculum experiment
    runner = CurriculumRunner(config.to_dict())
    logger.info("Starting curriculum experiment...")
    runner.run_curriculum()
    logger.info("Curriculum experiment completed. Results saved in 'results' directory.")

if __name__ == "__main__":
    main() 