import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import seaborn as sns

def load_curriculum_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load curriculum results from JSON files."""
    results = []
    # Find the most recent results directory
    result_dirs = sorted(Path(results_dir).glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not result_dirs:
        raise FileNotFoundError(f"No results found in {results_dir}")
    
    # Load results from the most recent directory
    result_file = result_dirs[0] / 'curriculum_results.json'
    if not result_file.exists():
        raise FileNotFoundError(f"No curriculum_results.json found in {result_dirs[0]}")
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    return results

def extract_learning_data(results: list) -> Dict[str, Dict[str, List[float]]]:
    """Extract learning curves data from results."""
    learning_data = {}
    
    for experiment in results:
        if 'task_metrics' not in experiment:
            continue  # skip invalid or incomplete entries
        topology = experiment['topology']
        if topology not in learning_data:
            learning_data[topology] = {}
            
        for task in experiment['task_metrics']:
            if task not in learning_data[topology]:
                learning_data[topology][task] = []
            
            # Extract training rewards
            rewards = [episode['reward'] for episode in experiment['task_metrics'][task]['training_metrics']]
            learning_data[topology][task].append(rewards)
            print(f"Learning data for {topology} - {task}: {len(rewards)} rewards")
    
    return learning_data

def extract_forgetting_data(results: list) -> Dict[str, Dict[str, List[float]]]:
    """Extract forgetting curves data from results."""
    forgetting_data = {}
    
    for experiment in results:
        if 'task_metrics' not in experiment:
            continue  # skip invalid or incomplete entries
        topology = experiment['topology']
        if topology not in forgetting_data:
            forgetting_data[topology] = {}
            
        for task in experiment['task_metrics']:
            if task not in forgetting_data[topology]:
                forgetting_data[topology][task] = []
            
            # Extract retention test rewards
            if 'retention_tests' in experiment['task_metrics'][task]:
                for test in experiment['task_metrics'][task]['retention_tests']:
                    rewards = [episode['reward'] for episode in test['episodes']]
                    forgetting_data[topology][task].append(rewards)
                    print(f"Forgetting data for {topology} - {task}: {len(rewards)} rewards")
    
    return forgetting_data

def normalize_rewards(rewards: List[float], task: str) -> List[float]:
    """Normalize rewards based on task type."""
    if task == 'cartpole':
        # CartPole: higher is better, normalize to [0,1]
        return [r/500.0 for r in rewards]  # 500 is max reward
    else:
        # MountainCar and Acrobot: lower is better, normalize to [0,1]
        return [1.0 - (r/-200.0) for r in rewards]  # -200 is typical min reward

def plot_learning_curves(learning_data: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Plot learning curves for each task and topology."""
    plt.style.use('seaborn')
    
    for task in ['cartpole', 'mountain_car', 'acrobot']:
        plt.figure(figsize=(12, 8))
        
        for topology in learning_data:
            if task in learning_data[topology]:
                # Calculate mean and std across seeds
                rewards = np.array(learning_data[topology][task])
                mean_rewards = np.mean(rewards, axis=0)
                std_rewards = np.std(rewards, axis=0)
                
                # Plot mean with confidence interval
                x = np.arange(len(mean_rewards))
                plt.plot(x, mean_rewards, label=topology, linewidth=2)
                plt.fill_between(x, 
                               mean_rewards - std_rewards,
                               mean_rewards + std_rewards,
                               alpha=0.2)
        
        plt.title(f'Learning Curves - {task.replace("_", " ").title()}')
        plt.xlabel('Training Episodes')
        plt.ylabel('Normalized Reward')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f'learning_curves_{task}.png'
        plt.savefig(output_path)
        plt.close()

def plot_forgetting_curves(forgetting_data: Dict[str, Dict[str, List[float]]], output_dir: str):
    """Plot forgetting curves for each task and topology."""
    plt.style.use('seaborn')
    
    for task in ['cartpole', 'mountain_car', 'acrobot']:
        plt.figure(figsize=(12, 8))
        
        for topology in forgetting_data:
            if task in forgetting_data[topology]:
                # Calculate mean and std across retention tests
                rewards = np.array(forgetting_data[topology][task])
                mean_rewards = np.mean(rewards, axis=0)
                std_rewards = np.std(rewards, axis=0)
                
                # Normalize rewards
                mean_rewards = normalize_rewards(mean_rewards, task)
                std_rewards = [std/500.0 if task == 'cartpole' else std/200.0 
                             for std in std_rewards]
                
                # Plot mean with confidence interval
                x = np.arange(len(mean_rewards))
                plt.plot(x, mean_rewards, label=topology, linewidth=2)
                plt.fill_between(x, 
                               np.maximum(0, mean_rewards - std_rewards),
                               np.minimum(1, mean_rewards + std_rewards),
                               alpha=0.2)
        
        plt.title(f'Forgetting Curves - {task.replace("_", " ").title()}')
        plt.xlabel('Retention Test Episodes')
        plt.ylabel('Normalized Performance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f'forgetting_curves_{task}.png'
        plt.savefig(output_path)
        plt.close()

def main():
    # Load results
    results_dir = 'results/test_curriculum'
    results = load_curriculum_results(results_dir)
    
    # Create output directory
    output_dir = 'analysis_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract and plot learning curves
    learning_data = extract_learning_data(results)
    plot_learning_curves(learning_data, output_dir)
    
    # Extract and plot forgetting curves
    forgetting_data = extract_forgetting_data(results)
    plot_forgetting_curves(forgetting_data, output_dir)

if __name__ == '__main__':
    main() 