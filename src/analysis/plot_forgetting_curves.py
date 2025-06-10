import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_curriculum_results(results_dir):
    """Load curriculum results from JSON files."""
    results = []
    for result_file in Path(results_dir).glob('**/curriculum_results.json'):
        with open(result_file, 'r') as f:
            results.extend(json.load(f))
    return results

def extract_forgetting_data(results):
    """Extract forgetting data from results."""
    forgetting_data = {}
    
    for result in results:
        topology = result['topology']
        if topology not in forgetting_data:
            forgetting_data[topology] = {
                'cartpole': {'rewards': [], 'iterations': []},
                'mountain_car': {'rewards': [], 'iterations': []},
                'acrobot': {'rewards': [], 'iterations': []}
            }
        
        # Extract performance history
        for task, performances in result['curriculum_results']['performance_history'].items():
            for eval_task, metrics in performances.items():
                if isinstance(metrics, dict) and 'mean_reward' in metrics:
                    forgetting_data[topology][eval_task]['rewards'].append(metrics['mean_reward'])
                    forgetting_data[topology][eval_task]['iterations'].append(len(forgetting_data[topology][eval_task]['rewards']))
    
    return forgetting_data

def normalize_rewards(rewards, task):
    """Normalize rewards based on task-specific thresholds."""
    if task == 'cartpole':
        # For CartPole, higher is better (max reward is 500)
        return np.array(rewards) / 500.0
    else:
        # For MountainCar and Acrobot, lower is better (min reward is -200)
        return np.array(rewards) / -200.0

def plot_forgetting_curves(forgetting_data, output_dir):
    """Plot forgetting curves for each task and topology with improved visualization."""
    tasks = ['cartpole', 'mountain_car', 'acrobot']
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    for task in tasks:
        plt.figure(figsize=(12, 8))
        
        for topology in topologies:
            if topology in forgetting_data and forgetting_data[topology][task]['rewards']:
                rewards = np.array(forgetting_data[topology][task]['rewards'])
                iterations = np.array(forgetting_data[topology][task]['iterations'])
                
                # Ensure rewards is 1D array
                if rewards.ndim > 1:
                    rewards = rewards.flatten()
                
                # Normalize rewards
                normalized_rewards = normalize_rewards(rewards, task)
                
                # Plot with confidence interval
                plt.plot(iterations, normalized_rewards, label=topology, linewidth=2)
                
                # Add confidence interval if we have multiple runs
                if len(rewards) > 1:
                    std_rewards = np.std(normalized_rewards)
                    plt.fill_between(iterations,
                                   np.maximum(0, normalized_rewards - std_rewards),
                                   np.minimum(1, normalized_rewards + std_rewards),
                                   alpha=0.2)
        
        plt.title(f'Forgetting Curves - {task.replace("_", " ").title()}', fontsize=14)
        plt.xlabel('Task Iteration', fontsize=12)
        plt.ylabel('Normalized Performance', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / f'forgetting_curves_{task}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to plot forgetting curves."""
    # Load results
    results_dir = 'results/test_curriculum'
    results = load_curriculum_results(results_dir)
    
    # Extract forgetting data
    forgetting_data = extract_forgetting_data(results)
    
    # Plot forgetting curves
    output_dir = 'analysis_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_forgetting_curves(forgetting_data, output_dir)

if __name__ == '__main__':
    main() 