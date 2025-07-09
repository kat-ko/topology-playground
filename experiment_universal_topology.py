"""
UNIVERSAL TOPOLOGY EXPERIMENT: Single Task with Universal Topology + Adapters
Objective: Test the universal topology approach with task-specific input/output adapters.

Key Concept:
- Universal topology backbone with maximum expected dimensions (6 inputs, 3 outputs)
- Task-specific input/output adapters that project to/from task dimensions
- No padding/masking - uses learnable projections instead
- Enables transfer learning across curriculum tasks

Factor	Setting
Task	CartPole-v1 (4 inputs, 2 outputs)
Universal topology	6 inputs, 3 outputs (max expected across curriculum)
Adapters	Input: 4->6, Output: 3->2 (learnable projections)
Network type	ffn
Layers	1
Size	100, 200
Topologies	small_world, modular, fully_connected, hybrid
Seed	42
Node selection	random
Episodes	1000
Env steps per task	5000
Evaluation episodes	20
"""

import logging
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import networkx as nx
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

# Import your existing modules
from src.curriculum.enhanced_runner import EnhancedCurriculumRunner
from src.utils.parameter_budget import ParameterBudgetCalculator
from src.utils.capacity_measurement import CapacityMeasurementManager
from src.agents.universal_topology_policy import create_universal_topology_policy
from src.tasks.rl_tasks import get_task_config

# GPU Support
try:
    from src.utils.device_manager import get_device_manager, get_device_info
    DEVICE_MANAGER = get_device_manager()
    DEVICE_INFO = get_device_info()
    GPU_SUPPORT_ENABLED = True
except ImportError as e:
    print(f"Warning: GPU support not available: {e}")
    DEVICE_MANAGER = None
    DEVICE_INFO = {'device': 'cpu', 'is_cuda': False, 'is_gpu_available': False}
    GPU_SUPPORT_ENABLED = False

class UniversalTopologyExperiment:
    """Experiment to test universal topology approach."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(f"results/universal_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.results_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the universal topology experiment."""
        print("🚀 Starting Universal Topology Experiment")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        results = {
            'config': self.config,
            'topologies': {},
            'comparison': {}
        }
        
        # Test each topology
        for topology_name in self.config['topologies']:
            print(f"\n🔬 Testing topology: {topology_name}")
            
            topology_results = self._test_topology(topology_name)
            results['topologies'][topology_name] = topology_results
        
        # Create comparison plots
        self._create_comparison_plots(results)
        
        # Save results
        with open(self.results_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Experiment complete! Results saved to: {self.results_dir}")
        return results
    
    def _test_topology(self, topology_name: str) -> Dict[str, Any]:
        """Test a specific topology with universal approach."""
        # Create environment
        env = gym.make(self.config['task_name'])
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create evaluation environment
        eval_env = gym.make(self.config['task_name'])
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # Create custom policy for this topology
        topology_params = self.config['topology_params'].get(topology_name, {})
        CustomPolicy = create_universal_topology_policy(
            topology_name=topology_name,
            universal_input_dim=self.config['universal_input_dim'],
            universal_output_dim=self.config['universal_output_dim'],
            hidden_size=self.config['hidden_size'],
            topology_params=topology_params
        )
        
        # Create PPO model with custom topology policy
        model = PPO(
            CustomPolicy,
            env,
            learning_rate=self.config['learning_rate'],
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            ent_coef=self.config['ent_coef'],
            verbose=1
        )
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.results_dir / f"{topology_name}_best"),
            log_path=str(self.results_dir / f"{topology_name}_logs"),
            eval_freq=max(self.config['total_timesteps'] // 10, 1),
            deterministic=True,
            render=False
        )
        
        # Train the model
        print(f"   Training for {self.config['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=eval_callback,
            progress_bar=True
        )
        
        # Evaluate the model
        print("   Evaluating model...")
        eval_results = self._evaluate_model(model, eval_env, self.config['eval_episodes'])
        
        # Get parameter counts
        param_counts = model.policy.get_parameter_count()
        
        # Get training info
        training_info = {
            'total_timesteps': self.config['total_timesteps'],
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'n_epochs': self.config['n_epochs']
        }
        
        return {
            'eval_results': eval_results,
            'training_info': training_info,
            'parameter_counts': param_counts,
            'model_path': str(self.results_dir / f"{topology_name}_best")
        }
    
    def _evaluate_model(self, model, env, num_episodes: int) -> Dict[str, Any]:
        """Evaluate the trained model."""
        episode_rewards = []
        episode_lengths = []
        solved_count = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward[0]  # VecEnv returns array
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # CartPole is solved if reward >= 195
            if episode_reward >= 195:
                solved_count += 1
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'solved_rate': (solved_count / num_episodes) * 100,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def _create_comparison_plots(self, results: Dict[str, Any]):
        """Create comparison plots for all topologies."""
        topologies = list(results['topologies'].keys())
        
        # 1. Final performance comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        final_rewards = [results['topologies'][t]['eval_results']['mean_reward'] for t in topologies]
        solved_rates = [results['topologies'][t]['eval_results']['solved_rate'] for t in topologies]
        
        x = np.arange(len(topologies))
        width = 0.35
        
        ax.bar(x - width/2, final_rewards, width, label='Mean Reward', alpha=0.7)
        ax.bar(x + width/2, solved_rates, width, label='Solved Rate (%)', alpha=0.7)
        
        ax.set_xlabel('Topology')
        ax.set_ylabel('Performance')
        ax.set_title('Universal Topology Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Reward distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, topology in enumerate(topologies):
            if i < len(axes):
                rewards = results['topologies'][topology]['eval_results']['episode_rewards']
                
                axes[i].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
                axes[i].axvline(np.mean(rewards), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(rewards):.1f}')
                axes[i].set_title(f'{topology} Reward Distribution')
                axes[i].set_xlabel('Reward')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "reward_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Parameter count comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        param_counts = [results['topologies'][t]['parameter_counts']['total'] for t in topologies]
        
        bars = ax.bar(topologies, param_counts, alpha=0.7)
        ax.set_xlabel('Topology')
        ax.set_ylabel('Total Parameters')
        ax.set_title('Parameter Count Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                   f'{count:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "parameter_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {self.results_dir}")

def main():
    """Run the universal topology experiment."""
    # Configuration
    config = {
        'name': 'Universal Topology Experiment',
        'description': 'Testing universal topology approach with task-specific adapters',
        
        # Task configuration
        'task_name': 'CartPole-v1',
        
        # Universal topology dimensions (maximum expected across curriculum)
        # Based on task specifications:
        # - CartPole: 4 inputs, 2 outputs
        # - Acrobot: 6 inputs, 3 outputs  
        # - Mountain Car: 2 inputs, 3 outputs
        'universal_input_dim': 6,   # Max observation space (Acrobot)
        'universal_output_dim': 3,  # Max action space (Acrobot/Mountain Car)
        
        # Task-specific dimensions (CartPole)
        'task_input_dim': 4,        # CartPole observation space
        'task_output_dim': 2,       # CartPole action space
        
        # Network configuration
        'hidden_size': 100,
        'topologies': ['small_world', 'modular', 'hybrid', 'fully_connected'],
        
        # PPO training parameters
        'total_timesteps': 50000,
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        
        # Evaluation parameters
        'eval_episodes': 100,
        
        # Topology parameters
        'topology_params': {
            'small_world': {'k': 4, 'p': 0.3, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'modular': {'num_modules': 4, 'intra_module_prob': 0.8, 'inter_module_prob': 0.1, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'hybrid': {'num_modules': 4, 'k': 4, 'p': 0.3, 'inter_module_prob': 0.1, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'fully_connected': {'intra_layer_prob': 0.8, 'inter_layer_prob': 0.5, 'num_layers': 1}
        }
    }
    
    # Run experiment
    experiment = UniversalTopologyExperiment(config)
    results = experiment.run_experiment()
    
    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("="*50)
    for topology, result in results['topologies'].items():
        eval_results = result['eval_results']
        param_counts = result['parameter_counts']
        print(f"{topology}:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Solved Rate: {eval_results['solved_rate']:.1f}%")
        print(f"  Mean Episode Length: {eval_results['mean_length']:.1f}")
        print(f"  Total Parameters: {param_counts['total']:,}")
        print()

if __name__ == "__main__":
    main() 