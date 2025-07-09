"""
CONTROLLED UNIVERSAL TOPOLOGY EXPERIMENT: Maximizing Topology Transfer Insight
Objective: Test topology transfer with minimal adapter influence and comprehensive analysis.

Control Measures Implemented:
✅ Minimal adapters (linear or tiny MLPs)
✅ Gradient norm tracking (topology vs adapter contribution)
✅ Ablation studies (frozen adapters/topology)
✅ Per-task performance tracking across phases
✅ Linear probe evaluation on topology outputs
✅ Consistent architecture across experiments

Transfer Learning Phases:
1. Task A Training: Full training (topology + adapters)
2. Task B Training: Frozen topology (adapters only)
3. Task C Training: Frozen adapters (topology only)
4. Backward Transfer: Return to Task A with frozen topology

This design ensures topology influence is maximized and adapter influence is minimized.
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
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import your existing modules
from src.agents.universal_topology_policy import create_universal_topology_policy

class GradientTrackingCallback(BaseCallback):
    """Callback to track gradient norms during training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.gradient_history = []
    
    def _on_step(self) -> bool:
        """Track gradients after each step."""
        if self.model.policy.training:
            gradient_analysis = self.model.policy.get_gradient_analysis()
            self.gradient_history.append(gradient_analysis)
        return True

class LinearProbeEvaluator:
    """Linear probe to evaluate topology feature quality."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.probe = LogisticRegression(max_iter=1000, random_state=42)
        self.is_trained = False
    
    def train_probe(self, features: np.ndarray, labels: np.ndarray):
        """Train linear probe on topology features."""
        self.probe.fit(features, labels)
        self.is_trained = True
    
    def evaluate_probe(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate probe performance."""
        if not self.is_trained:
            return {'accuracy': 0.0, 'error': 'Probe not trained'}
        
        predictions = self.probe.predict(features)
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'feature_dim': self.feature_dim,
            'num_classes': self.num_classes
        }

class ControlledUniversalTopologyExperiment:
    """Controlled experiment to maximize topology transfer insight."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(f"results/controlled_universal_topology_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.results_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize probe evaluators
        self.probe_evaluators = {}
        
        # Results storage
        self.results = {
            'config': config,
            'phases': {},
            'ablation_studies': {},
            'gradient_analysis': {},
            'probe_evaluations': {},
            'transfer_metrics': {}
        }
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the controlled universal topology experiment."""
        print("🔬 Starting Controlled Universal Topology Experiment")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        # Phase 1: Task A Training (Full training)
        print("\n🔄 PHASE 1: Task A Training (Full Training)")
        task_a_results = self._run_phase(
            task_name=self.config['task_sequence'][0],
            phase_name='task_a_full',
            freeze_topology=False,
            freeze_adapters=False,
            freeze_output_adapters=False
        )
        self.results['phases']['task_a_full'] = task_a_results
        
        # Phase 2: Task B Training (Frozen Topology)
        print("\n🔄 PHASE 2: Task B Training (Frozen Topology)")
        task_b_results = self._run_phase(
            task_name=self.config['task_sequence'][1],
            phase_name='task_b_frozen_topology',
            freeze_topology=True,
            freeze_adapters=False,
            freeze_output_adapters=False,
            load_model_path=task_a_results['model_path']
        )
        self.results['phases']['task_b_frozen_topology'] = task_b_results
        
        # Phase 3: Task C Training (Frozen Adapters)
        print("\n🔄 PHASE 3: Task C Training (Frozen Adapters)")
        task_c_results = self._run_phase(
            task_name=self.config['task_sequence'][2],
            phase_name='task_c_frozen_adapters',
            freeze_topology=False,
            freeze_adapters=True,
            freeze_output_adapters=True,
            load_model_path=task_b_results['model_path']
        )
        self.results['phases']['task_c_frozen_adapters'] = task_c_results
        
        # Phase 4: Backward Transfer (Return to Task A with Frozen Topology)
        print("\n🔄 PHASE 4: Backward Transfer (Task A with Frozen Topology)")
        backward_results = self._run_phase(
            task_name=self.config['task_sequence'][0],
            phase_name='task_a_backward_transfer',
            freeze_topology=True,
            freeze_adapters=False,
            freeze_output_adapters=False,
            load_model_path=task_c_results['model_path']
        )
        self.results['phases']['task_a_backward_transfer'] = backward_results
        
        # Run ablation studies
        print("\n🔬 Running Ablation Studies")
        self._run_ablation_studies()
        
        # Create comprehensive analysis
        print("\n📊 Creating Analysis")
        self._create_comprehensive_analysis()
        
        # Save results
        with open(self.results_dir / "results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Experiment complete! Results saved to: {self.results_dir}")
        return self.results
    
    def _run_phase(self, task_name: str, phase_name: str, 
                   freeze_topology: bool = False, freeze_adapters: bool = False,
                   freeze_output_adapters: bool = False, load_model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run a single phase of the experiment."""
        print(f"   Task: {task_name}")
        print(f"   Freeze Topology: {freeze_topology}")
        print(f"   Freeze Adapters: {freeze_adapters}")
        print(f"   Freeze Output Adapters: {freeze_output_adapters}")
        
        # Create environment
        env = gym.make(task_name)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create evaluation environment
        eval_env = gym.make(task_name)
        eval_env = DummyVecEnv([lambda: eval_env])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # Create custom policy for this topology
        topology_params = self.config['topology_params'].get(self.config['topology_name'], {})
        CustomPolicy = create_universal_topology_policy(
            topology_name=self.config['topology_name'],
            universal_input_dim=self.config['universal_input_dim'],
            universal_output_dim=self.config['universal_output_dim'],
            hidden_size=self.config['hidden_size'],
            topology_params=topology_params,
            adapter_type=self.config['adapter_type'],
            adapter_hidden_dim=self.config['adapter_hidden_dim'],
            freeze_adapters=freeze_adapters,
            freeze_output_adapters=freeze_output_adapters
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
        
        # Load previous model if specified
        if load_model_path and Path(load_model_path).exists():
            print(f"   Loading model from: {load_model_path}")
            model = PPO.load(load_model_path, env=env)
            
            # Apply freezing after loading
            if freeze_topology:
                model.policy.features_extractor.freeze_topology_weights()
                print("   ✅ Topology weights frozen")
            
            if freeze_adapters:
                model.policy.features_extractor.freeze_adapter_weights()
                print("   ✅ Input adapter weights frozen")
            
            if freeze_output_adapters:
                model.policy.freeze_output_adapter_weights()
                print("   ✅ Output adapter weights frozen")
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.results_dir / f"{phase_name}_best"),
            log_path=str(self.results_dir / f"{phase_name}_logs"),
            eval_freq=max(self.config['total_timesteps'] // 10, 1),
            deterministic=True,
            render=False
        )
        
        # Create gradient tracking callback
        gradient_callback = GradientTrackingCallback()
        
        # Train the model
        print(f"   Training for {self.config['total_timesteps']} timesteps...")
        model.learn(
            total_timesteps=self.config['total_timesteps'],
            callback=[eval_callback, gradient_callback],
            progress_bar=True
        )
        
        # Evaluate the model
        print("   Evaluating model...")
        eval_results = self._evaluate_model(model, eval_env, self.config['eval_episodes'])
        
        # Get parameter counts
        param_counts = model.policy.get_parameter_count()
        
        # Get gradient analysis
        gradient_analysis = model.policy.get_gradient_analysis()
        
        # Evaluate linear probe on topology features
        probe_results = self._evaluate_linear_probe(model, eval_env, task_name, phase_name)
        
        # Get training info
        training_info = {
            'total_timesteps': self.config['total_timesteps'],
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'n_epochs': self.config['n_epochs'],
            'freeze_topology': freeze_topology,
            'freeze_adapters': freeze_adapters,
            'freeze_output_adapters': freeze_output_adapters
        }
        
        return {
            'task_name': task_name,
            'eval_results': eval_results,
            'training_info': training_info,
            'parameter_counts': param_counts,
            'gradient_analysis': gradient_analysis,
            'probe_results': probe_results,
            'gradient_history': gradient_callback.gradient_history,
            'model_path': str(self.results_dir / f"{phase_name}_best")
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
            
            # Task-specific success criteria
            if episode_reward >= self.config['success_threshold']:
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
    
    def _evaluate_linear_probe(self, model, env, task_name: str, phase_name: str) -> Dict[str, Any]:
        """Evaluate linear probe on topology features."""
        print(f"   Evaluating linear probe for {task_name}...")
        
        # Collect features and labels
        features_list = []
        labels_list = []
        
        for episode in range(self.config['probe_eval_episodes']):
            obs = env.reset()
            done = False
            
            while not done:
                # Get topology features
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).to(model.device)
                    features = model.policy.features_extractor(obs_tensor)
                    features_list.append(features.cpu().numpy())
                
                # Get action as label
                action, _ = model.predict(obs, deterministic=True)
                labels_list.append(action[0])
                
                obs, reward, done, _ = env.step(action)
        
        # Convert to arrays
        features_array = np.vstack(features_list)
        labels_array = np.array(labels_list)
        
        # Create or get probe evaluator
        if task_name not in self.probe_evaluators:
            self.probe_evaluators[task_name] = LinearProbeEvaluator(
                feature_dim=features_array.shape[1],
                num_classes=len(np.unique(labels_array))
            )
        
        probe = self.probe_evaluators[task_name]
        
        # Train and evaluate probe
        probe.train_probe(features_array, labels_array)
        probe_results = probe.evaluate_probe(features_array, labels_array)
        
        return {
            'task_name': task_name,
            'phase_name': phase_name,
            'probe_accuracy': probe_results['accuracy'],
            'feature_dim': probe_results['feature_dim'],
            'num_classes': probe_results['num_classes'],
            'num_samples': len(features_array)
        }
    
    def _run_ablation_studies(self):
        """Run ablation studies to isolate topology vs adapter contributions."""
        print("   Running ablation studies...")
        
        ablation_results = {}
        
        # Test each topology with different adapter configurations
        for adapter_type in ['linear', 'tiny_mlp', 'identity']:
            print(f"     Testing adapter type: {adapter_type}")
            
            # Create environment
            env = gym.make(self.config['task_sequence'][0])
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            
            # Create policy with different adapter type
            topology_params = self.config['topology_params'].get(self.config['topology_name'], {})
            CustomPolicy = create_universal_topology_policy(
                topology_name=self.config['topology_name'],
                universal_input_dim=self.config['universal_input_dim'],
                universal_output_dim=self.config['universal_output_dim'],
                hidden_size=self.config['hidden_size'],
                topology_params=topology_params,
                adapter_type=adapter_type,
                adapter_hidden_dim=self.config['adapter_hidden_dim'],
                freeze_adapters=False,
                freeze_output_adapters=False
            )
            
            # Create and train model
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
                verbose=0
            )
            
            # Train for shorter time for ablation
            model.learn(total_timesteps=self.config['ablation_timesteps'], progress_bar=False)
            
            # Evaluate
            eval_results = self._evaluate_model(model, env, self.config['eval_episodes'])
            param_counts = model.policy.get_parameter_count()
            gradient_analysis = model.policy.get_gradient_analysis()
            
            ablation_results[adapter_type] = {
                'eval_results': eval_results,
                'parameter_counts': param_counts,
                'gradient_analysis': gradient_analysis
            }
        
        self.results['ablation_studies'] = ablation_results
    
    def _create_comprehensive_analysis(self):
        """Create comprehensive analysis plots and metrics."""
        print("   Creating comprehensive analysis...")
        
        # 1. Transfer Learning Performance Comparison
        self._plot_transfer_performance()
        
        # 2. Gradient Analysis Over Time
        self._plot_gradient_analysis()
        
        # 3. Ablation Study Results
        self._plot_ablation_results()
        
        # 4. Linear Probe Performance
        self._plot_probe_performance()
        
        # 5. Parameter Efficiency Analysis
        self._plot_parameter_efficiency()
        
        print("   ✅ Analysis plots created")
    
    def _plot_transfer_performance(self):
        """Plot transfer learning performance across phases."""
        phases = list(self.results['phases'].keys())
        rewards = [self.results['phases'][p]['eval_results']['mean_reward'] for p in phases]
        solved_rates = [self.results['phases'][p]['eval_results']['solved_rate'] for p in phases]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reward comparison
        bars1 = ax1.bar(phases, rewards, alpha=0.7, color='skyblue')
        ax1.set_title('Transfer Learning: Mean Reward')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, reward in zip(bars1, rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # Solved rate comparison
        bars2 = ax2.bar(phases, solved_rates, alpha=0.7, color='lightgreen')
        ax2.set_title('Transfer Learning: Solved Rate')
        ax2.set_ylabel('Solved Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rate in zip(bars2, solved_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "transfer_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_analysis(self):
        """Plot gradient analysis over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        phases = list(self.results['phases'].keys())
        
        for i, phase in enumerate(phases):
            if i < len(axes):
                gradient_history = self.results['phases'][phase]['gradient_history']
                
                if gradient_history:
                    steps = range(len(gradient_history))
                    topology_norms = [g.get('topology_norm', 0) for g in gradient_history]
                    adapter_norms = [g.get('input_adapter_norm', 0) for g in gradient_history]
                    
                    axes[i].plot(steps, topology_norms, label='Topology', linewidth=2)
                    axes[i].plot(steps, adapter_norms, label='Input Adapter', linewidth=2)
                    axes[i].set_title(f'{phase}: Gradient Norms')
                    axes[i].set_xlabel('Training Steps')
                    axes[i].set_ylabel('Gradient Norm')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "gradient_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_results(self):
        """Plot ablation study results."""
        if not self.results['ablation_studies']:
            return
        
        adapter_types = list(self.results['ablation_studies'].keys())
        rewards = [self.results['ablation_studies'][a]['eval_results']['mean_reward'] for a in adapter_types]
        topology_ratios = [self.results['ablation_studies'][a]['gradient_analysis']['topology_ratio'] for a in adapter_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        bars1 = ax1.bar(adapter_types, rewards, alpha=0.7, color='orange')
        ax1.set_title('Ablation Study: Performance by Adapter Type')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, reward in zip(bars1, rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{reward:.1f}', ha='center', va='bottom')
        
        # Topology contribution comparison
        bars2 = ax2.bar(adapter_types, topology_ratios, alpha=0.7, color='purple')
        ax2.set_title('Ablation Study: Topology Contribution Ratio')
        ax2.set_ylabel('Topology Gradient Ratio')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, ratio in zip(bars2, topology_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{ratio:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "ablation_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_probe_performance(self):
        """Plot linear probe performance."""
        phases = list(self.results['phases'].keys())
        probe_accuracies = [self.results['phases'][p]['probe_results']['probe_accuracy'] for p in phases]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(phases, probe_accuracies, alpha=0.7, color='red')
        ax.set_title('Linear Probe Performance on Topology Features')
        ax.set_ylabel('Probe Accuracy')
        ax.set_xlabel('Training Phase')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, probe_accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "probe_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_efficiency(self):
        """Plot parameter efficiency analysis."""
        phases = list(self.results['phases'].keys())
        total_params = [self.results['phases'][p]['parameter_counts']['total'] for p in phases]
        topology_params = [self.results['phases'][p]['parameter_counts']['topology'] for p in phases]
        adapter_params = [self.results['phases'][p]['parameter_counts']['input_adapter'] + 
                         self.results['phases'][p]['parameter_counts']['action_adapter'] + 
                         self.results['phases'][p]['parameter_counts']['value_adapter'] for p in phases]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(phases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, topology_params, width, label='Topology Parameters', alpha=0.7)
        bars2 = ax.bar(x + width/2, adapter_params, width, label='Adapter Parameters', alpha=0.7)
        
        ax.set_title('Parameter Distribution: Topology vs Adapters')
        ax.set_ylabel('Number of Parameters')
        ax.set_xlabel('Training Phase')
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "parameter_efficiency.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Run the controlled universal topology experiment."""
    # Configuration
    config = {
        'name': 'Controlled Universal Topology Experiment',
        'description': 'Maximizing topology transfer insight with minimal adapter influence',
        
        # Task sequence for transfer learning
        'task_sequence': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'],
        
        # Universal topology dimensions (maximum expected across curriculum)
        'universal_input_dim': 6,   # Max observation space (Acrobot)
        'universal_output_dim': 3,  # Max action space (Acrobot/Mountain Car)
        
        # Network configuration
        'topology_name': 'small_world',
        'hidden_size': 100,
        
        # Minimal adapter configuration
        'adapter_type': 'linear',  # 'linear', 'tiny_mlp', 'identity'
        'adapter_hidden_dim': 8,   # For tiny_mlp adapters
        
        # PPO training parameters
        'total_timesteps': 50000,
        'ablation_timesteps': 10000,  # Shorter for ablation studies
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
        'probe_eval_episodes': 50,
        'success_threshold': 195,  # CartPole success threshold
        
        # Topology parameters
        'topology_params': {
            'small_world': {'k': 4, 'p': 0.3, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'modular': {'num_modules': 4, 'intra_module_prob': 0.8, 'inter_module_prob': 0.1, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'hybrid': {'num_modules': 4, 'k': 4, 'p': 0.3, 'inter_module_prob': 0.1, 'num_layers': 1, 'inter_layer_prob': 0.1},
            'fully_connected': {'intra_layer_prob': 0.8, 'inter_layer_prob': 0.5, 'num_layers': 1}
        }
    }
    
    # Run experiment
    experiment = ControlledUniversalTopologyExperiment(config)
    results = experiment.run_experiment()
    
    # Print comprehensive summary
    print("\n📊 CONTROLLED EXPERIMENT SUMMARY")
    print("="*60)
    
    for phase, result in results['phases'].items():
        eval_results = result['eval_results']
        param_counts = result['parameter_counts']
        gradient_analysis = result['gradient_analysis']
        probe_results = result['probe_results']
        
        print(f"\n{phase.upper()}:")
        print(f"  Task: {result['task_name']}")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Solved Rate: {eval_results['solved_rate']:.1f}%")
        print(f"  Topology Parameters: {param_counts['topology']:,}")
        print(f"  Adapter Parameters: {param_counts['input_adapter'] + param_counts['action_adapter'] + param_counts['value_adapter']:,}")
        print(f"  Topology Gradient Ratio: {gradient_analysis['topology_ratio']:.3f}")
        print(f"  Linear Probe Accuracy: {probe_results['probe_accuracy']:.3f}")
    
    # Print ablation results
    if results['ablation_studies']:
        print(f"\n🔬 ABLATION STUDY RESULTS:")
        print("="*40)
        for adapter_type, ablation_result in results['ablation_studies'].items():
            eval_results = ablation_result['eval_results']
            gradient_analysis = ablation_result['gradient_analysis']
            print(f"  {adapter_type}:")
            print(f"    Mean Reward: {eval_results['mean_reward']:.2f}")
            print(f"    Topology Ratio: {gradient_analysis['topology_ratio']:.3f}")

if __name__ == "__main__":
    main() 