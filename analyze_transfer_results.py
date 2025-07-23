#!/usr/bin/env python3
"""
Transfer Learning Analysis: Comprehensive visualization of topology differences
Analyzes forward/backward transfer, parameter efficiency, and task difficulty
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TransferLearningAnalyzer:
    def __init__(self, results_file):
        """Initialize analyzer with results data."""
        self.df = pd.read_csv(results_file)
        self.tasks = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
        self.topologies = self.df['topology_type'].unique()
        self.experiment_types = self.df['experiment_type'].unique()
        
        # Separate data by experiment type
        self.same_size_df = self.df[self.df['experiment_type'] == 'same_size'].copy()
        self.match_small_world_df = self.df[self.df['experiment_type'] == 'match_small_world'].copy()
        
        # Calculate transfer metrics for each experiment type
        self.calculate_transfer_metrics()
        
    def calculate_transfer_metrics(self):
        """Calculate transfer learning metrics for each experiment type."""
        def process_dataframe(df, experiment_type):
            transfer_data = []
            
            for _, row in df.iterrows():
                train_task = row['train_task']
                train_performance = row[f'{train_task}_mean_reward']
                
                for test_task in self.tasks:
                    test_performance = row[f'{test_task}_mean_reward']
                    test_std = row[f'{test_task}_std_reward']
                    
                    # Calculate transfer ratio
                    if train_performance != 0:
                        transfer_ratio = test_performance / train_performance
                    else:
                        transfer_ratio = 0
                    
                    # Determine transfer type
                    is_forward = test_task != train_task
                    transfer_type = 'forward' if is_forward else 'same_task'
                    
                    transfer_data.append({
                        'topology_type': row['topology_type'],
                        'num_layers': row['num_layers'],
                        'train_task': train_task,
                        'test_task': test_task,
                        'train_performance': train_performance,
                        'test_performance': test_performance,
                        'test_std': test_std,
                        'transfer_ratio': transfer_ratio,
                        'transfer_type': transfer_type,
                        'is_forward': is_forward,
                        'network_size': row['network_size'],
                        'total_params': row['total_params'],
                        'training_time': row['training_time'],
                        'experiment_type': experiment_type,
                        'parameter_efficiency': row['total_params'] / row['network_size']
                    })
            
            return pd.DataFrame(transfer_data)
        
        # Process each experiment type separately
        self.same_size_transfer_df = process_dataframe(self.same_size_df, 'same_size')
        self.match_small_world_transfer_df = process_dataframe(self.match_small_world_df, 'match_small_world')
        
        # Combine for overall analysis
        self.transfer_df = pd.concat([self.same_size_transfer_df, self.match_small_world_transfer_df], ignore_index=True)
        
    def create_transfer_heatmaps(self, save_path="figures"):
        """Create transfer learning heatmaps for each experiment type."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_heatmaps_for_experiment(transfer_df, experiment_name, filename_suffix):
            # Get unique topologies
            topologies = transfer_df['topology_type'].unique()
            
            # Create subplot grid
            n_topologies = len(topologies)
            n_cols = min(2, n_topologies)
            n_rows = (n_topologies + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
            if n_topologies == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for idx, topology in enumerate(topologies):
                if idx >= len(axes):
                    break
                    
                # Filter data for this topology
                topology_data = transfer_df[transfer_df['topology_type'] == topology]
                
                # Create pivot table for heatmap
                pivot_data = topology_data.pivot_table(
                    values='transfer_ratio', 
                    index='train_task', 
                    columns='test_task', 
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(
                    pivot_data, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r', 
                    center=1.0,
                    ax=axes[idx],
                    cbar_kws={'label': 'Transfer Ratio'}
                )
                axes[idx].set_title(f'{topology.replace("_", " ").title()} - {experiment_name}')
                axes[idx].set_xlabel('Test Task')
                axes[idx].set_ylabel('Training Task')
            
            # Hide unused subplots
            for idx in range(n_topologies, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/transfer_heatmaps_{filename_suffix}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Create heatmaps for each experiment type
        create_heatmaps_for_experiment(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_heatmaps_for_experiment(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_topology_comparison(self, save_path="figures"):
        """Compare average transfer performance across topologies for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        # Function to create comparison plots for a given dataset
        def create_comparison_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Calculate average forward transfer for each topology
                forward_transfer = task_data[task_data['is_forward'] == True].groupby('topology_type').agg({
                    'transfer_ratio': ['mean', 'std'],
                    'test_performance': ['mean', 'std']
                }).round(3)
                
                if len(forward_transfer) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Transfer Ratio Comparison
                x_pos = np.arange(len(forward_transfer))
                means = forward_transfer[('transfer_ratio', 'mean')]
                stds = forward_transfer[('transfer_ratio', 'std')]
                
                bars1 = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax1.set_xlabel('Topology Type')
                ax1.set_ylabel('Average Transfer Ratio')
                ax1.set_title(f'Forward Transfer Learning by Topology\n{experiment_name} - Trained on {train_task}')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([t.replace('_', ' ').title() for t in forward_transfer.index], rotation=45)
                ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer (1.0)')
                ax1.legend()
                
                # Add value labels on bars
                for bar, mean in zip(bars1, means):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{mean:.3f}', ha='center', va='bottom')
                
                # Plot 2: Raw Performance Comparison
                means2 = forward_transfer[('test_performance', 'mean')]
                stds2 = forward_transfer[('test_performance', 'std')]
                
                bars2 = ax2.bar(x_pos, means2, yerr=stds2, capsize=5, alpha=0.7, color='orange')
                ax2.set_xlabel('Topology Type')
                ax2.set_ylabel('Average Test Performance')
                ax2.set_title(f'Raw Transfer Performance by Topology\n{experiment_name} - Trained on {train_task}')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([t.replace('_', ' ').title() for t in forward_transfer.index], rotation=45)
                
                # Add value labels on bars
                for bar, mean in zip(bars2, means2):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{mean:.1f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/topology_comparison_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create comparisons for each experiment type
        create_comparison_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_comparison_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_parameter_efficiency_analysis(self, save_path="figures"):
        """Analyze parameter efficiency vs transfer performance for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_efficiency_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Filter for forward transfer only
                forward_data = task_data[task_data['is_forward'] == True]
                
                if len(forward_data) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                
                topologies = forward_data['topology_type'].unique()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot: Parameter Efficiency vs Transfer Ratio
                for topology in topologies:
                    topology_data = forward_data[forward_data['topology_type'] == topology]
                    ax.scatter(
                        topology_data['parameter_efficiency'], 
                        topology_data['transfer_ratio'],
                        label=topology.replace('_', ' ').title(),
                        alpha=0.7,
                        s=50
                    )
                
                ax.set_xlabel('Parameters per Hidden Unit')
                ax.set_ylabel('Transfer Ratio')
                ax.set_title(f'Parameter Efficiency vs Transfer Learning\n{experiment_name} - Trained on {train_task}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/parameter_efficiency_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create efficiency plots for each experiment type
        create_efficiency_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_efficiency_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_task_difficulty_analysis(self, save_path="figures"):
        """Analyze task difficulty and performance patterns for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_task_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Calculate average performance for each task across all topologies
                task_performance = task_data.groupby(['test_task', 'topology_type']).agg({
                    'test_performance': 'mean',
                    'transfer_ratio': 'mean'
                }).reset_index()
                
                if len(task_performance) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                
                topologies = task_performance['topology_type'].unique()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Task Performance by Topology
                for topology in topologies:
                    topology_data = task_performance[task_performance['topology_type'] == topology]
                    ax1.plot(topology_data['test_task'], topology_data['test_performance'], 
                            marker='o', label=topology.replace('_', ' ').title(), linewidth=2, markersize=8)
                
                ax1.set_xlabel('Test Task')
                ax1.set_ylabel('Average Performance')
                ax1.set_title(f'Task Performance Across Topologies\n{experiment_name} - Trained on {train_task}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                plt.setp(ax1.get_xticklabels(), rotation=45)
                
                # Plot 2: Transfer Ratio by Task
                for topology in topologies:
                    topology_data = task_performance[task_performance['topology_type'] == topology]
                    ax2.plot(topology_data['test_task'], topology_data['transfer_ratio'], 
                            marker='s', label=topology.replace('_', ' ').title(), linewidth=2, markersize=8)
                
                ax2.set_xlabel('Test Task')
                ax2.set_ylabel('Average Transfer Ratio')
                ax2.set_title(f'Transfer Learning by Task\n{experiment_name} - Trained on {train_task}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer')
                plt.setp(ax2.get_xticklabels(), rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/task_difficulty_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create task plots for each experiment type
        create_task_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_task_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_forward_backward_analysis(self, save_path="figures"):
        """Compare forward vs backward transfer learning for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_forward_backward_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Calculate forward and backward transfer for each topology
                forward_transfer = task_data[task_data['is_forward'] == True].groupby('topology_type')['transfer_ratio'].mean()
                
                if len(forward_transfer) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                
                topologies = forward_transfer.index
                
                # For backward transfer, we need to calculate how well models perform when tested on their training task
                # vs when other models are trained on that task
                backward_data = []
                for topology in topologies:
                    # Get performance when this topology trains on this task
                    same_task_data = task_data[
                        (task_data['topology_type'] == topology) & 
                        (task_data['train_task'] == train_task) & 
                        (task_data['test_task'] == train_task)
                    ]
                    
                    if len(same_task_data) == 0:
                        continue
                        
                    same_task_perf = same_task_data['test_performance'].iloc[0]
                    
                    # Get average performance when other topologies train on this task
                    other_data = task_data[
                        (task_data['topology_type'] != topology) & 
                        (task_data['train_task'] == train_task) & 
                        (task_data['test_task'] == train_task)
                    ]
                    
                    if len(other_data) == 0:
                        continue
                        
                    other_perf = other_data['test_performance'].mean()
                    
                    if other_perf != 0:
                        backward_ratio = same_task_perf / other_perf
                    else:
                        backward_ratio = 1.0
                        
                    backward_data.append({
                        'topology_type': topology,
                        'train_task': train_task,
                        'backward_ratio': backward_ratio
                    })
                
                if len(backward_data) == 0:
                    print(f"No backward transfer data for {experiment_name} - {train_task}")
                    continue
                    
                backward_df = pd.DataFrame(backward_data)
                backward_transfer = backward_df.groupby('topology_type')['backward_ratio'].mean()
                
                # Create comparison plot
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x_pos = np.arange(len(topologies))
                width = 0.35
                
                bars1 = ax.bar(x_pos - width/2, forward_transfer, width, label='Forward Transfer', alpha=0.7)
                bars2 = ax.bar(x_pos + width/2, backward_transfer, width, label='Backward Transfer', alpha=0.7)
                
                ax.set_xlabel('Topology Type')
                ax.set_ylabel('Transfer Ratio')
                ax.set_title(f'Forward vs Backward Transfer Learning\n{experiment_name} - Trained on {train_task}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels([t.replace('_', ' ').title() for t in topologies], rotation=45)
                ax.legend()
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/forward_backward_comparison_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create forward/backward plots for each experiment type
        create_forward_backward_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_forward_backward_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_training_efficiency_analysis(self, save_path="figures"):
        """Analyze training efficiency vs transfer performance for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_training_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Filter for forward transfer only
                forward_data = task_data[task_data['is_forward'] == True]
                
                if len(forward_data) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                
                topologies = forward_data['topology_type'].unique()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot: Timesteps per Second vs Transfer Ratio
                # Calculate timesteps per second (assuming 1000 timesteps from config)
                for topology in topologies:
                    topology_data = forward_data[forward_data['topology_type'] == topology]
                    topology_tps = 1000 / topology_data['training_time']
                    ax.scatter(
                        topology_tps, 
                        topology_data['transfer_ratio'],
                        label=topology.replace('_', ' ').title(),
                        alpha=0.7,
                        s=50
                    )
                
                ax.set_xlabel('Training Speed (timesteps/second)')
                ax.set_ylabel('Transfer Ratio')
                ax.set_title(f'Training Speed vs Transfer Learning\n{experiment_name} - Trained on {train_task}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/training_efficiency_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create training plots for each experiment type
        create_training_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_training_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_summary_statistics(self, save_path="figures"):
        """Create summary statistics table for each experiment type and training task."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_summary_table(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                # Filter data for this training task
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Calculate comprehensive statistics
                summary_stats = []
                
                topologies = task_data['topology_type'].unique()
                
                for topology in topologies:
                    topology_data = task_data[task_data['topology_type'] == topology]
                    forward_data = topology_data[topology_data['is_forward'] == True]
                    
                    if len(forward_data) == 0:
                        continue
                    
                    # Find best and worst transfer tasks
                    best_task = forward_data.loc[forward_data['transfer_ratio'].idxmax(), 'test_task']
                    worst_task = forward_data.loc[forward_data['transfer_ratio'].idxmin(), 'test_task']
                    
                    stats = {
                        'Topology': topology.replace('_', ' ').title(),
                        'Avg Forward Transfer': forward_data['transfer_ratio'].mean(),
                        'Std Forward Transfer': forward_data['transfer_ratio'].std(),
                        'Best Transfer Task': best_task,
                        'Worst Transfer Task': worst_task,
                        'Avg Parameters': topology_data['total_params'].mean(),
                        'Avg Training Time': topology_data['training_time'].mean(),
                        'Parameter Efficiency': topology_data['parameter_efficiency'].mean()
                    }
                    summary_stats.append(stats)
                
                if len(summary_stats) == 0:
                    print(f"No data for {experiment_name} - {train_task}")
                    continue
                    
                summary_df = pd.DataFrame(summary_stats)
                
                # Create a nice table visualization
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.axis('tight')
                ax.axis('off')
                
                table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                
                # Color the header
                for i in range(len(summary_df.columns)):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                plt.title(f'Transfer Learning Summary Statistics\n{experiment_name} - Trained on {train_task}', fontsize=16, pad=20)
                plt.savefig(f'{save_path}/summary_statistics_{filename_suffix}_{train_task.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                return summary_df
        
        # Create summary tables for each experiment type
        same_size_summary = create_summary_table(self.same_size_transfer_df, "Same Size Networks", "same_size")
        matched_capacity_summary = create_summary_table(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
        # Return combined summary for overall analysis
        if same_size_summary is not None and matched_capacity_summary is not None:
            return pd.concat([same_size_summary, matched_capacity_summary], ignore_index=True)
        elif same_size_summary is not None:
            return same_size_summary
        elif matched_capacity_summary is not None:
            return matched_capacity_summary
        else:
            return pd.DataFrame()
        
    def generate_all_analyses(self, save_path="figures"):
        """Generate all analysis figures for each experiment type."""
        print("🎨 Generating comprehensive transfer learning analysis...")
        print(f"📊 Analyzing {len(self.same_size_df)} same_size experiments and {len(self.match_small_world_df)} matched capacity experiments")
        
        self.create_transfer_heatmaps(save_path)
        self.create_topology_comparison(save_path)
        self.create_parameter_efficiency_analysis(save_path)
        self.create_task_difficulty_analysis(save_path)
        self.create_forward_backward_analysis(save_path)
        self.create_training_efficiency_analysis(save_path)
        summary_df = self.create_summary_statistics(save_path)
        
        print(f"✅ All analyses saved to {save_path}/")
        print(f"📊 Generated separate analyses for same_size and matched_capacity experiments")
        
        return summary_df

def main():
    """Main analysis function."""
    # Find the most recent results file
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No results directory found!")
        return
    
    # Find the most recent results
    result_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not result_dirs:
        print("❌ No result directories found!")
        return
    
    latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
    results_file = latest_dir / "cross_task_results.csv"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Analyzing results from: {latest_dir}")
    
    # Create analyzer and generate all analyses
    analyzer = TransferLearningAnalyzer(results_file)
    summary_df = analyzer.generate_all_analyses()
    
    # Print key insights
    print("\n🔍 Key Insights:")
    print("=" * 50)
    
    # Best forward transfer topology
    best_forward = summary_df.loc[summary_df['Avg Forward Transfer'].idxmax()]
    print(f"🏆 Best Forward Transfer: {best_forward['Topology']} ({best_forward['Avg Forward Transfer']:.3f})")
    
    # Most parameter efficient
    best_efficiency = summary_df.loc[summary_df['Parameter Efficiency'].idxmax()]
    print(f"⚡ Most Parameter Efficient: {best_efficiency['Topology']} ({best_efficiency['Parameter Efficiency']:.1f} params/unit)")
    
    # Fastest training
    fastest = summary_df.loc[summary_df['Avg Training Time'].idxmin()]
    print(f"🚀 Fastest Training: {fastest['Topology']} ({fastest['Avg Training Time']:.2f}s)")
    
    print(f"\n📈 Analysis complete! Check the 'figures' directory for all visualizations.")

if __name__ == "__main__":
    main() 