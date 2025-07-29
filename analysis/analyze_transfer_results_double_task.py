#!/usr/bin/env python3
"""
Double-Task Transfer Learning Analysis: Comprehensive visualization of sequential training effects
Analyzes forward/backward transfer, forgetting, and sequential training performance
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

class DoubleTaskTransferLearningAnalyzer:
    def __init__(self, results_file):
        """Initialize analyzer with double-task results data."""
        self.df = pd.read_csv(results_file)
        self.tasks = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1']
        self.topologies = self.df['topology_type'].unique()
        self.experiment_types = self.df['experiment_type'].unique()
        
        # Separate data by experiment type
        self.same_size_df = self.df[self.df['experiment_type'] == 'same_size'].copy()
        self.match_small_world_df = self.df[self.df['experiment_type'] == 'match_small_world'].copy()
        
        # For matched capacity analysis, we need to include small_world from same_size
        # since other topologies were matched to its parameter count (~364)
        small_world_same_size = self.same_size_df[self.same_size_df['topology_type'] == 'small_world'].copy()
        self.match_small_world_df = pd.concat([self.match_small_world_df, small_world_same_size], ignore_index=True)
        
        # Calculate transfer metrics for each experiment type
        self.calculate_transfer_metrics()
        
    def calculate_transfer_metrics(self):
        """Calculate transfer learning metrics for double-task training."""
        def process_dataframe(df, experiment_type):
            transfer_data = []
            
            for _, row in df.iterrows():
                train_task_1 = row['train_task_1']
                train_task_2 = row['train_task_2']
                
                # Get performance after training on both tasks
                task1_performance_after_task1 = row.get(f'{train_task_1}_after_task1_mean_reward', 0)
                task1_performance_final = row.get(f'{train_task_1}_final_mean_reward', 0)
                task2_performance_final = row.get(f'{train_task_2}_final_mean_reward', 0)
                
                # Calculate forgetting on task 1
                forgetting_task1 = row.get(f'{train_task_1}_forgetting', 0)
                
                for test_task in self.tasks:
                    test_performance = row.get(f'{test_task}_final_mean_reward', 0)
                    test_std = row.get(f'{test_task}_final_std_reward', 0)
                    test_success = row.get(f'{test_task}_final_success_rate', 0)
                    test_length = row.get(f'{test_task}_final_mean_length', 0)
                    
                    # Determine transfer type
                    is_training_task_1 = test_task == train_task_1
                    is_training_task_2 = test_task == train_task_2
                    is_third_task = not (is_training_task_1 or is_training_task_2)
                    
                    # Calculate transfer ratios
                    if task1_performance_after_task1 != 0:
                        transfer_ratio_task1 = test_performance / task1_performance_after_task1
                    else:
                        transfer_ratio_task1 = 0
                        
                    if task2_performance_final != 0:
                        transfer_ratio_task2 = test_performance / task2_performance_final
                    else:
                        transfer_ratio_task2 = 0
                    
                    # Create a unique topology identifier that includes layer count for fully connected
                    topology_id = row['topology_type']
                    if row['topology_type'] == 'fully_connected':
                        topology_id = f"fully_connected_{row['num_layers']}layers"
                    
                    transfer_data.append({
                        'topology_type': row['topology_type'],
                        'topology_id': topology_id,  # Unique identifier including layer count
                        'num_layers': row['num_layers'],
                        'train_task_1': train_task_1,
                        'train_task_2': train_task_2,
                        'test_task': test_task,
                        'task1_performance_after_task1': task1_performance_after_task1,
                        'task1_performance_final': task1_performance_final,
                        'task2_performance_final': task2_performance_final,
                        'test_performance': test_performance,
                        'test_std': test_std,
                        'test_success': test_success,
                        'test_length': test_length,
                        'transfer_ratio_task1': transfer_ratio_task1,
                        'transfer_ratio_task2': transfer_ratio_task2,
                        'is_training_task_1': is_training_task_1,
                        'is_training_task_2': is_training_task_2,
                        'is_third_task': is_third_task,
                        'forgetting_task1': forgetting_task1,
                        'network_size': row['network_size'],
                        'total_params': row['total_params'],
                        'training_time': row['total_training_time'],
                        'experiment_type': experiment_type,
                        'parameter_efficiency': row['total_params'] / row['network_size']
                    })
            
            return pd.DataFrame(transfer_data)
        
        # Process each experiment type separately
        self.same_size_transfer_df = process_dataframe(self.same_size_df, 'same_size')
        self.match_small_world_transfer_df = process_dataframe(self.match_small_world_df, 'match_small_world')
        
        # Combine for overall analysis
        self.transfer_df = pd.concat([self.same_size_transfer_df, self.match_small_world_transfer_df], ignore_index=True)
        
    def create_forgetting_analysis(self, save_path="figures"):
        """Analyze forgetting patterns for each experiment type and training task combination."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_forgetting_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training task combinations
            task_combinations = transfer_df[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                # Filter data for this training combination
                combo_data = transfer_df[
                    (transfer_df['train_task_1'] == train_task_1) & 
                    (transfer_df['train_task_2'] == train_task_2)
                ]
                
                if len(combo_data) == 0:
                    continue
                
                # Calculate forgetting metrics
                forgetting_data = combo_data.groupby('topology_id').agg({
                    'forgetting_task1': 'mean',
                    'task1_performance_after_task1': 'mean',
                    'task1_performance_final': 'mean',
                    'task2_performance_final': 'mean'
                }).reset_index()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Forgetting on Task 1
                x_pos = np.arange(len(forgetting_data))
                forgetting_values = forgetting_data['forgetting_task1']
                
                bars1 = ax1.bar(x_pos, forgetting_values, alpha=0.7, color='red')
                ax1.set_xlabel('Topology Type')
                ax1.set_ylabel('Forgetting on Task 1')
                ax1.set_title(f'Catastrophic Forgetting Analysis\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([t.replace('_', ' ').title() for t in forgetting_data['topology_id']], rotation=45)
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars1, forgetting_values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{value:.2f}', ha='center', va='bottom')
                
                # Plot 2: Performance Comparison
                task1_after = forgetting_data['task1_performance_after_task1']
                task1_final = forgetting_data['task1_performance_final']
                task2_final = forgetting_data['task2_performance_final']
                
                x_pos = np.arange(len(forgetting_data))
                width = 0.25
                
                bars2 = ax2.bar(x_pos - width, task1_after, width, label=f'{train_task_1} after Task 1', alpha=0.7)
                bars3 = ax2.bar(x_pos, task1_final, width, label=f'{train_task_1} after Task 2', alpha=0.7)
                bars4 = ax2.bar(x_pos + width, task2_final, width, label=f'{train_task_2} after Task 2', alpha=0.7)
                
                ax2.set_xlabel('Topology Type')
                ax2.set_ylabel('Performance')
                ax2.set_title(f'Performance Evolution\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([t.replace('_', ' ').title() for t in forgetting_data['topology_id']], rotation=45)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/forgetting_analysis_{filename_suffix}_{train_task_1.replace("-", "_")}_{train_task_2.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create forgetting plots for each experiment type
        create_forgetting_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_forgetting_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_sequential_transfer_heatmaps(self, save_path="figures"):
        """Create transfer learning heatmaps showing performance after sequential training."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_heatmaps_for_experiment(transfer_df, experiment_name, filename_suffix):
            # Get unique topologies (including layer variants)
            topologies = transfer_df['topology_id'].unique()
            
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
                topology_data = transfer_df[transfer_df['topology_id'] == topology]
                
                # Create pivot table for heatmap (average performance across all training combinations)
                pivot_data = topology_data.pivot_table(
                    values='test_performance', 
                    index='train_task_1', 
                    columns='test_task', 
                    aggfunc='mean'
                )
                
                # Create heatmap
                sns.heatmap(
                    pivot_data, 
                    annot=True, 
                    fmt='.1f', 
                    cmap='RdYlBu_r', 
                    center=0,
                    ax=axes[idx],
                    cbar_kws={'label': 'Test Performance'}
                )
                # Create a nicer title that handles the layer count
                if 'fully_connected' in topology:
                    title = topology.replace('_', ' ').title()
                else:
                    title = topology.replace('_', ' ').title()
                axes[idx].set_title(f'{title} - {experiment_name}')
                axes[idx].set_xlabel('Test Task')
                axes[idx].set_ylabel('First Training Task')
            
            # Hide unused subplots
            for idx in range(n_topologies, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/sequential_transfer_heatmaps_{filename_suffix}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Create heatmaps for each experiment type
        create_heatmaps_for_experiment(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_heatmaps_for_experiment(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_third_task_analysis(self, save_path="figures"):
        """Analyze performance on the third task (not trained on) for each experiment type."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_third_task_plots(transfer_df, experiment_name, filename_suffix):
            # Filter for third task performance only
            third_task_data = transfer_df[transfer_df['is_third_task'] == True]
            
            if len(third_task_data) == 0:
                print(f"No third task data for {experiment_name}")
                return
            
            # Get unique training task combinations
            task_combinations = third_task_data[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                # Filter data for this training combination
                combo_data = third_task_data[
                    (third_task_data['train_task_1'] == train_task_1) & 
                    (third_task_data['train_task_2'] == train_task_2)
                ]
                
                if len(combo_data) == 0:
                    continue
                
                # Calculate average third task performance by topology
                topology_performance = combo_data.groupby('topology_id').agg({
                    'test_performance': ['mean', 'std'],
                    'test_success': 'mean'
                }).reset_index()
                
                # Flatten column names
                topology_performance.columns = ['topology_type', 'mean_performance', 'std_performance', 'mean_success']
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot 1: Third Task Performance
                x_pos = np.arange(len(topology_performance))
                means = topology_performance['mean_performance']
                stds = topology_performance['std_performance']
                
                bars1 = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax1.set_xlabel('Topology Type')
                ax1.set_ylabel('Third Task Performance')
                ax1.set_title(f'Third Task Transfer Learning\n{experiment_name} - Trained on {train_task_1} → {train_task_2}')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([t.replace('_', ' ').title() for t in topology_performance['topology_id']], rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean in zip(bars1, means):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                            f'{mean:.1f}', ha='center', va='bottom')
                
                # Plot 2: Success Rate
                success_rates = topology_performance['mean_success']
                
                bars2 = ax2.bar(x_pos, success_rates, alpha=0.7, color='green')
                ax2.set_xlabel('Topology Type')
                ax2.set_ylabel('Success Rate')
                ax2.set_title(f'Third Task Success Rate\n{experiment_name} - Trained on {train_task_1} → {train_task_2}')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([t.replace('_', ' ').title() for t in topology_performance['topology_id']], rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars2, success_rates):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{rate:.1%}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/third_task_analysis_{filename_suffix}_{train_task_1.replace("-", "_")}_{train_task_2.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create third task plots for each experiment type
        create_third_task_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_third_task_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_sequential_vs_single_analysis(self, save_path="figures"):
        """Compare sequential training performance vs single task training."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_comparison_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training task combinations
            task_combinations = transfer_df[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                # Filter data for this training combination
                combo_data = transfer_df[
                    (transfer_df['train_task_1'] == train_task_1) & 
                    (transfer_df['train_task_2'] == train_task_2)
                ]
                
                if len(combo_data) == 0:
                    continue
                
                # Calculate performance metrics
                performance_metrics = combo_data.groupby('topology_id').agg({
                    'task1_performance_final': 'mean',
                    'task2_performance_final': 'mean',
                    'forgetting_task1': 'mean',
                    'test_performance': 'mean'
                }).reset_index()
                
                # Filter for third task performance
                third_task_data = combo_data[combo_data['is_third_task'] == True]
                if len(third_task_data) > 0:
                    third_task_performance = third_task_data.groupby('topology_id')['test_performance'].mean().reset_index()
                    third_task_performance.columns = ['topology_id', 'third_task_performance']
                    performance_metrics = performance_metrics.merge(third_task_performance, on='topology_id', how='left')
                else:
                    performance_metrics['third_task_performance'] = 0
                
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Task 1 Performance
                x_pos = np.arange(len(performance_metrics))
                task1_perf = performance_metrics['task1_performance_final']
                
                bars1 = ax1.bar(x_pos, task1_perf, alpha=0.7, color='blue')
                ax1.set_xlabel('Topology Type')
                ax1.set_ylabel('Final Task 1 Performance')
                ax1.set_title(f'Task 1 Performance After Sequential Training\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels([t.replace('_', ' ').title() for t in performance_metrics['topology_id']], rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Task 2 Performance
                task2_perf = performance_metrics['task2_performance_final']
                
                bars2 = ax2.bar(x_pos, task2_perf, alpha=0.7, color='orange')
                ax2.set_xlabel('Topology Type')
                ax2.set_ylabel('Final Task 2 Performance')
                ax2.set_title(f'Task 2 Performance After Sequential Training\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels([t.replace('_', ' ').title() for t in performance_metrics['topology_id']], rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Forgetting on Task 1
                forgetting = performance_metrics['forgetting_task1']
                
                bars3 = ax3.bar(x_pos, forgetting, alpha=0.7, color='red')
                ax3.set_xlabel('Topology Type')
                ax3.set_ylabel('Forgetting on Task 1')
                ax3.set_title(f'Catastrophic Forgetting Analysis\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax3.set_xticks(x_pos)
                ax3.set_xticklabels([t.replace('_', ' ').title() for t in performance_metrics['topology_id']], rotation=45)
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Third Task Performance
                third_task_perf = performance_metrics['third_task_performance']
                
                bars4 = ax4.bar(x_pos, third_task_perf, alpha=0.7, color='green')
                ax4.set_xlabel('Topology Type')
                ax4.set_ylabel('Third Task Performance')
                ax4.set_title(f'Third Task Transfer Learning\n{experiment_name} - {train_task_1} → {train_task_2}')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([t.replace('_', ' ').title() for t in performance_metrics['topology_id']], rotation=45)
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'{save_path}/sequential_vs_single_{filename_suffix}_{train_task_1.replace("-", "_")}_{train_task_2.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
        
        # Create comparison plots for each experiment type
        create_comparison_plots(self.same_size_transfer_df, "Same Size Networks", "same_size")
        create_comparison_plots(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
    def create_summary_statistics(self, save_path="figures"):
        """Create summary statistics table for double-task training."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_summary_table(transfer_df, experiment_name, filename_suffix):
            # Get unique training task combinations
            task_combinations = transfer_df[['train_task_1', 'train_task_2']].drop_duplicates()
            
            all_summaries = []
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                # Filter data for this training combination
                combo_data = transfer_df[
                    (transfer_df['train_task_1'] == train_task_1) & 
                    (transfer_df['train_task_2'] == train_task_2)
                ]
                
                if len(combo_data) == 0:
                    continue
                
                # Calculate comprehensive statistics
                summary_stats = []
                
                topologies = combo_data['topology_id'].unique()
                
                for topology in topologies:
                    topology_data = combo_data[combo_data['topology_id'] == topology]
                    
                    # Task 1 performance
                    task1_final = topology_data['task1_performance_final'].mean()
                    task1_forgetting = topology_data['forgetting_task1'].mean()
                    
                    # Task 2 performance
                    task2_final = topology_data['task2_performance_final'].mean()
                    
                    # Third task performance
                    third_task_data = topology_data[topology_data['is_third_task'] == True]
                    third_task_perf = third_task_data['test_performance'].mean() if len(third_task_data) > 0 else 0
                    third_task_success = third_task_data['test_success'].mean() if len(third_task_data) > 0 else 0
                    
                    # Create a nicer topology name for display
                    if 'fully_connected' in topology:
                        display_name = topology.replace('_', ' ').title()
                    else:
                        display_name = topology.replace('_', ' ').title()
                    
                    stats = {
                        'Topology': display_name,
                        'Task 1 Final': f'{task1_final:.1f}',
                        'Task 1 Forgetting': f'{task1_forgetting:.2f}',
                        'Task 2 Final': f'{task2_final:.1f}',
                        'Third Task Perf': f'{third_task_perf:.1f}',
                        'Third Task Success': f'{third_task_success:.1%}',
                        'Avg Parameters': f'{topology_data["total_params"].mean():.0f}',
                        'Avg Training Time': f'{topology_data["training_time"].mean():.2f}s'
                    }
                    summary_stats.append(stats)
                
                if len(summary_stats) == 0:
                    continue
                    
                summary_df = pd.DataFrame(summary_stats)
                
                # Create a nice table visualization
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.axis('tight')
                ax.axis('off')
                
                table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                # Color the header
                for i in range(len(summary_df.columns)):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                plt.title(f'Double-Task Training Summary\n{experiment_name} - {train_task_1} → {train_task_2}', fontsize=16, pad=20)
                plt.savefig(f'{save_path}/summary_statistics_{filename_suffix}_{train_task_1.replace("-", "_")}_{train_task_2.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                all_summaries.append(summary_df)
            
            return all_summaries
        
        # Create summary tables for each experiment type
        same_size_summaries = create_summary_table(self.same_size_transfer_df, "Same Size Networks", "same_size")
        matched_capacity_summaries = create_summary_table(self.match_small_world_transfer_df, "Matched Capacity Networks", "matched_capacity")
        
        return same_size_summaries, matched_capacity_summaries
        
    def generate_all_analyses(self, save_path="figures"):
        """Generate all analysis figures for double-task training."""
        print("🎨 Generating comprehensive double-task transfer learning analysis...")
        print(f"📊 Analyzing {len(self.same_size_df)} same_size experiments and {len(self.match_small_world_df)} matched capacity experiments")
        
        self.create_forgetting_analysis(save_path)
        self.create_sequential_transfer_heatmaps(save_path)
        self.create_third_task_analysis(save_path)
        self.create_sequential_vs_single_analysis(save_path)
        same_size_summaries, matched_capacity_summaries = self.create_summary_statistics(save_path)
        
        print(f"✅ All analyses saved to {save_path}/")
        print(f"📊 Generated separate analyses for same_size and matched_capacity experiments")
        
        return same_size_summaries, matched_capacity_summaries

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
    results_file = latest_dir / "double_task_results.csv"
    
    if not results_file.exists():
        print(f"❌ Double-task results file not found: {results_file}")
        return
    
    print(f"📊 Analyzing double-task results from: {latest_dir}")
    
    # Create analyzer and generate all analyses
    analyzer = DoubleTaskTransferLearningAnalyzer(results_file)
    same_size_summaries, matched_capacity_summaries = analyzer.generate_all_analyses()
    
    # Print key insights
    print("\n🔍 Key Insights:")
    print("=" * 50)
    
    # Analyze forgetting patterns
    print("📉 Catastrophic Forgetting Analysis:")
    for experiment_type, summaries in [("Same Size", same_size_summaries), ("Matched Capacity", matched_capacity_summaries)]:
        if summaries:
            for summary_df in summaries:
                if 'Task 1 Forgetting' in summary_df.columns:
                    worst_forgetting = summary_df.loc[summary_df['Task 1 Forgetting'].astype(float).idxmin()]
                    best_forgetting = summary_df.loc[summary_df['Task 1 Forgetting'].astype(float).idxmax()]
                    print(f"  {experiment_type}: Worst forgetting: {worst_forgetting['Topology']} ({worst_forgetting['Task 1 Forgetting']})")
                    print(f"  {experiment_type}: Best forgetting: {best_forgetting['Topology']} ({best_forgetting['Task 1 Forgetting']})")
    
    # Analyze third task performance
    print("\n🎯 Third Task Transfer Analysis:")
    for experiment_type, summaries in [("Same Size", same_size_summaries), ("Matched Capacity", matched_capacity_summaries)]:
        if summaries:
            for summary_df in summaries:
                if 'Third Task Perf' in summary_df.columns:
                    best_third_task = summary_df.loc[summary_df['Third Task Perf'].astype(float).idxmax()]
                    print(f"  {experiment_type}: Best third task: {best_third_task['Topology']} ({best_third_task['Third Task Perf']})")
    
    print(f"\n📈 Analysis complete! Check the 'figures' directory for all visualizations.")

if __name__ == "__main__":
    main() 