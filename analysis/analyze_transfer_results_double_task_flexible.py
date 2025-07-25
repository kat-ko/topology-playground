#!/usr/bin/env python3
"""
Double-Task Transfer Learning Analysis: Comprehensive visualization of sequential training effects
Analyzes forward/backward transfer, forgetting, and sequential training performance
FLEXIBLE VERSION: Automatically detects tasks from data instead of hardcoding
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
        
        # FLEXIBLE: Automatically detect tasks from the data
        self.tasks = self._detect_tasks_from_data()
        print(f"🔍 Detected {len(self.tasks)} tasks: {', '.join(self.tasks)}")
        
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
    
    def _detect_tasks_from_data(self):
        """Automatically detect tasks from the data columns."""
        # Look for columns that end with '_final_mean_reward' to identify tasks
        task_columns = [col for col in self.df.columns if col.endswith('_final_mean_reward')]
        tasks = [col.replace('_final_mean_reward', '') for col in task_columns]
        
        # Validate that we found tasks
        if not tasks:
            raise ValueError("No tasks detected in the data. Expected columns ending with '_final_mean_reward'")
        
        # Sort tasks for consistent ordering
        tasks.sort()
        return tasks
        
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
        
    def create_forgetting_analysis(self, save_path="figures"):
        """Create forgetting analysis plots."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_forgetting_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training task combinations
            task_combinations = transfer_df[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                combo_data = transfer_df[
                    (transfer_df['train_task_1'] == train_task_1) & 
                    (transfer_df['train_task_2'] == train_task_2)
                ]
                
                # Create forgetting analysis plot
                plt.figure(figsize=(12, 8))
                
                unique_topologies = combo_data['topology_id'].unique()
                x_pos = np.arange(len(unique_topologies))
                
                # Plot forgetting on task 1
                forgetting_values = []
                forgetting_errors = []
                
                for topology in unique_topologies:
                    topology_data = combo_data[combo_data['topology_id'] == topology]
                    if not topology_data.empty:
                        mean_forgetting = topology_data['forgetting_task1'].mean()
                        std_forgetting = topology_data['forgetting_task1'].std()
                        forgetting_values.append(mean_forgetting)
                        forgetting_errors.append(std_forgetting)
                    else:
                        forgetting_values.append(0)
                        forgetting_errors.append(0)
                
                plt.errorbar(x_pos, forgetting_values, yerr=forgetting_errors, 
                           marker='o', capsize=5, label=f'Forgetting on {train_task_1}')
                
                plt.xlabel('Topology')
                plt.ylabel('Forgetting (Performance Drop)')
                plt.title(f'Forgetting Analysis - {train_task_1} → {train_task_2} ({experiment_name})')
                plt.xticks(x_pos, unique_topologies, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"forgetting_analysis_{experiment_name}_{train_task_1}_{train_task_2}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created forgetting analysis: {filename}")
        
        # Create forgetting plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_forgetting_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_forgetting_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_sequential_transfer_heatmaps(self, save_path="figures"):
        """Create sequential transfer learning heatmaps."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_heatmaps_for_experiment(transfer_df, experiment_name, filename_suffix):
            # Get unique topologies (including layer variants)
            unique_topologies = transfer_df['topology_id'].unique()
            
            for topology in unique_topologies:
                topology_data = transfer_df[transfer_df['topology_id'] == topology]
                
                # Create pivot table for heatmap (using transfer ratio from task 2)
                pivot_data = topology_data.pivot_table(
                    values='transfer_ratio_task2', 
                    index='train_task_1', 
                    columns='test_task', 
                    aggfunc='mean'
                )
                
                # Create the heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           center=1.0, vmin=0, vmax=2, cbar_kws={'label': 'Transfer Ratio (from Task 2)'})
                plt.title(f'Sequential Transfer Learning Heatmap - {topology} ({experiment_name})')
                plt.xlabel('Test Task')
                plt.ylabel('First Training Task')
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"sequential_transfer_heatmap_{topology}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created sequential transfer heatmap: {filename}")
        
        # Create heatmaps for each experiment type
        if not self.same_size_transfer_df.empty:
            create_heatmaps_for_experiment(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_heatmaps_for_experiment(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_third_task_analysis(self, save_path="figures"):
        """Create third task analysis (for cases with more than 2 tasks)."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_third_task_plots(transfer_df, experiment_name, filename_suffix):
            # Filter for third task performance only
            third_task_data = transfer_df[transfer_df['is_third_task'] == True]
            
            if third_task_data.empty:
                print(f"   ⚠️  No third task data found for {experiment_name}")
                return
            
            # Get unique training task combinations
            task_combinations = third_task_data[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                combo_data = third_task_data[
                    (third_task_data['train_task_1'] == train_task_1) & 
                    (third_task_data['train_task_2'] == train_task_2)
                ]
                
                # Create third task analysis plot
                plt.figure(figsize=(12, 8))
                
                unique_topologies = combo_data['topology_id'].unique()
                x_pos = np.arange(len(unique_topologies))
                
                # Plot third task performance for each topology
                third_task_performance = []
                third_task_errors = []
                
                for topology in unique_topologies:
                    topology_data = combo_data[combo_data['topology_id'] == topology]
                    if not topology_data.empty:
                        mean_perf = topology_data['test_performance'].mean()
                        std_perf = topology_data['test_performance'].std()
                        third_task_performance.append(mean_perf)
                        third_task_errors.append(std_perf)
                    else:
                        third_task_performance.append(0)
                        third_task_errors.append(0)
                
                plt.errorbar(x_pos, third_task_performance, yerr=third_task_errors, 
                           marker='o', capsize=5, label='Third Task Performance')
                
                plt.xlabel('Topology')
                plt.ylabel('Performance on Third Task')
                plt.title(f'Third Task Analysis - {train_task_1} → {train_task_2} ({experiment_name})')
                plt.xticks(x_pos, unique_topologies, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"third_task_analysis_{experiment_name}_{train_task_1}_{train_task_2}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created third task analysis: {filename}")
        
        # Create third task plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_third_task_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_third_task_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_sequential_vs_single_analysis(self, save_path="figures"):
        """Create comparison between sequential and single task training."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_comparison_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training task combinations
            task_combinations = transfer_df[['train_task_1', 'train_task_2']].drop_duplicates()
            
            for _, combo in task_combinations.iterrows():
                train_task_1 = combo['train_task_1']
                train_task_2 = combo['train_task_2']
                
                combo_data = transfer_df[
                    (transfer_df['train_task_1'] == train_task_1) & 
                    (transfer_df['train_task_2'] == train_task_2)
                ]
                
                # Create comparison plot
                plt.figure(figsize=(12, 8))
                
                unique_topologies = combo_data['topology_id'].unique()
                x_pos = np.arange(len(unique_topologies))
                
                # Plot performance on training tasks
                task1_performance = []
                task1_errors = []
                task2_performance = []
                task2_errors = []
                
                for topology in unique_topologies:
                    topology_data = combo_data[combo_data['topology_id'] == topology]
                    
                    # Task 1 performance
                    task1_data = topology_data[topology_data['is_training_task_1'] == True]
                    if not task1_data.empty:
                        mean_perf = task1_data['test_performance'].mean()
                        std_perf = task1_data['test_performance'].std()
                        task1_performance.append(mean_perf)
                        task1_errors.append(std_perf)
                    else:
                        task1_performance.append(0)
                        task1_errors.append(0)
                    
                    # Task 2 performance
                    task2_data = topology_data[topology_data['is_training_task_2'] == True]
                    if not task2_data.empty:
                        mean_perf = task2_data['test_performance'].mean()
                        std_perf = task2_data['test_performance'].std()
                        task2_performance.append(mean_perf)
                        task2_errors.append(std_perf)
                    else:
                        task2_performance.append(0)
                        task2_errors.append(0)
                
                plt.errorbar(x_pos - 0.2, task1_performance, yerr=task1_errors, 
                           label=f'{train_task_1} Performance', marker='o', capsize=5)
                plt.errorbar(x_pos + 0.2, task2_performance, yerr=task2_errors, 
                           label=f'{train_task_2} Performance', marker='s', capsize=5)
                
                plt.xlabel('Topology')
                plt.ylabel('Performance')
                plt.title(f'Sequential Training Performance - {train_task_1} → {train_task_2} ({experiment_name})')
                plt.xticks(x_pos, unique_topologies, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"sequential_vs_single_{experiment_name}_{train_task_1}_{train_task_2}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created sequential vs single analysis: {filename}")
        
        # Create comparison plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_comparison_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_comparison_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_summary_statistics(self, save_path="figures"):
        """Create summary statistics table."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_summary_table(transfer_df, experiment_name, filename_suffix):
            # Calculate summary statistics
            summary_stats = []
            
            for topology in transfer_df['topology_id'].unique():
                topology_data = transfer_df[transfer_df['topology_id'] == topology]
                
                for _, combo in topology_data[['train_task_1', 'train_task_2']].drop_duplicates().iterrows():
                    train_task_1 = combo['train_task_1']
                    train_task_2 = combo['train_task_2']
                    
                    combo_data = topology_data[
                        (topology_data['train_task_1'] == train_task_1) & 
                        (topology_data['train_task_2'] == train_task_2)
                    ]
                    
                    # Calculate metrics
                    avg_forgetting = combo_data['forgetting_task1'].mean()
                    avg_task1_performance = combo_data[combo_data['is_training_task_1'] == True]['test_performance'].mean()
                    avg_task2_performance = combo_data[combo_data['is_training_task_2'] == True]['test_performance'].mean()
                    avg_params = combo_data['total_params'].mean()
                    avg_training_time = combo_data['training_time'].mean()
                    
                    summary_stats.append({
                        'Topology': topology,
                        'Training Sequence': f"{train_task_1} → {train_task_2}",
                        'Avg Forgetting on Task 1': f"{avg_forgetting:.3f}",
                        f'Avg {train_task_1} Performance': f"{avg_task1_performance:.2f}",
                        f'Avg {train_task_2} Performance': f"{avg_task2_performance:.2f}",
                        'Avg Parameters': f"{avg_params:,.0f}",
                        'Avg Training Time (s)': f"{avg_training_time:.1f}"
                    })
            
            # Create summary table
            summary_df = pd.DataFrame(summary_stats)
            
            # Save as CSV
            filename = f"summary_statistics_{experiment_name}_{len(self.tasks)}tasks.csv"
            summary_df.to_csv(f"{save_path}/{filename}", index=False)
            
            # Create a nice formatted table
            fig, ax = plt.subplots(figsize=(16, len(summary_stats) * 0.4 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            plt.title(f'Summary Statistics - {experiment_name} ({len(self.tasks)} tasks)', pad=20)
            plt.tight_layout()
            
            # Save as PNG
            png_filename = f"summary_statistics_{experiment_name}_{len(self.tasks)}tasks.png"
            plt.savefig(f"{save_path}/{png_filename}", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Created summary statistics: {filename}, {png_filename}")
        
        # Create summary tables for each experiment type
        if not self.same_size_transfer_df.empty:
            create_summary_table(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_summary_table(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def generate_all_analyses(self, save_path="figures"):
        """Generate all analysis plots and tables."""
        print(f"🔍 Generating double-task transfer learning analysis for {len(self.tasks)} tasks...")
        print(f"   Tasks: {', '.join(self.tasks)}")
        
        self.create_forgetting_analysis(save_path)
        self.create_sequential_transfer_heatmaps(save_path)
        
        # Only create third task analysis if we have more than 2 tasks
        if len(self.tasks) > 2:
            self.create_third_task_analysis(save_path)
        else:
            print(f"   ⚠️  Skipping third task analysis (only {len(self.tasks)} tasks)")
        
        self.create_sequential_vs_single_analysis(save_path)
        self.create_summary_statistics(save_path)
        
        print(f"✅ All analyses completed! Results saved to: {save_path}")

def main():
    """Main function to run the analysis."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_transfer_results_double_task_flexible.py <results_file>")
        print("Example: python analyze_transfer_results_double_task_flexible.py results/double_task_training_20241201_120000/double_task_results.csv")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"🔍 Loading results from: {results_file}")
    
    # Create analyzer and generate all analyses
    analyzer = DoubleTaskTransferLearningAnalyzer(results_file)
    analyzer.generate_all_analyses()
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Analyzed {len(analyzer.tasks)} tasks: {', '.join(analyzer.tasks)}")
    print(f"🔬 Generated analyses for {len(analyzer.topologies)} topologies")
    print(f"📈 Created visualizations in: figures/")

if __name__ == "__main__":
    main() 