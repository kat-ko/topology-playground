#!/usr/bin/env python3
"""
Transfer Learning Analysis: Comprehensive visualization of topology differences
Analyzes forward/backward transfer, parameter efficiency, and task difficulty
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

class TransferLearningAnalyzer:
    def __init__(self, results_file):
        """Initialize analyzer with results data."""
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
        # Look for columns that end with '_mean_reward' to identify tasks
        task_columns = [col for col in self.df.columns if col.endswith('_mean_reward')]
        tasks = [col.replace('_mean_reward', '') for col in task_columns]
        
        # Validate that we found tasks
        if not tasks:
            raise ValueError("No tasks detected in the data. Expected columns ending with '_mean_reward'")
        
        # Sort tasks for consistent ordering
        tasks.sort()
        return tasks
        
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
                    
                    # Create a unique topology identifier that includes layer count for fully connected
                    topology_id = row['topology_type']
                    if row['topology_type'] == 'fully_connected':
                        topology_id = f"fully_connected_{row['num_layers']}layers"
                    
                    transfer_data.append({
                        'topology_type': row['topology_type'],
                        'topology_id': topology_id,  # Unique identifier including layer count
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
            # Get unique topologies (including layer variants)
            unique_topologies = transfer_df['topology_id'].unique()
            
            for topology in unique_topologies:
                topology_data = transfer_df[transfer_df['topology_id'] == topology]
                
                # Create pivot table for heatmap
                pivot_data = topology_data.pivot_table(
                    values='transfer_ratio', 
                    index='train_task', 
                    columns='test_task', 
                    aggfunc='mean'
                )
                
                # Create the heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                           center=1.0, vmin=0, vmax=2, cbar_kws={'label': 'Transfer Ratio'})
                plt.title(f'Transfer Learning Heatmap - {topology} ({experiment_name})')
                plt.xlabel('Test Task')
                plt.ylabel('Training Task')
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"transfer_heatmap_{topology}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created transfer heatmap: {filename}")
        
        # Create heatmaps for each experiment type
        if not self.same_size_transfer_df.empty:
            create_heatmaps_for_experiment(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_heatmaps_for_experiment(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_topology_comparison(self, save_path="figures"):
        """Create topology comparison plots."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_comparison_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Create comparison plot
                plt.figure(figsize=(12, 8))
                
                # Plot transfer ratios for each topology
                unique_topologies = task_data['topology_id'].unique()
                x_pos = np.arange(len(unique_topologies))
                
                for i, test_task in enumerate(self.tasks):
                    task_performance = []
                    task_errors = []
                    
                    for topology in unique_topologies:
                        topology_task_data = task_data[
                            (task_data['topology_id'] == topology) & 
                            (task_data['test_task'] == test_task)
                        ]
                        
                        if not topology_task_data.empty:
                            mean_perf = topology_task_data['transfer_ratio'].mean()
                            std_perf = topology_task_data['transfer_ratio'].std()
                            task_performance.append(mean_perf)
                            task_errors.append(std_perf)
                        else:
                            task_performance.append(0)
                            task_errors.append(0)
                    
                    # Plot with error bars
                    plt.errorbar(x_pos + i*0.15, task_performance, yerr=task_errors, 
                               label=test_task, marker='o', capsize=5)
                
                plt.xlabel('Topology')
                plt.ylabel('Transfer Ratio')
                plt.title(f'Topology Comparison - Training on {train_task} ({experiment_name})')
                plt.xticks(x_pos + 0.15, unique_topologies, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"topology_comparison_{train_task}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created topology comparison: {filename}")
        
        # Create comparison plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_comparison_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_comparison_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_parameter_efficiency_analysis(self, save_path="figures"):
        """Create parameter efficiency analysis."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_efficiency_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Create efficiency plot
                plt.figure(figsize=(12, 8))
                
                unique_topologies = task_data['topology_id'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_topologies)))
                
                for i, topology in enumerate(unique_topologies):
                    topology_data = task_data[task_data['topology_id'] == topology]
                    
                    # Plot parameter efficiency vs transfer ratio
                    plt.scatter(topology_data['parameter_efficiency'], 
                              topology_data['transfer_ratio'], 
                              label=topology, color=colors[i], alpha=0.7, s=100)
                
                plt.xlabel('Parameter Efficiency (params/node)')
                plt.ylabel('Transfer Ratio')
                plt.title(f'Parameter Efficiency vs Transfer - Training on {train_task} ({experiment_name})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"parameter_efficiency_{train_task}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created parameter efficiency plot: {filename}")
        
        # Create efficiency plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_efficiency_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_efficiency_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_task_difficulty_analysis(self, save_path="figures"):
        """Create task difficulty analysis."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_task_plots(transfer_df, experiment_name, filename_suffix):
            # Calculate average performance for each task
            task_performance = {}
            task_std = {}
            
            for task in self.tasks:
                task_data = transfer_df[transfer_df['test_task'] == task]
                task_performance[task] = task_data['test_performance'].mean()
                task_std[task] = task_data['test_performance'].std()
            
            # Create task difficulty plot
            plt.figure(figsize=(10, 6))
            tasks = list(task_performance.keys())
            performances = list(task_performance.values())
            stds = list(task_std.values())
            
            bars = plt.bar(tasks, performances, yerr=stds, capsize=5, alpha=0.7)
            plt.xlabel('Task')
            plt.ylabel('Average Performance')
            plt.title(f'Task Difficulty Analysis ({experiment_name})')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save with task count in filename
            filename = f"task_difficulty_{experiment_name}_{len(self.tasks)}tasks.png"
            plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Created task difficulty plot: {filename}")
        
        # Create task plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_task_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_task_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_forward_backward_analysis(self, save_path="figures"):
        """Create forward/backward transfer analysis."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_forward_backward_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Separate forward and backward transfer
                forward_data = task_data[task_data['is_forward'] == True]
                same_task_data = task_data[task_data['is_forward'] == False]
                
                # Create forward/backward comparison
                plt.figure(figsize=(12, 8))
                
                unique_topologies = task_data['topology_id'].unique()
                x_pos = np.arange(len(unique_topologies))
                
                # Plot same-task performance
                same_task_performance = []
                same_task_errors = []
                
                for topology in unique_topologies:
                    topology_data = same_task_data[same_task_data['topology_id'] == topology]
                    if not topology_data.empty:
                        mean_perf = topology_data['transfer_ratio'].mean()
                        std_perf = topology_data['transfer_ratio'].std()
                        same_task_performance.append(mean_perf)
                        same_task_errors.append(std_perf)
                    else:
                        same_task_performance.append(0)
                        same_task_errors.append(0)
                
                plt.errorbar(x_pos - 0.2, same_task_performance, yerr=same_task_errors, 
                           label='Same Task', marker='o', capsize=5, color='blue')
                
                # Plot forward transfer performance
                forward_performance = []
                forward_errors = []
                
                for topology in unique_topologies:
                    topology_data = forward_data[forward_data['topology_id'] == topology]
                    if not topology_data.empty:
                        mean_perf = topology_data['transfer_ratio'].mean()
                        std_perf = topology_data['transfer_ratio'].std()
                        forward_performance.append(mean_perf)
                        forward_errors.append(std_perf)
                    else:
                        forward_performance.append(0)
                        forward_errors.append(0)
                
                plt.errorbar(x_pos + 0.2, forward_performance, yerr=forward_errors, 
                           label='Forward Transfer', marker='s', capsize=5, color='red')
                
                plt.xlabel('Topology')
                plt.ylabel('Transfer Ratio')
                plt.title(f'Forward vs Same-Task Transfer - Training on {train_task} ({experiment_name})')
                plt.xticks(x_pos, unique_topologies, rotation=45, ha='right')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"forward_backward_{train_task}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created forward/backward analysis: {filename}")
        
        # Create forward/backward plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_forward_backward_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_forward_backward_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_training_efficiency_analysis(self, save_path="figures"):
        """Create training efficiency analysis."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_training_plots(transfer_df, experiment_name, filename_suffix):
            # Get unique training tasks
            train_tasks = transfer_df['train_task'].unique()
            
            for train_task in train_tasks:
                task_data = transfer_df[transfer_df['train_task'] == train_task]
                
                # Create training efficiency plot
                plt.figure(figsize=(12, 8))
                
                unique_topologies = task_data['topology_id'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_topologies)))
                
                for i, topology in enumerate(unique_topologies):
                    topology_data = task_data[task_data['topology_id'] == topology]
                    
                    # Plot training time vs transfer ratio
                    plt.scatter(topology_data['training_time'], 
                              topology_data['transfer_ratio'], 
                              label=topology, color=colors[i], alpha=0.7, s=100)
                
                plt.xlabel('Training Time (seconds)')
                plt.ylabel('Transfer Ratio')
                plt.title(f'Training Efficiency vs Transfer - Training on {train_task} ({experiment_name})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save with task count in filename
                filename = f"training_efficiency_{train_task}_{experiment_name}_{len(self.tasks)}tasks.png"
                plt.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Created training efficiency plot: {filename}")
        
        # Create training plots for each experiment type
        if not self.same_size_transfer_df.empty:
            create_training_plots(self.same_size_transfer_df, 'same_size', 'same_size')
        if not self.match_small_world_transfer_df.empty:
            create_training_plots(self.match_small_world_transfer_df, 'match_small_world', 'match_small_world')
    
    def create_summary_statistics(self, save_path="figures"):
        """Create summary statistics table."""
        Path(save_path).mkdir(exist_ok=True)
        
        def create_summary_table(transfer_df, experiment_name, filename_suffix):
            # Calculate summary statistics
            summary_stats = []
            
            for topology in transfer_df['topology_id'].unique():
                topology_data = transfer_df[transfer_df['topology_id'] == topology]
                
                for train_task in topology_data['train_task'].unique():
                    task_data = topology_data[topology_data['train_task'] == train_task]
                    
                    # Calculate metrics
                    avg_transfer_ratio = task_data['transfer_ratio'].mean()
                    std_transfer_ratio = task_data['transfer_ratio'].std()
                    avg_performance = task_data['test_performance'].mean()
                    avg_params = task_data['total_params'].mean()
                    avg_training_time = task_data['training_time'].mean()
                    
                    summary_stats.append({
                        'Topology': topology,
                        'Training Task': train_task,
                        'Avg Transfer Ratio': f"{avg_transfer_ratio:.3f} ± {std_transfer_ratio:.3f}",
                        'Avg Performance': f"{avg_performance:.2f}",
                        'Avg Parameters': f"{avg_params:,.0f}",
                        'Avg Training Time (s)': f"{avg_training_time:.1f}"
                    })
            
            # Create summary table
            summary_df = pd.DataFrame(summary_stats)
            
            # Save as CSV
            filename = f"summary_statistics_{experiment_name}_{len(self.tasks)}tasks.csv"
            summary_df.to_csv(f"{save_path}/{filename}", index=False)
            
            # Create a nice formatted table
            fig, ax = plt.subplots(figsize=(14, len(summary_stats) * 0.4 + 2))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns, 
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
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
        print(f"🔍 Generating transfer learning analysis for {len(self.tasks)} tasks...")
        print(f"   Tasks: {', '.join(self.tasks)}")
        
        self.create_transfer_heatmaps(save_path)
        self.create_topology_comparison(save_path)
        self.create_parameter_efficiency_analysis(save_path)
        self.create_task_difficulty_analysis(save_path)
        self.create_forward_backward_analysis(save_path)
        self.create_training_efficiency_analysis(save_path)
        self.create_summary_statistics(save_path)
        
        print(f"✅ All analyses completed! Results saved to: {save_path}")

def main():
    """Main function to run the analysis."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_transfer_results_single_task_flexible.py <results_file>")
        print("Example: python analyze_transfer_results_single_task_flexible.py results/single_task_training_20241201_120000/cross_task_results.csv")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        sys.exit(1)
    
    print(f"🔍 Loading results from: {results_file}")
    
    # Create analyzer and generate all analyses
    analyzer = TransferLearningAnalyzer(results_file)
    analyzer.generate_all_analyses()
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Analyzed {len(analyzer.tasks)} tasks: {', '.join(analyzer.tasks)}")
    print(f"🔬 Generated analyses for {len(analyzer.topologies)} topologies")
    print(f"📈 Created visualizations in: figures/")

if __name__ == "__main__":
    main() 