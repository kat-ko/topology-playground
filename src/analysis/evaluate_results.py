import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import argparse
from scipy import stats
import glob
from datetime import datetime

class ExperimentAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = self._load_results()
        self.df = self._prepare_dataframe()
        
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from JSON file."""
        with open(self.results_dir / 'experiment_results.json', 'r') as f:
            return json.load(f)
    
    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert nested results into a flat DataFrame."""
        rows = []
        for result in self.results:
            base_data = {
                'network_size': result['network_size'],
                'seed': result['seed'],
                'strategy': result['strategy'],
                'task': result['task'],
                'topology': result['topology']
            }
            
            for algo, perf in result['performance'].items():
                row = base_data.copy()
                row['algorithm'] = algo
                
                # Add all metrics from performance dictionary
                for metric, value in perf.items():
                    if metric not in ['input_nodes', 'output_nodes']:  # Skip node lists
                        row[metric] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_metric_distribution(self, metric: str, group_by: str, save_path: str = None):
        """Create boxplot for a specific metric grouped by a factor."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x=group_by, y=metric)
        plt.title(f'{metric} Distribution by {group_by}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_metric_heatmap(self, metric: str, x_group: str, y_group: str, save_path: str = None):
        """Create heatmap for a specific metric across two factors."""
        pivot_table = self.df.pivot_table(
            values=metric,
            index=y_group,
            columns=x_group,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title(f'{metric} Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def perform_statistical_analysis(self, metric: str, group_by: str) -> Dict[str, Any]:
        """Perform statistical analysis on a metric grouped by a factor."""
        groups = self.df.groupby(group_by)[metric]
        
        # Calculate summary statistics and convert to native Python types
        summary = {
            'mean': groups.mean().to_dict(),
            'std': groups.std().to_dict(),
            'min': groups.min().to_dict(),
            'max': groups.max().to_dict()
        }
        
        # Perform ANOVA if more than 2 groups
        if len(groups) > 2:
            groups_list = [group for _, group in groups]
            f_stat, p_value = stats.f_oneway(*groups_list)
            summary['anova'] = {
                'f_statistic': float(f_stat),  # Convert numpy float to Python float
                'p_value': float(p_value)      # Convert numpy float to Python float
            }
        
        return summary
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive analysis report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all numeric columns for analysis
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Generate plots for each metric
        for metric in numeric_columns:
            # Distribution by topology
            self.plot_metric_distribution(
                metric, 'topology',
                save_path=output_dir / f'{metric}_by_topology.png'
            )
            
            # Distribution by network size
            self.plot_metric_distribution(
                metric, 'network_size',
                save_path=output_dir / f'{metric}_by_size.png'
            )
            
            # Distribution by algorithm
            self.plot_metric_distribution(
                metric, 'algorithm',
                save_path=output_dir / f'{metric}_by_algorithm.png'
            )
            
            # Heatmap of strategy vs topology
            self.plot_metric_heatmap(
                metric, 'strategy', 'topology',
                save_path=output_dir / f'{metric}_heatmap.png'
            )
        
        # Statistical analysis
        stats_results = {}
        for metric in numeric_columns:
            stats_results[metric] = {
                'by_topology': self.perform_statistical_analysis(metric, 'topology'),
                'by_algorithm': self.perform_statistical_analysis(metric, 'algorithm'),
                'by_strategy': self.perform_statistical_analysis(metric, 'strategy')
            }
        
        # Save statistical results
        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        # Save summary statistics
        summary_stats = self.df.describe()
        summary_stats.to_csv(output_dir / 'summary_statistics.csv')

def find_oldest_results_folder(base_dir: str = "results") -> Path:
    """Find the oldest results folder in the base directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")
    
    # Get all timestamped folders (format: YYYYMMDD_HHMMSS)
    result_folders = [f for f in base_path.iterdir() 
                     if f.is_dir() and '_' in f.name and 
                     all(part.isdigit() for part in f.name.split('_'))]
    
    if not result_folders:
        raise FileNotFoundError(f"No results folders found in {base_dir}")
    
    # Sort by creation time and return the oldest
    return min(result_folders, key=lambda x: x.stat().st_ctime)

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results-dir', help='Path to the results directory (optional)')
    parser.add_argument('--output-dir', default='analysis_results',
                      help='Path to save analysis results')
    args = parser.parse_args()
    
    try:
        # If no results directory specified, find the oldest one
        if args.results_dir is None:
            results_dir = find_oldest_results_folder()
            print(f"No results directory specified. Using oldest results folder: {results_dir}")
        else:
            results_dir = Path(args.results_dir)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(args.output_dir) / timestamp
        
        analyzer = ExperimentAnalyzer(results_dir)
        analyzer.generate_report(output_dir)
        print(f"Analysis complete. Results saved to {output_dir}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    main() 