import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
from scipy import stats
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ComprehensiveAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = self._load_results()
        self.df = self._prepare_dataframe()
        self.aggregated_df = self._aggregate_by_seeds()
        
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
                
                # Add network metrics
                for metric, value in perf['network_metrics'].items():
                    row[f'network_{metric}'] = value
                
                # Add node metrics (averages for input/output nodes)
                for node_type in ['input_nodes', 'output_nodes']:
                    for metric, values in perf['node_metrics'][node_type].items():
                        if isinstance(values, list):
                            row[f'{node_type}_{metric}_mean'] = np.mean(values)
                            row[f'{node_type}_{metric}_std'] = np.std(values)
                        else:
                            row[f'{node_type}_{metric}'] = values
                
                # Add strategy metrics
                for metric, value in perf['strategy_metrics'].items():
                    row[f'strategy_{metric}'] = value
                
                # Add task metrics
                for metric, value in perf['task_metrics'].items():
                    row[f'task_{metric}'] = value
                
                # Add convergence metrics
                for metric, value in perf['convergence_metrics'].items():
                    row[f'convergence_{metric}'] = value
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _aggregate_by_seeds(self) -> pd.DataFrame:
        """Aggregate results by seeds to compute mean and standard deviation."""
        # Group by all factors except seed
        group_cols = ['network_size', 'strategy', 'task', 'topology', 'algorithm']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Compute mean and std for each metric
        agg_dict = {col: ['mean', 'std'] for col in numeric_cols}
        agg_df = self.df.groupby(group_cols).agg(agg_dict)
        
        # Flatten the MultiIndex columns
        agg_df.columns = [f"{col[0]}_{col[1]}" for col in agg_df.columns]
        return agg_df
    
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
    
    def plot_correlation_matrix(self, metrics: List[str], save_path: str = None):
        """Create correlation matrix for selected metrics."""
        corr_matrix = self.df[metrics].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Metrics')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_interactive_3d(self, x_metric: str, y_metric: str, z_metric: str, 
                          color_by: str, save_path: str = None):
        """Create interactive 3D scatter plot."""
        fig = px.scatter_3d(
            self.df,
            x=x_metric,
            y=y_metric,
            z=z_metric,
            color=color_by,
            title=f'3D Scatter Plot: {x_metric} vs {y_metric} vs {z_metric}'
        )
        
        if save_path:
            fig.write_html(save_path)
        return fig
    
    def perform_statistical_analysis(self, metric: str, group_by: str) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on a metric grouped by a factor."""
        groups = self.df.groupby(group_by)[metric]
        
        # Calculate summary statistics
        summary = {
            'mean': groups.mean().to_dict(),
            'std': groups.std().to_dict(),
            'min': groups.min().to_dict(),
            'max': groups.max().to_dict(),
            'median': groups.median().to_dict(),
            'q1': groups.quantile(0.25).to_dict(),
            'q3': groups.quantile(0.75).to_dict()
        }
        
        # Perform ANOVA if more than 2 groups and not all values are constant
        if len(groups) > 2:
            groups_list = [group for _, group in groups]
            # Check if any group has non-constant values
            if any(np.std(group) > 0 for group in groups_list):
                f_stat, p_value = stats.f_oneway(*groups_list)
                summary['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value)
                }
                
                # Perform post-hoc tests if ANOVA is significant and not all values are constant
                if p_value < 0.05 and all(np.std(group) > 0 for group in groups_list):
                    from statsmodels.stats.multicomp import MultiComparison
                    mc = MultiComparison(self.df[metric], self.df[group_by])
                    tukey = mc.tukeyhsd()
                    
                    # Convert Tukey HSD results to dictionary manually
                    summary['post_hoc'] = {
                        'groups': tukey.groupsunique.tolist(),
                        'meandiffs': tukey.meandiffs.tolist(),
                        'confint': tukey.confint.tolist(),
                        'reject': tukey.reject.tolist(),
                        'pvalues': tukey.pvalues.tolist()
                    }
            else:
                summary['anova'] = {
                    'note': 'All groups have constant values, ANOVA not performed'
                }
        
        return summary
    
    def analyze_node_selection_effectiveness(self, output_dir: Path):
        """Analyze how different node selection strategies perform."""
        # 1. Strategy effectiveness by topology
        for topology in self.df['topology'].unique():
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df[self.df['topology'] == topology],
                x='strategy',
                y='strategy_input_output_distance',
                hue='task'
            )
            plt.title(f'Input-Output Distance by Strategy ({topology})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / f'strategy_distance_{topology}.png')
            plt.close()
        
        # 2. Node centrality analysis
        centrality_metrics = ['betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank']
        for metric in centrality_metrics:
            plt.figure(figsize=(15, 10))
            g = sns.FacetGrid(
                self.df,
                col='topology',
                row='task',
                hue='strategy',
                height=4
            )
            g.map_dataframe(
                sns.scatterplot,
                x=f'input_nodes_{metric}_mean',
                y=f'output_nodes_{metric}_mean',
                alpha=0.6
            )
            g.add_legend()
            plt.savefig(output_dir / f'node_centrality_{metric}.png')
            plt.close()
        
        # 3. Module membership analysis (for modular topology)
        if 'module_membership' in self.df.columns:
            mod_df = self.df[self.df['topology'] == 'modular']
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=mod_df,
                x='strategy',
                y='input_nodes_module_membership_mean',
                hue='task'
            )
            plt.title('Input Node Module Membership by Strategy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'module_membership_input.png')
            plt.close()
    
    def analyze_network_performance(self, output_dir: Path):
        """Analyze how network properties affect task performance."""
        # 1. Network metrics vs task performance
        network_metrics = [col for col in self.df.columns if col.startswith('network_')]
        task_metrics = [col for col in self.df.columns if col.startswith('task_')]
        
        for task_metric in task_metrics:
            plt.figure(figsize=(15, 10))
            g = sns.FacetGrid(
                self.df,
                col='topology',
                row='task',
                hue='strategy',
                height=4
            )
            g.map_dataframe(
                sns.scatterplot,
                x='network_avg_path_length',
                y=task_metric,
                alpha=0.6
            )
            g.add_legend()
            plt.savefig(output_dir / f'network_vs_performance_{task_metric}.png')
            plt.close()
        
        # 2. Network size impact
        for task_metric in task_metrics:
            plt.figure(figsize=(12, 6))
            sns.boxplot(
                data=self.df,
                x='network_size',
                y=task_metric,
                hue='topology'
            )
            plt.title(f'Network Size Impact on {task_metric}')
            plt.tight_layout()
            plt.savefig(output_dir / f'size_impact_{task_metric}.png')
            plt.close()
    
    def generate_comprehensive_report(self, output_dir: str):
        """Generate comprehensive analysis report with all visualizations and statistics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all numeric columns for analysis
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # 1. Basic Distribution Analysis
        print("Generating distribution plots...")
        for metric in numeric_columns:
            for group_by in ['topology', 'network_size', 'strategy', 'algorithm', 'task']:
                self.plot_metric_distribution(
                    metric, group_by,
                    save_path=output_dir / f'dist_{metric}_by_{group_by}.png'
                )
        
        # 2. Heatmap Analysis
        print("Generating heatmaps...")
        for metric in numeric_columns:
            self.plot_metric_heatmap(
                metric, 'strategy', 'topology',
                save_path=output_dir / f'heatmap_{metric}.png'
            )
        
        # 3. Correlation Analysis
        print("Generating correlation matrix...")
        self.plot_correlation_matrix(
            numeric_columns,
            save_path=output_dir / 'correlation_matrix.png'
        )
        
        # 4. Interactive 3D Plots
        print("Generating interactive 3D plots...")
        if len(numeric_columns) >= 3:
            self.plot_interactive_3d(
                numeric_columns[0],
                numeric_columns[1],
                numeric_columns[2],
                'topology',
                save_path=output_dir / 'interactive_3d.html'
            )
        
        # 5. Node Selection Analysis
        print("Analyzing node selection effectiveness...")
        self.analyze_node_selection_effectiveness(output_dir)
        
        # 6. Network Performance Analysis
        print("Analyzing network performance...")
        self.analyze_network_performance(output_dir)
        
        # 7. Statistical Analysis
        print("Performing statistical analysis...")
        stats_results = {}
        for metric in numeric_columns:
            stats_results[metric] = {
                'by_topology': self.perform_statistical_analysis(metric, 'topology'),
                'by_algorithm': self.perform_statistical_analysis(metric, 'algorithm'),
                'by_strategy': self.perform_statistical_analysis(metric, 'strategy'),
                'by_network_size': self.perform_statistical_analysis(metric, 'network_size')
            }
        
        # Save statistical results
        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(stats_results, f, indent=2)
        
        # 8. Summary Statistics
        print("Generating summary statistics...")
        summary_stats = {
            'overall': self._convert_dataframe_to_dict(self.df.describe()),
            'by_topology': self._convert_dataframe_to_dict(self.df.groupby('topology').describe()),
            'by_algorithm': self._convert_dataframe_to_dict(self.df.groupby('algorithm').describe()),
            'by_strategy': self._convert_dataframe_to_dict(self.df.groupby('strategy').describe()),
            'by_network_size': self._convert_dataframe_to_dict(self.df.groupby('network_size').describe())
        }
        
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # 9. Generate HTML Report
        print("Generating HTML report...")
        self._generate_html_report(output_dir)
        
        print(f"Analysis complete. Results saved to {output_dir}")
    
    def _convert_dataframe_to_dict(self, df: pd.DataFrame) -> Dict:
        """Convert a DataFrame (including MultiIndex) to a nested dictionary."""
        def convert_value(v):
            if isinstance(v, (np.integer, np.floating)):
                return float(v)
            return v

        # Handle MultiIndex in index
        if isinstance(df.index, pd.MultiIndex):
            result = {}
            for idx in df.index.levels[0]:
                sub_df = df.xs(idx, level=0)
                result[str(idx)] = self._convert_dataframe_to_dict(sub_df)
            return result
        
        # Handle MultiIndex in columns
        if isinstance(df.columns, pd.MultiIndex):
            result = {}
            for col in df.columns.levels[0]:
                sub_df = df.xs(col, level=0, axis=1)
                result[str(col)] = self._convert_dataframe_to_dict(sub_df)
            return result
        
        # Regular DataFrame
        return {str(idx): {str(col): convert_value(val) for col, val in row.items()} 
                for idx, row in df.iterrows()}
    
    def _generate_html_report(self, output_dir: Path):
        """Generate an HTML report summarizing all findings."""
        html_content = f"""
        <html>
        <head>
            <title>Experiment Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Experiment Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Overview</h2>
                <p>Total experiments: {len(self.df)}</p>
                <p>Unique configurations: {len(self.aggregated_df)}</p>
            </div>
            
            <div class="section">
                <h2>Distribution Analysis</h2>
                {self._generate_distribution_section(output_dir)}
            </div>
            
            <div class="section">
                <h2>Correlation Analysis</h2>
                <img src="correlation_matrix.png" alt="Correlation Matrix">
            </div>
            
            <div class="section">
                <h2>Interactive Analysis</h2>
                <p>See <a href="interactive_3d.html">Interactive 3D Plot</a> for detailed exploration.</p>
            </div>
            
            <div class="section">
                <h2>Node Selection Analysis</h2>
                {self._generate_node_selection_section(output_dir)}
            </div>
            
            <div class="section">
                <h2>Network Performance Analysis</h2>
                {self._generate_network_performance_section(output_dir)}
            </div>
            
            <div class="section">
                <h2>Statistical Analysis</h2>
                {self._generate_statistical_section(output_dir)}
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html_content)
    
    def _generate_distribution_section(self, output_dir: Path) -> str:
        """Generate HTML section for distribution plots."""
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        html = ""
        
        for metric in numeric_columns:
            html += f"<h3>{metric}</h3>"
            for group_by in ['topology', 'network_size', 'strategy', 'algorithm']:
                html += f"""
                <div class="metric">
                    <h4>By {group_by}</h4>
                    <img src="dist_{metric}_by_{group_by}.png" alt="{metric} by {group_by}">
                </div>
                """
        
        return html
    
    def _generate_node_selection_section(self, output_dir: Path) -> str:
        """Generate HTML section for node selection analysis."""
        html = ""
        
        # 1. Strategy effectiveness by topology
        for topology in self.df['topology'].unique():
            html += f"<h3>Strategy Effectiveness by Topology ({topology})</h3>"
            html += f"""
            <div class="metric">
                <h4>Input-Output Distance</h4>
                <img src="strategy_distance_{topology}.png" alt="Input-Output Distance">
            </div>
            """
        
        # 2. Node centrality analysis
        centrality_metrics = ['betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank']
        for metric in centrality_metrics:
            html += f"<h3>Node Centrality Analysis ({metric})</h3>"
            html += f"""
            <div class="metric">
                <h4>Node Centrality</h4>
                <img src="node_centrality_{metric}.png" alt="{metric} Node Centrality">
            </div>
            """
        
        # 3. Module membership analysis (for modular topology)
        if 'module_membership' in self.df.columns:
            html += "<h3>Module Membership Analysis</h3>"
            html += f"""
            <div class="metric">
                <h4>Input Node Module Membership</h4>
                <img src="module_membership_input.png" alt="Input Node Module Membership">
            </div>
            """
        
        return html
    
    def _generate_network_performance_section(self, output_dir: Path) -> str:
        """Generate HTML section for network performance analysis."""
        html = ""
        
        # 1. Network metrics vs task performance
        network_metrics = [col for col in self.df.columns if col.startswith('network_')]
        task_metrics = [col for col in self.df.columns if col.startswith('task_')]
        
        for task_metric in task_metrics:
            html += f"<h3>Network Metrics vs Task Performance ({task_metric})</h3>"
            html += f"""
            <div class="metric">
                <h4>Network Metrics vs Task Performance</h4>
                <img src="network_vs_performance_{task_metric}.png" alt="{task_metric} Network Metrics vs Task Performance">
            </div>
            """
        
        # 2. Network size impact
        for task_metric in task_metrics:
            html += f"<h3>Network Size Impact on {task_metric}</h3>"
            html += f"""
            <div class="metric">
                <h4>Network Size Impact</h4>
                <img src="size_impact_{task_metric}.png" alt="{task_metric} Network Size Impact">
            </div>
            """
        
        return html
    
    def _generate_statistical_section(self, output_dir: Path) -> str:
        """Generate HTML section for statistical analysis."""
        with open(output_dir / 'statistical_analysis.json', 'r') as f:
            stats = json.load(f)
        
        html = "<div class='statistical-analysis'>"
        for metric, analyses in stats.items():
            html += f"<h3>{metric}</h3>"
            for factor, analysis in analyses.items():
                html += f"<h4>{factor}</h4>"
                html += "<ul>"
                for stat, value in analysis.items():
                    if isinstance(value, dict):
                        html += f"<li>{stat}: {json.dumps(value, indent=2)}</li>"
                    else:
                        html += f"<li>{stat}: {value}</li>"
                html += "</ul>"
        
        html += "</div>"
        return html

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
    parser = argparse.ArgumentParser(description='Perform comprehensive analysis of experiment results')
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
        
        analyzer = ComprehensiveAnalyzer(results_dir)
        analyzer.generate_comprehensive_report(output_dir)
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