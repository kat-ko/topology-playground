import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, silhouette_score, precision_score, recall_score, mean_absolute_error, davies_bouldin_score, calinski_harabasz_score
import time

from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..node_selection.strategies import NodeSelector
from ..tasks.task_definitions import TaskGenerator, TaskEvaluator

class ExperimentRunner:
    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.task_generator = TaskGenerator()
        self.task_evaluator = TaskEvaluator()
        
    def run_experiment(self):
        """Run the complete experiment with all combinations of parameters."""
        print("Executing run_experiment from:", __file__)
        print("Experiment parameters:")
        print(f"Network sizes: {self.config['network_sizes']}")
        print(f"Seeds: {self.config['seeds']}")
        print(f"Small-world parameters: {self.config['small_world_params']}")
        print(f"Modular parameters: {self.config['modular_params']}")
        print(f"Node selection strategies: {self.config['node_selection_strategies']}")
        print(f"Tasks: {self.config['tasks']}")
        results = []
        
        for size in tqdm(self.config['network_sizes'], desc="Network sizes"):
            for seed in tqdm(self.config['seeds'], desc="Seeds", leave=False):
                # Generate networks
                small_world = SmallWorldTopology(
                    size=size,
                    k=self.config['small_world_params']['k'],
                    p=self.config['small_world_params']['p'],
                    seed=seed
                )
                
                modular = ModularTopology(
                    size=size,
                    num_modules=self.config['modular_params']['num_modules'],
                    inter_module_prob=self.config['modular_params']['inter_module_prob'],
                    intra_module_prob=self.config['modular_params']['intra_module_prob'],
                    seed=seed
                )
                
                # Generate networks
                sw_graph = small_world.generate()
                mod_graph = modular.generate()
                
                # Test each node selection strategy
                for strategy in tqdm(self.config['node_selection_strategies'], desc="Strategies", leave=False):
                    # Small-world network
                    sw_selector = NodeSelector(sw_graph, self.config['num_io_nodes'], seed)
                    if strategy == 'module_based':
                        sw_input_nodes, sw_output_nodes = sw_selector.random_selection()
                    else:
                        sw_input_nodes, sw_output_nodes = getattr(sw_selector, f"{strategy}_selection")()
                    
                    # Modular network
                    mod_selector = NodeSelector(mod_graph, self.config['num_io_nodes'], seed)
                    if strategy == 'module_based':
                        mod_input_nodes, mod_output_nodes = mod_selector.module_based_selection(
                            modular.get_module_assignments()
                        )
                    else:
                        mod_input_nodes, mod_output_nodes = getattr(mod_selector, f"{strategy}_selection")()
                    
                    # Run tasks
                    for task in tqdm(self.config['tasks'], desc="Tasks", leave=False):
                        # Generate task data
                        X, y = getattr(self.task_generator, f"generate_{task}_task")()
                        # Evaluate performance for both networks
                        for topology, graph, input_nodes, output_nodes in [
                            ('small_world', sw_graph, sw_input_nodes, sw_output_nodes),
                            ('modular', mod_graph, mod_input_nodes, mod_output_nodes)
                        ]:
                            performance = self._evaluate_network_performance(
                                graph, input_nodes, output_nodes, X, y, task, size, seed, strategy, topology
                            )
                            results.append({
                                'network_size': size,
                                'seed': seed,
                                'strategy': strategy,
                                'task': task,
                                'topology': topology,
                                'performance': performance
                            })
        # Save results
        self._save_results(results)

    def _evaluate_network_performance(self, graph, input_nodes, output_nodes, X, y, task, size, seed, strategy, topology):
        """Evaluate the performance of a network configuration on a specific task."""
        algorithms = ['SAC', 'A2C', 'PPO']
        results = {}
        
        # Calculate network metrics
        network_metrics = {
            'avg_path_length': np.mean([
                nx.shortest_path_length(graph, source=i, target=o)
                for i in input_nodes
                for o in output_nodes
            ]),
            'clustering_coefficient': nx.average_clustering(graph),
            'density': nx.density(graph),
            'avg_degree': np.mean([d for n, d in graph.degree()]),
            'diameter': nx.diameter(graph),
            'avg_shortest_path': nx.average_shortest_path_length(graph)
        }
        
        # Calculate node-level metrics for input/output nodes
        node_metrics = {
            'input_nodes': {
                'degrees': [graph.degree(node) for node in input_nodes],
                'betweenness_centrality': list(nx.betweenness_centrality(graph, k=min(100, size)).values()),
                'closeness_centrality': list(nx.closeness_centrality(graph).values()),
                'eigenvector_centrality': list(nx.eigenvector_centrality(graph, max_iter=1000).values()),
                'pagerank': list(nx.pagerank(graph).values())
            },
            'output_nodes': {
                'degrees': [graph.degree(node) for node in output_nodes],
                'betweenness_centrality': list(nx.betweenness_centrality(graph, k=min(100, size)).values()),
                'closeness_centrality': list(nx.closeness_centrality(graph).values()),
                'eigenvector_centrality': list(nx.eigenvector_centrality(graph, max_iter=1000).values()),
                'pagerank': list(nx.pagerank(graph).values())
            }
        }
        
        # Add module membership for modular topology
        if topology == 'modular':
            from ..topologies.modular import ModularTopology
            modular = ModularTopology(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                intra_module_prob=self.config['modular_params']['intra_module_prob'],
                seed=seed
            )
            module_assignments = modular.get_module_assignments()
            node_metrics['input_nodes']['module_membership'] = [module_assignments[node] for node in input_nodes]
            node_metrics['output_nodes']['module_membership'] = [module_assignments[node] for node in output_nodes]
        
        # Calculate node selection strategy effectiveness
        strategy_metrics = {
            'input_output_distance': np.mean([
                nx.shortest_path_length(graph, source=i, target=o)
                for i in input_nodes
                for o in output_nodes
            ]),
            'input_node_centrality': np.mean(node_metrics['input_nodes']['betweenness_centrality']),
            'output_node_centrality': np.mean(node_metrics['output_nodes']['betweenness_centrality']),
            'input_output_centrality_diff': np.mean(node_metrics['input_nodes']['betweenness_centrality']) - 
                                          np.mean(node_metrics['output_nodes']['betweenness_centrality'])
        }
        
        for algo in algorithms:
            print(f"[{topology.upper()}] size={size} seed={seed} strategy={strategy} task={task} algo={algo}")
            
            # Simulate training time and convergence
            start_time = time.time()
            time.sleep(0.1)  # Simulate training time
            training_time = time.time() - start_time
            
            # Simulate convergence metrics
            convergence_metrics = {
                'training_time': training_time,
                'iterations_to_converge': np.random.randint(100, 1000),
                'final_loss': np.random.uniform(0.1, 0.5)
            }
            
            # Calculate task-specific metrics
            task_metrics = {}
            if task == 'classification':
                # Simulate predictions (replace with actual model predictions)
                y_pred = np.random.randint(0, 2, size=len(y))
                task_metrics.update({
                    'accuracy': accuracy_score(y, y_pred),
                    'f1_score': f1_score(y, y_pred, average='weighted'),
                    'precision': precision_score(y, y_pred, average='weighted'),
                    'recall': recall_score(y, y_pred, average='weighted')
                })
            elif task == 'regression':
                # Simulate predictions (replace with actual model predictions)
                y_pred = np.random.normal(np.mean(y), np.std(y), size=len(y))
                task_metrics.update({
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2_score': r2_score(y, y_pred)
                })
            elif task == 'clustering':
                # Simulate cluster assignments (replace with actual clustering)
                clusters = np.random.randint(0, 3, size=len(X))
                task_metrics.update({
                    'silhouette_score': silhouette_score(X, clusters),
                    'davies_bouldin_score': davies_bouldin_score(X, clusters),
                    'calinski_harabasz_score': calinski_harabasz_score(X, clusters)
                })
            
            # Store all metrics
            results[algo] = {
                'input_nodes': input_nodes,
                'output_nodes': output_nodes,
                'network_metrics': network_metrics,
                'node_metrics': node_metrics,
                'strategy_metrics': strategy_metrics,
                'task_metrics': task_metrics,
                'convergence_metrics': convergence_metrics
            }
            
        return results
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to files."""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create a subfolder with the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(results_dir / 'experiment_results.csv', index=False)
        
        # Convert all numpy types to native Python types for JSON
        def to_serializable(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_serializable(i) for i in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        serializable_results = to_serializable(results)
        # Save as JSON
        with open(results_dir / 'experiment_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2) 