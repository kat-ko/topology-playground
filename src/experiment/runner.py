import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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
                    print(f"\nProcessing strategy: {strategy}")
                    # Small-world network
                    sw_selector = NodeSelector(sw_graph, self.config['num_io_nodes'], seed)
                    if strategy == 'module_based':
                        print("Using random_selection for small-world network (module_based not applicable)")
                        sw_input_nodes, sw_output_nodes = sw_selector.random_selection()
                    else:
                        print(f"Using {strategy}_selection for small-world network")
                        sw_input_nodes, sw_output_nodes = getattr(sw_selector, f"{strategy}_selection")()
                    
                    # Modular network
                    mod_selector = NodeSelector(mod_graph, self.config['num_io_nodes'], seed)
                    if strategy == 'module_based':
                        print("Using module_based_selection for modular network")
                        mod_input_nodes, mod_output_nodes = mod_selector.module_based_selection(
                            modular.get_module_assignments()
                        )
                    else:
                        print(f"Using {strategy}_selection for modular network")
                        mod_input_nodes, mod_output_nodes = getattr(mod_selector, f"{strategy}_selection")()
                    
                    # Run tasks
                    for task in tqdm(self.config['tasks'], desc="Tasks", leave=False):
                        # Generate task data
                        X, y = getattr(self.task_generator, f"generate_{task}_task")()
                        
                        # Evaluate performance for both networks
                        sw_performance = self._evaluate_network_performance(
                            sw_graph, sw_input_nodes, sw_output_nodes, X, y, task
                        )
                        mod_performance = self._evaluate_network_performance(
                            mod_graph, mod_input_nodes, mod_output_nodes, X, y, task
                        )
                        
                        # Store results
                        results.append({
                            'network_size': size,
                            'seed': seed,
                            'strategy': strategy,
                            'task': task,
                            'topology': 'small_world',
                            'performance': sw_performance
                        })
                        results.append({
                            'network_size': size,
                            'seed': seed,
                            'strategy': strategy,
                            'task': task,
                            'topology': 'modular',
                            'performance': mod_performance
                        })
        
        # Save results
        self._save_results(results)
        
    def _evaluate_network_performance(self, graph, input_nodes, output_nodes, X, y, task):
        """Evaluate the performance of a network configuration on a specific task."""
        # Here you would implement the actual task execution using the network
        # For now, we'll return a dummy performance metric
        return {
            'input_nodes': input_nodes,
            'output_nodes': output_nodes,
            'avg_path_length': np.mean([
                nx.shortest_path_length(graph, source=i, target=o)
                for i in input_nodes
                for o in output_nodes
            ])
        }
    
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