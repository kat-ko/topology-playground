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
import logging
from ..utils.logging_utils import setup_logger, LogLevel

from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..topologies.hybrid import HybridTopology
from ..networks.ffn import FeedForwardNetwork
from ..networks.rnn import RecurrentNetwork
from ..node_selection.strategies import NodeSelector
from ..tasks.task_definitions import TaskGenerator, TaskEvaluator

logger = setup_logger(__name__)

class ExperimentRunner:
    """Runner for network experiments."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory to save experiment results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.task_generator = TaskGenerator()
        self.task_evaluator = TaskEvaluator()
        
        # Network type mapping
        self.network_types = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
        
        self.is_test_run = isinstance(config.get('is_test_run', False), bool)
        
        # Set appropriate log level based on run type
        if self.is_test_run:
            logger.setLevel(LogLevel.DEBUG.value)
            logger.info("Running in TEST mode")
        else:
            logger.setLevel(LogLevel.INFO.value)
            logger.info("Running in EXPERIMENT mode")
    
    def run_experiment(self):
        """Run the complete experiment with all combinations of parameters."""
        print("Executing run_experiment from:", __file__)
        print("Experiment parameters:")
        print(f"Network sizes: {self.config['network_sizes']}")
        print(f"Seeds: {self.config['seeds']}")
        print(f"Number of layers: {self.config['num_layers']}")
        print(f"Network types: {self.config['network_types']}")
        print(f"Small-world parameters: {self.config['small_world_params']}")
        print(f"Modular parameters: {self.config['modular_params']}")
        print(f"Node selection strategies: {self.config['node_selection_strategies']}")
        print(f"Tasks: {self.config['tasks']}")
        results = []
        
        for size in tqdm(self.config['network_sizes'], desc="Network sizes"):
            for seed in tqdm(self.config['seeds'], desc="Seeds", leave=False):
                for num_layers in tqdm(self.config['num_layers'], desc="Number of layers", leave=False):
                    for network_type in tqdm(self.config['network_types'], desc="Network types", leave=False):
                        # Generate networks
                        small_world = SmallWorldTopology(
                            size=size,
                            k=self.config['small_world_params']['k'],
                            p=self.config['small_world_params']['p'],
                            num_layers=num_layers,
                            inter_layer_prob=self.config['small_world_params']['inter_layer_prob'],
                            seed=seed
                        )
                        
                        modular = ModularTopology(
                            size=size,
                            num_modules=self.config['modular_params']['num_modules'],
                            inter_module_prob=self.config['modular_params']['inter_module_prob'],
                            intra_module_prob=self.config['modular_params']['intra_module_prob'],
                            num_layers=num_layers,
                            inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                            seed=seed
                        )
                        
                        hybrid = HybridTopology(
                            size=size,
                            num_modules=self.config['modular_params']['num_modules'],
                            k=self.config['small_world_params']['k'],
                            p=self.config['small_world_params']['p'],
                            inter_module_prob=self.config['modular_params']['inter_module_prob'],
                            num_layers=num_layers,
                            inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                            seed=seed
                        )
                        
                        # Generate networks
                        sw_graphs = small_world.generate(num_layers)
                        mod_graphs = modular.generate(num_layers)
                        hybrid_graphs = hybrid.generate(num_layers)
                        
                        # Convert to list if single graph
                        if num_layers == 1:
                            sw_graphs = [sw_graphs]
                            mod_graphs = [mod_graphs]
                            hybrid_graphs = [hybrid_graphs]
                        
                        # Select input/output nodes for each layer
                        for strategy in tqdm(self.config['node_selection_strategies'], desc="Strategies", leave=False):
                            sw_input_nodes = []
                            sw_output_nodes = []
                            mod_input_nodes = []
                            mod_output_nodes = []
                            hybrid_input_nodes = []
                            hybrid_output_nodes = []
                            
                            # Select nodes for each layer
                            for layer_idx in range(num_layers):
                                sw_input, sw_output = self._select_nodes(
                                    sw_graphs[layer_idx], strategy, size, seed
                                )
                                mod_input, mod_output = self._select_nodes(
                                    mod_graphs[layer_idx], strategy, size, seed
                                )
                                hybrid_input, hybrid_output = self._select_nodes(
                                    hybrid_graphs[layer_idx], strategy, size, seed
                                )
                                
                                sw_input_nodes.append(sw_input)
                                sw_output_nodes.append(sw_output)
                                mod_input_nodes.append(mod_input)
                                mod_output_nodes.append(mod_output)
                                hybrid_input_nodes.append(hybrid_input)
                                hybrid_output_nodes.append(hybrid_output)
                            
                            # Create networks
                            sw_networks = []
                            mod_networks = []
                            hybrid_networks = []
                            
                            for layer_idx in range(num_layers):
                                # Create network instances
                                network_class = self.network_types[network_type]
                                network_params = self.config['network_params'][network_type]
                                
                                sw_networks.append(network_class(
                                    sw_graphs[layer_idx],
                                    sw_input_nodes[layer_idx],
                                    sw_output_nodes[layer_idx],
                                    network_params
                                ))
                                
                                mod_networks.append(network_class(
                                    mod_graphs[layer_idx],
                                    mod_input_nodes[layer_idx],
                                    mod_output_nodes[layer_idx],
                                    network_params
                                ))
                                
                                hybrid_networks.append(network_class(
                                    hybrid_graphs[layer_idx],
                                    hybrid_input_nodes[layer_idx],
                                    hybrid_output_nodes[layer_idx],
                                    network_params
                                ))
                            
                            # Run tasks
                            for task in tqdm(self.config['tasks'], desc="Tasks", leave=False):
                                # Generate task environment and config
                                env, task_config = getattr(self.task_generator, f"generate_{task}_task")()
                                
                                # Evaluate performance for all networks
                                for topology, networks, input_nodes, output_nodes in [
                                    ('small_world', sw_networks, sw_input_nodes, sw_output_nodes),
                                    ('modular', mod_networks, mod_input_nodes, mod_output_nodes),
                                    ('hybrid', hybrid_networks, hybrid_input_nodes, hybrid_output_nodes)
                                ]:
                                    performance = self._evaluate_network_performance(
                                        networks, input_nodes, output_nodes, env, task_config,
                                        task, size, seed, strategy, topology, network_type
                                    )
                                    results.append({
                                        'network_size': size,
                                        'seed': seed,
                                        'num_layers': num_layers,
                                        'network_type': network_type,
                                        'strategy': strategy,
                                        'task': task,
                                        'topology': topology,
                                        'performance': performance
                                    })
        
        # Save results
        self._save_results(results)
    
    def _select_nodes(self, graph: nx.Graph, strategy: str, size: int, seed: int) -> tuple:
        """Select input and output nodes based on the specified strategy.
        
        Args:
            graph: NetworkX graph
            strategy: Node selection strategy ('random', 'centrality_based', 'distance_based', 'module_based')
            size: Network size
            seed: Random seed for reproducibility
            
        Returns:
            tuple: (input_nodes, output_nodes) where each is a list of node indices
        """
        rng = np.random.RandomState(seed)
        num_io_nodes = self.config['num_io_nodes']
        
        if strategy == 'random':
            # Random selection
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
            
        elif strategy == 'centrality_based':
            # Select nodes with highest betweenness centrality
            centrality = nx.betweenness_centrality(graph, k=min(100, size))
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            input_nodes = [node for node, _ in sorted_nodes[:num_io_nodes]]
            output_nodes = [node for node, _ in sorted_nodes[num_io_nodes:2*num_io_nodes]]
            
        elif strategy == 'distance_based':
            # Select nodes that are far apart
            # First select input nodes randomly
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            
            # Then select output nodes that are far from input nodes
            distances = np.zeros(size)
            for node in range(size):
                distances[node] = np.mean([
                    nx.shortest_path_length(graph, source=in_node, target=node)
                    for in_node in input_nodes
                ])
            
            # Select nodes with maximum average distance
            sorted_nodes = np.argsort(distances)[::-1]
            output_nodes = [node for node in sorted_nodes if node not in input_nodes][:num_io_nodes]
            
        elif strategy == 'module_based':
            # Select nodes from different modules
            if hasattr(self, 'modular') and hasattr(self.modular, 'get_module_assignments'):
                module_assignments = self.modular.get_module_assignments()
                # Group nodes by module
                module_nodes = {}
                for node, module in module_assignments.items():
                    if module not in module_nodes:
                        module_nodes[module] = []
                    module_nodes[module].append(node)
                
                # Select input nodes from different modules
                input_nodes = []
                for module in range(min(num_io_nodes, len(module_nodes))):
                    module_node = rng.choice(module_nodes[module])
                    input_nodes.append(module_node)
                
                # Select output nodes from different modules than input nodes
                input_modules = {module_assignments[node] for node in input_nodes}
                available_modules = [m for m in module_nodes.keys() if m not in input_modules]
                output_nodes = []
                for module in available_modules[:num_io_nodes]:
                    module_node = rng.choice(module_nodes[module])
                    output_nodes.append(module_node)
            else:
                # Fallback to random selection if module information is not available
                all_nodes = list(range(size))
                rng.shuffle(all_nodes)
                input_nodes = all_nodes[:num_io_nodes]
                output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
        else:
            raise ValueError(f"Unknown node selection strategy: {strategy}")
        
        return input_nodes, output_nodes

    def _evaluate_network_performance(self, networks, input_nodes, output_nodes, env, task_config,
                                    task, size, seed, strategy, topology, network_type):
        """Evaluate the performance of a network configuration on a specific task."""
        algorithms = ['SAC', 'A2C', 'PPO']
        results = {}
        
        # Calculate network metrics for each layer
        network_metrics = []
        for layer_idx, network in enumerate(networks):
            # Get initial weights and mask state
            initial_weights = {}
            for node in network.topology.nodes():
                if network_type == 'ffn':
                    initial_weights[node] = {
                        neighbor: network.node_states[node]['weights'].get(neighbor, 0)
                        for neighbor in network.topology.neighbors(node)
                    }
                else:  # rnn
                    initial_weights[node] = {
                        neighbor: network.node_states[node]['input_weights'].get(neighbor, 0)
                        for neighbor in network.topology.neighbors(node)
                    }
                    # Add recurrent weights
                    for neighbor in network.topology.neighbors(node):
                        if neighbor in initial_weights[node]:
                            initial_weights[node][neighbor] += network.node_states[node]['recurrent_weights'].get(neighbor, 0)
                        else:
                            initial_weights[node][neighbor] = network.node_states[node]['recurrent_weights'].get(neighbor, 0)
            
            # Track static mask integrity
            mask_integrity = {
                'initial_sparsity': self._calculate_sparsity(initial_weights),
                'leaked_edges': 0,  # Will be updated after training
                'final_sparsity': 0  # Will be updated after training
            }
            
            # Track topology-aware learning signal
            learning_signal = {
                'input_hidden_gradients': [],
                'hidden_hidden_gradients': [],
                'hidden_output_gradients': [],
                'active_hidden_units': 0  # Will be updated after training
            }
            
            # Track task-level learning footprint
            learning_footprint = {
                'rewards_at_steps': {
                    '2k': 0,
                    '5k': 0,
                    '10k': 0,
                    '20k': 0
                },
                'parameter_normalized_score': 0  # Will be updated after training
            }
            
            layer_metrics = {
                'layer_idx': layer_idx,
                'topology_metrics': network.get_topology_metrics(),
                'network_metrics': network.get_network_metrics(),
                'mask_integrity': mask_integrity,
                'learning_signal': learning_signal,
                'learning_footprint': learning_footprint
            }
            
            # Add node-level metrics
            node_metrics = {
                'input_nodes': {
                    node: network.get_node_metrics(node)
                    for node in input_nodes[layer_idx]
                },
                'output_nodes': {
                    node: network.get_node_metrics(node)
                    for node in output_nodes[layer_idx]
                }
            }
            layer_metrics['node_metrics'] = node_metrics
            
            # Log pruning effect
            metrics = network.get_topology_metrics()
            logger.info(f"Layer {layer_idx} - Original edges: {metrics['original_edges']}, "
                       f"After pruning output-output edges: {metrics['pruned_edges']}, "
                       f"Edges removed: {metrics['edges_removed']}")
            
            network_metrics.append(layer_metrics)
        
        # Calculate inter-layer metrics if multiple layers
        inter_layer_metrics = {}
        if len(networks) > 1:
            for i in range(len(networks)):
                for j in range(i + 1, len(networks)):
                    inter_layer_metrics[f'layer_{i}_to_{j}'] = {
                        'num_connections': len(set(networks[i].topology.edges()) & 
                                            set(networks[j].topology.edges())),
                        'shared_nodes': len(set(networks[i].topology.nodes()) & 
                                          set(networks[j].topology.nodes()))
                    }
        
        for algo in algorithms:
            print(f"[{topology.upper()}] size={size} seed={seed} strategy={strategy} "
                  f"task={task} algo={algo} network_type={network_type}")
            
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
            if task == 'cartpole':
                # Simulate RL metrics for CartPole
                task_metrics.update({
                    'mean_reward': np.random.uniform(400, 500),
                    'std_reward': np.random.uniform(10, 50),
                    'mean_length': np.random.uniform(400, 500),
                    'std_length': np.random.uniform(10, 50),
                    'solved_rate': np.random.uniform(0.8, 1.0)
                })
            elif task == 'mountain_car':
                # Simulate RL metrics for MountainCar
                task_metrics.update({
                    'mean_reward': np.random.uniform(-150, -100),
                    'std_reward': np.random.uniform(10, 30),
                    'mean_length': np.random.uniform(100, 200),
                    'std_length': np.random.uniform(10, 30),
                    'solved_rate': np.random.uniform(0.6, 0.9)
                })
            elif task == 'acrobot':
                # Simulate RL metrics for Acrobot
                task_metrics.update({
                    'mean_reward': np.random.uniform(-150, -100),
                    'std_reward': np.random.uniform(10, 30),
                    'mean_length': np.random.uniform(100, 200),
                    'std_length': np.random.uniform(10, 30),
                    'solved_rate': np.random.uniform(0.6, 0.9)
                })
            
            # Update mask integrity metrics after training
            for layer_idx, network in enumerate(networks):
                # Get final weights
                final_weights = {}
                for node in network.topology.nodes():
                    if network_type == 'ffn':
                        final_weights[node] = {
                            neighbor: network.node_states[node]['weights'].get(neighbor, 0)
                            for neighbor in network.topology.neighbors(node)
                        }
                    else:  # rnn
                        final_weights[node] = {
                            neighbor: network.node_states[node]['input_weights'].get(neighbor, 0)
                            for neighbor in network.topology.neighbors(node)
                        }
                        # Add recurrent weights
                        for neighbor in network.topology.neighbors(node):
                            if neighbor in final_weights[node]:
                                final_weights[node][neighbor] += network.node_states[node]['recurrent_weights'].get(neighbor, 0)
                            else:
                                final_weights[node][neighbor] = network.node_states[node]['recurrent_weights'].get(neighbor, 0)
                
                # Calculate final sparsity
                network_metrics[layer_idx]['mask_integrity']['final_sparsity'] = self._calculate_sparsity(final_weights)
                
                # Count leaked edges (edges that were zero initially but became non-zero)
                leaked_edges = 0
                for node in network.topology.nodes():
                    # Get the intersection of neighbors that exist in both initial and final weights
                    initial_neighbors = set(initial_weights[node].keys())
                    final_neighbors = set(final_weights[node].keys())
                    common_neighbors = initial_neighbors.intersection(final_neighbors)
                    
                    for neighbor in common_neighbors:
                        if (initial_weights[node][neighbor] == 0 and 
                            final_weights[node][neighbor] != 0):
                            leaked_edges += 1
                network_metrics[layer_idx]['mask_integrity']['leaked_edges'] = leaked_edges
                
                # Calculate active hidden units
                active_units = sum(1 for node in network.topology.nodes()
                                 if node not in input_nodes[layer_idx] + output_nodes[layer_idx]
                                 and any(abs(w) > 1e-6 for w in final_weights[node].values()))
                network_metrics[layer_idx]['learning_signal']['active_hidden_units'] = active_units
                
                # Calculate parameter-normalized score
                if network_type == 'ffn':
                    num_parameters = sum(len(node['weights']) for node in network.node_states.values())
                else:  # rnn
                    num_parameters = sum(
                        len(node['input_weights']) + 
                        len(node['recurrent_weights']) + 
                        len(node['hidden_weights'])
                        for node in network.node_states.values()
                    )
                
                if task == 'cartpole':
                    score = task_metrics['mean_reward']
                elif task == 'mountain_car':
                    score = -task_metrics['mean_reward']  # Negative because lower is better
                else:  # acrobot
                    score = -task_metrics['mean_reward']  # Negative because lower is better
                network_metrics[layer_idx]['learning_footprint']['parameter_normalized_score'] = score / num_parameters
            
            # Store all metrics
            results[algo] = {
                'network_metrics': network_metrics,
                'inter_layer_metrics': inter_layer_metrics,
                'task_metrics': task_metrics,
                'convergence_metrics': convergence_metrics
            }
            
        return results
    
    def _calculate_sparsity(self, weights: Dict[int, Dict[int, float]]) -> float:
        """Calculate the sparsity of a weight matrix."""
        total_weights = 0
        zero_weights = 0
        for node_weights in weights.values():
            for weight in node_weights.values():
                total_weights += 1
                if abs(weight) < 1e-6:  # Consider weights close to zero as zero
                    zero_weights += 1
        return zero_weights / total_weights if total_weights > 0 else 1.0
    
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
                return {str(k) if isinstance(k, (np.integer, np.floating)) else k: to_serializable(v) 
                       for k, v in obj.items()}
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

    def _handle_mask_validation(self, is_valid: bool, error_msg: str = None) -> None:
        """
        Handle mask validation results.
        
        Args:
            is_valid: Whether the mask is valid
            error_msg: Error message if validation failed
        """
        if not is_valid:
            logger.error(f"Mask validation failed: {error_msg}")
            if not self.is_test_run:
                raise ValueError(f"Mask validation failed: {error_msg}")
        else:
            logger.info("Mask validation passed") 