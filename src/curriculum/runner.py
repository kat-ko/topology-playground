import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import logging
from ..utils.logging_utils import setup_logger, LogLevel
from collections import defaultdict

from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..topologies.hybrid import HybridTopology
from ..topologies.fully_connected import FullyConnectedTopology
from ..networks.ffn import FeedForwardNetwork
from ..networks.rnn import RecurrentNetwork
from ..node_selection.strategies import NodeSelector
from ..tasks.task_definitions import TaskGenerator, TaskEvaluator
from ..tasks.rl_tasks import RLTaskGenerator, RLTaskEvaluator
from ..agents.network_agent import NetworkAgent

logger = setup_logger(__name__)

class CurriculumRunner:
    """Runner for curriculum learning experiments."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "results"):
        """
        Initialize the curriculum runner.
        
        Args:
            config: Curriculum configuration
            output_dir: Directory to save experiment results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.task_generator = RLTaskGenerator()
        self.task_evaluator = RLTaskEvaluator()
        
        # Network type mapping
        self.network_types = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
    
    def run_curriculum(self):
        """Run the complete curriculum experiment."""
        print("Executing run_curriculum from:", __file__)
        print("\nCurriculum parameters:")
        print(f"Task sequence: {self.config['task_sequence']}")
        print(f"Network sizes: {self.config['network_sizes']}")
        print(f"Seeds: {self.config['seeds']}")
        print(f"Number of layers: {self.config['num_layers']}")
        print(f"Network types: {self.config['network_types']}")
        
        # Print parameter budget information
        print("\nParameter budget settings:")
        print(f"Budget type: {self.config['parameter_budget']['budget_type']}")
        print(f"Target budget: {self.config['parameter_budget']['target_budget']}")
        print(f"Normalize by size: {self.config['parameter_budget']['normalize_by_size']}")
        
        # Print experiment types and their capacity matching
        print("\nExperiment types and capacity matching:")
        for exp_type in self.config['experiment_types']:
            print(f"\n{exp_type}:")
            if exp_type == 'same_size':
                print("  All networks will have the same node size")
            else:
                target = '_'.join(exp_type.split('_')[1:])
                print(f"  All networks will match {target} capacity")
        
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
                        
                        fully_connected = FullyConnectedTopology(
                            size=size,
                            num_layers=num_layers,
                            inter_layer_prob=self.config['fully_connected_params']['inter_layer_prob'],
                            intra_layer_prob=self.config['fully_connected_params']['intra_layer_prob'],
                            seed=seed
                        )
                        
                        # Generate networks
                        sw_graphs = small_world.generate(num_layers)
                        mod_graphs = modular.generate(num_layers)
                        hybrid_graphs = hybrid.generate(num_layers)
                        fc_graphs = fully_connected.generate(num_layers)
                        
                        # Convert to list if single graph
                        if num_layers == 1:
                            sw_graphs = [sw_graphs]
                            mod_graphs = [mod_graphs]
                            hybrid_graphs = [hybrid_graphs]
                            fc_graphs = [fc_graphs]
                        
                        # Select input/output nodes for each layer
                        for strategy in tqdm(self.config['node_selection_strategies'], desc="Strategies", leave=False):
                            sw_input_nodes = []
                            sw_output_nodes = []
                            mod_input_nodes = []
                            mod_output_nodes = []
                            hybrid_input_nodes = []
                            hybrid_output_nodes = []
                            fc_input_nodes = []
                            fc_output_nodes = []
                            
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
                                fc_input, fc_output = self._select_nodes(
                                    fc_graphs[layer_idx], strategy, size, seed
                                )
                                
                                sw_input_nodes.append(sw_input)
                                sw_output_nodes.append(sw_output)
                                mod_input_nodes.append(mod_input)
                                mod_output_nodes.append(mod_output)
                                hybrid_input_nodes.append(hybrid_input)
                                hybrid_output_nodes.append(hybrid_output)
                                fc_input_nodes.append(fc_input)
                                fc_output_nodes.append(fc_output)
                            
                            # Create networks
                            sw_networks = []
                            mod_networks = []
                            hybrid_networks = []
                            fc_networks = []
                            
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
                                
                                fc_networks.append(network_class(
                                    fc_graphs[layer_idx],
                                    fc_input_nodes[layer_idx],
                                    fc_output_nodes[layer_idx],
                                    network_params
                                ))
                            
                            # Run curriculum for each topology
                            for topology, networks, input_nodes, output_nodes in [
                                ('small_world', sw_networks, sw_input_nodes, sw_output_nodes),
                                ('modular', mod_networks, mod_input_nodes, mod_output_nodes),
                                ('hybrid', hybrid_networks, hybrid_input_nodes, hybrid_output_nodes),
                                ('fully_connected', fc_networks, fc_input_nodes, fc_output_nodes)
                            ]:
                                curriculum_results = self._run_task_sequence(
                                    networks, input_nodes, output_nodes,
                                    size, seed, strategy, topology, network_type
                                )
                                results.append({
                                    'network_size': size,
                                    'seed': seed,
                                    'num_layers': num_layers,
                                    'network_type': network_type,
                                    'strategy': strategy,
                                    'topology': topology,
                                    'curriculum_results': curriculum_results
                                })
        
        # Save results
        self._save_results(results)
    
    def _run_task_sequence(self, networks, input_nodes, output_nodes,
                          size, seed, strategy, topology, network_type):
        """Run the sequence of tasks."""
        performance_history = {}
        baseline_performance = {}
        
        # Get baseline performance
        for task in self.config['task_sequence']:
            env, task_config = getattr(self.task_generator, f"generate_{task}_task")()
            baseline = self._evaluate_performance(networks, input_nodes, output_nodes,
                                                env, task_config, task, size, seed,
                                                strategy, topology, network_type)
            baseline_performance[task] = baseline
        
        # Run curriculum
        for task in self.config['task_sequence']:
            # Train on current task
            env, task_config = getattr(self.task_generator, f"generate_{task}_task")()
            task_metrics = self._train_task(networks, input_nodes, output_nodes,
                           env, task_config, task, size, seed,
                           strategy, topology, network_type)
            
            # Evaluate on all tasks
            current_performance = {}
            for eval_task in self.config['task_sequence']:
                eval_env, eval_config = getattr(self.task_generator, f"generate_{eval_task}_task")()
                performance = self._evaluate_performance(networks, input_nodes, output_nodes,
                                                       eval_env, eval_config, eval_task,
                                                       size, seed, strategy, topology,
                                                       network_type)
                current_performance[eval_task] = performance
            
            performance_history[task] = current_performance
        
        # Calculate transfer metrics
        transfer_metrics = self._calculate_transfer_metrics(
            baseline_performance,
            performance_history
        )
        
        return {
            'performance_history': performance_history,
            'transfer_metrics': transfer_metrics,
            'task_metrics': task_metrics
        }
    
    def _train_task(self, networks, input_nodes, output_nodes,
                   env, task_config, task, size, seed,
                   strategy, topology, network_type):
        """Train networks on a specific task."""
        print(f"Training on {task} with {topology} topology")
        
        # Simulate training (replace with actual training)
        time.sleep(0.1)  # Simulate training time
        
        # Update network metrics
        for layer_idx, network in enumerate(networks):
            metrics = network.get_network_metrics()
            logger.info(f"Layer {layer_idx} - Training metrics: {metrics}")
        
        # Save training metrics under task_metrics
        task_metrics = {
            'training_metrics': metrics
        }
        return task_metrics
    
    def _evaluate_performance(self, networks, input_nodes, output_nodes,
                            env, task_config, task, size, seed,
                            strategy, topology, network_type):
        """Evaluate network performance on a specific task using real agent and environment."""
        from ..tasks.rl_tasks import RLTaskEvaluator
        
        # If networks is a list, evaluate each network and aggregate results
        if isinstance(networks, list):
            results = []
            for network in networks:
                # Wrap network in agent
                agent = NetworkAgent(network, task_config)
                # Evaluate single network
                result = RLTaskEvaluator.evaluate_episodes(env, agent, task_config, num_episodes=10)
                results.append(result)
            
            # Aggregate results
            mean_rewards = [r['mean_reward'] for r in results]
            std_rewards = [r['std_reward'] for r in results]
            mean_lengths = [r['mean_length'] for r in results]
            std_lengths = [r['std_length'] for r in results]
            solved_rates = [r['solved_rate'] for r in results]
            
            return {
                'mean_reward': np.mean(mean_rewards),
                'std_reward': np.std(mean_rewards),  # Std of means across networks
                'mean_length': np.mean(mean_lengths),
                'std_length': np.std(mean_lengths),
                'solved_rate': np.mean(solved_rates)
            }
        else:
            # Single network case - wrap in agent and evaluate
            agent = NetworkAgent(networks, task_config)
            return RLTaskEvaluator.evaluate_episodes(env, agent, task_config, num_episodes=10)
    
    def _calculate_transfer_metrics(self, baseline_performance, performance_history):
        """Calculate transfer learning metrics."""
        transfer_metrics = {
            'backward_transfer': {},
            'forward_transfer': {},
            'final_performance': {}
        }
        
        # Calculate backward transfer (how well previous tasks are retained)
        for task in self.config['backward_transfer_tasks']:
            final_performance = performance_history[self.config['task_sequence'][-1]][task]
            baseline = baseline_performance[task]
            
            # For CartPole, higher is better
            if task == 'cartpole':
                transfer_metrics['backward_transfer'][task] = (
                    final_performance['mean_reward'] / baseline['mean_reward']
                )
            # For MountainCar and Acrobot, lower is better
            else:
                transfer_metrics['backward_transfer'][task] = (
                    baseline['mean_reward'] / final_performance['mean_reward']
                )
        
        # Calculate forward transfer (how quickly new tasks are learned)
        for task in self.config['forward_transfer_tasks']:
            task_idx = self.config['task_sequence'].index(task)
            if task_idx > 0:
                # Compare performance after previous task vs baseline
                after_prev = performance_history[self.config['task_sequence'][task_idx-1]][task]
                baseline = baseline_performance[task]
                
                # For CartPole, higher is better
                if task == 'cartpole':
                    transfer_metrics['forward_transfer'][task] = (
                        after_prev['mean_reward'] / baseline['mean_reward']
                    )
                # For MountainCar and Acrobot, lower is better
                else:
                    transfer_metrics['forward_transfer'][task] = (
                        baseline['mean_reward'] / after_prev['mean_reward']
                    )
        
        # Store final performance
        final_task = self.config['task_sequence'][-1]
        transfer_metrics['final_performance'] = performance_history[final_task]
        
        return transfer_metrics
    
    def _select_nodes(self, graph: nx.Graph, strategy: str, size: int, seed: int) -> tuple:
        """Select input and output nodes based on the specified strategy."""
        rng = np.random.RandomState(seed)
        num_io_nodes = self.config['num_io_nodes']
        
        if strategy == 'random':
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
            
        elif strategy == 'centrality_based':
            centrality = nx.betweenness_centrality(graph, k=min(100, size))
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            input_nodes = [node for node, _ in sorted_nodes[:num_io_nodes]]
            output_nodes = [node for node, _ in sorted_nodes[num_io_nodes:2*num_io_nodes]]
            
        elif strategy == 'distance_based':
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            input_nodes = all_nodes[:num_io_nodes]
            
            distances = np.zeros(size)
            for node in range(size):
                distances[node] = np.mean([
                    nx.shortest_path_length(graph, source=in_node, target=node)
                    for in_node in input_nodes
                ])
            
            sorted_nodes = np.argsort(distances)[::-1]
            output_nodes = [node for node in sorted_nodes if node not in input_nodes][:num_io_nodes]
            
        elif strategy == 'module_based':
            if hasattr(self, 'modular') and hasattr(self.modular, 'get_module_assignments'):
                module_assignments = self.modular.get_module_assignments()
                module_nodes = {}
                for node, module in module_assignments.items():
                    if module not in module_nodes:
                        module_nodes[module] = []
                    module_nodes[module].append(node)
                
                input_nodes = []
                for module in range(min(num_io_nodes, len(module_nodes))):
                    module_node = rng.choice(module_nodes[module])
                    input_nodes.append(module_node)
                
                input_modules = {module_assignments[node] for node in input_nodes}
                available_modules = [m for m in module_nodes.keys() if m not in input_modules]
                output_nodes = []
                for module in available_modules[:num_io_nodes]:
                    module_node = rng.choice(module_nodes[module])
                    output_nodes.append(module_node)
            else:
                all_nodes = list(range(size))
                rng.shuffle(all_nodes)
                input_nodes = all_nodes[:num_io_nodes]
                output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
        else:
            raise ValueError(f"Unknown node selection strategy: {strategy}")
        
        return input_nodes, output_nodes
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save experiment results to files."""
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create a subfolder with the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = self.output_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        df.to_csv(results_dir / 'curriculum_results.csv', index=False)
        
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
        with open(results_dir / 'curriculum_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)

    def _run_single_experiment(self, algorithm: str, experiment_type: str):
        """Run a single curriculum experiment."""
        # Create agent with capacity-matched network
        agent_config = self.config.copy()
        agent_config['experiment_type'] = experiment_type
        
        # Create network with correct capacity
        network = self.parameter_budget.calculator.create_network(
            topology=self.config['network_types'][0],
            size=self.config['network_sizes'][0],
            experiment_type=experiment_type
        )
        
        # Initialize agent with the capacity-matched network
        agent = self.agents[algorithm]
        agent.network = network
        
        task_generator = RLTaskGenerator(self.config)
        evaluator = RLTaskEvaluator(self.config)
        
        # Initialize metrics for this experiment
        exp_key = f"{algorithm}_{experiment_type}"
        self.metrics[exp_key] = defaultdict(list)
        
        # Run curriculum
        for task in self.config['task_sequence']:
            # Generate task
            env, task_config = task_generator.generate_task(task)
            
            # Set environment for agent
            agent.model.set_env(env)
            
            # Initialize environment step counter
            env_steps = 0
            env_steps_so_far = 0
            
            # Train agent
            while env_steps_so_far < self.config['max_env_steps_per_task']:
                # Calculate remaining steps
                remaining_steps = self.config['max_env_steps_per_task'] - env_steps_so_far
                steps_to_train = min(remaining_steps, 2048)  # SB3's default n_steps
                
                # Train for steps_to_train steps
                agent.model.learn(total_timesteps=steps_to_train)
                env_steps_so_far += steps_to_train
                
                # Evaluate current performance
                eval_rewards = []
                for _ in range(self.config['evaluation_episodes']):
                    obs = env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        action, _ = agent.model.predict(obs, deterministic=True)
                        obs, reward, done, _ = env.step(action)
                        episode_reward += reward
                    
                    eval_rewards.append(episode_reward)
                
                # Log evaluation metrics
                mean_reward = np.mean(eval_rewards)
                self.metrics[exp_key]['eval_rewards'].append(mean_reward)
                self.metrics[exp_key]['env_steps_so_far'].append(env_steps_so_far)
                
                # Check if task is learned using task-specific threshold
                if mean_reward >= self.config['task_memory_thresholds'][task]:
                    self.metrics[exp_key]['learning_episodes'].append(env_steps_so_far)
                    # Log physics parameters when task is learned
                    if hasattr(env, 'get_physics_params'):
                        physics_params = env.get_physics_params()
                        self.metrics[exp_key]['physics_params'].append({
                            'task': task,
                            'steps': env_steps_so_far,
                            'params': physics_params
                        })
                    break
            
            # Get parameter budget stats
            budget_stats = self.parameter_budget.get_budget_stats(
                agent.network,
                self.config['network_sizes'][0],
                agent.topology
            )
            self.metrics[exp_key]['parameter_stats'].append(budget_stats)
            
            # Test transfer learning if applicable
            if task in self.config['backward_transfer_tasks']:
                self._test_transfer_learning(agent, task, exp_key)
            
            # Test forgetting and retention
            self._test_forgetting_retention(agent, task, exp_key) 