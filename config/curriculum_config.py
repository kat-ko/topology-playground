from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning experiments."""
    
    # Experiment types
    experiment_types: List[str] = field(default_factory=lambda: [
        'same_size',  # All topologies use the same node count (not matched capacities)
        'match_hybrid',  # All topologies matched to hybrid capacity
        'match_small_world',  # All topologies matched to small world capacity
        'match_modular',  # All topologies matched to modular capacity
        'match_fully_connected'  # All topologies matched to fully connected capacity
    ])
    
    # Parameter budget settings
    parameter_budget: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # Whether to enforce parameter budget
        'budget_type': 'edges',  # 'edges' or 'weights'
        'target_budget': 10000,  # Target number of parameters
        'padding_strategy': 'zero',  # How to pad: 'zero' or 'random'
        'normalize_by_size': True,  # Whether to normalize budget by network size
    })
    
    # Task sequence
    task_sequence: List[str] = field(default_factory=lambda: [
        'cartpole',
        'mountain_car',
        'acrobot'
    ])
    
    # Network parameters (reusing from ExperimentConfig)
    network_sizes: List[int] = field(default_factory=lambda: [25, 50, 100])
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    network_types: List[str] = field(default_factory=lambda: ['ffn', 'rnn'])
    
    # Training parameters
    episodes_per_task: int = 2000  # Increased from 1000 for better learning
    evaluation_episodes: int = 100
    max_env_steps_per_task: int = 100000  # Increased from 50000 for proper CartPole solving
    
    # Task memory and difficulty parameters
    task_memory_size: int = 6  # Number of tasks to remember (2 per task type)
    task_memory_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cartpole': 400.0,  # Reward threshold to store task in memory
        'mountain_car': -110.0,  # Reward threshold to store task in memory
        'acrobot': -110.0  # Reward threshold to store task in memory
    })
    difficulty_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cartpole': 450.0,  # Reward threshold to increase difficulty
        'mountain_car': -100.0,  # Reward threshold to increase difficulty
        'acrobot': -100.0  # Reward threshold to increase difficulty
    })
    difficulty_increase: float = 0.1
    
    # Transfer learning parameters
    backward_transfer_tasks: List[str] = field(default_factory=lambda: ['cartpole'])
    forward_transfer_tasks: List[str] = field(default_factory=lambda: ['mountain_car', 'acrobot'])
    
    # Forgetting and retention testing parameters
    forgetting_test: Dict[str, Any] = field(default_factory=lambda: {
        'retention_interval': 10,  # How often to test retention (iterations)
        'retention_episodes': 20,  # Number of episodes for retention testing (increased from 5 to 20)
        'forgetting_threshold': 0.8,  # Performance threshold to consider as forgetting
        'retention_threshold': 0.9  # Performance threshold to consider as retained
    })
    
    # Reuse existing parameters
    network_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'ffn': {
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'rnn': {
            'hidden_size': 32,
            'sequence_length': 10,
            'learning_rate': 0.001,
            'batch_size': 32
        }
    })
    
    # Fully connected network parameters
    fully_connected_params: Dict[str, Any] = field(default_factory=lambda: {
        'inter_layer_prob': 1.0,  # Fully connected between layers
        'intra_layer_prob': 1.0   # Fully connected within layers
    })
    
    small_world_params: Dict[str, Any] = field(default_factory=lambda: {
        'k': 4,
        'p': 0.1,
        'inter_layer_prob': 0.1
    })
    
    modular_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_modules': 4,
        'inter_module_prob': 0.1,
        'intra_module_prob': 0.3,
        'inter_layer_prob': 0.1
    })
    
    # Node selection strategies
    node_selection_strategies: List[str] = field(default_factory=lambda: [
        'random',
        'centrality_based',
        'distance_based',
        'module_based'
    ])
    
    # Number of input/output nodes
    num_io_nodes: int = 5
    
    # RL-specific parameters
    rl_params: Dict[str, Any] = field(default_factory=lambda: {
        'ppo': {
            'learning_rate': 0.0002,  # Balanced learning rate for stability and learning
            'gamma': 0.99,
            'clip_ratio': 0.15,  # Balanced clip range for learning
            'batch_size': 64,
            'max_episode_steps': 500,  # Increased for CartPole (solves at 475+ steps)
            'n_epochs': 4,  # Reduced from 10 to prevent overfitting
            'entropy_coef': 0.02,  # Increased for better exploration
            'n_steps': 1024,  # Reduced from 2048 for more frequent updates
            'gae_lambda': 0.95
        },
        'a2c': {
            'learning_rate': 0.0005,  # Balanced learning rate
            'gamma': 0.99,
            'batch_size': 64,
            'max_episode_steps': 500,
            'entropy_coef': 0.02
        },
        'sac': {
            'learning_rate': 0.0002,  # Balanced learning rate
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 64,
            'tau': 0.005,
            'max_episode_steps': 500
        }
    })
    
    def __post_init__(self):
        # Update module size based on network sizes
        self.modular_params['module_size'] = {
            size: size // self.modular_params['num_modules']
            for size in self.network_sizes
        }
        
        # Calculate parameter budgets for each network size
        if self.parameter_budget['enabled']:
            self.parameter_budget['size_budgets'] = self._calculate_size_budgets()
    
    def _calculate_size_budgets(self) -> Dict[int, int]:
        """Calculate parameter budget for each network size."""
        budgets = {}
        base_budget = self.parameter_budget['target_budget']
        
        for size in self.network_sizes:
            if self.parameter_budget['normalize_by_size']:
                # Normalize budget by network size
                # This ensures fair comparison across different sizes
                normalized_budget = int(base_budget * (size / min(self.network_sizes)))
            else:
                normalized_budget = base_budget
            
            budgets[size] = normalized_budget
        
        return budgets
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'experiment_types': self.experiment_types,
            'parameter_budget': self.parameter_budget,
            'task_sequence': self.task_sequence,
            'network_sizes': self.network_sizes,
            'seeds': self.seeds,
            'num_layers': self.num_layers,
            'network_types': self.network_types,
            'episodes_per_task': self.episodes_per_task,
            'evaluation_episodes': self.evaluation_episodes,
            'backward_transfer_tasks': self.backward_transfer_tasks,
            'forward_transfer_tasks': self.forward_transfer_tasks,
            'forgetting_test': self.forgetting_test,
            'network_params': self.network_params,
            'fully_connected_params': self.fully_connected_params,
            'small_world_params': self.small_world_params,
            'modular_params': self.modular_params,
            'node_selection_strategies': self.node_selection_strategies,
            'num_io_nodes': self.num_io_nodes,
            'rl_params': self.rl_params
        } 