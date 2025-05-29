from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class ExperimentConfig:
    """Configuration for network experiments."""
    
    # Network sizes to test
    network_sizes: List[int] = field(default_factory=lambda: [50, 100, 200])
    
    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Number of layers in the network
    num_layers: List[int] = field(default_factory=lambda: [1, 2])
    
    # Network types to test
    network_types: List[str] = field(default_factory=lambda: ['ffn', 'rnn'])
    
    # Network-specific parameters
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
    
    # Small-world network parameters
    small_world_params: Dict[str, Any] = field(default_factory=lambda: {
        'k': 4,  # Number of nearest neighbors
        'p': 0.1,  # Rewiring probability
        'inter_layer_prob': 0.1  # Probability of inter-layer connections
    })
    
    # Modular network parameters
    modular_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_modules': 4,  # Number of modules
        'inter_module_prob': 0.1,  # Probability of inter-module connections
        'intra_module_prob': 0.3,  # Probability of intra-module connections
        'inter_layer_prob': 0.1  # Probability of inter-layer connections
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
    
    # Tasks to test
    tasks: List[str] = field(default_factory=lambda: [
        'cartpole',
        'mountain_car',
        'acrobot'
    ])
    
    # RL-specific parameters
    rl_params: Dict[str, Any] = field(default_factory=lambda: {
        'state_dim': 4,  # For CartPole
        'action_dim': 2,  # For CartPole
        'ppo': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'clip_ratio': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'batch_size': 64,
            'max_episode_steps': 500  # Increased from 100
        },
        'a2c': {
            'learning_rate': 0.0007,
            'gamma': 0.99,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'batch_size': 64,
            'max_episode_steps': 500  # Increased from 100
        },
        'sac': {
            'learning_rate': 0.0003,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 64,
            'target_update_freq': 1000,
            'tau': 0.005,
            'entropy_coef': 0.01,
            'max_episode_steps': 500  # Increased from 100
        },
        'curriculum': {
            'num_iterations': 100,
            'eval_episodes': 10,
            'task_difficulty_increase': 0.1,
            'episodes_per_iteration': 5,  # Number of episodes to run per iteration
            'task_memory_size': 6,  # Number of tasks to remember (2 per task type)
            'task_memory_threshold': 400,  # Reward threshold to store task in memory
            'difficulty_threshold': 450,  # Reward threshold to increase difficulty
            'replay_episodes': 2,  # Number of episodes to replay from memory
            'replay_frequency': 5,  # How often to replay from memory (iterations)
            'forgetting_test': {
                'retention_interval': 10,  # How often to test retention (iterations)
                'retention_episodes': 5,  # Number of episodes for retention testing
                'forgetting_threshold': 0.8,  # Performance threshold to consider as forgetting
                'retention_threshold': 0.9  # Performance threshold to consider as retained
            }
        }
    })
    
    def __post_init__(self):
        # Update module size based on network sizes
        self.modular_params['module_size'] = {
            size: size // self.modular_params['num_modules']
            for size in self.network_sizes
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'network_sizes': self.network_sizes,
            'seeds': self.seeds,
            'num_layers': self.num_layers,
            'network_types': self.network_types,
            'network_params': self.network_params,
            'small_world_params': self.small_world_params,
            'modular_params': self.modular_params,
            'node_selection_strategies': self.node_selection_strategies,
            'num_io_nodes': self.num_io_nodes,
            'tasks': self.tasks,
            'rl_params': self.rl_params
        }

# Network types to test
NETWORK_TYPES = ['rnn', 'ffn']

# Network topologies to test
TOPOLOGIES = ['small_world', 'modular', 'hybrid']

# Tasks to test
TASKS = ['classification', 'regression', 'clustering']

# Algorithms to test
ALGORITHMS = ['SAC', 'A2C', 'PPO']

# Experiment configuration
EXPERIMENT_CONFIG: Dict[str, Any] = {
    'network_sizes': [50, 100, 200],
    'seeds': [42, 123, 456],
    'num_layers': [1, 2],  # Test with 1 and 2 layers
    'strategies': ['random', 'centrality_based', 'distance_based', 'module_based'],
    'tasks': TASKS,
    'algorithms': ALGORITHMS,
    'topologies': TOPOLOGIES,
    'network_types': NETWORK_TYPES,
    
    # Topology-specific parameters
    'topology_params': {
        'small_world': {
            'k': 4,  # Number of nearest neighbors
            'p': 0.1,  # Rewiring probability
            'inter_layer_prob': 0.1  # Probability of inter-layer connections
        },
        'modular': {
            'num_modules': 4,
            'inter_module_prob': 0.1,
            'intra_module_prob': 0.3,
            'inter_layer_prob': 0.1  # Probability of inter-layer connections
        },
        'hybrid': {
            'num_modules': 4,
            'k': 4,
            'p': 0.1,
            'inter_module_prob': 0.1,
            'inter_layer_prob': 0.1  # Probability of inter-layer connections
        }
    },
    
    # Task-specific parameters
    'task_params': {
        'classification': {
            'num_classes': 3,
            'input_dim': 10,
            'num_samples': 1000
        },
        'regression': {
            'input_dim': 10,
            'num_samples': 1000
        },
        'clustering': {
            'num_clusters': 3,
            'num_samples': 1000
        }
    },
    
    # Algorithm-specific parameters
    'algorithm_params': {
        'SAC': {
            'learning_rate': 0.001,
            'buffer_size': 10000,
            'batch_size': 64
        },
        'A2C': {
            'learning_rate': 0.001,
            'entropy_coef': 0.01
        },
        'PPO': {
            'learning_rate': 0.001,
            'clip_ratio': 0.2,
            'value_coef': 0.5
        }
    },
    
    # Training parameters
    'training_params': {
        'max_epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    }
} 