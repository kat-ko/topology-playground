from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning experiments."""
    
    # Task sequence
    task_sequence: List[str] = field(default_factory=lambda: [
        'cartpole',
        'mountain_car',
        'acrobot'
    ])
    
    # Network parameters (reusing from ExperimentConfig)
    network_sizes: List[int] = field(default_factory=lambda: [100])
    seeds: List[int] = field(default_factory=lambda: [42])
    num_layers: List[int] = field(default_factory=lambda: [2])
    network_types: List[str] = field(default_factory=lambda: ['ffn'])
    
    # Training parameters
    episodes_per_task: int = 1000
    evaluation_episodes: int = 100
    
    # Transfer learning parameters
    backward_transfer_tasks: List[str] = field(default_factory=lambda: ['cartpole'])
    forward_transfer_tasks: List[str] = field(default_factory=lambda: ['mountain_car', 'acrobot'])
    
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
    
    def __post_init__(self):
        # Update module size based on network sizes
        self.modular_params['module_size'] = {
            size: size // self.modular_params['num_modules']
            for size in self.network_sizes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'task_sequence': self.task_sequence,
            'network_sizes': self.network_sizes,
            'seeds': self.seeds,
            'num_layers': self.num_layers,
            'network_types': self.network_types,
            'episodes_per_task': self.episodes_per_task,
            'evaluation_episodes': self.evaluation_episodes,
            'backward_transfer_tasks': self.backward_transfer_tasks,
            'forward_transfer_tasks': self.forward_transfer_tasks,
            'network_params': self.network_params,
            'small_world_params': self.small_world_params,
            'modular_params': self.modular_params,
            'node_selection_strategies': self.node_selection_strategies,
            'num_io_nodes': self.num_io_nodes
        } 