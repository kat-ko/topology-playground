from dataclasses import dataclass, field
from typing import List, Dict, Any
from .curriculum_config import CurriculumConfig

@dataclass
class TestCurriculumConfig(CurriculumConfig):
    """Test configuration for quick curriculum learning experiments."""
    
    # Reduced network sizes for testing
    network_sizes: List[int] = field(default_factory=lambda: [50])
    
    # Single seed for testing
    seeds: List[int] = field(default_factory=lambda: [42])
    
    # Test with single layer only
    num_layers: List[int] = field(default_factory=lambda: [1])
    
    # Test with one network type
    network_types: List[str] = field(default_factory=lambda: ['rnn', 'ffn'])
    
    # Include all tasks for curriculum testing
    task_sequence: List[str] = field(default_factory=lambda: [
        'cartpole',
        'mountain_car',
        'acrobot'
    ])
    
    # Reduced training parameters
    episodes_per_task: int = 200
    evaluation_episodes: int = 20
    max_env_steps_per_task: int = 20000
    
    # Reduced node selection strategies
    node_selection_strategies: List[str] = field(default_factory=lambda: ['random'])
    
    # Reduced experiment types
    experiment_types: List[str] = field(default_factory=lambda: [
        'match_fully_connected',  # Test both basic and capacity matching
        'same_size',
    ])
    
    # Include transfer learning tasks for testing
    backward_transfer_tasks: List[str] = field(default_factory=lambda: ['cartpole'])
    forward_transfer_tasks: List[str] = field(default_factory=lambda: ['mountain_car', 'acrobot'])
    
    # Reduced retention testing
    forgetting_test: Dict[str, Any] = field(default_factory=lambda: {
        'retention_interval': 5,
        'retention_episodes': 5,
        'forgetting_threshold': 0.8,
        'retention_threshold': 0.9
    })
    
    # Added for the new test configuration
    topologies: List[str] = field(default_factory=lambda: ['small_world', 'modular', 'hybrid', 'fully_connected'])
    
    def __post_init__(self):
        super().__post_init__()
        # Mark as test run
        self.is_test_run = True 

    def to_dict(self):
        return {
            'experiment_types': self.experiment_types,
            'parameter_budget': self.parameter_budget,
            'task_sequence': self.task_sequence,
            'network_sizes': self.network_sizes,
            'seeds': self.seeds,
            'num_layers': self.num_layers,
            'network_types': self.network_types,
            'topologies': self.topologies,
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