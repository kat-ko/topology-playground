from dataclasses import dataclass, field
from typing import List, Dict, Any
from .curriculum_config import CurriculumConfig

@dataclass
class TestCurriculumConfig(CurriculumConfig):
    """Configuration for curriculum learning test experiments - focused on problematic cases."""
    
    # Focus on the sizes that showed large divergences
    network_sizes: List[int] = field(default_factory=lambda: [25, 50, 100])
    
    # Use the same seeds as main curriculum
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Focus on the layer counts that showed issues
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Test both network types that showed problems
    network_types: List[str] = field(default_factory=lambda: ['ffn', 'rnn'])
    
    # Include all tasks for curriculum testing
    task_sequence: List[str] = field(default_factory=lambda: [
        'cartpole',
        'mountain_car',
        'acrobot'
    ])
    
    # Reduced training parameters for faster testing
    episodes_per_task: int = 20  # Very small for debugging
    evaluation_episodes: int = 100
    max_env_steps_per_task: int = 500
    
    # Test all node selection strategies to see if issue is consistent
    node_selection_strategies: List[str] = field(default_factory=lambda: [
        'random',
        'centrality_based', 
        'distance_based',
        'module_based'
    ])
    
    # Focus on the experiment types that showed large divergences
    experiment_types: List[str] = field(default_factory=lambda: [
        'match_fully_connected',  # This showed 100%+ divergences
        'match_small_world',      # This showed 40-200% divergences
        'match_modular',          # This showed 90-200% divergences
        'match_hybrid'            # This showed 90-200% divergences
    ])
    
    # Include transfer learning tasks for testing
    backward_transfer_tasks: List[str] = field(default_factory=lambda: ['cartpole'])
    forward_transfer_tasks: List[str] = field(default_factory=lambda: ['mountain_car', 'acrobot'])
    
    # Minimal retention testing for debugging
    forgetting_test: Dict[str, Any] = field(default_factory=lambda: {
        'retention_interval': 2,
        'retention_episodes': 2,
        'forgetting_threshold': 0.8,
        'retention_threshold': 0.9
    })
    
    # Added for the new test configuration
    topologies: List[str] = field(default_factory=lambda: ['small_world', 'modular', 'hybrid', 'fully_connected'])
    
    def __post_init__(self):
        super().__post_init__()

        self.network_sizes = [25]  
        self.seeds = [42] 
        self.num_layers = [1]  
        self.network_types = ['ffn']  
        self.node_selection_strategies = [
            'random',
            ] 
        self.experiment_types = [
            'match_small_world',
        ]  
        
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
            'rl_params': self.rl_params,
            'use_capacity_mapping': False  # Temporarily disable for testing
        } 