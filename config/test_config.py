from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class TestConfig:
    """Configuration for quick testing of the system."""
    
    # Network sizes to test (minimal set)
    network_sizes: List[int] = field(default_factory=lambda: [50])  # One small size for quick testing
    
    # Random seeds (just one for testing)
    seeds: List[int] = field(default_factory=lambda: [42])
    
    # Number of layers to test
    num_layers: List[int] = field(default_factory=lambda: [1, 2])  # Test both single and multi-layer
    
    # Network types to test
    network_types: List[str] = field(default_factory=lambda: ['rnn', 'ffn'])  # Test both network types
    
    # Network-specific parameters
    network_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'rnn': {
            'hidden_size': 16,  # Reduced for testing
            'sequence_length': 5,  # Reduced for testing
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'ffn': {
            'activation': 'relu',
            'learning_rate': 0.001,
            'batch_size': 32
        }
    })
    
    # Small-world network parameters
    small_world_params: Dict[str, Any] = field(default_factory=lambda: {
        'k': 4,
        'p': 0.1,
        'inter_layer_prob': 0.1
    })
    
    # Modular network parameters
    modular_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_modules': 2,  # Reduced for testing
        'inter_module_prob': 0.1,
        'intra_module_prob': 0.3,
        'inter_layer_prob': 0.1
    })
    
    # Node selection strategies (one of each type)
    node_selection_strategies: List[str] = field(default_factory=lambda: [
        'random',  # Basic strategy
        'centrality_based',  # Complex strategy
        'module_based'  # Strategy requiring module information
    ])
    
    # Number of input/output nodes
    num_io_nodes: int = 5
    
    # Tasks to test (one of each type)
    tasks: List[str] = field(default_factory=lambda: [
        'classification',  # Basic task
        'regression'  # Complex task
    ])
    
    def __post_init__(self):
        # Update module size based on network sizes
        self.modular_params['module_size'] = {
            size: size // self.modular_params['num_modules']
            for size in self.network_sizes
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
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
            'is_test_run': True  # Mark this as a test run
        } 