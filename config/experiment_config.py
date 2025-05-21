from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class ExperimentConfig:
    # Network sizes to test
    network_sizes: List[int] = field(default_factory=lambda: [50, 100, 200])
    
    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    
    # Small-world network parameters
    small_world_params: Dict[str, Any] = field(default_factory=lambda: {
        'k': 4,  # Number of nearest neighbors
        'p': 0.1  # Rewiring probability
    })
    
    # Modular network parameters
    modular_params: Dict[str, Any] = field(default_factory=lambda: {
        'num_modules': 4,
        'module_size': None,  # Will be set based on network size
        'inter_module_prob': 0.1,
        'intra_module_prob': 0.3
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
    
    # Task parameters
    tasks: List[str] = field(default_factory=lambda: [
        'classification',
        'regression',
        'clustering'
    ])
    
    def __post_init__(self):
        # Update module size based on network sizes
        self.modular_params['module_size'] = {
            size: size // self.modular_params['num_modules']
            for size in self.network_sizes
        } 