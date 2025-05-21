from .experiment_config import ExperimentConfig

class TestConfig(ExperimentConfig):
    """Minimal test configuration for quick validation of the system."""
    
    def __init__(self):
        super().__init__()
        
        # Test with smallest network size only
        self.network_sizes = [50]
        
        # Single seed for reproducibility
        self.seeds = [42]
        
        # Test both single and multi-layer
        self.num_layers = [1, 2]
        
        # Test both network types
        self.network_types = ['ffn', 'rnn']
        
        # Test one strategy from each category
        self.node_selection_strategies = [
            'random',           # Basic strategy
            'centrality_based'  # Complex strategy
        ]
        
        # Test one task from each category
        self.tasks = [
            'classification',   # Supervised learning
            'clustering'        # Unsupervised learning
        ]
        
        # Simplified network parameters
        self.network_params = {
            'ffn': {
                'activation': 'relu',
                'learning_rate': 0.01,
                'batch_size': 32
            },
            'rnn': {
                'activation': 'tanh',
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_size': 16,
                'sequence_length': 5
            }
        }
        
        # Simplified topology parameters
        self.small_world_params = {
            'k': 4,
            'p': 0.1,
            'inter_layer_prob': 0.1
        }
        
        self.modular_params = {
            'num_modules': 2,
            'inter_module_prob': 0.1,
            'intra_module_prob': 0.3,
            'inter_layer_prob': 0.1,
            'module_size': {50: 25}  # Only for size 50
        } 