from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import networkx as nx
import math
from .capacity_mapping import CapacityMapper

@dataclass
class ParameterBudgetCalculator:
    """Pre-computes and manages parameter budgets for different topologies and experiment types."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize the calculator with base parameters and capacity mapping."""
        self.base_size = min(self.config['network_sizes'])
        self.topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
        self.experiment_types = self.config['experiment_types']
        # Capacity mapping system (default: enabled)
        self.use_capacity_mapping = self.config.get('use_capacity_mapping', True)
        if self.use_capacity_mapping:
            self.capacity_mapper = CapacityMapper(self.config)
        else:
            self.capacity_mapper = None
        
        # Pre-compute base capacities for each topology
        self.base_capacities = self._compute_base_capacities()
        
        # Pre-compute budgets for each experiment type and network size
        self.budgets = self._compute_all_budgets()
    
    def _compute_base_capacities(self) -> Dict[str, int]:
        """Compute the base capacity (parameters) for each topology at base size."""
        capacities = {}
        
        for topology in self.topologies:
            # Calculate parameters based on actual topology structure
            capacities[topology] = self._calculate_topology_parameters(topology, self.base_size)
        
        return capacities
    
    def _calculate_topology_parameters(self, topology: str, size: int) -> int:
        """Calculate actual parameters for a given topology and size using empirical models."""
        num_layers = self.config.get('num_layers', [1])[0]
        
        # Use empirical models if available
        if topology in EMPIRICAL_SCALING_MODELS:
            base_params = EMPIRICAL_SCALING_MODELS[topology]['formula'](size)
            
            # Apply dynamic multipliers based on capacity range
            capacity_range = self._get_capacity_range(base_params)
            if capacity_range in EMPIRICAL_SCALING_MODELS[topology]['dynamic_multipliers']:
                multiplier = EMPIRICAL_SCALING_MODELS[topology]['dynamic_multipliers'][capacity_range](base_params)
                base_params = int(base_params * multiplier)
            
            # Adjust for number of layers
            if num_layers > 1:
                # For multi-layer networks, add inter-layer connections
                # This is a simplified approximation
                inter_layer_params = size * size * (num_layers - 1) * 0.1  # 10% connectivity between layers
                return int(base_params + inter_layer_params)
            return base_params
        
        # Fallback to original calculations
        if topology == 'small_world':
            # Small world: k connections per node + biases, but reduced due to acyclicity constraint
            k = max(2, size // 10)  # Number of local connections
            # Due to acyclicity constraint, only ~45% of expected connections are created
            # Each node has approximately 0.45 * k incoming connections + 1 bias
            return int(size * k * 0.45 + size)  # weights + biases
            
        elif topology == 'modular':
            # Modular: connections within modules + inter-module connections
            num_modules = max(2, size // 20)
            module_size = size // num_modules
            # Intra-module connections: each module has module_size^2 connections
            intra_module_connections = num_modules * (module_size * module_size)
            # Inter-module connections: sparse connections between modules
            inter_module_connections = num_modules * (num_modules - 1) * module_size * 2
            # Biases: one per node
            return intra_module_connections + inter_module_connections + size
            
        elif topology == 'hybrid':
            # Hybrid: combines small world and modular
            k = max(2, size // 10)
            num_modules = max(2, size // 20)
            module_size = size // num_modules
            # Small world connections: k per node
            sw_connections = size * k
            # Modular connections within modules
            modular_connections = num_modules * (module_size * module_size)
            # Biases: one per node
            return sw_connections + modular_connections + size
            
        elif topology == 'fully_connected':
            # Fully connected: all nodes connected to all nodes
            # For each layer: size^2 connections + size biases
            total_connections = 0
            for layer in range(num_layers):
                if layer == 0:
                    # Input layer: size^2 connections
                    total_connections += size * size
                else:
                    # Hidden layers: size^2 connections
                    total_connections += size * size
                # Add biases for each layer
                total_connections += size
            return total_connections
        
        else:
            raise ValueError(f"Unknown topology: {topology}")
    
    def _get_capacity_range(self, capacity: int) -> str:
        """Get the capacity range for applying dynamic multipliers."""
        if capacity < 1000:
            return 'small'
        elif capacity < 5000:
            return 'medium'
        else:
            return 'large'
    
    def _compute_all_budgets(self) -> Dict[str, Dict[str, Dict[int, Any]]]:
        """Compute budgets for all experiment types and network sizes."""
        budgets = {}
        
        for experiment_type in self.experiment_types:
            budgets[experiment_type] = {}
            
            if experiment_type.startswith('match_'):
                reference_topology = experiment_type[len('match_'):]
                for topology in self.topologies:
                    budgets[experiment_type][topology] = {}
                    for size in self.config['network_sizes']:
                        budgets[experiment_type][topology][size] = {}
                        for network_type in self.config['network_types']:
                            budgets[experiment_type][topology][size][network_type] = {}
                            for num_layers in self.config['num_layers']:
                                # Always use reference topology's capacity as target
                                target_capacity = self._get_reference_capacity(
                                    reference_topology, size, network_type, num_layers
                                )
                                budgets[experiment_type][topology][size][network_type][num_layers] = {
                                    'target_capacity': target_capacity
                                }
            else:
                for topology in self.topologies:
                    budgets[experiment_type][topology] = {}
                    for size in self.config['network_sizes']:
                        budgets[experiment_type][topology][size] = self._compute_budget(
                            experiment_type, topology, size
                        )
        return budgets
    
    def _pre_calculate_matching_size(self, topology: str, target_capacity: int, network_type: str = 'ffn', num_layers: int = 1) -> int:
        """
        Pre-calculate the matching size for a topology to achieve target capacity.
        Uses incremental adjustment to find the right size step by step.
        """
        # Use capacity mapper if available
        if self.use_capacity_mapping and self.capacity_mapper is not None:
            try:
                return self.capacity_mapper.find_matching_size(topology, target_capacity, network_type, num_layers)
            except Exception as e:
                print(f"[CapacityMapper fallback] {e}")
                # Fallback to incremental adjustment below
        
        # Use incremental adjustment to find the right size
        return self._find_matching_size_incremental(topology, target_capacity, network_type, num_layers)
    
    def _find_matching_size_incremental(self, topology: str, target_capacity: int, network_type: str = 'ffn', num_layers: int = 1) -> int:
        """
        Find matching size using incremental adjustment with adaptive search parameters.
        Designed to work robustly across all network sizes and topologies.
        Optimized for performance with early termination and adaptive search.
        """
        # Allow config override for min/max size
        min_size = self.config.get('min_search_size', 10)
        max_size = self.config.get('max_search_size', 2000)
        # But don't go below topology's minimum viable size
        min_size = max(min_size, self._get_minimum_viable_size(topology))

        # Start with a reasonable estimate based on topology type and number of layers
        if num_layers > 1:
            # For multi-layer networks, use different scaling
            if topology == 'fully_connected':
                # Adjust scaling for small target capacities (like matching to small_world)
                if target_capacity < 1000:
                    # For small capacities, use more conservative scaling
                    estimated_size = int((target_capacity / (1.8 * num_layers)) ** 0.5)
                else:
                    estimated_size = int((target_capacity / (2.05 * num_layers)) ** 0.5)
            elif topology in ['small_world', 'modular', 'hybrid']:
                estimated_size = int((target_capacity / (1.5 * num_layers)) ** 0.67)
            else:
                estimated_size = min_size
        else:
            # Single layer networks use original scaling
            if topology == 'fully_connected':
                estimated_size = int((target_capacity / 2.05) ** 0.5)
            elif topology in ['small_world', 'modular', 'hybrid']:
                estimated_size = int((target_capacity / 1.5) ** 0.67)
            else:
                estimated_size = min_size

        start_size = max(min_size, estimated_size)
        
        # ADAPTIVE SEARCH PARAMETERS based on target capacity and topology
        # For small world topology, we need much larger search ranges due to acyclicity constraint
        if topology == 'small_world':
            # Small world has highly variable parameter scaling due to acyclicity
            if target_capacity > 10000:
                search_range = max(2000, start_size * 4)  # Very large range for high capacity
                step_size = 20
                fine_step = 5
            elif target_capacity > 5000:
                search_range = max(1500, start_size * 3)  # Large range for medium-high capacity
                step_size = 15
                fine_step = 3
            elif target_capacity > 1000:
                search_range = max(1000, start_size * 2.5)  # Large range for medium capacity
                step_size = 10
                fine_step = 2
            else:
                search_range = max(500, start_size * 2)  # Medium range for low capacity
                step_size = 3  # Much finer step size for small capacities
                fine_step = 1  # Very fine step for small capacities
        elif topology == 'modular':
            # Modular has more predictable scaling
            if target_capacity > 10000:
                search_range = max(1000, start_size * 2)
                step_size = 15
                fine_step = 3
            elif target_capacity > 5000:
                search_range = max(800, start_size * 1.8)
                step_size = 12
                fine_step = 2
            elif target_capacity > 1000:
                search_range = max(500, start_size * 1.5)
                step_size = 8
                fine_step = 2
            else:
                search_range = max(500, start_size * 1.5)
                step_size = 3  # Much finer step size for small capacities
                fine_step = 1  # Very fine step for small capacities
        elif topology == 'hybrid':
            # Hybrid combines small world and modular characteristics
            if target_capacity > 10000:
                search_range = max(1500, start_size * 3)
                step_size = 18
                fine_step = 4
            elif target_capacity > 5000:
                search_range = max(1200, start_size * 2.5)
                step_size = 15
                fine_step = 3
            elif target_capacity > 1000:
                search_range = max(800, start_size * 2)
                step_size = 10
                fine_step = 2
            else:
                search_range = max(800, start_size * 2)
                step_size = 3  # Much finer step size for small capacities
                fine_step = 1  # Very fine step for small capacities
        else:  # fully_connected
            # Fully connected has the most predictable scaling
            if target_capacity > 10000:
                search_range = max(800, start_size * 1.5)
                step_size = 12
                fine_step = 2
            elif target_capacity > 5000:
                search_range = max(600, start_size * 1.3)
                step_size = 10
                fine_step = 2
            elif target_capacity > 1000:
                search_range = max(400, start_size * 1.2)
                step_size = 8
                fine_step = 2
            else:
                # For very small target capacities (like matching to small world), use finer search
                search_range = max(300, start_size * 1.5)  # Slightly larger range for small targets
                step_size = 3  # Much finer step size for small targets
                fine_step = 1  # Very fine step for small targets
        
        # Adjust for multi-layer networks
        if num_layers > 1:
            search_range = int(search_range * 1.5)  # Increase search range for multi-layer
            step_size = max(step_size // 2, 5)  # Use finer steps for multi-layer
            fine_step = max(fine_step // 2, 1)  # Use even finer steps for multi-layer
        
        best_size = start_size
        best_divergence = float('inf')
        
        # OPTIMIZATION 1: Try the estimated size first (often very close)
        try:
            network = self._create_test_network(topology, start_size, network_type, num_layers)
            if network is not None:
                metrics = network.get_network_metrics()
                actual_capacity = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                best_divergence = abs(actual_capacity - target_capacity) / target_capacity
                best_size = start_size
                if best_divergence < 0.01:
                    return best_size
        except Exception as e:
            pass
        
        # OPTIMIZATION 3: Adaptive binary search for coarse search (much faster than linear)
        left = min_size
        right = min(max_size, start_size + search_range)
        last_actual_capacity = None
        
        # Adaptive binary search with dynamic step size based on relative error
        while left < right - 1:  # Continue until we can't divide further
            mid = (left + right) // 2
            try:
                network = self._create_test_network(topology, mid, network_type, num_layers)
                if network is None:
                    right = mid
                    continue
                    
                metrics = network.get_network_metrics()
                actual_capacity = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                last_actual_capacity = actual_capacity
                
                divergence = abs(actual_capacity - target_capacity) / target_capacity
                if divergence < best_divergence:
                    best_divergence = divergence
                    best_size = mid
                
                # OPTIMIZATION 4: Early termination for excellent matches
                if divergence < 0.005:  # 0.5% threshold
                    return best_size
                
                # Adaptive step size based on relative error
                relative_error = abs(actual_capacity - target_capacity) / target_capacity
                if relative_error < 0.1:  # Within 10% of target
                    # Use finer search when close to target
                    step_threshold = 2
                elif relative_error < 0.3:  # Within 30% of target
                    # Use medium search when moderately close
                    step_threshold = 5
                else:
                    # Use coarse search when far from target
                    step_threshold = 10
                
                if actual_capacity < target_capacity:
                    left = mid
                else:
                    right = mid
                    
                # Stop if we can't make meaningful progress
                if right - left <= step_threshold:
                    break
                    
            except Exception as e:
                right = mid
        
        # OPTIMIZATION 5: Adaptive fine search around best size found
        # For small target capacities, use fine_step=1 and a larger fine_range
        if target_capacity < 1000:
            fine_step = 1
            if topology == 'fully_connected':
                # For fully_connected matching small capacities, use much wider search range
                fine_range = max(100, start_size * 4)  # Increased from max(50, start_size * 2)
            else:
                fine_range = max(50, start_size * 2)  # Keep original for other topologies
        else:
            if best_divergence < 0.05:  # 5% threshold
                fine_range = min(50, max(20, start_size // 4))  # Smaller range for good matches
            else:
                fine_range = min(200, max(50, start_size // 2))  # Larger range for poor matches
            fine_range = max(fine_range, step_size * 2)  # Ensure fine range is at least 2x step size
        
        consecutive_no_improvement = 0
        max_no_improvement = 5  # Stop if no improvement for 5 consecutive attempts
        
        for fine_size in range(max(min_size, best_size - fine_range), min(max_size, best_size + fine_range), fine_step):
            try:
                network = self._create_test_network(topology, fine_size, network_type, num_layers)
                if network is None:
                    consecutive_no_improvement += 1
                    if consecutive_no_improvement >= max_no_improvement:
                        break
                    continue
                metrics = network.get_network_metrics()
                actual_capacity = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                divergence = abs(actual_capacity - target_capacity) / target_capacity
                if divergence < best_divergence:
                    best_divergence = divergence
                    best_size = fine_size
                    consecutive_no_improvement = 0  # Reset counter
                else:
                    consecutive_no_improvement += 1
                if divergence < 0.002:  # 0.2% threshold
                    return best_size
                if consecutive_no_improvement >= max_no_improvement:
                    break
            except Exception as e:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= max_no_improvement:
                    break
                continue
        
        # Final local optimality check: test a wider range around best_size
        candidate_sizes = [best_size - 5, best_size - 4, best_size - 3, best_size - 2, best_size - 1, best_size, best_size + 1, best_size + 2, best_size + 3, best_size + 4, best_size + 5]
        optimal_size = best_size
        optimal_divergence = best_divergence
        for candidate in candidate_sizes:
            if candidate < min_size or candidate > max_size:
                continue
            try:
                network = self._create_test_network(topology, candidate, network_type, num_layers)
                if network is not None:
                    metrics = network.get_network_metrics()
                    actual_capacity = sum(
                        metrics.get(k, 0) for k in metrics if k.startswith('num_')
                    )
                    divergence = abs(actual_capacity - target_capacity) / target_capacity
                    if divergence < optimal_divergence:
                        optimal_divergence = divergence
                        optimal_size = candidate
            except Exception:
                continue
        return optimal_size
    
    def _create_test_network(self, topology: str, size: int, network_type: str = 'ffn', num_layers: int = 1) -> torch.nn.Module:
        """
        Create a test network for capacity measurement.
        Returns None if creation fails.
        """
        try:
            # Import here to avoid circular imports
            from ..topologies.small_world import SmallWorldTopology
            from ..topologies.modular import ModularTopology
            from ..topologies.hybrid import HybridTopology
            from ..topologies.fully_connected import FullyConnectedTopology
            from ..networks.ffn import FeedForwardNetwork
            from ..networks.rnn import RecurrentNetwork
            import numpy as np
            
            # Network class mapping
            network_class_map = {
                'ffn': FeedForwardNetwork,
                'rnn': RecurrentNetwork
            }
            
            # Create the actual topology
            topo_map = {
                'small_world': SmallWorldTopology(
                    size=size,
                    k=self.config['small_world_params']['k'],
                    p=self.config['small_world_params']['p'],
                    seed=42
                ),
                'modular': ModularTopology(
                    size=size,
                    num_modules=self.config['modular_params']['num_modules'],
                    inter_module_prob=self.config['modular_params']['inter_module_prob'],
                    intra_module_prob=self.config['modular_params']['intra_module_prob'],
                    seed=42
                ),
                'hybrid': HybridTopology(
                    size=size,
                    num_modules=self.config['modular_params']['num_modules'],
                    k=self.config['small_world_params']['k'],
                    p=self.config['small_world_params']['p'],
                    inter_module_prob=self.config['modular_params']['inter_module_prob'],
                    seed=42
                ),
                'fully_connected': FullyConnectedTopology(
                    size=size,
                    num_layers=num_layers,
                    seed=42
                )
            }
            
            # Generate graphs
            if topology == 'fully_connected':
                # FullyConnectedTopology.generate() always returns a single unified graph
                graphs = [topo_map[topology].generate(num_layers)]
            else:
                # Non-FC topologies generate a single graph
                graphs = [topo_map[topology].generate()]
            
            # Select input/output nodes for each layer (like training does)
            def select_nodes(graph, strategy, size, seed):
                rng = np.random.RandomState(seed)
                all_nodes = list(range(size))
                rng.shuffle(all_nodes)
                num_io_nodes = self.config['num_io_nodes']
                input_nodes = all_nodes[:num_io_nodes]
                output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
                return input_nodes, output_nodes
            
            # Create networks for each layer (like training does)
            networks = []
            total_params = 0
            
            # For non-FC topologies, we only have one graph
            num_graphs = len(graphs)
            
            for layer_idx in range(num_graphs):
                input_nodes, output_nodes = select_nodes(graphs[layer_idx], 'random', size, 42)
                
                network = network_class_map[network_type](
                    graphs[layer_idx],
                    input_nodes,
                    output_nodes,
                    self.config['network_params'][network_type]
                )
                
                networks.append(network)
                
                # Get metrics for this layer
                metrics = network.get_network_metrics()
                layer_params = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                total_params += layer_params
            
            # Return the first network but with total parameter count
            # We'll modify the first network's metrics to reflect total capacity
            first_network = networks[0]
            
            # Create a wrapper that returns the total capacity
            class MultiLayerNetworkWrapper:
                def __init__(self, network, total_capacity):
                    self.network = network
                    self.total_capacity = total_capacity
                
                def get_network_metrics(self):
                    # Return metrics with total capacity
                    metrics = self.network.get_network_metrics()
                    # Scale up the parameter counts to reflect total capacity
                    current_total = sum(
                        metrics.get(k, 0) for k in metrics if k.startswith('num_')
                    )
                    if current_total > 0:
                        scale_factor = self.total_capacity / current_total
                        for key in metrics:
                            if key.startswith('num_'):
                                metrics[key] = int(metrics[key] * scale_factor)
                    return metrics
                
                def __getattr__(self, name):
                    return getattr(self.network, name)
            
            return MultiLayerNetworkWrapper(first_network, total_params)
            
        except Exception as e:
            return None
    
    def _get_reference_capacity(self, reference_topology: str, size: int, network_type: str = 'ffn', num_layers: int = 1, seed: int = 42) -> int:
        """
        Get the actual parameter count of the reference topology at the given size.
        Uses theoretical calculations to avoid networkx compatibility issues.
        """
        # Use capacity mapper if available
        if self.use_capacity_mapping and self.capacity_mapper is not None:
            cap = self.capacity_mapper.get_capacity_at_size(reference_topology, size, network_type, num_layers)
            if cap is not None:
                # Apply dynamic multiplier to the measured capacity if available
                if reference_topology in EMPIRICAL_SCALING_MODELS:
                    capacity_range = self._get_capacity_range(cap)
                    dynamic_multipliers = EMPIRICAL_SCALING_MODELS[reference_topology].get('dynamic_multipliers', {})
                    if capacity_range in dynamic_multipliers:
                        multiplier = dynamic_multipliers[capacity_range](cap)
                        cap = int(cap * multiplier)
                return cap
        
        # Use theoretical calculations instead of creating actual networks
        # This avoids networkx compatibility issues and is much faster
        theoretical_capacity = self._calculate_topology_parameters(reference_topology, size)
        
        # Apply dynamic multiplier to the theoretical capacity if available
        if reference_topology in EMPIRICAL_SCALING_MODELS:
            capacity_range = self._get_capacity_range(theoretical_capacity)
            dynamic_multipliers = EMPIRICAL_SCALING_MODELS[reference_topology].get('dynamic_multipliers', {})
            if capacity_range in dynamic_multipliers:
                multiplier = dynamic_multipliers[capacity_range](theoretical_capacity)
                theoretical_capacity = int(theoretical_capacity * multiplier)
        
        return theoretical_capacity
    
    def _compute_budget(self, experiment_type: str, topology: str, size: int, network_type: str = None, num_layers: int = None) -> int:
        """Compute the parameter budget for a specific experiment type, topology, and size."""
        if experiment_type == 'same_size':
            if network_type is None:
                network_type = self.config['network_types'][0] if self.config['network_types'] else 'ffn'
            if num_layers is None:
                num_layers = self.config['num_layers'][0] if self.config['num_layers'] else 1
            actual_capacity = self._get_reference_capacity(topology, size, network_type, num_layers)
            return actual_capacity
        if experiment_type.startswith('match_'):
            # Always use reference topology's capacity as target
            reference_topology = experiment_type[len('match_'):]
            if network_type is None:
                network_type = self.config['network_types'][0] if self.config['network_types'] else 'ffn'
            if num_layers is None:
                num_layers = self.config['num_layers'][0] if self.config['num_layers'] else 1
            target_capacity = self._get_reference_capacity(
                reference_topology, size, network_type, num_layers
            )
            return target_capacity
        else:
            target_capacity = self.base_capacities[topology]
            scale_factor = size / self.base_size
            return int(target_capacity * scale_factor)
    
    def create_network(self, topology: str, size: int, experiment_type: str, network_type: str = 'ffn', num_layers: int = 1, seed: int = 42) -> torch.nn.Module:
        """Create a network with the specified topology and size, using the same logic as training."""
        # Import here to avoid circular imports
        from ..topologies.small_world import SmallWorldTopology
        from ..topologies.modular import ModularTopology
        from ..topologies.hybrid import HybridTopology
        from ..topologies.fully_connected import FullyConnectedTopology
        from ..networks.ffn import FeedForwardNetwork
        from ..networks.rnn import RecurrentNetwork
        import numpy as np
        
        # For match_* experiments, use pre-calculated matching size
        if experiment_type.startswith('match_'):
            # Pass the original size to get_matching_size, not the matching size
            matching_size = self.get_matching_size(experiment_type, topology, size, network_type, num_layers)
            # Use the matching size for network creation
            size = matching_size
        
        # Network class mapping
        network_class_map = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
        
        # Create the actual topology
        topo_map = {
            'small_world': SmallWorldTopology(
                size=size,
                k=self.config['small_world_params']['k'],
                p=self.config['small_world_params']['p'],
                seed=seed
            ),
            'modular': ModularTopology(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                intra_module_prob=self.config['modular_params']['intra_module_prob'],
                seed=seed
            ),
            'hybrid': HybridTopology(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                k=self.config['small_world_params']['k'],
                p=self.config['small_world_params']['p'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                seed=seed
            ),
            'fully_connected': FullyConnectedTopology(
                size=size,
                num_layers=num_layers,
                seed=seed
            )
        }
        
        # Generate graphs
        if topology == 'fully_connected':
            # FullyConnectedTopology.generate() always returns a single unified graph
            graphs = [topo_map[topology].generate(num_layers)]
        else:
            # Non-FC topologies generate a single graph
            graphs = [topo_map[topology].generate()]
        
        # Select input/output nodes for each layer
        def select_nodes(graph, strategy, size, seed):
            rng = np.random.RandomState(seed)
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            num_io_nodes = self.config['num_io_nodes']
            input_nodes = all_nodes[:num_io_nodes]
            output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
            return input_nodes, output_nodes
        
        # Create networks for each layer
        networks = []
        total_params = 0
        
        # For non-FC topologies, we only have one graph
        num_graphs = len(graphs)
        
        for layer_idx in range(num_graphs):
            input_nodes, output_nodes = select_nodes(graphs[layer_idx], 'random', size, seed)
            
            # Create network for this layer
            network_class = network_class_map[network_type]
            network_params = self.config['network_params'][network_type]
            
            try:
                network = network_class(graphs[layer_idx], input_nodes, output_nodes, network_params)
                networks.append(network)
                
                # Get metrics for this layer
                metrics = network.get_network_metrics()
                layer_params = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                total_params += layer_params
                
            except Exception as e:
                print(f"Error creating network for {topology} layer {layer_idx}: {e}")
                raise
        
        # For single layer, return the network directly
        if num_layers == 1:
            return networks[0]
        
        # For multi-layer, return a wrapper that represents the total capacity
        first_network = networks[0]
        
        # Create a wrapper that returns the total capacity
        class MultiLayerNetworkWrapper:
            def __init__(self, network, total_capacity):
                self.network = network
                self.total_capacity = total_capacity
            
            def get_network_metrics(self):
                # Return metrics with total capacity
                metrics = self.network.get_network_metrics()
                # Scale up the parameter counts to reflect total capacity
                current_total = sum(
                    metrics.get(k, 0) for k in metrics if k.startswith('num_')
                )
                if current_total > 0:
                    scale_factor = self.total_capacity / current_total
                    for key in metrics:
                        if key.startswith('num_'):
                            metrics[key] = int(metrics[key] * scale_factor)
                return metrics
            
            def __getattr__(self, name):
                return getattr(self.network, name)
        
        return MultiLayerNetworkWrapper(first_network, total_params)
    
    def get_budget(self, experiment_type: str, topology: str, size: int, network_type: str = None, num_layers: int = None) -> int:
        """Get the budget for a specific configuration."""
        if experiment_type.startswith('match_') and network_type is not None and num_layers is not None:
            # For capacity matching experiments with specific network type and layer count
            budget_info = self.budgets[experiment_type][topology][size][network_type][num_layers]
            return budget_info['target_capacity']
        else:
            # For other experiments or when network_type/num_layers not specified, use the first available
            if experiment_type.startswith('match_'):
                network_type = network_type or self.config['network_types'][0]
                num_layers = num_layers or self.config['num_layers'][0]
                budget_info = self.budgets[experiment_type][topology][size][network_type][num_layers]
                return budget_info['target_capacity']
            else:
                return self.budgets[experiment_type][topology][size]
    
    def get_matching_size(self, experiment_type: str, topology: str, size: int, network_type: str = None, num_layers: int = None) -> int:
        """Get the matching size for a specific configuration (only for match_* experiments)."""
        if not experiment_type.startswith('match_'):
            return size  # For non-matching experiments, return original size
        
        if network_type is None:
            network_type = self.config['network_types'][0]
        if num_layers is None:
            num_layers = self.config['num_layers'][0]
        
        # Get target capacity from pre-computed budget
        budget_info = self.budgets[experiment_type][topology][size][network_type][num_layers]
        target_capacity = budget_info['target_capacity']
        
        # Calculate matching size on-demand using incremental adjustment
        return self.calculate_matching_size(topology, target_capacity, network_type, num_layers)
    
    def get_budget_stats(self, experiment_type: str, topology: str, size: int) -> Dict[str, Any]:
        """Get statistics about the budget for a specific configuration."""
        budget = self.get_budget(experiment_type, topology, size)
        base_capacity = self.base_capacities[topology]
        scale_factor = size / self.base_size
        scaled_capacity = int(base_capacity * scale_factor)
        
        return {
            'experiment_type': experiment_type,
            'topology': topology,
            'size': size,
            'budget': budget,
            'base_capacity': base_capacity,
            'scaled_capacity': scaled_capacity,
            'budget_usage': scaled_capacity / budget if budget > 0 else float('inf')
        }
    
    def _count_parameters(self, network: torch.nn.Module) -> int:
        """Count total number of parameters in the network."""
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    def calculate_matching_size(self, topology: str, target_capacity: int, network_type: str = 'ffn', num_layers: int = 1) -> int:
        """
        Calculate the number of nodes needed for a topology to achieve the target capacity.
        Uses incremental adjustment to find the right size step by step.
        """
        # Use capacity mapper if available
        if self.use_capacity_mapping and self.capacity_mapper is not None:
            try:
                return self.capacity_mapper.find_matching_size(topology, target_capacity, network_type, num_layers)
            except Exception as e:
                print(f"[CapacityMapper fallback] {e}")
                # Fallback to incremental adjustment below
        
        # Use incremental adjustment to find the right size
        return self._find_matching_size_incremental(topology, target_capacity, network_type, num_layers)
    
    def _get_minimum_viable_size(self, topology: str) -> int:
        """Get the minimum viable size for a topology based on its parameters."""
        if topology == 'small_world':
            k = self.config['small_world_params']['k']
            # Small world needs at least k+1 nodes, but allow smaller sizes
            return max(10, k + 1)  # Reduced from 50 to 10
            
        elif topology == 'modular':
            num_modules = self.config['modular_params']['num_modules']
            # Modular needs at least num_modules nodes, but allow smaller sizes
            return max(10, num_modules)  # Reduced from 50 to 10
            
        elif topology == 'hybrid':
            k = self.config['small_world_params']['k']
            num_modules = self.config['modular_params']['num_modules']
            # Hybrid needs at least max(k, num_modules) nodes, but allow smaller sizes
            return max(10, max(k, num_modules))  # Reduced from 50 to 10
            
        elif topology == 'fully_connected':
            # Fully connected can work with smaller sizes
            return 10  # Reduced from 30 to 10
            
        else:
            return 10  # Default minimum reduced from 50 to 10

@dataclass
class ParameterBudget:
    """Handles parameter budget enforcement for fair comparison."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize budget tracking."""
        self.calculator = ParameterBudgetCalculator(self.config)
        self.budget_type = self.config['parameter_budget']['budget_type']
        self.padding_strategy = self.config['parameter_budget']['padding_strategy']
        self.experiment_type = self.config.get('experiment_type', 'same_size')
    
    def count_parameters(self, network: torch.nn.Module, size: int) -> int:
        """Count actual parameters in the network."""
        if self.budget_type == 'edges':
            return self._count_edges(network)
        else:  # weights
            return self._count_weights(network)
    
    def _count_edges(self, network: torch.nn.Module) -> int:
        """Count number of non-zero edges in the network."""
        edge_count = 0
        for param in network.parameters():
            if param.requires_grad:
                edge_count += torch.count_nonzero(param).item()
        return edge_count
    
    def _count_weights(self, network: torch.nn.Module) -> int:
        """Count total number of weights in the network."""
        return sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    def enforce_budget(self, network: torch.nn.Module, size: int, topology: str = None) -> torch.nn.Module:
        """Enforce parameter budget on the network."""
        if not self.config['parameter_budget']['enabled']:
            return network
        
        # Get target budget from calculator
        if topology is None:
            raise ValueError("Topology must be specified for budget enforcement")
        
        target_budget = self.calculator.get_budget(self.experiment_type, topology, size)
        current_params = self.count_parameters(network, size)
        
        if current_params <= target_budget:
            return network
        
        # Need to reduce parameters
        if self.budget_type == 'edges':
            return self._enforce_edge_budget(network, target_budget)
        else:
            return self._enforce_weight_budget(network, target_budget)
    
    def pad_to_budget(self, network: torch.nn.Module, size: int, topology: str = None) -> torch.nn.Module:
        """Pad network to meet parameter budget if needed."""
        if not self.config['parameter_budget']['enabled']:
            return network
        
        # Get target budget from calculator
        if topology is None:
            raise ValueError("Topology must be specified for budget padding")
        
        target_budget = self.calculator.get_budget(self.experiment_type, topology, size)
        current_params = self.count_parameters(network, size)
        
        if current_params >= target_budget:
            return network
        
        # Need to add parameters
        if self.budget_type == 'edges':
            return self._pad_edge_budget(network, target_budget)
        else:
            return self._pad_weight_budget(network, target_budget)
    
    def get_budget_stats(self, network: torch.nn.Module, size: int, topology: str = None) -> Dict[str, Any]:
        """Get statistics about parameter budget usage."""
        if topology is None:
            raise ValueError("Topology must be specified for budget stats")
        
        # Get stats from calculator
        calculator_stats = self.calculator.get_budget_stats(self.experiment_type, topology, size)
        
        # Add current network stats
        current_params = self.count_parameters(network, size)
        
        return {
            **calculator_stats,
            'current_params': current_params,
            'budget_usage': current_params / calculator_stats['budget'],
            'budget_type': self.budget_type,
            'padding_strategy': self.padding_strategy
        }
    
    def _enforce_edge_budget(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Enforce edge budget by zeroing out excess edges."""
        # Get all parameters
        params = []
        for param in network.parameters():
            if param.requires_grad:
                params.append(param.view(-1))
        
        # Concatenate all parameters
        all_params = torch.cat(params)
        
        # Get threshold for top k parameters
        k = target_budget
        if k >= len(all_params):
            return network
        
        # Get threshold value
        threshold = torch.kthvalue(torch.abs(all_params), len(all_params) - k).values
        
        # Zero out parameters below threshold
        for param in network.parameters():
            if param.requires_grad:
                mask = torch.abs(param) < threshold
                param.data[mask] = 0
        
        return network
    
    def _enforce_weight_budget(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Enforce weight budget by removing excess weights."""
        # This is more complex as we need to maintain network structure
        # For now, we'll use a simple approach of removing entire layers
        # if they exceed the budget
        
        total_params = 0
        layers_to_keep = []
        
        for name, module in network.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_params = sum(p.numel() for p in module.parameters())
                if total_params + layer_params <= target_budget:
                    total_params += layer_params
                    layers_to_keep.append(name)
                else:
                    break
        
        # Create new network with only kept layers
        new_network = type(network)()
        for name in layers_to_keep:
            setattr(new_network, name, getattr(network, name))
        
        return new_network
    
    def _pad_edge_budget(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Pad edge budget by adding random edges."""
        current_edges = self._count_edges(network)
        edges_to_add = target_budget - current_edges
        
        if edges_to_add <= 0:
            return network
        
        # Add random edges
        for param in network.parameters():
            if param.requires_grad:
                # Get number of zero elements
                zero_mask = param == 0
                num_zeros = zero_mask.sum().item()
                
                if num_zeros > 0:
                    # Calculate how many edges to add to this parameter
                    edges_for_param = min(edges_to_add, num_zeros)
                    
                    # Generate random values
                    if self.padding_strategy == 'random':
                        new_values = torch.randn(edges_for_param) * 0.01
                    else:  # zero
                        new_values = torch.zeros(edges_for_param)
                    
                    # Get random indices of zero elements
                    zero_indices = torch.nonzero(zero_mask.view(-1))[torch.randperm(num_zeros)[:edges_for_param]]
                    
                    # Add new edges
                    param.view(-1)[zero_indices] = new_values
                    
                    edges_to_add -= edges_for_param
                    if edges_to_add <= 0:
                        break
        
        return network
    
    def _pad_weight_budget(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Pad weight budget by adding new layers."""
        # This is more complex as we need to maintain network structure
        # For now, we'll use a simple approach of adding new layers
        # if we're under the budget
        
        current_params = self._count_weights(network)
        params_to_add = target_budget - current_params
        
        if params_to_add <= 0:
            return network
        
        # Add new layers until we reach the budget
        # This is a simplified approach - in practice, you'd want to
        # add layers in a way that maintains network structure
        while self._count_weights(network) < target_budget:
            # Add a small layer
            new_layer = torch.nn.Linear(10, 10)
            network.add_module(f'padding_layer_{len(list(network.modules()))}', new_layer)
        
        return network

SCALING_TABLE = {
    'modular': {
        'match_fully_connected': {25: 1.0, 50: 1.1, 100: 1.5},
        'match_small_world':     {25: 1.05, 50: 0.65, 100: 0.45},  # Further fine-tuned for size 25
        'match_modular':         {25: 1.0, 50: 1.0, 100: 1.0},
        'match_hybrid':          {25: 1.0, 50: 0.65, 100: 0.65},
    },
    'hybrid': {
        'match_fully_connected': {25: 0.85, 50: 0.75, 100: 0.70},
        'match_small_world':     {25: 1.2, 50: 1.2, 100: 1.1},
        'match_modular':         {25: 1.0, 50: 0.95, 100: 0.98},
        'match_hybrid':          {25: 1.0, 50: 1.0, 100: 1.0},
    },
    'small_world': {
        'match_fully_connected': {25: 1.0, 50: 1.0, 100: 1.0},
        'match_small_world':     {25: 1.0, 50: 1.0, 100: 1.0},
        'match_modular':         {25: 1.0, 50: 1.0, 100: 1.0},
        'match_hybrid':          {25: 1.0, 50: 1.0, 100: 1.0},
    },
    'fully_connected': {
        'match_fully_connected': {25: 1.0, 50: 1.0, 100: 1.0},
        'match_small_world':     {25: 1.0, 50: 1.0, 100: 1.0},
        'match_modular':         {25: 1.0, 50: 1.0, 100: 1.0},
        'match_hybrid':          {25: 1.0, 50: 1.0, 100: 1.0},
    },
}

SIZE_BINS = [25, 50, 100]

def get_closest_bin(size: int) -> int:
    return min(SIZE_BINS, key=lambda x: abs(x - size))

def calculate_network_size(size: int, topology: str, experiment_type: str, target_capacity: int) -> int:
    """
    Calculate the network size needed for a topology to achieve target capacity.
    Uses empirical models and dynamic scaling for better accuracy.
    """
    if experiment_type == 'same_size':
        return size

    if topology not in EMPIRICAL_SCALING_MODELS:
        raise ValueError(f"Unknown topology: {topology}")

    # Get empirical model
    model = EMPIRICAL_SCALING_MODELS[topology]
    
    # Determine capacity range for dynamic scaling
    capacity_range = get_capacity_range(target_capacity)
    
    # Get dynamic multiplier based on capacity range
    dynamic_multiplier = model['dynamic_multipliers'][capacity_range](target_capacity)
    
    # Calculate base size using empirical formula
    # We need to solve: target_capacity = model['formula'](base_size)
    # For most models, we can use inverse approximation
    
    if topology == 'small_world':
        # Inverse of 0.30 * size^1.92
        base_size = int((target_capacity / 0.30) ** (1/1.92))
    elif topology == 'modular':
        # For modular, we need to estimate module size first
        # Approximate: target_capacity ≈ 2.05 * size * (size/num_modules)
        # Assuming num_modules ≈ size/20
        base_size = int((target_capacity / 2.05) ** 0.5)
    elif topology == 'hybrid':
        # Inverse of 11.03 * size^1.25
        base_size = int((target_capacity / 11.03) ** (1/1.25))
    elif topology == 'fully_connected':
        # Inverse of 2.05 * size^2
        base_size = int((target_capacity / 2.05) ** 0.5)
    else:
        base_size = size
    
    # Apply dynamic multiplier
    scaled_size = int(base_size * dynamic_multiplier)
    
    # Ensure minimum viable size
    min_size = 30
    scaled_size = max(min_size, scaled_size)
    
    return scaled_size

def get_capacity_range(capacity: int) -> str:
    """Determine capacity range for dynamic scaling."""
    if capacity < 1000:
        return 'small'
    elif capacity < 5000:
        return 'medium'
    else:
        return 'large'

def calculate_divergence(actual: float, target: float) -> float:
    return abs((actual - target) / target) * 100

def adjust_scaling_factor(current_scaling: float, divergence: float, threshold: float = 5.0, step: float = 0.1) -> float:
    if divergence > threshold:
        if divergence > 0:  # Overshooting
            return current_scaling - step
        else:  # Undershooting
            return current_scaling + step
    return current_scaling

def optimize_scaling_factors(max_iterations: int = 10, divergence_threshold: float = 5.0) -> Dict[str, Dict[str, Dict[int, float]]]:
    optimized_table = SCALING_TABLE.copy()
    for _ in range(max_iterations):
        for topology in optimized_table:
            for match_target in optimized_table[topology]:
                for node_size in optimized_table[topology][match_target]:
                    # Simulate capacity matching (replace with actual test)
                    actual_capacity = simulate_capacity_matching(topology, match_target, node_size)
                    target_capacity = get_target_capacity(match_target, node_size)
                    divergence = calculate_divergence(actual_capacity, target_capacity)
                    optimized_table[topology][match_target][node_size] = adjust_scaling_factor(
                        optimized_table[topology][match_target][node_size],
                        divergence,
                        divergence_threshold
                    )
    return optimized_table

def simulate_capacity_matching(topology: str, match_target: str, node_size: int) -> float:
    # Placeholder for actual capacity matching simulation
    # Replace with actual implementation
    return 1000.0

def get_target_capacity(match_target: str, node_size: int) -> float:
    # Placeholder for target capacity calculation
    # Replace with actual implementation
    return 1000.0

# Optimize scaling factors
OPTIMIZED_SCALING_TABLE = optimize_scaling_factors()

# Update SCALING_TABLE with optimized values
SCALING_TABLE.update(OPTIMIZED_SCALING_TABLE)

# Empirical parameter growth models based on actual measurements
EMPIRICAL_SCALING_MODELS = {
    'small_world': {
        'formula': lambda size: int(0.135 * size**1.92),
        'dynamic_multipliers': {
            'small': lambda capacity: 0.8 if capacity < 1000 else 0.9,
            'medium': lambda capacity: 0.9 if capacity < 5000 else 1.0,
            'large': lambda capacity: 1.0 if capacity < 10000 else 1.1
        }
    },
    'modular': {
        'formula': lambda size: int(2.8 * size * (size // max(2, size // 20))),  # Increased from 2.05 to 2.8
        'dynamic_multipliers': {
            'small': lambda capacity: 0.93 if capacity < 1000 else 0.95,
            'medium': lambda capacity: 0.95 if capacity < 5000 else 1.0,
            'large': lambda capacity: 1.0 if capacity < 10000 else 1.05
        }
    },
    'hybrid': {
        'formula': lambda size: int(1.6 * size * (size // max(2, size // 15))),  # Increased from 1.2 to 1.6
        'dynamic_multipliers': {
            'small': lambda capacity: 0.82 if capacity < 1000 else 0.9,
            'medium': lambda capacity: 0.9 if capacity < 5000 else 1.0,
            'large': lambda capacity: 1.0 if capacity < 10000 else 1.1
        }
    },
    'fully_connected': {
        'formula': lambda size: int(2.05 * size**2),
        'dynamic_multipliers': {
            'small': lambda capacity: 1.05 if capacity < 1000 else 1.02,  # Increased from 0.95 to 1.05 to fix undershooting
            'medium': lambda capacity: 0.98 if capacity < 5000 else 1.0,
            'large': lambda capacity: 1.0 if capacity < 10000 else 1.02
        }
    }
}

# ... existing code ... 