from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import networkx as nx
import math

@dataclass
class ParameterBudgetCalculator:
    """Pre-computes and manages parameter budgets for different topologies and experiment types."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize the calculator with base parameters."""
        self.base_size = min(self.config['network_sizes'])
        self.topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
        self.experiment_types = self.config['experiment_types']
        
        # Pre-compute base capacities for each topology
        self.base_capacities = self._compute_base_capacities()
        
        # Pre-compute budgets for each experiment type and network size
        self.budgets = self._compute_all_budgets()
    
    def _compute_base_capacities(self) -> Dict[str, int]:
        """Compute the base capacity (parameters) for each topology at base size."""
        capacities = {}
        
        for topology in self.topologies:
            if topology == 'small_world':
                network = self._create_sample_small_world(self.base_size)
            elif topology == 'modular':
                network = self._create_sample_modular(self.base_size)
            elif topology == 'hybrid':
                network = self._create_sample_hybrid(self.base_size)
            elif topology == 'fully_connected':
                network = self._create_sample_fully_connected(self.base_size)
            
            capacities[topology] = self._count_parameters(network)
        
        return capacities
    
    def _compute_all_budgets(self) -> Dict[str, Dict[str, Dict[int, int]]]:
        """Compute budgets for all experiment types and network sizes."""
        budgets = {}
        
        for experiment_type in self.experiment_types:
            budgets[experiment_type] = {}
            
            for topology in self.topologies:
                budgets[experiment_type][topology] = {}
                
                for size in self.config['network_sizes']:
                    budgets[experiment_type][topology][size] = self._compute_budget(
                        experiment_type, topology, size
                    )
        
        return budgets
    
    def _compute_budget(self, experiment_type: str, topology: str, size: int) -> int:
        """Compute the parameter budget for a specific experiment type, topology, and size."""
        if experiment_type == 'same_size':
            # For same_size, use the base budget scaled by size
            base_budget = self.config['parameter_budget']['target_budget']
            scale_factor = size / self.base_size
            return int(base_budget * scale_factor)
        
        # For capacity matching experiments
        num_layers = self.config.get('num_layers', [1])[0]
        
        if experiment_type == 'match_fully_connected':
            # Calculate fully connected network parameters including biases
            # Input layer parameters
            target_capacity = size * size  # Input to hidden weights
            target_capacity += size  # Input layer biases
            # Hidden layers parameters
            for _ in range(num_layers - 1):
                target_capacity += size * size  # Hidden to hidden weights
                target_capacity += size  # Hidden layer biases
            # Output layer parameters
            target_capacity += size * size  # Hidden to output weights
            target_capacity += size  # Output layer biases
            
        elif experiment_type == 'match_small_world':
            # Calculate small world network parameters
            k = max(2, size // 10)  # Number of local connections
            # Input layer parameters
            target_capacity = size * k  # Input to hidden weights
            target_capacity += k  # Input layer biases
            # Hidden layers parameters
            for _ in range(num_layers - 1):
                target_capacity += k * k  # Hidden to hidden weights
                target_capacity += k  # Hidden layer biases
            # Output layer parameters
            target_capacity += k * size  # Hidden to output weights
            target_capacity += size  # Output layer biases
            
        elif experiment_type == 'match_modular':
            # Calculate modular network parameters
            num_modules = max(2, size // 20)  # Number of modules
            module_size = size // num_modules
            # Input layer parameters
            target_capacity = size * module_size  # Input to first module
            target_capacity += module_size  # Input layer biases
            # Hidden layers parameters
            for _ in range(num_layers - 1):
                target_capacity += module_size * module_size  # Module to module
                target_capacity += module_size  # Module biases
            # Output layer parameters
            target_capacity += module_size * size  # Module to output
            target_capacity += size  # Output layer biases
            
        elif experiment_type == 'match_hybrid':
            # Calculate hybrid network parameters
            k = max(2, size // 10)  # Number of local connections
            num_modules = max(2, size // 20)  # Number of modules
            module_size = size // num_modules
            # Input layer parameters
            target_capacity = size * k  # Input to local
            target_capacity += k  # Local biases
            # Hidden layers parameters
            for _ in range(num_layers - 1):
                target_capacity += k * module_size  # Local to module
                target_capacity += module_size  # Module biases
            # Output layer parameters
            target_capacity += module_size * size  # Module to output
            target_capacity += size  # Output layer biases
            
        else:
            # Get target topology from experiment type
            target = '_'.join(experiment_type.split('_')[1:])
            target_capacity = self.base_capacities[target]
            # Scale the target capacity by the ratio of current size to base size
            scale_factor = size / self.base_size
            target_capacity = int(target_capacity * scale_factor)
        
        return target_capacity
    
    def create_network(self, topology: str, size: int, experiment_type: str) -> torch.nn.Module:
        """Create a network with the specified topology and size, respecting the experiment type's budget."""
        # Get target capacity
        target_capacity = self.get_budget(experiment_type, topology, size)
        
        # Create base network to measure initial capacity
        if topology == 'small_world':
            base_network = self._create_sample_small_world(size)
        elif topology == 'modular':
            base_network = self._create_sample_modular(size)
        elif topology == 'hybrid':
            base_network = self._create_sample_hybrid(size)
        elif topology == 'fully_connected':
            base_network = self._create_sample_fully_connected(size)
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # Count initial parameters
        initial_capacity = self._count_parameters(base_network)
        
        # Calculate scaling factor based on topology and experiment type
        if experiment_type.startswith('match_'):
            # For capacity matching, use topology-specific scaling
            if topology == 'fully_connected':
                # Reduce base scaling for fully connected
                scale_factor = (target_capacity / (2.05 * (size * size))) ** 0.5 * 0.7  # Added 0.7 factor
            elif topology == 'small_world':
                # Keep current small_world scaling as it works well for smaller networks
                multiplier = 1.2 + 5.0 * (target_capacity / (target_capacity + 1000))
                scale_factor = target_capacity / (0.30 * size**1.92 * multiplier)
            elif topology == 'modular':
                # Reduce base scaling for modular
                num_modules = max(2, size // 20)  # Number of modules
                module_size = size // num_modules
                scale_factor = target_capacity / (2.05 * (size * module_size)) * 0.8  # Added 0.8 factor
            elif topology == 'hybrid':
                # Reduce base scaling for hybrid
                multiplier = 1.1 + 0.25 * (target_capacity / (target_capacity + 3000))
                scale_factor = target_capacity / (11.03 * size**1.25 * multiplier) * 0.75  # Added 0.75 factor
        else:
            # For same size, use topology-specific scaling
            if topology == 'fully_connected':
                scale_factor = (target_capacity / (2.05 * (size * size))) ** 0.5 * 0.7
            elif topology == 'small_world':
                multiplier = 1.2 + 5.0 * (target_capacity / (target_capacity + 1000))
                scale_factor = target_capacity / (0.30 * size**1.92 * multiplier)
            elif topology == 'modular':
                num_modules = max(2, size // 20)
                module_size = size // num_modules
                scale_factor = target_capacity / (2.05 * (size * module_size)) * 0.8
            elif topology == 'hybrid':
                multiplier = 1.1 + 0.25 * (target_capacity / (target_capacity + 3000))
                scale_factor = target_capacity / (11.03 * size**1.25 * multiplier) * 0.75
        
        # Scale network size while preserving topology
        scaled_size = max(1, int(size * scale_factor))
        
        # Create network with scaled size
        if topology == 'small_world':
            network = self._create_sample_small_world(scaled_size)
        elif topology == 'modular':
            network = self._create_sample_modular(scaled_size)
        elif topology == 'hybrid':
            network = self._create_sample_hybrid(scaled_size)
        elif topology == 'fully_connected':
            network = self._create_sample_fully_connected(scaled_size)
        
        # Verify final parameter count
        final_capacity = self._count_parameters(network)
        print(f"\nCreating {topology} network for {experiment_type}:")
        print(f"Initial size: {size}, Scaled size: {scaled_size}")
        print(f"Initial capacity: {initial_capacity}")
        print(f"Target capacity: {target_capacity}")
        print(f"Final capacity: {final_capacity}")
        print(f"Capacity match: {abs(final_capacity - target_capacity) <= 1}\n")
        
        return network
    
    def _create_sample_fully_connected(self, size: int) -> torch.nn.Module:
        """Create a sample fully connected network."""
        # Fully connected network has all possible connections
        network = torch.nn.Sequential(
            torch.nn.Linear(size, size),  # Input to hidden
            torch.nn.ReLU(),
            torch.nn.Linear(size, size)   # Hidden to output
        )
        return network
    
    def _create_sample_small_world(self, size: int) -> torch.nn.Module:
        """Create a sample small world network."""
        # Small world has sparse local connections with some long-range connections
        k = max(2, size // 10)  # Number of local connections
        network = torch.nn.Sequential(
            torch.nn.Linear(size, k),     # Sparse input connections
            torch.nn.ReLU(),
            torch.nn.Linear(k, size)      # Sparse output connections
        )
        return network
    
    def _create_sample_modular(self, size: int) -> torch.nn.Module:
        """Create a sample modular network."""
        # Modular network has dense connections within modules
        num_modules = max(2, size // 20)  # Number of modules
        module_size = size // num_modules
        network = torch.nn.Sequential(
            torch.nn.Linear(size, module_size),  # Input to first module
            torch.nn.ReLU(),
            torch.nn.Linear(module_size, size)   # Module to output
        )
        return network
    
    def _create_sample_hybrid(self, size: int) -> torch.nn.Module:
        """Create a sample hybrid network."""
        # Hybrid combines small world and modular characteristics
        k = max(2, size // 10)  # Number of local connections
        num_modules = max(2, size // 20)  # Number of modules
        module_size = size // num_modules
        network = torch.nn.Sequential(
            torch.nn.Linear(size, k),           # Sparse input connections
            torch.nn.ReLU(),
            torch.nn.Linear(k, module_size),    # Local to module
            torch.nn.ReLU(),
            torch.nn.Linear(module_size, size)  # Module to output
        )
        return network
    
    def get_budget(self, experiment_type: str, topology: str, size: int) -> int:
        """Get the budget for a specific configuration."""
        return self.budgets[experiment_type][topology][size]
    
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
        'match_small_world':     {25: 0.6, 50: 0.4, 100: 0.3},
        'match_modular':         {25: 1.0, 50: 1.0, 100: 1.0},
        'match_hybrid':          {25: 0.6, 50: 0.4, 100: 0.3},
    },
    'hybrid': {
        'match_fully_connected': {25: 1.2, 50: 1.2, 100: 1.2},
        'match_small_world':     {25: 1.2, 50: 1.2, 100: 1.1},
        'match_modular':         {25: 1.2, 50: 1.2, 100: 1.2},
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
    if experiment_type == 'same_size':
        return size

    # Table-driven scaling for all topologies
    if topology in SCALING_TABLE and experiment_type in SCALING_TABLE[topology]:
        size_bin = get_closest_bin(size)
        scale = SCALING_TABLE[topology][experiment_type].get(size_bin, 1.0)
        # Use the same base formula as before, but multiply by the table scale
        if topology == 'modular':
            num_modules = max(2, size // 10)
            module_size = size // num_modules
            base = (target_capacity / (2.05 * (size * module_size))) ** 0.5
            scale_factor = base * scale
        elif topology == 'hybrid':
            base = target_capacity / (11.03 * size**1.25)
            scale_factor = base * scale
        elif topology == 'small_world':
            if size >= 100:
                base = (target_capacity / (0.12 * size**2.1)) ** 0.5
            elif size >= 50:
                base = (target_capacity / (0.13 * size**2.1)) ** 0.5
            else:
                base = (target_capacity / (0.15 * size**2.1)) ** 0.5
            scale_factor = base * scale
        elif topology == 'fully_connected':
            base = (target_capacity / (2.05 * (size * size))) ** 0.5
            scale_factor = base * scale
        return int(size * scale_factor)

    raise ValueError(f"Unknown topology: {topology}") 