from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import networkx as nx

@dataclass
class ParameterBudgetCalculator:
    """Pre-computes and manages parameter budgets for different topologies and experiment types."""
    
    config: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize the calculator with base parameters."""
        self.base_size = min(self.config['network_sizes'])
        self.topologies = ['small_world', 'modular', 'hybrid']
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
            else:  # hybrid
                network = self._create_sample_hybrid(self.base_size)
            
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
        if experiment_type == 'match_hybrid':
            target_capacity = self.base_capacities['hybrid']
        elif experiment_type == 'match_small_world':
            target_capacity = self.base_capacities['small_world']
        elif experiment_type == 'match_modular':
            target_capacity = self.base_capacities['modular']
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        # Scale the target capacity by the ratio of current size to base size
        scale_factor = size / self.base_size
        return int(target_capacity * scale_factor)
    
    def create_network(self, topology: str, size: int, experiment_type: str) -> torch.nn.Module:
        """Create a network with the specified topology and size, respecting the experiment type's budget."""
        # Get target budget for this configuration
        target_budget = self.get_budget(experiment_type, topology, size)
        
        # Create base network
        if topology == 'small_world':
            network = self._create_sample_small_world(size)
        elif topology == 'modular':
            network = self._create_sample_modular(size)
        elif topology == 'hybrid':
            network = self._create_sample_hybrid(size)
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # Count initial parameters
        initial_params = self._count_parameters(network)
        print(f"\nCreating {topology} network for {experiment_type}:")
        print(f"Initial parameters: {initial_params}")
        print(f"Target budget: {target_budget}")
        
        # Adjust parameters to match budget
        if initial_params > target_budget:
            network = self._reduce_parameters(network, target_budget)
        elif initial_params < target_budget:
            network = self._add_parameters(network, target_budget)
        
        # Verify final parameter count
        final_params = self._count_parameters(network)
        print(f"Final parameters: {final_params}")
        print(f"Budget match: {abs(final_params - target_budget) <= 1}\n")
        
        return network
    
    def _reduce_parameters(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Reduce network parameters to match target budget."""
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
    
    def _add_parameters(self, network: torch.nn.Module, target_budget: int) -> torch.nn.Module:
        """Add parameters to network to match target budget."""
        current_params = self._count_parameters(network)
        params_to_add = target_budget - current_params
        
        if params_to_add <= 0:
            return network
        
        # Add random edges
        for param in network.parameters():
            if param.requires_grad:
                # Get number of zero elements
                zero_mask = param == 0
                num_zeros = zero_mask.sum().item()
                
                if num_zeros > 0:
                    # Calculate how many edges to add to this parameter
                    edges_for_param = min(params_to_add, num_zeros)
                    
                    # Generate random values
                    if self.config['parameter_budget']['padding_strategy'] == 'random':
                        new_values = torch.randn(edges_for_param) * 0.01
                    else:  # zero
                        new_values = torch.zeros(edges_for_param)
                    
                    # Get random indices of zero elements
                    zero_indices = torch.nonzero(zero_mask.view(-1))[torch.randperm(num_zeros)[:edges_for_param]]
                    
                    # Add new edges
                    param.view(-1)[zero_indices] = new_values
                    
                    params_to_add -= edges_for_param
                    if params_to_add <= 0:
                        break
        
        return network
    
    def get_budget(self, experiment_type: str, topology: str, size: int) -> int:
        """Get the pre-computed budget for a specific configuration."""
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
    
    def _create_sample_small_world(self, size: int) -> torch.nn.Module:
        """Create a sample small world network."""
        network = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            torch.nn.ReLU(),
            torch.nn.Linear(size, size)
        )
        return network
    
    def _create_sample_modular(self, size: int) -> torch.nn.Module:
        """Create a sample modular network."""
        network = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            torch.nn.ReLU(),
            torch.nn.Linear(size, size)
        )
        return network
    
    def _create_sample_hybrid(self, size: int) -> torch.nn.Module:
        """Create a sample hybrid network."""
        network = torch.nn.Sequential(
            torch.nn.Linear(size, size),
            torch.nn.ReLU(),
            torch.nn.Linear(size, size)
        )
        return network
    
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