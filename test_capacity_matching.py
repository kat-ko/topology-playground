import torch
from src.utils.parameter_budget import ParameterBudgetCalculator, calculate_network_size
from config.test_curriculum_config import TestCurriculumConfig
import numpy as np

def test_parameter_matching():
    """Test parameter matching across different topologies."""
    # Create test configuration
    config = TestCurriculumConfig().to_dict()
    
    # Modify config for testing
    config['network_sizes'] = [50]  # Single size for testing
    config['num_layers'] = [2]  # Two layers for testing
    config['experiment_types'] = [
        'match_fully_connected',
        'match_small_world',
        'match_modular',
        'match_hybrid'
    ]
    
    # Create calculator
    calculator = ParameterBudgetCalculator(config)
    
    # Test each topology
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    size = config['network_sizes'][0]
    
    print("\nTesting Parameter Matching")
    print("=========================")
    
    for experiment_type in config['experiment_types']:
        print(f"\nExperiment Type: {experiment_type}")
        print("-" * 50)
        
        # Get target capacity
        target_capacity = calculator.get_budget(experiment_type, 'fully_connected', size)
        print(f"Target Capacity: {target_capacity}")
        
        # Test each topology
        for topology in topologies:
            # Create network
            network = calculator.create_network(topology, size, experiment_type)
            
            # Count parameters
            actual_capacity = sum(p.numel() for p in network.parameters() if p.requires_grad)
            
            # Print results
            print(f"\nTopology: {topology}")
            print(f"Actual Capacity: {actual_capacity}")
            print(f"Match: {abs(actual_capacity - target_capacity) <= 1}")
            
            # Print network structure
            print("\nNetwork Structure:")
            for name, param in network.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.shape}")

# Mapping from experiment type to reference topology
EXPERIMENT_TYPE_TO_REF = {
    'match_fully_connected': 'fully_connected',
    'match_small_world': 'small_world',
    'match_modular': 'modular',
    'match_hybrid': 'hybrid',
}

def test_capacity_matching():
    # Test cases for different node sizes
    node_sizes = [25, 50, 100]
    topologies = ['fully_connected', 'small_world', 'modular', 'hybrid']
    experiment_types = ['same_size', 'match_fully_connected', 'match_small_world', 'match_modular', 'match_hybrid']
    
    print("\nTesting capacity matching for different node sizes:")
    print("-" * 80)
    
    for size in node_sizes:
        print(f"\nNode size: {size}")
        print("-" * 40)
        
        # First calculate target capacities for each topology
        target_capacities = {}
        for topology in topologies:
            # Calculate what the parameter count would be for this topology at this size
            scaled_size = calculate_network_size(size, topology, 'same_size', size)
            # For fully connected, this is approximately 2.05 * size^2
            if topology == 'fully_connected':
                target_capacities[topology] = int(2.05 * size * size)
            # For small world, this is approximately 0.30 * size**1.92
            elif topology == 'small_world':
                target_capacities[topology] = int(0.30 * size**1.92)
            # For modular, this is approximately 2.05 * size * (size/num_modules)
            elif topology == 'modular':
                num_modules = max(2, size // 20)
                module_size = size // num_modules
                target_capacities[topology] = int(2.05 * size * module_size)
            # For hybrid, this is approximately 11.03 * size**1.25
            elif topology == 'hybrid':
                target_capacities[topology] = int(11.03 * size**1.25)
        
        # Now test capacity matching for each experiment type
        for exp_type in experiment_types:
            print(f"\nExperiment: {exp_type}")
            if exp_type == 'same_size':
                for topology in topologies:
                    print(f"  {topology:16s} | size={size:4d} | params={target_capacities[topology]:6d}")
            else:
                ref_topology = EXPERIMENT_TYPE_TO_REF[exp_type]
                target_capacity = target_capacities[ref_topology]
                for topology in topologies:
                    if topology == ref_topology:
                        # Reference topology: use reference size
                        scaled_size = size
                        params = target_capacities[ref_topology]
                    else:
                        scaled_size = calculate_network_size(size, topology, exp_type, target_capacity)
                        # Recompute parameter count at this scaled size
                        if topology == 'fully_connected':
                            params = int(2.05 * scaled_size * scaled_size)
                        elif topology == 'small_world':
                            params = int(0.30 * scaled_size**1.92)
                        elif topology == 'modular':
                            num_modules = max(2, scaled_size // 20)
                            module_size = scaled_size // num_modules
                            params = int(2.05 * scaled_size * module_size)
                        elif topology == 'hybrid':
                            params = int(11.03 * scaled_size**1.25)
                    print(f"  {topology:16s} | size={scaled_size:4d} | params={params:6d} | target={target_capacity:6d}")

if __name__ == "__main__":
    test_parameter_matching()
    test_capacity_matching() 