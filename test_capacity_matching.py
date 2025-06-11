import torch
from src.utils.parameter_budget import ParameterBudgetCalculator
from config.test_curriculum_config import TestCurriculumConfig
import numpy as np

def test_parameter_matching():
    """Test parameter matching across different topologies using the exact same logic as main_curriculum."""
    # Create test configuration
    config = TestCurriculumConfig().to_dict()
    
    # Create calculator (same as used in main_curriculum)
    calculator = ParameterBudgetCalculator(config)
    
    # Test configurations from main_curriculum
    sizes = [25, 50, 100]
    seeds = [42, 123, 456]
    layers = [1, 2, 3]  # Test different numbers of layers
    network_types = ['ffn', 'rnn']
    
    # Test topologies
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    
    # Test experiment types
    experiment_types = [
        'same_size',
        'match_fully_connected',
        'match_small_world',
        'match_modular',
        'match_hybrid'
    ]
    
    print("\nTesting Parameter Matching (using main curriculum logic)")
    print("======================================================")
    
    # Test each combination
    for size in sizes:
        print(f"\nTesting size: {size}")
        print("=" * 20)
        
        for topology in topologies:
            print(f"\nTopology: {topology}")
            print("-" * 20)
            
            for exp_type in experiment_types:
                for seed in seeds:
                    for num_layers in layers:
                        for network_type in network_types:
                            # Set random seed
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            
                            # Update config with current number of layers
                            config['num_layers'] = [num_layers]
                            calculator = ParameterBudgetCalculator(config)
                            
                            # Use the exact same network creation logic as main_curriculum
                            network = calculator.create_network(
                                topology=topology,
                                size=size,
                                experiment_type=exp_type
                            )
                            
                            # Get target capacity
                            target_capacity = calculator._compute_budget(exp_type, topology, size)
                            
                            # Count actual parameters
                            actual_capacity = sum(p.numel() for p in network.parameters() if p.requires_grad)
                            
                            # Calculate divergence
                            divergence = abs(actual_capacity - target_capacity) / target_capacity * 100 if target_capacity > 0 else 0
                            
                            # Print results
                            print(f"\nConfiguration:")
                            print(f"  Size: {size}")
                            print(f"  Seed: {seed}")
                            print(f"  Layers: {num_layers}")
                            print(f"  Network Type: {network_type}")
                            print(f"  Experiment: {exp_type}")
                            print(f"  Target capacity: {target_capacity}")
                            print(f"  Actual capacity: {actual_capacity}")
                            print(f"  Divergence: {divergence:.2f}%")
                            if divergence <= 5.0:
                                print("✓ Verification passed")
                            else:
                                print("⚠ Verification failed")
                            
                            # Print network structure and topology characteristics
                            print("\nNetwork structure:")
                            for name, param in network.named_parameters():
                                if param.requires_grad:
                                    print(f"  {name}: {param.shape}")
                            
                            # Print topology characteristics
                            print("\nTopology characteristics:")
                            if hasattr(network, 'graph'):
                                print(f"  Number of nodes: {network.graph.number_of_nodes()}")
                                print(f"  Number of edges: {network.graph.number_of_edges()}")
                                if hasattr(network, 'input_nodes'):
                                    print(f"  Input nodes: {len(network.input_nodes)}")
                                if hasattr(network, 'output_nodes'):
                                    print(f"  Output nodes: {len(network.output_nodes)}")
                            
                            print("-" * 40)

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