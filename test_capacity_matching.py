from src.utils.parameter_budget import ParameterBudgetCalculator
from config.curriculum_config import CurriculumConfig

def test_capacity_matching():
    # Create config
    config = CurriculumConfig().to_dict()
    
    # Create calculator
    calculator = ParameterBudgetCalculator(config)
    
    # Test each experiment type
    experiment_types = ['same_size', 'match_hybrid', 'match_small_world', 'match_modular']
    topologies = ['small_world', 'modular', 'hybrid']
    size = 100  # Use smallest size for testing
    
    print("\nTesting capacity matching across experiment types:")
    print("=" * 80)
    
    for exp_type in experiment_types:
        print(f"\nExperiment type: {exp_type}")
        print("-" * 40)
        
        # Create networks for each topology
        networks = {}
        for topology in topologies:
            network = calculator.create_network(topology, size, exp_type)
            networks[topology] = network
        
        # Print parameter counts
        print("\nParameter counts:")
        for topology, network in networks.items():
            params = calculator._count_parameters(network)
            print(f"{topology}: {params} parameters")
        
        # Verify matching
        if exp_type == 'same_size':
            print("\nVerifying same size:")
            params = [calculator._count_parameters(network) for network in networks.values()]
            print(f"All networks have same size: {len(set(params)) == 1}")
        else:
            target = '_'.join(exp_type.split('_')[1:])
            target_params = calculator._count_parameters(networks[target])
            print(f"\nVerifying matching to {target}:")
            for topology, network in networks.items():
                if topology != target:
                    params = calculator._count_parameters(network)
                    print(f"{topology} matches {target}: {abs(params - target_params) <= 1}")

if __name__ == "__main__":
    test_capacity_matching() 