def main():
    """Run curriculum learning experiments with different configurations."""
    # Load base configuration
    config = load_config('config/curriculum_config.py')
    
    # Define variations to test
    network_sizes = [100, 200, 500, 1000]
    network_types = ['small_world', 'modular', 'hybrid']
    experiment_types = [
        'same_size',  # All topologies with same node size
        'match_hybrid',  # All topologies matched to hybrid capacity
        'match_small_world',  # All topologies matched to small world capacity
        'match_modular'  # All topologies matched to modular capacity
    ]
    algorithms = ['ppo', 'sac', 'a2c']
    node_selection_strategies = ['random', 'centrality_based', 'distance_based', 'module_based']
    num_io_nodes = [5, 10, 20]
    task_memory_sizes = [4, 6, 8, 10]
    
    # Network parameter variations
    network_params = {
        'ffn': {
            'activation': ['relu', 'tanh', 'sigmoid'],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128]
        },
        'rnn': {
            'hidden_size': [32, 64, 128],
            'sequence_length': [10, 20, 30],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [32, 64, 128]
        }
    }
    
    # Initialize results storage
    results = {}
    
    # Run experiments for each configuration
    for size in network_sizes:
        for network_type in network_types:
            for exp_type in experiment_types:
                for algo in algorithms:
                    for strategy in node_selection_strategies:
                        for num_io in num_io_nodes:
                            for memory_size in task_memory_sizes:
                                # Update configuration
                                config['network_sizes'] = [size]
                                config['network_types'] = [network_type]
                                config['experiment_type'] = exp_type
                                config['node_selection_strategy'] = strategy
                                config['num_io_nodes'] = num_io
                                config['task_memory_size'] = memory_size
                                
                                # Update network parameters
                                for net_type, params in network_params.items():
                                    for param_name, values in params.items():
                                        for value in values:
                                            config['network_params'][net_type][param_name] = value
                                            
                                            # Create experiment key
                                            exp_key = f"{size}_{network_type}_{exp_type}_{algo}_{strategy}_{num_io}_{memory_size}_{net_type}_{param_name}_{value}"
                                            
                                            # Run experiment
                                            print(f"\nRunning experiment: {exp_key}")
                                            runner = CurriculumRunner(config)
                                            result = runner.run_curriculum([algo], [exp_type])
                                            
                                            # Store results
                                            results[exp_key] = {
                                                'config': config.copy(),
                                                'result': result
                                            }
                                            
                                            # Save results periodically
                                            if len(results) % 10 == 0:
                                                save_results(results, 'results/curriculum_results.json')
    
    # Save final results
    save_results(results, 'results/curriculum_results.json')
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main() 