#!/usr/bin/env python3
"""
Weights & Biases Sweep Configuration for Triple-Task Training

This file defines clean, simplified sweep configurations for triple-task training only.
Focusing on sweeps 4-5 (fixed network sizes) and 4-6 (fixed capacities).
"""

import wandb

def create_fixed_network_sizes_triple_task_sweep():
    """
    Create sweep configuration for comparing topologies with fixed network sizes (Sweep 4-5).
    Uses normalized metrics as primary optimization target.
    
    Returns:
        dict: Fixed network sizes comparison sweep configuration
    """
    
    return {
        'program': 'topologies_triple_task_training_sweep.py',
        'method': 'grid',  # Grid search for systematic comparison
        'metric': {
            'name': 'fixed_network_sizes_triple_task/normalized/final_normalized_score',
            'goal': 'maximize'
        },
        'parameters': {
            # ============================================================================
            # PRIMARY VARIABLES: TOPOLOGY TYPE AND NETWORK SIZE
            # ============================================================================
            'topology_type': {
                'values': ['modular', 'small_world', 'hybrid', 'fully_connected']
            },
            'hidden_size': {
                'values': [64, 128, 256]  # 3 sizes as requested
            },
            
            # ============================================================================
            # TASK SEQUENCE VARIATION (3 orders as requested)
            # ============================================================================
            'task_order': {
                'values': [
                    'CartPole-v1_Acrobot-v1_LunarLander-v2',
                    'Acrobot-v1_LunarLander-v2_CartPole-v1',
                    'LunarLander-v2_CartPole-v1_Acrobot-v1'
                ]
            },
            
            # ============================================================================
            # FIXED TRAINING PARAMETERS (Single values for faster evaluation)
            # ============================================================================
            'learning_rate': {'value': 3e-4},
            'batch_size': {'value': 64},
            'n_steps': {'value': 2048},
            'n_epochs': {'value': 10},
            'gamma': {'value': 0.99},
            'gae_lambda': {'value': 0.95},
            'clip_range': {'value': 0.2},
            'ent_coef': {'value': 0.01},
            'max_grad_norm': {'value': 0.5},
            'activation': {'value': 'relu'},
            'dropout': {'value': 0.0},
            'num_layers': {'value': 3},

            # ============================================================================
            # FIXED TOPOLOGY PARAMETERS
            # ============================================================================
            'small_world_k': {'value': 4},
            'small_world_p': {'value': 0.1},
            'modular_num_modules': {'value': 4},
            'modular_inter_module_prob': {'value': 0.05},
            'modular_intra_module_prob': {'value': 0.7},
            'hybrid_num_modules': {'value': 4},
            'hybrid_k': {'value': 4},
            'hybrid_p': {'value': 0.1},
            'hybrid_inter_module_prob': {'value': 0.05},
            
            # ============================================================================
            # SEED PARAMETER FOR REPRODUCIBILITY
            # ============================================================================
            'seed': {
                'values': [42, 123, 456, 789, 101112]  # 5 seeds for statistical robustness
            },
            
            # ============================================================================
            # EVALUATION PARAMETERS
            # ============================================================================
            'total_timesteps': {'value': 600000},
            'n_eval_episodes': {'value': 15},
        }
    }
    
def create_fixed_capacities_triple_task_sweep():
    """
    Create sweep configuration for comparing topologies with fixed parameter capacities (Sweep 4-6).
    Uses normalized metrics as primary optimization target.
    
    Returns:
        dict: Fixed capacities comparison sweep configuration
    """
    
        return {
        'program': 'topologies_triple_task_training_sweep.py',
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
            'name': 'fixed_capacities_triple_task/normalized/final_normalized_score',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
            # PRIMARY VARIABLES: TOPOLOGY TYPE AND TARGET CAPACITY
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'target_capacity': {
                'values': [1000, 5000, 10000]  # 3 capacities as requested
                },
                
                # ============================================================================
            # TASK SEQUENCE VARIATION (3 orders as requested)
                # ============================================================================
            'task_order': {
                'values': [
                    'CartPole-v1_Acrobot-v1_LunarLander-v2',
                    'Acrobot-v1_LunarLander-v2_CartPole-v1',
                    'LunarLander-v2_CartPole-v1_Acrobot-v1'
                ]
                },
                
                # ============================================================================
            # FIXED TRAINING PARAMETERS (Single values for faster evaluation)
                # ============================================================================
                'learning_rate': {'value': 3e-4},
                'batch_size': {'value': 64},
                'n_steps': {'value': 2048},
            'n_epochs': {'value': 10},
                'gamma': {'value': 0.99},
            'gae_lambda': {'value': 0.95},
            'clip_range': {'value': 0.2},
            'ent_coef': {'value': 0.01},
            'max_grad_norm': {'value': 0.5},
            'activation': {'value': 'relu'},
            'dropout': {'value': 0.0},
            'num_layers': {'value': 3},
                
                # ============================================================================
            # FIXED TOPOLOGY PARAMETERS
                # ============================================================================
            'small_world_k': {'value': 4},
            'small_world_p': {'value': 0.1},
            'modular_num_modules': {'value': 4},
            'modular_inter_module_prob': {'value': 0.05},
            'modular_intra_module_prob': {'value': 0.7},
            'hybrid_num_modules': {'value': 4},
            'hybrid_k': {'value': 4},
            'hybrid_p': {'value': 0.1},
            'hybrid_inter_module_prob': {'value': 0.05},

# ============================================================================
            # SEED PARAMETER FOR REPRODUCIBILITY
# ============================================================================
            'seed': {
                'values': [42, 123, 456, 789, 101112]  # 5 seeds for statistical robustness
            },
            
            # ============================================================================
            # EVALUATION PARAMETERS
            # ============================================================================
            'total_timesteps': {'value': 600000},
            'n_eval_episodes': {'value': 15},
        }
    }
    
def create_continual_learning_sweep():
    """
    Create sweep configuration for continual learning with observation shifts.
    
    Single continuous lifelong run per task with piecewise-constant observation shifts.
    No tuning between shifts - training proceeds unchanged across the whole stream.
    
    Returns:
        dict: Continual learning sweep configuration
    """
    
    return {
        'program': 'topologies_continual_task_training_sweep.py',
        'method': 'grid',  # Grid search for systematic comparison
        'metric': {
            'name': 'continual_learning/cumulative_lifetime_reward',
            'goal': 'maximize'
        },
        'parameters': {
            # ============================================================================
            # PRIMARY VARIABLES: TOPOLOGY TYPE AND NETWORK SIZE
            # ============================================================================
            'topology_type': {
                'values': ['modular', 'small_world', 'hybrid', 'fully_connected']
            },
            'hidden_size': {
                'values': [64, 128, 256]  # 3 sizes as requested
            },
            
            # ============================================================================
            # SINGLE TASK PER RUN (continual learning)
            # ============================================================================
            'task_name': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2']
            },
            
            # ============================================================================
            # CONTINUAL LEARNING PARAMETERS
            # ============================================================================
            'total_lifetime_steps': {'value': 3000},  # Very short as requested
            'segment_length': {'value': 200},  # Fixed at 200 steps
            'shift_range': {'value': [0, 2]},  # Fixed range [0, 2]
            'continual_learning': {'value': True},  # Enable continual learning mode
            
            # ============================================================================
            # FIXED TRAINING PARAMETERS (unchanged across shifts)
            # ============================================================================
            'learning_rate': {'value': 3e-4},
            'batch_size': {'value': 64},
            'n_steps': {'value': 2048},
            'n_epochs': {'value': 10},
            'gamma': {'value': 0.99},
            'gae_lambda': {'value': 0.95},
            'clip_range': {'value': 0.2},
            'ent_coef': {'value': 0.01},
            'max_grad_norm': {'value': 0.5},
            'activation': {'value': 'relu'},
            'dropout': {'value': 0.0},
            'num_layers': {'value': 3},
            
            # ============================================================================
            # FIXED TOPOLOGY PARAMETERS
            # ============================================================================
            'small_world_k': {'value': 4},
            'small_world_p': {'value': 0.1},
            'modular_num_modules': {'value': 4},
            'modular_inter_module_prob': {'value': 0.05},
            'modular_intra_module_prob': {'value': 0.7},
            'hybrid_num_modules': {'value': 4},
            'hybrid_k': {'value': 4},
            'hybrid_p': {'value': 0.1},
            'hybrid_inter_module_prob': {'value': 0.05},
            
            # ============================================================================
            # SEED PARAMETER FOR REPRODUCIBILITY (15 seeds as requested)
            # ============================================================================
            'seed': {
                'values': [42, 123, 456, 789, 101112, 131415, 161718, 192021, 222324, 252627, 282930, 313233, 343536, 373839, 404142]
            },
        }
    }

            # ============================================================================
# INDIVIDUAL RUN CONFIGURATIONS (Reusing sweep parameter structure)
            # ============================================================================

def create_single_run_config():
    """
    Create configuration for a single individual run.
    Reuses exact same parameter structure as sweep configs.
    
    Returns:
        dict: Single run configuration with all parameters
    """
    # Use the same parameters as the fixed network sizes sweep
    sweep_config = create_fixed_network_sizes_triple_task_sweep()
    
    # Extract parameters and convert to individual run format
    params = sweep_config['parameters']
    
    config = {}
    for key, value_dict in params.items():
        if 'values' in value_dict:
            # For list parameters, take the first value
            config[key] = value_dict['values'][0]
        elif 'value' in value_dict:
            # For single parameters, extract the value
            config[key] = value_dict['value']
    
    return config

def create_batch_run_config():
    """
    Create configuration for batch runs with parameter variations.
    Reuses exact same parameter structure as sweep configs.
    
    Returns:
        dict: Batch run configuration with parameter lists
    """
    # Use the same parameters as the fixed network sizes sweep
    sweep_config = create_fixed_network_sizes_triple_task_sweep()
    
    # Extract parameters and convert to individual run format
    params = sweep_config['parameters']
    
    config = {}
    for key, value_dict in params.items():
        if 'values' in value_dict:
            # For list parameters, use all values
            config[key] = value_dict['values']
        elif 'value' in value_dict:
            # For single parameters, extract the value
            config[key] = value_dict['value']
    
    return config

def create_fixed_capacity_batch_config():
    """
    Create configuration for batch runs with fixed capacities.
    Reuses exact same parameter structure as sweep configs.
    
    Returns:
        dict: Fixed capacity batch run configuration
    """
    # Use the same parameters as the fixed capacities sweep
    sweep_config = create_fixed_capacities_triple_task_sweep()
    
    # Extract parameters and convert to individual run format
    params = sweep_config['parameters']
    
    config = {}
    for key, value_dict in params.items():
        if 'values' in value_dict:
            # For list parameters, use all values
            config[key] = value_dict['values']
        elif 'value' in value_dict:
            # For single parameters, extract the value
            config[key] = value_dict['value']
    
    return config

def get_config_by_name(config_name):
    """
    Get configuration by name.
    
    Args:
        config_name (str): Name of the configuration to load
                          ('single', 'batch', or 'fixed_capacity_batch')
    
    Returns:
        dict: Configuration dictionary
        
    Raises:
        ValueError: If config_name is not recognized
    """
    configs = {
        'single': create_single_run_config,
        'batch': create_batch_run_config,
        'fixed_capacity_batch': create_fixed_capacity_batch_config,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]()

def generate_parameter_combinations(config):
    """
    Generate all possible combinations of parameters from a configuration.
    
    Args:
        config (dict): Configuration dictionary with single values or lists
    
    Returns:
        list: List of configuration dictionaries, one for each combination
    """
    import itertools
    
    # Separate list parameters from single values
    list_params = {}
    single_params = {}
    
    for key, value in config.items():
        if isinstance(value, list):
            list_params[key] = value
        else:
            single_params[key] = value
    
    # If no list parameters, return single config
    if not list_params:
        return [config]
    
    # Generate all combinations of list parameters
    param_names = list(list_params.keys())
    param_values = list(list_params.values())
    
    combinations = []
    for combination in itertools.product(*param_values):
        # Create config for this combination
        combo_config = single_params.copy()
        for i, param_name in enumerate(param_names):
            combo_config[param_name] = combination[i]
        combinations.append(combo_config)
    
    return combinations

if __name__ == "__main__":
    # Example usage
    print("Available triple-task sweep configurations:")
    print("1. Fixed network sizes sweep: create_fixed_network_sizes_triple_task_sweep()")
    print("2. Fixed capacities sweep: create_fixed_capacities_triple_task_sweep()")
    print("\nAvailable individual run configurations:")
    print("3. Single run: create_single_run_config()")
    print("4. Batch run: create_batch_run_config()")
    print("5. Fixed capacity batch: create_fixed_capacity_batch_config()")
    
    # Create and print sample configurations
    fixed_sizes_config = create_fixed_network_sizes_triple_task_sweep()
    print(f"\nFixed network sizes sweep configuration:")
    print(f"Method: {fixed_sizes_config['method']}")
    print(f"Metric: {fixed_sizes_config['metric']}")
    print(f"Number of parameters: {len(fixed_sizes_config['parameters'])}")
    print(f"Topology types: {fixed_sizes_config['parameters']['topology_type']['values']}")
    print(f"Hidden sizes: {fixed_sizes_config['parameters']['hidden_size']['values']}")
    print(f"Task orders: {fixed_sizes_config['parameters']['task_order']['values']}")
    
    fixed_capacities_config = create_fixed_capacities_triple_task_sweep()
    print(f"\nFixed capacities sweep configuration:")
    print(f"Method: {fixed_capacities_config['method']}")
    print(f"Number of parameters: {len(fixed_capacities_config['parameters'])}")
    print(f"Topology types: {fixed_capacities_config['parameters']['topology_type']['values']}")
    print(f"Target capacities: {fixed_capacities_config['parameters']['target_capacity']['values']}")
    print(f"Task orders: {fixed_capacities_config['parameters']['task_order']['values']}")
    
    # Test individual run configs
    single_config = create_single_run_config()
    print(f"\nSingle run configuration:")
    print(f"Topology: {single_config['topology_type']}")
    print(f"Hidden size: {single_config['hidden_size']}")
    print(f"Task order: {single_config['task_order']}")
    
    batch_config = create_batch_run_config()
    print(f"\nBatch run configuration:")
    print(f"Topologies: {batch_config['topology_type']}")
    print(f"Hidden sizes: {batch_config['hidden_size']}")
    print(f"Task orders: {batch_config['task_order']}")
    
    combinations = generate_parameter_combinations(batch_config)
    print(f"Total batch combinations: {len(combinations)}")