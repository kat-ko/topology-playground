#!/usr/bin/env python3
"""
Weights & Biases Sweep Configuration for Topology Network Hyperparameter Optimization

This file defines sweep configurations for optimizing hyperparameters in the topology training scripts.
"""

import wandb

def create_sweep_config(program='topologies--single-task-training-sweep.py'):
    """
    Create a comprehensive sweep configuration for topology network hyperparameter optimization.
    This is for single-task training with cross-task evaluation (tests on all tasks).
    
    Args:
        program (str): The training script to run (default: single-task training with cross-task evaluation)
    
    Returns:
        dict: Sweep configuration dictionary
    """
    
    sweep_config = {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for efficient hyperparameter search
        'metric': {
            'name': 'testing/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # PPO TRAINING PARAMETERS
            # ============================================================================
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            },
            'n_steps': {
                'values': [1024, 2048, 4096, 8192]
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'n_epochs': {
                'values': [10, 15]
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 0.999
            },
            'gae_lambda': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            'clip_range': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'ent_coef': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-1
            },
            'max_grad_norm': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            
            # ============================================================================
            # NETWORK ARCHITECTURE PARAMETERS
            # ============================================================================
            'hidden_size': {
                'values': [64, 128, 256]
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            
            # ============================================================================
            # TOPOLOGY-SPECIFIC PARAMETERS
            # ============================================================================
            # Small World parameters
            'small_world_k': {
                'values': [2, 4, 6, 8]
            },
            'small_world_p': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3  # Keep in small-world range (0.1-0.3)
            },
            
            # Modular parameters
            'modular_num_modules': {
                'values': [2, 4, 6, 8]
            },
            'modular_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.2  # Keep low to maintain modularity
            },
            'modular_intra_module_prob': {
                'distribution': 'uniform',
                'min': 0.7,
                'max': 0.9  # Keep high to maintain modularity
            },
            
            # Hybrid parameters
            'hybrid_num_modules': {
                'values': [2, 4, 6, 8]
            },
            'hybrid_k': {
                'values': [2, 4, 6, 8]
            },
            'hybrid_p': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3  # Keep in small-world range (0.1-0.3)
            },
            'hybrid_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.2  # Keep low to maintain modularity
            },
            
            # ============================================================================
            # NETWORK PARAMETERS
            # ============================================================================
            'activation': {
                'values': ['relu', 'tanh', 'leaky_relu']
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            
            # ============================================================================
            # TRAINING CONFIGURATION
            # ============================================================================
            'total_timesteps': {
                'values': [500000, 700000]
            },
            'n_eval_episodes': {
                'values': [15]
            },
            
            # ============================================================================
            # TOPOLOGY SELECTION
            # ============================================================================
            'topology_type': {
                'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
            },
            
            # ============================================================================
            # TASK SELECTION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
        }
    }
    
    return sweep_config

def create_baseline_sweep_config(program='topologies--baseline-training-sweep.py'):
    """
    Create a baseline-specific sweep configuration for topology network verification.
    Focused on single-task training with same-task evaluation only (no cross-task testing).
    Uses the same parameter ranges as comprehensive sweeps.
    
    Args:
        program (str): The baseline training script to run
    
    Returns:
        dict: Baseline sweep configuration dictionary
    """
    
    sweep_config = {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for efficient hyperparameter search
        'metric': {
            'name': 'testing/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # PPO TRAINING PARAMETERS (Same as comprehensive)
            # ============================================================================
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            },
            'n_steps': {
                'values': [1024, 2048, 4096, 8192]
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'n_epochs': {
                'values': [10, 15]
            },
            'gamma': {
                'distribution': 'uniform',
                'min': 0.9,
                'max': 0.999
            },
            'gae_lambda': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            'clip_range': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3
            },
            'ent_coef': {
                'distribution': 'log_uniform_values',
                'min': 1e-4,
                'max': 1e-1
            },
            'max_grad_norm': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            
            # ============================================================================
            # NETWORK ARCHITECTURE PARAMETERS (Same as comprehensive)
            # ============================================================================
            'hidden_size': {
                'values': [64, 128, 256]
            },
            'num_layers': {
                'values': [1, 2, 3]
            },
            
            # ============================================================================
            # TOPOLOGY-SPECIFIC PARAMETERS (Same as comprehensive)
            # ============================================================================
            # Small World parameters
            'small_world_k': {
                'values': [2, 4, 6, 8]
            },
            'small_world_p': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3  # Keep in small-world range (0.1-0.3)
            },
            
            # Modular parameters
            'modular_num_modules': {
                'values': [2, 4, 6, 8]
            },
            'modular_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.2  # Keep low to maintain modularity
            },
            'modular_intra_module_prob': {
                'distribution': 'uniform',
                'min': 0.7,
                'max': 0.9  # Keep high to maintain modularity
            },
            
            # Hybrid parameters
            'hybrid_num_modules': {
                'values': [2, 4, 6, 8]
            },
            'hybrid_k': {
                'values': [2, 4, 6, 8]
            },
            'hybrid_p': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.3  # Keep in small-world range (0.1-0.3)
            },
            'hybrid_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.2  # Keep low to maintain modularity
            },
            
            # ============================================================================
            # NETWORK PARAMETERS (Same as comprehensive)
            # ============================================================================
            'activation': {
                'values': ['relu', 'tanh', 'leaky_relu']
            },
            'dropout': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            
            # ============================================================================
            # TRAINING CONFIGURATION (Same as comprehensive)
            # ============================================================================
            'total_timesteps': {
                'values': [500000, 700000]
            },
            'n_eval_episodes': {
                'values': [15]
            },
            
            # ============================================================================
            # TOPOLOGY SELECTION (Same as comprehensive)
            # ============================================================================
            'topology_type': {
                'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
            },
            
            # ============================================================================
            # TASK SELECTION (Same as comprehensive)
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
        }
    }
    
    return sweep_config

def create_baseline_focused_sweep_config(focus_area='topology_comparison', program='topologies--baseline-training-sweep.py'):
    """
    Create focused sweep configurations for baseline training with topology analysis.
    
    Baseline training uses same-task evaluation only (no cross-task testing).
    This ensures we test basic learning capability without generalization effects.
    
    Args:
        focus_area (str): Analysis type ('topology_comparison', 'topology_optimization', 'meta_analysis', 'capacity_matched')
        program (str): The training script to run
    
    Returns:
        dict: Focused sweep configuration for baseline training
    """
    
    if focus_area == 'topology_comparison':
        # Fair head-to-head topology comparison with standardized parameters
        return {
            'program': program,
            'method': 'grid',  # Grid search for fair comparison
            'metric': {
                'name': 'baseline/topology_comparison/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # PRIMARY VARIABLE: TOPOLOGY TYPE
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # STANDARDIZED ARCHITECTURE (Fixed for fair comparison)
                # ============================================================================
                'hidden_size': {'value': 128},  # Fixed for fair comparison
                'num_layers': {'value': 2},     # Fixed for fair comparison
                'activation': {'value': 'relu'}, # Fixed for fair comparison
                'dropout': {'value': 0.0},      # Fixed for fair comparison
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS (Fixed for fair comparison)
                # ============================================================================
                'learning_rate': {'value': 3e-4},    # Fixed for fair comparison
                'n_steps': {'value': 2048},          # Fixed for fair comparison
                'batch_size': {'value': 64},         # Fixed for fair comparison
                'n_epochs': {'value': 10},           # Fixed for fair comparison
                'gamma': {'value': 0.99},            # Fixed for fair comparison
                'gae_lambda': {'value': 0.95},       # Fixed for fair comparison
                'clip_range': {'value': 0.2},        # Fixed for fair comparison
                'ent_coef': {'value': 0.01},         # Fixed for fair comparison
                'max_grad_norm': {'value': 0.5},     # Fixed for fair comparison
                
                # ============================================================================
                # TASK VARIATION (To test generalization across tasks)
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (Optimized defaults)
                # ============================================================================
                'small_world_k': {'value': 4},           # Good default
                'small_world_p': {'value': 0.2},         # Good default
                'modular_num_modules': {'value': 4},     # Good default
                'modular_inter_module_prob': {'value': 0.1},  # Good default
                'modular_intra_module_prob': {'value': 0.8},  # Good default
                'hybrid_num_modules': {'value': 4},      # Good default
                'hybrid_k': {'value': 4},                # Good default
                'hybrid_p': {'value': 0.2},              # Good default
                'hybrid_inter_module_prob': {'value': 0.1},   # Good default
                
                # ============================================================================
                # EVALUATION CONFIGURATION (Fixed for fair comparison)
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'topology_optimization':
        # Individual topology optimization with topology-specific parameters
        return {
            'program': program,
            'method': 'bayes',  # Bayesian optimization for parameter tuning
            'metric': {
                'name': 'baseline/topology_optimization/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY SELECTION (User selects which topology to optimize)
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # VARIABLE ARCHITECTURE (Optimize for selected topology)
                # ============================================================================
                'hidden_size': {'values': [64, 128, 256]},
                'num_layers': {'values': [1, 2, 3]},
                'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
                'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
                
                # ============================================================================
                # VARIABLE TRAINING PARAMETERS (Optimize for selected topology)
                # ============================================================================
                'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
                'n_steps': {'values': [1024, 2048, 4096]},
                'batch_size': {'values': [32, 64, 128, 256]},
                'n_epochs': {'values': [5, 10, 15]},
                'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
                'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
                'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
                'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
                'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (All included, filtered in training script)
                # ============================================================================
                'small_world_k': {'values': [2, 4, 6, 8, 10]},
                'small_world_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'modular_num_modules': {'values': [2, 4, 6, 8, 10]},
                'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
                'hybrid_num_modules': {'values': [2, 4, 6, 8, 10]},
                'hybrid_k': {'values': [2, 4, 6, 8, 10]},
                'hybrid_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                
                # ============================================================================
                # EVALUATION CONFIGURATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'meta_analysis':
        # Compare optimized topologies against each other
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'baseline/meta_analysis/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CONFIGURATION SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'configuration_preset': {
                    'values': ['optimized', 'standard', 'minimal']
                },
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'capacity_matched':
        # Compare topologies with matched parameter counts
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'baseline/capacity_matched/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CAPACITY SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'target_capacity': {
                    'values': [1000, 5000, 10000, 50000]  # Parameter count targets
                },
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS
                # ============================================================================
                'learning_rate': {'value': 3e-4},
                'batch_size': {'value': 64},
                'n_steps': {'value': 2048},
                'gamma': {'value': 0.99},
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Return comprehensive baseline sweep (all parameters)
        return create_baseline_sweep_config(program)

def create_focused_sweep_config(focus_area='topology_comparison', program='topologies--single-task-training-sweep.py'):
    """
    Create focused sweep configurations for single-task training with topology analysis.
    
    Single-task training uses cross-task evaluation (tests on all tasks).
    This ensures we test generalization capability across different environments.
    
    Args:
        focus_area (str): Analysis type ('topology_comparison', 'topology_optimization', 'meta_analysis', 'capacity_matched')
        program (str): The training script to run
    
    Returns:
        dict: Focused sweep configuration for single-task training
    """
    
    if focus_area == 'topology_comparison':
        # Fair head-to-head topology comparison with standardized parameters
        return {
            'program': program,
            'method': 'grid',  # Grid search for fair comparison
            'metric': {
                'name': 'single_task/topology_comparison/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # PRIMARY VARIABLE: TOPOLOGY TYPE
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # STANDARDIZED ARCHITECTURE (Fixed for fair comparison)
                # ============================================================================
                'hidden_size': {'value': 128},  # Fixed for fair comparison
                'num_layers': {'value': 2},     # Fixed for fair comparison
                'activation': {'value': 'relu'}, # Fixed for fair comparison
                'dropout': {'value': 0.0},      # Fixed for fair comparison
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS (Fixed for fair comparison)
                # ============================================================================
                'learning_rate': {'value': 3e-4},    # Fixed for fair comparison
                'n_steps': {'value': 2048},          # Fixed for fair comparison
                'batch_size': {'value': 64},         # Fixed for fair comparison
                'n_epochs': {'value': 10},           # Fixed for fair comparison
                'gamma': {'value': 0.99},            # Fixed for fair comparison
                'gae_lambda': {'value': 0.95},       # Fixed for fair comparison
                'clip_range': {'value': 0.2},        # Fixed for fair comparison
                'ent_coef': {'value': 0.01},         # Fixed for fair comparison
                'max_grad_norm': {'value': 0.5},     # Fixed for fair comparison
                
                # ============================================================================
                # TASK VARIATION (To test generalization across tasks)
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (Optimized defaults)
                # ============================================================================
                'small_world_k': {'value': 4},           # Good default
                'small_world_p': {'value': 0.2},         # Good default
                'modular_num_modules': {'value': 4},     # Good default
                'modular_inter_module_prob': {'value': 0.1},  # Good default
                'modular_intra_module_prob': {'value': 0.8},  # Good default
                'hybrid_num_modules': {'value': 4},      # Good default
                'hybrid_k': {'value': 4},                # Good default
                'hybrid_p': {'value': 0.2},              # Good default
                'hybrid_inter_module_prob': {'value': 0.1},   # Good default
                
                # ============================================================================
                # EVALUATION CONFIGURATION (Fixed for fair comparison)
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'topology_optimization':
        # Individual topology optimization with topology-specific parameters
        return {
            'program': program,
            'method': 'bayes',  # Bayesian optimization for parameter tuning
            'metric': {
                'name': 'single_task/topology_optimization/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY SELECTION (User selects which topology to optimize)
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # VARIABLE ARCHITECTURE (Optimize for selected topology)
                # ============================================================================
                'hidden_size': {'values': [64, 128, 256]},
                'num_layers': {'values': [1, 2, 3]},
                'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
                'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
                
                # ============================================================================
                # VARIABLE TRAINING PARAMETERS (Optimize for selected topology)
                # ============================================================================
                'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
                'n_steps': {'values': [1024, 2048, 4096]},
                'batch_size': {'values': [32, 64, 128, 256]},
                'n_epochs': {'values': [5, 10, 15]},
                'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
                'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
                'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
                'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
                'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (All included, filtered in training script)
                # ============================================================================
                'small_world_k': {'values': [2, 4, 6, 8, 10]},
                'small_world_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'modular_num_modules': {'values': [2, 4, 6, 8, 10]},
                'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
                'hybrid_num_modules': {'values': [2, 4, 6, 8, 10]},
                'hybrid_k': {'values': [2, 4, 6, 8, 10]},
                'hybrid_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                
                # ============================================================================
                # EVALUATION CONFIGURATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'meta_analysis':
        # Compare optimized topologies against each other
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'single_task/meta_analysis/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CONFIGURATION SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'configuration_preset': {
                    'values': ['optimized', 'standard', 'minimal']
                },
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'capacity_matched':
        # Compare topologies with matched parameter counts
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'single_task/capacity_matched/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CAPACITY SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'target_capacity': {
                    'values': [1000, 5000, 10000, 50000]  # Parameter count targets
                },
                
                # ============================================================================
                # TASK VARIATION
                # ============================================================================
                'train_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS
                # ============================================================================
                'learning_rate': {'value': 3e-4},
                'batch_size': {'value': 64},
                'n_steps': {'value': 2048},
                'gamma': {'value': 0.99},
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Return comprehensive single-task sweep (all parameters)
        return create_sweep_config(program)

def create_task_specific_sweep_config(task='CartPole-v1', program='topologies--single-task-training-sweep.py'):
    """
    Create task-specific sweep configurations optimized for particular environments.
    
    Args:
        task (str): Task name ('CartPole-v1', 'Acrobot-v1', 'MountainCar-v0')
        program (str): The training script to run
    
    Returns:
        dict: Task-specific sweep configuration
    """
    
    if task == 'CartPole-v1':
        # CartPole-specific optimization
        return {
            'program': program,
            'method': 'bayes',
            'metric': {
                'name': 'testing/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3,
                },
                'n_steps': {
                    'values': [1024, 2048, 4096]
                },
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'n_epochs': {
                    'values': [3, 5, 10]
                },
                'gamma': {
                    'distribution': 'uniform',
                    'min': 0.95,
                    'max': 0.999
                },
                'hidden_size': {
                    'values': [64, 128, 256]
                },
                'num_layers': {
                    'values': [1, 2]
                },
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'total_timesteps': {
                    'values': [200000, 400000]
                },
                # Fixed values
                'gae_lambda': {'value': 0.95},
                'clip_range': {'value': 0.2},
                'ent_coef': {'value': 0.05},
                'max_grad_norm': {'value': 0.5},
                'train_task': {'value': task},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif task == 'Acrobot-v1':
        # Acrobot-specific optimization (longer episodes, different reward structure)
        return {
            'program': program,
            'method': 'bayes',
            'metric': {
                'name': 'testing/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                'learning_rate': {
                'distribution': 'log_uniform_values',
                'values': [1e-6, 1e-3]
            },
                'n_steps': {
                    'values': [2048, 4096, 8192]
                },
                'batch_size': {
                    'values': [128, 256, 512]
                },
                'n_epochs': {
                    'values': [5, 10, 15]
                },
                'gamma': {
                    'distribution': 'uniform',
                    'min': 0.98,
                    'max': 0.999
                },
                'hidden_size': {
                    'values': [128, 256, 512]
                },
                'num_layers': {
                    'values': [1, 2, 3]
                },
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'total_timesteps': {
                    'values': [400000, 600000, 800000]
                },
                # Fixed values
                'gae_lambda': {'value': 0.95},
                'clip_range': {'value': 0.2},
                'ent_coef': {'value': 0.01},  # Lower entropy for Acrobot
                'max_grad_norm': {'value': 0.5},
                'train_task': {'value': task},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif task == 'MountainCar-v0':
        # MountainCar-specific optimization (sparse rewards, exploration important)
        return {
            'program': program,
            'method': 'bayes',
            'metric': {
                'name': 'testing/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-5,
                    'max': 1e-3,
                },
                'n_steps': {
                    'values': [2048, 4096, 8192]
                },
                'batch_size': {
                    'values': [64, 128, 256]
                },
                'n_epochs': {
                    'values': [5, 10, 15]
                },
                'gamma': {
                    'distribution': 'uniform',
                    'min': 0.99,
                    'max': 0.999
                },
                'ent_coef': {
                'distribution': 'log_uniform_values',
                'values': [1e-3, 1e-1]
            },
                'hidden_size': {
                    'values': [128, 256, 512]
                },
                'num_layers': {
                    'values': [1, 2]
                },
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'total_timesteps': {
                    'values': [400000, 600000, 800000]
                },
                # Fixed values
                'gae_lambda': {'value': 0.95},
                'clip_range': {'value': 0.2},
                'max_grad_norm': {'value': 0.5},
                'train_task': {'value': task},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Default to comprehensive sweep
        return create_sweep_config(program)

def create_sweep_agent_config():
    """
    Create configuration for the sweep agent.
    
    Returns:
        dict: Sweep agent configuration
    """
    return {
        'entity': 'katko-it-universitetet-i-k-benhavn',
        'project': 'topologies--hyperparameter-optimization'
    }

def create_baseline_sweep_agent_config():
    """
    Create configuration for the baseline sweep agent.
    
    Returns:
        dict: Baseline sweep agent configuration
    """
    return {
        'entity': 'katko-it-universitetet-i-k-benhavn',
        'project': 'topologies--baseline-training'
    }

# ============================================================================
# CONVENIENCE FUNCTIONS FOR DIFFERENT TRAINING TYPES
# ============================================================================

def create_double_task_sweep_config(program='topologies--double-task-training-sweep.py'):
    """
    Create sweep configuration for double-task training.
    
    Args:
        program (str): The double-task training script to run
    
    Returns:
        dict: Double-task sweep configuration
    """
    # Get the base comprehensive configuration
    sweep_config = create_sweep_config(program)
    
    # Add double-task specific parameters
    sweep_config['parameters'].update({
        'train_task_1': {
            'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
        },
        'train_task_2': {
            'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
        }
    })
    
    # Remove single-task parameter if present
    if 'train_task' in sweep_config['parameters']:
        del sweep_config['parameters']['train_task']
    
    return sweep_config

def create_triple_task_sweep_config(program='topologies--triple-task-training-sweep.py'):
    """
    Create sweep configuration for triple-task training.
    
    Args:
        program (str): The triple-task training script to run
    
    Returns:
        dict: Triple-task sweep configuration
    """
    # Get the base comprehensive configuration
    sweep_config = create_sweep_config(program)
    
    # Add triple-task specific parameters
    sweep_config['parameters'].update({
        'train_task_1': {
            'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
        },
        'train_task_2': {
            'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
        },
        'train_task_3': {
            'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
        }
    })
    
    # Remove single-task parameter if present
    if 'train_task' in sweep_config['parameters']:
        del sweep_config['parameters']['train_task']
    
    return sweep_config

def create_focused_double_task_sweep_config(focus_area='topology_comparison', program='topologies--double-task-training-sweep.py'):
    """
    Create focused sweep configurations for double-task training with topology analysis.
    
    Double-task training uses sequential training on two distinct tasks.
    This tests transfer learning and sequential adaptation capabilities.
    
    Args:
        focus_area (str): Analysis type ('topology_comparison', 'topology_optimization', 'meta_analysis', 'capacity_matched')
        program (str): The training script to run
    
    Returns:
        dict: Focused sweep configuration for double-task training
    """
    
    if focus_area == 'topology_comparison':
        # Fair head-to-head topology comparison with standardized parameters
        return {
            'program': program,
            'method': 'grid',  # Grid search for fair comparison
            'metric': {
                'name': 'double_task/topology_comparison/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # PRIMARY VARIABLE: TOPOLOGY TYPE
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # STANDARDIZED ARCHITECTURE (Fixed for fair comparison)
                # ============================================================================
                'hidden_size': {'value': 128},  # Fixed for fair comparison
                'num_layers': {'value': 2},     # Fixed for fair comparison
                'activation': {'value': 'relu'}, # Fixed for fair comparison
                'dropout': {'value': 0.0},      # Fixed for fair comparison
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS (Fixed for fair comparison)
                # ============================================================================
                'learning_rate': {'value': 3e-4},    # Fixed for fair comparison
                'n_steps': {'value': 2048},          # Fixed for fair comparison
                'batch_size': {'value': 64},         # Fixed for fair comparison
                'n_epochs': {'value': 10},           # Fixed for fair comparison
                'gamma': {'value': 0.99},            # Fixed for fair comparison
                'gae_lambda': {'value': 0.95},       # Fixed for fair comparison
                'clip_range': {'value': 0.2},        # Fixed for fair comparison
                'ent_coef': {'value': 0.01},         # Fixed for fair comparison
                'max_grad_norm': {'value': 0.5},     # Fixed for fair comparison
                
                # ============================================================================
                # TASK SEQUENCE VARIATION (To test transfer learning)
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (Optimized defaults)
                # ============================================================================
                'small_world_k': {'value': 4},           # Good default
                'small_world_p': {'value': 0.2},         # Good default
                'modular_num_modules': {'value': 4},     # Good default
                'modular_inter_module_prob': {'value': 0.1},  # Good default
                'modular_intra_module_prob': {'value': 0.8},  # Good default
                'hybrid_num_modules': {'value': 4},      # Good default
                'hybrid_k': {'value': 4},                # Good default
                'hybrid_p': {'value': 0.2},              # Good default
                'hybrid_inter_module_prob': {'value': 0.1},   # Good default
                
                # ============================================================================
                # EVALUATION CONFIGURATION (Fixed for fair comparison)
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'topology_optimization':
        # Individual topology optimization with topology-specific parameters
        return {
            'program': program,
            'method': 'bayes',  # Bayesian optimization for parameter tuning
            'metric': {
                'name': 'double_task/topology_optimization/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY SELECTION (User selects which topology to optimize)
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # VARIABLE ARCHITECTURE (Optimize for selected topology)
                # ============================================================================
                'hidden_size': {'values': [64, 128, 256]},
                'num_layers': {'values': [1, 2, 3]},
                'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
                'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
                
                # ============================================================================
                # VARIABLE TRAINING PARAMETERS (Optimize for selected topology)
                # ============================================================================
                'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
                'n_steps': {'values': [1024, 2048, 4096]},
                'batch_size': {'values': [32, 64, 128, 256]},
                'n_epochs': {'values': [5, 10, 15]},
                'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
                'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
                'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
                'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
                'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (All included, filtered in training script)
                # ============================================================================
                'small_world_k': {'values': [2, 4, 6, 8, 10]},
                'small_world_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'modular_num_modules': {'values': [2, 4, 6, 8, 10]},
                'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
                'hybrid_num_modules': {'values': [2, 4, 6, 8, 10]},
                'hybrid_k': {'values': [2, 4, 6, 8, 10]},
                'hybrid_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                
                # ============================================================================
                # EVALUATION CONFIGURATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'meta_analysis':
        # Compare optimized topologies against each other
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'double_task/meta_analysis/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CONFIGURATION SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'configuration_preset': {
                    'values': ['optimized', 'standard', 'minimal']
                },
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'capacity_matched':
        # Compare topologies with matched parameter counts
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'double_task/capacity_matched/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CAPACITY SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'target_capacity': {
                    'values': [1000, 5000, 10000, 50000]  # Parameter count targets
                },
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS
                # ============================================================================
                'learning_rate': {'value': 3e-4},
                'batch_size': {'value': 64},
                'n_steps': {'value': 2048},
                'gamma': {'value': 0.99},
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Return comprehensive double-task sweep (all parameters)
        return create_double_task_sweep_config(program)

def create_focused_triple_task_sweep_config(focus_area='topology_comparison', program='topologies--triple-task-training-sweep.py'):
    """
    Create focused sweep configurations for triple-task training with topology analysis.
    
    Triple-task training uses sequential training on three distinct tasks.
    This tests advanced transfer learning and sequential adaptation capabilities.
    
    Args:
        focus_area (str): Analysis type ('topology_comparison', 'topology_optimization', 'meta_analysis', 'capacity_matched')
        program (str): The training script to run
    
    Returns:
        dict: Focused sweep configuration for triple-task training
    """
    
    if focus_area == 'topology_comparison':
        # Fair head-to-head topology comparison with standardized parameters
        return {
            'program': program,
            'method': 'grid',  # Grid search for fair comparison
            'metric': {
                'name': 'triple_task/topology_comparison/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # PRIMARY VARIABLE: TOPOLOGY TYPE
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # STANDARDIZED ARCHITECTURE (Fixed for fair comparison)
                # ============================================================================
                'hidden_size': {'value': 128},  # Fixed for fair comparison
                'num_layers': {'value': 2},     # Fixed for fair comparison
                'activation': {'value': 'relu'}, # Fixed for fair comparison
                'dropout': {'value': 0.0},      # Fixed for fair comparison
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS (Fixed for fair comparison)
                # ============================================================================
                'learning_rate': {'value': 3e-4},    # Fixed for fair comparison
                'n_steps': {'value': 2048},          # Fixed for fair comparison
                'batch_size': {'value': 64},         # Fixed for fair comparison
                'n_epochs': {'value': 10},           # Fixed for fair comparison
                'gamma': {'value': 0.99},            # Fixed for fair comparison
                'gae_lambda': {'value': 0.95},       # Fixed for fair comparison
                'clip_range': {'value': 0.2},        # Fixed for fair comparison
                'ent_coef': {'value': 0.01},         # Fixed for fair comparison
                'max_grad_norm': {'value': 0.5},     # Fixed for fair comparison
                
                # ============================================================================
                # TASK SEQUENCE VARIATION (To test advanced transfer learning)
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'third_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (Optimized defaults)
                # ============================================================================
                'small_world_k': {'value': 4},           # Good default
                'small_world_p': {'value': 0.2},         # Good default
                'modular_num_modules': {'value': 4},     # Good default
                'modular_inter_module_prob': {'value': 0.1},  # Good default
                'modular_intra_module_prob': {'value': 0.8},  # Good default
                'hybrid_num_modules': {'value': 4},      # Good default
                'hybrid_k': {'value': 4},                # Good default
                'hybrid_p': {'value': 0.2},              # Good default
                'hybrid_inter_module_prob': {'value': 0.1},   # Good default
                
                # ============================================================================
                # EVALUATION CONFIGURATION (Fixed for fair comparison)
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'topology_optimization':
        # Individual topology optimization with topology-specific parameters
        return {
            'program': program,
            'method': 'bayes',  # Bayesian optimization for parameter tuning
            'metric': {
                'name': 'triple_task/topology_optimization/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY SELECTION (User selects which topology to optimize)
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                
                # ============================================================================
                # VARIABLE ARCHITECTURE (Optimize for selected topology)
                # ============================================================================
                'hidden_size': {'values': [64, 128, 256]},
                'num_layers': {'values': [1, 2, 3]},
                'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
                'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
                
                # ============================================================================
                # VARIABLE TRAINING PARAMETERS (Optimize for selected topology)
                # ============================================================================
                'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
                'n_steps': {'values': [1024, 2048, 4096]},
                'batch_size': {'values': [32, 64, 128, 256]},
                'n_epochs': {'values': [5, 10, 15]},
                'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
                'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
                'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
                'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
                'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'third_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # TOPOLOGY-SPECIFIC PARAMETERS (All included, filtered in training script)
                # ============================================================================
                'small_world_k': {'values': [2, 4, 6, 8, 10]},
                'small_world_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'modular_num_modules': {'values': [2, 4, 6, 8, 10]},
                'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
                'hybrid_num_modules': {'values': [2, 4, 6, 8, 10]},
                'hybrid_k': {'values': [2, 4, 6, 8, 10]},
                'hybrid_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
                'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
                
                # ============================================================================
                # EVALUATION CONFIGURATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'meta_analysis':
        # Compare optimized topologies against each other
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'triple_task/meta_analysis/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CONFIGURATION SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'configuration_preset': {
                    'values': ['optimized', 'standard', 'minimal']
                },
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'third_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'capacity_matched':
        # Compare topologies with matched parameter counts
        return {
            'program': program,
            'method': 'grid',  # Grid search for systematic comparison
            'metric': {
                'name': 'triple_task/capacity_matched/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                # ============================================================================
                # TOPOLOGY AND CAPACITY SELECTION
                # ============================================================================
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'target_capacity': {
                    'values': [1000, 5000, 10000, 50000]  # Parameter count targets
                },
                
                # ============================================================================
                # TASK SEQUENCE VARIATION
                # ============================================================================
                'first_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'second_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                'third_task': {
                    'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
                },
                
                # ============================================================================
                # STANDARDIZED TRAINING PARAMETERS
                # ============================================================================
                'learning_rate': {'value': 3e-4},
                'batch_size': {'value': 64},
                'n_steps': {'value': 2048},
                'gamma': {'value': 0.99},
                
                # ============================================================================
                # FIXED EVALUATION
                # ============================================================================
                'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Return comprehensive triple-task sweep (all parameters)
        return create_triple_task_sweep_config(program)

# ============================================================================
# TOPOLOGY COMPARISON SWEEP CONFIGURATIONS
# ============================================================================

def create_topology_comparison_sweep_config(program='topologies--topology-comparison-sweep.py'):
    """
    Create a topology comparison sweep configuration for fair head-to-head comparison.
    
    This sweep uses standardized parameters across all topologies to ensure fair comparison.
    Only the topology_type varies, with all other parameters fixed at good defaults.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Topology comparison sweep configuration
    """
    return {
        'program': program,
        'method': 'grid',  # Grid search for fair comparison
        'metric': {
            'name': 'topology_comparison/mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            # ============================================================================
            # PRIMARY VARIABLE: TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {
                'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
            },
            
            # ============================================================================
            # STANDARDIZED ARCHITECTURE (Fixed for fair comparison)
            # ============================================================================
            'hidden_size': {'value': 128},  # Fixed for fair comparison
            'num_layers': {'value': 2},     # Fixed for fair comparison
            'activation': {'value': 'relu'}, # Fixed for fair comparison
            'dropout': {'value': 0.0},      # Fixed for fair comparison
            
            # ============================================================================
            # STANDARDIZED TRAINING PARAMETERS (Fixed for fair comparison)
            # ============================================================================
            'learning_rate': {'value': 3e-4},    # Fixed for fair comparison
            'n_steps': {'value': 2048},          # Fixed for fair comparison
            'batch_size': {'value': 64},         # Fixed for fair comparison
            'n_epochs': {'value': 10},           # Fixed for fair comparison
            'gamma': {'value': 0.99},            # Fixed for fair comparison
            'gae_lambda': {'value': 0.95},       # Fixed for fair comparison
            'clip_range': {'value': 0.2},        # Fixed for fair comparison
            'ent_coef': {'value': 0.01},         # Fixed for fair comparison
            'max_grad_norm': {'value': 0.5},     # Fixed for fair comparison
            
            # ============================================================================
            # TASK VARIATION (To test generalization across tasks)
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # TOPOLOGY-SPECIFIC PARAMETERS (Optimized defaults)
            # ============================================================================
            # Small World parameters (good defaults)
            'small_world_k': {'value': 4},           # Good default
            'small_world_p': {'value': 0.2},         # Good default
            
            # Modular parameters (good defaults)
            'modular_num_modules': {'value': 4},     # Good default
            'modular_inter_module_prob': {'value': 0.1},  # Good default
            'modular_intra_module_prob': {'value': 0.8},  # Good default
            
            # Hybrid parameters (good defaults)
            'hybrid_num_modules': {'value': 4},      # Good default
            'hybrid_k': {'value': 4},                # Good default
            'hybrid_p': {'value': 0.2},              # Good default
            'hybrid_inter_module_prob': {'value': 0.1},   # Good default
            
            # ============================================================================
            # EVALUATION CONFIGURATION (Fixed for fair comparison)
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }
    

def create_topology_optimization_sweep_config(topology_type='small_world', program=None):
    """
    Create a topology-specific optimization sweep configuration.
    
    This sweep optimizes the parameters specific to a given topology type,
    while varying architecture and training parameters to find the best
    configuration for that topology.
    
    Args:
        topology_type (str): The topology type to optimize ('small_world', 'modular', 'hybrid', 'fully_connected')
        program (str): The training script to run (auto-generated if None)
    
    Returns:
        dict: Topology-specific optimization sweep configuration
    """
    if program is None:
        program = f'topologies--{topology_type}-optimization-sweep.py'
    
    base_config = {
            'program': program,
        'method': 'bayes',  # Bayesian optimization for parameter tuning
            'metric': {
            'name': f'{topology_type}_optimization/mean_reward',
                'goal': 'maximize'
            },
            'early_terminate': {
                'type': 'hyperband',
                'min_iter': 10
            },
            'parameters': {
            # ============================================================================
            # FIXED TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {'value': topology_type},
            
            # ============================================================================
            # VARIABLE ARCHITECTURE (Optimize for this topology)
            # ============================================================================
            'hidden_size': {'values': [64, 128, 256]},
            'num_layers': {'values': [1, 2, 3]},
            'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
            
            # ============================================================================
            # VARIABLE TRAINING PARAMETERS (Optimize for this topology)
            # ============================================================================
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_steps': {'values': [1024, 2048, 4096]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'n_epochs': {'values': [5, 10, 15]},
            'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # EVALUATION CONFIGURATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    # ============================================================================
    # TOPOLOGY-SPECIFIC PARAMETERS (ONLY for this topology)
    # ============================================================================
    if topology_type == 'small_world':
        base_config['parameters'].update({
            'small_world_k': {'values': [2, 4, 6, 8, 10]},
            'small_world_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
        })
    elif topology_type == 'modular':
        base_config['parameters'].update({
            'modular_num_modules': {'values': [2, 4, 6, 8, 10]},
            'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
            'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.6, 'max': 0.95},
        })
    elif topology_type == 'hybrid':
        base_config['parameters'].update({
            'hybrid_num_modules': {'values': [2, 4, 6, 8, 10]},
            'hybrid_k': {'values': [2, 4, 6, 8, 10]},
            'hybrid_p': {'distribution': 'uniform', 'min': 0.1, 'max': 0.4},
            'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.05, 'max': 0.3},
        })
    # fully_connected has no additional parameters
    
    return base_config


def create_meta_analysis_sweep_config(program='topologies--meta-analysis-sweep.py'):
    """
    Create a meta-analysis sweep configuration to compare optimized topologies.
    
    This sweep compares the best configurations from each topology optimization
    against each other to determine which optimized topology performs best.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Meta-analysis sweep configuration
    """
    return {
        'program': program,
        'method': 'grid',  # Grid search for systematic comparison
        'metric': {
            'name': 'meta_analysis/mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            # ============================================================================
            # TOPOLOGY AND CONFIGURATION SELECTION
            # ============================================================================
            'topology_type': {
                'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
            },
            'configuration_preset': {
                'values': ['optimized', 'standard', 'minimal']
            },
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # FIXED EVALUATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
                'n_eval_episodes': {'value': 15},
            }
        }
    

def create_capacity_matched_comparison_sweep(program='topologies--capacity-matched-sweep.py'):
    """
    Create a capacity-matched comparison sweep configuration.
    
    This sweep compares topologies with matched parameter counts to ensure
    fair comparison regardless of topology-specific parameter efficiency.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Capacity-matched comparison sweep configuration
    """
    return {
        'program': program,
        'method': 'grid',  # Grid search for systematic comparison
        'metric': {
            'name': 'capacity_matched/mean_reward',
            'goal': 'maximize'
        },
        'parameters': {
            # ============================================================================
            # TOPOLOGY AND CAPACITY SELECTION
            # ============================================================================
            'topology_type': {
                'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
            },
            'target_capacity': {
                'values': [1000, 5000, 10000, 50000]  # Parameter count targets
            },

# ============================================================================
            # TASK VARIATION
# ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # STANDARDIZED TRAINING PARAMETERS
            # ============================================================================
            'learning_rate': {'value': 3e-4},
            'batch_size': {'value': 64},
            'n_steps': {'value': 2048},
            'gamma': {'value': 0.99},
            
            # ============================================================================
            # FIXED EVALUATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }

def create_small_world_optimization_sweep_config(program='topologies--single-task-training-sweep.py'):
    """
    Create Small World topology optimization sweep configuration.
    
    Only includes Small World specific parameters for efficient optimization.
    Parameters refined based on network theory and biological plausibility.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Small World optimization sweep configuration
    """
    return {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for parameter tuning
        'metric': {
            'name': 'small_world_optimization/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # FIXED TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {'value': 'small_world'},
            
            # ============================================================================
            # VARIABLE ARCHITECTURE (Optimize for Small World)
            # ============================================================================
            'hidden_size': {'values': [64, 128, 256]},
            'num_layers': {'value': 1},  # Fixed: Small World always generates single graph
            'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
            
            # ============================================================================
            # VARIABLE TRAINING PARAMETERS (Optimize for Small World)
            # ============================================================================
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_steps': {'values': [1024, 2048, 4096]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'n_epochs': {'values': [5, 10, 15]},
            'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # SMALL WORLD SPECIFIC PARAMETERS (Refined bounds)
            # ============================================================================
            # k (local neighborhood size): [4, 6, 8]
            # Rationale: In Watts-Strogatz, too low (k=2) leads to near-chain networks (inefficient),
            # while too high (k=10) approaches dense random graphs.
            # Cortical microcircuits show ~4-8 strong local synapses per neuron (sparse but not minimal).
            'small_world_k': {'values': [4, 6, 8]},
            
            # p (rewiring probability): uniform(0.05-0.25)
            # Rationale: Empirical brain networks: p ≈ 0.05-0.2 gives small-world index > 1 (clustering high, paths short).
            # Above p > 0.3, the graph loses its clustering and behaves random (Erdős-Rényi).
            # Staying in low-to-mid small-world regime preserves biologically relevant balance.
            'small_world_p': {'distribution': 'uniform', 'min': 0.05, 'max': 0.25},
            
            # ============================================================================
            # EVALUATION CONFIGURATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }


def create_modular_optimization_sweep_config(program='topologies--single-task-training-sweep.py'):
    """
    Create Modular topology optimization sweep configuration.
    
    Only includes Modular specific parameters for efficient optimization.
    Parameters refined based on network theory and biological plausibility.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Modular optimization sweep configuration
    """
    return {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for parameter tuning
        'metric': {
            'name': 'modular_optimization/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # FIXED TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {'value': 'modular'},
            
            # ============================================================================
            # VARIABLE ARCHITECTURE (Optimize for Modular)
            # ============================================================================
            'hidden_size': {'values': [64, 128, 256]},
            'num_layers': {'value': 1},  # Fixed: Modular always generates single graph
            'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
            
            # ============================================================================
            # VARIABLE TRAINING PARAMETERS (Optimize for Modular)
            # ============================================================================
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_steps': {'values': [1024, 2048, 4096]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'n_epochs': {'values': [5, 10, 15]},
            'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # MODULAR SPECIFIC PARAMETERS (Refined bounds)
            # ============================================================================
            # num_modules: [4, 6, 8]
            # Rationale: Cortical networks show 4-8 mesoscopic modules in many tasks
            # (e.g., sensory/motor areas subdivided into modules).
            'modular_num_modules': {'values': [4, 6, 8]},
            
            # intra_module_prob: uniform(0.5-0.8)
            # Rationale: Extremely high intra-module density (≥0.9) collapses modules into near cliques,
            # eliminating sparseness. Biological cortical areas show moderate intra-area connection density (~30-50%).
            'modular_intra_module_prob': {'distribution': 'uniform', 'min': 0.5, 'max': 0.8},
            
            # inter_module_prob: uniform(0.02-0.15)
            # Rationale: Biological connectivity between modules is very sparse (e.g., ~5-15% of cortical projections).
            # Higher values (>0.2) risk destroying modularity by creating too many cross-module links.
            'modular_inter_module_prob': {'distribution': 'uniform', 'min': 0.02, 'max': 0.15},
            
            # ============================================================================
            # EVALUATION CONFIGURATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }


def create_hybrid_optimization_sweep_config(program='topologies--single-task-training-sweep.py'):
    """
    Create Hybrid topology optimization sweep configuration.
    
    Only includes Hybrid specific parameters for efficient optimization.
    Parameters refined based on network theory and biological plausibility.
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Hybrid optimization sweep configuration
    """
    return {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for parameter tuning
        'metric': {
            'name': 'hybrid_optimization/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # FIXED TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {'value': 'hybrid'},
            
            # ============================================================================
            # VARIABLE ARCHITECTURE (Optimize for Hybrid)
            # ============================================================================
            'hidden_size': {'values': [64, 128, 256]},
            'num_layers': {'value': 1},  # Fixed: Hybrid always generates single graph
            'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
            
            # ============================================================================
            # VARIABLE TRAINING PARAMETERS (Optimize for Hybrid)
            # ============================================================================
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_steps': {'values': [1024, 2048, 4096]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'n_epochs': {'values': [5, 10, 15]},
            'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # HYBRID SPECIFIC PARAMETERS (Refined bounds)
            # ============================================================================
            # num_modules: [4, 6] (to keep modules meaningful)
            'hybrid_num_modules': {'values': [4, 6]},
            
            # k: [4, 6] (preserve local small-world neighborhoods within modules)
            'hybrid_k': {'values': [4, 6]},
            
            # p: uniform(0.05-0.2) (avoid randomness)
            'hybrid_p': {'distribution': 'uniform', 'min': 0.05, 'max': 0.2},
            
            # inter_module_prob: uniform(0.02-0.12) (strong modular integrity with sparse global bridges)
            'hybrid_inter_module_prob': {'distribution': 'uniform', 'min': 0.02, 'max': 0.12},
            
            # ============================================================================
            # EVALUATION CONFIGURATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }


def create_fully_connected_optimization_sweep_config(program='topologies--single-task-training-sweep.py'):
    """
    Create Fully Connected topology optimization sweep configuration.
    
    Only includes architecture and training parameters (no topology-specific ones).
    
    Args:
        program (str): The training script to run
    
    Returns:
        dict: Fully Connected optimization sweep configuration
    """
    return {
        'program': program,
        'method': 'bayes',  # Bayesian optimization for parameter tuning
        'metric': {
            'name': 'fully_connected_optimization/mean_reward',
            'goal': 'maximize'
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        },
        'parameters': {
            # ============================================================================
            # FIXED TOPOLOGY TYPE
            # ============================================================================
            'topology_type': {'value': 'fully_connected'},
            
            # ============================================================================
            # VARIABLE ARCHITECTURE (Optimize for Fully Connected)
            # ============================================================================
            'hidden_size': {'values': [64, 128, 256]},
            'num_layers': {'values': [1, 2, 3]},
            'activation': {'values': ['relu', 'tanh', 'leaky_relu']},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.3},
            
            # ============================================================================
            # VARIABLE TRAINING PARAMETERS (Optimize for Fully Connected)
            # ============================================================================
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'n_steps': {'values': [1024, 2048, 4096]},
            'batch_size': {'values': [32, 64, 128, 256]},
            'n_epochs': {'values': [5, 10, 15]},
            'gamma': {'distribution': 'uniform', 'min': 0.9, 'max': 0.999},
            'gae_lambda': {'distribution': 'uniform', 'min': 0.8, 'max': 0.99},
            'clip_range': {'distribution': 'uniform', 'min': 0.1, 'max': 0.3},
            'ent_coef': {'distribution': 'log_uniform_values', 'min': 1e-4, 'max': 1e-1},
            'max_grad_norm': {'distribution': 'uniform', 'min': 0.1, 'max': 1.0},
            
            # ============================================================================
            # TASK VARIATION
            # ============================================================================
            'train_task': {
                'values': ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
            },
            
            # ============================================================================
            # NO TOPOLOGY-SPECIFIC PARAMETERS (Fully Connected doesn't need them)
            # ============================================================================
            
            # ============================================================================
            # EVALUATION CONFIGURATION
            # ============================================================================
            'total_timesteps': {'value': 500000},
            'n_eval_episodes': {'value': 15},
        }
    }

if __name__ == "__main__":
    # Example usage
    print("Available sweep configurations:")
    print("1. Comprehensive sweep: create_sweep_config()")
    print("2. PPO-focused sweep: create_focused_sweep_config('ppo')")
    print("3. Architecture-focused sweep: create_focused_sweep_config('architecture')")
    print("4. Topology-focused sweep: create_focused_sweep_config('topology')")
    print("5. Task-specific sweeps: create_task_specific_sweep_config('CartPole-v1')")
    print("\nBaseline sweep configurations:")
    print("6. Comprehensive baseline sweep: create_baseline_sweep_config()")
    print("7. PPO-focused baseline sweep: create_baseline_focused_sweep_config('ppo')")
    print("8. Architecture-focused baseline sweep: create_baseline_focused_sweep_config('architecture')")
    print("9. Topology-focused baseline sweep: create_baseline_focused_sweep_config('topology')")
    print("10. Task-specific baseline sweeps: create_baseline_focused_sweep_config('task_specific')")
    print("\nDouble-task sweep configurations:")
    print("11. Double-task sweep: create_double_task_sweep_config()")
    print("12. PPO-focused double-task sweep: create_focused_double_task_sweep_config('ppo')")
    print("13. Architecture-focused double-task sweep: create_focused_double_task_sweep_config('architecture')")
    print("14. Topology-focused double-task sweep: create_focused_double_task_sweep_config('topology')")
    print("\nTriple-task sweep configurations:")
    print("15. Triple-task sweep: create_triple_task_sweep_config()")
    print("16. PPO-focused triple-task sweep: create_focused_triple_task_sweep_config('ppo')")
    print("17. Architecture-focused triple-task sweep: create_focused_triple_task_sweep_config('architecture')")
    print("18. Topology-focused triple-task sweep: create_focused_triple_task_sweep_config('topology')")
    print("\nTopology comparison sweep configurations:")
    print("19. Topology comparison sweep: create_topology_comparison_sweep_config()")
    print("20. Topology optimization sweep (small_world): create_topology_optimization_sweep_config('small_world')")
    print("21. Topology optimization sweep (modular): create_topology_optimization_sweep_config('modular')")
    print("22. Topology optimization sweep (hybrid): create_topology_optimization_sweep_config('hybrid')")
    print("23. Topology optimization sweep (fully_connected): create_topology_optimization_sweep_config('fully_connected')")
    print("24. Meta-analysis sweep: create_meta_analysis_sweep_config()")
    print("25. Capacity-matched comparison sweep: create_capacity_matched_comparison_sweep()")
    
    # Create and print a sample configuration
    config = create_sweep_config()
    print(f"\nSample comprehensive sweep configuration:")
    print(f"Method: {config['method']}")
    print(f"Metric: {config['metric']}")
    print(f"Number of parameters: {len(config['parameters'])}")
    
    # Create and print a sample baseline configuration
    baseline_config = create_baseline_sweep_config()
    print(f"\nSample baseline sweep configuration:")
    print(f"Method: {baseline_config['method']}")
    print(f"Metric: {baseline_config['metric']}")
    print(f"Number of parameters: {len(baseline_config['parameters'])}")
    
    # Create and print a sample double-task configuration
    double_task_config = create_double_task_sweep_config()
    print(f"\nSample double-task sweep configuration:")
    print(f"Method: {double_task_config['method']}")
    print(f"Metric: {double_task_config['metric']}")
    print(f"Number of parameters: {len(double_task_config['parameters'])}")
    
    # Create and print a sample triple-task configuration
    triple_task_config = create_triple_task_sweep_config()
    print(f"\nSample triple-task sweep configuration:")
    print(f"Method: {triple_task_config['method']}")
    print(f"Metric: {triple_task_config['metric']}")
    print(f"Number of parameters: {len(triple_task_config['parameters'])}")