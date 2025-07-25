#!/usr/bin/env python3
"""
Weights & Biases Sweep Configuration for Topology Network Hyperparameter Optimization

This file defines sweep configurations for optimizing hyperparameters in the topology training scripts.
"""

import wandb

def create_sweep_config():
    """
    Create a comprehensive sweep configuration for topology network hyperparameter optimization.
    
    Returns:
        dict: Sweep configuration dictionary
    """
    
    sweep_config = {
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
                'distribution': 'log_uniform',
                'min': -6,  # 1e-6
                'max': -2,  # 1e-2
            },
            'n_steps': {
                'values': [1024, 2048, 4096, 8192]
            },
            'batch_size': {
                'values': [32, 64, 128, 256]
            },
            'n_epochs': {
                'values': [3, 5, 10, 15]
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
                'distribution': 'log_uniform',
                'min': -4,  # 1e-4
                'max': -1,  # 1e-1
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
                'max': 0.5
            },
            
            # Modular parameters
            'modular_num_modules': {
                'values': [2, 4, 6, 8]
            },
            'modular_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.4
            },
            'modular_intra_module_prob': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 0.9
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
                'max': 0.5
            },
            'hybrid_inter_module_prob': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.4
            },
            
            # ============================================================================
            # NETWORK PARAMETERS
            # ============================================================================
            'activation': {
                'values': ['relu', 'tanh', 'sigmoid', 'leaky_relu']
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
                'values': [200000, 400000, 600000]
            },
            'n_eval_episodes': {
                'values': [10, 15, 20]
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

def create_focused_sweep_config(focus_area='ppo'):
    """
    Create focused sweep configurations for specific areas of hyperparameter optimization.
    
    Args:
        focus_area (str): Area to focus on ('ppo', 'architecture', 'topology', 'comprehensive')
    
    Returns:
        dict: Focused sweep configuration
    """
    
    if focus_area == 'ppo':
        # Focus only on PPO training parameters
        return {
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
                    'distribution': 'log_uniform',
                    'min': -6,
                    'max': -2,
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
                    'distribution': 'log_uniform',
                    'min': -4,
                    'max': -1,
                },
                'max_grad_norm': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 1.0
                },
                # Fixed values for other parameters
                'hidden_size': {'value': 128},
                'num_layers': {'value': 1},
                'topology_type': {'value': 'small_world'},
                'train_task': {'value': 'CartPole-v1'},
                'total_timesteps': {'value': 400000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'architecture':
        # Focus on network architecture parameters
        return {
            'method': 'grid',  # Grid search for discrete architecture choices
            'metric': {
                'name': 'testing/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                'hidden_size': {
                    'values': [64, 128, 256, 512]
                },
                'num_layers': {
                    'values': [1, 2, 3]
                },
                'activation': {
                    'values': ['relu', 'tanh', 'sigmoid']
                },
                'dropout': {
                    'values': [0.0, 0.1, 0.2]
                },
                # Fixed PPO parameters
                'learning_rate': {'value': 3e-4},
                'n_steps': {'value': 2048},
                'batch_size': {'value': 128},
                'n_epochs': {'value': 5},
                'gamma': {'value': 0.99},
                'gae_lambda': {'value': 0.95},
                'clip_range': {'value': 0.2},
                'ent_coef': {'value': 0.05},
                'max_grad_norm': {'value': 0.5},
                'topology_type': {'value': 'small_world'},
                'train_task': {'value': 'CartPole-v1'},
                'total_timesteps': {'value': 400000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    elif focus_area == 'topology':
        # Focus on topology-specific parameters
        return {
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
                'topology_type': {
                    'values': ['small_world', 'modular', 'hybrid', 'fully_connected']
                },
                'small_world_k': {
                    'values': [2, 4, 6, 8]
                },
                'small_world_p': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'modular_num_modules': {
                    'values': [2, 4, 6, 8]
                },
                'modular_inter_module_prob': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.4
                },
                'modular_intra_module_prob': {
                    'distribution': 'uniform',
                    'min': 0.6,
                    'max': 0.9
                },
                'hybrid_num_modules': {
                    'values': [2, 4, 6, 8]
                },
                'hybrid_k': {
                    'values': [2, 4, 6, 8]
                },
                'hybrid_p': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'hybrid_inter_module_prob': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.4
                },
                # Fixed values for other parameters
                'hidden_size': {'value': 128},
                'num_layers': {'value': 1},
                'learning_rate': {'value': 3e-4},
                'n_steps': {'value': 2048},
                'batch_size': {'value': 128},
                'n_epochs': {'value': 5},
                'gamma': {'value': 0.99},
                'gae_lambda': {'value': 0.95},
                'clip_range': {'value': 0.2},
                'ent_coef': {'value': 0.05},
                'max_grad_norm': {'value': 0.5},
                'train_task': {'value': 'CartPole-v1'},
                'total_timesteps': {'value': 400000},
                'n_eval_episodes': {'value': 15},
            }
        }
    
    else:
        # Return comprehensive sweep
        return create_sweep_config()

def create_task_specific_sweep_config(task='CartPole-v1'):
    """
    Create task-specific sweep configurations optimized for particular environments.
    
    Args:
        task (str): Task name ('CartPole-v1', 'Acrobot-v1', 'MountainCar-v0')
    
    Returns:
        dict: Task-specific sweep configuration
    """
    
    if task == 'CartPole-v1':
        # CartPole-specific optimization
        return {
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
                    'distribution': 'log_uniform',
                    'min': -5,  # 1e-5
                    'max': -3,  # 1e-3
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
                    'distribution': 'log_uniform',
                    'min': -6,  # 1e-6
                    'max': -3,  # 1e-3
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
                    'distribution': 'log_uniform',
                    'min': -5,  # 1e-5
                    'max': -3,  # 1e-3
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
                    'distribution': 'log_uniform',
                    'min': -3,  # 1e-3
                    'max': -1,  # 1e-1
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
        return create_sweep_config()

def create_sweep_agent_config():
    """
    Create configuration for the sweep agent.
    
    Returns:
        dict: Sweep agent configuration
    """
    return {
        'entity': 'katko-it-universitetet-i-k-benhavn',
        'project': 'topologies--hyperparameter-optimization',
        'program': 'topologies--single-task-training-sweep.py',
        'count': 50,  # Number of runs to perform
        'name': 'topology_hyperparameter_optimization'
    }

if __name__ == "__main__":
    # Example usage
    print("Available sweep configurations:")
    print("1. Comprehensive sweep: create_sweep_config()")
    print("2. PPO-focused sweep: create_focused_sweep_config('ppo')")
    print("3. Architecture-focused sweep: create_focused_sweep_config('architecture')")
    print("4. Topology-focused sweep: create_focused_sweep_config('topology')")
    print("5. Task-specific sweeps: create_task_specific_sweep_config('CartPole-v1')")
    
    # Create and print a sample configuration
    config = create_sweep_config()
    print(f"\nSample comprehensive sweep configuration:")
    print(f"Method: {config['method']}")
    print(f"Metric: {config['metric']}")
    print(f"Number of parameters: {len(config['parameters'])}") 