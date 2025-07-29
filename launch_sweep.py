#!/usr/bin/env python3
"""
Launch Weights & Biases Sweep for Topology Network Hyperparameter Optimization

This script launches wandb sweeps to optimize hyperparameters for the topology training scripts.
"""

import wandb
from wandb_sweep_config import (
    create_sweep_config,
    create_focused_sweep_config,
    create_task_specific_sweep_config,
    create_sweep_agent_config
)

def launch_comprehensive_sweep():
    """Launch a comprehensive sweep over all hyperparameters."""
    print("🚀 Launching comprehensive hyperparameter sweep...")
    
    # Create sweep configuration
    sweep_config = create_sweep_config()
    sweep_config['name'] = 'comprehensive_topology_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Comprehensive sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_focused_sweep(focus_area='ppo'):
    """Launch a focused sweep on specific hyperparameter areas."""
    print(f"🚀 Launching {focus_area}-focused hyperparameter sweep...")
    
    # Create focused sweep configuration
    sweep_config = create_focused_sweep_config(focus_area)
    sweep_config['name'] = f'{focus_area}_focused_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {focus_area}-focused sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_task_specific_sweep(task='CartPole-v1'):
    """Launch a task-specific sweep optimized for particular environments."""
    print(f"🚀 Launching {task}-specific hyperparameter sweep...")
    
    # Create task-specific sweep configuration
    sweep_config = create_task_specific_sweep_config(task)
    sweep_config['name'] = f'{task}_optimization'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ {task}-specific sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_custom_sweep(sweep_config, name="custom_optimization"):
    """Launch a custom sweep with user-defined configuration."""
    print(f"🚀 Launching custom hyperparameter sweep: {name}...")
    
    # Add name to sweep configuration
    sweep_config['name'] = name
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Custom sweep '{name}' created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def main():
    """Main function to launch sweeps based on user input."""
    print("🎯 Weights & Biases Sweep Launcher for Topology Networks")
    print("=" * 60)
    
    print("\nAvailable sweep types:")
    print("1. Comprehensive sweep (all hyperparameters)")
    print("2. PPO-focused sweep (training parameters + all topologies/architectures)")
    print("3. Architecture-focused sweep (network structure + all topologies/tasks)")
    print("4. Topology-focused sweep (topology parameters + all architectures/tasks)")
    print("5. Task-specific sweep (CartPole-v1)")
    print("6. Task-specific sweep (Acrobot-v1)")
    print("7. Task-specific sweep (MountainCar-v0)")
    print("8. Task-specific comprehensive (all tasks, topologies, architectures)")
    print("9. Custom sweep")
    
    choice = input("\nEnter your choice (1-9): ").strip()
    
    if choice == '1':
        launch_comprehensive_sweep()
    elif choice == '2':
        launch_focused_sweep('ppo')
    elif choice == '3':
        launch_focused_sweep('architecture')
    elif choice == '4':
        launch_focused_sweep('topology')
    elif choice == '5':
        launch_task_specific_sweep('CartPole-v1')
    elif choice == '6':
        launch_task_specific_sweep('Acrobot-v1')
    elif choice == '7':
        launch_task_specific_sweep('MountainCar-v0')
    elif choice == '8':
        launch_focused_sweep('task_specific')
    elif choice == '9':
        print("\nCustom sweep configuration:")
        print("You can modify the sweep configuration in wandb_sweep_config.py")
        print("or create a custom configuration here.")
        
        # Example custom configuration
        custom_config = {
            'method': 'bayes',
            'metric': {
                'name': 'testing/mean_reward',
                'goal': 'maximize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': -5,
                    'max': -3,
                },
                'hidden_size': {
                    'values': [64, 128, 256]
                },
                'topology_type': {
                    'values': ['small_world', 'modular']
                },
                'train_task': {
                    'value': 'CartPole-v1'
                },
                # Add more parameters as needed
            }
        }
        
        name = input("Enter sweep name: ").strip() or "custom_optimization"
        launch_custom_sweep(custom_config, name)
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main() 