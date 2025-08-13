#!/usr/bin/env python3
"""
Launch Weights & Biases Sweep for Triple-Task Training

This script launches wandb sweeps for triple-task training with topology comparison.
Focusing on sweeps 4-5 (fixed network sizes) and 4-6 (fixed capacities).
"""

import wandb
from wandb_sweep_config import (
    create_fixed_network_sizes_triple_task_sweep,
    create_fixed_capacities_triple_task_sweep
)

def create_sweep_agent_config():
    """Create configuration for the sweep agent."""
    return {
        'entity': 'katko-it-universitetet-i-k-benhavn',
        'project': 'topologies--triple-task-training'
    }

def launch_fixed_network_sizes_triple_task_sweep():
    """Launch sweep 4-5: Fixed network sizes comparison for triple-task training."""
    print("🚀 Launching Fixed Network Sizes Triple-Task Sweep (4-5)...")
    
    # Create sweep configuration
    sweep_config = create_fixed_network_sizes_triple_task_sweep()
    sweep_config['name'] = 'fixed_network_sizes_triple_task_comparison'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Calculate total runs
    topology_count = len(sweep_config['parameters']['topology_type']['values'])
    size_count = len(sweep_config['parameters']['hidden_size']['values'])
    task_count = len(sweep_config['parameters']['task_order']['values'])
    total_runs = topology_count * size_count * task_count
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Topology types: {topology_count} ({', '.join(sweep_config['parameters']['topology_type']['values'])})")
    print(f"   • Hidden sizes: {size_count} ({', '.join(map(str, sweep_config['parameters']['hidden_size']['values']))})")
    print(f"   • Task orders: {task_count}")
    print(f"   • Total runs: {total_runs}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Fixed Network Sizes Triple-Task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def launch_fixed_capacities_triple_task_sweep():
    """Launch sweep 4-6: Fixed capacities comparison for triple-task training."""
    print("🚀 Launching Fixed Capacities Triple-Task Sweep (4-6)...")
    
    # Create sweep configuration
    sweep_config = create_fixed_capacities_triple_task_sweep()
    sweep_config['name'] = 'fixed_capacities_triple_task_comparison'
    
    # Create sweep agent configuration
    agent_config = create_sweep_agent_config()
    
    # Calculate total runs
    topology_count = len(sweep_config['parameters']['topology_type']['values'])
    capacity_count = len(sweep_config['parameters']['target_capacity']['values'])
    task_count = len(sweep_config['parameters']['task_order']['values'])
    total_runs = topology_count * capacity_count * task_count
    
    print(f"   • Project: {agent_config['project']}")
    print(f"   • Entity: {agent_config['entity']}")
    print(f"   • Method: {sweep_config['method']}")
    print(f"   • Topology types: {topology_count} ({', '.join(sweep_config['parameters']['topology_type']['values'])})")
    print(f"   • Target capacities: {capacity_count} ({', '.join(map(str, sweep_config['parameters']['target_capacity']['values']))})")
    print(f"   • Task orders: {task_count}")
    print(f"   • Total runs: {total_runs}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, **agent_config)
    
    print(f"✅ Fixed Capacities Triple-Task sweep created with ID: {sweep_id}")
    print(f"🔗 View sweep at: https://wandb.ai/{agent_config['entity']}/{agent_config['project']}/sweeps/{sweep_id}")
    
    return sweep_id

def main():
    """Main function to launch sweeps."""
    print("🎯 Triple-Task Training Sweep Launcher")
    print("=" * 50)
    print("Available sweeps:")
    print("1. Fixed Network Sizes Triple-Task Sweep (4-5)")
    print("2. Fixed Capacities Triple-Task Sweep (4-6)")
    print("3. Launch both sweeps")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                launch_fixed_network_sizes_triple_task_sweep()
                break
            elif choice == '2':
                launch_fixed_capacities_triple_task_sweep()
                break
            elif choice == '3':
                print("\n🚀 Launching both sweeps...")
                sweep1_id = launch_fixed_network_sizes_triple_task_sweep()
                print()
                sweep2_id = launch_fixed_capacities_triple_task_sweep()
                print(f"\n✅ Both sweeps launched successfully!")
                print(f"   • Fixed Network Sizes: {sweep1_id}")
                print(f"   • Fixed Capacities: {sweep2_id}")
                break
            elif choice == '4':
                print("👋 Exiting...")
                break
        else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main() 