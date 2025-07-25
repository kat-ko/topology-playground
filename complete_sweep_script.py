#!/usr/bin/env python3
"""
Complete Sweep Script Generator

This script helps you complete the topologies--single-task-training-sweep.py file
by copying the necessary classes and functions from the original training script.
"""

import os

def copy_classes_from_original():
    """
    Copy the necessary classes and functions from the original training script
    to complete the sweep training script.
    """
    
    # Read the original training script
    original_file = "topologies--single-task-training.py"
    sweep_file = "topologies--single-task-training-sweep.py"
    
    if not os.path.exists(original_file):
        print(f"❌ Original file {original_file} not found!")
        return
    
    print(f"📖 Reading original file: {original_file}")
    
    with open(original_file, 'r') as f:
        original_content = f.read()
    
    # Define the classes and functions to copy
    classes_to_copy = [
        "UniversalActionWrapper",
        "DebugTopologyPolicy", 
        "EnhancedDebugCallback",
        "cross_task_testing",
        "evaluate_model",
        "evaluate_model_enhanced",
        "calculate_success_rate",
        "verify_capacity_matching_debug",
        "create_network_visualization",
        "create_connection_heatmap",
        "create_layer_analysis_visualization",
        "create_connection_list",
        "create_simple_connection_list",
        "save_simple_connection_files",
        "save_connection_lists",
        "make_env"
    ]
    
    # Extract the classes and functions
    extracted_code = []
    
    # Add imports and setup
    extracted_code.append("""
# ============================================================================
# COPIED CLASSES AND FUNCTIONS FROM ORIGINAL SCRIPT
# ============================================================================

# Copy all the classes and functions from topologies--single-task-training.py here
# This includes:
# - UniversalActionWrapper
# - DebugTopologyPolicy
# - EnhancedDebugCallback
# - cross_task_testing
# - evaluate_model
# - evaluate_model_enhanced
# - calculate_success_rate
# - verify_capacity_matching_debug
# - create_network_visualization
# - create_connection_heatmap
# - create_layer_analysis_visualization
# - create_connection_list
# - create_simple_connection_list
# - save_simple_connection_files
# - save_connection_lists
# - make_env
# - All utility functions

""")
    
    # Find and extract each class/function
    for class_name in classes_to_copy:
        print(f"🔍 Looking for {class_name}...")
        
        # Simple extraction - you'll need to manually copy the exact content
        start_marker = f"class {class_name}" if class_name[0].isupper() else f"def {class_name}"
        
        if start_marker in original_content:
            start_idx = original_content.find(start_marker)
            
            # Find the end of the class/function
            if class_name[0].isupper():  # Class
                # Look for the next class definition or end of file
                next_class_start = original_content.find("\nclass ", start_idx + 1)
                if next_class_start == -1:
                    next_class_start = len(original_content)
                
                class_content = original_content[start_idx:next_class_start]
            else:  # Function
                # Look for the next function definition or end of file
                next_func_start = original_content.find("\ndef ", start_idx + 1)
                if next_func_start == -1:
                    next_func_start = len(original_content)
                
                class_content = original_content[start_idx:next_func_start]
            
            extracted_code.append(f"# {class_name}")
            extracted_code.append(class_content)
            extracted_code.append("")
            print(f"   ✅ Found {class_name}")
        else:
            print(f"   ⚠️  {class_name} not found in original file")
    
    # Create the complete sweep script
    print(f"📝 Creating complete sweep script...")
    
    # Read the current sweep script
    with open(sweep_file, 'r') as f:
        sweep_content = f.read()
    
    # Insert the extracted code after the imports
    import_end = sweep_content.find("# Import the original classes and functions")
    if import_end == -1:
        import_end = sweep_content.find("def create_sweep_config()")
    
    if import_end != -1:
        # Insert the extracted code
        complete_content = (
            sweep_content[:import_end] + 
            "\n".join(extracted_code) + 
            "\n" + 
            sweep_content[import_end:]
        )
        
        # Write the complete script
        with open(sweep_file, 'w') as f:
            f.write(complete_content)
        
        print(f"✅ Complete sweep script written to: {sweep_file}")
        print(f"📋 Next steps:")
        print(f"   1. Review the copied classes and functions")
        print(f"   2. Test the script with a single run")
        print(f"   3. Launch your sweep!")
    else:
        print(f"❌ Could not find insertion point in sweep script")

def create_minimal_sweep_example():
    """
    Create a minimal working example for testing sweeps.
    """
    
    minimal_example = '''#!/usr/bin/env python3
"""
Minimal Sweep Example for Topology Networks

This is a simplified version for testing wandb sweeps.
"""

import wandb
import numpy as np
import time

def minimal_train_function():
    """Minimal training function for testing sweeps."""
    
    # Initialize wandb run
    wandb.init(
        entity="katko-it-universitetet-i-k-benhavn",
        project="topologies--hyperparameter-optimization",
        config={
            'learning_rate': 3e-4,
            'hidden_size': 128,
            'topology_type': 'small_world',
            'train_task': 'CartPole-v1',
        }
    )
    
    print(f"🎯 Minimal sweep run:")
    print(f"   • Learning rate: {wandb.config.learning_rate}")
    print(f"   • Hidden size: {wandb.config.hidden_size}")
    print(f"   • Topology: {wandb.config.topology_type}")
    print(f"   • Task: {wandb.config.train_task}")
    
    # Simulate training (replace with actual training)
    time.sleep(2)  # Simulate training time
    
    # Simulate results based on hyperparameters
    base_reward = 100.0
    
    # Adjust reward based on hyperparameters
    if wandb.config.topology_type == 'small_world':
        base_reward += 20
    elif wandb.config.topology_type == 'modular':
        base_reward += 15
    elif wandb.config.topology_type == 'hybrid':
        base_reward += 10
    
    # Learning rate effect
    if 1e-4 <= wandb.config.learning_rate <= 1e-3:
        base_reward += 30
    elif 1e-5 <= wandb.config.learning_rate < 1e-4:
        base_reward += 20
    else:
        base_reward -= 10
    
    # Hidden size effect
    if wandb.config.hidden_size >= 256:
        base_reward += 25
    elif wandb.config.hidden_size >= 128:
        base_reward += 15
    else:
        base_reward += 5
    
    # Add some noise
    noise = np.random.normal(0, 5)
    final_reward = max(0, base_reward + noise)
    
    # Log results
    wandb.log({
        'testing/mean_reward': final_reward,
        'training/step': 1000,
        'training/loss': 0.1,
    })
    
    print(f"✅ Minimal run completed! Reward: {final_reward:.2f}")
    
    wandb.finish()

if __name__ == "__main__":
    minimal_train_function()
'''
    
    with open("minimal_sweep_example.py", 'w') as f:
        f.write(minimal_example)
    
    print("✅ Created minimal_sweep_example.py")
    print("📋 You can use this to test your sweep setup before running the full training")

def main():
    """Main function to help complete the sweep script."""
    
    print("🔧 Sweep Script Completion Helper")
    print("=" * 50)
    
    print("\nThis script helps you complete the sweep training script.")
    print("Choose an option:")
    print("1. Copy classes from original script")
    print("2. Create minimal sweep example")
    print("3. Both")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        copy_classes_from_original()
    elif choice == '2':
        create_minimal_sweep_example()
    elif choice == '3':
        copy_classes_from_original()
        create_minimal_sweep_example()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 