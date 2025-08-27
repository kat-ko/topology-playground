#!/usr/bin/env python3
"""
Test script to verify network metadata collection in topologies_continual_task_training_sweep.py

This script tests:
1. Enhanced metadata collection during training
2. Topology parameter extraction
3. Network architecture details
4. Local data collection with W&B run names
5. Metadata file creation and content
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path

def test_metadata_collection():
    """Test the enhanced metadata collection functionality."""
    print("🧪 Testing Network Metadata Collection")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'max_iterations': 4,  # Very short test
        'level_switch': 2,
        'shift_range': [0, 2],
        'episode_cap': 100,
        'reward_scale': 20.0,
        'n_steps': 800,
        'n_epochs': 1,  # Reduced for testing
        'batch_size': 32,
        'learning_rate': 0.01,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'max_grad_norm': 0.5,
        'num_layers': 1,
        'hidden_size': 64,  # Reduced for testing
        'small_world_k': 4,
        'small_world_p': 0.2,
        'modular_num_modules': 4,
        'modular_inter_module_prob': 0.1,
        'modular_intra_module_prob': 0.8,
        'hybrid_num_modules': 4,
        'hybrid_k': 4,
        'hybrid_p': 0.2,
        'hybrid_inter_module_prob': 0.1
    }
    
    # Test different topology types
    topology_tests = [
        'small_world',
        'modular', 
        'hybrid',
        'fully_connected',
        'standard_mlp'
    ]
    
    results = {}
    
    for topology_type in topology_tests:
        print(f"\n🎯 Testing {topology_type} topology...")
        
        try:
            # Import the training function
            from topologies_continual_task_training_sweep import continual_learning_training
            
            # Create temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Backup original test_experiments directory
                original_test_dir = "test_experiments"
                backup_test_dir = f"test_experiments_backup_{int(time.time())}"
                
                if os.path.exists(original_test_dir):
                    shutil.move(original_test_dir, backup_test_dir)
                
                try:
                    # Create fresh test_experiments directory
                    os.makedirs(original_test_dir, exist_ok=True)
                    
                    # Run training with minimal iterations
                    print(f"   🚀 Starting training for {topology_type}...")
                    
                    model, env = continual_learning_training(
                        config=test_config,
                        task_name="CartPole-v1",
                        topology_type=topology_type,
                        seed=42,
                        use_wandb=False,  # Disable W&B for testing
                        enable_phase3=False,  # Disable Phase 3 for testing
                        device="cpu",  # Use CPU for testing
                        no_noise=False
                    )
                    
                    print(f"   ✅ Training completed for {topology_type}")
                    
                    # Check if metadata files were created
                    test_dirs = [d for d in os.listdir(original_test_dir) if d.startswith(f"CartPole-v1_{topology_type}_seed42")]
                    
                    if test_dirs:
                        test_dir = test_dirs[0]
                        metadata_file = os.path.join(original_test_dir, test_dir, "run_metadata.json")
                        
                        if os.path.exists(metadata_file):
                            print(f"   📁 Metadata file found: {metadata_file}")
                            
                            # Load and analyze metadata
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Test metadata structure
                            test_results = test_metadata_structure(metadata, topology_type, test_config)
                            results[topology_type] = test_results
                            
                            if test_results['passed']:
                                print(f"   ✅ {topology_type} metadata test PASSED")
                            else:
                                print(f"   ❌ {topology_type} metadata test FAILED")
                                for error in test_results['errors']:
                                    print(f"      - {error}")
                        else:
                            print(f"   ❌ Metadata file not found for {topology_type}")
                            results[topology_type] = {'passed': False, 'errors': ['Metadata file not created']}
                    else:
                        print(f"   ❌ Test directory not found for {topology_type}")
                        results[topology_type] = {'passed': False, 'errors': ['Test directory not created']}
                
                finally:
                    # Restore original test_experiments directory
                    if os.path.exists(original_test_dir):
                        shutil.rmtree(original_test_dir)
                    if os.path.exists(backup_test_dir):
                        shutil.move(backup_test_dir, original_test_dir)
        
        except Exception as e:
            print(f"   ❌ Test failed for {topology_type}: {e}")
            import traceback
            traceback.print_exc()
            results[topology_type] = {'passed': False, 'errors': [str(e)]}
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(topology_tests)
    
    for topology, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{topology:15} : {status}")
        if result['passed']:
            passed += 1
        else:
            for error in result['errors']:
                print(f"                - {error}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All network metadata tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

def test_metadata_structure(metadata, topology_type, config):
    """Test the structure and content of the metadata."""
    errors = []
    
    # Test required top-level keys
    required_keys = ['run_id', 'timestamp', 'total_episodes', 'total_shifts', 
                    'episode_data_file', 'shift_data_file', 'training_config', 
                    'network_architecture', 'topology_parameters']
    
    for key in required_keys:
        if key not in metadata:
            errors.append(f"Missing required key: {key}")
    
    # Test training_config
    if 'training_config' in metadata:
        training_config = metadata['training_config']
        expected_training_keys = ['task_name', 'topology_type', 'seed', 'max_iterations', 
                                'level_switch', 'shift_range', 'reward_scale', 'episode_cap', 'no_noise']
        
        for key in expected_training_keys:
            if key not in training_config:
                errors.append(f"Missing training_config key: {key}")
        
        # Verify specific values
        if training_config.get('topology_type') != topology_type:
            errors.append(f"Topology type mismatch: expected {topology_type}, got {training_config.get('topology_type')}")
    
    # Test network_architecture
    if 'network_architecture' in metadata:
        network_arch = metadata['network_architecture']
        expected_arch_keys = ['hidden_size', 'num_layers', 'activation', 'dropout']
        
        for key in expected_arch_keys:
            if key not in network_arch:
                errors.append(f"Missing network_architecture key: {key}")
        
        # Verify specific values
        if network_arch.get('hidden_size') != config['hidden_size']:
            errors.append(f"Hidden size mismatch: expected {config['hidden_size']}, got {network_arch.get('hidden_size')}")
        
        if network_arch.get('num_layers') != config['num_layers']:
            errors.append(f"Num layers mismatch: expected {config['num_layers']}, got {network_arch.get('num_layers')}")
    
    # Test topology_parameters
    if 'topology_parameters' in metadata:
        topology_params = metadata['topology_parameters']
        
        if topology_type == 'small_world':
            expected_params = ['k', 'p']
            for param in expected_params:
                if param not in topology_params:
                    errors.append(f"Missing small_world parameter: {param}")
            
            if 'k' in topology_params and topology_params['k'] != config['small_world_k']:
                errors.append(f"Small world k mismatch: expected {config['small_world_k']}, got {topology_params['k']}")
        
        elif topology_type == 'modular':
            expected_params = ['num_modules', 'inter_module_prob', 'intra_module_prob']
            for param in expected_params:
                if param not in topology_params:
                    errors.append(f"Missing modular parameter: {param}")
        
        elif topology_type == 'hybrid':
            expected_params = ['num_modules', 'k', 'p', 'inter_module_prob']
            for param in expected_params:
                if param not in topology_params:
                    errors.append(f"Missing hybrid parameter: {param}")
        
        elif topology_type == 'fully_connected':
            if 'type' not in topology_params or topology_params['type'] != 'fully_connected':
                errors.append("Fully connected type not correctly specified")
        
        elif topology_type == 'standard_mlp':
            expected_params = ['type', 'num_layers']
            for param in expected_params:
                if param not in topology_params:
                    errors.append(f"Missing standard_mlp parameter: {param}")
    
    # Test that CSV files exist
    if 'episode_data_file' in metadata:
        episode_file = metadata['episode_data_file']
        if not os.path.exists(episode_file):
            errors.append(f"Episode data file not found: {episode_file}")
    
    if 'shift_data_file' in metadata:
        shift_file = metadata['shift_data_file']
        if not os.path.exists(shift_file):
            errors.append(f"Shift data file not found: {shift_file}")
    
    return {
        'passed': len(errors) == 0,
        'errors': errors
    }

def test_metadata_file_content(metadata_file):
    """Test the content of a specific metadata file."""
    print(f"\n📄 Testing metadata file: {metadata_file}")
    
    if not os.path.exists(metadata_file):
        print("❌ Metadata file not found")
        return False
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print("✅ Metadata file loaded successfully")
        print(f"📊 Run ID: {metadata.get('run_id', 'N/A')}")
        print(f"📊 Total Episodes: {metadata.get('total_episodes', 'N/A')}")
        print(f"📊 Total Shifts: {metadata.get('total_shifts', 'N/A')}")
        
        if 'training_config' in metadata:
            config = metadata['training_config']
            print(f"🎯 Task: {config.get('task_name', 'N/A')}")
            print(f"🎯 Topology: {config.get('topology_type', 'N/A')}")
            print(f"🎯 Seed: {config.get('seed', 'N/A')}")
        
        if 'network_architecture' in metadata:
            arch = metadata['network_architecture']
            print(f"🔧 Hidden Size: {arch.get('hidden_size', 'N/A')}")
            print(f"🔧 Num Layers: {arch.get('num_layers', 'N/A')}")
            print(f"🔧 Total Parameters: {arch.get('total_parameters', 'N/A')}")
        
        if 'topology_parameters' in metadata:
            params = metadata['topology_parameters']
            print(f"🌐 Topology Parameters: {params}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading metadata file: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Network Metadata Test Suite")
    print("=" * 60)
    
    # Run the main test
    success = test_metadata_collection()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("The network metadata collection is working correctly.")
    else:
        print("\n⚠️  Some tests failed.")
        print("Check the errors above to identify issues.")
    
    # Optional: Test a specific metadata file if it exists
    print("\n" + "=" * 60)
    print("🔍 OPTIONAL: Test Specific Metadata File")
    print("=" * 60)
    
    # Look for any existing metadata files
    test_experiments_dir = "test_experiments"
    if os.path.exists(test_experiments_dir):
        for item in os.listdir(test_experiments_dir):
            item_path = os.path.join(test_experiments_dir, item)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, "run_metadata.json")
                if os.path.exists(metadata_file):
                    print(f"\nFound existing metadata file: {metadata_file}")
                    test_metadata_file_content(metadata_file)
                    break
        else:
            print("No existing metadata files found to test.")
    else:
        print("No test_experiments directory found.")
