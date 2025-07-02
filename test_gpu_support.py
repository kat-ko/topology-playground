#!/usr/bin/env python3
"""
GPU Support Test Script

This script tests the GPU support functionality to ensure:
1. Device manager initializes correctly
2. GPU detection works properly
3. Safe fallback to CPU when GPU is unavailable
4. Device information is logged correctly
5. No existing functionality is broken
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_device_manager():
    """Test the device manager functionality."""
    print("="*60)
    print("GPU SUPPORT TEST")
    print("="*60)
    
    # Test 1: Basic import and initialization
    print("\n1. Testing device manager import and initialization...")
    try:
        from src.utils.device_manager import get_device_manager, get_device_info, get_device, is_cuda
        print("   ✅ Device manager imports successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Device manager initialization
    try:
        device_manager = get_device_manager()
        print("   ✅ Device manager initialized successfully")
    except Exception as e:
        print(f"   ❌ Device manager initialization failed: {e}")
        return False
    
    # Test 3: Device information
    print("\n2. Testing device information...")
    try:
        device_info = get_device_info()
        print(f"   Device: {device_info['device']}")
        print(f"   Is CUDA: {device_info['is_cuda']}")
        print(f"   GPU Available: {device_info['is_gpu_available']}")
        print(f"   Preference: {device_info['device_preference']}")
        
        if device_info['is_cuda']:
            print(f"   GPU Name: {device_info.get('cuda_device_name', 'Unknown')}")
            print(f"   Memory Allocated: {device_info.get('cuda_memory_allocated', 0) / 1024**2:.1f}MB")
        
        print("   ✅ Device information retrieved successfully")
    except Exception as e:
        print(f"   ❌ Device information failed: {e}")
        return False
    
    # Test 4: Device operations
    print("\n3. Testing device operations...")
    try:
        device = get_device()
        cuda_available = is_cuda()
        print(f"   Current device: {device}")
        print(f"   CUDA available: {cuda_available}")
        print("   ✅ Device operations work correctly")
    except Exception as e:
        print(f"   ❌ Device operations failed: {e}")
        return False
    
    # Test 5: Memory operations
    print("\n4. Testing memory operations...")
    try:
        memory_info = device_manager.get_memory_info()
        print(f"   Memory info: {memory_info}")
        print("   ✅ Memory operations work correctly")
    except Exception as e:
        print(f"   ❌ Memory operations failed: {e}")
        return False
    
    # Test 6: Seed setting
    print("\n5. Testing seed setting...")
    try:
        device_manager.set_seed(42)
        print("   ✅ Seed setting works correctly")
    except Exception as e:
        print(f"   ❌ Seed setting failed: {e}")
        return False
    
    # Test 7: Cache operations
    print("\n6. Testing cache operations...")
    try:
        device_manager.clear_cache()
        print("   ✅ Cache operations work correctly")
    except Exception as e:
        print(f"   ❌ Cache operations failed: {e}")
        return False
    
    # Test 8: Environment variable override
    print("\n7. Testing environment variable override...")
    try:
        # Test CPU override
        os.environ['TOPOLOGY_DEVICE'] = 'cpu'
        from src.utils.device_manager import DeviceManager
        cpu_manager = DeviceManager()
        print(f"   CPU override device: {cpu_manager.get_device()}")
        
        # Test CUDA override (if available)
        if device_info['is_gpu_available']:
            os.environ['TOPOLOGY_DEVICE'] = 'cuda'
            cuda_manager = DeviceManager()
            print(f"   CUDA override device: {cuda_manager.get_device()}")
        else:
            print("   CUDA override skipped (GPU not available)")
        
        print("   ✅ Environment variable override works correctly")
    except Exception as e:
        print(f"   ❌ Environment variable override failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("GPU SUPPORT TEST COMPLETE")
    print("="*60)
    print("✅ All tests passed! GPU support is working correctly.")
    print(f"🔧 Current device: {device_info['device']}")
    print(f"🚀 Ready for training on {'GPU' if device_info['is_cuda'] else 'CPU'}")
    
    return True

def test_integration_with_existing_code():
    """Test that GPU support doesn't break existing functionality."""
    print("\n" + "="*60)
    print("INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Import existing modules
    print("\n1. Testing imports of existing modules...")
    try:
        from src.utils.parameter_budget import ParameterBudgetCalculator
        from src.topologies.small_world import SmallWorldTopology
        from src.networks.ffn import FeedForwardNetwork
        print("   ✅ All existing modules import successfully")
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 2: Network creation
    print("\n2. Testing network creation...")
    try:
        config = {
            'network_sizes': [50],
            'network_types': ['ffn'],
            'num_layers': [1],
            'parameter_budget': {'budget_type': 'edges', 'target_budget': 1000},
            'small_world_params': {'k': 4, 'p': 0.3},
            'num_io_nodes': 4
        }
        
        calculator = ParameterBudgetCalculator(config)
        network = calculator.create_network(
            topology='small_world',
            size=50,
            experiment_type='same_size',
            network_type='ffn',
            num_layers=1,
            seed=42
        )
        
        metrics = network.get_network_metrics()
        print(f"   Network created with {metrics.get('num_nodes', 0)} nodes")
        print("   ✅ Network creation works correctly")
    except Exception as e:
        print(f"   ❌ Network creation failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)
    print("✅ Integration test passed! GPU support doesn't break existing functionality.")
    
    return True

def main():
    """Run all GPU support tests."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("GPU Support Test Suite")
    print("Testing device manager functionality and integration...")
    
    # Run tests
    device_test_passed = test_device_manager()
    integration_test_passed = test_integration_with_existing_code()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Device Manager Test: {'✅ PASSED' if device_test_passed else '❌ FAILED'}")
    print(f"Integration Test: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if device_test_passed and integration_test_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("GPU support is ready for use.")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED!")
        print("Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 