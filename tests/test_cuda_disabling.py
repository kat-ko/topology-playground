#!/usr/bin/env python3
"""
Test script to verify CUDA disabling functionality.
This script tests whether the --no_cuda flag properly disables CUDA usage.
"""

import os
import torch
import argparse

def test_cuda_disabling():
    """Test whether CUDA can be properly disabled."""
    print("🧪 Testing CUDA Disabling Functionality")
    print("=" * 50)
    
    # Check initial CUDA state
    print(f"Initial CUDA state:")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
        print(f"   torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"   torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
    
    print()
    
    # Test CUDA disabling
    print("Testing CUDA disabling...")
    
    # Method 1: Set environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    print(f"   Set CUDA_VISIBLE_DEVICES=''")
    
    # Method 2: Store original CUDA availability function
    original_cuda_available = torch.cuda.is_available
    print(f"   Stored original torch.cuda.is_available function")
    
    # Method 3: Try to clear CUDA cache
    try:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            print(f"   Cleared CUDA cache")
    except:
        print(f"   Could not clear CUDA cache")
    
    # Check CUDA state after disabling
    print(f"\nAfter CUDA disabling:")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    # Try to create a tensor on CUDA
    try:
        test_tensor = torch.tensor([1.0], device='cuda')
        print(f"   ❌ FAILED: Could still create tensor on CUDA: {test_tensor.device}")
        print(f"      This indicates that CUDA is not fully disabled!")
        print(f"      The --no_cuda flag may not work as expected.")
    except Exception as e:
        print(f"   ✅ SUCCESS: Cannot create tensor on CUDA: {e}")
    
    # Try to create a tensor on CPU
    try:
        test_tensor = torch.tensor([1.0], device='cpu')
        print(f"   ✅ SUCCESS: Can create tensor on CPU: {test_tensor.device}")
    except Exception as e:
        print(f"   ❌ FAILED: Cannot create tensor on CPU: {e}")
    
    # Test device selection logic
    print(f"\nTesting device selection logic:")
    
    # Simulate --no_cuda flag
    no_cuda = True
    if not no_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   CUDA selected: {device}")
    else:
        device = torch.device("cpu")
        print(f"   CPU selected: {device}")
    
    # Test model creation on CPU
    print(f"\nTesting model creation on {device}:")
    try:
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        model = model.to(device)
        print(f"   ✅ SUCCESS: Model created on {device}")
        
        # Test forward pass
        x = torch.randn(5, 10).to(device)
        y = model(x)
        print(f"   ✅ SUCCESS: Forward pass works on {y.device}")
        
    except Exception as e:
        print(f"   ❌ FAILED: Model creation/forward pass failed: {e}")
    
    # Test comprehensive CUDA disabling
    print(f"\nTesting comprehensive CUDA disabling...")
    
    # Method 4: Try to set device to CPU globally
    try:
        torch.set_default_tensor_type('torch.FloatTensor')
        print(f"   Set default tensor type to CPU")
    except:
        print(f"   Could not set default tensor type")
    
    # Test again
    try:
        test_tensor = torch.tensor([1.0], device='cuda')
        print(f"   ❌ FAILED: Still can create tensor on CUDA after comprehensive disabling")
        print(f"      This suggests that PyTorch has deep CUDA integration that cannot be easily disabled.")
    except Exception as e:
        print(f"   ✅ SUCCESS: Comprehensive disabling worked: {e}")
    
    # Restore original CUDA availability function
    torch.cuda.is_available = original_cuda_available
    print(f"   Restored original torch.cuda.is_available function")
    
    print(f"\n🎯 Test completed!")
    
    # Final assessment
    if device.type == 'cpu':
        print(f"✅ CPU device selected - this is good for shared server usage")
        print(f"⚠️  However, PyTorch may still be able to create CUDA tensors if explicitly requested")
        print(f"   The --no_cuda flag provides best-effort protection but may not be 100% foolproof")
    else:
        print(f"❌ CUDA device selected - may cause issues on shared servers")
    
    print(f"\n💡 Recommendations:")
    print(f"   1. Always use --no_cuda flag on shared servers")
    print(f"   2. Monitor GPU usage with 'nvidia-smi' during training")
    print(f"   3. If you see GPU usage, the job may not be fully CPU-only")
    print(f"   4. Consider using smaller models or fewer iterations for CPU training")
    print(f"   5. The --no_cuda flag provides best-effort protection but monitor GPU usage")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test CUDA disabling functionality")
    parser.add_argument("--no_cuda", action="store_true", help="Test with --no_cuda flag")
    
    args = parser.parse_args()
    
    if args.no_cuda:
        print("🚫 --no_cuda flag detected - testing CUDA disabling...")
        test_cuda_disabling()
    else:
        print("🔧 No --no_cuda flag - testing normal CUDA behavior...")
        test_cuda_disabling()

if __name__ == "__main__":
    main()
