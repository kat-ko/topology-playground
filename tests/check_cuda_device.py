#!/usr/bin/env python3
"""
CUDA Device Checker
This script shows which CUDA device you're currently using and provides detailed device information.
"""

import torch
import os
import subprocess
import sys

def get_nvidia_smi_info():
    """Get GPU information from nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        else:
            return None
    except FileNotFoundError:
        return None

def check_cuda_device():
    """Check which CUDA device is currently being used."""
    print("🔍 CUDA Device Checker")
    print("=" * 50)
    
    # Check if CUDA is available
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Using CPU only.")
        return
    
    # Get CUDA device information
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    
    print(f"Total CUDA Devices: {device_count}")
    print(f"Current CUDA Device: {current_device}")
    
    # Get current device name
    try:
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Current Device Name: {device_name}")
    except Exception as e:
        print(f"Could not get device name: {e}")
    
    # Check memory usage
    try:
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**2    # MB
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**2  # MB
        
        print(f"\nMemory Usage on Device {current_device}:")
        print(f"  Allocated: {memory_allocated:.1f} MB")
        print(f"  Reserved:  {memory_reserved:.1f} MB")
        print(f"  Total:     {memory_total:.1f} MB")
        print(f"  Free:      {memory_total - memory_reserved:.1f} MB")
        
    except Exception as e:
        print(f"Could not get memory info: {e}")
    
    # Check environment variables
    print(f"\nEnvironment Variables:")
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # Check if any tensors are on GPU
    print(f"\nGPU Tensor Check:")
    gpu_tensors = 0
    cpu_tensors = 0
    
    # Check if there are any existing tensors
    try:
        # Create a test tensor to see where it goes
        test_tensor = torch.tensor([1.0])
        if test_tensor.device.type == 'cuda':
            gpu_tensors += 1
            print(f"  Test tensor created on: {test_tensor.device}")
        else:
            cpu_tensors += 1
            print(f"  Test tensor created on: {test_tensor.device}")
        
        # Try to create a tensor on specific device
        if current_device < device_count:
            try:
                specific_tensor = torch.tensor([1.0], device=f'cuda:{current_device}')
                print(f"  Specific tensor on cuda:{current_device}: {specific_tensor.device}")
            except Exception as e:
                print(f"  Could not create tensor on cuda:{current_device}: {e}")
    except Exception as e:
        print(f"  Error checking tensors: {e}")
    
    # Show all available devices
    print(f"\nAll Available CUDA Devices:")
    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory = props.total_memory / 1024**2  # MB
            print(f"  Device {i}: {name} ({memory:.0f} MB)")
        except Exception as e:
            print(f"  Device {i}: Error getting info - {e}")

def check_python_process_gpu():
    """Check which GPU the current Python process is using."""
    print(f"\n🐍 Current Python Process GPU Usage:")
    print("=" * 50)
    
    # Get current process ID
    pid = os.getpid()
    print(f"Process ID: {pid}")
    
    # Try to get GPU usage from nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            found = False
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4 and parts[0] == str(pid):
                        gpu_uuid = parts[2]
                        memory = parts[3]
                        print(f"  GPU UUID: {gpu_uuid}")
                        print(f"  Memory Used: {memory} MB")
                        found = True
                        break
            
            if not found:
                print("  No GPU usage found for this process")
        else:
            print("  Could not get GPU usage info")
    except Exception as e:
        print(f"  Error checking GPU usage: {e}")

def check_environment_gpu():
    """Check GPU environment and configuration."""
    print(f"\n🌍 GPU Environment Check:")
    print("=" * 50)
    
    # Check common GPU-related environment variables
    gpu_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CUDA_LAUNCH_BLOCKING',
        'CUDA_CACHE_DISABLE',
        'TOPOLOGY_DEVICE'
    ]
    
    for var in gpu_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")
    
    # Check if we're in a container
    if os.path.exists('/.dockerenv'):
        print("  Running in Docker container")
    elif os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup').read():
        print("  Running in Docker container")
    else:
        print("  Running on host system")

def main():
    """Main function."""
    print("🚀 Comprehensive CUDA Device Checker")
    print("=" * 60)
    
    # Check CUDA device
    check_cuda_device()
    
    # Check Python process GPU usage
    check_python_process_gpu()
    
    # Check environment
    check_environment_gpu()
    
    # Summary
    print(f"\n📋 Summary:")
    print("=" * 30)
    
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"✅ Currently using: CUDA Device {current_device} ({device_name})")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        print(f"   • You're currently using GPU {current_device}")
        print(f"   • To use specific GPUs, set: export CUDA_VISIBLE_DEVICES=0,6")
        print(f"   • To use only GPU 0: export CUDA_VISIBLE_DEVICES=0")
        print(f"   • To use only GPU 6: export CUDA_VISIBLE_DEVICES=6")
        print(f"   • To use CPU only: export CUDA_VISIBLE_DEVICES=''")
    else:
        print(f"❌ Currently using: CPU only")
        print(f"\n💡 To enable GPU usage, ensure CUDA is properly installed")

if __name__ == "__main__":
    main()
