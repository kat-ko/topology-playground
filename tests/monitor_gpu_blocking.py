#!/usr/bin/env python3
"""
Real-time GPU Blocking Monitor
Shows how GPU blocking works and monitors usage.
"""

import os
import time
import subprocess
import torch

def show_gpu_blocking_status():
    """Show current GPU blocking status."""
    print("🔒 GPU Blocking Status")
    print("=" * 40)
    
    # Check environment variable
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    if cuda_visible == 'Not set':
        print("❌ No GPU blocking active!")
        print("   Run: source gpu_block.sh 0 6")
        return False
    
    # Check PyTorch devices
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ PyTorch sees {device_count} CUDA devices")
        
        # Show device mapping
        print(f"\n📊 Device Mapping:")
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory = props.total_memory / 1024**2  # MB
            print(f"   PyTorch cuda:{i} -> Physical GPU with {memory:.0f} MB total")
        
        return True
    else:
        print("❌ CUDA not available")
        return False

def show_gpu_usage():
    """Show current GPU usage for blocked GPUs."""
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cuda_visible:
        print("❌ No GPU blocking active")
        return
    
    # Parse blocked GPUs
    blocked_gpus = cuda_visible.split(',')
    print(f"\n📊 GPU Usage for Blocked GPUs: {blocked_gpus}")
    print("=" * 50)
    
    try:
        # Get GPU usage for blocked GPUs
        for gpu in blocked_gpus:
            if gpu.strip():
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw',
                    '--format=csv,noheader,nounits',
                    '-i', gpu.strip()
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    line = result.stdout.strip()
                    if line:
                        parts = line.split(', ')
                        if len(parts) >= 5:
                            gpu_idx, memory_used, memory_total, utilization, power = parts
                            memory_free = int(memory_total) - int(memory_used)
                            memory_percent = (int(memory_used) / int(memory_total)) * 100
                            
                            print(f"GPU {gpu_idx}:")
                            print(f"   Memory: {memory_used}MB used / {memory_total}MB total ({memory_percent:.1f}%)")
                            print(f"   Free: {memory_free}MB")
                            print(f"   Utilization: {utilization}%")
                            print(f"   Power: {power}W")
                            print()
                else:
                    print(f"   Could not get info for GPU {gpu}")
                    
    except Exception as e:
        print(f"Error getting GPU usage: {e}")

def monitor_realtime(duration=30):
    """Monitor GPU usage in real-time."""
    print(f"🔄 Real-time Monitoring (Press Ctrl+C to stop)")
    print("=" * 50)
    
    start_time = time.time()
    try:
        while time.time() - start_time < duration:
            # Clear screen (works on most terminals)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"🕐 Time: {time.strftime('%H:%M:%S')}")
            print(f"⏱️  Elapsed: {int(time.time() - start_time)}s")
            print()
            
            # Show status
            show_gpu_blocking_status()
            
            # Show usage
            show_gpu_usage()
            
            # Show PyTorch device info
            if torch.cuda.is_available():
                print(f"🐍 PyTorch Process:")
                print(f"   Current device: cuda:{torch.cuda.current_device()}")
                print(f"   Device name: {torch.cuda.get_device_name()}")
                
                # Check if any tensors are on GPU
                try:
                    # Create a small test tensor to see memory usage
                    test_tensor = torch.randn(1000, 1000)
                    if test_tensor.device.type == 'cuda':
                        memory = torch.cuda.memory_allocated() / 1024**2
                        print(f"   Test tensor on: {test_tensor.device}")
                        print(f"   Memory allocated: {memory:.1f} MB")
                    else:
                        print(f"   Test tensor on: {test_tensor.device}")
                    
                    # Clean up
                    del test_tensor
                    if test_tensor.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"   Error testing tensor: {e}")
            
            print(f"\n🔄 Refreshing in 2 seconds... (Press Ctrl+C to stop)")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")

def main():
    """Main function."""
    print("🚀 GPU Blocking Real-time Monitor")
    print("=" * 60)
    
    # Show current status
    if show_gpu_blocking_status():
        print(f"\n✅ GPU blocking is active and working!")
        
        # Show current usage
        show_gpu_usage()
        
        # Ask if user wants real-time monitoring
        print(f"\n💡 Options:")
        print(f"   1. Show current status (above)")
        print(f"   2. Monitor in real-time (30 seconds)")
        print(f"   3. Exit")
        
        try:
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '2':
                monitor_realtime(30)
            elif choice == '1':
                print("Current status shown above.")
            else:
                print("Exiting...")
                
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        print(f"\n❌ GPU blocking is not active!")
        print(f"   Please run: source gpu_block.sh 0 6")

if __name__ == "__main__":
    main()
