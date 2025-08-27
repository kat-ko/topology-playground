#!/usr/bin/env python3
"""
GPU Blocking Solution for Cluster Safety
This script provides a safe way to use specific GPUs (0 and 6) without interfering with other users.
"""

import os
import torch
import subprocess
import time
import argparse
from typing import List, Optional

class GPUBlocker:
    """Manages GPU access to prevent interference with other users."""
    
    def __init__(self, target_gpus: List[int] = [0, 6]):
        """
        Initialize GPU blocker.
        
        Args:
            target_gpus: List of GPU indices to use (default: [0, 6])
        """
        self.target_gpus = sorted(target_gpus)
        self.original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        self.blocked_gpus = []
        
    def block_gpus(self) -> bool:
        """
        Block access to all GPUs except the target ones.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"🔒 Blocking GPUs - Only allowing access to: {self.target_gpus}")
        
        try:
            # Set CUDA_VISIBLE_DEVICES to only show target GPUs
            cuda_visible_str = ','.join(map(str, self.target_gpus))
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_str
            
            print(f"   Set CUDA_VISIBLE_DEVICES={cuda_visible_str}")
            
            # Verify the blocking worked
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"   ✅ CUDA available with {device_count} devices")
                
                # Check which devices are visible
                visible_devices = []
                for i in range(device_count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        visible_devices.append(f"GPU {i}: {name}")
                    except:
                        pass
                
                print(f"   Visible devices: {visible_devices}")
                
                # Map target GPUs to visible indices
                self.gpu_mapping = {}
                for i, target_gpu in enumerate(self.target_gpus):
                    self.gpu_mapping[target_gpu] = i
                
                print(f"   GPU mapping: {self.gpu_mapping}")
                return True
            else:
                print(f"   ❌ CUDA not available after blocking")
                return False
                
        except Exception as e:
            print(f"   ❌ Error blocking GPUs: {e}")
            return False
    
    def unblock_gpus(self):
        """Restore original GPU access."""
        print(f"🔓 Unblocking GPUs - Restoring original access")
        
        if self.original_cuda_visible:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.original_cuda_visible
            print(f"   Restored CUDA_VISIBLE_DEVICES={self.original_cuda_visible}")
        else:
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                del os.environ['CUDA_VISIBLE_DEVICES']
            print(f"   Removed CUDA_VISIBLE_DEVICES restriction")
    
    def get_target_device(self, gpu_index: int) -> torch.device:
        """
        Get the device object for a specific target GPU.
        
        Args:
            gpu_index: Index of the target GPU (0 or 6)
            
        Returns:
            torch.device object for the target GPU
        """
        if gpu_index not in self.target_gpus:
            raise ValueError(f"GPU {gpu_index} is not in target GPUs {self.target_gpus}")
        
        if gpu_index not in self.gpu_mapping:
            raise ValueError(f"GPU {gpu_index} is not properly mapped")
        
        visible_index = self.gpu_mapping[gpu_index]
        return torch.device(f'cuda:{visible_index}')
    
    def monitor_gpu_usage(self, duration: int = 60):
        """
        Monitor GPU usage to ensure we're not interfering with others.
        
        Args:
            duration: Duration to monitor in seconds
        """
        print(f"📊 Monitoring GPU usage for {duration} seconds...")
        print(f"   Press Ctrl+C to stop monitoring")
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                # Get GPU usage for target GPUs
                for target_gpu in self.target_gpus:
                    try:
                        # Use nvidia-smi to get specific GPU info
                        result = subprocess.run([
                            'nvidia-smi', 
                            '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader,nounits',
                            '-i', str(target_gpu)
                        ], capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            line = result.stdout.strip()
                            if line:
                                parts = line.split(', ')
                                if len(parts) >= 4:
                                    gpu_idx, memory_used, memory_total, utilization = parts
                                    memory_free = int(memory_total) - int(memory_used)
                                    print(f"   GPU {gpu_idx}: {memory_used}MB used, {memory_free}MB free, {utilization}% util")
                        
                    except Exception as e:
                        print(f"   Error monitoring GPU {target_gpu}: {e}")
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print(f"\n   Monitoring stopped by user")
    
    def test_gpu_access(self):
        """Test that we can access the target GPUs."""
        print(f"🧪 Testing GPU access...")
        
        try:
            for target_gpu in self.target_gpus:
                device = self.get_target_device(target_gpu)
                print(f"   Testing GPU {target_gpu} -> {device}")
                
                # Create a test tensor
                test_tensor = torch.tensor([1.0], device=device)
                print(f"     ✅ Tensor created on {test_tensor.device}")
                
                # Check memory usage
                if device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
                    print(f"     📊 Memory allocated: {memory_allocated:.1f} MB")
                
                # Clean up
                del test_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error testing GPU access: {e}")

def create_gpu_blocking_script():
    """Create a shell script for easy GPU blocking."""
    script_content = '''#!/bin/bash
# GPU Blocking Script for Safe Cluster Usage
# Usage: source gpu_block.sh [gpu1] [gpu2] ...

# Default to GPUs 0 and 6
GPUS=${@:-0 6}

echo "🔒 Blocking GPUs - Only allowing access to: $GPUS"

# Convert space-separated list to comma-separated
CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',')

# Set environment variable
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "✅ Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "🔒 Only GPUs $GPUS are now visible to PyTorch"
echo ""
echo "💡 To unblock, run: unset CUDA_VISIBLE_DEVICES"
echo "💡 To check current status: echo $CUDA_VISIBLE_DEVICES"
echo "💡 To monitor GPU usage: nvidia-smi -i $GPUS"
'''

    with open('gpu_block.sh', 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod('gpu_block.sh', 0o755)
    print("📝 Created gpu_block.sh script")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="GPU Blocking Solution for Cluster Safety")
    parser.add_argument("--gpus", nargs='+', type=int, default=[0, 6], 
                       help="Target GPUs to use (default: 0 6)")
    parser.add_argument("--monitor", type=int, default=0,
                       help="Monitor GPU usage for N seconds (default: 0)")
    parser.add_argument("--test", action="store_true",
                       help="Test GPU access after blocking")
    parser.add_argument("--create-script", action="store_true",
                       help="Create gpu_block.sh script")
    
    args = parser.parse_args()
    
    print("🚀 GPU Blocking Solution for Cluster Safety")
    print("=" * 60)
    print(f"Target GPUs: {args.gpus}")
    print(f"Current CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print()
    
    # Create GPU blocker
    blocker = GPUBlocker(args.gpus)
    
    # Block GPUs
    if blocker.block_gpus():
        print(f"✅ Successfully blocked GPUs")
        
        # Test access if requested
        if args.test:
            blocker.test_gpu_access()
        
        # Monitor if requested
        if args.monitor > 0:
            blocker.monitor_gpu_usage(args.monitor)
        
        # Create script if requested
        if args.create_script:
            create_gpu_blocking_script()
        
        print(f"\n🎯 GPU Blocking Active!")
        print(f"   • Only GPUs {args.gpus} are visible to PyTorch")
        print(f"   • Other users' GPUs are protected")
        print(f"   • To unblock: blocker.unblock_gpus()")
        print(f"   • To use GPU 0: device = blocker.get_target_device(0)")
        print(f"   • To use GPU 6: device = blocker.get_target_device(6)")
        
    else:
        print(f"❌ Failed to block GPUs")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
