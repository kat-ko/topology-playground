# CUDA Management and Shared Server Safety

## Overview

This document explains how the topology playground handles CUDA devices and provides guidelines for safe usage on shared GPU servers.

## The Problem

When running machine learning training on shared GPU servers, even when using the `--no_cuda` flag, some parts of the code or underlying libraries might still try to use CUDA devices. This can cause:

1. **Resource Conflicts**: Your job might interfere with other users' GPU jobs
2. **Server Crashes**: Excessive GPU memory usage can crash the entire server
3. **Job Failures**: Other users' jobs might fail due to GPU resource conflicts
4. **Poor Performance**: CPU jobs might be slower than expected due to GPU interference

## Solution: Comprehensive CUDA Disabling

The topology playground now includes a comprehensive CUDA management system that ensures complete CPU-only operation when the `--no_cuda` flag is used.

### How It Works

1. **Environment Variable Control**: Sets `CUDA_VISIBLE_DEVICES=''` to hide all CUDA devices
2. **PyTorch Override**: Overrides `torch.cuda.is_available()` to return `False`
3. **Device Enforcement**: Periodically checks and moves any GPU tensors to CPU
4. **Model Verification**: Ensures all models and topology networks are on CPU

### Key Functions

#### `force_cpu_usage()`
- Hides all CUDA devices from PyTorch
- Overrides CUDA availability checks
- Clears any existing CUDA cache

#### `enforce_cpu_usage(model, device)`
- Periodically checks if any model parameters are on GPU
- Automatically moves GPU parameters to CPU
- Handles topology networks and their parameters

## Usage

### Basic Usage with --no_cuda

```bash
# Safe for shared servers - uses CPU only
python topologies_continual_task_training_sweep.py --single --task CartPole-v1 --no_cuda

# Normal usage - may use GPU if available
python topologies_continual_task_training_sweep.py --single --task CartPole-v1
```

### Testing CUDA Disabling

```bash
# Test the CUDA disabling functionality
python test_cuda_disabling.py --no_cuda
```

## Safety Features

### 1. Complete CUDA Hiding
- All CUDA devices are hidden from PyTorch
- `torch.cuda.is_available()` returns `False`
- No accidental GPU tensor creation

### 2. Periodic Enforcement
- Every 100 training iterations, the system checks for GPU usage
- Automatically moves any GPU tensors to CPU
- Prevents gradual GPU memory accumulation

### 3. Model Verification
- Ensures all model parameters are on the correct device
- Handles topology networks and their parameters
- Provides clear feedback about device status

### 4. Environment Configuration
- Sets appropriate environment variables
- Clears any existing CUDA cache
- Provides detailed logging of device status

## Monitoring and Debugging

### Check Device Status
The system provides detailed logging about device configuration:

```
🔒 CUDA devices have been completely hidden from PyTorch
   All operations will use CPU only
⚠️ CUDA is not available or disabled. Using CPU: cpu
🔒 CPU-only mode confirmed - CUDA devices are hidden from PyTorch
✅ CUDA successfully disabled - safe for shared server usage
```

### Verify CUDA Disabling
You can verify that CUDA is properly disabled by checking:

1. **Environment Variables**: `echo $CUDA_VISIBLE_DEVICES` should be empty
2. **PyTorch Status**: `torch.cuda.is_available()` should return `False`
3. **GPU Usage**: `nvidia-smi` should not show your process using GPU memory

### Troubleshooting

#### CUDA Still Available
If you see warnings like:
```
❌ WARNING: CUDA is still available despite --no_cuda flag!
   This might cause issues on shared GPU servers.
```

This indicates that the CUDA disabling didn't work properly. Check:
1. Are you using the latest version of the code?
2. Are there any conflicting environment variables?
3. Is PyTorch properly installed and configured?

#### Performance Issues
If training is very slow with `--no_cuda`:
1. This is expected - CPU training is slower than GPU training
2. Consider using smaller models or fewer training iterations
3. The trade-off is safety vs. speed on shared servers

## Best Practices for Shared Servers

### 1. Always Use --no_cuda
```bash
# ✅ Good - Safe for shared servers
python script.py --no_cuda

# ❌ Bad - May interfere with other users
python script.py
```

### 2. Monitor Resource Usage
```bash
# Check if your process is using GPU
nvidia-smi

# Check CPU usage
htop
```

### 3. Use Appropriate Resources
```bash
# Request CPU-only resources in your job submission
# Example for SLURM:
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
```

### 4. Test Before Running
```bash
# Test CUDA disabling
python test_cuda_disabling.py --no_cuda

# Run a small test job
python script.py --no_cuda --test
```

## Technical Details

### Environment Variables
- `CUDA_VISIBLE_DEVICES=''`: Hides all CUDA devices
- `TOPOLOGY_DEVICE='cpu'`: Forces CPU device preference

### PyTorch Overrides
- `torch.cuda.is_available = lambda: False`: Overrides CUDA availability
- Device enforcement in model parameters and topology networks

### Memory Management
- Automatic GPU memory clearing when CUDA is disabled
- Periodic checks for GPU tensor usage
- Safe fallback to CPU for all operations

## Support

If you encounter issues with CUDA management:

1. **Check the logs**: Look for device-related messages
2. **Test CUDA disabling**: Run `test_cuda_disabling.py --no_cuda`
3. **Verify environment**: Check environment variables and PyTorch configuration
4. **Report issues**: Include logs and error messages

## Conclusion

The topology playground now provides comprehensive CUDA management that ensures safe operation on shared GPU servers. By using the `--no_cuda` flag, you can:

- ✅ Safely run training jobs without interfering with other users
- ✅ Prevent server crashes and resource conflicts
- ✅ Ensure predictable CPU-only operation
- ✅ Monitor and enforce device usage throughout training

Remember: **Always use `--no_cuda` on shared servers to be a good neighbor!**
