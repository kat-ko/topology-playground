"""
Device Management Utility for GPU Support

This module provides centralized device management for the topology playground,
allowing seamless switching between CPU and GPU training without breaking
existing functionality.
"""

import torch
import os
from typing import Optional, Union, Dict, Any
import logging
import warnings

logger = logging.getLogger(__name__)

class DeviceManager:
    """Centralized device management for GPU/CPU support."""
    
    def __init__(self, device_preference: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            device_preference: 'auto', 'cuda', 'cpu', or None (uses environment variable)
        """
        self.device_preference = device_preference or os.getenv('TOPOLOGY_DEVICE', 'auto')
        self.is_gpu_available = self._check_gpu_availability()
        self.device = self._determine_device()
        
        logger.info(f"Device Manager initialized: {self.device}")
        if self.is_gpu_available:
            logger.info(f"GPU devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("No GPU devices available, using CPU")
    
    def _check_gpu_availability(self) -> bool:
        """Safely check GPU availability with comprehensive error handling."""
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                return False
            
            # Check if we can actually use CUDA
            if torch.cuda.device_count() == 0:
                return False
            
            # Test basic CUDA operations
            try:
                test_tensor = torch.tensor([1.0], device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                logger.warning(f"CUDA test failed: {e}")
                return False
                
        except Exception as e:
            logger.warning(f"GPU availability check failed: {e}")
            return False
    
    def _determine_device(self) -> torch.device:
        """Determine the best available device based on preference with safe fallback."""
        try:
            if self.device_preference == 'cpu':
                logger.info("Using CPU (explicitly requested)")
                return torch.device('cpu')
            elif self.device_preference == 'cuda':
                if not self.is_gpu_available:
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    return torch.device('cpu')
                logger.info("Using CUDA (explicitly requested)")
                return torch.device('cuda')
            elif self.device_preference == 'auto':
                if self.is_gpu_available:
                    logger.info("Using CUDA (auto-detected)")
                    return torch.device('cuda')
                else:
                    logger.info("Using CPU (auto-detected, no GPU available)")
                    return torch.device('cpu')
            else:
                logger.warning(f"Unknown device preference '{self.device_preference}', using auto")
                # Recursive call with 'auto' preference
                self.device_preference = 'auto'
                return self._determine_device()
        except Exception as e:
            logger.error(f"Device determination failed: {e}, falling back to CPU")
            return torch.device('cpu')
    
    def get_device(self) -> torch.device:
        """Get the current device."""
        return self.device
    
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device.type == 'cuda'
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information with safe error handling."""
        try:
            info = {
                'device': str(self.device),
                'is_cuda': self.is_cuda(),
                'is_gpu_available': self.is_gpu_available,
                'device_preference': self.device_preference
            }
            
            if self.is_cuda():
                try:
                    info.update({
                        'cuda_device_count': torch.cuda.device_count(),
                        'current_cuda_device': torch.cuda.current_device(),
                        'cuda_device_name': torch.cuda.get_device_name(),
                        'cuda_memory_allocated': torch.cuda.memory_allocated(),
                        'cuda_memory_reserved': torch.cuda.memory_reserved()
                    })
                except Exception as e:
                    logger.warning(f"Failed to get CUDA info: {e}")
                    info.update({
                        'cuda_device_count': 0,
                        'current_cuda_device': -1,
                        'cuda_device_name': 'Unknown',
                        'cuda_memory_allocated': 0,
                        'cuda_memory_reserved': 0
                    })
            
            return info
        except Exception as e:
            logger.error(f"Failed to get device info: {e}")
            return {
                'device': 'cpu',
                'is_cuda': False,
                'is_gpu_available': False,
                'device_preference': 'cpu',
                'error': str(e)
            }
    
    def to_device(self, tensor_or_module: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
        """Move tensor or module to the current device with safe error handling."""
        try:
            if hasattr(tensor_or_module, 'to'):
                return tensor_or_module.to(self.device)
            return tensor_or_module
        except Exception as e:
            logger.warning(f"Failed to move to device {self.device}: {e}")
            # Return original object if device transfer fails
            return tensor_or_module
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for the current device with safe error handling."""
        try:
            torch.manual_seed(seed)
            if self.is_cuda():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except Exception as e:
            logger.warning(f"Failed to set seed: {e}")
    
    def clear_cache(self) -> None:
        """Clear CUDA cache if using GPU with safe error handling."""
        try:
            if self.is_cuda():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Failed to clear CUDA cache: {e}")
    
    def synchronize(self) -> None:
        """Synchronize GPU operations if using CUDA with safe error handling."""
        try:
            if self.is_cuda():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Failed to synchronize CUDA: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information with safe error handling."""
        try:
            if self.is_cuda():
                return {
                    'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                    'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
                    'max_allocated': torch.cuda.max_memory_allocated() / 1024**2,  # MB
                    'max_reserved': torch.cuda.max_memory_reserved() / 1024**2     # MB
                }
            else:
                return {
                    'allocated': 0,
                    'reserved': 0,
                    'max_allocated': 0,
                    'max_reserved': 0
                }
        except Exception as e:
            logger.warning(f"Failed to get memory info: {e}")
            return {
                'allocated': 0,
                'reserved': 0,
                'max_allocated': 0,
                'max_reserved': 0,
                'error': str(e)
            }

# Global device manager instance
_device_manager: Optional[DeviceManager] = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance with safe initialization."""
    global _device_manager
    if _device_manager is None:
        try:
            _device_manager = DeviceManager()
        except Exception as e:
            logger.error(f"Failed to initialize device manager: {e}")
            # Create a safe fallback device manager
            _device_manager = DeviceManager('cpu')
    return _device_manager

def set_device_manager(device_manager: DeviceManager) -> None:
    """Set the global device manager instance."""
    global _device_manager
    _device_manager = device_manager

def get_device() -> torch.device:
    """Get the current device with safe fallback."""
    try:
        return get_device_manager().get_device()
    except Exception as e:
        logger.error(f"Failed to get device: {e}")
        return torch.device('cpu')

def is_cuda() -> bool:
    """Check if using CUDA device with safe fallback."""
    try:
        return get_device_manager().is_cuda()
    except Exception as e:
        logger.error(f"Failed to check CUDA status: {e}")
        return False

def to_device(tensor_or_module: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    """Move tensor or module to the current device with safe fallback."""
    try:
        return get_device_manager().to_device(tensor_or_module)
    except Exception as e:
        logger.error(f"Failed to move to device: {e}")
        return tensor_or_module

def get_device_info() -> Dict[str, Any]:
    """Get device information with safe fallback."""
    try:
        return get_device_manager().get_device_info()
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return {
            'device': 'cpu',
            'is_cuda': False,
            'is_gpu_available': False,
            'device_preference': 'cpu',
            'error': str(e)
        }
