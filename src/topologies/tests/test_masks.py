import pytest
import torch
from ..utils import mask_shape_sanity

def test_mask_shape_sanity_ok():
    """Test that mask_shape_sanity passes for correctly shaped masks."""
    # Test case 1: Small network
    mask1 = torch.zeros((10, 10))
    mask_shape_sanity(mask1, n_in=2, n_hidden=6, n_out=2)
    
    # Test case 2: Larger network
    mask2 = torch.zeros((100, 100))
    mask_shape_sanity(mask2, n_in=20, n_hidden=60, n_out=20)
    
    # Test case 3: Network with no hidden nodes
    mask3 = torch.zeros((10, 10))
    mask_shape_sanity(mask3, n_in=5, n_hidden=0, n_out=5)

def test_mask_shape_sanity_bad():
    """Test that mask_shape_sanity raises ValueError for incorrectly shaped masks."""
    # Test case 1: Wrong dimensions (1D tensor)
    mask1 = torch.zeros(10)
    with pytest.raises(ValueError, match="Mask shape mismatch"):
        mask_shape_sanity(mask1, n_in=2, n_hidden=6, n_out=2)
    
    # Test case 2: Wrong size (1-off)
    mask2 = torch.zeros((11, 11))
    with pytest.raises(ValueError, match="Mask shape mismatch"):
        mask_shape_sanity(mask2, n_in=2, n_hidden=6, n_out=2)
    
    # Test case 3: Non-square matrix
    mask3 = torch.zeros((10, 11))
    with pytest.raises(ValueError, match="Mask shape mismatch"):
        mask_shape_sanity(mask3, n_in=2, n_hidden=6, n_out=2) 