import torch
from typing import Tuple, Optional, List
from ..utils.logging_utils import setup_logger, log_mask_validation
import networkx as nx
import numpy as np

logger = setup_logger(__name__)

def mask_shape_sanity(mask: torch.Tensor,
                     n_in: int,
                     n_hidden: int,
                     n_out: int,
                     is_test_run: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Verify that the mask tensor has the correct shape for the given network dimensions.
    
    Args:
        mask: 2D tensor representing the adjacency mask
        n_in: Number of input nodes
        n_hidden: Number of hidden nodes
        n_out: Number of output nodes
        is_test_run: Whether this is a test run
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    expected = n_in + n_hidden + n_out
    expected_shape = (expected, expected)
    
    # Log the validation attempt
    log_mask_validation(logger, mask.shape, expected_shape, is_test_run)
    
    if not (mask.dim() == 2 and mask.shape[0] == expected and mask.shape[1] == expected):
        error_msg = (
            f"Mask shape mismatch. Expected square matrix of size {expected}x{expected}, "
            f"got shape {mask.shape}"
        )
        return False, error_msg
    
    return True, None

def prune_output_edges(topology: nx.Graph, output_nodes: List[int]) -> nx.Graph:
    """
    Prune edges between output nodes while preserving other connections.
    
    Args:
        topology: NetworkX graph representing the network topology
        output_nodes: List of output node indices
        
    Returns:
        NetworkX graph with output-output edges removed
    """
    # Create a copy of the topology to avoid modifying the original
    pruned_topology = topology.copy()
    
    # Remove edges between output nodes
    for i in output_nodes:
        for j in output_nodes:
            if pruned_topology.has_edge(i, j):
                pruned_topology.remove_edge(i, j)
    
    return pruned_topology

def mask_shape_sanity(mask: torch.Tensor, expected_shape: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Validate the shape of a mask tensor against expected dimensions.
    
    Args:
        mask: The mask tensor to validate
        expected_shape: Expected (height, width) of the mask
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(mask, torch.Tensor):
        return False, "Mask must be a PyTorch tensor"
    
    if mask.dim() != 2:
        return False, f"Mask must be 2D, got {mask.dim()}D"
    
    if mask.shape != expected_shape:
        return False, f"Expected shape {expected_shape}, got {mask.shape}"
    
    if not torch.all((mask == 0) | (mask == 1)):
        return False, "Mask must contain only binary values (0 or 1)"
    
    return True, "" 