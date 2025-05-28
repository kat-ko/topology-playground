from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Any, List, Optional, Union, Tuple
import torch
from .utils import mask_shape_sanity

class BaseTopology(ABC):
    """Abstract base class for network topologies."""
    
    def __init__(self, 
                 n_in: int,
                 n_hidden: int,
                 n_out: int,
                 is_test_run: bool = False):
        """
        Initialize the topology.
        
        Args:
            n_in: Number of input nodes
            n_hidden: Number of hidden nodes
            n_out: Number of output nodes
            is_test_run: Whether this is a test run
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.is_test_run = is_test_run
        self._validate_dimensions()
    
    def _validate_dimensions(self) -> None:
        """Validate the network dimensions."""
        if self.n_in < 0 or self.n_hidden < 0 or self.n_out < 0:
            raise ValueError("Network dimensions must be non-negative")
    
    def validate_mask(self, mask: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validate the mask shape.
        
        Args:
            mask: The adjacency mask to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return mask_shape_sanity(
            mask=mask,
            n_in=self.n_in,
            n_hidden=self.n_hidden,
            n_out=self.n_out,
            is_test_run=self.is_test_run
        )
    
    @abstractmethod
    def generate(self, num_layers: int = 1) -> Union[nx.Graph, List[nx.Graph]]:
        """Generate the network topology.
        
        Args:
            num_layers: Number of layers to generate (default: 1)
            
        Returns:
            If num_layers=1: networkx.Graph
            If num_layers>1: List[networkx.Graph]
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the topology parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of topology parameters
        """
        pass
    
    @abstractmethod
    def get_layer_connections(self, layer1: int, layer2: int) -> Optional[nx.Graph]:
        """Get the inter-layer connections between two layers.
        
        Args:
            layer1: Index of first layer
            layer2: Index of second layer
            
        Returns:
            networkx.Graph or None: Graph representing inter-layer connections,
                                  or None if layers are not connected
        """
        pass
    
    @abstractmethod
    def get_layer_metrics(self, layer: int) -> Dict[str, Any]:
        """Get metrics specific to a particular layer.
        
        Args:
            layer: Index of the layer
            
        Returns:
            Dict[str, Any]: Dictionary of layer-specific metrics
        """
        pass
    
    @abstractmethod
    def generate_adjacency_mask(self) -> torch.Tensor:
        """
        Generate the adjacency mask for the network.
        
        Returns:
            Binary adjacency mask tensor
        """
        pass 