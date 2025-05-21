from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Any, List, Optional, Union

class BaseTopology(ABC):
    """Abstract base class for network topologies."""
    
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