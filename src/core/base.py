from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePlugin(ABC):
    """Base class for all plugins in the system."""
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get the plugin's parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of plugin parameters
        """
        pass 