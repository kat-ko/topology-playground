from typing import Dict, Any, Type, Optional
from .base import BasePlugin

class PluginRegistry:
    """Registry for managing plugins in the system."""
    
    _instance = None
    _plugins: Dict[str, Dict[str, Type[BasePlugin]]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, plugin_type: str, name: str):
        """Decorator to register a plugin class.
        
        Args:
            plugin_type: Type of plugin (e.g., 'topologies', 'tasks')
            name: Name of the plugin
        """
        def decorator(plugin_class: Type[BasePlugin]):
            if plugin_type not in cls._plugins:
                cls._plugins[plugin_type] = {}
            cls._plugins[plugin_type][name] = plugin_class
            return plugin_class
        return decorator
    
    @classmethod
    def get_plugin(cls, plugin_type: str, name: str) -> Optional[Type[BasePlugin]]:
        """Get a plugin class by type and name.
        
        Args:
            plugin_type: Type of plugin
            name: Name of the plugin
            
        Returns:
            The plugin class if found, None otherwise
        """
        return cls._plugins.get(plugin_type, {}).get(name)
    
    @classmethod
    def list_plugins(cls, plugin_type: str) -> Dict[str, Type[BasePlugin]]:
        """List all plugins of a given type.
        
        Args:
            plugin_type: Type of plugin
            
        Returns:
            Dictionary of plugin names to plugin classes
        """
        return cls._plugins.get(plugin_type, {}) 