"""
Capacity Measurement System

This module manages the measurement and storage of actual network capacities
from same_size experiments for use in match_* experiments.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CapacityMeasurement:
    """Represents a single capacity measurement."""
    topology: str
    size: int
    network_type: str
    num_layers: int
    actual_capacity: int
    measurement_time: str
    seed: int = 42

class CapacityMeasurementManager:
    """
    Manages capacity measurements from same_size experiments.
    Stores and retrieves measured capacities for use in match_* experiments.
    """
    
    def __init__(self, config: Dict[str, Any], cache_dir: str = "capacity_measurements"):
        """
        Initialize the capacity measurement manager.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory to store capacity measurements
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store measurements in memory for fast access
        self.measurements: Dict[str, CapacityMeasurement] = {}
        
        # Load existing measurements if available
        self._load_measurements()
    
    def _get_measurement_key(self, topology: str, size: int, network_type: str, num_layers: int) -> str:
        """Generate a key for storing/retrieving measurements."""
        return f"{topology}_{size}_{network_type}_{num_layers}"
    
    def _get_cache_file(self) -> Path:
        """Get the cache file path for measurements."""
        config_hash = self._get_config_hash()
        return self.cache_dir / f"capacity_measurements_{config_hash}.json"
    
    def _get_config_hash(self) -> str:
        """Generate a hash of the configuration for caching."""
        import hashlib
        
        # Create a simplified config for hashing (only relevant parameters)
        hash_config = {
            'network_types': self.config['network_types'],
            'num_layers': self.config['num_layers'],
            'num_io_nodes': self.config['num_io_nodes'],
            'small_world_params': self.config['small_world_params'],
            'modular_params': self.config['modular_params'],
            'fully_connected_params': self.config['fully_connected_params'],
            'network_params': self.config['network_params']
        }
        
        config_str = json.dumps(hash_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _load_measurements(self):
        """Load existing measurements from cache file."""
        cache_file = self._get_cache_file()
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                for key, measurement_data in data.items():
                    self.measurements[key] = CapacityMeasurement(**measurement_data)
                
                logger.info(f"Loaded {len(self.measurements)} capacity measurements from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load capacity measurements from {cache_file}: {e}")
    
    def _save_measurements(self):
        """Save measurements to cache file."""
        cache_file = self._get_cache_file()
        
        # Convert to serializable format
        data = {}
        for key, measurement in self.measurements.items():
            data[key] = asdict(measurement)
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.measurements)} capacity measurements to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save capacity measurements to {cache_file}: {e}")
    
    def measure_capacity(self, topology: str, size: int, network_type: str, num_layers: int, seed: int = 42) -> int:
        """
        Measure the actual capacity of a network configuration.
        
        Args:
            topology: The topology type
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            seed: Random seed
            
        Returns:
            Actual capacity (number of parameters)
        """
        # Import here to avoid circular imports
        from ..topologies.small_world import SmallWorldTopology
        from ..topologies.modular import ModularTopology
        from ..topologies.hybrid import HybridTopology
        from ..topologies.fully_connected import FullyConnectedTopology
        from ..networks.ffn import FeedForwardNetwork
        from ..networks.rnn import RecurrentNetwork
        
        # Set random seed for consistent results
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Create topology
        topology_classes = {
            'small_world': SmallWorldTopology,
            'modular': ModularTopology,
            'hybrid': HybridTopology,
            'fully_connected': FullyConnectedTopology
        }
        
        network_classes = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
        
        topology_class = topology_classes[topology]
        
        # Create topology instance
        if topology == 'small_world':
            topology_instance = topology_class(
                size=size,
                k=self.config['small_world_params']['k'],
                p=self.config['small_world_params']['p'],
                num_layers=num_layers,
                inter_layer_prob=self.config['small_world_params']['inter_layer_prob'],
                seed=seed
            )
        elif topology == 'modular':
            topology_instance = topology_class(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                intra_module_prob=self.config['modular_params']['intra_module_prob'],
                num_layers=num_layers,
                inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                seed=seed
            )
        elif topology == 'hybrid':
            topology_instance = topology_class(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                k=self.config['small_world_params']['k'],
                p=self.config['small_world_params']['p'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                num_layers=num_layers,
                inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                seed=seed
            )
        elif topology == 'fully_connected':
            topology_instance = topology_class(
                size=size,
                num_layers=num_layers,
                inter_layer_prob=self.config['fully_connected_params']['inter_layer_prob'],
                intra_layer_prob=self.config['fully_connected_params']['intra_layer_prob'],
                seed=seed
            )
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # Generate graphs
        graphs = topology_instance.generate(num_layers)
        if num_layers == 1:
            graphs = [graphs]
        
        # Select input/output nodes for each layer
        def select_nodes(graph, strategy, size, seed):
            rng = np.random.RandomState(seed)
            all_nodes = list(range(size))
            rng.shuffle(all_nodes)
            num_io_nodes = self.config['num_io_nodes']
            input_nodes = all_nodes[:num_io_nodes]
            output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
            return input_nodes, output_nodes
        
        # Create networks for each layer and sum their parameters
        total_capacity = 0
        
        for layer_idx in range(num_layers):
            input_nodes, output_nodes = select_nodes(graphs[layer_idx], 'random', size, seed)
            
            # Create network for this layer
            network_class = network_classes[network_type]
            network_params = self.config['network_params'][network_type]
            
            network = network_class(graphs[layer_idx], input_nodes, output_nodes, network_params)
            
            # Get actual capacity for this layer
            metrics = network.get_network_metrics()
            layer_capacity = sum(
                metrics.get(k, 0) for k in metrics if k.startswith('num_')
            )
            total_capacity += layer_capacity
        
        return total_capacity
    
    def store_measurement(self, topology: str, size: int, network_type: str, num_layers: int, 
                         actual_capacity: int, seed: int = 42):
        """
        Store a capacity measurement.
        
        Args:
            topology: The topology type
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            actual_capacity: Measured capacity
            seed: Random seed
        """
        key = self._get_measurement_key(topology, size, network_type, num_layers)
        
        measurement = CapacityMeasurement(
            topology=topology,
            size=size,
            network_type=network_type,
            num_layers=num_layers,
            actual_capacity=actual_capacity,
            measurement_time=datetime.now().isoformat(),
            seed=seed
        )
        
        self.measurements[key] = measurement
        logger.debug(f"Stored measurement: {key} = {actual_capacity} params")
    
    def get_measurement(self, topology: str, size: int, network_type: str, num_layers: int) -> Optional[int]:
        """
        Get a stored capacity measurement.
        
        Args:
            topology: The topology type
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            Stored capacity measurement, or None if not found
        """
        key = self._get_measurement_key(topology, size, network_type, num_layers)
        measurement = self.measurements.get(key)
        
        if measurement:
            return measurement.actual_capacity
        return None
    
    def get_target_capacity(self, reference_topology: str, size: int, network_type: str, num_layers: int) -> Optional[int]:
        """
        Get the target capacity for a match_* experiment.
        
        Args:
            reference_topology: The topology to match (e.g., 'small_world' for 'match_small_world')
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            Target capacity from the reference topology, or None if not found
        """
        return self.get_measurement(reference_topology, size, network_type, num_layers)
    
    def has_all_measurements(self, size: int, network_type: str, num_layers: int) -> bool:
        """
        Check if all topology measurements are available for a given configuration.
        
        Args:
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            True if all topology measurements are available
        """
        topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
        
        for topology in topologies:
            if not self.get_measurement(topology, size, network_type, num_layers):
                return False
        
        return True
    
    def get_measurement_summary(self) -> Dict[str, Any]:
        """Get a summary of all stored measurements."""
        summary = {
            'total_measurements': len(self.measurements),
            'measurements_by_topology': {},
            'measurements_by_network_type': {},
            'measurements_by_size': {}
        }
        
        for key, measurement in self.measurements.items():
            # Group by topology
            if measurement.topology not in summary['measurements_by_topology']:
                summary['measurements_by_topology'][measurement.topology] = []
            summary['measurements_by_topology'][measurement.topology].append({
                'size': measurement.size,
                'network_type': measurement.network_type,
                'num_layers': measurement.num_layers,
                'capacity': measurement.actual_capacity
            })
            
            # Group by network type
            if measurement.network_type not in summary['measurements_by_network_type']:
                summary['measurements_by_network_type'][measurement.network_type] = []
            summary['measurements_by_network_type'][measurement.network_type].append({
                'topology': measurement.topology,
                'size': measurement.size,
                'num_layers': measurement.num_layers,
                'capacity': measurement.actual_capacity
            })
            
            # Group by size
            if measurement.size not in summary['measurements_by_size']:
                summary['measurements_by_size'][measurement.size] = []
            summary['measurements_by_size'][measurement.size].append({
                'topology': measurement.topology,
                'network_type': measurement.network_type,
                'num_layers': measurement.num_layers,
                'capacity': measurement.actual_capacity
            })
        
        return summary
    
    def save_to_cache(self):
        """Save all measurements to cache file."""
        self._save_measurements()
    
    def clear_cache(self):
        """Clear all measurements from memory and cache."""
        self.measurements.clear()
        cache_file = self._get_cache_file()
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cleared all capacity measurements") 