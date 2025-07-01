"""
Capacity Mapping System

This module provides systematic capacity mapping by measuring actual network capacities
at different sizes and caching the results for use in training and parameter matching.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

from ..topologies.small_world import SmallWorldTopology
from ..topologies.modular import ModularTopology
from ..topologies.hybrid import HybridTopology
from ..topologies.fully_connected import FullyConnectedTopology
from ..networks.ffn import FeedForwardNetwork
from ..networks.rnn import RecurrentNetwork

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

@dataclass
class CapacityMap:
    """Represents the complete capacity mapping for a configuration."""
    config_hash: str
    measurements: List[CapacityMeasurement]
    created_time: str
    size_range: Tuple[int, int]
    size_step: int
    fine_step: int
    max_divergence: float

class CapacityMapper:
    """
    Systematic capacity mapping system that measures actual network capacities
    at different sizes and provides efficient lookup for parameter matching.
    """
    
    def __init__(self, config: Dict[str, Any], cache_dir: str = "capacity_cache"):
        """
        Initialize the capacity mapper.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory to store capacity mappings
        """
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration for systematic measurement - use more reasonable ranges
        self.size_range = (50, 500)  # Start from 50 to ensure minimum viable sizes
        self.coarse_step = 10  # Step size for coarse measurement
        self.fine_step = 2     # Step size for fine-tuning
        self.max_divergence = 5.0  # Maximum acceptable divergence in %
        
        # Network type mapping
        self.network_types = {
            'ffn': FeedForwardNetwork,
            'rnn': RecurrentNetwork
        }
        
        # Topology mapping
        self.topology_classes = {
            'small_world': SmallWorldTopology,
            'modular': ModularTopology,
            'hybrid': HybridTopology,
            'fully_connected': FullyConnectedTopology
        }
        
        # Load or create capacity mappings
        self.capacity_maps = self._load_or_create_mappings()
    
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
    
    def _load_or_create_mappings(self) -> Dict[str, CapacityMap]:
        """Load existing capacity mappings or create new ones."""
        config_hash = self._get_config_hash()
        cache_file = self.cache_dir / f"capacity_map_{config_hash}.json"
        
        if cache_file.exists():
            logger.info(f"Loading existing capacity mappings from {cache_file}")
            return self._load_mappings_from_file(cache_file)
        else:
            logger.info(f"Creating new capacity mappings for config {config_hash}")
            return self._create_new_mappings(config_hash)
    
    def _load_mappings_from_file(self, cache_file: Path) -> Dict[str, CapacityMap]:
        """Load capacity mappings from a JSON file."""
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            mappings = {}
            for key, map_data in data.items():
                measurements = [CapacityMeasurement(**m) for m in map_data['measurements']]
                mappings[key] = CapacityMap(
                    config_hash=map_data['config_hash'],
                    measurements=measurements,
                    created_time=map_data['created_time'],
                    size_range=tuple(map_data['size_range']),
                    size_step=map_data['size_step'],
                    fine_step=map_data['fine_step'],
                    max_divergence=map_data['max_divergence']
                )
            
            return mappings
        except Exception as e:
            logger.warning(f"Failed to load capacity mappings from {cache_file}: {e}")
            return {}
    
    def _create_new_mappings(self, config_hash: str) -> Dict[str, CapacityMap]:
        """Create new capacity mappings by systematically measuring capacities."""
        mappings = {}
        
        for topology in ['small_world', 'modular', 'hybrid', 'fully_connected']:
            for network_type in self.config['network_types']:
                for num_layers in self.config['num_layers']:
                    key = f"{topology}_{network_type}_{num_layers}"
                    logger.info(f"Creating capacity mapping for {key}")
                    
                    measurements = self._measure_capacity_range(
                        topology, network_type, num_layers
                    )
                    
                    mappings[key] = CapacityMap(
                        config_hash=config_hash,
                        measurements=measurements,
                        created_time=datetime.now().isoformat(),
                        size_range=self.size_range,
                        size_step=self.coarse_step,
                        fine_step=self.fine_step,
                        max_divergence=self.max_divergence
                    )
        
        # Save mappings to file
        self._save_mappings(mappings, config_hash)
        
        return mappings
    
    def _measure_capacity_range(self, topology: str, network_type: str, num_layers: int) -> List[CapacityMeasurement]:
        """Systematically measure capacities across a range of sizes."""
        measurements = []
        
        # Get minimum viable size for this topology
        min_viable_size = self._get_minimum_viable_size(topology)
        
        # Ensure we start from a valid size
        start_size = max(self.size_range[0], min_viable_size)
        
        # Coarse measurement: measure every coarse_step sizes
        for size in range(start_size, self.size_range[1] + 1, self.coarse_step):
            try:
                capacity = self._measure_single_capacity(topology, network_type, num_layers, size)
                measurements.append(CapacityMeasurement(
                    topology=topology,
                    size=size,
                    network_type=network_type,
                    num_layers=num_layers,
                    actual_capacity=capacity,
                    measurement_time=datetime.now().isoformat()
                ))
                logger.debug(f"Measured {topology} {network_type} {num_layers}L at size {size}: {capacity} params")
            except Exception as e:
                logger.warning(f"Failed to measure {topology} {network_type} {num_layers}L at size {size}: {e}")
                continue
        
        # Fine measurement: measure around key sizes for better precision
        fine_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450]
        for target_size in fine_sizes:
            if target_size < start_size or target_size > self.size_range[1]:
                continue
            
            # Measure sizes around the target for fine-tuning
            for offset in range(-self.fine_step, self.fine_step + 1, self.fine_step):
                size = target_size + offset
                if size < start_size or size > self.size_range[1]:
                    continue
                
                # Skip if already measured
                if any(m.size == size for m in measurements):
                    continue
                
                try:
                    capacity = self._measure_single_capacity(topology, network_type, num_layers, size)
                    measurements.append(CapacityMeasurement(
                        topology=topology,
                        size=size,
                        network_type=network_type,
                        num_layers=num_layers,
                        actual_capacity=capacity,
                        measurement_time=datetime.now().isoformat()
                    ))
                    logger.debug(f"Fine measurement: {topology} {network_type} {num_layers}L at size {size}: {capacity} params")
                except Exception as e:
                    logger.warning(f"Failed fine measurement {topology} {network_type} {num_layers}L at size {size}: {e}")
                    continue
        
        # Sort by size
        measurements.sort(key=lambda m: m.size)
        return measurements
    
    def _measure_single_capacity(self, topology: str, network_type: str, num_layers: int, size: int) -> int:
        """Measure the actual capacity of a single network configuration."""
        # Set random seed for consistent results
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create topology
        topology_class = self.topology_classes[topology]
        
        if topology == 'small_world':
            topology_instance = topology_class(
                size=size,
                k=self.config['small_world_params']['k'],
                p=self.config['small_world_params']['p'],
                num_layers=num_layers,
                inter_layer_prob=self.config['small_world_params']['inter_layer_prob'],
                seed=42
            )
        elif topology == 'modular':
            topology_instance = topology_class(
                size=size,
                num_modules=self.config['modular_params']['num_modules'],
                inter_module_prob=self.config['modular_params']['inter_module_prob'],
                intra_module_prob=self.config['modular_params']['intra_module_prob'],
                num_layers=num_layers,
                inter_layer_prob=self.config['modular_params']['inter_layer_prob'],
                seed=42
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
                seed=42
            )
        elif topology == 'fully_connected':
            topology_instance = topology_class(
                size=size,
                num_layers=num_layers,
                inter_layer_prob=self.config['fully_connected_params']['inter_layer_prob'],
                intra_layer_prob=self.config['fully_connected_params']['intra_layer_prob'],
                seed=42
            )
        else:
            raise ValueError(f"Unknown topology: {topology}")
        
        # Generate graphs
        graphs = topology_instance.generate(num_layers)
        if num_layers == 1:
            graphs = [graphs]
        
        # Select input/output nodes
        rng = np.random.RandomState(42)
        all_nodes = list(range(size))
        rng.shuffle(all_nodes)
        num_io_nodes = self.config['num_io_nodes']
        input_nodes = all_nodes[:num_io_nodes]
        output_nodes = all_nodes[num_io_nodes:2*num_io_nodes]
        
        # Create network
        network_class = self.network_types[network_type]
        network_params = self.config['network_params'][network_type]
        
        network = network_class(graphs[0], input_nodes, output_nodes, network_params)
        
        # Get actual capacity
        metrics = network.get_network_metrics()
        actual_capacity = sum(
            metrics.get(k, 0) for k in metrics if k.startswith('num_')
        )
        
        return actual_capacity
    
    def _save_mappings(self, mappings: Dict[str, CapacityMap], config_hash: str):
        """Save capacity mappings to a JSON file."""
        cache_file = self.cache_dir / f"capacity_map_{config_hash}.json"
        
        # Convert to serializable format
        data = {}
        for key, capacity_map in mappings.items():
            data[key] = {
                'config_hash': capacity_map.config_hash,
                'measurements': [asdict(m) for m in capacity_map.measurements],
                'created_time': capacity_map.created_time,
                'size_range': list(capacity_map.size_range),
                'size_step': capacity_map.size_step,
                'fine_step': capacity_map.fine_step,
                'max_divergence': capacity_map.max_divergence
            }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved capacity mappings to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save capacity mappings to {cache_file}: {e}")
    
    def find_matching_size(self, topology: str, target_capacity: int, network_type: str, num_layers: int) -> int:
        """
        Find the size needed for a topology to achieve the target capacity.
        
        Args:
            topology: The topology type
            target_capacity: Target parameter count
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            Size needed to achieve target capacity
        """
        key = f"{topology}_{network_type}_{num_layers}"
        
        if key not in self.capacity_maps:
            # If no mapping exists, create a basic one with minimum viable sizes
            return self._get_minimum_viable_size(topology)
        
        measurements = self.capacity_maps[key].measurements
        
        if not measurements:
            # If no measurements, return minimum viable size
            return self._get_minimum_viable_size(topology)
        
        # Find the closest measurement
        best_size = measurements[0].size
        best_divergence = float('inf')
        
        for measurement in measurements:
            divergence = abs(measurement.actual_capacity - target_capacity) / target_capacity
            if divergence < best_divergence:
                best_divergence = divergence
                best_size = measurement.size
        
        # If we're not close enough, interpolate or extrapolate
        if best_divergence > self.max_divergence / 100:
            best_size = self._interpolate_size(measurements, target_capacity)
        
        # Ensure minimum viable size
        min_size = self._get_minimum_viable_size(topology)
        best_size = max(min_size, best_size)
        
        return best_size
    
    def _get_minimum_viable_size(self, topology: str) -> int:
        """Get the minimum viable size for a topology based on its parameters."""
        if topology == 'small_world':
            k = self.config['small_world_params']['k']
            # Small world needs at least k+1 nodes
            return max(50, k * 2)  # Use a reasonable minimum
            
        elif topology == 'modular':
            num_modules = self.config['modular_params']['num_modules']
            # Modular needs at least num_modules * 2 nodes
            return max(50, num_modules * 3)
            
        elif topology == 'hybrid':
            k = self.config['small_world_params']['k']
            num_modules = self.config['modular_params']['num_modules']
            # Hybrid needs at least max(k, num_modules) * 2 nodes
            return max(50, max(k, num_modules) * 3)
            
        elif topology == 'fully_connected':
            # Fully connected can work with smaller sizes
            return 30
            
        else:
            return 50  # Default minimum
    
    def _interpolate_size(self, measurements: List[CapacityMeasurement], target_capacity: int) -> int:
        """Interpolate size based on capacity measurements."""
        if len(measurements) < 2:
            return measurements[0].size if measurements else 50
        
        # Sort by capacity
        sorted_measurements = sorted(measurements, key=lambda m: m.actual_capacity)
        
        # Find the two measurements that bracket the target capacity
        lower_measurement = None
        upper_measurement = None
        
        for i, measurement in enumerate(sorted_measurements):
            if measurement.actual_capacity <= target_capacity:
                lower_measurement = measurement
            else:
                upper_measurement = measurement
                break
        
        # If target is below all measurements, extrapolate down
        if lower_measurement is None:
            if len(sorted_measurements) >= 2:
                m1, m2 = sorted_measurements[0], sorted_measurements[1]
                # Linear extrapolation: size = m1.size + (target - m1.capacity) * (m2.size - m1.size) / (m2.capacity - m1.capacity)
                size_diff = m2.size - m1.size
                capacity_diff = m2.actual_capacity - m1.actual_capacity
                if capacity_diff != 0:
                    extrapolated_size = m1.size + (target_capacity - m1.actual_capacity) * size_diff / capacity_diff
                    return max(30, int(extrapolated_size))
            return sorted_measurements[0].size
        
        # If target is above all measurements, extrapolate up
        if upper_measurement is None:
            if len(sorted_measurements) >= 2:
                m1, m2 = sorted_measurements[-2], sorted_measurements[-1]
                size_diff = m2.size - m1.size
                capacity_diff = m2.actual_capacity - m1.actual_capacity
                if capacity_diff != 0:
                    extrapolated_size = m2.size + (target_capacity - m2.actual_capacity) * size_diff / capacity_diff
                    return min(1000, int(extrapolated_size))
            return sorted_measurements[-1].size
        
        # Interpolate between the two measurements
        size_diff = upper_measurement.size - lower_measurement.size
        capacity_diff = upper_measurement.actual_capacity - lower_measurement.actual_capacity
        
        if capacity_diff == 0:
            return lower_measurement.size
        
        # Linear interpolation
        interpolation_factor = (target_capacity - lower_measurement.actual_capacity) / capacity_diff
        interpolated_size = lower_measurement.size + interpolation_factor * size_diff
        
        return int(interpolated_size)
    
    def get_capacity_at_size(self, topology: str, size: int, network_type: str, num_layers: int) -> Optional[int]:
        """
        Get the capacity at a specific size from the measurements.
        
        Args:
            topology: The topology type
            size: Network size
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            Capacity at the given size, or None if not measured
        """
        key = f"{topology}_{network_type}_{num_layers}"
        
        if key not in self.capacity_maps:
            return None
        
        measurements = self.capacity_maps[key].measurements
        
        for measurement in measurements:
            if measurement.size == size:
                return measurement.actual_capacity
        
        return None
    
    def get_capacity_curve(self, topology: str, network_type: str, num_layers: int) -> List[Tuple[int, int]]:
        """
        Get the capacity curve (size vs capacity) for a topology.
        
        Args:
            topology: The topology type
            network_type: Network type (ffn/rnn)
            num_layers: Number of layers
            
        Returns:
            List of (size, capacity) tuples
        """
        key = f"{topology}_{network_type}_{num_layers}"
        
        if key not in self.capacity_maps:
            return []
        
        measurements = self.capacity_maps[key].measurements
        return [(m.size, m.actual_capacity) for m in measurements]
    
    def refresh_mappings(self):
        """Refresh all capacity mappings by re-measuring."""
        logger.info("Refreshing all capacity mappings")
        config_hash = self._get_config_hash()
        self.capacity_maps = self._create_new_mappings(config_hash)
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the capacity mappings."""
        stats = {
            'total_mappings': len(self.capacity_maps),
            'mappings': {}
        }
        
        for key, capacity_map in self.capacity_maps.items():
            measurements = capacity_map.measurements
            if measurements:
                capacities = [m.actual_capacity for m in measurements]
                sizes = [m.size for m in measurements]
                
                stats['mappings'][key] = {
                    'num_measurements': len(measurements),
                    'size_range': (min(sizes), max(sizes)),
                    'capacity_range': (min(capacities), max(capacities)),
                    'created_time': capacity_map.created_time
                }
        
        return stats 