from typing import Dict, Any, List, Set, Tuple
import networkx as nx
import numpy as np
from .utils import prune_output_edges

class TopologyValidator:
    """Validates network topology constraints and edge connections."""
    
    def __init__(self, topology: nx.Graph, input_nodes: List[int], output_nodes: List[int]):
        """
        Initialize the topology validator.
        
        Args:
            topology: NetworkX graph representing the network topology
            input_nodes: List of input node indices
            output_nodes: List of output node indices
        """
        self.topology = topology
        self.input_nodes = set(input_nodes)
        self.output_nodes = set(output_nodes)
        self.hidden_nodes = set(topology.nodes()) - self.input_nodes - self.output_nodes
    
    def validate_forbidden_edges(self) -> Tuple[bool, str]:
        """
        Validate that no forbidden edges exist in the topology.
        Forbidden edges are:
        1. Input -> Input edges
        2. Output -> Output edges
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for input-input edges
        for i in self.input_nodes:
            for j in self.input_nodes:
                if self.topology.has_edge(i, j):
                    return False, f"Forbidden edge detected: Input node {i} -> Input node {j}"
        
        # Check for output-output edges
        for i in self.output_nodes:
            for j in self.output_nodes:
                if self.topology.has_edge(i, j):
                    return False, f"Forbidden edge detected: Output node {i} -> Output node {j}"
        
        return True, "No forbidden edges detected"
    
    def validate_forward_influence(self, test_inputs: Dict[int, float]) -> Tuple[bool, str]:
        """
        Validate forward influence by running a deterministic forward pass and checking
        that forbidden edges don't carry signal.
        
        Args:
            test_inputs: Dictionary mapping input node indices to their input values
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Store original activations
        original_activations = self._run_forward_pass(test_inputs)
        
        # Zero out hidden activations
        modified_activations = self._run_forward_pass(test_inputs, zero_hidden=True)
        
        # Compare outputs
        for node in self.output_nodes:
            if not np.isclose(original_activations[node], modified_activations[node], atol=1e-6):
                return False, f"Output node {node} activation changed when hidden nodes were zeroed"
        
        return True, "Forward influence validation passed"
    
    def _run_forward_pass(self, inputs: Dict[int, float], zero_hidden: bool = False) -> Dict[int, float]:
        """
        Run a deterministic forward pass through the network.
        
        Args:
            inputs: Dictionary mapping input node indices to their input values
            zero_hidden: Whether to zero out hidden node activations
            
        Returns:
            Dictionary mapping node indices to their activations
        """
        # Initialize activations
        activations = {node: 0.0 for node in self.topology.nodes()}
        
        # Set input node activations
        for node, value in inputs.items():
            if node in self.input_nodes:
                activations[node] = value
        
        # Process through network layers
        for layer in nx.topological_sort(self.topology):
            if layer not in self.input_nodes:
                if zero_hidden and layer in self.hidden_nodes:
                    activations[layer] = 0.0
                    continue
                
                # Sum weighted inputs
                weighted_sum = 0.0
                for neighbor in self.topology.predecessors(layer):
                    weighted_sum += activations[neighbor]
                
                # Apply activation function (ReLU)
                activations[layer] = max(0, weighted_sum)
        
        return activations
    
    def get_forward_influence_map(self) -> Dict[int, Set[int]]:
        """
        Generate a forward influence map showing which nodes influence each output node.
        
        Returns:
            Dictionary mapping output node indices to sets of nodes that influence them
        """
        influence_map = {node: set() for node in self.output_nodes}
        
        # For each output node, find all nodes that can reach it
        for output_node in self.output_nodes:
            # Get all nodes that can reach this output node
            for node in self.topology.nodes():
                if nx.has_path(self.topology, node, output_node):
                    influence_map[output_node].add(node)
        
        return influence_map 