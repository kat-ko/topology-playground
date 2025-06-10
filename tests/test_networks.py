import networkx as nx
import pytest
from src.networks.ffn import FeedForwardNetwork

def test_weight_initialization_matches_topology():
    """Test that weight initialization matches the topology structure."""
    # Create a simple DAG topology
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])  # Simple DAG: 0 -> 1 -> 2 and 0 -> 2
    
    # Create network
    network = FeedForwardNetwork(
        topology=G,
        input_nodes=[0],
        output_nodes=[2],
        network_params={}
    )
    
    # Verify weight initialization
    for node in G.nodes():
        # Check that all weights correspond to actual edges
        for neighbor in network.node_states[node]['weights'].keys():
            assert G.has_edge(neighbor, node), f"Weight exists for non-existent edge {neighbor} -> {node}"
        
        # Check that all incoming edges have weights
        for neighbor in G.predecessors(node):
            assert neighbor in network.node_states[node]['weights'], f"Missing weight for edge {neighbor} -> {node}"
    
    # Test with invalid topology (cycle)
    G_cycle = nx.DiGraph()
    G_cycle.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Cycle: 0 -> 1 -> 2 -> 0
    
    with pytest.raises(ValueError):
        FeedForwardNetwork(
            topology=G_cycle,
            input_nodes=[0],
            output_nodes=[2],
            network_params={}
        )
    
    # Test with missing weights
    G_missing = nx.DiGraph()
    G_missing.add_edges_from([(0, 1), (1, 2)])
    
    network = FeedForwardNetwork(
        topology=G_missing,
        input_nodes=[0],
        output_nodes=[2],
        network_params={}
    )
    
    # Manually remove a weight to simulate missing weight
    del network.node_states[1]['weights'][0]
    
    with pytest.raises(ValueError):
        network._validate_weight_initialization() 