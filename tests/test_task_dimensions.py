#!/usr/bin/env python3
"""
Test script to verify that all topologies work correctly with different task dimensions.
Tests CartPole-v1 (4+2), Acrobot-v1 (6+3), and LunarLander-v2 (8+4) dimensions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import networkx as nx
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.topologies.fully_connected import FullyConnectedTopology
from src.topologies.standard_mlp import StandardMLPTopology
from src.networks.ffn import FeedForwardNetwork

def test_topology_dimensions(topology_type: str, topology_class, config: dict):
    """Test a topology with different task dimensions."""
    print(f"\n🔍 Testing {topology_type.upper()} Topology")
    print("=" * 50)
    
    # Task configurations
    tasks = [
        ("CartPole-v1", 4, 2, 134),
        ("Acrobot-v1", 6, 3, 137), 
        ("LunarLander-v2", 8, 4, 140)
    ]
    
    results = {}
    
    for task_name, input_dim, output_dim, expected_nodes in tasks:
        print(f"\n📊 {task_name}: {input_dim} input + 128 hidden + {output_dim} output = {expected_nodes} nodes")
        
        try:
            # Create topology
            topology = topology_class(**config)
            
            # Generate graph
            graph = topology.generate(input_dim=input_dim, output_dim=output_dim)
            
            # Verify node count
            actual_nodes = len(graph.nodes())
            node_status = "✅" if actual_nodes == expected_nodes else "❌"
            print(f"  {node_status} Nodes: {actual_nodes} (expected: {expected_nodes})")
            
            # Verify edge count > 0
            actual_edges = len(graph.edges())
            edge_status = "✅" if actual_edges > 0 else "❌"
            print(f"  {edge_status} Edges: {actual_edges}")
            
            # Verify DAG properties
            is_directed = graph.is_directed()
            is_acyclic = nx.is_directed_acyclic_graph(graph)
            is_connected = nx.is_weakly_connected(graph)
            
            print(f"  {'✅' if is_directed else '❌'} Directed: {is_directed}")
            print(f"  {'✅' if is_acyclic else '❌'} Acyclic: {is_acyclic}")
            print(f"  {'✅' if is_connected else '❌'} Connected: {is_connected}")
            
            # Test FFN creation
            input_nodes = list(range(input_dim))
            output_nodes = list(range(input_dim + 128, input_dim + 128 + output_dim))
            
            network = FeedForwardNetwork(
                graph, 
                input_nodes, 
                output_nodes, 
                {'learning_rate': 0.001, 'activation': 'relu'}
            )
            
            # Test forward pass
            test_input = {node: torch.randn(1) for node in input_nodes}
            output = network.forward(test_input)
            
            forward_success = len(output) == output_dim
            print(f"  {'✅' if forward_success else '❌'} Forward Pass: {len(output)} outputs (expected: {output_dim})")
            
            # Count parameters
            actual_params = 0
            if hasattr(network, 'node_states'):
                for node, state in network.node_states.items():
                    if 'bias' in state:
                        actual_params += 1
                    if 'weights' in state:
                        actual_params += len(state['weights'])
            
            print(f"  🔢 Parameters: {actual_params}")
            
            # Store results
            results[task_name] = {
                'success': actual_nodes == expected_nodes and actual_edges > 0 and is_directed and is_acyclic and is_connected and forward_success,
                'nodes': actual_nodes,
                'edges': actual_edges,
                'parameters': actual_params,
                'forward_pass': forward_success
            }
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[task_name] = {
                'success': False,
                'error': str(e)
            }
    
    return results

def main():
    """Run comprehensive topology dimension testing."""
    print("🧪 Topology Task Dimension Compatibility Test")
    print("=" * 60)
    
    # Topology configurations
    topologies = {
        'small_world': (SmallWorldTopology, {'size': 128, 'k': 4, 'p': 0.2, 'seed': 42}),
        'modular': (ModularTopology, {'size': 128, 'num_modules': 4, 'inter_module_prob': 0.1, 'intra_module_prob': 0.8, 'seed': 42}),
        'hybrid': (HybridTopology, {'size': 128, 'num_modules': 4, 'k': 4, 'p': 0.2, 'inter_module_prob': 0.1, 'seed': 42}),
        'fully_connected': (FullyConnectedTopology, {'size': 128, 'seed': 42}),
        'standard_mlp': (StandardMLPTopology, {'size': 128, 'num_layers': 3, 'activation': 'relu', 'seed': 42})
    }
    
    all_results = {}
    
    for topology_type, (topology_class, config) in topologies.items():
        results = test_topology_dimensions(topology_type, topology_class, config)
        all_results[topology_type] = results
    
    # Summary report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    for topology_type, results in all_results.items():
        print(f"\n🎯 {topology_type.upper()}")
        success_count = sum(1 for task_result in results.values() if task_result.get('success', False))
        total_tasks = len(results)
        print(f"  Success Rate: {success_count}/{total_tasks} tasks")
        
        for task_name, task_result in results.items():
            status = "✅ PASS" if task_result.get('success', False) else "❌ FAIL"
            print(f"    {task_name}: {status}")
            if not task_result.get('success', False) and 'error' in task_result:
                print(f"      Error: {task_result['error']}")
    
    # Overall assessment
    total_success = sum(
        sum(1 for task_result in results.values() if task_result.get('success', False))
        for results in all_results.values()
    )
    total_tests = sum(len(results) for results in all_results.values())
    
    print(f"\n🎉 OVERALL SUCCESS RATE: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
    
    if total_success == total_tests:
        print("🎯 ALL TOPOLOGIES ARE FULLY COMPATIBLE WITH ALL TASK DIMENSIONS!")
    else:
        print("⚠️  Some compatibility issues detected. Check individual results above.")

if __name__ == "__main__":
    main()
