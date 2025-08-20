#!/usr/bin/env python3
"""
Systematic Topology Testing Script
Tests each topology type to ensure methodologically sound integration.
"""

import sys
import os
import traceback
import networkx as nx
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

# Add src to path and set up proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import topology classes directly from their modules
try:
    from src.topologies.small_world import SmallWorldTopology
    from src.topologies.modular import ModularTopology
    from src.topologies.hybrid import HybridTopology
    from src.topologies.fully_connected import FullyConnectedTopology
    from src.topologies.standard_mlp import StandardMLPTopology
    from src.networks.ffn import FeedForwardNetwork
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import paths...")
    
    # Try alternative import paths
    try:
        from topologies.small_world import SmallWorldTopology
        from topologies.modular import ModularTopology
        from topologies.hybrid import HybridTopology
        from topologies.fully_connected import FullyConnectedTopology
        from topologies.standard_mlp import StandardMLPTopology
        from networks.ffn import FeedForwardNetwork
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("Please check the project structure and import paths.")
        sys.exit(1)

class TopologySystematicTester:
    """Systematic tester for all topology types."""
    
    def __init__(self):
        self.test_results = {}
        self.test_configs = {
            'small_world': {
                'size': 128,
                'k': 4,
                'p': 0.2,
                'seed': 42
            },
            'modular': {
                'size': 128,
                'num_modules': 4,
                'inter_module_prob': 0.1,
                'intra_module_prob': 0.8,
                'seed': 42
            },
            'hybrid': {
                'size': 128,
                'num_modules': 4,
                'k': 4,
                'p': 0.2,
                'inter_module_prob': 0.1,
                'seed': 42
            },
            'fully_connected': {
                'size': 128,
                'seed': 42
            },
            'standard_mlp': {
                'size': 128,
                'num_layers': 3,
                'activation': 'relu',
                'seed': 42
            }
        }
        
        # CartPole environment dimensions
        self.obs_dim = 4
        self.action_dim = 2
        
    def test_topology_creation(self, topology_type: str) -> Dict[str, Any]:
        """Test basic topology creation and validation."""
        print(f"🔍 Testing {topology_type} topology creation...")
        
        try:
            # Create topology instance
            config = self.test_configs[topology_type].copy()
            
            if topology_type == 'small_world':
                topology = SmallWorldTopology(**config)
            elif topology_type == 'modular':
                topology = ModularTopology(**config)
            elif topology_type == 'hybrid':
                topology = HybridTopology(**config)
            elif topology_type == 'fully_connected':
                topology = FullyConnectedTopology(**config)
            elif topology_type == 'standard_mlp':
                topology = StandardMLPTopology(**config)
            else:
                raise ValueError(f"Unknown topology type: {topology_type}")
            
            # Test basic properties
            result = {
                'success': True,
                'topology_class': topology.__class__.__name__,
                'size': getattr(topology, 'size', None),
                'parameters': topology.get_parameters(),
                'errors': []
            }
            
            print(f"  ✅ {topology_type} created successfully")
            print(f"  📊 Size: {result['size']}")
            print(f"  ⚙️  Parameters: {result['parameters']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to create {topology_type}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {
                'success': False,
                'topology_class': None,
                'size': None,
                'parameters': {},
                'errors': [error_msg]
            }
    
    def test_graph_generation(self, topology_type: str) -> Dict[str, Any]:
        """Test graph generation and validation."""
        print(f"🔄 Testing {topology_type} graph generation...")
        
        try:
            # Create topology and generate graph
            config = self.test_configs[topology_type].copy()
            
            if topology_type == 'small_world':
                topology = SmallWorldTopology(**config)
            elif topology_type == 'modular':
                topology = ModularTopology(**config)
            elif topology_type == 'hybrid':
                topology = HybridTopology(**config)
            elif topology_type == 'fully_connected':
                topology = FullyConnectedTopology(**config)
            elif topology_type == 'standard_mlp':
                topology = StandardMLPTopology(**config)
            
            # Generate graph with input/output dimensions
            graph = topology.generate(input_dim=self.obs_dim, output_dim=self.action_dim)
            
            # Validate graph structure
            result = {
                'success': True,
                'graph_type': type(graph).__name__,
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'is_directed': graph.is_directed(),
                'is_acyclic': nx.is_directed_acyclic_graph(graph) if graph.is_directed() else True,
                'is_connected': nx.is_weakly_connected(graph) if graph.is_directed() else nx.is_connected(graph),
                'errors': []
            }
            
            print(f"  ✅ Graph generated successfully")
            print(f"  📊 Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")
            print(f"  🔗 Directed: {result['is_directed']}, Acyclic: {result['is_acyclic']}")
            print(f"  🌐 Connected: {result['is_connected']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate graph for {topology_type}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {
                'success': False,
                'graph_type': None,
                'num_nodes': 0,
                'num_edges': 0,
                'is_directed': False,
                'is_acyclic': False,
                'is_connected': False,
                'errors': [error_msg]
            }
    
    def test_network_creation(self, topology_type: str) -> Dict[str, Any]:
        """Test FeedForwardNetwork creation from topology."""
        print(f"🧠 Testing {topology_type} network creation...")
        
        try:
            # Create topology and generate graph
            config = self.test_configs[topology_type].copy()
            
            if topology_type == 'small_world':
                topology = SmallWorldTopology(**config)
            elif topology_type == 'modular':
                topology = ModularTopology(**config)
            elif topology_type == 'hybrid':
                topology = HybridTopology(**config)
            elif topology_type == 'fully_connected':
                topology = FullyConnectedTopology(**config)
            elif topology_type == 'standard_mlp':
                topology = StandardMLPTopology(**config)
            
            # Generate graph with input/output dimensions
            graph = topology.generate(input_dim=self.obs_dim, output_dim=self.action_dim)
            
            # Create FeedForwardNetwork
            network = FeedForwardNetwork(
                graph,
                list(range(self.obs_dim)),
                list(range(self.obs_dim + config['size'], self.obs_dim + config['size'] + self.action_dim)),
                {'learning_rate': 0.001, 'activation': 'relu'}
            )
            
            # Test network properties
            result = {
                'success': True,
                'network_type': type(network).__name__,
                'input_nodes': network.input_nodes,
                'output_nodes': network.output_nodes,
                'num_nodes': network.num_nodes,
                'total_params': sum(p.numel() for p in network.parameters()),
                'trainable_params': sum(p.numel() for p in network.parameters() if p.requires_grad),
                'errors': []
            }
            
            print(f"  ✅ Network created successfully")
            print(f"  📊 Input nodes: {result['input_nodes']}, Output nodes: {result['output_nodes']}")
            print(f"  🔢 Total nodes: {result['num_nodes']}")
            print(f"  🔢 Total Parameters: {result['total_params']}")
            print(f"  🎯 Trainable Parameters: {result['trainable_params']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to create network for {topology_type}: {str(e)}"
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
            return {
                'success': False,
                'network_type': None,
                'input_dim': 0,
                'output_dim': 0,
                'hidden_dim': 0,
                'total_params': 0,
                'trainable_params': 0,
                'errors': [error_msg]
            }
    
    def test_forward_pass(self, topology_type: str) -> Dict[str, Any]:
        """Test forward pass through the network."""
        print(f"➡️  Testing {topology_type} forward pass...")
        
        try:
            # Create topology and network
            config = self.test_configs[topology_type].copy()
            
            if topology_type == 'small_world':
                topology = SmallWorldTopology(**config)
            elif topology_type == 'modular':
                topology = ModularTopology(**config)
            elif topology_type == 'hybrid':
                topology = HybridTopology(**config)
            elif topology_type == 'fully_connected':
                topology = FullyConnectedTopology(**config)
            elif topology_type == 'standard_mlp':
                topology = StandardMLPTopology(**config)
            
            graph = topology.generate(input_dim=self.obs_dim, output_dim=self.action_dim)
            network = FeedForwardNetwork(
                graph,
                list(range(self.obs_dim)),
                list(range(self.obs_dim + config['size'], self.obs_dim + config['size'] + self.action_dim)),
                {'learning_rate': 0.001, 'activation': 'relu'}
            )
            
            # Create test input (dictionary mapping input node indices to values)
            # Use 1D tensors [1] instead of 2D [1,1] to match FFN expectations
            test_input = {node: torch.randn(1) for node in network.input_nodes}
            
            # Forward pass
            with torch.no_grad():
                output = network.forward(test_input)
            
            # Validate output
            result = {
                'success': True,
                'input_shape': {node: val.shape for node, val in test_input.items()},
                'output_shape': {node: val.shape for node, val in output.items()},
                'output_range': (min(val.min().item() for val in output.values()), max(val.max().item() for val in output.values())),
                'output_mean': np.mean([val.mean().item() for val in output.values()]),
                'output_std': np.mean([val.std().item() for val in output.values()]),
                'errors': []
            }
            
            print(f"  ✅ Forward pass successful")
            print(f"  📥 Input nodes: {list(test_input.keys())}")
            print(f"  📤 Output nodes: {list(output.keys())}")
            print(f"  📊 Output range: [{result['output_range'][0]:.3f}, {result['output_range'][1]:.3f}]")
            print(f"  📈 Output mean: {result['output_mean']:.3f}, std: {result['output_std']:.3f}")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed forward pass for {topology_type}: {str(e)}"
            print(f"  ❌ {error_msg}")
            traceback.print_exc()
            return {
                'success': False,
                'input_shape': None,
                'output_shape': None,
                'output_range': (0, 0),
                'output_mean': 0,
                'output_std': 0,
                'errors': [error_msg]
            }
    
    def test_parameter_counting(self, topology_type: str) -> Dict[str, Any]:
        """Test parameter counting - use actual parameters from node_states as ground truth."""
        print(f"🔢 Testing {topology_type} parameter counting...")
        
        try:
            # Create topology and network
            config = self.test_configs[topology_type].copy()
            
            if topology_type == 'small_world':
                topology = SmallWorldTopology(**config)
            elif topology_type == 'modular':
                topology = ModularTopology(**config)
            elif topology_type == 'hybrid':
                topology = HybridTopology(**config)
            elif topology_type == 'fully_connected':
                topology = FullyConnectedTopology(**config)
            elif topology_type == 'standard_mlp':
                topology = StandardMLPTopology(**config)
            
            graph = topology.generate(input_dim=self.obs_dim, output_dim=self.action_dim)
            network = FeedForwardNetwork(
                graph,
                list(range(self.obs_dim)),
                list(range(self.obs_dim + config['size'], self.obs_dim + config['size'] + self.action_dim)),
                {'learning_rate': 0.001, 'activation': 'relu'}
            )
            
            # Count actual parameters from node_states (this is the ground truth)
            actual_params = 0
            if hasattr(network, 'node_states'):
                for node, state in network.node_states.items():
                    if 'bias' in state:
                        actual_params += 1
                    if 'weights' in state:
                        actual_params += len(state['weights'])
            
            result = {
                'success': True,
                'actual_total': actual_params,
                'actual_trainable': actual_params,  # All parameters are trainable
                'network_edges': len(list(network.topology.edges())),
                'network_nodes': len(list(network.topology.nodes())),
                'errors': []
            }
            
            print(f"  ✅ Parameter counting successful")
            print(f"  🔢 Actual Parameters: {actual_params}")
            print(f"  🌐 Network Structure: {result['network_nodes']} nodes, {result['network_edges']} edges")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed parameter counting for {topology_type}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {
                'success': False,
                'actual_total': 0,
                'actual_trainable': 0,
                'network_edges': 0,
                'network_nodes': 0,
                'errors': [error_msg]
            }
    
    # Parameter estimation removed - we use actual parameters from node_states as ground truth

    def run_systematic_tests(self) -> Dict[str, Any]:
        """Run all tests for all topology types."""
        print("🚀 Starting Systematic Topology Testing")
        print("=" * 60)
        
        topology_types = ['small_world', 'modular', 'hybrid', 'fully_connected', 'standard_mlp']
        
        for topology_type in topology_types:
            print(f"\n🎯 Testing {topology_type.upper()} Topology")
            print("-" * 40)
            
            # Run all tests for this topology
            creation_result = self.test_topology_creation(topology_type)
            graph_result = self.test_graph_generation(topology_type)
            network_result = self.test_network_creation(topology_type)
            forward_result = self.test_forward_pass(topology_type)
            param_result = self.test_parameter_counting(topology_type)
            
            # Store results
            self.test_results[topology_type] = {
                'creation': creation_result,
                'graph_generation': graph_result,
                'network_creation': network_result,
                'forward_pass': forward_result,
                'parameter_counting': param_result
            }
            
            # Summary for this topology
            success_count = sum([
                creation_result['success'],
                graph_result['success'],
                network_result['success'],
                forward_result['success'],
                param_result['success']
            ])
            
            print(f"\n📊 {topology_type.upper()} Summary: {success_count}/5 tests passed")
            
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("# 🧪 Systematic Topology Testing Report")
        report.append("")
        report.append(f"**Test Date**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall summary
        total_tests = len(self.test_results) * 5
        total_passed = sum(
            sum(
                result['success'] 
                for result in topology_results.values()
            )
            for topology_results in self.test_results.values()
        )
        
        report.append(f"**Overall Results**: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
        report.append("")
        
        # Detailed results for each topology
        for topology_type, results in self.test_results.items():
            report.append(f"## 🎯 {topology_type.upper()} Topology")
            report.append("")
            
            # Count successes
            success_count = sum(result['success'] for result in results.values())
            report.append(f"**Status**: {success_count}/5 tests passed")
            report.append("")
            
            # Test details
            for test_name, test_result in results.items():
                status = "✅ PASS" if test_result['success'] else "❌ FAIL"
                report.append(f"### {test_name.replace('_', ' ').title()}: {status}")
                
                if test_result['success']:
                    # Show key metrics
                    if 'total_params' in test_result:
                        report.append(f"- Total Parameters: {test_result['total_params']}")
                    if 'parameter_accuracy' in test_result:
                        report.append(f"- Parameter Accuracy: {test_result['parameter_accuracy']:.2f}%")
                    if 'num_nodes' in test_result and 'num_edges' in test_result:
                        report.append(f"- Graph Nodes: {test_result['num_nodes']}, Edges: {test_result['num_edges']}")
                    elif 'num_nodes' in test_result:
                        report.append(f"- Graph Nodes: {test_result['num_nodes']}")
                    if 'input_nodes' in test_result:
                        report.append(f"- Input Nodes: {test_result['input_nodes']}")
                    if 'output_nodes' in test_result:
                        report.append(f"- Output Nodes: {test_result['output_nodes']}")
                else:
                    # Show errors
                    for error in test_result.get('errors', []):
                        report.append(f"- Error: {error}")
                
                report.append("")
        
        # Recommendations
        report.append("## 🎯 Recommendations")
        report.append("")
        
        failed_topologies = [
            topo for topo, results in self.test_results.items()
            if any(not result['success'] for result in results.values())
        ]
        
        if failed_topologies:
            report.append("**Topologies requiring attention**:")
            for topo in failed_topologies:
                report.append(f"- {topo}")
            report.append("")
        else:
            report.append("**All topologies are working correctly!** 🎉")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main testing function."""
    print("🧪 Topology Systematic Testing Suite")
    print("=" * 50)
    
    # Create tester
    tester = TopologySystematicTester()
    
    # Run tests
    results = tester.run_systematic_tests()
    
    # Generate report
    print("\n" + "=" * 60)
    print("📊 GENERATING COMPREHENSIVE REPORT")
    print("=" * 60)
    
    report = tester.generate_report()
    
    # Save report
    with open('topology_systematic_test_report.md', 'w') as f:
        f.write(report)
    
    print("📄 Report saved to: topology_systematic_test_report.md")
    
    # Print summary
    print("\n🎯 TESTING COMPLETE")
    print("=" * 30)
    
    total_tests = len(results) * 5
    total_passed = sum(
        sum(
            result['success'] 
            for result in topology_results.values()
        )
        for topology_results in results.values()
    )
    
    print(f"Total Tests: {total_tests}")
    print(f"Tests Passed: {total_passed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TOPOLOGIES ARE WORKING CORRECTLY!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Check the report for details.")

if __name__ == "__main__":
    main()
