import unittest
import networkx as nx
import numpy as np
from src.topologies.small_world import SmallWorldTopology
from src.topologies.modular import ModularTopology
from src.topologies.hybrid import HybridTopology
from src.networks.ffn import FeedForwardNetwork
from src.networks.rnn import RecurrentNetwork

class TestTopologyIntegrity(unittest.TestCase):
    def setUp(self):
        """Set up test parameters."""
        self.size = 50
        self.num_layers = 2
        self.seed = 42
        
        # Small world parameters
        self.k = 4
        self.p = 0.1
        
        # Modular parameters
        self.num_modules = 2
        self.inter_module_prob = 0.1
        self.intra_module_prob = 0.3
        
        # Common parameters
        self.inter_layer_prob = 0.1
        
        # Network parameters
        self.network_params = {
            'ffn': {
                'activation': 'relu',
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_layers': [16, 8]
            },
            'rnn': {
                'hidden_size': 16,
                'sequence_length': 5,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        # Input/output nodes
        self.input_nodes = [0, 1, 2]
        self.output_nodes = [48, 49]
    
    def _verify_topology_constraints(self, network, original_topology):
        """Verify that network weights respect topology constraints."""
        # Get all valid edges from original topology
        valid_edges = set(original_topology.edges())
        
        # Check FFN weights
        if isinstance(network, FeedForwardNetwork):
            for node in network.topology.nodes():
                # Check that weights only exist for valid connections
                valid_neighbors = set(original_topology.neighbors(node))
                actual_neighbors = set(network.node_states[node]['weights'].keys())
                self.assertEqual(actual_neighbors, valid_neighbors)
                
                # Check that no new connections were created
                for neighbor in network.node_states[node]['weights']:
                    self.assertIn((node, neighbor), valid_edges)
        
        # Check RNN weights
        elif isinstance(network, RecurrentNetwork):
            for node in network.topology.nodes():
                # Check input weights
                valid_neighbors = set(original_topology.neighbors(node))
                actual_neighbors = set(network.node_states[node]['input_weights'].keys())
                self.assertEqual(actual_neighbors, valid_neighbors)
                
                # Check recurrent weights
                actual_recurrent = set(network.node_states[node]['recurrent_weights'].keys())
                self.assertEqual(actual_recurrent, valid_neighbors)
                
                # Check that no new connections were created
                for neighbor in network.node_states[node]['input_weights']:
                    self.assertIn((node, neighbor), valid_edges)
                for neighbor in network.node_states[node]['recurrent_weights']:
                    self.assertIn((node, neighbor), valid_edges)
    
    def test_small_world_training_integrity(self):
        """Test that small-world topology constraints are maintained during training."""
        # Create topology
        topology = SmallWorldTopology(
            size=self.size,
            k=self.k,
            p=self.p,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test both network types
        for network_type, network_class in [('ffn', FeedForwardNetwork), ('rnn', RecurrentNetwork)]:
            # Create network
            network = network_class(
                topology=layers[0],
                input_nodes=self.input_nodes,
                output_nodes=self.output_nodes,
                network_params=self.network_params[network_type]
            )
            
            # Generate random input
            inputs = {node: np.random.randn() for node in self.input_nodes}
            
            # Run forward pass
            outputs = network.forward(inputs)
            
            # Verify topology constraints are maintained
            self._verify_topology_constraints(network, layers[0])
    
    def test_modular_training_integrity(self):
        """Test that modular topology constraints are maintained during training."""
        # Create topology
        topology = ModularTopology(
            size=self.size,
            num_modules=self.num_modules,
            inter_module_prob=self.inter_module_prob,
            intra_module_prob=self.intra_module_prob,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test both network types
        for network_type, network_class in [('ffn', FeedForwardNetwork), ('rnn', RecurrentNetwork)]:
            # Create network
            network = network_class(
                topology=layers[0],
                input_nodes=self.input_nodes,
                output_nodes=self.output_nodes,
                network_params=self.network_params[network_type]
            )
            
            # Generate random input
            inputs = {node: np.random.randn() for node in self.input_nodes}
            
            # Run forward pass
            outputs = network.forward(inputs)
            
            # Verify topology constraints are maintained
            self._verify_topology_constraints(network, layers[0])
    
    def test_hybrid_training_integrity(self):
        """Test that hybrid topology constraints are maintained during training."""
        # Create topology
        topology = HybridTopology(
            size=self.size,
            num_modules=self.num_modules,
            k=self.k,
            p=self.p,
            inter_module_prob=self.inter_module_prob,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test both network types
        for network_type, network_class in [('ffn', FeedForwardNetwork), ('rnn', RecurrentNetwork)]:
            # Create network
            network = network_class(
                topology=layers[0],
                input_nodes=self.input_nodes,
                output_nodes=self.output_nodes,
                network_params=self.network_params[network_type]
            )
            
            # Generate random input
            inputs = {node: np.random.randn() for node in self.input_nodes}
            
            # Run forward pass
            outputs = network.forward(inputs)
            
            # Verify topology constraints are maintained
            self._verify_topology_constraints(network, layers[0])
    
    def test_small_world_integrity(self):
        """Test small-world topology integrity."""
        # Create topology
        topology = SmallWorldTopology(
            size=self.size,
            k=self.k,
            p=self.p,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test layer integrity
        for i, layer in enumerate(layers):
            # Check number of nodes
            self.assertEqual(len(layer.nodes()), self.size)
            
            # Check average degree is close to k
            avg_degree = np.mean([d for n, d in layer.degree()])
            self.assertAlmostEqual(avg_degree, self.k, delta=1)
            
            # Check clustering coefficient is high (small-world property)
            clustering = nx.average_clustering(layer)
            self.assertGreater(clustering, 0.1)
            
            # Check graph is connected
            self.assertTrue(nx.is_connected(layer))
        
        # Test inter-layer connections
        inter_layer = topology.get_layer_connections(0, 1)
        self.assertIsNotNone(inter_layer)
        self.assertEqual(len(inter_layer.nodes()), self.size)
        
        # Check inter-layer connection probability
        expected_edges = self.size * self.size * self.inter_layer_prob
        actual_edges = len(inter_layer.edges())
        self.assertAlmostEqual(actual_edges / (self.size * self.size), 
                             self.inter_layer_prob, delta=0.1)
    
    def test_modular_integrity(self):
        """Test modular topology integrity."""
        # Create topology
        topology = ModularTopology(
            size=self.size,
            num_modules=self.num_modules,
            inter_module_prob=self.inter_module_prob,
            intra_module_prob=self.intra_module_prob,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test layer integrity
        for i, layer in enumerate(layers):
            # Check number of nodes
            self.assertEqual(len(layer.nodes()), self.size)
            
            # Check module assignments
            module_assignments = topology.get_module_assignments()
            self.assertEqual(len(module_assignments), self.size)
            
            # Check module sizes
            module_sizes = np.bincount(list(module_assignments.values()))
            self.assertTrue(all(size >= self.size // self.num_modules for size in module_sizes))
            
            # Check intra-module connections
            for module in range(self.num_modules):
                module_nodes = [n for n, m in module_assignments.items() if m == module]
                subgraph = layer.subgraph(module_nodes)
                expected_edges = len(module_nodes) * (len(module_nodes) - 1) * self.intra_module_prob / 2
                actual_edges = len(subgraph.edges())
                self.assertAlmostEqual(actual_edges / (len(module_nodes) * (len(module_nodes) - 1) / 2),
                                     self.intra_module_prob, delta=0.1)
            
            # Check inter-module connections
            for m1 in range(self.num_modules):
                for m2 in range(m1 + 1, self.num_modules):
                    m1_nodes = [n for n, m in module_assignments.items() if m == m1]
                    m2_nodes = [n for n, m in module_assignments.items() if m == m2]
                    expected_edges = len(m1_nodes) * len(m2_nodes) * self.inter_module_prob
                    actual_edges = sum(1 for n1 in m1_nodes for n2 in m2_nodes 
                                     if layer.has_edge(n1, n2))
                    self.assertAlmostEqual(actual_edges / (len(m1_nodes) * len(m2_nodes)),
                                         self.inter_module_prob, delta=0.1)
        
        # Test inter-layer connections
        inter_layer = topology.get_layer_connections(0, 1)
        self.assertIsNotNone(inter_layer)
        self.assertEqual(len(inter_layer.nodes()), self.size)
        
        # Check inter-layer connection probability
        expected_edges = self.size * self.size * self.inter_layer_prob
        actual_edges = len(inter_layer.edges())
        self.assertAlmostEqual(actual_edges / (self.size * self.size),
                             self.inter_layer_prob, delta=0.1)
    
    def test_hybrid_integrity(self):
        """Test hybrid topology integrity."""
        # Create topology
        topology = HybridTopology(
            size=self.size,
            num_modules=self.num_modules,
            k=self.k,
            p=self.p,
            inter_module_prob=self.inter_module_prob,
            num_layers=self.num_layers,
            inter_layer_prob=self.inter_layer_prob,
            seed=self.seed
        )
        
        # Generate topology
        layers = topology.generate(self.num_layers)
        
        # Test layer integrity
        for i, layer in enumerate(layers):
            # Check number of nodes
            self.assertEqual(len(layer.nodes()), self.size)
            
            # Check module assignments
            module_assignments = topology.get_module_assignments()
            self.assertEqual(len(module_assignments), self.size)
            
            # Check module sizes
            module_sizes = np.bincount(list(module_assignments.values()))
            self.assertTrue(all(size >= self.size // self.num_modules for size in module_sizes))
            
            # Check small-world properties within modules
            for module in range(self.num_modules):
                module_nodes = [n for n, m in module_assignments.items() if m == module]
                subgraph = layer.subgraph(module_nodes)
                
                # Check average degree is close to k
                avg_degree = np.mean([d for n, d in subgraph.degree()])
                self.assertAlmostEqual(avg_degree, self.k, delta=1)
                
                # Check clustering coefficient is high
                clustering = nx.average_clustering(subgraph)
                self.assertGreater(clustering, 0.1)
            
            # Check inter-module connections
            for m1 in range(self.num_modules):
                for m2 in range(m1 + 1, self.num_modules):
                    m1_nodes = [n for n, m in module_assignments.items() if m == m1]
                    m2_nodes = [n for n, m in module_assignments.items() if m == m2]
                    expected_edges = len(m1_nodes) * len(m2_nodes) * self.inter_module_prob
                    actual_edges = sum(1 for n1 in m1_nodes for n2 in m2_nodes 
                                     if layer.has_edge(n1, n2))
                    self.assertAlmostEqual(actual_edges / (len(m1_nodes) * len(m2_nodes)),
                                         self.inter_module_prob, delta=0.1)
        
        # Test inter-layer connections
        inter_layer = topology.get_layer_connections(0, 1)
        self.assertIsNotNone(inter_layer)
        self.assertEqual(len(inter_layer.nodes()), self.size)
        
        # Check inter-layer connection probability
        expected_edges = self.size * self.size * self.inter_layer_prob
        actual_edges = len(inter_layer.edges())
        self.assertAlmostEqual(actual_edges / (self.size * self.size),
                             self.inter_layer_prob, delta=0.1)

if __name__ == '__main__':
    unittest.main() 