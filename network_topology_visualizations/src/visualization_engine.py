"""
Visualization Engine for Network Topologies

This module provides three different visualization approaches:
1. Connection Density Visualization
2. Structural Pattern Visualization  
3. Information Flow Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class VisualizationEngine:
    """Core visualization engine for network topologies."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize the visualization engine.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Dots per inch for output
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Color schemes
        self.colors = {
            'nodes': {
                'low_degree': '#87CEEB',    # Sky blue
                'high_degree': '#DC143C',   # Crimson
                'default': '#4682B4'        # Steel blue
            },
            'edges': {
                'thin': '#FFB3B3',          # Light red
                'medium': '#FF6666',        # Medium red
                'thick': '#FF0000',         # Bright red
                'highlight': '#CC0000'      # Dark red
            },
            'modules': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
            'layers': ['#E17055', '#74B9FF', '#00B894', '#FDCB6E', '#E84393'],
            'flow': {
                'input': '#3498DB',         # Blue
                'hidden': '#2ECC71',        # Green  
                'output': '#E74C3C'         # Red
            }
        }
    
    def create_ring_layout(self, graph: nx.Graph, radius: float = 1.0) -> Dict[int, Tuple[float, float]]:
        """
        Create circular ring layout for nodes.
        
        Args:
            graph: NetworkX graph
            radius: Radius of the circle
            
        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        n_nodes = len(graph.nodes())
        if n_nodes == 0:
            return {}
        
        # Create circular positions
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        positions = {}
        
        for i, node in enumerate(sorted(graph.nodes())):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            positions[node] = (x, y)
        
        return positions
    
    def create_layered_layout(self, graph: nx.Graph, topology_name: str) -> Dict[int, Tuple[float, float]]:
        """
        Create layered layout for MLP topologies.
        
        Args:
            graph: NetworkX graph
            topology_name: Name of the topology
            
        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        positions = {}
        nodes = sorted(graph.nodes())
        
        if 'mlp' in topology_name:
            # Determine number of layers based on actual graph structure
            if '3layers' in topology_name:
                num_layers = 5  # Input + 3 Hidden + Output
                # Calculate actual layer sizes based on total nodes
                total_nodes = len(nodes)
                # Estimate layer sizes: input(4) + hidden1(~16) + hidden2(~16) + hidden3(~16) + output(4)
                layer_sizes = [4, (total_nodes - 8) // 3, (total_nodes - 8) // 3, (total_nodes - 8) // 3, 4]
                # Adjust for remainder
                remainder = (total_nodes - 8) % 3
                layer_sizes[1] += remainder // 3
                layer_sizes[2] += remainder // 3 + remainder % 3
            else:
                num_layers = 3  # Input + 1 Hidden + Output
                total_nodes = len(nodes)
                layer_sizes = [4, total_nodes - 8, 4]
            
            # Create horizontal layered layout
            layer_width = 2.0
            layer_spacing = 1.5
            
            node_idx = 0
            for layer in range(num_layers):
                layer_size = layer_sizes[layer]
                
                # Vertical spacing within layer
                if layer_size > 1:
                    y_positions = np.linspace(-1, 1, layer_size)
                else:
                    y_positions = [0]
                
                for i in range(layer_size):
                    if node_idx < len(nodes):
                        node = nodes[node_idx]
                        x = layer * layer_spacing - (num_layers - 1) * layer_spacing / 2
                        y = y_positions[i] if i < len(y_positions) else 0
                        positions[node] = (x, y)
                        node_idx += 1
        else:
            # Fallback to ring layout
            positions = self.create_ring_layout(graph)
        
        return positions
    
    def visualize_connection_density(self, graph: nx.Graph, topology_name: str, 
                                   title: str = None) -> plt.Figure:
        """
        Create connection density visualization.
        
        Args:
            graph: NetworkX graph
            topology_name: Name of the topology
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Choose layout based on topology
        if 'mlp' in topology_name:
            pos = self.create_layered_layout(graph, topology_name)
        else:
            pos = self.create_ring_layout(graph)
        
        # Calculate node degrees for color mapping
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 0
        
        # Create color map for nodes based on degree
        node_colors = []
        for node in sorted(graph.nodes()):
            if max_degree > min_degree:
                normalized_degree = (degrees[node] - min_degree) / (max_degree - min_degree)
            else:
                normalized_degree = 0.5
            
            # Interpolate between low and high degree colors
            color = self._interpolate_color(
                self.colors['nodes']['low_degree'],
                self.colors['nodes']['high_degree'],
                normalized_degree
            )
            node_colors.append(color)
        
        # Draw edges with varying thickness based on importance
        edge_weights = self._calculate_edge_importance(graph)
        
        for edge in graph.edges():
            weight = edge_weights.get(edge, 0.5)
            thickness = max(0.5, weight * 3)
            
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            
            ax.plot(x_coords, y_coords, 
                   color=self.colors['edges']['medium'],
                   linewidth=thickness,
                   alpha=0.6)
        
        # Draw nodes
        for i, node in enumerate(sorted(graph.nodes())):
            x, y = pos[node]
            ax.scatter(x, y, 
                      c=node_colors[i],
                      s=100,
                      edgecolors='black',
                      linewidth=0.5,
                      zorder=3)
            
            # Add node labels for small networks
            if len(graph.nodes()) <= 20:
                ax.annotate(str(node), (x, y), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title is None:
            title = f"{topology_name.replace('_', ' ').title()} - Connection Density"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar for degree
        if max_degree > min_degree:
            sm = plt.cm.ScalarMappable(
                cmap=LinearSegmentedColormap.from_list(
                    'degree_cmap', 
                    [self.colors['nodes']['low_degree'], self.colors['nodes']['high_degree']]
                ),
                norm=plt.Normalize(vmin=min_degree, vmax=max_degree)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
            cbar.set_label('Node Degree', rotation=270, labelpad=15)
        
        plt.tight_layout()
        return fig
    
    def visualize_structural_patterns(self, graph: nx.Graph, topology_name: str,
                                    title: str = None) -> plt.Figure:
        """
        Create structural pattern visualization.
        
        Args:
            graph: NetworkX graph
            topology_name: Name of the topology
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Choose layout based on topology
        if 'mlp' in topology_name:
            pos = self.create_layered_layout(graph, topology_name)
        else:
            pos = self.create_ring_layout(graph)
        
        # Identify structural elements
        structural_info = self._analyze_structural_elements(graph, topology_name)
        
        # Draw edges with structural highlighting
        for edge in graph.edges():
            edge_type = structural_info['edge_types'].get(edge, 'secondary')
            
            if edge_type == 'structural':
                color = self.colors['edges']['highlight']
                linewidth = 2.5
                alpha = 0.8
            else:
                color = self.colors['edges']['thin']
                linewidth = 0.5
                alpha = 0.3
            
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            
            ax.plot(x_coords, y_coords, 
                   color=color, linewidth=linewidth, alpha=alpha)
        
        # Draw nodes with structural grouping
        if 'modules' in structural_info:
            for module_id, module_nodes in structural_info['modules'].items():
                module_color = self.colors['modules'][module_id % len(self.colors['modules'])]
                
                for node in module_nodes:
                    x, y = pos[node]
                    ax.scatter(x, y, c=module_color, s=120, 
                             edgecolors='black', linewidth=1, zorder=3)
        
        elif 'layers' in structural_info:
            for layer_id, layer_nodes in structural_info['layers'].items():
                layer_color = self.colors['layers'][layer_id % len(self.colors['layers'])]
                
                for node in layer_nodes:
                    x, y = pos[node]
                    ax.scatter(x, y, c=layer_color, s=120,
                             edgecolors='black', linewidth=1, zorder=3)
        else:
            # Default node drawing
            for node in graph.nodes():
                x, y = pos[node]
                ax.scatter(x, y, c=self.colors['nodes']['default'], s=120,
                         edgecolors='black', linewidth=1, zorder=3)
        
        # Add structural annotations
        if 'modules' in structural_info:
            self._add_module_annotations(ax, pos, structural_info['modules'])
        elif 'layers' in structural_info:
            self._add_layer_annotations(ax, pos, structural_info['layers'])
        
        # Customize plot
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title is None:
            title = f"{topology_name.replace('_', ' ').title()} - Structural Patterns"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def visualize_information_flow(self, graph: nx.Graph, topology_name: str,
                                 title: str = None) -> plt.Figure:
        """
        Create information flow visualization.
        
        Args:
            graph: NetworkX graph
            topology_name: Name of the topology
            title: Custom title for the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Choose layout based on topology
        if 'mlp' in topology_name:
            pos = self.create_layered_layout(graph, topology_name)
        else:
            pos = self.create_ring_layout(graph)
        
        # Analyze information flow
        flow_info = self._analyze_information_flow(graph, topology_name)
        
        # Draw edges with flow direction and importance
        for edge in graph.edges():
            flow_importance = flow_info['edge_importance'].get(edge, 0.5)
            flow_direction = flow_info['flow_direction'].get(edge, 1)
            
            # Determine edge properties based on flow
            alpha = min(0.9, 0.3 + flow_importance * 0.6)
            linewidth = max(0.5, flow_importance * 3)
            
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            
            # Draw edge
            ax.plot(x_coords, y_coords, 
                   color=self.colors['edges']['medium'],
                   linewidth=linewidth, alpha=alpha)
            
            # Add arrow for directed edges
            if graph.is_directed():
                self._add_arrow(ax, pos[edge[0]], pos[edge[1]], 
                              color=self.colors['edges']['medium'], alpha=alpha)
        
        # Draw nodes with flow-based coloring
        for node in graph.nodes():
            node_flow = flow_info['node_flow_level'].get(node, 0.5)
            
            # Color based on flow level (input=blue, hidden=green, output=red)
            if node_flow < 0.33:
                color = self._interpolate_color(
                    self.colors['flow']['input'],
                    self.colors['flow']['hidden'],
                    node_flow * 3
                )
            else:
                color = self._interpolate_color(
                    self.colors['flow']['hidden'],
                    self.colors['flow']['output'],
                    (node_flow - 0.33) * 1.5
                )
            
            x, y = pos[node]
            ax.scatter(x, y, c=color, s=120,
                     edgecolors='black', linewidth=1, zorder=3)
        
        # Customize plot
        ax.set_aspect('equal')
        ax.axis('off')
        
        if title is None:
            title = f"{topology_name.replace('_', ' ').title()} - Information Flow"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add flow legend
        self._add_flow_legend(ax)
        
        plt.tight_layout()
        return fig
    
    def _interpolate_color(self, color1: str, color2: str, factor: float) -> str:
        """Interpolate between two hex colors."""
        import matplotlib.colors as mcolors
        
        rgb1 = mcolors.hex2color(color1)
        rgb2 = mcolors.hex2color(color2)
        
        interpolated = [
            max(0, min(1, rgb1[i] + factor * (rgb2[i] - rgb1[i]))) 
            for i in range(3)
        ]
        
        return mcolors.rgb2hex(interpolated)
    
    def _calculate_edge_importance(self, graph: nx.Graph) -> Dict[Tuple, float]:
        """Calculate importance of edges based on connectivity."""
        edge_importance = {}
        
        for edge in graph.edges():
            # Simple importance based on node degrees
            degree1 = graph.degree(edge[0])
            degree2 = graph.degree(edge[1])
            importance = (degree1 + degree2) / (2 * len(graph.nodes()))
            edge_importance[edge] = importance
        
        return edge_importance
    
    def _analyze_structural_elements(self, graph: nx.Graph, topology_name: str) -> Dict[str, Any]:
        """Analyze structural elements of the graph."""
        structural_info = {
            'edge_types': {},
            'modules': {},
            'layers': {}
        }
        
        # Identify structural edges
        for edge in graph.edges():
            structural_info['edge_types'][edge] = 'secondary'
        
        if 'modular' in topology_name or 'hybrid' in topology_name:
            # Try to identify modules
            try:
                # Simple module detection based on clustering
                undirected = graph.to_undirected() if graph.is_directed() else graph
                communities = nx.community.greedy_modularity_communities(undirected)
                
                for i, community in enumerate(communities):
                    structural_info['modules'][i] = list(community)
                    
                    # Mark inter-module edges as structural
                    for edge in graph.edges():
                        if (edge[0] in community) != (edge[1] in community):
                            structural_info['edge_types'][edge] = 'structural'
                            
            except:
                pass
        
        elif 'mlp' in topology_name:
            # Identify layers for MLP
            nodes = sorted(graph.nodes())
            if '3layers' in topology_name:
                # Assume 3 hidden layers
                layer_size = len(nodes) // 5  # Rough estimate
                for i in range(5):
                    start_idx = i * layer_size
                    end_idx = start_idx + layer_size if i < 4 else len(nodes)
                    structural_info['layers'][i] = nodes[start_idx:end_idx]
            else:
                # Assume 1 hidden layer
                layer_size = len(nodes) // 3
                for i in range(3):
                    start_idx = i * layer_size
                    end_idx = start_idx + layer_size if i < 2 else len(nodes)
                    structural_info['layers'][i] = nodes[start_idx:end_idx]
        
        return structural_info
    
    def _analyze_information_flow(self, graph: nx.Graph, topology_name: str) -> Dict[str, Any]:
        """Analyze information flow through the graph."""
        flow_info = {
            'edge_importance': {},
            'flow_direction': {},
            'node_flow_level': {}
        }
        
        # Calculate node flow levels (0=input, 0.5=hidden, 1=output)
        nodes = sorted(graph.nodes())
        
        if 'mlp' in topology_name:
            # For MLP, flow is clear: input -> hidden -> output
            if '3layers' in topology_name:
                # 5 layers: input, hidden1, hidden2, hidden3, output
                layer_size = len(nodes) // 5
                for i, node in enumerate(nodes):
                    layer_idx = i // layer_size
                    flow_info['node_flow_level'][node] = layer_idx / 4.0
            else:
                # 3 layers: input, hidden, output
                layer_size = len(nodes) // 3
                for i, node in enumerate(nodes):
                    layer_idx = i // layer_size
                    flow_info['node_flow_level'][node] = layer_idx / 2.0
        else:
            # For other topologies, estimate flow based on position
            for i, node in enumerate(nodes):
                flow_info['node_flow_level'][node] = i / len(nodes)
        
        # Calculate edge importance based on shortest paths
        try:
            # Use betweenness centrality as proxy for flow importance
            betweenness = nx.betweenness_centrality(graph.to_undirected() if graph.is_directed() else graph)
            
            for edge in graph.edges():
                importance = (betweenness.get(edge[0], 0) + betweenness.get(edge[1], 0)) / 2
                flow_info['edge_importance'][edge] = importance
                flow_info['flow_direction'][edge] = 1
                
        except:
            # Fallback: simple importance based on node degrees
            for edge in graph.edges():
                degree1 = graph.degree(edge[0])
                degree2 = graph.degree(edge[1])
                importance = (degree1 + degree2) / (2 * len(graph.nodes()))
                flow_info['edge_importance'][edge] = importance
                flow_info['flow_direction'][edge] = 1
        
        return flow_info
    
    def _add_arrow(self, ax, start_pos: Tuple[float, float], end_pos: Tuple[float, float],
                   color: str, alpha: float = 1.0):
        """Add arrow to indicate direction."""
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Arrow properties
        arrow_length = 0.1
        head_width = 0.05
        
        ax.arrow(start_pos[0] + dx * 0.8, start_pos[1] + dy * 0.8,
                dx * 0.15, dy * 0.15,
                head_width=head_width, head_length=arrow_length,
                fc=color, ec=color, alpha=alpha)
    
    def _add_module_annotations(self, ax, pos: Dict, modules: Dict):
        """Add annotations for modules."""
        for module_id, module_nodes in modules.items():
            if not module_nodes:
                continue
                
            # Calculate module center
            center_x = np.mean([pos[node][0] for node in module_nodes])
            center_y = np.mean([pos[node][1] for node in module_nodes])
            
            # Add module label
            ax.annotate(f'Module {module_id}', 
                       (center_x, center_y),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8))
    
    def _add_layer_annotations(self, ax, pos: Dict, layers: Dict):
        """Add annotations for layers."""
        layer_names = ['Input', 'Hidden1', 'Hidden2', 'Hidden3', 'Output']
        
        for layer_id, layer_nodes in layers.items():
            if not layer_nodes:
                continue
                
            # Calculate layer center
            center_x = np.mean([pos[node][0] for node in layer_nodes])
            center_y = np.mean([pos[node][1] for node in layer_nodes])
            
            # Add layer label
            layer_name = layer_names[layer_id] if layer_id < len(layer_names) else f'Layer {layer_id}'
            ax.annotate(layer_name, 
                       (center_x, center_y),
                       xytext=(0, 20), textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8))
    
    def _add_flow_legend(self, ax):
        """Add legend for information flow."""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor=self.colors['flow']['input'], label='Input Layer'),
            Patch(facecolor=self.colors['flow']['hidden'], label='Hidden Layer'),
            Patch(facecolor=self.colors['flow']['output'], label='Output Layer')
        ]
        
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0, 1), fontsize=9)


if __name__ == "__main__":
    # Test the visualization engine
    print("Testing Visualization Engine...")
    
    # Create a simple test graph
    G = nx.erdos_renyi_graph(16, 0.3)
    
    engine = VisualizationEngine()
    
    # Test all three visualization types
    fig1 = engine.visualize_connection_density(G, "test_topology")
    fig1.savefig("test_density.png", dpi=150, bbox_inches='tight')
    
    fig2 = engine.visualize_structural_patterns(G, "test_topology")
    fig2.savefig("test_structural.png", dpi=150, bbox_inches='tight')
    
    fig3 = engine.visualize_information_flow(G, "test_topology")
    fig3.savefig("test_flow.png", dpi=150, bbox_inches='tight')
    
    print("✓ Test visualizations created successfully!")
