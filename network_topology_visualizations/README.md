# Network Topology Visualization System

This package provides comprehensive visualization tools for comparing different neural network topologies at size 64, demonstrating the fundamental differences between various network architectures used in training.

## 🎯 Overview

The visualization system compares 6 distinct network topologies:

1. **Fully Connected** - Complete DAG where every node connects to every higher-indexed node
2. **Small World** - Ring lattice with some rewired connections (k=4, p=0.2)
3. **Modular** - Nodes grouped into 4 modules with high intra-module and low inter-module connectivity
4. **Hybrid** - Combines small-world within modules and modular structure between modules
5. **Standard MLP (1 layer)** - Traditional feedforward network with 1 hidden layer
6. **Standard MLP (3 layers)** - Multi-layer feedforward network with 3 hidden layers

## 🔍 Visualization Approaches

Each topology is visualized using three different approaches:

### 1. Connection Density Visualization
- **Ring Layout**: Nodes arranged in a circle for consistent comparison
- **Color Coding**: Node color represents degree (number of connections)
- **Edge Thickness**: Connection importance/strength
- **Purpose**: Shows overall connectivity patterns and density distribution

### 2. Structural Pattern Visualization
- **Ring Layout**: Same circular arrangement maintained
- **Highlighting**: Different colors for structural elements (modules, layers)
- **Bold Edges**: Key structural connections emphasized
- **Purpose**: Emphasizes unique structural characteristics of each topology

### 3. Information Flow Visualization
- **Ring Layout**: Circular arrangement with flow representation
- **Directed Arrows**: Information flow direction
- **Gradient Colors**: Input (blue) → Hidden (green) → Output (red)
- **Purpose**: Shows how information propagates through each network type

## 🚀 Quick Start

### Basic Usage

```bash
# Generate all visualizations with default settings (size=64, seed=42)
python demo_visualize_topologies.py

# Customize network size and seed
python demo_visualize_topologies.py --size 128 --seed 123

# Specify output directory
python demo_visualize_topologies.py --output-dir my_results

# Quick mode (only comparison grids, faster)
python demo_visualize_topologies.py --quick
```

### Advanced Usage

```bash
# Generate only specific visualization type
python demo_visualize_topologies.py --type density
python demo_visualize_topologies.py --type structural
python demo_visualize_topologies.py --type flow

# Test individual components
python demo_visualize_topologies.py --test
```

## 📁 Output Structure

The system generates the following outputs:

```
topology_visualization_results/
├── connection_density_comparison.png      # Side-by-side density comparison
├── structural_patterns_comparison.png     # Side-by-side structural comparison
├── information_flow_comparison.png        # Side-by-side flow comparison
├── topology_summary_report.txt            # Quantitative analysis report
├── fully_connected/                       # Individual topology folder
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
├── small_world/
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
├── modular/
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
├── hybrid/
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
├── standard_mlp_1layer/
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
├── standard_mlp_3layers/
│   ├── connection_density.png
│   ├── structural_patterns.png
│   └── information_flow.png
└── *.graphml, *.edgelist, *_metrics.json  # Raw graph data exports
```

## 🔧 Technical Details

### Dependencies

- `networkx` - Graph manipulation and analysis
- `matplotlib` - Visualization and plotting
- `numpy` - Numerical computations
- `torch` - Tensor operations (for topology classes)

### Architecture

The system consists of three main components:

1. **NetworkGenerator** (`src/network_generator.py`)
   - Creates actual network topologies using existing topology classes
   - Ensures consistency with training code
   - Calculates network metrics

2. **VisualizationEngine** (`src/visualization_engine.py`)
   - Implements the three visualization approaches
   - Handles layout algorithms (ring, layered)
   - Manages color schemes and styling

3. **ComparisonInterface** (`src/comparison_interface.py`)
   - Orchestrates the complete comparison workflow
   - Generates side-by-side comparison grids
   - Creates summary reports and exports

### Key Features

- **Consistent Layouts**: Ring-based layout for easy comparison
- **Real Network Data**: Uses actual topology classes from training code
- **Multiple Views**: Three complementary visualization approaches
- **High Quality**: 300 DPI individual images, 150 DPI comparison grids
- **Comprehensive**: Metrics, reports, and raw data exports

## 📊 Expected Results

### Topology Characteristics

| Topology | Key Visual Features | Density | Clustering |
|----------|-------------------|---------|------------|
| Fully Connected | Dense web, uniform connectivity | Very High | Low |
| Small World | Ring structure with shortcuts | Medium | High |
| Modular | Clear module boundaries | Medium | Very High |
| Hybrid | Modular + small-world within | Medium | High |
| MLP (1 layer) | Horizontal bands, feedforward | Medium | Low |
| MLP (3 layers) | Multiple horizontal bands | Medium | Low |

### Visual Distinctions

- **Fully Connected**: Uniformly dense connections, no structural patterns
- **Small World**: Mostly local connections with scattered long-range links
- **Modular**: Color-coded modules with clear separation
- **Hybrid**: Modular organization with internal small-world structure
- **MLP Topologies**: Distinct horizontal layer bands showing feedforward flow

## 🎨 Customization

### Color Schemes

The visualization system uses carefully chosen color schemes:

- **Node Colors**: Degree-based gradient (light blue → dark red)
- **Module Colors**: Distinct color families for easy identification
- **Flow Colors**: Input (blue) → Hidden (green) → Output (red)
- **Edge Colors**: Importance-based thickness and transparency

### Layout Adaptations

- **MLP Topologies**: Override circular layout with layered horizontal bands
- **Modular Topologies**: Group nodes by modules while maintaining ring structure
- **Other Topologies**: Standard circular ring layout

## 🔍 Analysis Guide

### Reading the Visualizations

1. **Connection Density**: Look for overall connectivity patterns
   - High density = many connections
   - Color intensity = node importance
   - Edge thickness = connection strength

2. **Structural Patterns**: Focus on organizational features
   - Color groupings = modules/layers
   - Bold edges = structural connections
   - Layout adaptations = topology-specific organization

3. **Information Flow**: Observe data propagation
   - Arrow direction = information flow
   - Color gradients = layer progression
   - Edge opacity = flow importance

### Metrics Interpretation

- **Density**: Fraction of possible connections present
- **Clustering**: Tendency for nodes to form tightly connected groups
- **Diameter**: Longest shortest path in the network
- **Average Path Length**: Typical distance between nodes

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the correct directory
2. **Memory Issues**: Use smaller network sizes for testing
3. **Missing Dependencies**: Install required packages with pip
4. **Path Issues**: Check that topology classes are accessible

### Testing

```bash
# Test individual components
python demo_visualize_topologies.py --test

# Test with smaller size
python demo_visualize_topologies.py --size 16

# Test single visualization type
python demo_visualize_topologies.py --type density
```

## 📈 Future Enhancements

Potential improvements for the visualization system:

- Interactive web-based visualizations
- Animation showing network evolution
- 3D network layouts
- Custom topology parameter exploration
- Performance benchmarking integration
- Export to additional formats (SVG, PDF)

## 📝 Citation

If you use this visualization system in your research, please cite the topology playground repository and acknowledge the visualization framework.

---

**Note**: This visualization system is designed to work with the existing topology classes in the main repository. Make sure the topology classes are accessible and properly configured before running the visualizations.
