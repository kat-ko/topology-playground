# Network Topology Visualization Tools

This directory contains two tools for visualizing network topologies that would be used in training, without actually training them:

1. **`network_visualization.ipynb`** - Jupyter notebook for interactive visualization
2. **`network_visualization_cli.py`** - Command-line script for batch visualization

## Features

Both tools:
- ✅ **Accept the same command line arguments** as the continual task training script
- ✅ **Create the actual networks** using the same topology creation logic
- ✅ **Generate comprehensive visualizations** showing network structure
- ✅ **Provide topology-specific analysis** and statistics
- ✅ **Support all topology types** and task environments

## Supported Topologies

- **`small_world`**: Small-world network with configurable k and p parameters
- **`modular`**: Modular network with configurable module count and connection probabilities
- **`hybrid`**: Hybrid network combining small-world and modular properties
- **`fully_connected`**: Fully connected network
- **`standard_mlp`**: Standard multi-layer perceptron with configurable layers

## Supported Tasks

- **`CartPole-v1`**: 4-dimensional observation space, 2 discrete actions
- **`Acrobot-v1`**: 6-dimensional observation space, 3 discrete actions
- **`LunarLander-v2`**: 8-dimensional observation space, 4 discrete actions

## Usage

### 1. Jupyter Notebook (Interactive)

```bash
# Start Jupyter
jupyter notebook network_visualization.ipynb

# Then modify the config dictionary in the Configuration cell and run all cells
```

**Advantages:**
- Interactive exploration
- Easy parameter modification
- Inline visualization
- Step-by-step execution

### 2. Command Line Script (Batch/Headless)

```bash
# Basic usage
python network_visualization_cli.py --topology small_world --task CartPole-v1

# With custom parameters
python network_visualization_cli.py \
    --topology modular \
    --task LunarLander-v2 \
    --hidden_size 256 \
    --num_layers 2 \
    --seed 123

# Save visualization to file
python network_visualization_cli.py \
    --topology hybrid \
    --task Acrobot-v1 \
    --output hybrid_network.png
```

**Advantages:**
- Command-line interface
- Batch processing
- File output support
- Headless execution

## Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--topology` | Network topology type | `small_world` | `small_world`, `modular`, `hybrid`, `fully_connected`, `standard_mlp` |
| `--task` | Environment to visualize | `CartPole-v1` | `CartPole-v1`, `Acrobot-v1`, `LunarLander-v2` |
| `--seed` | Random seed | `42` | Any integer |
| `--num_layers` | Number of layers | `1` | Any positive integer |
| `--hidden_size` | Hidden layer size | `128` | Any positive integer |
| `--output` | Output file path | None (display only) | Any valid file path |

## Topology-Specific Parameters

### Small-World Network
- **k**: Number of nearest neighbors (default: 4)
- **p**: Rewiring probability (default: 0.2)

### Modular Network
- **num_modules**: Number of modules (default: 4)
- **inter_module_prob**: Inter-module connection probability (default: 0.1)
- **intra_module_prob**: Intra-module connection probability (default: 0.8)

### Hybrid Network
- **k**: Small-world k parameter (default: 4)
- **p**: Small-world p parameter (default: 0.2)
- **num_modules**: Number of modules (default: 4)
- **inter_module_prob**: Inter-module connection probability (default: 0.1)

### Standard MLP
- **num_layers**: Number of hidden layers (default: 1)
- **activation**: Activation function (default: 'leaky_relu')
- **dropout**: Dropout rate (default: 0.0)

## Output

### Visualization Components

1. **Network Graph**: 
   - Input nodes (light blue)
   - Hidden nodes (light green)
   - Output nodes (light coral)
   - Connections between nodes

2. **Network Statistics**:
   - Total nodes and edges
   - Network density
   - Average degree
   - Clustering coefficient
   - Node distribution

3. **Topology-Specific Analysis**:
   - Small-world properties
   - Modularity scores
   - Layer distributions
   - Connection patterns

### File Output

When using `--output <filename>`, the script saves a high-resolution PNG image (300 DPI) that can be used in papers, presentations, or further analysis.

## Examples

### Example 1: Visualize Small-World Network
```bash
python network_visualization_cli.py \
    --topology small_world \
    --task CartPole-v1 \
    --hidden_size 64
```

### Example 2: Visualize Multi-Layer MLP
```bash
python network_visualization_cli.py \
    --topology standard_mlp \
    --task LunarLander-v2 \
    --num_layers 3 \
    --hidden_size 256
```

### Example 3: Visualize Modular Network and Save
```bash
python network_visualization_cli.py \
    --topology modular \
    --task Acrobot-v1 \
    --hidden_size 128 \
    --output modular_acrobot.png
```

## Integration with Training Script

These visualization tools use the **exact same network creation logic** as the training script (`topologies_continual_task_training_sweep.py`):

- Same topology classes and parameters
- Same task dimension detection
- Same graph generation process
- Same configuration structure

This ensures that what you visualize is exactly what would be trained.

## Requirements

- Python 3.7+
- PyTorch
- NetworkX
- Matplotlib
- Seaborn
- Gymnasium
- NumPy

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and the `src/` folder is accessible
2. **Display Issues**: The CLI script uses non-interactive matplotlib backend to avoid tkinter issues
3. **Memory Issues**: Large networks (hidden_size > 512) may require more memory for visualization

### Getting Help

```bash
# Show help
python network_visualization_cli.py --help

# Test with minimal parameters
python network_visualization_cli.py --topology fully_connected --task CartPole-v1
```

## Contributing

To add new topology types or visualization features:

1. Add the topology class to the `create_topology_network` function
2. Add topology-specific analysis to `analyze_topology_specific_features`
3. Update the argument parser if new parameters are needed
4. Test with various configurations

---

**Note**: These tools are designed for **visualization and analysis only**. They do not perform any training or use computational resources beyond what's needed for network creation and visualization.
