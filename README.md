# Topology Playground

A reinforcement learning framework for exploring network topologies in multi-task learning scenarios with clean, research-ready experimental design.

## 🎯 **Current Status: PRODUCTION READY** ✅

**Latest Updates (August 2024)**:
- ✅ **Episode Counter Bug Fixed**: All topology levels now complete properly
- ✅ **Level Counting Consistency**: Run names and completion messages now match
- ✅ **W&B Integration**: Fully functional with accurate parameter counting
- ✅ **Multi-Layer MLP Support**: Standard MLP baseline with configurable layers
- ✅ **Task Compatibility**: Works with CartPole, Acrobot, and LunarLander
- ✅ **Parameter Accuracy**: Actual parameter counts, not estimates
- ✅ **Indentation Errors**: All syntax issues resolved

## Overview

This project implements and evaluates different network topologies (fully connected, small world, modular, hybrid, and standard MLP) for reinforcement learning agents trained on multiple tasks. The framework supports single-task, double-task, triple-task, and continual learning training scenarios with streamlined analysis and visualization capabilities.

## Features

- **Multiple Network Topologies**: Fully connected, small world, modular, hybrid, and **standard MLP baseline**
- **Multi-Task Learning**: Support for single, double, and triple task training scenarios
- **Continual Learning**:single task training with observation shifts for adaptation analysis
- **Accurate Parameter Counting**: **Real parameter counts** from network states, not estimates
- **Multi-Layer Support**: Standard MLP supports 1, 2, 3, 5+ layers with proper scaling
- **Capacity Matching**: Automatic parameter budget matching across different topologies
- **Clean Slate Logging**: Minimal, research-ready W&B logging for publication-quality plots
- **Experiment Tracking**: Streamlined integration with Weights & Biases
- **Task-Specific Dimensions**: Dynamic input/output handling for different environments

## Continual Learning Protocol (Paper-Accurate)

### **Experimental Setup**
- **Total Iterations**: 3,000 iterations (maintained from current setup)
- **Iterations per Shift**: 200 iterations per perturbation level
- **Total Perturbation Levels**: 15 levels (including clean baseline)
- **Environment Steps per Iteration**: ~800 steps (2 episodes × 400 steps max)
- **Total Environment Steps**: ~2.4M steps (3000 × 800)

### **Key Features**
- **Clean Baseline**: First 200 iterations (0-199) have **NO NOISE** for proper learning
- **Iteration-Based Switching**: Perturbation changes every 200 iterations, not every 200 env-steps
- **Proper Reward Scaling**: Raw rewards **divided by 20** (creates small gradients for slow adaptation)
- **Realistic Scale**: 2.4M total environment steps vs previous 3K steps

### **Perturbation Schedule**
- **Level 0 (Iterations 0-199)**: Clean baseline learning
- **Level 1 (Iterations 200-399)**: First perturbation applied
- **Level 2 (Iterations 400-599)**: Second perturbation applied
- **...and so on...**
- **Level 14 (Iterations 2800-2999)**: Final perturbation level

### **Research Value**
- **Paper Accuracy**: Matches the reference implementation exactly
- **Realistic Learning**: Slow adaptation due to small gradients (reward/20)
- **Proper Continual Learning**: Clean baseline followed by gradual perturbation introduction
- **Publication Ready**: Correct experimental protocol for continual learning research

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd topology-playground
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:
   ```bash
   pip install -e .
   ```

### Alternative Installation Methods

**Using conda**:
```bash
conda create -n topology-playground python=3.9
conda activate topology-playground
pip install -r requirements.txt
```

**Using pip with specific versions**:
```bash
 pip install -r requirements.txt --no-cache-dir
```

## Usage

### Basic Training

**Continual Learning Training** (Recommended):
```bash
# Basic continual learning with 5 levels
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5

# Multi-layer MLP baseline
python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task CartPole-v1 --seed 42 --num_levels 5 --num_layers 3

# Full experiment with 15 levels
python topologies_continual_task_training_sweep.py --single --topology modular --task Acrobot-v1 --seed 42
```

**Available Topologies**:
- `small_world`: Small world networks with configurable k and p parameters
- `modular`: Modular networks with distinct functional modules
- `hybrid`: Combination of small world and modular properties
- `fully_connected`: Traditional dense neural networks
- `standard_mlp`: Standard MLP baseline with configurable layers (1, 2, 3, 5+)

**Available Tasks**:
- `CartPole-v1`: Cart-pole balancing (4D observation, 2D action)
- `Acrobot-v1`: Acrobot swing-up (6D observation, 3D action)
- `LunarLander-v2`: Lunar lander (8D observation, 4D action)

**Training Parameters**:
- `--num_levels`: Number of perturbation levels (default: 15)
- `--num_layers`: Number of layers for standard_mlp (default: 1)
- `--seed`: Random seed for reproducibility
- `--no_wandb`: Disable W&B logging if needed

### Advanced Training

**Test Experiment Mode** (Multiple seeds and topologies):
```bash
python topologies_continual_task_training_sweep.py --test --task CartPole-v1 --num_levels 5
```

**W&B Sweep Integration**:
```bash
# Launch hyperparameter sweep
python wandb_sweep_config.py
```

## Recent Fixes and Improvements

### **Episode Counter Bug (August 2024)** ✅
**Problem**: Training was stopping after 4 levels instead of completing all 5 levels due to episode counter accumulation across iterations.

**Solution**: Added episode counter reset in `ContinualLearningWrapper.set_iteration()` method.

**Result**: All topology levels now complete properly, ensuring full training coverage.

### **Level Counting Consistency** ✅
**Problem**: Discrepancy between run names (showing L5) and completion messages (showing 6 levels).

**Solution**: Fixed level calculation in training completion message to match run naming logic.

**Result**: Consistent level counting throughout the system.

### **Parameter Counting Accuracy** ✅
**Problem**: System was estimating parameters instead of calculating actual counts from network states.

**Solution**: Implemented accurate parameter counting from `FeedForwardNetwork.node_states`.

**Result**: Run names now show real parameter counts (e.g., P1804 instead of P256).

### **Multi-Layer MLP Support** ✅
**Problem**: Only fully connected topology supported multiple layers, limiting baseline comparisons.

**Solution**: Introduced `StandardMLPTopology` class with configurable layers (1, 2, 3, 5+).

**Result**: Proper MLP baseline for comparing against graph-based topologies.

### **Task-Specific Dimension Handling** ✅
**Problem**: Universal action/observation spaces limited flexibility across different tasks.

**Solution**: Dynamic input/output dimension handling based on actual task requirements.

**Result**: Each topology adapts to task-specific dimensions (CartPole: 4D→2D, Acrobot: 6D→3D, LunarLander: 8D→4D).

## Analysis Scripts

**Systematic Topology Testing**:
```bash
python test_topology_systematic.py
```

**Task Dimension Compatibility**:
```bash
python test_task_dimensions.py
```

**MLP Multi-Layer Validation**:
```bash
python test_mlp_multilayer.py
```

**Analyze Single Task Results**:
```bash
python analyze_transfer_results_single_task.py
```

**Analyze Double Task Results**:
```bash
python analyze_transfer_results_double_task.py
```

**Analyze Topology Depth**:
```bash
python analyze_topology_depth.py
```

## Configuration

The training scripts use centralized configuration through the `create_debug_config()` function. Key parameters include:

- **Tasks**: CartPole-v1, Acrobot-v1, LunarLander-v2
- **Topologies**: fully_connected, small_world, modular, hybrid, standard_mlp
- **Training Parameters**: Learning rate, batch size, timesteps, etc.
- **Capacity Matching**: Automatic parameter budget matching
- **Multi-Layer Support**: Configurable layers for standard_mlp topology

## Clean Slate Experimental Design

### Philosophy
Our new clean slate approach focuses on **minimal essential metrics** for research-quality results:

- **Reduced Complexity**: Eliminates redundant and noisy metrics
- **Faster Training**: Less logging overhead during training
- **Cleaner W&B Interface**: Easier to navigate and analyze
- **Research Focused**: Metrics directly support publication needs
- **Incremental Enhancement**: Can add specific metrics as needed

### Clean Logging Structure
```
config/                    # System configuration (one-time)
├── topology_type
├── hidden_size
├── num_layers
├── total_parameters
├── task_name
└── seed

training/                  # Core training metrics
├── timestep
├── episode_return
├── episode_length
├── mean_episode_reward
└── total_episodes

continual_learning/        # Essential continual learning only
├── current_segment
├── shift_boundary
└── total_shifts
```

### Continual Learning Setup
- **Single Task Training**: Focus on one task with observation shifts
- **Segment Length**: 200 iterations per segment (configurable)
- **Shift Range**: [0, 2] observation space modifications
- **Total Lifetime**: 3,000 iterations for comprehensive experimentation
- **Adaptation Analysis**: Measure how well networks adapt to changes

## Project Structure

```
topology-playground/
├── src/                          # Source code
│   ├── topologies/              # Network topology implementations
│   │   ├── small_world.py      # Small world networks
│   │   ├── modular.py          # Modular networks
│   │   ├── hybrid.py           # Hybrid networks
│   │   ├── fully_connected.py  # Fully connected networks
│   │   └── standard_mlp.py     # Standard MLP baseline
│   ├── networks/                # Network architectures
│   │   ├── ffn.py              # FeedForwardNetwork implementation
│   │   └── base.py             # Base network classes
│   ├── utils/                   # Utility functions
│   ├── analysis/                # Analysis tools
│   └── ...
├── test_*.py                    # Testing and validation scripts
├── wandb_sweep_config.py        # W&B sweep configurations
├── results/                     # Experiment results
├── logs/                        # Training logs
├── figures/                     # Generated visualizations
├── config/                      # Configuration files
├── tests/                       # Test files
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
└── README.md                    # This file
```

## Key Components

### Network Topologies

- **Small World**: Networks with high clustering and short path lengths
- **Modular**: Networks with distinct functional modules
- **Hybrid**: Combination of small world and modular properties
- **Fully Connected**: Traditional dense neural networks
- **Standard MLP**: Multi-layer perceptron baseline with configurable layers

### Training Scripts

- **Continual Learning**: Single task with observation shifts (RECOMMENDED)
- **Single Task**: Train on one task, evaluate on all tasks
- **Double Task**: Train on two tasks sequentially, evaluate on all tasks
- **Triple Task**: Train on all three tasks sequentially

### Analysis Tools

- **Transfer Learning Analysis**: Measure knowledge transfer between tasks
- **Network Visualization**: Generate network structure plots
- **Performance Metrics**: Comprehensive evaluation metrics
- **Capacity Analysis**: Parameter budget and efficiency analysis
- **Adaptation Analysis**: Continual learning performance metrics

## Environment Variables

Set these environment variables for optimal performance:

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export WANDB_API_KEY=your_key   # Weights & Biases API key
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Add project to Python path
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Memory Issues**:
   ```bash
   # Force CPU usage if GPU memory is insufficient
   CUDA_VISIBLE_DEVICES="" python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5
   ```

2. **Import Errors**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Memory Issues**:
   - Reduce batch size in configuration
   - Use smaller network sizes
   - Use fewer layers for standard_mlp
   - Enable gradient checkpointing

4. **WandB Issues**:
   ```bash
   wandb login  # Login to Weights & Biases
   ```

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch sizes based on available memory
- Use appropriate network sizes for your hardware
- Use fewer layers if memory is constrained
- Enable mixed precision training for faster execution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{topology_playground,
  title={Topology Playground: A Reinforcement Learning Framework for Network Topologies},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/topology-playground}
}
```

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the example scripts in the root directory
- See `METHODOLOGY.md` for detailed experimental approach

## Acknowledgments

- Stable Baselines3 for the RL framework
- NetworkX for graph analysis
- Weights & Biases for experiment tracking
- The reinforcement learning community for inspiration and tools