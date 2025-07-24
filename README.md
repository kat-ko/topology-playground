# Topology Playground

A reinforcement learning framework for exploring network topologies in multi-task learning scenarios.

## Overview

This project implements and evaluates different network topologies (fully connected, small world, modular, and hybrid) for reinforcement learning agents trained on multiple tasks. The framework supports single-task, double-task, and triple-task training scenarios with comprehensive analysis and visualization capabilities.

## Features

- **Multiple Network Topologies**: Fully connected, small world, modular, and hybrid networks
- **Multi-Task Learning**: Support for single, double, and triple task training scenarios
- **Capacity Matching**: Automatic parameter budget matching across different topologies
- **Comprehensive Analysis**: Detailed performance metrics, transfer learning analysis, and network visualization
- **Experiment Tracking**: Integration with Weights & Biases for experiment logging
- **Universal Action Space**: Unified action and observation spaces across different environments

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

**Single Task Training**:
```bash
python topologies--single-task-training.py
```

**Double Task Training**:
```bash
python topologies--double-task-training.py
```

**Triple Task Training**:
```bash
python topologies--triple-task-training.py
```

### Analysis Scripts

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

### Configuration

The training scripts use centralized configuration through the `create_debug_config()` function. Key parameters include:

- **Tasks**: CartPole-v1, MountainCar-v0, Acrobot-v1
- **Topologies**: fully_connected, small_world, modular, hybrid
- **Training Parameters**: Learning rate, batch size, timesteps, etc.
- **Capacity Matching**: Automatic parameter budget matching

## Project Structure

```
topology-playground/
├── src/                          # Source code
│   ├── topologies/              # Network topology implementations
│   ├── networks/                # Network architectures
│   ├── utils/                   # Utility functions
│   ├── analysis/                # Analysis tools
│   └── ...
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

- **Fully Connected**: Traditional dense neural networks
- **Small World**: Networks with high clustering and short path lengths
- **Modular**: Networks with distinct functional modules
- **Hybrid**: Combination of small world and modular properties

### Training Scripts

- **Single Task**: Train on one task, evaluate on all tasks
- **Double Task**: Train on two tasks sequentially, evaluate on all tasks
- **Triple Task**: Train on all three tasks sequentially

### Analysis Tools

- **Transfer Learning Analysis**: Measure knowledge transfer between tasks
- **Network Visualization**: Generate network structure plots
- **Performance Metrics**: Comprehensive evaluation metrics
- **Capacity Analysis**: Parameter budget and efficiency analysis

## Environment Variables

Set these environment variables for optimal performance:

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export WANDB_API_KEY=your_key   # Weights & Biases API key
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Add project to Python path
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   ```bash
   python gpu_test.ipynb  # Test GPU availability
   ```

2. **Import Errors**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Memory Issues**:
   - Reduce batch size in configuration
   - Use smaller network sizes
   - Enable gradient checkpointing

4. **WandB Issues**:
   ```bash
   wandb login  # Login to Weights & Biases
   ```

### Performance Optimization

- Use GPU acceleration when available
- Adjust batch sizes based on available memory
- Use appropriate network sizes for your hardware
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

## Acknowledgments

- Stable Baselines3 for the RL framework
- NetworkX for graph analysis
- Weights & Biases for experiment tracking
- The reinforcement learning community for inspiration and tools
