# Topology Playground

A reinforcement learning framework for exploring network topologies in continual learning scenarios with clean, research-ready experimental design.

## 🚀 **Quick Start**

### **Baseline MLP Training (Their Implementation)**
```bash
# Train their exact MLP with continual learning
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5

# No-noise ablation study
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5 --no_noise

# Disable CUDA if GPU memory issues
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5 --no_cuda
```

### **Topology Network Training**
```bash
# Small World topology
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5

# Modular topology
python topologies_continual_task_training_sweep.py --single --topology modular --task CartPole-v1 --seed 42 --num_levels 5

# Hybrid topology
python topologies_continual_task_training_sweep.py --single --topology hybrid --task CartPole-v1 --seed 42 --num_levels 5

# Fully Connected topology
python topologies_continual_task_training_sweep.py --single --topology fully_connected --task CartPole-v1 --seed 42 --num_levels 5

# Standard MLP (our baseline)
python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task CartPole-v1 --seed 42 --num_levels 5 --num_layers 3
```

## 🎯 **What This Framework Does**

### **Core Functionality**
- **Network Topology Research**: Compare different network architectures (Small World, Modular, Hybrid, Fully Connected, Standard MLP)
- **Continual Learning**: Train agents that adapt to changing observation distributions
- **Baseline Comparison**: Compare your topologies against their exact MLP implementation
- **Methodological Soundness**: Use the same activation functions (LeakyReLU) and training protocols across all experiments

### **Research Scenarios**
1. **Continual Learning**: Single task with observation shifts every 200 iterations
2. **No-Noise Ablation**: Train without perturbations to validate system functionality
3. **Multi-Layer Analysis**: Compare different layer depths for MLP baselines
4. **Task Compatibility**: Works with CartPole-v1, Acrobot-v1, and LunarLander-v2

## 📋 **Command Line Arguments**

### **Baseline MLP Script (`baseline_mlp_test.py`)**

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--task` | Environment to train on | CartPole-v1 | `--task Acrobot-v1` |
| `--seed` | Random seed for reproducibility | 42 | `--seed 123` |
| `--num_levels` | Number of perturbation levels | 15 | `--num_levels 5` |
| `--no_noise` | Disable all perturbation noise | False | `--no_noise` |
| `--no_cuda` | Force CPU usage | False | `--no_cuda` |
| `--no_wandb` | Disable W&B logging | False | `--no_wandb` |

### **Topology Training Script (`topologies_continual_task_training_sweep.py`)**

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--single` | Run single training instead of sweep | False | `--single` |
| `--topology` | Network topology type | small_world | `--topology modular` |
| `--task` | Environment to train on | CartPole-v1 | `--task LunarLander-v2` |
| `--seed` | Random seed for reproducibility | 42 | `--seed 456` |
| `--num_levels` | Number of perturbation levels | 15 | `--num_levels 10` |
| `--num_layers` | Number of layers for standard_mlp | 1 | `--num_layers 3` |
| `--no_noise` | Disable all perturbation noise | False | `--no_noise` |
| `--no_cuda` | Force CPU usage | False | `--no_cuda` |
| `--no_wandb` | Disable W&B logging | False | `--no_wandb` |
| `--phase3` | Enable advanced analysis | False | `--phase3` |
| `--test` | Run test experiment with multiple seeds | False | `--test` |

## 🔬 **Available Topologies**

### **Small World Networks**
- **Description**: Networks with high clustering and short path lengths
- **Parameters**: `k` (neighbors), `p` (rewiring probability)
- **Use Case**: Balance between local structure and global connectivity
- **Command**: `--topology small_world`

### **Modular Networks**
- **Description**: Networks with distinct functional modules
- **Parameters**: Number of modules, inter/intra-module connection probabilities
- **Use Case**: Tasks with distinct functional components
- **Command**: `--topology modular`

### **Hybrid Networks**
- **Description**: Combination of small world and modular properties
- **Parameters**: Combines both topology types
- **Use Case**: Complex tasks requiring both local and modular structure
- **Command**: `--topology hybrid`

### **Fully Connected Networks**
- **Description**: Traditional dense neural networks
- **Parameters**: Network size
- **Use Case**: Standard baseline for comparison
- **Command**: `--topology fully_connected`

### **Standard MLP**
- **Description**: Multi-layer perceptron baseline
- **Parameters**: Number of layers (1, 2, 3, 5+)
- **Use Case**: Traditional MLP comparison baseline
- **Command**: `--topology standard_mlp --num_layers 3`

## 🌍 **Available Environments**

| Environment | Observation Dim | Action Dim | Description |
|-------------|----------------|------------|-------------|
| `CartPole-v1` | 4 | 2 | Cart-pole balancing task |
| `Acrobot-v1` | 6 | 3 | Acrobot swing-up task |
| `LunarLander-v2` | 8 | 4 | Lunar lander control task |

## 📊 **Experiment Types**

### **1. Single Training Run**
```bash
# Basic single experiment
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5

# With specific parameters
python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task Acrobot-v1 --seed 123 --num_levels 10 --num_layers 3
```

### **2. No-Noise Ablation Study**
```bash
# Train without perturbations to validate system
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5 --no_noise

# Baseline MLP without noise
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5 --no_noise
```

### **3. Test Experiment Mode**
```bash
# Run multiple seeds and topologies automatically
python topologies_continual_task_training_sweep.py --test --task CartPole-v1 --num_levels 5
```

### **4. CPU-Only Training**
```bash
# Force CPU usage if GPU memory issues
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5 --no_cuda

# Baseline MLP on CPU
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5 --no_cuda
```

## 🔧 **Technical Details**

### **Activation Functions**
- **All networks** use **LeakyReLU(α=0.1)** for consistency
- **Baseline MLP** matches their exact implementation from `main.ipynb`
- **All topologies** use the same activation function for fair comparison

### **Continual Learning Protocol**
- **Iteration-based**: 200 iterations per perturbation level
- **Clean baseline**: First 200 iterations have no noise
- **Observation shifts**: Gaussian noise (mean=0, std=2) applied to observations
- **Reward scaling**: Raw rewards divided by 20 for small gradients

### **Training Parameters**
- **PPO algorithm** with Stable-Baselines3
- **Learning rate**: 0.01 
- **Batch size**: 32
- **Epochs per update**: 5
- **Steps per update**: 800

## 📈 **Output and Logging**

### **W&B Integration**
- **Project**: `topologies--continual-learning-training`
- **Run names**: Include topology type, parameters, and activation function
- **Example**: `SW_L1_S128_P822_CP_seed42_L5_I1000_LS200_N00_LReLU`
  - `SW`: Small World topology
  - `L1`: 1 layer
  - `S128`: 128 hidden nodes
  - `P822`: 822 parameters
  - `CP`: CartPole task
  - `N00`: No noise (ablation study)
  - `LReLU`: LeakyReLU activation

### **Local Data Collection**
- **CSV files**: Saved to `test_experiments/` directory
- **Batch updates**: Every 200 episodes
- **Offline analysis**: Available even without W&B

## 🚨 **Troubleshooting**

### **Common Issues and Solutions**

1. **CUDA Out of Memory**
   ```bash
   # Force CPU usage
   --no_cuda
   ```

2. **Import Errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **W&B Login Issues**
   ```bash
   wandb login
   ```

4. **Slow Training**
   - Use `--no_wandb` for faster execution
   - Reduce `--num_levels` for quick testing
   - Use smaller networks

### **Performance Tips**
- **GPU**: Use CUDA for faster training (default)
- **CPU**: Use `--no_cuda` if GPU memory is insufficient
- **W&B**: Disable with `--no_wandb` for faster execution
- **Testing**: Use `--num_levels 2` for quick validation

## 📁 **Project Structure**

```
topology-playground/
├── baseline_mlp_test.py              # Their exact MLP implementation
├── topologies_continual_task_training_sweep.py  # Main topology training
├── src/                              # Source code
│   ├── topologies/                   # Network topology implementations
│   ├── networks/                     # Network architectures
│   └── utils/                        # Utility functions
├── test_*.py                         # Testing and validation scripts
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🎯 **Research Use Cases**

### **Continual Learning Research**
- Compare network topologies under observation shifts
- Analyze adaptation capabilities of different architectures
- Study catastrophic forgetting in various network structures

### **Architecture Comparison**
- Evaluate efficiency of different topology types
- Compare parameter efficiency across architectures
- Analyze learning dynamics of different network structures

### **Baseline Validation**
- Compare your topologies against their exact MLP implementation
- Ensure methodological consistency across experiments
- Validate experimental protocols

## 📚 **Examples and Tutorials**

### **Quick Comparison Run**
```bash
# Train baseline MLP
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5

# Train small world topology
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5

# Compare results in W&B
```

### **Ablation Study**
```bash
# With noise
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5

# Without noise
python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5 --no_noise
```

### **Multi-Layer Analysis**
```bash
# 1-layer MLP
python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task CartPole-v1 --seed 42 --num_levels 5 --num_layers 1

# 3-layer MLP
python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task CartPole-v1 --seed 42 --num_levels 5 --num_layers 3
```

## 🤝 **Support and Contributing**

- **Issues**: Create GitHub issues for bugs or questions
- **Contributions**: Pull requests welcome for improvements
- **Documentation**: Help improve this README and other docs

## 📄 **License**

MIT License - see LICENSE file for details.