# Weights & Biases Sweeps for Topology Network Hyperparameter Optimization

This guide explains how to use Weights & Biases (wandb) sweeps to optimize hyperparameters for your topology network training scripts.

## Overview

The wandb sweep system allows you to automatically search through different hyperparameter combinations to find the best performing configuration for your topology networks. This is particularly useful for optimizing:

- **PPO Training Parameters**: learning rate, batch size, number of steps, etc.
- **Network Architecture**: hidden size, number of layers, activation functions
- **Topology-Specific Parameters**: small world parameters, modular structure, etc.
- **Training Configuration**: timesteps, evaluation episodes, etc.

## Files Overview

1. **`wandb_sweep_config.py`** - Contains sweep configuration definitions
2. **`topologies--single-task-training-sweep.py`** - Modified training script for sweeps
3. **`launch_sweep.py`** - Script to launch different types of sweeps
4. **`README_wandb_sweeps.md`** - This documentation file

## Setup

### 1. Install Dependencies

Make sure you have wandb installed:

```bash
pip install wandb
```

### 2. Login to Weights & Biases

```bash
wandb login
```

### 3. Prepare Your Training Script

The sweep training script (`topologies--single-task-training-sweep.py`) needs to be modified to work with wandb sweeps. You'll need to:

1. Copy the necessary classes from your original training script:
   - `UniversalActionWrapper`
   - `DebugTopologyPolicy`
   - `EnhancedDebugCallback`
   - `cross_task_testing` function
   - Other utility functions

2. Modify the script to read hyperparameters from `wandb.config`

## Available Sweep Types

### 1. Comprehensive Sweep

Optimizes all hyperparameters simultaneously:

```python
from launch_sweep import launch_comprehensive_sweep

sweep_id = launch_comprehensive_sweep()
```

**Parameters optimized:**
- PPO training parameters (learning_rate, n_steps, batch_size, etc.)
- Network architecture (hidden_size, num_layers)
- Topology-specific parameters (small_world_k, modular_num_modules, etc.)
- Network parameters (activation, dropout)
- Training configuration (total_timesteps, n_eval_episodes)
- Topology type and training task

### 2. Focused Sweeps

Optimize specific areas of hyperparameters:

#### PPO-Focused Sweep
```python
from launch_sweep import launch_focused_sweep

sweep_id = launch_focused_sweep('ppo')
```

**Parameters optimized:**
- learning_rate
- n_steps
- batch_size
- n_epochs
- gamma
- gae_lambda
- clip_range
- ent_coef
- max_grad_norm

#### Architecture-Focused Sweep
```python
sweep_id = launch_focused_sweep('architecture')
```

**Parameters optimized:**
- hidden_size
- num_layers
- activation
- dropout

#### Topology-Focused Sweep
```python
sweep_id = launch_focused_sweep('topology')
```

**Parameters optimized:**
- topology_type
- small_world_k, small_world_p
- modular_num_modules, modular_inter_module_prob, modular_intra_module_prob
- hybrid_num_modules, hybrid_k, hybrid_p, hybrid_inter_module_prob

### 3. Task-Specific Sweeps

Optimized for particular environments:

```python
from launch_sweep import launch_task_specific_sweep

# CartPole-v1 optimization
sweep_id = launch_task_specific_sweep('CartPole-v1')

# Acrobot-v1 optimization
sweep_id = launch_task_specific_sweep('Acrobot-v1')

# MountainCar-v0 optimization
sweep_id = launch_task_specific_sweep('MountainCar-v0')
```

## How to Launch a Sweep

### Method 1: Interactive Launcher

Run the interactive launcher:

```bash
python launch_sweep.py
```

This will present you with options to choose the type of sweep you want to launch.

### Method 2: Programmatic Launch

```python
import wandb
from wandb_sweep_config import create_sweep_config, create_sweep_agent_config

# Create sweep configuration
sweep_config = create_sweep_config()

# Create agent configuration
agent_config = create_sweep_agent_config()
agent_config['count'] = 30  # Number of runs

# Login to wandb
wandb.login()

# Launch sweep
sweep_id = wandb.sweep(sweep_config, **agent_config)
print(f"Sweep ID: {sweep_id}")
```

### Method 3: Custom Configuration

```python
from launch_sweep import launch_custom_sweep

# Define your custom sweep configuration
custom_config = {
    'method': 'bayes',
    'metric': {
        'name': 'testing/mean_reward',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': -5,
            'max': -3,
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'topology_type': {
            'values': ['small_world', 'modular']
        },
        'train_task': {
            'value': 'CartPole-v1'
        }
    }
}

sweep_id = launch_custom_sweep(custom_config, "my_custom_sweep")
```

## Running Sweep Agents

After launching a sweep, you need to run agents to execute the hyperparameter combinations:

### Method 1: Local Agent

```bash
wandb agent your_entity/your_project/sweep_id
```

### Method 2: Multiple Local Agents

```bash
# Run multiple agents in parallel
wandb agent your_entity/your_project/sweep_id &
wandb agent your_entity/your_project/sweep_id &
wandb agent your_entity/your_project/sweep_id &
```

### Method 3: Programmatic Agent

```python
import wandb

# Initialize agent
wandb.agent("your_entity/your_project/sweep_id", function=train_with_sweep, count=10)
```

## Sweep Configuration Details

### Optimization Methods

1. **Bayesian Optimization** (`method: 'bayes'`)
   - Efficient for continuous parameters
   - Learns from previous runs
   - Good for expensive evaluations

2. **Grid Search** (`method: 'grid'`)
   - Exhaustive search over discrete values
   - Good for categorical parameters
   - Can be computationally expensive

3. **Random Search** (`method: 'random'`)
   - Random sampling from parameter space
   - Good baseline method
   - Simple to implement

### Early Termination

The sweeps use Hyperband early termination to stop poorly performing runs early:

```python
'early_terminate': {
    'type': 'hyperband',
    'min_iter': 10
}
```

### Metrics

The sweeps optimize for `testing/mean_reward` (maximization). You can modify this in the sweep configuration.

## Monitoring Sweeps

### 1. WandB Dashboard

Visit your sweep dashboard at:
```
https://wandb.ai/your_entity/your_project/sweeps/sweep_id
```

### 2. Key Metrics to Monitor

- **Best Run**: Shows the best performing hyperparameter combination
- **Parallel Coordinates Plot**: Visualizes parameter relationships
- **Parameter Importance**: Shows which parameters matter most
- **Learning Curves**: Compare training progress across runs

### 3. Sweep Analysis

```python
import wandb

# Get sweep results
api = wandb.Api()
sweep = api.sweep("your_entity/your_project/sweep_id")

# Get best run
best_run = sweep.best_run
print(f"Best reward: {best_run.summary['testing/mean_reward']}")
print(f"Best config: {best_run.config}")
```

## Best Practices

### 1. Start with Focused Sweeps

Begin with focused sweeps (e.g., PPO parameters only) before running comprehensive sweeps. This helps you understand which parameters matter most.

### 2. Use Appropriate Search Spaces

- **Learning rates**: Use log-uniform distribution
- **Discrete parameters**: Use specific values
- **Continuous parameters**: Use uniform distribution

### 3. Set Reasonable Limits

- **Number of runs**: Start with 20-50 runs
- **Early termination**: Use Hyperband to stop poor runs early
- **Resource constraints**: Consider training time and computational resources

### 4. Monitor Resource Usage

- Track GPU/CPU usage
- Monitor memory consumption
- Set appropriate time limits

### 5. Analyze Results

- Look for parameter interactions
- Identify robust configurations
- Consider transfer learning performance

## Example Workflow

1. **Start with PPO optimization**:
   ```bash
   python launch_sweep.py
   # Choose option 2 (PPO-focused)
   ```

2. **Run the agent**:
   ```bash
   wandb agent your_entity/your_project/sweep_id
   ```

3. **Analyze results** and identify best PPO parameters

4. **Run architecture optimization** with fixed PPO parameters:
   ```bash
   python launch_sweep.py
   # Choose option 3 (Architecture-focused)
   ```

5. **Run topology optimization** with fixed PPO and architecture parameters

6. **Validate best configuration** with a final comprehensive sweep

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all required classes are copied to the sweep training script
2. **Configuration Errors**: Check that all parameters in `wandb.config` are properly handled
3. **Resource Issues**: Monitor memory and GPU usage
4. **Early Termination**: Adjust `min_iter` if runs are stopping too early

### Debugging

1. **Check logs**: Look at individual run logs in the WandB dashboard
2. **Test locally**: Run a single configuration locally before launching sweep
3. **Validate configuration**: Ensure sweep parameters match your training script

## Advanced Configuration

### Custom Metrics

You can optimize for different metrics by modifying the sweep configuration:

```python
'metric': {
    'name': 'testing/success_rate',  # or any other metric
    'goal': 'maximize'
}
```

### Conditional Parameters

You can make parameters conditional on other parameters:

```python
'parameters': {
    'topology_type': {
        'values': ['small_world', 'modular']
    },
    'small_world_k': {
        'values': [2, 4, 6, 8],
        'conditions': {
            'topology_type': 'small_world'
        }
    }
}
```

### Resource Constraints

Set resource limits for your sweeps:

```python
'resource': 'gpu',
'resource_args': {
    'gpu_count': 1
}
```

## Conclusion

WandB sweeps provide a powerful way to optimize hyperparameters for your topology networks. Start with focused sweeps to understand parameter importance, then move to comprehensive optimization. Monitor results carefully and use the insights to improve your models.

For more information, visit the [WandB Sweeps documentation](https://docs.wandb.ai/guides/sweeps). 