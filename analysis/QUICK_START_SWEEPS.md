# Quick Start Guide: WandB Sweeps for Topology Networks

This guide will get you up and running with wandb sweeps for hyperparameter optimization in under 10 minutes.

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install wandb
```

### 2. Login to WandB
```bash
wandb login
```

### 3. Test with Minimal Example
```bash
python complete_sweep_script.py
# Choose option 2 to create minimal example
```

### 4. Run Minimal Sweep Test
```bash
# Launch a simple sweep
python launch_sweep.py
# Choose option 8 (Custom sweep)

# Run the agent
wandb agent your_entity/your_project/sweep_id
```

## 📋 Step-by-Step Process

### Step 1: Complete Your Sweep Script
```bash
python complete_sweep_script.py
# Choose option 1 to copy classes from original script
```

This will copy the necessary classes from `topologies--single-task-training.py` to `topologies--single-task-training-sweep.py`.

### Step 2: Test Single Run
```bash
python topologies--single-task-training-sweep.py
```

Make sure the script runs without errors before launching sweeps.

### Step 3: Launch Your First Sweep
```bash
python launch_sweep.py
```

Choose from:
- **Option 2**: PPO-focused (recommended to start)
- **Option 3**: Architecture-focused
- **Option 4**: Topology-focused
- **Option 5-7**: Task-specific sweeps

### Step 4: Run Sweep Agents
```bash
# Copy the sweep ID from the output and run:
wandb agent your_entity/your_project/sweep_id
```

### Step 5: Monitor Results
Visit the sweep dashboard at:
```
https://wandb.ai/your_entity/your_project/sweeps/sweep_id
```

## 🎯 Recommended Workflow

### Phase 1: PPO Optimization (Start Here)
```bash
python launch_sweep.py
# Choose option 2 (PPO-focused)
```

This optimizes:
- Learning rate
- Batch size
- Number of steps
- Number of epochs
- Gamma, GAE lambda, etc.

### Phase 2: Architecture Optimization
```bash
python launch_sweep.py
# Choose option 3 (Architecture-focused)
```

This optimizes:
- Hidden size
- Number of layers
- Activation functions
- Dropout

### Phase 3: Topology Optimization
```bash
python launch_sweep.py
# Choose option 4 (Topology-focused)
```

This optimizes:
- Topology type
- Small world parameters (k, p)
- Modular parameters (num_modules, probabilities)
- Hybrid parameters

### Phase 4: Task-Specific Optimization
```bash
python launch_sweep.py
# Choose option 5-7 for specific tasks
```

## 🔧 Configuration Examples

### Quick PPO Sweep
```python
from launch_sweep import launch_focused_sweep

sweep_id = launch_focused_sweep('ppo')
```

### Quick Architecture Sweep
```python
from launch_sweep import launch_focused_sweep

sweep_id = launch_focused_sweep('architecture')
```

### Custom Sweep
```python
from launch_sweep import launch_custom_sweep

custom_config = {
    'method': 'bayes',
    'metric': {'name': 'testing/mean_reward', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'distribution': 'log_uniform', 'min': -5, 'max': -3},
        'hidden_size': {'values': [64, 128, 256]},
        'topology_type': {'values': ['small_world', 'modular']},
        'train_task': {'value': 'CartPole-v1'}
    }
}

sweep_id = launch_custom_sweep(custom_config, "my_sweep")
```

## 📊 Monitoring Your Sweeps

### Key Metrics to Watch
- **Best Run**: Shows the best performing configuration
- **Parallel Coordinates**: Visualizes parameter relationships
- **Parameter Importance**: Shows which parameters matter most

### Example Analysis
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

## 🚨 Common Issues & Solutions

### Issue: Import Errors
**Solution**: Make sure all classes are copied from the original script
```bash
python complete_sweep_script.py
```

### Issue: Configuration Errors
**Solution**: Check that all parameters in `wandb.config` are handled
```python
# In your sweep script, add defaults:
learning_rate = wandb.config.get('learning_rate', 3e-4)
```

### Issue: Runs Stopping Early
**Solution**: Adjust early termination settings
```python
'early_terminate': {
    'type': 'hyperband',
    'min_iter': 20  # Increase from 10
}
```

### Issue: Resource Problems
**Solution**: Monitor and limit resources
```python
'resource': 'gpu',
'resource_args': {
    'gpu_count': 1
}
```

## 🎯 Best Practices

1. **Start Small**: Begin with focused sweeps (20-30 runs)
2. **Test First**: Always test a single run before launching sweeps
3. **Monitor Resources**: Watch GPU/CPU usage
4. **Use Early Termination**: Let Hyperband stop poor runs early
5. **Analyze Results**: Look for parameter interactions and patterns

## 📈 Expected Results

### PPO Sweep (20-30 runs)
- Should find learning rate in 1e-4 to 1e-3 range
- Batch size typically 64-256
- Number of steps 1024-4096

### Architecture Sweep (20-30 runs)
- Hidden size typically 128-512
- Number of layers 1-3
- ReLU usually performs best

### Topology Sweep (20-30 runs)
- Small world: k=4-8, p=0.1-0.3
- Modular: 4-8 modules
- Hybrid: combination of above

## 🔗 Useful Links

- [WandB Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Sweep Dashboard](https://wandb.ai/your_entity/your_project/sweeps)
- [Parameter Importance Guide](https://docs.wandb.ai/guides/sweeps/analyze-results)

## 🎉 Success Checklist

- [ ] WandB installed and logged in
- [ ] Sweep script completed and tested
- [ ] First sweep launched successfully
- [ ] Agents running without errors
- [ ] Results appearing in dashboard
- [ ] Best configuration identified
- [ ] Results validated with final run

## 💡 Pro Tips

1. **Parallel Agents**: Run multiple agents to speed up sweeps
2. **Resource Monitoring**: Use `nvidia-smi` to monitor GPU usage
3. **Result Analysis**: Export results to CSV for further analysis
4. **Configuration Reuse**: Save best configurations for future experiments
5. **Incremental Optimization**: Use results from one sweep to inform the next

Happy optimizing! 🚀 