# Baseline MLP Test Script

This script implements **their exact MLP architecture** from `main.ipynb` and integrates it with our continual learning W&B system for fair comparison with our topology networks.

## 🎯 Purpose

- **Compare their exact MLP implementation** with our topology networks
- **Use the same continual learning setup** (distribution shifts, iteration-based training)
- **Log results to the same W&B project** for direct comparison
- **Focus on Adam optimizer** (no TRAC for now)

## 🏗️ Architecture

Their exact MLP implementation:
- **PolicyNetwork**: 3 hidden layers of 128 nodes each, LeakyReLU(0.1)
- **ValueNetwork**: 3 hidden layers of 128 nodes each, LeakyReLU(0.1)
- **Total Parameters**: 67,715 (33,922 + 33,793)
- **Optimizer**: Adam (lr=0.01)

## 🚀 Usage

### Basic Usage
```bash
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5
```

### All Options
```bash
python baseline_mlp_test.py \
    --task CartPole-v1 \           # Environment: CartPole-v1, Acrobot-v1, LunarLander-v2
    --seed 42 \                    # Random seed for reproducibility
    --num_levels 15 \              # Number of distribution shift levels
    --no_wandb                     # Disable W&B logging (optional)
```

### Examples

**CartPole with 5 levels:**
```bash
python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5
```

**Acrobot with 10 levels:**
```bash
python baseline_mlp_test.py --task Acrobot-v1 --seed 123 --num_levels 10
```

**LunarLander with 15 levels (no W&B):**
```bash
python baseline_mlp_test.py --task LunarLander-v2 --seed 456 --num_levels 15 --no_wandb
```

## 📊 Output

### Console Output
- Training progress with iteration-level rewards
- Level changes every 200 iterations
- Final parameter counts and results

### W&B Integration
- **Project**: `topologies--continual-learning-training`
- **Run Name**: `BASELINE_MLP_L3_S128_P33922_CA_seed42_L5_I1000_LS200_N02`
- **Metrics**: Episode rewards, lengths, levels, perturbation levels
- **Plots**: Iteration vs. mean episode rewards (same as our topologies)

## 🔄 Continual Learning Setup

- **Level 0** (iterations 0-199): NO NOISE - Clean baseline learning
- **Level 1+** (iterations 200+): Random perturbations in [0, 2] range
- **Level Switch**: Every 200 iterations
- **Reward Scaling**: Division by 20 (creates small gradients)
- **Episode Cap**: 400 steps maximum
- **Episodes per Iteration**: 2

## 📈 Comparison with Our Topologies

Now you can directly compare:

1. **Their MLP Baseline**: `python baseline_mlp_test.py --task CartPole-v1 --seed 42 --num_levels 5`
2. **Our Standard MLP**: `python topologies_continual_task_training_sweep.py --single --topology standard_mlp --task CartPole-v1 --seed 42 --num_levels 5`
3. **Our Topologies**: `python topologies_continual_task_training_sweep.py --single --topology small_world --task CartPole-v1 --seed 42 --num_levels 5`

All results will be logged to the same W&B project with consistent naming conventions and metrics.

## 🎯 Key Differences from Our System

| Aspect | Their MLP (This Script) | Our Topologies |
|--------|-------------------------|----------------|
| **Architecture** | Direct PyTorch (3×128) | Graph-based (networkx) |
| **Implementation** | Their exact code | Our custom system |
| **Training** | Simple iteration loop | PPO with Stable-Baselines3 |
| **W&B Integration** | Same project, same metrics | Same project, same metrics |
| **Comparability** | ✅ Direct comparison | ✅ Direct comparison |

## 🔧 Technical Details

- **Device**: Forces CPU usage to avoid CUDA memory issues
- **Parameter Verification**: Includes assertions to ensure exact parameter counts
- **Episode Tracking**: Custom step counter for episode length capping
- **Plot Generation**: Same plotting function as our topology system
- **W&B Logging**: Consistent with our existing logging patterns

## 📝 Notes

- This script is **completely separate** from our topology system
- It reuses our `ContinualLearningWrapper` for consistency
- Results are logged to the same W&B project for fair comparison
- Parameter counts are verified to match their exact implementation
- The script is designed for **baseline comparison**, not production use
