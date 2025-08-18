# Methodology

## Summary

This document tracks the project's approach to various challenges, ensuring consistency, reusability, and coherent variable naming across all components.

## Topology Network Creation

### Network Types
- **Small World**: Watts-Strogatz model with rewiring probability
- **Modular**: Community-based structure with inter/intra-module connections
- **Hybrid**: Combination of Small World and Modular approaches
- **Fully Connected**: Standard feedforward networks

### Implementation
- Custom PyTorch modules for each topology type
- Universal action wrapper for consistent action spaces
- Parameter budget calculator for capacity matching

## Training Customization with SB3 PPO

### Custom Policy Class
- `DebugTopologyPolicy` extends SB3's `ActorCriticPolicy`
- Integrates topology networks as actor and critic components
- Supports universal observation and action spaces

### Clean Slate Callback System
- **SimplifiedCallback**: Minimal essential logging for system configuration
- **CleanTrainingCallback**: Core training metrics only (episode returns, lengths)
- **ContinualLearningProgressBarCallback**: Progress tracking for continual learning
- **ShiftLoggingCallback**: Essential continual learning metrics only
- **TrainingTerminationCallback**: Automatic training termination

### Clean Slate Logging Philosophy
- **Minimal Essential Metrics**: Only log what's absolutely necessary for research
- **Clean Hierarchy**: Clear, logical grouping of metrics
- **Research Ready**: Metrics that directly support paper figures
- **Performance Focused**: Avoid logging noise and redundant data

#### Clean Logging Structure
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

#### Benefits of Clean Slate Approach
- **Reduced Complexity**: Eliminates redundant and noisy metrics
- **Faster Training**: Less logging overhead during training
- **Cleaner W&B Interface**: Easier to navigate and analyze
- **Research Focused**: Metrics directly support publication needs
- **Incremental Enhancement**: Can add specific metrics as needed

## Continual Learning with Observation Shifts

Our continual learning protocol follows the paper-accurate approach with **iteration-based training**:

### **Training Structure**
- **Total Iterations**: 3,000 outer-loop iterations
- **Iterations per Level**: 200 iterations per perturbation level
- **Total Levels**: 15 perturbation levels (including clean baseline)
- **Environment Steps per Iteration**: ~800 steps (2 episodes × 400 max steps)
- **Total Environment Steps**: ~2.4 million steps (3,000 × 800)

### **Perturbation Protocol**
- **Level 0 (Iterations 0-199)**: Clean baseline with **NO NOISE** applied
- **Level 1 (Iterations 200-399)**: First perturbation level
- **Level 2 (Iterations 400-599)**: Second perturbation level
- **...and so on...**

**Key Insight**: Perturbation switches occur every **200 iterations**, not every 200 environment steps. This means each perturbation level lasts approximately **160,000 environment steps** (200 iterations × 800 steps), creating realistic continual learning scenarios.

### **Reward Scaling Strategy**
- **Training**: Rewards are **divided by 20** to create smaller gradients for stable learning
- **Logging**: Raw returns are logged by **multiplying back by 20**, so plots show actual environment performance
- **Net Effect**: Training uses down-scaled rewards, but analysis shows raw performance

### **Pre-Generated Perturbations**
All 15 perturbation vectors are generated at initialization using the run's seed, ensuring reproducibility. Each vector contains per-dimension additive offsets sampled from Uniform[0, 20].

### Training Configuration
```python
CORRECTED_CONTINUAL_LEARNING_CONFIG = {
    'max_iterations': 3000,          # Total iterations (maintained)
    'level_switch': 200,             # Switch perturbation every 200 iterations
    'levels': 15,                    # Total perturbation levels
    'max_episodes_per_iteration': 2, # Episodes per iteration
    'max_timesteps_per_episode': 400, # Max steps per episode
    'total_env_steps': 2400000,      # ~2.4M total environment steps
    'shift_boundaries': [0, 160000, 320000, 480000, 640000, 800000, 
                         960000, 1120000, 1280000, 1440000, 1600000,
                         1760000, 1920000, 2080000, 2240000, 2400000]
}
```

### Key Differences from Previous Implementation
1. **Iteration-Based vs Step-Based**: Outer loop is iterations, not environment steps
2. **Clean Baseline**: First 200 iterations have no perturbation
3. **Proper Reward Scaling**: Division by 20, not multiplication
4. **Correct Shift Timing**: Every 200 iterations ≈ 160K environment steps
5. **Realistic Scale**: 2.4M total environment steps vs 3K steps

## Adaptive Training with Early Stopping

### Task-Specific Training Times
- **CartPole-v1**: 200K timesteps (target), 300K (max)
- **Acrobot-v1**: 800K timesteps (target), 1M (max)
- **MountainCar-v0**: 600K timesteps (target), 800K (max)

### Convergence Monitoring
- Real-time performance tracking every 5K steps
- Automatic termination when tasks converge or timeout
- Resource optimization to prevent overtraining

## Capacity Matching with Incremental Adjustment

### Algorithm
1. Start with base network size
2. Calculate parameter count for current topology
3. Adjust hidden size incrementally
4. Match target parameter count within tolerance
5. Validate network structure

### Implementation
- `ParameterBudgetCalculator` class
- Support for all topology types
- Pre-calculation before wandb initialization

## Experimental Variants and Architectures

### Training Types
- **Baseline**: Single task training with comprehensive evaluation
- **Single-Task**: Train on one task, test on all tasks
- **Double-Task**: Sequential training on two tasks with intermediate testing
- **Triple-Task**: Sequential training on three tasks with intermediate testing
- **Continual Learning**: Single task with observation shifts

### Enhanced Intermediate Testing System
- **Testing Schedule**: After each training phase, test on ALL tasks
- **Temporal Tracking**: Clear phase-based metric naming
- **Transfer Analysis**: Forward and backward transfer measurement

## Logging and Results Management

### Clean Slate Metric Structure
```
training/
├── mean_episode_reward
├── total_episodes
├── episode_return
├── episode_length
└── episode_number

continual_learning/
├── current_segment
├── shift_boundary
└── total_shifts
```

### Benefits of Clean Structure
- **Minimal Overhead**: Only essential metrics logged
- **Clear Organization**: Logical grouping by function
- **Easy Analysis**: Simple to create research plots
- **Scalable**: Easy to add specific metrics as needed

## Training and Sweep Management

### Sweep Types
- **Fixed Network Sizes**: Compare topologies across fixed hidden sizes
- **Fixed Capacities**: Compare topologies across fixed parameter counts
- **Continual Learning**: Single task with observation shifts

### Sweep Configuration
- **Method**: Grid search for systematic comparison
- **Primary Metric**: Clean, essential metrics only
- **Fixed Hyperparameters**: Optimal settings for fair comparison

## Code Organization Principles

### File Structure
- **Main Script**: `topologies_continual_task_training_sweep.py`
- **Configuration**: Clean, minimal configuration
- **Callbacks**: Streamlined callback system
- **Utilities**: Essential utilities only

### Function Organization
- **Unified Training Functions**: Main entry points for training
- **Clean Logging Functions**: Minimal essential metric logging
- **Streamlined Callbacks**: Focused on core functionality
- **Utility Functions**: Capacity matching, etc. 