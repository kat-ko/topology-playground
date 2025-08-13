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

### Experimental Setup
- **Single Task Training**: Focus on one task with observation shifts
- **Segment Length**: 200 steps per segment (configurable)
- **Shift Range**: [0, 2] observation space modifications
- **Total Lifetime**: 3,000 steps for rapid experimentation

### Continual Learning Wrapper
- **Observation Shifts**: Random modifications to input space every segment
- **Segment Tracking**: Clear boundaries between learning phases
- **Adaptation Analysis**: Measure how well networks adapt to changes

### Training Configuration
```python
CONTINUAL_LEARNING_CONFIG = {
    'segment_length': 200,           # Steps per segment
    'shift_range': [0, 2],          # Observation modification range
    'total_lifetime_steps': 3000,    # Total training budget
    'log_frequency': 100,            # Log every 100 steps
    'episode_simulation': 500        # Simulate episodes every 500 steps
}
```

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