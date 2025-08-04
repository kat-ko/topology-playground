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

### Enhanced Callback System
- `EnhancedDebugCallback` for comprehensive logging
- Phase tracking for sequential training
- Graph metrics and network analysis

### Adaptive Training with Early Stopping
- **Task-Specific Training Times**: Individual timesteps per task based on complexity
- **Convergence Monitoring**: Real-time performance tracking every 5K steps
- **Early Stopping**: Automatic termination when tasks converge or timeout
- **Resource Optimization**: Prevents overtraining and resource waste

#### Early Stopping Triggers
1. **Convergence Detection**: Task reaches target performance threshold
2. **Performance Plateau**: No improvement for specified patience period
3. **Maximum Timeout**: Reaches task-specific maximum timesteps
4. **Target Achievement**: Reaches task-specific target timesteps

#### Task-Specific Configuration
```python
TASK_TRAINING_CONFIG = {
    'CartPole-v1': {
        'total_timesteps': 200000,      # Target convergence time
        'max_timesteps': 300000,        # Maximum allowed time
        'min_timesteps': 50000,         # Minimum training time
        'early_stopping_patience': 10000, # Steps without improvement
        'convergence_window': 1000,     # Window for convergence check
        'reward_threshold': 450,        # Performance threshold
    },
    'Acrobot-v1': {
        'total_timesteps': 800000,      # Slower convergence
        'max_timesteps': 1000000,       # Higher complexity
        'min_timesteps': 200000,
        'early_stopping_patience': 20000,
        'convergence_window': 2000,
        'reward_threshold': -100,
    },
    'MountainCar-v0': {
        'total_timesteps': 600000,      # Medium complexity
        'max_timesteps': 800000,
        'min_timesteps': 150000,
        'early_stopping_patience': 15000,
        'convergence_window': 1500,
        'reward_threshold': -200,
    }
}
```

#### Convergence Monitoring Process
1. **Step-by-Step Tracking**: Monitor training progress every step
2. **Evaluation Integration**: Connect with existing evaluation callbacks
3. **Performance Analysis**: Track reward trends and stability
4. **Decision Making**: Trigger early stopping based on criteria
5. **Logging**: Record convergence events and timing

#### Resource Efficiency Benefits
- **CartPole-v1**: ~60% time reduction (200K vs 500K timesteps)
- **Acrobot-v1**: Appropriate complexity-based timing
- **MountainCar-v0**: Balanced training duration
- **Overall**: Significant resource savings across all training types

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

## Capacity Matching to Fixed Targets

### Fixed Target Capacities
- **Targets**: 1K, 5K, 10K, 50K parameters
- **Method**: Grid search with fixed optimal hyperparameters
- **Advantage**: Systematic comparison across topologies

### Pre-calculation Process
- Calculate effective hidden size for each topology
- Use fixed optimal hyperparameters
- Ensure fair comparison across different network structures

## Experimental Variants and Architectures

### Training Types
- **Baseline**: Single task training with comprehensive evaluation
- **Single-Task**: Train on one task, test on all tasks
- **Double-Task**: Sequential training on two tasks with intermediate testing
- **Triple-Task**: Sequential training on three tasks with intermediate testing

### Enhanced Intermediate Testing System
- **Testing Schedule**: After each training phase, test on ALL tasks
- **Temporal Tracking**: Clear phase-based metric naming
- **Transfer Analysis**: Forward and backward transfer measurement

#### Double-Task Training Flow
```
Phase 1: Train on Task 1 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 2: Train on Task 2 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
```

#### Triple-Task Training Flow
```
Phase 1: Train on Task 1 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 2: Train on Task 2 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 3: Train on Task 3 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
```

## Logging and Results Management

### Topology-Aware Metric Structure
```
{topology_type}/{task_sequence}/phase{phase_number}/testing/{task}/{metric}
```

### Example Metrics
```
small_world/CartPole-v1_Acrobot-v1/phase1/testing/CartPole-v1/mean_reward: 450
small_world/CartPole-v1_Acrobot-v1/phase2/testing/CartPole-v1/mean_reward: 440
modular/CartPole-v1_Acrobot-v1/phase1/testing/CartPole-v1/mean_reward: 420
modular/CartPole-v1_Acrobot-v1/phase2/testing/CartPole-v1/mean_reward: 410
```

### Transfer Learning Metrics
- **Forward Transfer**: How well does training on A help with B?
- **Backward Transfer**: How well does training on B affect A retention?
- **Catastrophic Forgetting**: Measure of performance degradation

### Benefits
- **Complete temporal visibility**: Every phase tested on all tasks
- **Clear progression tracking**: Learning and forgetting patterns
- **Easy comparative analysis**: Same task sequence, different topologies
- **Systematic transfer analysis**: Forward and backward transfer patterns

## Training and Sweep Management

### Sweep Types
- **Fixed Network Sizes**: Compare topologies across fixed hidden sizes (64, 128, 256, 512)
- **Fixed Capacities**: Compare topologies across fixed parameter counts (1K, 5K, 10K, 50K)

### Sweep Configuration
- **Method**: Grid search for systematic comparison
- **Primary Metric**: `normalized/final_normalized_score`
- **Fixed Hyperparameters**: Optimal settings for fair comparison

### Task Order Combinations
- **Double-Task**: 6 valid combinations (no duplicates)
- **Triple-Task**: 6 valid permutations (no duplicates)
- **Validation**: Ensure no duplicate tasks in sequence

## Task Logic Variants

### Baseline Logic
- Train on single task
- Evaluate on training task
- Comprehensive network analysis

### Single-Task Logic
- Train on one task
- Test on all available tasks
- Cross-task transfer analysis

### Double-Task Logic
- Sequential training on two tasks
- Intermediate testing after each phase
- Forward and backward transfer analysis

### Triple-Task Logic
- Sequential training on three tasks
- Intermediate testing after each phase
- Comprehensive transfer learning analysis

## Variable Naming Conventions

### Topology-Aware Structure
```
{topology_type}/{task_sequence}/phase{phase_number}/testing/{task}/{metric}
```

### Training Phases
```
phase1/ - After training on first task
phase2/ - After training on second task
phase3/ - After training on third task (triple-task only)
```

### Testing Contexts
```
testing/ - Final evaluation on all tasks
transfer/ - Transfer learning metrics
training/ - Training metadata and configuration
```

### Task-Specific Metrics
```
{task}/mean_reward - Average reward for specific task
{task}/success_rate - Success rate for specific task
{task}/mean_length - Average episode length for specific task
```

## Code Organization Principles

### File Structure
- **Training Scripts**: `topologies--{type}-training-sweep.py`
- **Configuration**: `wandb_sweep_config.py`
- **Launch Script**: `launch_sweep.py`
- **Utilities**: `src/utils/` directory

### Function Organization
- **Unified Training Functions**: Main entry points for wandb sweeps
- **Logging Functions**: Topology-aware metric logging
- **Evaluation Functions**: Comprehensive task evaluation
- **Utility Functions**: Capacity matching, normalization, etc.

### Consistency Requirements
- **Same metric structure** across all training types
- **Consistent parameter naming** across all files
- **Unified logging approach** with topology context
- **Backward compatibility** with legacy metric names 