# Sweep Refactoring Plan

## Overview

This document outlines the plan for refactoring the 4 topology sweep files to align with the continual learning concept, implement comparison sweeps, and improve wandb logging.

## Goals

1. **Unified Training Function Structure**: Consistent approach across all training types
2. **Comparison Sweeps**: Fixed network sizes and fixed capacities for systematic comparison
3. **Improved Logging**: Comprehensive, topology-aware metric logging
4. **Enhanced Intermediate Testing**: Temporal tracking of learning progression

## Enhanced Intermediate Testing System

### Testing Schedule
- **After each training phase**: Test on ALL tasks
- **Temporal tracking**: Clear phase-based metric naming
- **Transfer analysis**: Forward and backward transfer measurement

### Double-Task Training Flow
```
Phase 1: Train on Task 1 (200K timesteps) - Adaptive with early stopping
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 2: Train on Task 2 (800K timesteps) - Adaptive with early stopping
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
```

### Triple-Task Training Flow
```
Phase 1: Train on Task 1 (200K timesteps) - Adaptive with early stopping
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 2: Train on Task 2 (800K timesteps) - Adaptive with early stopping
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 3: Train on Task 3 (600K timesteps) - Adaptive with early stopping
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
```

## Adaptive Training with Early Stopping

### Task-Specific Training Times
- **CartPole-v1**: 200K timesteps (fast convergence)
- **Acrobot-v1**: 800K timesteps (complex task)
- **MountainCar-v0**: 600K timesteps (medium complexity)

### Early Stopping Triggers
1. **Convergence Detection**: Task reaches target performance threshold
2. **Performance Plateau**: No improvement for specified patience period
3. **Maximum Timeout**: Reaches task-specific maximum timesteps
4. **Target Achievement**: Reaches task-specific target timesteps

### Resource Efficiency Benefits
- **CartPole-v1**: ~60% time reduction (200K vs 500K timesteps)
- **Acrobot-v1**: Appropriate complexity-based timing
- **MountainCar-v0**: Balanced training duration
- **Overall**: Significant resource savings across all training types

## Topology-Aware Logging Structure

### Metric Naming Convention
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
```
small_world/CartPole-v1_Acrobot-v1/transfer/forward_transfer_score: 150
small_world/CartPole-v1_Acrobot-v1/transfer/backward_transfer_score: 0.978
small_world/CartPole-v1_Acrobot-v1/transfer/catastrophic_forgetting: 0.022
```

## Comparison Sweep Configurations

### Fixed Network Sizes Comparison
- **Purpose**: Compare topologies across fixed hidden sizes
- **Sizes**: [64, 128, 256, 512]
- **Method**: Grid search with fixed optimal hyperparameters
- **Total Runs**: 4 topologies × 4 sizes × N tasks = 16N runs

### Fixed Capacities Comparison
- **Purpose**: Compare topologies across fixed parameter counts
- **Capacities**: [1K, 5K, 10K, 50K] parameters
- **Method**: Grid search with fixed optimal hyperparameters
- **Total Runs**: 4 topologies × 4 capacities × N tasks = 16N runs

## Improved Logging System

### WandB Initialization
- **Project**: topology-playground
- **Entity**: katko-it-universitetet-i-k-benhavn
- **Run Naming**: Descriptive names with topology and task information
- **Tags**: Easy filtering and organization

### Run Naming Convention
```
{training_type}_{topology_type}_{size/capacity}_{task_sequence}
```

### Examples
```
double_task_small_world_size64_CartPole-v1_Acrobot-v1
triple_task_modular_cap1000_CartPole-v1_Acrobot-v1_MountainCar-v0
```

### Hierarchical Logging Structure
```
{topology_type}/{task_sequence}/phase{phase_number}/testing/{task}/{metric}
{topology_type}/{task_sequence}/transfer/{transfer_metric}
{topology_type}/{task_sequence}/training/{training_metadata}
```

### Dual Logging (Raw + Normalized)
- **Raw Metrics**: Actual performance values
- **Normalized Metrics**: Task-normalized performance (0-1 scale)
- **Primary Optimization**: Use normalized metrics for sweep optimization

### Legacy Compatibility
- **Backward Compatibility**: Maintain legacy metric names
- **Gradual Migration**: Add new metrics alongside existing ones
- **No Breaking Changes**: Existing dashboards continue to work

## File-Specific Refactoring

### 1. topologies--baseline-training-sweep.py
- **Enhanced Logging**: Add topology-aware metric structure
- **Capacity Matching**: Integrate with fixed capacity sweeps
- **Normalized Metrics**: Add reward scaling and task normalization

### 2. topologies--single-task-training-sweep.py
- **Enhanced Logging**: Add topology-aware metric structure
- **Cross-Task Testing**: Test on all tasks after training
- **Transfer Analysis**: Measure transfer learning effects

### 3. topologies--double-task-training-sweep.py
- **Intermediate Testing**: Test after each training phase
- **Temporal Tracking**: Phase-based metric naming
- **Transfer Metrics**: Forward and backward transfer analysis

### 4. topologies--triple-task-training-sweep.py
- **Intermediate Testing**: Test after each training phase
- **Temporal Tracking**: Phase-based metric naming
- **Comprehensive Transfer**: Multi-phase transfer learning analysis

## Integration with Launch Functions

### Launch Script Updates
- **Comparison Sweeps**: Add fixed network sizes and fixed capacities options
- **Sweep Configuration**: Use new comparison sweep configs
- **Run Calculation**: Display total runs for each sweep type
- **Metric Display**: Show primary normalized metric

### Menu Structure
```
Topology Analysis Suite:
1. Fixed Network Sizes Comparison
2. Fixed Capacities Comparison
3. Hyperparameter Optimization
4. Custom Configuration
```

## Benefits of Enhanced System

### Complete Temporal Visibility
- **Every phase** tested on all tasks
- **Clear progression** of learning and forgetting
- **Exact timing** of when each measurement was taken

### Comprehensive Transfer Analysis
- **Forward transfer**: How does training on A help with B?
- **Backward transfer**: How does training on B affect A retention?
- **Catastrophic forgetting**: Is performance degrading over time?

### Easy Comparative Analysis
- **Same task sequence, different topologies** → Compare learning patterns
- **Same topology, different task sequences** → Compare transfer effects
- **Cross-phase analysis** → Track temporal evolution

### Systematic Comparison
- **Fixed hyperparameters** ensure fair comparison
- **Grid search** covers all topology-capacity combinations
- **Normalized metrics** enable meaningful aggregation

## Implementation Status

### ✅ Completed
- [x] Enhanced intermediate testing system
- [x] Topology-aware logging structure
- [x] Transfer learning metrics
- [x] Comparison sweep configurations
- [x] Unified training function structure
- [x] Backward compatibility with legacy metrics

### 🔄 In Progress
- [ ] Integration with launch functions
- [ ] Menu structure updates
- [ ] Documentation updates

### 📋 Planned
- [ ] Dashboard template creation
- [ ] Analysis script development
- [ ] Performance optimization

## Dashboard Analysis Examples

### Track Learning Progression
```
Filter: task = "CartPole-v1"
Metrics:
├── small_world/*/phase1/testing/CartPole-v1/mean_reward (baseline)
├── small_world/*/phase2/testing/CartPole-v1/mean_reward (after Acrobot)
└── small_world/*/phase3/testing/CartPole-v1/mean_reward (after MountainCar)
```

### Compare Forward Transfer
```
Filter: training_sequence = "CartPole-v1_Acrobot-v1"
Metrics:
├── small_world/CartPole-v1_Acrobot-v1/phase1/testing/Acrobot-v1/mean_reward (baseline)
└── small_world/CartPole-v1_Acrobot-v1/phase2/testing/Acrobot-v1/mean_reward (after training)
```

### Analyze Catastrophic Forgetting
```
Filter: task = "CartPole-v1"
Metrics:
├── small_world/*/phase1/testing/CartPole-v1/mean_reward (baseline)
├── small_world/*/phase2/testing/CartPole-v1/mean_reward (after Acrobot)
└── small_world/*/phase3/testing/CartPole-v1/mean_reward (after MountainCar)
```

## Conclusion

This enhanced system provides complete temporal visibility into the continual learning process, enabling comprehensive analysis of topology performance, transfer learning patterns, and learning progression. The topology-aware logging structure makes it easy to compare performance across different topologies and task sequences, while the intermediate testing system captures the temporal evolution of learning and forgetting. 