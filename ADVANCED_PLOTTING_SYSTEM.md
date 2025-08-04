# Advanced Plotting System for Topology Network Analysis

## Overview

This document describes the comprehensive plotting system implemented for analyzing topology network performance across different training phases and task orders. The system generates **separate plots for each task order combination** to handle the complexity of continual learning experiments.

## 🎯 Key Features

### **Task Order Complexity Handling**
- **Double-Task**: 6 task order combinations (e.g., CartPole→Acrobot, Acrobot→CartPole)
- **Triple-Task**: 6 task order combinations (e.g., CartPole→Acrobot→MountainCar)
- **Each plot type** is generated **separately for each task order**
- **Total**: 60+ plots per sweep for comprehensive analysis

### **Temporal Learning Progression**
- **Multi-phase learning curves** showing performance evolution
- **Clear phase separation** with training annotations
- **Complete temporal visibility** of learning and forgetting patterns

### **Transfer Learning Analysis**
- **Forward transfer** comparison across topologies
- **Backward transfer** (retention) analysis
- **Catastrophic forgetting** measurement
- **Overall transfer scores** for ranking

## 📊 Plot Types

### **1. Multi-Phase Learning Curves (CRITICAL)**
**Purpose**: Show how each topology learns and forgets across phases

**Features**:
- Performance lines for each task (CartPole, Acrobot, MountainCar)
- Vertical lines marking training phase transitions
- Annotations showing which task was trained in each phase
- Interactive hover information

**Example**:
```
Learning Progression: small_world - CartPole-v1_Acrobot-v1
├── CartPole-v1: 450 → 440 (slight forgetting)
├── Acrobot-v1: 50 → 200 (strong forward transfer)
└── MountainCar-v0: 30 → 35 (minimal transfer)
```

### **2. Transfer Learning Comparison (CRITICAL)**
**Purpose**: Compare transfer learning performance across topologies

**Subplots**:
- **Forward Transfer**: How well does training on A help with B?
- **Backward Transfer**: How well is A retained after training on B?
- **Catastrophic Forgetting**: Measure of performance degradation
- **Overall Transfer Score**: Combined metric for ranking

**Example**:
```
Transfer Learning Comparison: CartPole-v1_Acrobot-v1
├── Small World: Forward=150, Retention=0.978, Forgetting=0.022
├── Modular: Forward=120, Retention=0.950, Forgetting=0.050
├── Hybrid: Forward=140, Retention=0.965, Forgetting=0.035
└── Fully Connected: Forward=100, Retention=0.920, Forgetting=0.080
```

### **3. Topology Performance Matrix (CRITICAL)**
**Purpose**: Heatmap showing topology performance across all tasks and phases

**Features**:
- **Rows**: Topologies (small_world, modular, hybrid, fully_connected)
- **Columns**: Tasks (CartPole, Acrobot, MountainCar)
- **Subplots**: One for each training phase
- **Color intensity**: Performance level (higher = better)

**Example**:
```
Phase 1 Matrix: CartPole-v1_Acrobot-v1
┌─────────────┬─────────────┬─────────────┐
│             │ CartPole    │ Acrobot     │
├─────────────┼─────────────┼─────────────┤
│ Small World │    450      │     50      │
│ Modular     │    420      │     45      │
│ Hybrid      │    430      │     48      │
│ Fully Conn  │    400      │     40      │
└─────────────┴─────────────┴─────────────┘
```

### **4. Capacity Scaling Analysis (IMPORTANT)**
**Purpose**: Show how topology performance scales with different capacities

**Features**:
- **X-axis**: Parameter capacity (1K, 5K, 10K, 50K)
- **Y-axis**: Mean reward
- **Lines**: One for each task
- **Log scale**: For better visualization of capacity ranges

**Example**:
```
Capacity Scaling: small_world - CartPole-v1_Acrobot-v1
├── CartPole-v1: 400 → 450 → 480 → 500
├── Acrobot-v1: 30 → 150 → 200 → 220
└── MountainCar-v0: 25 → 30 → 35 → 40
```

### **5. Task Order Effects (IMPORTANT)**
**Purpose**: Analyze how different task orders affect topology performance

**Features**:
- **Lines**: One for each task order combination
- **X-axis**: Training phases
- **Y-axis**: Overall performance
- **Legend**: Shows task sequence

**Example**:
```
Task Order Effects: small_world (double_task)
├── CartPole→Acrobot: Phase1=450, Phase2=440
├── CartPole→MountainCar: Phase1=450, Phase2=435
├── Acrobot→CartPole: Phase1=50, Phase2=420
└── ...
```

## 🚀 Implementation

### **File Structure**
```
src/utils/advanced_plotting.py
├── Core plotting functions
├── Task order parsing utilities
├── Color schemes and constants
└── WandB integration functions
```

### **Integration Points**
All training scripts now include advanced plotting:

```python
# In each training script
from src.utils.advanced_plotting import (
    log_comprehensive_plots_for_run, create_multi_phase_learning_curves
)

# After training completion
log_comprehensive_plots_for_run(
    wandb_run=wandb.run,
    phase_results=all_phase_results,
    transfer_metrics=transfer_metrics,
    topology_type=topology_type,
    task_sequence=task_order,
    sweep_results=None
)
```

### **WandB Logging Structure**
```
plots/
├── {topology_type}/{task_sequence}/learning_progression
├── {topology_type}/{task_sequence}/transfer_analysis
├── {topology_type}/{task_sequence}/capacity_scaling
├── sweep_analysis/performance_matrix_{task_sequence}
└── {topology_type}/task_order_effects
```

## 📈 Analysis Workflow

### **1. Individual Run Analysis**
Each training run generates:
- **Learning progression plot** for that specific topology/task order
- **Transfer metrics** for that specific combination
- **Capacity scaling** (if capacity data available)

### **2. Sweep-Level Analysis**
When sweep results are available:
- **Performance matrices** comparing all topologies
- **Transfer learning comparisons** across all topologies
- **Task order effects** showing robustness to task sequence

### **3. Comparative Analysis**
Easy comparison across:
- **Same task sequence, different topologies** → Which topology is best?
- **Same topology, different task sequences** → Which task order works best?
- **Same topology, different capacities** → How does performance scale?

## 🎨 Visualization Features

### **Color Schemes**
- **Topology Colors**: Consistent colors for each topology type
- **Task Colors**: Consistent colors for each task
- **Interactive Elements**: Hover information, zoom, pan

### **Layout Optimization**
- **Responsive Design**: Adapts to different screen sizes
- **Subplot Organization**: Logical grouping of related metrics
- **Clear Titles**: Descriptive titles with topology and task information

### **Interactive Features**
- **Hover Information**: Detailed metrics on mouse hover
- **Zoom and Pan**: Interactive exploration of data
- **Legend Controls**: Show/hide specific traces
- **Export Options**: Save plots as images

## 🔍 Dashboard Analysis Examples

### **Track Learning Progression**
```
Filter: task = "CartPole-v1"
Metrics:
├── small_world/*/phase1/testing/CartPole-v1/mean_reward (baseline)
├── small_world/*/phase2/testing/CartPole-v1/mean_reward (after Acrobot)
└── small_world/*/phase3/testing/CartPole-v1/mean_reward (after MountainCar)
```

### **Compare Forward Transfer**
```
Filter: training_sequence = "CartPole-v1_Acrobot-v1"
Metrics:
├── small_world/CartPole-v1_Acrobot-v1/phase1/testing/Acrobot-v1/mean_reward (baseline)
└── small_world/CartPole-v1_Acrobot-v1/phase2/testing/Acrobot-v1/mean_reward (after training)
```

### **Analyze Catastrophic Forgetting**
```
Filter: task = "CartPole-v1"
Metrics:
├── small_world/*/phase1/testing/CartPole-v1/mean_reward (baseline)
├── small_world/*/phase2/testing/CartPole-v1/mean_reward (after Acrobot)
└── small_world/*/phase3/testing/CartPole-v1/mean_reward (after MountainCar)
```

## 🎯 Benefits

### **Complete Temporal Visibility**
- **Every phase** tested on all tasks
- **Clear progression** of learning and forgetting
- **Exact timing** of when each measurement was taken

### **Comprehensive Transfer Analysis**
- **Forward transfer**: How does training on A help with B?
- **Backward transfer**: How does training on B affect A retention?
- **Catastrophic forgetting**: Is performance degrading over time?

### **Easy Comparative Analysis**
- **Same task sequence, different topologies** → Compare learning patterns
- **Same topology, different task sequences** → Compare transfer effects
- **Cross-phase analysis** → Track temporal evolution

### **Systematic Comparison**
- **Fixed hyperparameters** ensure fair comparison
- **Grid search** covers all topology-capacity combinations
- **Normalized metrics** enable meaningful aggregation

## 🚀 Usage

### **Automatic Generation**
Plots are automatically generated after each training run:
```python
# No manual intervention needed
# Plots are created and logged to wandb automatically
```

### **Manual Generation**
For custom analysis:
```python
from src.utils.advanced_plotting import create_multi_phase_learning_curves

# Create custom plot
fig = create_multi_phase_learning_curves(phase_results, topology_type, task_sequence)
fig.show()
```

### **Sweep-Level Analysis**
For comprehensive sweep analysis:
```python
from src.utils.advanced_plotting import generate_all_plots_for_sweep

# Generate all plots for all task orders
all_plots = generate_all_plots_for_sweep(sweep_results)
log_all_plots_to_wandb(wandb_run, all_plots)
```

## 📋 Summary

This advanced plotting system provides:

1. **Complete temporal visibility** into the continual learning process
2. **Comprehensive transfer learning analysis** across all topologies
3. **Easy comparative analysis** across different task orders and topologies
4. **Systematic visualization** of topology performance patterns
5. **Interactive dashboards** for detailed exploration

The system handles the complexity of task order combinations by generating **separate plots for each task order**, ensuring that every learning scenario is properly visualized and analyzed. This enables researchers to understand which topologies work best for which task sequences and how transfer learning patterns vary across different network architectures. 