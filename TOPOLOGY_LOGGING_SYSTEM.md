# Topology Logging System

## Overview

The **Topology Logging System** is a centralized, comprehensive logging handler that consolidates all logging logic for topology network training experiments. It provides consistent, hierarchical logging across individual runs, batch runs, and W&B sweeps.

## 🎯 **Key Benefits**

### **1. Centralized Management**
- **Single source of truth** for all logging logic
- **Easy maintenance** and updates
- **Consistent behavior** across all training modes

### **2. Hierarchical Organization**
- **Clean W&B workspace structure** with proper subsections
- **Organized metrics** by training phase, task order, and topology type
- **Easy data analysis** and comparison

### **3. Flexible Usage**
- **Individual runs**: Single experiment logging
- **Batch runs**: Multiple parameter combinations
- **W&B sweeps**: Hyperparameter optimization
- **All modes use identical logging structure**

## 🏗️ **Architecture**

### **Core Components**

#### **1. TopologyLoggingHandler (Main Class)**
```python
class TopologyLoggingHandler:
    """Main logging handler that coordinates all logging activities."""
    
    def __init__(self, config, topology_type, training_type='triple_task')
    def initialize_run(self, model=None, total_params=None)
    def update_run_name(self, model, total_params)
    def set_task_phase(self, task_name, phase_number)
    def log_training_step(self, step, metrics)
    def log_rollout_end(self, step, metrics)
    def log_network_info(self, step, model, hidden_size, num_layers)
    def log_cross_task_evaluation(self, step, task, metrics)
    def log_all_tables(self, model, hidden_size, num_layers)
```

#### **2. RunNamingManager (Naming & Tagging)**
```python
class RunNamingManager:
    """Manages run naming and tagging for consistent identification."""
    
    @staticmethod
    def create_initial_run_name(config, topology_type, training_type)
    @staticmethod
    def create_final_run_name(config, topology_type, training_type, model, total_params)
    @staticmethod
    def create_run_tags(config, topology_type, training_type, model, total_params)
```

#### **3. MetricsLogger (Metrics Logging)**
```python
class MetricsLogger:
    """Handles all metrics logging with consistent hierarchical structure."""
    
    @staticmethod
    def log_training_metrics(step, metrics, task_order, current_task, current_phase)
    @staticmethod
    def log_rollout_metrics(step, metrics, task_order)
    @staticmethod
    def log_network_metrics(step, metrics, task_order)
    @staticmethod
    def log_learning_progression(step, metrics, task_order, current_task, current_phase)
    @staticmethod
    def log_cross_task_comparison(step, metrics, task)
```

#### **4. TableCreator (Data Tables)**
```python
class TableCreator:
    """Creates and logs W&B tables for structured data analysis."""
    
    @staticmethod
    def create_training_summary_table(training_results, topology_type, task_order)
    @staticmethod
    def create_network_architecture_table(model, topology_type, hidden_size, num_layers, config)
    @staticmethod
    def create_phase_results_table(phase_results, topology_type, task_order)
    @staticmethod
    def create_transfer_learning_table(transfer_metrics, topology_type, task_order)
```

#### **5. EnhancedDebugCallback (Training Integration)**
```python
class EnhancedDebugCallback(BaseCallback):
    """Enhanced callback that uses the centralized logging handler."""
    
    def __init__(self, logging_handler, verbose=0, log_freq=1000)
    def set_task_phase(self, task_name, phase_number)
    def _on_step(self) -> bool
    def _on_rollout_end(self) -> None
    def _on_training_end(self) -> None
```

## 📊 **Logging Structure**

### **Hierarchical Paths**

```
train/
├── global/                    # Global training metrics
└── task_orders/
    └── {task_order}/
        └── phase_{phase}_{task}/  # Task-specific metrics

network/
├── global/                    # Global network metrics
├── architecture/              # Network architecture details
└── capacity/                  # Capacity matching analysis

rollout/
├── global/                    # Global rollout metrics
└── task_orders/
    └── {task_order}/          # Task-specific rollout metrics

learning_progression/
├── global/                    # Global learning progression
└── task_orders/
    └── {task_order}/
        └── phase_{phase}_{task}/  # Phase-specific progression

cross_task_comparison/
└── {task}/                    # Cross-task evaluation results

tables/
├── training_summary           # Training summary table
├── network_architecture       # Network architecture table
├── phase_results              # Phase results table
└── transfer_learning          # Transfer learning table
```

### **Key Metrics Logged**

#### **Training Metrics**
- **Timesteps, episodes, phases**
- **PPO metrics**: loss, entropy, learning rate, value, policy, clip
- **Performance**: mean reward, mean length, success rate, completion percentage
- **Progress**: training progress, phase transitions

#### **Network Metrics**
- **Architecture**: topology type, hidden size, layers
- **Parameters**: total, actor, critic, efficiency ratios
- **Capacity**: target vs. actual, matching ratios, differences

#### **Rollout Metrics**
- **Observations**: mean, standard deviation
- **Phase tracking**: current phase, task information

#### **Learning Progression**
- **Sequential training**: phase-by-phase performance
- **Task transitions**: learning transfer between tasks
- **Convergence**: success rates and completion percentages

## 🚀 **Usage Examples**

### **1. Individual Run**
```python
from src.utils.topology_logging_handler import create_logging_handler

# Create logging handler
logging_handler = create_logging_handler(config, topology_type, 'triple_task')
logging_handler.initialize_run()

# Create callback
callback = EnhancedDebugCallback(logging_handler=logging_handler, log_freq=1000)

# During training
logging_handler.set_task_phase(task_name, phase_number)
logging_handler.log_training_step(step, metrics)

# At the end
logging_handler.log_all_tables(model, hidden_size, num_layers)
logging_handler.finish_run()
```

### **2. Batch Run**
```python
# Same logging handler works for all runs
for config in config_combinations:
    logging_handler = create_logging_handler(config, topology_type, 'triple_task')
    logging_handler.initialize_run()
    
    # ... training ...
    
    logging_handler.finish_run()
```

### **3. W&B Sweep**
```python
# Sweep automatically uses the same logging structure
# No changes needed in sweep configuration
```

## 🔧 **Configuration**

### **Constants & Abbreviations**

#### **Topology Types**
```python
TOPOLOGY_ABBREVIATIONS = {
    'small_world': 'SW',
    'modular': 'MOD', 
    'hybrid': 'HYB',
    'fully_connected': 'FC'
}
```

#### **Task Names**
```python
TASK_ABBREVIATIONS = {
    'LunarLander-v2': 'LL',
    'Acrobot-v1': 'AC', 
    'CartPole-v1': 'CP',
    'MountainCar-v0': 'MC'
}
```

#### **Logging Paths**
```python
LOGGING_PATHS = {
    'train': {'global': 'train/global', 'task_orders': 'train/task_orders'},
    'network': {'global': 'network/global', 'architecture': 'network/architecture'},
    'rollout': {'global': 'rollout/global', 'task_orders': 'rollout/task_orders'},
    'learning_progression': {'global': 'learning_progression/global'},
    'cross_task_comparison': 'cross_task_comparison',
    'tables': 'tables'
}
```

## 📈 **Run Naming Convention**

### **Format**
```
{Topology}_{Capacity}_{Size}_{TaskOrder}
```

### **Examples**
- **Fixed Size**: `SW_C1477_S64_CP-AC-LL`
- **Fixed Capacity**: `SW_C5000_S128_AC-LL-CP`
- **Initial Placeholder**: `SW_C?_S64_CP-AC-LL`

### **Components**
- **Topology**: SW (Small World), MOD (Modular), HYB (Hybrid), FC (Fully Connected)
- **Capacity**: Target capacity for fixed-capacity runs, actual for fixed-size runs
- **Size**: Hidden layer size
- **Task Order**: Abbreviated task sequence (CP-AC-LL = CartPole → Acrobot → LunarLander)

## 🏷️ **Tagging System**

### **Primary Tags**
- **Topology type**: `small_world`, `modular`, `hybrid`, `fully_connected`
- **Training type**: `triple_task`
- **Metrics**: `normalized_metrics`

### **Capacity Tags**
- **Fixed Capacity**: `fixed_capacity`, `target_capacity_{N}`, `capacity_matched`
- **Fixed Size**: `fixed_size`, `size_{N}`, `size_matched`
- **Actual Capacity**: `actual_capacity_{N}`, `capacity_achieved`

### **Task Tags**
- **Individual tasks**: `CartPole-v1`, `Acrobot-v1`, `LunarLander-v2`
- **Task order**: `order_CartPole-Acrobot-LunarLander`
- **Sweep type**: `sweep_fixed_capacity` or `sweep_fixed_size`

## 📋 **Table Generation**

### **Training Summary Table**
- Topology type, task order, training statistics
- Performance metrics, success rates, completion percentages
- Training time and episode counts

### **Network Architecture Table**
- Network structure details
- Parameter counts and ratios
- Capacity matching analysis

### **Phase Results Table**
- Phase-by-phase performance
- Task-specific metrics
- Success rates and completion percentages

### **Transfer Learning Table**
- Forward transfer scores
- Retention (backward transfer) metrics
- Overall transfer performance

## 🔄 **Migration Guide**

### **From Old System**

#### **Before (Old Callback)**
```python
callback = EnhancedDebugCallback(wandb_run=wandb.run, log_freq=1000, task_order=task_order)
```

#### **After (New System)**
```python
logging_handler = create_logging_handler(config, topology_type, 'triple_task')
logging_handler.initialize_run()
callback = EnhancedDebugCallback(logging_handler=logging_handler, log_freq=1000)
```

#### **Before (Manual Table Logging)**
```python
log_streamlined_tables(wandb_run, training_results, phase_results, transfer_metrics, 
                      model, topology_type, hidden_size, num_layers, task_order, config)
```

#### **After (New System)**
```python
logging_handler.log_all_tables(model, hidden_size, num_layers)
```

### **Benefits of Migration**
- ✅ **Cleaner code**: No more scattered logging functions
- ✅ **Consistent behavior**: Same logging across all modes
- ✅ **Easy maintenance**: Single file to update
- ✅ **Better organization**: Hierarchical W&B structure
- ✅ **Fixed run naming**: Proper capacity tracking

## 🧪 **Testing**

### **Import Test**
```bash
python3 -c "from src.utils.topology_logging_handler import TopologyLoggingHandler; print('✅ Success!')"
```

### **Functionality Test**
```bash
# Test individual run
python3 topologies_triple_task_training_sweep.py

# Test batch run
python3 -c "from topologies_triple_task_training_sweep import unified_training_function; unified_training_function('batch')"
```

## 🎯 **Success Criteria**

### **✅ Run Naming**
- [ ] Capacity shows actual values (not `C?`)
- [ ] Size reflects actual hidden layer size
- [ ] Task order is correctly abbreviated
- [ ] No duplicate run names

### **✅ W&B Organization**
- [ ] Hierarchical structure with proper subsections
- [ ] `train/global/` and `train/task_orders/` paths exist
- [ ] `network/global/` and `network/capacity/` paths exist
- [ ] `rollout/global/` and `rollout/task_orders/` paths exist

### **✅ Tables**
- [ ] Training summary table appears
- [ ] Network architecture table shows capacity analysis
- [ ] Phase results table displays sequential training
- [ ] Transfer learning table shows cross-task performance

### **✅ Consistency**
- [ ] Individual runs use same logging as batch runs
- [ ] Batch runs use same logging as sweeps
- [ ] All modes produce identical W&B structure
- [ ] No divergence between different training modes

## 🚀 **Next Steps**

1. **Test the new system** with a simple individual run
2. **Verify W&B workspace** shows proper hierarchical structure
3. **Check run names** display actual capacity values
4. **Confirm tables** are generated and visible
5. **Test batch runs** to ensure consistency
6. **Validate sweep integration** works seamlessly

## 📚 **File Locations**

- **Main Handler**: `src/utils/topology_logging_handler.py`
- **Training Script**: `topologies_triple_task_training_sweep.py`
- **Sweep Config**: `wandb_sweep_config.py`
- **Documentation**: `TOPOLOGY_LOGGING_SYSTEM.md`

---

**The new logging system provides a robust, maintainable foundation for all topology training experiments while ensuring consistent data organization and easy analysis in W&B.**
