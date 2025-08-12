# Streamlined Logging Concept for Topology Research

## 🎯 **Research Methodology & Logging Strategy**

### **Core Research Question:**
*"How do different network topologies (Small World, Modular, Hybrid, Fully Connected) perform across different task orders and parameter configurations?"*

### **Key Variables:**
1. **Topology Type**: SW, MOD, HYB, FC
2. **Task Order**: CP-AC-LL, AC-LL-CP, LL-CP-AC
3. **Parameter Configuration**: Fixed Size vs. Fixed Capacity
4. **Network Size/Capacity**: 64/128/256 or 1000/5000/10000

## 🏗️ **W&B Workspace Structure**

### **1. Main Sections (Clean, Organized)**
```
📊 Topology Comparison (Main Research Focus)
├── Task Order: CP-AC-LL
├── Task Order: AC-LL-CP  
└── Task Order: LL-CP-AC

🔧 Network Architecture (Tables Only)
├── Topology Parameters
├── Capacity Analysis
└── Architecture Details

📈 Training Progress (Essential Plots Only)
├── Learning Curves by Task Order
├── Transfer Learning Analysis
└── Convergence Patterns

📋 Run Summary (Tables Only)
├── Training Statistics
├── Performance Metrics
└── Cross-Topology Comparison
```

### **2. What Gets Plotted vs. Tabled**

#### **📈 PLOTS (Dynamic Data - Changes During Training)**
- **Learning curves** by task order and topology
- **Transfer learning** performance across phases
- **Convergence patterns** by topology type
- **Cross-task comparison** during evaluation

#### **📋 TABLES (Static Data - Same Throughout Run)**
- **Network architecture** details
- **Parameter counts** and capacity matching
- **Training configuration** settings
- **Run metadata** and identifiers

## 🚀 **Implementation Strategy**

### **Phase 1: Fix Run Naming & Basic Structure**
1. **Ensure capacity is properly calculated** and displayed in run names
2. **Create proper hierarchical logging paths** for W&B organization
3. **Separate data by task order** for meaningful comparison

### **Phase 2: Streamline Data Collection**
1. **Collect only essential metrics** during training
2. **Aggregate data across runs** for topology comparison
3. **Create meaningful cross-run analysis**

### **Phase 3: Optimize W&B Display**
1. **Use tables for static information**
2. **Use plots for dynamic, comparable data**
3. **Organize by research focus** (topology comparison)

## 📊 **Specific Logging Paths**

### **Training Metrics (By Task Order)**
```
train/
├── task_order/CP-AC-LL/
│   ├── phase_1_CartPole-v1/
│   ├── phase_2_Acrobot-v1/
│   └── phase_3_LunarLander-v2/
├── task_order/AC-LL-CP/
│   ├── phase_1_Acrobot-v1/
│   ├── phase_2_LunarLander-v2/
│   └── phase_3_CartPole-v1/
└── task_order/LL-CP-AC/
    ├── phase_1_LunarLander-v2/
    ├── phase_2_CartPole-v1/
    └── phase_3_Acrobot-v1/
```

### **Network Information (Tables Only)**
```
network/
├── architecture/          # Table: Topology details
├── capacity/             # Table: Parameter counts
└── configuration/        # Table: Training settings
```

### **Cross-Topology Analysis (Plots)**
```
topology_comparison/
├── task_order/CP-AC-LL/
│   ├── learning_curves/  # Plot: All topologies on same task order
│   ├── transfer_learning/ # Plot: Forward/backward transfer
│   └── final_performance/ # Plot: End-of-training comparison
└── [repeat for other task orders]
```

## 🔧 **Technical Implementation**

### **1. Fix Run Naming**
```python
def update_run_name_with_actual_capacity(model, config):
    """Ensure run name shows actual capacity, not placeholder."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if 'target_capacity' in config:
        # Fixed capacity run - show target in name
        run_name = f"{topology}_{config['target_capacity']}_{actual_size}_{task_order}"
    else:
        # Fixed size run - show actual capacity in name
        run_name = f"{topology}_{total_params}_{config['hidden_size']}_{task_order}"
    
    wandb.run.name = run_name
    return run_name
```

### **2. Hierarchical Logging**
```python
def log_metrics_by_task_order(metrics, task_order, phase, task):
    """Log metrics in proper hierarchical structure."""
    path = f"train/task_order/{task_order}/phase_{phase}_{task}"
    wandb.log({path: metrics})
```

### **3. Cross-Run Aggregation**
```python
def log_topology_comparison(run_data, task_order):
    """Aggregate data across topologies for comparison."""
    # This will create plots comparing all topologies on same task order
    path = f"topology_comparison/task_order/{task_order}"
    wandb.log({path: run_data})
```

## 📋 **Essential Tables (No Plots)**

### **1. Network Architecture Table**
- Topology type, hidden size, layers
- Actual vs. target parameters
- Capacity matching ratio
- Architecture efficiency metrics

### **2. Training Configuration Table**
- Learning rate, batch size, timesteps
- Task-specific settings
- Convergence criteria
- Early stopping parameters

### **3. Run Summary Table**
- Training duration, episodes, steps
- Final performance metrics
- Success rates, completion percentages
- Cross-task transfer scores

## 📈 **Essential Plots (No Tables)**

### **1. Learning Curves by Task Order**
- **X-axis**: Training timesteps
- **Y-axis**: Reward/Completion percentage
- **Lines**: Different topologies (SW, MOD, HYB, FC)
- **Panels**: One per task order (CP-AC-LL, AC-LL-CP, LL-CP-AC)

### **2. Transfer Learning Analysis**
- **X-axis**: Training phases
- **Y-axis**: Performance on each task
- **Bars**: Different topologies
- **Colors**: Task performance (before vs. after training)

### **3. Topology Comparison**
- **X-axis**: Topology types
- **Y-axis**: Final performance metrics
- **Bars**: Different task orders
- **Error bars**: Standard deviation across runs

## 🎯 **Expected W&B Output**

### **Clean, Organized Workspace:**
```
📊 Topology Comparison (3 panels)
├── Learning Curves: CP-AC-LL
├── Learning Curves: AC-LL-CP
└── Learning Curves: LL-CP-AC

🔧 Network Architecture (3 tables)
├── Topology Parameters
├── Capacity Analysis
└── Training Configuration

📈 Training Progress (3 panels)
├── Transfer Learning: CP-AC-LL
├── Transfer Learning: AC-LL-CP
└── Transfer Learning: LL-CP-AC

📋 Run Summary (1 table)
└── Cross-Topology Comparison
```

### **Total: 7 panels + 4 tables = 11 organized sections**

## 🚀 **Implementation Priority**

### **Immediate (Fix Current Issues)**
1. ✅ Fix run naming (capacity display)
2. ✅ Create proper hierarchical paths
3. ✅ Separate data by task order

### **Short Term (Streamline)**
1. 🔄 Convert network stats to tables
2. 🔄 Aggregate data across runs
3. 🔄 Create topology comparison plots

### **Medium Term (Optimize)**
1. 🔄 Optimize W&B panel organization
2. 🔄 Add cross-run analysis
3. 🔄 Create research-focused dashboards

## ✅ **Success Criteria**

### **Clean W&B Workspace**
- [ ] **≤15 total panels** (down from 26)
- [ ] **Clear section organization** by research focus
- [ ] **Proper hierarchical structure** for task orders
- [ ] **Tables for static data**, plots for dynamic data

### **Meaningful Research Data**
- [ ] **Topology comparison** across task orders
- [ ] **Transfer learning analysis** by topology type
- [ ] **Capacity vs. performance** relationships
- [ ] **Cross-run aggregation** for statistical significance

### **Maintainable System**
- [ ] **Consistent logging paths** across all runs
- [ ] **Easy data retrieval** for analysis
- [ ] **Clear separation** of concerns
- [ ] **Scalable structure** for future experiments

---

**This streamlined approach focuses on what matters for topology research: comparing performance across different network architectures, task orders, and parameter configurations, while maintaining a clean, organized W&B workspace.**
