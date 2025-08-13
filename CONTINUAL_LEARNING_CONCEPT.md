# Continual Learning Concept for Topology Networks

## Core Principles

### Sequential Training with Comprehensive Evaluation
- **Sequential training**: Train on one task, then the next, then the next
- **Comprehensive evaluation**: After each training phase, test on ALL tasks
- **Transfer analysis**: Measure forward transfer (help) and backward transfer (retention/forgetting)
- **Consistent logging**: Reliable saving of results in Weights & Biases (wandb)
- **Reliable state management**: Proper I/O handling for sequential training on the same network

### Adaptive Training with Early Stopping
- **Task-specific training times**: Individual timesteps per task based on complexity
- **Convergence monitoring**: Real-time performance tracking every 5K steps
- **Early stopping**: Automatic termination when tasks converge or timeout
- **Resource optimization**: Prevents overtraining and resource waste

## Training Flows

### Double-Task Training Flow
```
Phase 1: Train on Task 1 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
         ↓
Phase 2: Train on Task 2 (600K timesteps)
         ↓
Testing: Test on ALL tasks (CartPole, Acrobot, MountainCar)
```

**Key Metrics Tracked:**
- **Phase 1 Results**: Baseline performance on all tasks after training on Task 1
- **Phase 2 Results**: Performance on all tasks after training on Task 2
- **Forward Transfer**: How well does training on Task 1 help with Task 2?
- **Backward Transfer**: How well does training on Task 2 affect Task 1 retention?
- **Catastrophic Forgetting**: Is performance on Task 1 degrading after training on Task 2?

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

## Early Stopping Implementation

### When Early Stopping Occurs

#### 1. **Convergence Detection**
- **Trigger**: Task reaches target performance threshold
- **Example**: CartPole-v1 achieves mean reward ≥ 450
- **Action**: Stop training and proceed to testing phase
- **Benefit**: Prevents overtraining on simple tasks

#### 2. **Performance Plateau**
- **Trigger**: No improvement for specified patience period
- **Example**: Acrobot-v1 shows no improvement for 20K steps
- **Action**: Stop training to avoid resource waste
- **Benefit**: Prevents training on converged or stuck models

#### 3. **Maximum Timeout**
- **Trigger**: Reaches task-specific maximum timesteps
- **Example**: MountainCar-v0 reaches 800K timesteps
- **Action**: Force stop training regardless of performance
- **Benefit**: Ensures finite training time for complex tasks

#### 4. **Target Achievement**
- **Trigger**: Reaches task-specific target timesteps
- **Example**: CartPole-v1 reaches 200K timesteps
- **Action**: Stop training as planned
- **Benefit**: Follows predetermined optimal training schedule

### How Early Stopping is Handled

#### **ConvergenceCallback Implementation**
```python
class ConvergenceCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Monitor training progress every step
        # Check convergence every 5K steps
        # Trigger early stopping based on criteria
        return not self.should_stop
    
    def _check_convergence_via_evaluation(self):
        # Check if target timesteps reached
        # Check if maximum timesteps exceeded
        # Log convergence events
```

#### **Integration with Training Pipeline**
1. **Callback Registration**: ConvergenceCallback added to training callbacks
2. **Step-by-Step Monitoring**: Every training step is monitored
3. **Evaluation Integration**: Connects with existing evaluation callbacks
4. **Decision Making**: Determines when to stop training
5. **Logging**: Records convergence events and timing

#### **Resource Efficiency Benefits**
- **CartPole-v1**: ~60% time reduction (200K vs 500K timesteps)
- **Acrobot-v1**: Appropriate complexity-based timing (800K timesteps)
- **MountainCar-v0**: Balanced training duration (600K timesteps)
- **Overall**: Significant resource savings across all training types

#### **Impact on Continual Learning Analysis**
- **Consistent Evaluation**: Early stopping doesn't affect testing phases
- **Fair Comparison**: All topologies use same early stopping criteria
- **Resource Optimization**: More efficient use of computational resources
- **Quality Assurance**: Prevents overtraining and maintains model quality

**Key Metrics Tracked:**
- **Phase 1 Results**: Baseline performance on all tasks after training on Task 1
- **Phase 2 Results**: Performance on all tasks after training on Task 2
- **Phase 3 Results**: Performance on all tasks after training on Task 3
- **Forward Transfer Task 2**: How well does training on Task 1 help with Task 2?
- **Forward Transfer Task 3**: How well does training on Tasks 1+2 help with Task 3?
- **Retention Task 1 after Task 2**: How well is Task 1 retained after training on Task 2?
- **Retention Task 1 after Task 3**: How well is Task 1 retained after training on Task 3?
- **Retention Task 2 after Task 3**: How well is Task 2 retained after training on Task 3?

## Enhanced Metric Structure

### Topology-Aware Temporal Tracking
```
{topology_type}/{task_sequence}/phase{phase_number}/testing/{task}/{metric}
```

### Example Metrics

#### Double-Task Training:
```
small_world/CartPole-v1_Acrobot-v1/phase1/testing/CartPole-v1/mean_reward: 450
small_world/CartPole-v1_Acrobot-v1/phase1/testing/Acrobot-v1/mean_reward: 50
small_world/CartPole-v1_Acrobot-v1/phase1/testing/MountainCar-v0/mean_reward: 30

small_world/CartPole-v1_Acrobot-v1/phase2/testing/CartPole-v1/mean_reward: 440
small_world/CartPole-v1_Acrobot-v1/phase2/testing/Acrobot-v1/mean_reward: 200
small_world/CartPole-v1_Acrobot-v1/phase2/testing/MountainCar-v0/mean_reward: 35
```

#### Triple-Task Training:
```
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase1/testing/CartPole-v1/mean_reward: 450
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase1/testing/Acrobot-v1/mean_reward: 50
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase1/testing/MountainCar-v0/mean_reward: 30

small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase2/testing/CartPole-v1/mean_reward: 440
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase2/testing/Acrobot-v1/mean_reward: 200
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase2/testing/MountainCar-v0/mean_reward: 35

small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase3/testing/CartPole-v1/mean_reward: 435
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase3/testing/Acrobot-v1/mean_reward: 180
small_world/CartPole-v1_Acrobot-v1_MountainCar-v0/phase3/testing/MountainCar-v0/mean_reward: 180
```

## Transfer Learning Metrics

### Forward Transfer
- **Definition**: How well does training on previous tasks help with current task?
- **Double-Task**: `forward_transfer_score = task2_final - task2_baseline`
- **Triple-Task**: 
  - `forward_transfer_task2 = task2_after_task1 - task2_baseline`
  - `forward_transfer_task3 = task3_after_task2 - task3_baseline`

### Backward Transfer (Retention)
- **Definition**: How well does training on later tasks affect retention of earlier tasks?
- **Double-Task**: `backward_transfer_score = task1_final / task1_baseline`
- **Triple-Task**:
  - `retention_task1_after_task2 = task1_phase2 / task1_phase1`
  - `retention_task1_after_task3 = task1_phase3 / task1_phase2`
  - `retention_task2_after_task3 = task2_phase3 / task2_phase2`

### Catastrophic Forgetting
- **Definition**: Measure of performance degradation on previously learned tasks
- **Calculation**: `catastrophic_forgetting = 1 - retention_rate`

## Task Order Combinations

### Non-Duplicated Permutations
- **Double-Task**: 6 valid combinations (no duplicate tasks in sequence)
  - CartPole-v1 → Acrobot-v1
  - CartPole-v1 → MountainCar-v0
  - Acrobot-v1 → CartPole-v1
  - Acrobot-v1 → MountainCar-v0
  - MountainCar-v0 → CartPole-v1
  - MountainCar-v0 → Acrobot-v1

- **Triple-Task**: 6 valid permutations (no duplicate tasks in sequence)
  - CartPole-v1 → Acrobot-v1 → MountainCar-v0
  - CartPole-v1 → MountainCar-v0 → Acrobot-v1
  - Acrobot-v1 → CartPole-v1 → MountainCar-v0
  - Acrobot-v1 → MountainCar-v0 → CartPole-v1
  - MountainCar-v0 → CartPole-v1 → Acrobot-v1
  - MountainCar-v0 → Acrobot-v1 → CartPole-v1

### Sweep Configuration
- **Grid search** with fixed hyperparameters
- **Task order validation** to ensure no duplicates
- **Systematic coverage** of all valid combinations

## Hierarchical Variable Naming Convention

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

## Benefits of Intermediate Testing

### Complete Temporal Visibility
- **Every phase** is tested on all tasks
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