# LunarLander Migration Concept

## 🎯 Overview
Replace MountainCar-v0 with LunarLander-v2 in all training configurations to improve PPO solvability and provide richer learning signals.

## 📊 Task Comparison

### MountainCar-v0 (Current)
- **Observation Space**: 2D (position, velocity)
- **Action Space**: Discrete (3 actions: left, no-op, right)
- **Reward Structure**: Sparse (-1 per step, +0 at goal)
- **Solved Threshold**: -110 (very difficult for PPO)
- **Max Episode Length**: 200 steps
- **Difficulty**: ⭐⭐⭐⭐⭐ (Very Hard)

### LunarLander-v2 (Proposed)
- **Observation Space**: 8D (position, velocity, angle, angular velocity, leg contacts)
- **Action Space**: Discrete (4 actions: no-op, main engine, left engine, right engine)
- **Reward Structure**: Rich (+100-140 for landing, fuel penalties, crash penalties)
- **Solved Threshold**: 200 (much more achievable)
- **Max Episode Length**: 1000 steps
- **Difficulty**: ⭐⭐⭐ (Medium)

## 🔧 Required Changes

### 1. Task Configuration Updates

#### `src/utils/task_normalization.py`
```python
# Update task constants
R_MIN = {
    "CartPole-v1": 0,
    "Acrobot-v1": -500,
    "LunarLander-v2": -1000  # Replace MountainCar-v0
}

R_SOLVED = {
    "CartPole-v1": 500,
    "Acrobot-v1": -80,
    "LunarLander-v2": 200  # Replace MountainCar-v0
}
```

#### `src/utils/task_training_config.py`
```python
TASK_TRAINING_CONFIG = {
    'CartPole-v1': {
        'total_timesteps': 100000,
        'convergence_threshold': 0.95,
        'min_timesteps': 50000,
        'max_timesteps': 200000,
        'early_stopping_patience': 10,
        'convergence_window': 3,
        'reward_threshold': 500
    },
    'Acrobot-v1': {
        'total_timesteps': 200000,
        'convergence_threshold': 0.9,
        'min_timesteps': 100000,
        'max_timesteps': 500000,
        'early_stopping_patience': 15,
        'convergence_window': 5,
        'reward_threshold': -80
    },
    'LunarLander-v2': {  # Replace MountainCar-v0
        'total_timesteps': 300000,  # Longer training needed
        'convergence_threshold': 0.85,
        'min_timesteps': 150000,
        'max_timesteps': 800000,
        'early_stopping_patience': 20,
        'convergence_window': 8,
        'reward_threshold': 200
    }
}
```

### 2. Universal Action/Observation Space Updates

#### `UniversalActionWrapper` Updates
```python
class UniversalActionWrapper(gym.Wrapper):
    def __init__(self, env, task_name):
        super().__init__(env)
        self.task_name = task_name
        
        # Update action space mapping
        self.action_mapping = {
            'CartPole-v1': {0: 0, 1: 1},  # 2 actions
            'Acrobot-v1': {0: 0, 1: 1, 2: 2},  # 3 actions
            'LunarLander-v2': {0: 0, 1: 1, 2: 2, 3: 3}  # 4 actions
        }
        
        # Update observation space padding
        self.obs_padding = {
            'CartPole-v1': 2,  # 4D → 6D
            'Acrobot-v1': 1,   # 6D → 7D
            'LunarLander-v2': 0  # 8D → 8D (no padding needed!)
        }
        
        # Update action space
        max_actions = max(len(mapping) for mapping in self.action_mapping.values())
        self.action_space = gym.spaces.Discrete(max_actions)
        
        # Update observation space
        max_obs = max(4 + padding for padding in self.obs_padding.values())
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_obs,), dtype=np.float32
        )
```

### 3. Training Script Updates

#### Success Rate Calculation
```python
def calculate_success_rate(rewards, episode_lengths, task_name):
    """Calculate success rate based on task-specific criteria."""
    if task_name == 'CartPole-v1':
        return np.mean([reward >= 500 for reward in rewards])
    elif task_name == 'Acrobot-v1':
        return np.mean([reward >= -80 for reward in rewards])
    elif task_name == 'LunarLander-v2':  # Replace MountainCar-v0
        return np.mean([reward >= 200 for reward in rewards])
    else:
        mean_reward = np.mean(rewards)
        return np.mean([reward >= mean_reward for reward in rewards])
```

### 4. Sweep Configuration Updates

#### `wandb_sweep_config.py`
```python
# Update task combinations
task_combinations = {
    'double_task': [
        'CartPole-v1_Acrobot-v1',
        'CartPole-v1_LunarLander-v2',  # Replace MountainCar-v0
        'Acrobot-v1_CartPole-v1',
        'Acrobot-v1_LunarLander-v2',   # Replace MountainCar-v0
        'LunarLander-v2_CartPole-v1',  # Replace MountainCar-v0
        'LunarLander-v2_Acrobot-v1'    # Replace MountainCar-v0
    ],
    'triple_task': [
        'CartPole-v1_Acrobot-v1_LunarLander-v2',  # Replace MountainCar-v0
        'CartPole-v1_LunarLander-v2_Acrobot-v1',  # Replace MountainCar-v0
        'Acrobot-v1_CartPole-v1_LunarLander-v2',  # Replace MountainCar-v0
        'Acrobot-v1_LunarLander-v2_CartPole-v1',  # Replace MountainCar-v0
        'LunarLander-v2_CartPole-v1_Acrobot-v1',  # Replace MountainCar-v0
        'LunarLander-v2_Acrobot-v1_CartPole-v1'   # Replace MountainCar-v0
    ]
}
```

## 🎯 Benefits of LunarLander-v2

### 1. **Better PPO Compatibility**
- Rich reward structure with immediate feedback
- Continuous state space with meaningful gradients
- More forgiving learning dynamics

### 2. **Richer Learning Signals**
- Multiple reward components (landing, fuel, crash)
- Continuous state representation
- More complex but learnable dynamics

### 3. **Better Transfer Learning Potential**
- More complex state space (8D vs 2D)
- Multiple control objectives
- Realistic physics simulation

### 4. **Improved Convergence**
- Achievable solved threshold (200 vs -110)
- Longer episodes allow for more learning
- Better exploration-exploitation balance

## 📋 Implementation Plan

### Phase 1: Configuration Updates
1. ✅ Update `task_normalization.py` constants
2. ✅ Update `task_training_config.py` parameters
3. ✅ Update `UniversalActionWrapper` for new dimensions
4. ✅ Update success rate calculations in all training scripts

### Phase 2: Training Script Updates
1. ✅ Update all 4 training scripts (`baseline`, `single`, `double`, `triple`)
2. ✅ Update sweep configurations
3. ✅ Test standalone execution

### Phase 3: Validation
1. ✅ Test LunarLander-v2 standalone training
2. ✅ Verify early stopping works correctly
3. ✅ Test multi-task training combinations
4. ✅ Validate sweep execution

### Phase 4: Documentation
1. ✅ Update methodology documents
2. ✅ Update sweep refactoring plan
3. ✅ Document new task characteristics

## 🚀 Expected Outcomes

### Training Performance
- **Faster Convergence**: LunarLander-v2 should converge in 200-400K steps
- **Higher Success Rates**: Expected 80-95% success rate vs 0-20% for MountainCar
- **Better Transfer**: More meaningful forward/backward transfer analysis

### Analysis Quality
- **Richer Metrics**: More nuanced performance analysis
- **Better Comparisons**: Fairer topology comparisons
- **More Reliable Results**: Consistent convergence across topologies

### Resource Efficiency
- **Shorter Training**: Less time spent on unsolvable tasks
- **Better Utilization**: More meaningful use of computational resources
- **Clearer Insights**: Easier to distinguish topology differences

## ⚠️ Considerations

### 1. **Task Difficulty Balance**
- LunarLander-v2 is easier than MountainCar-v0 but harder than CartPole-v1
- Good middle ground for topology comparison
- Still challenging enough to show differences

### 2. **State Space Complexity**
- 8D observation space vs 2D for MountainCar
- May require larger networks for optimal performance
- Better test of topology capacity

### 3. **Training Time**
- Longer episodes (1000 vs 200 steps)
- May require more timesteps for convergence
- Better for learning complex behaviors

## 🎯 Next Steps

1. **Implement Configuration Changes**: Update all constants and parameters
2. **Test Standalone Training**: Verify LunarLander-v2 works correctly
3. **Update Training Scripts**: Apply changes to all 4 training types
4. **Validate Sweeps**: Ensure sweep configurations work
5. **Update Documentation**: Reflect changes in methodology docs

This migration will significantly improve the quality and reliability of our topology comparison experiments! 🚀 