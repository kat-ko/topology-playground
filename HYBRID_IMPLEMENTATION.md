# Hybrid Implementation Guide

## Core Changes Needed

### 1. PPO Configuration Updates
```python
# Change from current to paper-aligned
'n_steps': 800,        # Was 2048
'n_epochs': 5,          # Was 10
'batch_size': 32,       # Keep our standard
'learning_rate': 0.0003 # Keep our standard (don't use paper's 0.01)
```

### 2. Reward Scaling Implementation
```python
# Apply 20.0 scaling in episode storage
reward_scale = 20.0

# In environment wrapper:
raw_reward = env.step(action)[1]
scaled_reward = raw_reward * reward_scale
# Store and log scaled_reward
```

### 3. Enhanced Continual Learning Wrapper
```python
class EnhancedContinualLearningWrapper(gym.Wrapper):
    def __init__(self, env, segment_length=200, shift_range=[0, 2], 
                 reward_scale=20.0, episode_cap=400):
        # Initialize with paper-aligned settings
        
    def step(self, action):
        # Check for shift every 200 steps
        if self.global_step % 200 == 0:
            self._resample_shift()
        
        # Apply shift to observation
        # Apply reward scaling
        # Cap episodes at 400 steps
```

### 4. Multi-Granularity Logging
```python
# Log at three levels:
# 1. Per-shift (every 200 steps)
# 2. Per-episode (individual episode completion)
# 3. Per-update (every 800 steps)
```

## Implementation Priority

1. **Week 1**: Update PPO config and implement wrapper
2. **Week 2**: Add enhanced logging system
3. **Week 3**: Integrate with existing seed system
4. **Week 4**: Test and validate

## Key Benefits

- **Paper-aligned**: Matches distribution shift methodology
- **Our hyperparameters**: Appropriate for custom topology networks
- **Methodological consistency**: Fair comparison across topologies
- **Research-ready**: Publication-quality experimental setup
