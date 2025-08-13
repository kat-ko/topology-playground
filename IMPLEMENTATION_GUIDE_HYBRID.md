# Implementation Guide: Hybrid Approach for Topology Comparison

## 🎯 **Overview**

This document provides a **hybrid implementation approach** that combines:
- **Paper-aligned methodology** for noise/distribution shifts
- **Our existing infrastructure** for topology networks and seed management
- **Methodological consistency** across all topology comparisons
- **Flexible logging** for comprehensive analysis

## 🧪 **Core Experimental Design**

### **A) Distribution Shifts (Paper-Aligned)**
```python
# Noise distribution: Uniform[0, 2] per observation dimension
shift_range = [0, 2]

# Shift cadence: Every 200 environment steps (strictly by global step)
segment_length = 200

# Implementation: Apply same offset vector for all steps in segment
if global_step % 200 == 0:
    new_offset = np.random.uniform(0, 2, size=obs_shape)
    # Use this offset for steps [global_step, global_step + 199]
```

### **B) Training Structure (Hybrid)**
```python
# Paper-aligned settings
rollout_length = 800        # 2 × 400-step episodes max
ppo_epochs = 5              # Fixed across all runs

# Our hyperparameters (consistent across topologies)
learning_rate = 0.0003      # Our standard for custom networks
batch_size = 32             # Our standard
n_steps = 800               # Match paper rollout length

# Why this hybrid?
# - 800-step rollouts match paper's training loop pacing
# - Our learning rate is appropriate for custom topology networks
# - Fixed hyperparameters ensure fair topology comparison
```

### **C) Reward Scaling (Paper-Aligned)**
```python
# Apply 20.0 scaling in episode storage (before logging)
reward_scale = 20.0

# Implementation flow:
# 1. Environment returns raw reward (e.g., +1 for CartPole)
# 2. Scale reward: scaled_reward = raw_reward * 20.0
# 3. Store scaled reward in episode buffer
# 4. Log scaled episode return
# 5. Training uses scaled rewards for PPO updates

# Benefits:
# - Matches paper methodology exactly
# - Larger magnitude helps PPO convergence
# - Consistent with Control task benchmarks
```

## 🏗️ **Implementation Phases**

### **Phase 1: Core Infrastructure Updates (Week 1)**

#### **1.1 Update PPO Configuration**
```python
# In create_debug_config() function
def create_debug_config():
    config = {
        # Paper-aligned settings
        'n_steps': 800,              # Was 2048, now 800
        'n_epochs': 5,               # Was 10, now 5
        'batch_size': 32,            # Keep our standard
        
        # Our hyperparameters (consistent across topologies)
        'learning_rate': 0.0003,     # Keep our standard
        'gamma': 0.99,               # Keep our standard
        'gae_lambda': 0.95,          # Keep our standard
        
        # Continual learning settings
        'segment_length': 200,       # Shift every 200 steps
        'shift_range': [0, 2],       # Uniform[0, 2] per dimension
        'episode_cap': 400,          # Max episode length
        'reward_scale': 20.0,        # Paper-aligned scaling
        
        # Experimental settings
        'total_timesteps': 100000,   # Our choice for research goals
    }
    return config
```

#### **1.2 Enhanced Continual Learning Wrapper**
```python
class EnhancedContinualLearningWrapper(gym.Wrapper):
    def __init__(self, env, segment_length=200, shift_range=[0, 2], 
                 reward_scale=20.0, episode_cap=400):
        super().__init__(env)
        self.segment_length = segment_length
        self.shift_range = shift_range
        self.reward_scale = reward_scale
        self.episode_cap = episode_cap
        
        # Global step tracking
        self.global_step = 0
        self.current_shift = np.zeros(env.observation_space.shape)
        self.shift_id = 0
        self.shift_history = []
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_returns = []
        
    def step(self, action):
        # Check for shift boundary
        if self.global_step % self.segment_length == 0:
            self._resample_shift()
        
        # Take environment step
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Apply observation shift (before any scaling)
        shifted_obs = obs + self.current_shift
        
        # Apply reward scaling
        scaled_reward = reward * self.reward_scale
        
        # Update tracking
        self.global_step += 1
        self.episode_step += 1
        self.episode_reward += scaled_reward
        
        # Handle episode termination (cap at 400 steps)
        if done or truncated or self.episode_step >= self.episode_cap:
            self._log_episode()
            self._reset_episode()
        
        return shifted_obs, scaled_reward, done, truncated, info
    
    def _resample_shift(self):
        """Resample observation offset vector every 200 steps."""
        self.current_shift = np.random.uniform(
            self.shift_range[0], 
            self.shift_range[1], 
            size=self.observation_space.shape
        )
        self.shift_id += 1
        
        # Log shift event
        self.shift_history.append({
            'shift_step': self.global_step,
            'shift_id': self.shift_id,
            'offset_vector': self.current_shift.copy()
        })
        
        print(f"🔄 Shift {self.shift_id} at step {self.global_step}")
    
    def _log_episode(self):
        """Log episode completion."""
        self.episode_returns.append({
            'step_end': self.global_step,
            'episode_return': self.episode_reward,
            'episode_length': self.episode_step,
            'shift_id': self.shift_id,
            'raw_episode_return': self.episode_reward / self.reward_scale  # For reference
        })
    
    def _reset_episode(self):
        """Reset episode tracking."""
        self.episode_step = 0
        self.episode_reward = 0.0
```

### **Phase 2: Enhanced Logging System (Week 2)**

#### **2.1 Multi-Granularity Logging**
```python
class EnhancedLoggingCallback(BaseCallback):
    def __init__(self, task_name, topology_type, seed, reward_scale=20.0):
        super().__init__()
        self.task_name = task_name
        self.topology_type = topology_type
        self.seed = seed
        self.reward_scale = reward_scale
        
        # Tracking
        self.update_index = 0
        self.last_update_step = 0
        self.episode_buffer = []
        
    def _on_step(self) -> bool:
        # Per-shift logging (every 200 steps)
        if self.num_timesteps % 200 == 0:
            self._log_shift_event()
        
        # Per-update logging (every 800 steps)
        if self.num_timesteps - self.last_update_step >= 800:
            self._log_update_event()
            self.last_update_step = self.num_timesteps
            self.update_index += 1
        
        return True
    
    def _log_shift_event(self):
        """Log shift boundary events."""
        if wandb.run:
            wandb.log({
                'continual_learning/shift_step': self.num_timesteps,
                'continual_learning/shift_id': self.num_timesteps // 200,
                'continual_learning/global_step': self.num_timesteps
            }, step=self.num_timesteps)
    
    def _log_update_event(self):
        """Log PPO update events."""
        if wandb.run:
            # Calculate mean episode return over last update
            recent_episodes = self._get_recent_episodes(800)
            if recent_episodes:
                mean_return = np.mean([ep['episode_return'] for ep in recent_episodes])
                mean_raw_return = np.mean([ep['raw_episode_return'] for ep in recent_episodes])
                
                wandb.log({
                    'ppo/update_index': self.update_index,
                    'ppo/global_step_end': self.num_timesteps,
                    'ppo/mean_episode_return': mean_return,        # Scaled
                    'ppo/mean_raw_episode_return': mean_raw_return, # Raw (for reference)
                    'ppo/rollout_size': 800,
                    'ppo/epochs_per_update': 5,
                    'ppo/reward_scale': self.reward_scale
                }, step=self.num_timesteps)
    
    def _log_episode_completion(self, episode_data):
        """Log individual episode completion."""
        if wandb.run:
            wandb.log({
                'episodes/step_end': episode_data['step_end'],
                'episodes/episode_return': episode_data['episode_return'],      # Scaled
                'episodes/raw_episode_return': episode_data['raw_episode_return'], # Raw
                'episodes/episode_length': episode_data['episode_length'],
                'episodes/shift_id': episode_data['shift_id'],
                'episodes/seed': self.seed,
                'episodes/task_id': self.task_name,
                'episodes/topology_id': self.topology_type,
                'episodes/reward_scale': self.reward_scale
            }, step=episode_data['step_end'])
    
    def _get_recent_episodes(self, window_steps):
        """Get episodes that ended within the last window_steps."""
        current_step = self.num_timesteps
        return [ep for ep in self.episode_buffer 
                if current_step - ep['step_end'] <= window_steps]
```

## 🎯 **Implementation Checklist**

### **Phase 1: Core Infrastructure**
- [ ] Update PPO configuration (800 steps, 5 epochs)
- [ ] Implement EnhancedContinualLearningWrapper
- [ ] Add reward scaling (20.0) and episode capping (400)
- [ ] Test shift resampling every 200 steps

### **Phase 2: Enhanced Logging**
- [ ] Implement EnhancedLoggingCallback
- [ ] Add multi-granularity logging (per-shift, per-episode, per-update)
- [ ] Update W&B metric structure
- [ ] Test logging at all granularities

### **Phase 3: Integration**
- [ ] Update main training function
- [ ] Integrate with existing seed system
- [ ] Test end-to-end training
- [ ] Validate metric consistency

## 🚀 **Next Steps**

1. **Start with Phase 1**: Core infrastructure updates
2. **Test thoroughly**: Ensure shifts happen every 200 steps
3. **Validate scaling**: Confirm reward scaling works correctly
4. **Iterate**: Refine based on testing results

This hybrid approach gives us the **best of both worlds**: paper-aligned methodology for distribution shifts while maintaining our existing infrastructure and hyperparameters for topology networks! 🎯
