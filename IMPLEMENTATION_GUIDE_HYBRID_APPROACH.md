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

#### **2.2 W&B Metric Structure**
```python
# Clean, organized metric structure
WANDB_METRIC_STRUCTURE = {
    'config/': {
        'topology_type': 'string',
        'hidden_size': 'int',
        'num_layers': 'int',
        'total_parameters': 'int',
        'task_name': 'string',
        'seed': 'int',
        'reward_scale': 'float',
        'segment_length': 'int',
        'shift_range': 'list'
    },
    
    'training/': {
        'global_step': 'int',
        'episode_return': 'float',        # Scaled
        'raw_episode_return': 'float',    # Raw (for reference)
        'episode_length': 'int',
        'episode_number': 'int'
    },
    
    'continual_learning/': {
        'shift_step': 'int',
        'shift_id': 'int',
        'global_step': 'int'
    },
    
    'ppo/': {
        'update_index': 'int',
        'global_step_end': 'int',
        'mean_episode_return': 'float',   # Scaled
        'mean_raw_episode_return': 'float', # Raw
        'rollout_size': 'int',
        'epochs_per_update': 'int',
        'reward_scale': 'float'
    }
}
```

### **Phase 3: Integration with Existing System (Week 3)**

#### **3.1 Update Main Training Function**
```python
def continual_learning_training(policy_class, topology_type, config, seed, task_name):
    """Enhanced continual learning training with hybrid approach."""
    
    # Update config with hybrid settings
    config.update({
        'n_steps': 800,              # Paper-aligned
        'n_epochs': 5,               # Paper-aligned
        'segment_length': 200,       # Paper-aligned
        'shift_range': [0, 2],       # Paper-aligned
        'episode_cap': 400,          # Paper-aligned
        'reward_scale': 20.0,        # Paper-aligned
        'total_timesteps': 100000    # Our choice
    })
    
    # Create enhanced environment wrapper
    env = gym.make(task_name)
    env = EnhancedContinualLearningWrapper(
        env, 
        segment_length=config['segment_length'],
        shift_range=config['shift_range'],
        reward_scale=config['reward_scale'],
        episode_cap=config['episode_cap']
    )
    
    # Create model with updated PPO config
    model = PPO(
        policy_class,
        env,
        n_steps=config['n_steps'],
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        verbose=1
    )
    
    # Enhanced callback system
    combined_callback = CallbackList([
        SimplifiedCallback(logging_handler=logging_handler, log_freq=100),
        ContinualLearningProgressBarCallback(config['total_timesteps'], task_name, config['segment_length']),
        ShiftLoggingCallback(env, log_interval=200),
        TrainingTerminationCallback(config['total_timesteps']),
        EnhancedLoggingCallback(task_name, topology_type, seed, config['reward_scale'])
    ])
    
    # Train with enhanced logging
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=combined_callback,
        progress_bar=False
    )
    
    return model, env
```

## 📊 **Analysis and Visualization**

### **A) Figure-6 Style Plots**
```python
class HybridFigure6Plotter:
    def __init__(self, results_data):
        self.results = results_data
        
    def create_learning_curves(self, topology, task, seeds):
        """Create Figure-6 style learning curves with hybrid approach."""
        # Gather per-seed traces (using scaled returns)
        seed_traces = {}
        for seed in seeds:
            trace = self._extract_seed_trace(topology, task, seed, metric='episode_return')
            seed_traces[seed] = trace
        
        # Resample to common step grid (every 200 steps for shift boundaries)
        step_grid = np.arange(0, max_timesteps, 200)
        resampled_traces = self._resample_to_grid(seed_traces, step_grid)
        
        # Aggregate across seeds
        mean_returns = []
        std_returns = []
        
        for step in step_grid:
            step_returns = [trace[step] for trace in resampled_traces.values() if step in trace]
            if step_returns:
                mean_returns.append(np.mean(step_returns))
                std_returns.append(np.std(step_returns))
            else:
                mean_returns.append(np.nan)
                std_returns.append(np.nan)
        
        # Create plot with shift boundaries
        self._plot_hybrid_learning_curve(step_grid, mean_returns, std_returns, topology, task)
    
    def _plot_hybrid_learning_curve(self, steps, means, stds, topology, task):
        """Plot learning curve with shift boundaries and hybrid approach."""
        plt.figure(figsize=(14, 8))
        
        # Main learning curve (scaled returns)
        plt.plot(steps, means, 'b-', linewidth=2, label=f'{topology} on {task} (scaled)')
        plt.fill_between(steps, 
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, color='blue')
        
        # Shift boundaries (every 200 steps)
        shift_steps = np.arange(0, max(steps), 200)
        for shift_step in shift_steps:
            plt.axvline(x=shift_step, color='red', linestyle='--', alpha=0.5, label='Shift Boundary' if shift_step == 0 else "")
        
        # Dual Y-axis for scaled vs raw returns
        ax2 = plt.twinx()
        # Convert scaled to raw for reference
        raw_means = np.array(means) / 20.0
        ax2.plot(steps, raw_means, 'g--', linewidth=1, alpha=0.7, label='Raw returns (reference)')
        ax2.set_ylabel('Raw Episode Return', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        plt.xlabel('Environment Steps')
        plt.ylabel('Scaled Episode Return (×20)', color='blue')
        plt.title(f'Hybrid Learning Curve: {topology} on {task}\n(Paper-aligned shifts + Our hyperparameters)')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.show()
```

### **B) Adaptation Metrics**
```python
def calculate_adaptation_metrics(episode_data, shift_boundaries):
    """Calculate adaptation metrics for continual learning analysis."""
    
    adaptation_metrics = {}
    
    for shift_id in range(len(shift_boundaries)):
        # Pre-shift performance (last 1000 steps before shift)
        pre_shift_steps = shift_boundaries[shift_id] - 1000
        pre_shift_episodes = [ep for ep in episode_data 
                             if pre_shift_steps <= ep['step_end'] < shift_boundaries[shift_id]]
        
        # Post-shift performance (first 1000 steps after shift)
        post_shift_steps = shift_boundaries[shift_id] + 1000
        post_shift_episodes = [ep for ep in episode_data 
                              if shift_boundaries[shift_id] <= ep['step_end'] < post_shift_steps]
        
        if pre_shift_episodes and post_shift_episodes:
            pre_mean = np.mean([ep['episode_return'] for ep in pre_shift_episodes])
            post_mean = np.mean([ep['episode_return'] for ep in post_shift_episodes])
            
            adaptation_metrics[f'shift_{shift_id}'] = {
                'immediate_drop': pre_mean - post_mean,
                'recovery_ratio': post_mean / pre_mean if pre_mean > 0 else 0,
                'pre_shift_performance': pre_mean,
                'post_shift_performance': post_mean
            }
    
    return adaptation_metrics
```

## 🔧 **Configuration and Testing**

### **A) Configuration File**
```python
# config/hybrid_continual_learning.py
HYBRID_CONTINUAL_LEARNING_CONFIG = {
    # Paper-aligned settings
    'shift_distribution': 'Uniform[0, 2]',
    'shift_cadence': 200,
    'rollout_length': 800,
    'ppo_epochs': 5,
    'episode_cap': 400,
    'reward_scale': 20.0,
    
    # Our hyperparameters (consistent across topologies)
    'learning_rate': 0.0003,
    'batch_size': 32,
    'n_steps': 800,
    
    # Experimental settings
    'total_timesteps': 100000,
    'segment_length': 200,
    'shift_range': [0, 2]
}
```

### **B) Testing Strategy**
```python
def test_hybrid_implementation():
    """Test the hybrid implementation step by step."""
    
    # Test 1: Environment wrapper
    env = gym.make('CartPole-v1')
    env = EnhancedContinualLearningWrapper(env)
    
    # Test 2: Shift resampling
    for step in range(0, 1000, 200):
        env.step(env.action_space.sample())
        print(f"Step {step}: Shift ID {env.shift_id}")
    
    # Test 3: Reward scaling
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    print(f"Raw reward: {reward}, Scaled reward: {reward * 20.0}")
    
    # Test 4: Episode capping
    while not done and env.episode_step < 400:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
    print(f"Episode ended at step {env.episode_step}")
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

### **Phase 4: Analysis**
- [ ] Implement Figure-6 style plotting
- [ ] Add adaptation metrics
- [ ] Create publication-ready visualizations
- [ ] Validate statistical analysis

## 🚀 **Next Steps**

1. **Start with Phase 1**: Core infrastructure updates
2. **Test thoroughly**: Ensure shifts happen every 200 steps
3. **Validate scaling**: Confirm reward scaling works correctly
4. **Iterate**: Refine based on testing results

This hybrid approach gives us the **best of both worlds**: paper-aligned methodology for distribution shifts while maintaining our existing infrastructure and hyperparameters for topology networks! 🎯
