"""
Task-specific training configuration and convergence monitoring.

This module provides task-specific training parameters and convergence monitoring
to optimize training time per task based on their individual characteristics.
"""

import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================================
# TASK-SPECIFIC TRAINING CONFIGURATIONS
# ============================================================================

TASK_TRAINING_CONFIG = {
    'CartPole-v1': {
        'total_timesteps': 200000,  # Faster convergence - typically solves quickly
        'convergence_threshold': 0.95,  # Success rate threshold
        'min_timesteps': 50000,  # Minimum training time
        'max_timesteps': 300000,  # Maximum training time
        'early_stopping_patience': 10000,  # Timesteps without improvement
        'convergence_window': 3,  # Window for convergence checking (3 evaluations = 60K steps)
        'reward_threshold': 500,  # Reward threshold for convergence (actual solved threshold)
    },
    'Acrobot-v1': {
        'total_timesteps': 800000,  # Slower convergence - more complex task
        'convergence_threshold': 0.8,  # Lower success rate threshold
        'min_timesteps': 200000,  # Longer minimum training time
        'max_timesteps': 1200000,  # Higher maximum training time
        'early_stopping_patience': 50000,  # More patience for improvement
        'convergence_window': 5,  # Larger window for convergence checking (5 evaluations = 100K steps)
        'reward_threshold': -80,  # Reward threshold for convergence (actual solved threshold)
    },
    'MountainCar-v0': {
        'total_timesteps': 600000,  # Medium convergence - moderate complexity
        'convergence_threshold': 0.9,  # Medium success rate threshold
        'min_timesteps': 150000,  # Medium minimum training time
        'max_timesteps': 900000,  # Medium maximum training time
        'early_stopping_patience': 30000,  # Medium patience for improvement
        'convergence_window': 4,  # Medium window for convergence checking (4 evaluations = 80K steps)
        'reward_threshold': -110,  # Reward threshold for convergence
    }
}

# Default configuration for unknown tasks
DEFAULT_TASK_CONFIG = {
    'total_timesteps': 500000,
    'convergence_threshold': 0.9,
    'min_timesteps': 100000,
    'max_timesteps': 1000000,
    'early_stopping_patience': 20000,
    'convergence_window': 1000,
    'reward_threshold': 0,
}


def get_task_training_config(task_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get task-specific training configuration.
    
    Args:
        task_name: Name of the task (e.g., 'CartPole-v1')
        config: General configuration dictionary
        
    Returns:
        Task-specific training configuration
    """
    # Get base task configuration
    task_config = TASK_TRAINING_CONFIG.get(task_name, DEFAULT_TASK_CONFIG).copy()
    
    # Override with sweep config if provided
    if 'task_timesteps' in config and task_name in config['task_timesteps']:
        task_config['total_timesteps'] = config['task_timesteps'][task_name]
    
    # Override with individual parameters if provided
    if 'task_convergence_threshold' in config and task_name in config['task_convergence_threshold']:
        task_config['convergence_threshold'] = config['task_convergence_threshold'][task_name]
    
    if 'task_early_stopping_patience' in config and task_name in config['task_early_stopping_patience']:
        task_config['early_stopping_patience'] = config['task_early_stopping_patience'][task_name]
    
    # Ensure timesteps are within bounds
    task_config['total_timesteps'] = max(
        task_config['min_timesteps'],
        min(task_config['total_timesteps'], task_config['max_timesteps'])
    )
    
    return task_config


def get_task_timesteps(task_name: str, config: Dict[str, Any]) -> int:
    """
    Get the total timesteps for a specific task.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary
        
    Returns:
        Total timesteps for the task
    """
    task_config = get_task_training_config(task_name, config)
    return task_config['total_timesteps']


# ============================================================================
# CONVERGENCE MONITORING CALLBACK
# ============================================================================

class ConvergenceCallback(BaseCallback):
    """
    Callback for monitoring training convergence and early stopping.
    
    This callback tracks performance during training and can trigger early stopping
    when the model has converged or stopped improving.
    """
    
    def __init__(self, task_name: str, task_config: Dict[str, Any], 
                 convergence_window: Optional[int] = None, verbose: int = 0):
        """
        Initialize the convergence callback.
        
        Args:
            task_name: Name of the task being trained
            task_config: Task-specific training configuration
            convergence_window: Window size for convergence checking (overrides config)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.task_name = task_name
        self.task_config = task_config
        self.convergence_window = convergence_window or task_config.get('convergence_window', 1000)
        
        # Performance tracking
        self.recent_rewards = []
        self.best_reward = float('-inf')
        self.steps_without_improvement = 0
        self.should_stop = False
        
        # Convergence metrics
        self.convergence_threshold = task_config.get('convergence_threshold', 0.9)
        self.reward_threshold = task_config.get('reward_threshold', 0)
        self.early_stopping_patience = task_config.get('early_stopping_patience', 20000)
        self.min_timesteps = task_config.get('min_timesteps', 100000)
        
        # Logging
        self.convergence_logged = False
        
    def _on_step(self) -> bool:
        """
        Called on each training step.
        
        Returns:
            False if training should stop, True otherwise
        """
        # Get number of environments safely
        n_envs = getattr(self.model, 'n_envs', 1)
        
        # Track training progress
        if self.num_timesteps >= self.min_timesteps:
            # Check if we should trigger an evaluation
            if not hasattr(self, '_last_eval_step'):
                self._last_eval_step = 0
            
            # Evaluate every 10K steps to check convergence (less frequent to avoid overhead)
            eval_interval = 10000
            if self.num_timesteps - self._last_eval_step >= eval_interval:
                self._last_eval_step = self.num_timesteps
                
                # Trigger evaluation to check convergence
                self._trigger_evaluation()
        
        return not self.should_stop
    
    def _trigger_evaluation(self):
        """Trigger a quick evaluation to check convergence."""
        if self.verbose > 0:
            print(f"📊 {self.task_name}: Checking convergence at {self.num_timesteps:,} timesteps")
        
        # Get target timesteps for this task
        target_timesteps = self.task_config.get('total_timesteps', 200000)
        max_timesteps = self.task_config.get('max_timesteps', 300000)
        
        # Show current state
        if self.verbose > 0:
            print(f"   📋 Target: {target_timesteps:,}, Max: {max_timesteps:,}, Current: {self.num_timesteps:,}")
            if self.recent_rewards:
                recent_mean = np.mean(self.recent_rewards[-5:])  # Last 5 evaluations
                print(f"   📊 Recent rewards: {recent_mean:.2f} (threshold: {self.reward_threshold})")
        
        # Check if we've reached the target timesteps (convergence)
        if self.num_timesteps >= target_timesteps:
            if not self.convergence_logged:
                if self.verbose > 0:
                    print(f"🎯 {self.task_name}: Reached target timesteps at {self.num_timesteps:,}")
                self.convergence_logged = True
                self.should_stop = True
        
        # Check if we've exceeded the maximum timesteps (timeout)
        elif self.num_timesteps >= max_timesteps:
            if self.verbose > 0:
                print(f"⏹️  {self.task_name}: Reached max timesteps at {self.num_timesteps:,}")
            self.should_stop = True
    
    def _check_convergence_via_evaluation(self):
        """Check convergence by triggering an evaluation."""
        if self.verbose > 0:
            print(f"📊 {self.task_name}: Evaluating at {self.num_timesteps} timesteps")
        
        # Get target timesteps for this task
        target_timesteps = self.task_config.get('total_timesteps', 200000)
        max_timesteps = self.task_config.get('max_timesteps', 300000)
        
        # Check if we've reached the target timesteps (convergence)
        if self.num_timesteps >= target_timesteps:
            if not self.convergence_logged:
                if self.verbose > 0:
                    print(f"🎯 {self.task_name}: Converged at {self.num_timesteps} timesteps")
                self.convergence_logged = True
                self.should_stop = True
        
        # Check if we've exceeded the maximum timesteps (timeout)
        elif self.num_timesteps >= max_timesteps:
            if self.verbose > 0:
                print(f"⏹️  {self.task_name}: Timeout at {self.num_timesteps} timesteps")
            self.should_stop = True
    
    def on_evaluation_result(self, eval_result: Dict[str, Any]):
        """
        Called when an evaluation result is available.
        This allows integration with other evaluation callbacks.
        
        Args:
            eval_result: Dictionary containing evaluation metrics
        """
        if 'mean_reward' in eval_result:
            reward = eval_result['mean_reward']
            self.recent_rewards.append(reward)
            
            # Keep only recent rewards
            if len(self.recent_rewards) > self.convergence_window:
                self.recent_rewards.pop(0)
            
            # Check for improvement
            if reward > self.best_reward:
                self.best_reward = reward
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
            
            # Check convergence criteria - robust approach with stability requirements
            if len(self.recent_rewards) >= 1:
                # Check for immediate high performance (but require stability)
                if self._check_convergence(reward):
                    # For immediate convergence, require at least 2 evaluations at threshold
                    if len(self.recent_rewards) >= 2:
                        # Check if previous evaluation was also good
                        prev_reward = self.recent_rewards[-2]
                        if self._check_convergence(prev_reward):
                            if not self.convergence_logged:
                                if self.verbose > 0:
                                    print(f"🎯 {self.task_name}: Converged with stability at {self.num_timesteps} timesteps (recent: {reward:.2f}, prev: {prev_reward:.2f})")
                                self.convergence_logged = True
                                self.should_stop = True
                        else:
                            if self.verbose > 0:
                                print(f"📊 {self.task_name}: High performance but waiting for stability (recent: {reward:.2f}, prev: {prev_reward:.2f})")
                    else:
                        if self.verbose > 0:
                            print(f"📊 {self.task_name}: High performance but waiting for second evaluation (reward: {reward:.2f})")
                
                # Check convergence over window (for stability)
                elif len(self.recent_rewards) >= self.convergence_window:
                    recent_mean = np.mean(self.recent_rewards[-self.convergence_window:])
                    
                    if self._check_convergence(recent_mean):
                        if not self.convergence_logged:
                            if self.verbose > 0:
                                print(f"🎯 {self.task_name}: Converged based on window performance at {self.num_timesteps} timesteps (mean: {recent_mean:.2f})")
                            self.convergence_logged = True
                            self.should_stop = True
                
                # Check early stopping
                elif self.steps_without_improvement >= self.early_stopping_patience:
                    if self.verbose > 0:
                        print(f"⏹️  {self.task_name}: Early stopping at {self.num_timesteps} timesteps")
                    self.should_stop = True
    
    def update_with_evaluation(self, mean_reward: float, success_rate: float = None):
        """
        Update convergence monitoring with evaluation results.
        This method can be called from training scripts when evaluation is performed.
        
        Args:
            mean_reward: Mean reward from evaluation
            success_rate: Success rate from evaluation (optional)
        """
        eval_result = {'mean_reward': mean_reward}
        if success_rate is not None:
            eval_result['success_rate'] = success_rate
        
        self.on_evaluation_result(eval_result)
    
    def _check_convergence(self, recent_mean: float) -> bool:
        """
        Check if the model has converged based on recent performance.
        
        Args:
            recent_mean: Mean reward over recent window
            
        Returns:
            True if converged, False otherwise
        """
        # Check reward threshold (primary criterion) - handle both positive and negative thresholds
        if self.reward_threshold != 0:  # Only check if threshold is set
            if self.reward_threshold > 0:
                # For positive thresholds (e.g., CartPole-v1: 450)
                if recent_mean >= self.reward_threshold:
                    return True
            else:
                # For negative thresholds (e.g., Acrobot-v1: -100, MountainCar-v0: -110)
                if recent_mean >= self.reward_threshold:
                    return True
        
        # Check stability (low variance in recent performance) - secondary criterion
        # Only use stability if we have enough evaluations and performance is reasonable
        if len(self.recent_rewards) >= self.convergence_window:
            recent_std = np.std(self.recent_rewards[-self.convergence_window:])
            # More conservative stability check: require very low variance
            if recent_std < 0.05 * abs(recent_mean):  # Very low variance (5% instead of 10%)
                # Additional safety check: don't converge on poor performance
                if self.reward_threshold > 0 and recent_mean > 0.5 * self.reward_threshold:
                    return True
                elif self.reward_threshold < 0 and recent_mean > 0.5 * self.reward_threshold:
                    # For negative thresholds, we want recent_mean to be better (less negative) than 50% of threshold
                    return True
        
        return False
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            final_reward = np.mean(self.recent_rewards[-self.convergence_window:]) if self.recent_rewards else 0
            print(f"🏁 {self.task_name}: Training completed at {self.num_timesteps} timesteps "
                  f"(final reward: {final_reward:.2f})")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_convergence_callback(task_name: str, config: Dict[str, Any], 
                               convergence_window: Optional[int] = None, verbose: int = 0) -> ConvergenceCallback:
    """
    Create a convergence callback for a specific task.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary
        convergence_window: Optional window size override
        
    Returns:
        Configured convergence callback
    """
    task_config = get_task_training_config(task_name, config)
    return ConvergenceCallback(task_name, task_config, convergence_window, verbose)


def get_all_task_timesteps(config: Dict[str, Any]) -> Dict[str, int]:
    """
    Get timesteps for all supported tasks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping task names to their timesteps
    """
    return {task: get_task_timesteps(task, config) for task in TASK_TRAINING_CONFIG.keys()}


def validate_task_config(task_name: str, config: Dict[str, Any]) -> bool:
    """
    Validate that a task configuration is valid.
    
    Args:
        task_name: Name of the task
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    try:
        task_config = get_task_training_config(task_name, config)
        required_keys = ['total_timesteps', 'min_timesteps', 'max_timesteps']
        
        for key in required_keys:
            if key not in task_config:
                return False
            if not isinstance(task_config[key], (int, float)) or task_config[key] <= 0:
                return False
        
        # Check that min <= total <= max
        if not (task_config['min_timesteps'] <= task_config['total_timesteps'] <= task_config['max_timesteps']):
            return False
        
        return True
    except Exception:
        return False 