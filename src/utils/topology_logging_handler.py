#!/usr/bin/env python3
"""
Simplified Topology Logging Handler

Minimal logging system that only logs standard training performance metrics.
Removes all custom topology and task-specific logging to keep W&B clean.
"""

import wandb
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Topology abbreviations for consistent naming
TOPOLOGY_ABBREVIATIONS = {
    'small_world': 'SW',
    'modular': 'MOD', 
    'hybrid': 'HYB',
    'fully_connected': 'FC'
}

# Task abbreviations for consistent naming
TASK_ABBREVIATIONS = {
    'LunarLander-v2': 'LL',
    'Acrobot-v1': 'AC', 
    'CartPole-v1': 'CP',
    'MountainCar-v0': 'MC'
}

# ============================================================================
# SIMPLIFIED LOGGING HANDLER
# ============================================================================

class SimplifiedLoggingHandler:
    """
    Minimal logging handler that only logs standard training metrics.
    No custom topology metrics, no task-specific logging, no complex hierarchies.
    """
    
    def __init__(self, config: Dict, topology_type: str, training_type: str = 'triple_task'):
        self.config = config
        self.topology_type = topology_type
        self.training_type = training_type
        self.global_timesteps = 0
        self.current_phase = 1
        self.current_task = None
        
    def initialize_run(self):
        """Initialize the logging handler."""
        print(f"🔧 Simplified logging initialized for {self.topology_type} topology")
        
    def set_task_phase(self, task: str, phase: int):
        """Set current task and phase."""
        self.current_task = task
        self.current_phase = phase
        print(f"🔄 Phase {phase} transition: {task} at global timestep {self.global_timesteps}")
        
    def update_global_timesteps(self, additional_timesteps: int):
        """Update global timestep counter."""
        self.global_timesteps += additional_timesteps
        print(f"📊 Task {self.current_task} completed: {additional_timesteps:,} timesteps")
        print(f"📈 Global timesteps now: {self.global_timesteps:,}")
        
    def log_standard_metrics(self, metrics: Dict):
        """
        Log only standard training metrics to W&B.
        No custom topology or task-specific logging.
        """
        if not wandb.run:
            return
            
        # Only log essential training metrics
        standard_metrics = {}
        
        # Basic training progress
        if 'timesteps' in metrics:
            standard_metrics['train/timesteps'] = metrics['timesteps']
        if 'episodes' in metrics:
            standard_metrics['train/episodes'] = metrics['episodes']
            
        # Standard PPO metrics
        for key in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip', 'learning_rate']:
            if key in metrics:
                standard_metrics[f'train/{key}'] = metrics[key]
                
        # Reward metrics (if available)
        if 'mean_reward' in metrics:
            standard_metrics['train/mean_reward'] = metrics['mean_reward']
        if 'mean_length' in metrics:
            standard_metrics['train/mean_length'] = metrics['mean_length']
            
        # Log with proper step validation
        if standard_metrics:
            safe_step = max(1, self.global_timesteps)
            wandb.log(standard_metrics, step=safe_step)

    def update_run_name(self, model, total_params):
        """Update the run name with actual network capacity, preserving task order and adding num_layers."""
        if wandb.run:
            # Create a comprehensive run name with all information
            topology_abbr = TOPOLOGY_ABBREVIATIONS.get(self.topology_type, self.topology_type.upper())
            hidden_size = self.config.get('hidden_size', 'unknown')
            num_layers = self.config.get('num_layers', 'unknown')
            
            # Include task order for triple-task training
            if self.training_type == 'triple_task':
                run_name = f"{topology_abbr}_C{total_params}_S{hidden_size}_L{num_layers}_CP-AC-LL"
            else:
                run_name = f"{topology_abbr}_C{total_params}_S{hidden_size}_L{num_layers}"
            
            # Update the run name
            wandb.run.name = run_name
            print(f"✅ Updated run name: {run_name}")
            
        return f"{self.topology_type}_C{total_params}"

# ============================================================================
# SIMPLIFIED CALLBACK
# ============================================================================

class SimplifiedCallback(BaseCallback):
    """
    Minimal callback that only logs standard training metrics.
    No custom topology logging, no complex evaluation structures.
    """
    
    def __init__(self, logging_handler, log_freq: int = 1000):
        super().__init__()
        self.logging_handler = logging_handler
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """Log standard metrics every log_freq steps."""
        if self.n_calls % self.log_freq == 0:
            # Collect basic metrics
            metrics = {
                'timesteps': self.num_timesteps,
                'episodes': len(self.episode_rewards)
            }
            
            # Add PPO metrics if available
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = self.model.logger.name_to_value
                for key, value in name_to_value.items():
                    if any(term in key.lower() for term in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip']):
                        metrics[key] = value
                        
            # Add learning rate if available
            if hasattr(self.model, 'lr_schedule'):
                current_lr = self.model.lr_schedule(self.num_timesteps)
                metrics['learning_rate'] = current_lr
                
            # Add reward metrics if available
            if self.episode_rewards:
                recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                recent_lengths = self.episode_lengths[-100:]  # Last 100 episodes
                metrics.update({
                    'mean_reward': np.mean(recent_rewards),
                    'mean_length': np.mean(recent_lengths)
                })
                
            # Log standard metrics
            self.logging_handler.log_standard_metrics(metrics)
            
        return True
        
    def _on_rollout_end(self) -> None:
        """Collect episode rewards and lengths."""
        if hasattr(self.training_env, 'get_episode_rewards'):
            rewards = self.training_env.get_episode_rewards()
            if rewards:
                self.episode_rewards.extend(rewards)
                
        if hasattr(self.training_env, 'get_episode_lengths'):
            lengths = self.training_env.get_episode_lengths()
            if lengths:
                self.episode_lengths.extend(lengths)

    def set_task_phase(self, task: str, phase: int):
        """Set the current task phase for the callback."""
        self.task_name = task
        self.phase = phase

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_logging_handler(config: Dict, topology_type: str, training_type: str = 'triple_task'):
    """Create a simplified logging handler."""
    return SimplifiedLoggingHandler(config, topology_type, training_type)

def create_run_name(config: Dict, topology_type: str, training_type: str = 'triple_task') -> str:
    """Create a simple run name with num_layers for consistency."""
    topology_abbr = TOPOLOGY_ABBREVIATIONS.get(topology_type, topology_type.upper())
    hidden_size = config.get('hidden_size', 'unknown')
    num_layers = config.get('num_layers', 'unknown')
    
    if training_type == 'triple_task':
        return f"{topology_abbr}_S{hidden_size}_L{num_layers}_CP-AC-LL"
    else:
        return f"{topology_abbr}_S{hidden_size}_L{num_layers}"

def create_run_tags(config: Dict, topology_type: str, training_type: str = 'triple_task') -> List[str]:
    """Create simple run tags."""
    tags = [topology_type, training_type]
    
    if 'hidden_size' in config:
        tags.append(f"size_{config['hidden_size']}")
        
    if training_type == 'triple_task':
        tags.extend(['CartPole-v1', 'Acrobot-v1', 'LunarLander-v2'])
        
    return tags
