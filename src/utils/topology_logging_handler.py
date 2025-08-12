#!/usr/bin/env python3
"""
Topology Logging Handler

Centralized logging system for topology network training experiments.
Handles run naming, tagging, metrics logging, and table creation for:
- Individual runs
- Batch runs  
- W&B sweeps

Provides consistent hierarchical logging structure and easy data analysis.
"""

import wandb
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
import time
from datetime import datetime

# 🚨 COMPREHENSIVE W&B STEP VALIDATION: Global wrapper to catch all step 0 calls
import wandb

# Store original wandb.log function
_original_wandb_log = wandb.log

def _validated_wandb_log(*args, **kwargs):
    """
    Wrapper around wandb.log that validates all steps before logging.
    This prevents step 0 warnings by ensuring all steps are valid.
    """
    # Extract step from kwargs
    step = kwargs.get('step', None)
    
    if step is not None and step <= 0:
        # 🚨 CRITICAL: Step 0 detected - debug and fix it
        import traceback
        
        print(f"🚨 W&B Step 0 Detected: step={step}")
        print(f"🚨 Call stack:")
        traceback.print_stack()
        
        if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step > 0:
            # Use W&B's internal step
            fixed_step = wandb.run.step
            print(f"🚨 W&B Step Fix: {step} → {fixed_step} (using W&B internal step)")
        else:
            # Use fallback step
            fixed_step = 1
            print(f"🚨 W&B Step Fix: {step} → {fixed_step} (using fallback)")
        
        # Update kwargs with fixed step
        kwargs['step'] = fixed_step
    
    # Call original wandb.log with potentially fixed step
    return _original_wandb_log(*args, **kwargs)

# Replace wandb.log with our validated version
wandb.log = _validated_wandb_log

print("🔧 W&B Step Validation: Global wrapper installed to prevent step 0 warnings")

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

# Hierarchical logging paths for organized W&B structure
LOGGING_PATHS = {
    'train': {
        'global': 'train/global',
        'task_orders': 'train/task_order',
        'phases': 'train/phases'
    },
    'network': {
        'global': 'network/global',
        'architecture': 'network/architecture',
        'capacity': 'network/capacity'
    },
    'rollout': {
        'global': 'rollout/global',
        'task_orders': 'rollout/task_order'
    },
    'learning_progression': {
        'global': 'learning_progression/global',
        'task_orders': 'learning_progression/task_order'
    },
    'cross_task_comparison': 'cross_task_comparison',
    'topology_comparison': 'topology_comparison',
    'tables': 'tables'
}

# ============================================================================
# RUN NAMING AND TAGGING
# ============================================================================

class RunNamingManager:
    """Manages run naming and tagging for consistent identification."""
    
    @staticmethod
    def create_initial_run_name(config: Dict, topology_type: str, training_type: str = 'triple_task') -> str:
        """Create initial run name with available config values."""
        
        topology_abbrev = TOPOLOGY_ABBREVIATIONS.get(topology_type, topology_type.upper())
        
        # Get initial size from config
        initial_size = config.get('hidden_size', 'unknown')
        
        # Build initial name parts
        name_parts = [topology_abbrev]
        
        # Add placeholder capacity (will be updated later)
        name_parts.append("C?")
        
        # Add initial size
        name_parts.append(f"S{initial_size}")
        
        # Add task sequence (in correct order)
        if training_type == 'triple_task':
            task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
            tasks = task_order.split('_')
            task_abbrevs = [TASK_ABBREVIATIONS.get(task, task) for task in tasks]
            name_parts.append("-".join(task_abbrevs))
        
        return "_".join(name_parts)
    
    @staticmethod
    def create_final_run_name(config: Dict, topology_type: str, training_type: str, 
                            model: Any, total_params: int) -> str:
        """Create final run name with ACTUAL capacity and size from the model."""
        
        topology_abbrev = TOPOLOGY_ABBREVIATIONS.get(topology_type, topology_type.upper())
        
        # Determine capacity for run name
        if 'target_capacity' in config:
            # Fixed capacity run - show target capacity in name
            capacity_for_name = config['target_capacity']
        else:
            # Fixed size run - show actual capacity in name
            capacity_for_name = total_params if total_params is not None else '?'
        
        # Get ACTUAL size from the model for run name
        actual_size = 'unknown'
        if model is not None and hasattr(model, 'policy'):
            try:
                policy = model.policy
                
                # Get actual hidden size from the policy directly
                if hasattr(policy, 'hidden_size'):
                    actual_size = policy.hidden_size
                    
            except Exception as e:
                print(f"   ⚠️  Could not get actual size from model: {e}")
                # Fallback to config values
                actual_size = config.get('hidden_size', 'unknown')
        
        # Build name parts
        name_parts = [topology_abbrev]
        
        # Add capacity (target for fixed-capacity, actual for fixed-size)
        name_parts.append(f"C{capacity_for_name}")
        
        # Add ACTUAL size
        name_parts.append(f"S{actual_size}")
        
        # Add task sequence (in correct order)
        if training_type == 'triple_task':
            task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
            tasks = task_order.split('_')
            task_abbrevs = [TASK_ABBREVIATIONS.get(task, task) for task in tasks]
            name_parts.append("-".join(task_abbrevs))
        
        return "_".join(name_parts)
    
    @staticmethod
    def create_run_tags(config: Dict, topology_type: str, training_type: str, 
                       model: Any = None, total_params: int = None) -> List[str]:
        """Create enhanced tags for easy filtering and organization with actual capacity tracking."""
        
        # Primary tags
        tags = [
            topology_type,
            training_type,
            "normalized_metrics"
        ]
        
        # Capacity and size tags
        if 'target_capacity' in config:
            tags.extend([
                "fixed_capacity",
                f"target_capacity_{config.get('target_capacity')}",
                "capacity_matched"
            ])
            
            # Add actual capacity tag if available
            if model is not None and total_params is not None:
                tags.extend([
                    f"actual_capacity_{total_params}",
                    f"capacity_{total_params}",  # General capacity tag for easy filtering
                    "capacity_achieved"
                ])
                
        elif 'hidden_size' in config:
            tags.extend([
                "fixed_size", 
                f"size_{config.get('hidden_size')}",
                "size_matched"
            ])
            
            # Add actual capacity tag if available
            if model is not None and total_params is not None:
                tags.extend([
                    f"actual_capacity_{total_params}",
                    f"capacity_{total_params}",  # General capacity tag for easy filtering
                    "capacity_achieved"
                ])
        
        # Add general capacity category tags for easy filtering
        if model is not None and total_params is not None:
            # Categorize capacity into ranges for easy filtering
            if total_params < 1000:
                tags.append("capacity_small")
            elif total_params < 5000:
                tags.append("capacity_medium")
            elif total_params < 10000:
                tags.append("capacity_large")
            else:
                tags.append("capacity_xlarge")
            
            # Add exact capacity as a tag for precise filtering
            tags.append(f"capacity_exact_{total_params}")
        
        # Task tags
        if training_type == 'triple_task':
            task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
            tasks = task_order.split('_')
            tags.extend(tasks)
            
            # Add task order tag for easy filtering
            tags.append(f"order_{task_order.replace('_', '-')}")
        
        # Add sweep type tag
        if 'target_capacity' in config:
            tags.append("sweep_fixed_capacity")
        else:
            tags.append("sweep_fixed_size")
        
        return tags

# ============================================================================
# METRICS LOGGING
# ============================================================================

class MetricsLogger:
    """Handles all metrics logging with consistent hierarchical structure."""
    
    @staticmethod
    def log_training_metrics(step: int, metrics: Dict, task_order: str = None, 
                           current_task: str = None, current_phase: int = 0) -> None:
        """Log training metrics with hierarchical organization."""
        
        # Global training metrics
        global_metrics = {
            "timesteps": metrics.get('timesteps', 0),
            "episodes": metrics.get('episodes', 0),
            "phase": current_phase,
            "current_task": current_task,
            "mean_reward": metrics.get('mean_reward', 0),
            "mean_length": metrics.get('mean_length', 0),
            "success_rate": metrics.get('success_rate', 0),
            "completion_percentage": metrics.get('completion_percentage', 0),
            "training_progress": metrics.get('training_progress', 0),
        }
        
        # Add PPO-specific metrics
        for key in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip', 'learning_rate']:
            if key in metrics:
                global_metrics[key] = metrics[key]
        
        # Log global metrics
        wandb.log({LOGGING_PATHS['train']['global']: global_metrics}, step=step)
        
        # Log task-order specific metrics if available
        if task_order and current_task:
            task_order_path = f"{LOGGING_PATHS['train']['task_orders']}/{task_order}/phase_{current_phase}_{current_task}"
            task_metrics = global_metrics.copy()
            task_metrics.update({
                "task_order": task_order,
                "phase": current_phase,
                "task": current_task
            })
            wandb.log({task_order_path: task_metrics}, step=step)
    
    @staticmethod
    def log_rollout_metrics(step: int, metrics: Dict, task_order: str = None) -> None:
        """Log rollout metrics with hierarchical organization."""
        
        # Global rollout metrics
        global_metrics = {
            "timesteps": metrics.get('timesteps', 0),
            "rollout_count": metrics.get('rollout_count', 0),
            "episode_rewards": metrics.get('episode_rewards', []),
            "episode_lengths": metrics.get('episode_lengths', []),
            "mean_reward": metrics.get('mean_reward', 0),
            "mean_length": metrics.get('mean_length', 0),
        }
        
        # Log global metrics
        wandb.log({LOGGING_PATHS['rollout']['global']: global_metrics}, step=step)
        
        # Log task-order specific metrics if available
        if task_order:
            task_order_path = f"{LOGGING_PATHS['rollout']['task_orders']}/{task_order}"
            task_metrics = global_metrics.copy()
            task_metrics.update({"task_order": task_order})
            wandb.log({task_order_path: task_metrics}, step=step)
    
    @staticmethod
    def log_network_metrics(step: int, metrics: Dict, task_order: str = None) -> None:
        """Log network architecture and capacity metrics."""
        
        # Global network metrics
        global_metrics = {
            "topology_type": metrics.get('topology_type', 'unknown'),
            "hidden_size": metrics.get('hidden_size', 0),
            "num_layers": metrics.get('num_layers', 0),
            "total_params": metrics.get('total_params', 0),
            "actor_params": metrics.get('actor_params', 0),
            "critic_params": metrics.get('critic_params', 0),
            "capacity_match_ratio": metrics.get('capacity_match_ratio', 0),
            "capacity_difference": metrics.get('capacity_difference', 0),
        }
        
        # Log global metrics
        wandb.log({LOGGING_PATHS['network']['global']: global_metrics}, step=step)
        
        # Log capacity-specific metrics
        capacity_metrics = {
            "target_capacity": metrics.get('target_capacity', 0),
            "actual_capacity": metrics.get('total_params', 0),
            "capacity_match_ratio": metrics.get('capacity_match_ratio', 0),
            "capacity_difference": metrics.get('capacity_difference', 0),
        }
        wandb.log({LOGGING_PATHS['network']['capacity']: capacity_metrics}, step=step)
    
    @staticmethod
    def log_learning_progression(step: int, metrics: Dict, task_order: str = None, 
                               current_task: str = None, current_phase: int = 0) -> None:
        """Log learning progression metrics for sequential training analysis."""
        
        # Global learning progression
        global_metrics = {
            "timesteps": metrics.get('timesteps', 0),
            "phase": current_phase,
            "current_task": current_task,
            "mean_reward": metrics.get('mean_reward', 0),
            "success_rate": metrics.get('success_rate', 0),
            "completion_percentage": metrics.get('completion_percentage', 0),
        }
        
        # Log global metrics
        wandb.log({LOGGING_PATHS['learning_progression']['global']: global_metrics}, step=step)
        
        # Log task-order specific metrics if available
        if task_order and current_task:
            task_order_path = f"{LOGGING_PATHS['learning_progression']['task_orders']}/{task_order}/phase_{current_phase}_{current_task}"
            task_metrics = global_metrics.copy()
            task_metrics.update({
                "task_order": task_order,
                "phase": current_phase,
                "task": current_task
            })
            wandb.log({task_order_path: task_metrics}, step=step)
    
    @staticmethod
    def log_cross_task_comparison(step: int, metrics: Dict, task: str) -> None:
        """Log cross-task comparison metrics."""
        
        task_metrics = {
            "task": task,
            "mean_reward": metrics.get('mean_reward', 0),
            "std_reward": metrics.get('std_reward', 0),
            "success_rate": metrics.get('success_rate', 0),
            "completion_percentage": metrics.get('completion_percentage', 0),
            "episode_count": metrics.get('episode_count', 0),
        }
        
        wandb.log({f"{LOGGING_PATHS['cross_task_comparison']}/{task}": task_metrics}, step=step)

# ============================================================================
# TABLE CREATION
# ============================================================================

class TableCreator:
    """Creates and logs W&B tables for structured data analysis."""
    
    @staticmethod
    def create_training_summary_table(training_results: Dict, topology_type: str, task_order: str) -> wandb.Table:
        """Create a training summary table."""
        try:
            training_table = wandb.Table(columns=["Metric", "Value", "Description"])
            
            # Training statistics
            training_table.add_data("Topology Type", topology_type, "Network topology used")
            training_table.add_data("Task Order", task_order, "Sequence of tasks trained")
            training_table.add_data("Total Episodes", str(training_results.get('total_episodes', 0)), "Total training episodes")
            training_table.add_data("Total Steps", str(training_results.get('total_steps', 0)), "Total training steps")
            training_table.add_data("Training Time", f"{training_results.get('training_time', 0):.2f}s", "Total training duration")
            
            # Performance metrics
            if 'episode_rewards' in training_results and training_results['episode_rewards']:
                recent_rewards = training_results['episode_rewards'][-100:]  # Last 100 episodes
                training_table.add_data("Final Mean Reward", f"{np.mean(recent_rewards):.2f}", "Average reward of last 100 episodes")
                training_table.add_data("Final Std Reward", f"{np.std(recent_rewards):.2f}", "Standard deviation of last 100 episodes")
            
            if 'episode_lengths' in training_results and training_results['episode_lengths']:
                recent_lengths = training_results['episode_lengths'][-100:]  # Last 100 episodes
                training_table.add_data("Final Mean Length", f"{np.mean(recent_lengths):.1f}", "Average episode length of last 100 episodes")
            
            # Success metrics
            training_table.add_data("Final Success Rate", f"{training_results.get('success_rate', 0):.1%}", "Final success rate")
            training_table.add_data("Final Completion %", f"{training_results.get('completion_percentage', 0):.1%}", "Final completion percentage")
            
            return training_table
            
        except Exception as e:
            print(f"⚠️ Error creating training summary table: {e}")
            return None
    
    @staticmethod
    def create_network_architecture_table(model: Any, topology_type: str, hidden_size: int, 
                                        num_layers: int, config: Dict = None) -> wandb.Table:
        """Create a network architecture details table with capacity tracking."""
        try:
            network_table = wandb.Table(columns=["Component", "Value", "Description"])
            
            # Basic network info
            network_table.add_data("Topology Type", topology_type, "Network topology architecture")
            network_table.add_data("Hidden Size", str(hidden_size), "Number of hidden units per layer")
            network_table.add_data("Number of Layers", str(num_layers), "Number of hidden layers")
            
            # Capacity information
            if config:
                if 'target_capacity' in config:
                    network_table.add_data("Target Capacity", f"{config['target_capacity']:,}", "Intended total parameters")
                    network_table.add_data("Sweep Type", "Fixed Capacity", "Capacity-matched sweep")
                else:
                    network_table.add_data("Sweep Type", "Fixed Size", "Size-fixed sweep")
            
            # Parameter counts
            if hasattr(model, 'policy'):
                actor_params = model.policy._get_topology_params(model.policy.actor_topology)
                critic_params = model.policy._get_topology_params(model.policy.critic_topology)
                
                if isinstance(actor_params, (int, float)) and isinstance(critic_params, (int, float)):
                    total_params = actor_params + critic_params
                    network_table.add_data("Actual Total Parameters", f"{total_params:,}", "Total trainable parameters achieved")
                    network_table.add_data("Actor Parameters", f"{actor_params:,}", "Actor network parameters")
                    network_table.add_data("Critic Parameters", f"{critic_params:,}", "Critic network parameters")
                    network_table.add_data("Actor-Critic Ratio", f"{actor_params/critic_params:.2f}", "Ratio of actor to critic parameters")
                    network_table.add_data("Parameter Efficiency", f"{total_params/hidden_size:.1f}", "Parameters per hidden unit")
                    
                    # Capacity matching analysis
                    if config and 'target_capacity' in config:
                        target_cap = config['target_capacity']
                        capacity_ratio = total_params / target_cap
                        network_table.add_data("Capacity Match Ratio", f"{capacity_ratio:.3f}", "Actual/Target capacity ratio")
                        network_table.add_data("Capacity Difference", f"{total_params - target_cap:,}", "Actual - Target parameters")
                        
                elif isinstance(actor_params, dict) and isinstance(critic_params, dict):
                    actor_size = actor_params.get('size', 0)
                    critic_size = critic_params.get('size', 0)
                    total_params = actor_size + critic_size
                    network_table.add_data("Actual Total Parameters", f"{total_params:,}", "Total trainable parameters achieved")
                    network_table.add_data("Actor Parameters", f"{actor_size:,}", "Actor network parameters")
                    network_table.add_data("Critic Parameters", f"{critic_size:,}", "Critic network parameters")
                    if critic_size > 0:
                        network_table.add_data("Actor-Critic Ratio", f"{actor_size/critic_size:.2f}", "Ratio of actor to critic parameters")
                    network_table.add_data("Parameter Efficiency", f"{total_params/hidden_size:.1f}", "Parameters per hidden unit")
                    
                    # Capacity matching analysis
                    if config and 'target_capacity' in config:
                        target_cap = config['target_capacity']
                        capacity_ratio = total_params / target_cap
                        network_table.add_data("Capacity Match Ratio", f"{capacity_ratio:.3f}", "Actual/Target capacity ratio")
                        network_table.add_data("Capacity Difference", f"{total_params - target_cap:,}", "Actual - Target parameters")
            
            return network_table
            
        except Exception as e:
            print(f"⚠️ Error creating network architecture table: {e}")
            return None
    
    @staticmethod
    def create_phase_results_table(phase_results: Dict, topology_type: str, task_order: str) -> wandb.Table:
        """Create a phase results table for triple-task training."""
        try:
            phase_table = wandb.Table(columns=["Phase", "Task", "Mean Reward", "Std Reward", "Success Rate", "Completion %", "Description"])
            
            tasks = task_order.split('_')
            
            for phase_idx, task in enumerate(tasks):
                phase_num = phase_idx + 1
                
                # Get results for this phase and task
                phase_key = f'phase{phase_num}'
                if task in phase_results:
                    results = phase_results[task]
                    phase_table.add_data(
                        f"Phase {phase_num}",
                        task,
                        f"{results.get('mean_reward', 0):.2f}",
                        f"{results.get('std_reward', 0):.2f}",
                        f"{results.get('success_rate', 0):.1%}",
                        f"{results.get('completion_percentage', 0):.1%}",
                        f"Performance on {task} after training phase {phase_num}"
                    )
                else:
                    phase_table.add_data(
                        f"Phase {phase_num}",
                        task,
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        f"No results available for {task} in phase {phase_num}"
                    )
            
            return phase_table
            
        except Exception as e:
            print(f"⚠️ Error creating phase results table: {e}")
            return None
    
    @staticmethod
    def create_transfer_learning_table(transfer_metrics: Dict, topology_type: str, task_order: str) -> wandb.Table:
        """Create a transfer learning summary table."""
        try:
            transfer_table = wandb.Table(columns=["Transfer Type", "From Task", "To Task", "Transfer Score", "Description"])
            
            tasks = task_order.split('_')
            
            # Forward transfer metrics
            for i in range(len(tasks) - 1):
                from_task = tasks[i]
                to_task = tasks[i + 1]
                transfer_key = f'forward_transfer_{from_task}_to_{to_task}'
                transfer_score = transfer_metrics.get(transfer_key, 0)
                
                transfer_table.add_data(
                    "Forward Transfer",
                    from_task,
                    to_task,
                    f"{transfer_score:.3f}",
                    f"Transfer from {from_task} to {to_task}"
                )
            
            # Backward transfer (retention) metrics
            for i in range(len(tasks) - 1):
                task = tasks[i]
                retention_key = f'retention_{task}_after_{tasks[i+1]}'
                retention_score = transfer_metrics.get(retention_key, 0)
                
                transfer_table.add_data(
                    "Retention",
                    task,
                    f"After {tasks[i+1]}",
                    f"{retention_score:.3f}",
                    f"Retention of {task} after training {tasks[i+1]}"
                )
            
            # Overall transfer metrics
            if 'overall_transfer_score' in transfer_metrics:
                transfer_table.add_data(
                    "Overall Transfer",
                    "All Tasks",
                    "All Tasks",
                    f"{transfer_metrics['overall_transfer_score']:.3f}",
                    "Overall transfer learning performance"
                )
            
            return transfer_table
            
        except Exception as e:
            print(f"⚠️ Error creating transfer learning table: {e}")
            return None

# ============================================================================
# MAIN LOGGING HANDLER
# ============================================================================

class TopologyLoggingHandler:
    """
    Centralized logging handler for topology training experiments.
    
    Manages:
    - Run naming and tagging
    - Metric logging with proper hierarchical paths
    - Table creation and logging
    - Timestep management (global vs local)
    """
    
    def __init__(self, config: Dict, topology_type: str, training_type: str = 'triple_task'):
        """
        Initialize the topology logging handler.
        
        Args:
            config: Configuration dictionary
            topology_type: Type of topology (small_world, modular, hybrid, fully_connected)
            training_type: Type of training (triple_task, double_task, single_task)
        """
        self.config = config
        self.topology_type = topology_type
        self.training_type = training_type
        
        # Extract task order from config
        self.task_order = config.get('task_order', 'CartPole-v1_Acrobot-v1_LunarLander-v2')
        
        # Initialize components
        self.run_naming_manager = RunNamingManager()
        self.metrics_logger = MetricsLogger()
        self.table_creator = TableCreator()
        
        # 🚨 CRITICAL FIX: Proper timestep management
        self.global_timesteps = 0          # Continuous across all tasks (never resets)
        self.task_start_timesteps = []     # Track global timestep when each task starts
        self.current_task_local_step = 0   # Local step within current task (resets per task)
        self.task_durations = []           # Track actual duration of each task
        
        # Current task and phase tracking
        self.current_task = None
        self.current_phase = 0
        
        # Data storage for tables
        self.training_results = {}
        self.phase_results = {}
        self.transfer_metrics = {}
        
        # 🚨 NEW: Logging frequency control to reduce verbose output
        self.log_freq = config.get('log_freq', 1000)  # Log every 1000 steps by default
        self.last_logged_step = 0  # Track when we last logged to avoid spam
        
        # 🚨 NEW: Model reference for SB3 timestep access
        self.model = None  # Will be set during update_run_name
    
    def _ensure_valid_step(self, step: int) -> int:
        """
        Ensure step is valid for W&B logging.
        
        Args:
            step (int): Input step value
            
        Returns:
            int: Valid step value (guaranteed > 0)
        """
        # 🚨 CRITICAL: W&B requires steps > 0 and monotonically increasing
        if step <= 0:
            # Use W&B's internal step if available, otherwise use fallback
            if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step > 0:
                valid_step = wandb.run.step
                print(f"⚠️  Step validation: {step} → {valid_step} (using W&B internal step)")
            else:
                # Fallback: use global timesteps + 1 to ensure > 0
                valid_step = max(1, self.global_timesteps + 1)
                print(f"⚠️  Step validation: {step} → {valid_step} (using fallback)")
            return valid_step
        
        return step
    
    def _get_wandb_step(self, fallback_step: int = None) -> int:
        """
        Get the current W&B step, with fallback validation.
        
        Args:
            fallback_step (int): Fallback step if W&B step unavailable
            
        Returns:
            int: Valid W&B step
        """
        # 🚨 CRITICAL: Primary approach - use W&B's internal step
        if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step > 0:
            wb_step = wandb.run.step
            return wb_step
        
        # 🚨 CRITICAL: Fallback approach - validate and use fallback
        if fallback_step is None:
            fallback_step = max(1, self.global_timesteps + 1)
        
        # Ensure fallback step is valid
        validated_step = self._ensure_valid_step(fallback_step)
        return validated_step
    
    def initialize_run(self, model: Any = None, total_params: int = None) -> str:
        """Initialize the W&B run with proper naming and tagging."""
        
        # Create initial run name
        initial_name = self.run_naming_manager.create_initial_run_name(
            self.config, self.topology_type, self.training_type
        )
        
        # Create run tags
        tags = self.run_naming_manager.create_run_tags(
            self.config, self.topology_type, self.training_type, model, total_params
        )
        
        # Initialize W&B run
        wandb.init(
            project="topologies--triple-task-training",
            name=initial_name,
            tags=tags,
            config=self.config
        )
        
        return initial_name
    
    def update_run_name(self, model: Any, total_params: int) -> str:
        """Update the run name with actual model parameters."""
        
        # Store model reference for timestep access
        self.model = model
        
        final_name = self.run_naming_manager.create_final_run_name(
            self.config, self.topology_type, self.training_type, model, total_params
        )
        
        # Update W&B run name
        wandb.run.name = final_name
        print(f"✅ Updated run name: {final_name}")
        
        return final_name
    
    def set_task_phase(self, task_name: str, phase_number: int) -> None:
        """
        Set the current task phase for sequential training tracking.
        
        This is called when switching between tasks and is crucial for proper timestep management.
        """
        # Record the global timestep when this task starts
        self.task_start_timesteps.append(self.global_timesteps)
        
        # Reset local step counter for the new task
        self.current_task_local_step = 0
        
        # Update phase and task info
        self.current_phase = phase_number
        self.current_task = task_name
        
        # 🚨 CRITICAL FIX: Use SB3 timesteps for phase transition logging
        # This ensures monotonically increasing steps across task switches
        sb3_step = None
        if hasattr(self, 'model') and self.model is not None:
            sb3_step = self.model.num_timesteps
        else:
            # Fallback: use current global timesteps
            sb3_step = self.global_timesteps
        
        # Log phase transition with SB3 step for proper alignment
        wandb.log({
            'train/global/phase': phase_number,
            'train/global/current_task': task_name,
            'train/global/task_start_timestep': self.global_timesteps,
            'train/global/phase_transition': True,
        }, step=sb3_step)
        
        print(f"🔄 Phase {phase_number} transition: {task_name} at global timestep {self.global_timesteps:,}")
    
    def log_training_step(self, local_step: int, metrics: Dict) -> None:
        """Log training metrics at each step with enhanced step validation."""
        # Skip logging if not at log frequency
        if local_step % self.log_freq != 0:
            return
        
        # 🚨 ENHANCED STEP VALIDATION: Use new validation system
        wb_step = self._get_wandb_step(fallback_step=local_step)
        
        # Log training metrics with validated step
        self.metrics_logger.log_training_metrics(
            wb_step, metrics, self.task_order, self.current_task, self.current_phase
        )
        
        # Log learning progression with validated step
        self.metrics_logger.log_learning_progression(
            wb_step, metrics, self.task_order, self.current_task, self.current_phase
        )
        
        self.last_logged_step = local_step
    
    def log_rollout_end(self, step: int, metrics: Dict) -> None:
        """Log metrics at the end of each rollout with enhanced step validation."""
        # 🚨 ENHANCED STEP VALIDATION: Use new validation system
        wb_step = self._get_wandb_step(fallback_step=step)
        
        # Log rollout metrics with validated step
        self.metrics_logger.log_rollout_metrics(
            wb_step, metrics, self.task_order, self.current_task, self.current_phase
        )
    
    def update_global_timesteps(self, task_duration: int) -> None:
        """
        Update global timesteps after completing a task.
        
        This is called when a task finishes to accumulate the global timestep counter.
        
        Args:
            task_duration: The actual number of timesteps used for this task
        """
        # Store the actual duration of this task
        self.task_durations.append(task_duration)
        
        # Update global timesteps to include this completed task
        self.global_timesteps += task_duration
        
        print(f"📊 Task {self.current_task} completed: {task_duration:,} timesteps")
        print(f"📈 Global timesteps now: {self.global_timesteps:,}")
        
        # 🚨 CRITICAL FIX: Use SB3 timesteps for task completion logging
        # This ensures monotonically increasing steps across task switches
        sb3_step = None
        if hasattr(self, 'model') and self.model is not None:
            sb3_step = self.model.num_timesteps
        else:
            # Fallback: use updated global timesteps
            sb3_step = self.global_timesteps
        
        # Log task completion metrics with SB3 step for proper alignment
        wandb.log({
            'train/global/task_completed': self.current_task,
            'train/global/task_duration': task_duration,
            'train/global/total_timesteps': self.global_timesteps,
            'train/global/task_completion': True,
        }, step=sb3_step)
    
    def get_timestep_info(self) -> Dict[str, int]:
        """
        Get current timestep information for debugging and analysis.
        
        Returns:
            Dictionary with current timestep state
        """
        return {
            'global_timesteps': self.global_timesteps,
            'current_task_local_step': self.current_task_local_step,
            'task_start_timesteps': self.task_start_timesteps.copy(),
            'task_durations': self.task_durations.copy(),
            'current_phase': self.current_phase,
            'current_task': self.current_task,
        }
    
    def log_network_info(self, step: int, model: Any, hidden_size: int, num_layers: int) -> None:
        """Log network architecture and capacity information with W&B-native step management."""
        # 🚨 W&B-NATIVE STEP MANAGEMENT: Use W&B's internal step counter
        # This ensures perfect synchronization and eliminates step warnings
        if wandb.run and hasattr(wandb.run, 'step'):
            # Use W&B's own step counter for perfect alignment
            wb_step = wandb.run.step
            print(f"🔄 W&B-Native: Using W&B step {wb_step} for network info")
        else:
            # Fallback to provided step if W&B step not available
            wb_step = max(1, step)  # Ensure step > 0
        
        # Calculate network capacity
        try:
            policy = model.policy
            total_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
            
            # Get individual network info
            actor_params = policy._get_topology_params(policy.actor_topology)
            critic_params = policy._get_topology_params(policy.critic_topology)
            
            # Create network metrics
            network_metrics = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'total_parameters': total_params,
                'actor_parameters': actor_params,
                'critic_parameters': critic_params,
                'topology_type': self.topology_type,
                'activation': 'relu',  # Default activation
                'dropout': 0,  # Default dropout
                'actor_topology_type': self.topology_type,
                'critic_topology_type': self.topology_type
            }
            
            # Log network metrics with W&B-native step for perfect alignment
            self.metrics_logger.log_network_metrics(wb_step, network_metrics, self.task_order)
            
            # Create and log network architecture table
            network_table = self.table_creator.create_network_architecture_table(
                model, self.topology_type, hidden_size, num_layers, self.config
            )
            
            # Log the table with W&B-native step
            if wandb.run:
                wandb.log({f"{LOGGING_PATHS['tables']}/network_architecture": network_table}, step=wb_step)
            
            print(f"📊 Network info logged at W&B step {wb_step}")
            
        except Exception as e:
            print(f"⚠️  Could not log network info: {e}")
            # Still log basic info even if detailed logging fails
            basic_metrics = {
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'topology_type': self.topology_type
            }
            self.metrics_logger.log_network_metrics(wb_step, basic_metrics, self.task_order)
    
    def log_cross_task_evaluation(self, step: int, metrics: Dict, task: str) -> None:
        """Log cross-task evaluation results with W&B-native step management."""
        # 🚨 W&B-NATIVE STEP MANAGEMENT: Use W&B's internal step counter
        # This ensures perfect synchronization and eliminates step warnings
        if wandb.run and hasattr(wandb.run, 'step'):
            # Use W&B's own step counter for perfect alignment
            wb_step = wandb.run.step
            print(f"🔄 W&B-Native: Using W&B step {wb_step} for cross-task evaluation")
        else:
            # Fallback to provided step if W&B step not available
            wb_step = max(1, step)  # Ensure step > 0
        
        # Log cross-task comparison with W&B-native step for perfect alignment
        self.metrics_logger.log_cross_task_comparison(wb_step, metrics, task)
    
    def log_topology_comparison(self, step: int, metrics: Dict, comparison_type: str = 'learning_curve') -> None:
        """Log topology comparison data for cross-run analysis with W&B-native step management."""
        # 🚨 W&B-NATIVE STEP MANAGEMENT: Use W&B's internal step counter
        # This ensures perfect synchronization and eliminates step warnings
        if wandb.run and hasattr(wandb.run, 'step'):
            # Use W&B's own step counter for perfect alignment
            wb_step = wandb.run.step
            print(f"🔄 W&B-Native: Using W&B step {wb_step} for topology comparison")
        else:
            # Fallback to provided step if W&B step not available
            wb_step = max(1, step)  # Ensure step > 0
        
        # Create topology comparison path
        comparison_path = f"{LOGGING_PATHS['topology_comparison']}/task_order/{self.task_order}/{comparison_type}"
        
        # Add metadata for comparison
        comparison_metrics = metrics.copy()
        comparison_metrics.update({
            'topology_type': self.topology_type,
            'task_order': self.task_order,
            'comparison_type': comparison_type,
            'step': wb_step
        })
        
        # Log to topology comparison path with W&B-native step for perfect alignment
        wandb.log({comparison_path: comparison_metrics}, step=wb_step)
    
    def log_cross_run_aggregation(self, step: int, aggregated_data: Dict) -> None:
        """Log aggregated data across multiple runs for topology comparison with W&B-native step management."""
        # 🚨 W&B-NATIVE STEP MANAGEMENT: Use W&B's internal step counter
        # This ensures perfect synchronization and eliminates step warnings
        if wandb.run and hasattr(wandb.run, 'step'):
            # Use W&B's own step counter for perfect alignment
            wb_step = wandb.run.step
            print(f"🔄 W&B-Native: Using W&B step {wb_step} for cross-run aggregation")
        else:
            # Fallback to provided step if W&B step not available
            wb_step = max(1, step)  # Ensure step > 0
        
        # This will be used to create plots comparing different topologies
        # on the same task order and parameter configuration
        aggregation_path = f"{LOGGING_PATHS['topology_comparison']}/aggregated/{self.task_order}"
        
        # Add metadata
        aggregated_data.update({
            'task_order': self.task_order,
            'step': wb_step,
            'timestamp': time.time()
        })
        
        # Log with W&B-native step for perfect alignment
        wandb.log({aggregation_path: aggregated_data}, step=wb_step)

    def log_all_tables(self, model: Any, hidden_size: int, num_layers: int) -> None:
        """Create and log all tables for comprehensive analysis with W&B-native step management."""
        # 🚨 W&B-NATIVE STEP MANAGEMENT: Use W&B's internal step counter
        # This ensures perfect synchronization and eliminates step warnings
        if wandb.run and hasattr(wandb.run, 'step'):
            # Use W&B's own step counter for perfect alignment
            wb_step = wandb.run.step
            print(f"🔄 W&B-Native: Using W&B step {wb_step} for comprehensive tables")
        else:
            # Fallback to current global timesteps if W&B step not available
            wb_step = max(1, self.global_timesteps)
        
        print(f"📊 Logging comprehensive tables for {self.topology_type} - {self.task_order}...")
        
        # Create tables
        training_table = self.table_creator.create_training_summary_table(
            self.training_results, self.topology_type, self.task_order
        )
        
        network_table = self.table_creator.create_network_architecture_table(
            model, self.topology_type, hidden_size, num_layers, self.config
        )
        
        phase_table = self.table_creator.create_phase_results_table(
            self.phase_results, self.topology_type, self.task_order
        )
        
        transfer_table = self.table_creator.create_transfer_learning_table(
            self.transfer_metrics, self.topology_type, self.task_order
        )
        
        # Log tables with W&B-native step for perfect alignment
        if training_table:
            wandb.log({f"{LOGGING_PATHS['tables']}/training_summary": training_table}, step=wb_step)
        
        if network_table:
            wandb.log({f"{LOGGING_PATHS['tables']}/network_architecture": network_table}, step=wb_step)
        
        if phase_table:
            wandb.log({f"{LOGGING_PATHS['tables']}/phase_results": phase_table}, step=wb_step)
        
        if transfer_table:
            wandb.log({f"{LOGGING_PATHS['tables']}/transfer_learning": transfer_table}, step=wb_step)
        
        print(f"✅ All tables logged at W&B step {wb_step}")
    
    def store_training_results(self, results: Dict) -> None:
        """Store training results for later table creation."""
        self.training_results.update(results)
    
    def store_phase_results(self, phase: int, task: str, results: Dict) -> None:
        """Store phase-specific results for later table creation."""
        if task not in self.phase_results:
            self.phase_results[task] = {}
        self.phase_results[task].update(results)
    
    def store_transfer_metrics(self, metrics: Dict) -> None:
        """Store transfer learning metrics for later table creation."""
        self.transfer_metrics.update(metrics)
    
    def finish_run(self) -> None:
        """Clean up and finish the logging session."""
        if wandb.run:
            wandb.finish()
            print("✅ W&B run finished")

# ============================================================================
# ENHANCED DEBUG CALLBACK (UPDATED)
# ============================================================================

class EnhancedDebugCallback(BaseCallback):
    """
    Enhanced debug callback with comprehensive logging and step validation.
    """
    
    def __init__(self, logging_handler: TopologyLoggingHandler, log_freq: int = 1000):
        super().__init__()
        self.logging_handler = logging_handler
        self.log_freq = log_freq
        self.last_logged_step = 0
        
        # 🚨 ENHANCED STEP VALIDATION: Add step validation for callback
        self._step_validation_enabled = True
        
        # 🚨 CRITICAL: Restore required attributes for compatibility
        self.rollout_count = 0
        self.step_count = 0
        
        # 🚨 CRITICAL: Add missing episode tracking attributes
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_started = False
        self.n_envs = 1
    
    def _ensure_valid_step(self, step: int) -> int:
        """
        Ensure step is valid for W&B logging in callback.
        
        Args:
            step (int): Input step value
            
        Returns:
            int: Valid step value (guaranteed > 0)
        """
        if not self._step_validation_enabled:
            return step
        
        # 🚨 CRITICAL: W&B requires steps > 0 and monotonically increasing
        if step <= 0:
            # Use W&B's internal step if available, otherwise use fallback
            if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step > 0:
                valid_step = wandb.run.step
                print(f"⚠️  Callback step validation: {step} → {valid_step} (using W&B internal step)")
            else:
                # Fallback: use current timesteps + 1 to ensure > 0
                valid_step = max(1, self.num_timesteps + 1)
                print(f"⚠️  Callback step validation: {step} → {valid_step} (using fallback)")
            return valid_step
        
        return step
    
    def _get_wandb_step(self, fallback_step: int = None) -> int:
        """
        Get the current W&B step for callback logging.
        
        Args:
            fallback_step (int): Fallback step if W&B step unavailable
            
        Returns:
            int: Valid W&B step
        """
        # 🚨 CRITICAL: Primary approach - use W&B's internal step
        if wandb.run and hasattr(wandb.run, 'step') and wandb.run.step > 0:
            wb_step = wandb.run.step
            return wb_step
        
        # 🚨 CRITICAL: Fallback approach - validate and use fallback
        if fallback_step is None:
            fallback_step = max(1, self.num_timesteps + 1)
        
        # Ensure fallback step is valid
        validated_step = self._ensure_valid_step(fallback_step)
        return validated_step
    
    def set_task_phase(self, task_name: str, phase_number: int) -> None:
        """Set the current task phase for sequential training tracking."""
        self.logging_handler.set_task_phase(task_name, phase_number)
    
    def _on_step(self) -> bool:
        """
        Called at each step during training with enhanced step validation.
        """
        # 🚨 ENHANCED STEP VALIDATION: Only log at specified frequency
        if self.num_timesteps % self.log_freq != 0:
            return True
        
        # 🚨 CRITICAL: Get validated step for logging
        wb_step = self._get_wandb_step(fallback_step=self.num_timesteps)
        
        # Update logging handler with current step
        self.logging_handler.current_task_local_step = self.num_timesteps
        
        # Log training step with validated step
        training_metrics = {
            'train/global/timesteps': self.num_timesteps,
            'train/global/step_count': self.num_timesteps,
            'train/global/rollout_count': getattr(self, 'rollout_count', 0)
        }
        
        # Use validated step for all W&B logging
        self.logging_handler.log_training_step(self.num_timesteps, training_metrics)
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Log metrics at the end of each rollout."""
        self.rollout_count += 1
        
        # 🚨 NEW: Only log every log_freq rollouts to reduce verbose output
        # This keeps the progress bar clean while preserving important data
        if self.rollout_count % self.log_freq != 0:
            return
        
        if wandb.run:
            self._log_rollout_metrics()
    
    def _on_training_end(self) -> None:
        """Log final training summary."""
        if wandb.run:
            self._log_final_training_summary()
    
    def _track_episode_progress(self):
        """Track episode rewards and lengths for proper logging."""
        # Get current reward and done status from the environment
        if hasattr(self, 'training_env') and self.training_env is not None:
            try:
                # Get the last observation to check if episode is done
                if hasattr(self.training_env, 'get_attr'):
                    dones = self.training_env.get_attr('done')[0]
                    rewards = self.training_env.get_attr('rewards')[0] if hasattr(self.training_env, 'rewards') else [0]
                    
                    if dones:
                        # Episode ended, store the episode data
                        if self.episode_started:
                            self.episode_rewards.append(self.current_episode_reward)
                            self.episode_lengths.append(self.current_episode_length)
                            
                            # Reset for next episode
                            self.current_episode_reward = 0
                            self.current_episode_length = 0
                            self.episode_started = False
                        else:
                            # Episode started
                            self.episode_started = True
                            self.current_episode_reward = rewards[0] if rewards else 0
                            self.current_episode_length = 1
                    else:
                        # Episode continues
                        if self.episode_started:
                            self.current_episode_reward += rewards[0] if rewards else 0
                            self.current_episode_length += 1
                        else:
                            # Episode started
                            self.episode_started = True
                            self.current_episode_reward = rewards[0] if rewards else 0
                            self.current_episode_length = 1
            except Exception as e:
                # Fallback: just increment step count
                pass
    
    def _log_training_metrics(self):
        """Log streamlined training metrics using the logging handler."""
        try:
            # Get metrics from the model's logger
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                name_to_value = self.model.logger.name_to_value
                
                # Collect essential metrics
                metrics = {
                    "timesteps": self.num_timesteps,  # This is the LOCAL task timestep
                    "episodes": len(self.episode_rewards),
                }
                
                # Add essential PPO metrics
                for key, value in name_to_value.items():
                    if any(term in key.lower() for term in ['loss', 'entropy', 'lr', 'value', 'policy', 'clip']):
                        metrics[key] = value
                
                # Add learning rate
                if hasattr(self.model, 'lr_schedule'):
                    current_lr = self.model.lr_schedule(self.num_timesteps)
                    metrics["learning_rate"] = current_lr
                
                # Add reward and length metrics if available
                if self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                    recent_lengths = self.episode_lengths[-100:]  # Last 100 episodes
                    
                    metrics.update({
                        "mean_reward": np.mean(recent_rewards),
                        "mean_length": np.mean(recent_lengths),
                        "training_progress": self.num_timesteps / self.model.total_timesteps if hasattr(self.model, 'total_timesteps') else 0.0
                    })
                    
                    # Add success rate and completion percentage if we have task info
                    if self.logging_handler.current_task:
                        try:
                            from src.utils.task_normalization import calculate_success_rate, calculate_reward_completion_percentage
                            current_task = self.logging_handler.current_task
                            success_rate = calculate_success_rate(recent_rewards, recent_lengths, current_task)
                            completion_pct = calculate_reward_completion_percentage(recent_rewards, current_task)
                            metrics.update({
                                "success_rate": success_rate,
                                "completion_percentage": completion_pct
                            })
                        except ImportError:
                            # Fallback if normalization module not available
                            pass
                
                # 🚨 CRITICAL: Don't log during first few steps to prevent W&B step warnings
                # Only log when we have meaningful progress (step > 0)
                if self.num_timesteps > 0:
                    self.logging_handler.log_training_step(self.num_timesteps, metrics)
                
        except Exception as e:
            print(f"   ⚠️  Error logging training metrics: {e}")
    
    def _log_rollout_metrics(self):
        """Log rollout metrics using the logging handler."""
        try:
            metrics = {
                "timesteps": self.num_timesteps,  # LOCAL timestep
                "rollout_count": self.rollout_count,
                "episode_rewards": self.episode_rewards[-10:] if self.episode_rewards else [],  # Last 10 episodes
                "episode_lengths": self.episode_lengths[-10:] if self.episode_lengths else [],  # Last 10 episodes
            }
            
            # 🚨 CRITICAL: Pass LOCAL timestep to logging handler
            # The handler will convert it to global timestep for continuous progression
            # 🚨 CRITICAL: Don't log during first few steps to prevent W&B step warnings
            # Only log when we have meaningful progress (step > 0)
            if self.num_timesteps > 0:
                self.logging_handler.log_rollout_end(self.num_timesteps, metrics)
            
        except Exception as e:
            print(f"   ⚠️  Error logging rollout metrics: {e}")
    
    def _log_final_training_summary(self):
        """Log final training summary using the logging handler."""
        try:
            # Get final metrics
            final_metrics = {
                "total_steps": self.step_count,
                "total_rollouts": self.rollout_count,
                "total_episodes": len(self.episode_rewards),
                "final_mean_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                "final_mean_length": np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            }
            
            # Log final summary
            if wandb.run:
                # Log final training summary with validated step
                final_metrics = {
                    'train/global/final_summary': final_metrics
                }
                
                # 🚨 CRITICAL FIX: Ensure step is never 0 for final summary
                final_step = max(1, self.logging_handler.global_timesteps)
                
                # Use validated step for final summary logging
                wandb.log(final_metrics, step=final_step)
                
        except Exception as e:
            print(f"   ⚠️  Error logging final training summary: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_logging_handler(config: Dict, topology_type: str, training_type: str = 'triple_task') -> TopologyLoggingHandler:
    """Factory function to create a logging handler."""
    return TopologyLoggingHandler(config, topology_type, training_type)

def log_streamlined_tables(logging_handler: TopologyLoggingHandler, model: Any, 
                          hidden_size: int, num_layers: int) -> None:
    """Log all streamlined tables using the logging handler."""
    logging_handler.log_all_tables(model, hidden_size, num_layers)
