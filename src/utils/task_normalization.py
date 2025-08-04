#!/usr/bin/env python3
"""
Task Normalization Utilities

This module provides utilities for normalizing rewards and computing efficiency metrics
across different tasks with different reward scales.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# ============================================================================
# NORMALIZATION CONSTANTS
# ============================================================================

# Minimum and solved reward values for each task
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

# Threshold fraction for efficiency calculation
ALPHA = 0.8

# Threshold values (pre-calculated for efficiency)
THRESHOLDS = {
    task: R_MIN[task] + ALPHA * (R_SOLVED[task] - R_MIN[task])
    for task in R_MIN.keys()
}

# Rolling mean window size
ROLLING_WINDOW = 100


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_reward(reward: float, task: str) -> float:
    """
    Normalize reward to 0-1 scale based on task-specific min and solved values.
    
    Args:
        reward: Raw reward value
        task: Task name (e.g., 'CartPole-v1')
    
    Returns:
        Normalized reward (0-1 scale)
    """
    if task not in R_MIN or task not in R_SOLVED:
        raise ValueError(f"Unknown task: {task}")
    
    r_min = R_MIN[task]
    r_solved = R_SOLVED[task]
    
    # Handle edge cases
    if r_solved == r_min:
        return 0.0 if reward <= r_min else 1.0
    
    normalized = (reward - r_min) / (r_solved - r_min)
    return np.clip(normalized, 0.0, 1.0)


def calculate_reward_completion_percentage(reward: float, task: str) -> float:
    """
    Calculate what percentage of the maximum possible reward has been achieved.
    
    Args:
        reward: Raw reward value
        task: Task name (e.g., 'CartPole-v1')
    
    Returns:
        Completion percentage (0-100 scale)
    """
    if task not in R_MIN or task not in R_SOLVED:
        raise ValueError(f"Unknown task: {task}")
    
    r_min = R_MIN[task]
    r_solved = R_SOLVED[task]
    
    # Handle edge cases
    if r_solved == r_min:
        return 0.0 if reward <= r_min else 100.0
    
    # Calculate completion percentage
    completion = (reward - r_min) / (r_solved - r_min) * 100.0
    return np.clip(completion, 0.0, 100.0)


def calculate_success_rate_with_completion(rewards: List[float], task: str) -> Tuple[float, float]:
    """
    Calculate both traditional success rate and reward completion percentage.
    
    Args:
        rewards: List of episodic rewards
        task: Task name (e.g., 'CartPole-v1')
    
    Returns:
        Tuple of (success_rate, completion_percentage)
    """
    if not rewards:
        return 0.0, 0.0
    
    # Traditional success rate (episodes that reached solved threshold)
    success_count = sum(1 for r in rewards if r >= R_SOLVED[task])
    success_rate = (success_count / len(rewards)) * 100.0
    
    # Average completion percentage across all episodes
    completion_percentages = [calculate_reward_completion_percentage(r, task) for r in rewards]
    avg_completion = np.mean(completion_percentages)
    
    return success_rate, avg_completion


def compute_rolling_mean(rewards: List[float], window_size: int = ROLLING_WINDOW) -> List[float]:
    """
    Compute rolling mean of rewards.
    
    Args:
        rewards: List of episodic rewards
        window_size: Size of rolling window
    
    Returns:
        List of rolling mean values
    """
    if len(rewards) == 0:
        return []
    
    rolling_means = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window_rewards = rewards[start_idx:i + 1]
        rolling_means.append(np.mean(window_rewards))
    
    return rolling_means


def find_steps_to_threshold(rewards: List[float], task: str, max_steps: int) -> int:
    """
    Find the first training step where rolling mean reaches threshold.
    
    Args:
        rewards: List of episodic rewards
        task: Task name
        max_steps: Maximum training steps
    
    Returns:
        Steps to threshold (or max_steps if never reached)
    """
    if task not in THRESHOLDS:
        raise ValueError(f"Unknown task: {task}")
    
    if len(rewards) == 0:
        return max_steps
    
    threshold = THRESHOLDS[task]
    rolling_means = compute_rolling_mean(rewards)
    
    # Find first step where rolling mean >= threshold
    for i, rolling_mean in enumerate(rolling_means):
        if rolling_mean >= threshold:
            # Convert episode index to approximate step count
            # Assuming each episode has roughly the same number of steps
            steps_per_episode = max_steps / len(rewards) if len(rewards) > 0 else 0
            return int(i * steps_per_episode)
    
    return max_steps


def compute_task_metrics(rewards: List[float], task: str, max_steps: int) -> Dict:
    """
    Compute comprehensive metrics for a single task.
    
    Args:
        rewards: List of episodic rewards
        task: Task name
        max_steps: Maximum training steps
    
    Returns:
        Dictionary with task metrics
    """
    if len(rewards) == 0:
        return {
            "normalized_reward": 0.0,
            "steps_to_threshold": max_steps,
            "final_reward": 0.0,
            "rolling_mean_final": 0.0
        }
    
    final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
    normalized_reward = normalize_reward(final_reward, task)
    steps_to_threshold = find_steps_to_threshold(rewards, task, max_steps)
    rolling_mean_final = compute_rolling_mean(rewards)[-1] if len(rewards) > 0 else 0.0
    
    return {
        "normalized_reward": normalized_reward,
        "steps_to_threshold": steps_to_threshold,
        "final_reward": final_reward,
        "rolling_mean_final": rolling_mean_final
    }


def compute_multi_task_metrics(task_rewards: Dict[str, List[float]], max_steps: int) -> Dict:
    """
    Compute aggregated metrics across multiple tasks.
    
    Args:
        task_rewards: Dictionary mapping task names to lists of rewards
        max_steps: Maximum training steps
    
    Returns:
        Dictionary with aggregated metrics
    """
    task_metrics = {}
    
    # Compute metrics for each task
    for task, rewards in task_rewards.items():
        task_metrics[task] = compute_task_metrics(rewards, task, max_steps)
    
    # Compute aggregated scores
    normalized_rewards = [metrics["normalized_reward"] for metrics in task_metrics.values()]
    final_normalized_score = np.mean(normalized_rewards) if normalized_rewards else 0.0
    
    steps_to_threshold_values = [metrics["steps_to_threshold"] for metrics in task_metrics.values()]
    efficiency_scores = [steps / max_steps for steps in steps_to_threshold_values]
    efficiency_score = np.mean(efficiency_scores) if efficiency_scores else 1.0
    
    return {
        "task_metrics": task_metrics,
        "final_normalized_score": final_normalized_score,
        "efficiency_score": efficiency_score
    }


def get_task_thresholds() -> Dict[str, float]:
    """
    Get threshold values for all tasks.
    
    Returns:
        Dictionary mapping task names to threshold values
    """
    return THRESHOLDS.copy()


def get_normalization_constants() -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Get normalization constants.
    
    Returns:
        Tuple of (R_MIN, R_SOLVED) dictionaries
    """
    return R_MIN.copy(), R_SOLVED.copy()


# ============================================================================
# UTILITY FUNCTIONS FOR TRAINING INTEGRATION
# ============================================================================

def log_normalized_metrics(wandb_run, task_metrics: Dict, final_normalized_score: float, 
                          efficiency_score: float, training_phase: str = "final"):
    """
    Log normalized metrics to WandB.
    
    Args:
        wandb_run: WandB run object
        task_metrics: Dictionary with task-specific metrics
        final_normalized_score: Aggregated normalized score
        efficiency_score: Aggregated efficiency score
        training_phase: Training phase identifier (e.g., "phase1", "phase2", "final")
    """
    if wandb_run is None:
        return
    
    # Log task-specific metrics
    for task, metrics in task_metrics.items():
        wandb_run.log({
            f'normalized/{training_phase}/{task}/normalized_reward': metrics["normalized_reward"],
            f'normalized/{training_phase}/{task}/steps_to_threshold': metrics["steps_to_threshold"],
            f'normalized/{training_phase}/{task}/final_reward': metrics["final_reward"],
            f'normalized/{training_phase}/{task}/rolling_mean_final': metrics["rolling_mean_final"],
        })
    
    # Log aggregated metrics
    wandb_run.log({
        f'normalized/{training_phase}/final_normalized_score': final_normalized_score,
        f'normalized/{training_phase}/efficiency_score': efficiency_score,
    })


def print_normalized_summary(task_metrics: Dict, final_normalized_score: float, 
                           efficiency_score: float, training_phase: str = "Final"):
    """
    Print a summary of normalized metrics.
    
    Args:
        task_metrics: Dictionary with task-specific metrics
        final_normalized_score: Aggregated normalized score
        efficiency_score: Aggregated efficiency score
        training_phase: Training phase identifier
    """
    print(f"📊 {training_phase.upper()} NORMALIZED METRICS:")
    
    for task, metrics in task_metrics.items():
        print(f"   • {task}:")
        print(f"     - Normalized Reward: {metrics['normalized_reward']:.3f}")
        print(f"     - Steps to Threshold: {metrics['steps_to_threshold']:,}")
        print(f"     - Final Reward: {metrics['final_reward']:.2f}")
        print(f"     - Rolling Mean Final: {metrics['rolling_mean_final']:.2f}")
    
    print(f"   • Final Normalized Score: {final_normalized_score:.3f}")
    print(f"   • Efficiency Score: {efficiency_score:.3f}") 