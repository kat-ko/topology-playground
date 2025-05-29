import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from src.agents.rl_agents import PPOAgent, A2CAgent, SACAgent, AgentConfig
from src.tasks.rl_tasks import RLTaskGenerator, RLTaskEvaluator
from src.experiment.runner import ExperimentRunner
from src.utils.logging import get_logger, setup_logging
from collections import deque, defaultdict
from src.utils.parameter_budget import ParameterBudget

logger = get_logger(__name__)

class CurriculumRunner:
    """Runs curriculum learning experiments with RL agents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the curriculum runner."""
        self.config = config
        self.parameter_budget = ParameterBudget(config)
        
        # Initialize agents
        self.agents = {
            'ppo': PPOAgent(config),
            'sac': SACAgent(config),
            'a2c': A2CAgent(config)
        }
        
        # Initialize metrics tracking
        self.metrics = defaultdict(lambda: defaultdict(list))
    
    def run_curriculum(self, algorithms: List[str] = None, experiment_types: List[str] = None):
        """Run curriculum learning experiments for specified algorithms and experiment types."""
        if algorithms is None:
            algorithms = list(self.agents.keys())
        
        if experiment_types is None:
            experiment_types = self.config['experiment_types']
        
        for algo in algorithms:
            for exp_type in experiment_types:
                self.config['experiment_type'] = exp_type
                self._run_single_experiment(algo, exp_type)
    
    def _run_single_experiment(self, algorithm: str, experiment_type: str):
        """Run a single curriculum experiment."""
        # Create agent with capacity-matched network
        agent_config = self.config.copy()
        agent_config['experiment_type'] = experiment_type
        
        # Create network with correct capacity
        network = self.parameter_budget.calculator.create_network(
            topology=self.config['network_types'][0],
            size=self.config['network_sizes'][0],
            experiment_type=experiment_type
        )
        
        # Initialize agent with the capacity-matched network
        agent = self.agents[algorithm]
        agent.network = network
        
        task_generator = RLTaskGenerator(self.config)
        evaluator = RLTaskEvaluator(self.config)
        
        # Initialize metrics for this experiment
        exp_key = f"{algorithm}_{experiment_type}"
        self.metrics[exp_key] = defaultdict(list)
        
        # Run curriculum
        for task in self.config['task_sequence']:
            # Generate task
            task_env = task_generator.generate_task(task)
            
            # Train agent
            for episode in range(self.config['episodes_per_task']):
                # Get action from agent
                state = task_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = task_env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    episode_reward += reward
                
                # Log metrics
                self.metrics[exp_key]['episode_rewards'].append(episode_reward)
                
                # Check if task is learned
                if episode_reward >= self.config['task_memory_threshold']:
                    self.metrics[exp_key]['learning_episodes'].append(episode)
                    break
            
            # Evaluate agent
            eval_rewards = []
            for _ in range(self.config['evaluation_episodes']):
                state = task_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = task_env.step(action)
                    state = next_state
                    episode_reward += reward
                
                eval_rewards.append(episode_reward)
            
            # Log evaluation metrics
            self.metrics[exp_key]['eval_rewards'].append(np.mean(eval_rewards))
            
            # Get parameter budget stats
            budget_stats = self.parameter_budget.get_budget_stats(
                agent.network,
                self.config['network_sizes'][0],
                agent.topology
            )
            self.metrics[exp_key]['parameter_stats'].append(budget_stats)
            
            # Test transfer learning if applicable
            if task in self.config['backward_transfer_tasks']:
                self._test_transfer_learning(agent, task, exp_key)
            
            # Test forgetting and retention
            self._test_forgetting_retention(agent, task, exp_key)
    
    def _test_transfer_learning(self, agent: RLAgent, task: str, exp_key: str):
        """Test transfer learning capabilities."""
        task_generator = RLTaskGenerator(self.config)
        evaluator = RLTaskEvaluator(self.config)
        
        # Test backward transfer
        if task in self.config['backward_transfer_tasks']:
            for prev_task in self.config['task_sequence'][:self.config['task_sequence'].index(task)]:
                prev_env = task_generator.generate_task(prev_task)
                transfer_reward = evaluator.evaluate(agent, prev_env)
                self.metrics[exp_key]['backward_transfer'].append({
                    'from_task': task,
                    'to_task': prev_task,
                    'reward': transfer_reward
                })
        
        # Test forward transfer
        if task in self.config['forward_transfer_tasks']:
            for next_task in self.config['task_sequence'][self.config['task_sequence'].index(task)+1:]:
                next_env = task_generator.generate_task(next_task)
                transfer_reward = evaluator.evaluate(agent, next_env)
                self.metrics[exp_key]['forward_transfer'].append({
                    'from_task': task,
                    'to_task': next_task,
                    'reward': transfer_reward
                })
    
    def _test_forgetting_retention(self, agent: RLAgent, task: str, exp_key: str):
        """Test forgetting and retention of learned tasks."""
        if self.config['forgetting_test']['enabled']:
            task_generator = RLTaskGenerator(self.config)
            evaluator = RLTaskEvaluator(self.config)
            
            # Test retention
            if len(self.metrics[exp_key]['episode_rewards']) % self.config['forgetting_test']['retention_interval'] == 0:
                task_env = task_generator.generate_task(task)
                retention_reward = evaluator.evaluate(agent, task_env)
                
                self.metrics[exp_key]['retention'].append({
                    'task': task,
                    'reward': retention_reward,
                    'iteration': len(self.metrics[exp_key]['episode_rewards'])
                })
                
                # Check for forgetting
                if retention_reward < self.config['forgetting_test']['forgetting_threshold']:
                    self.metrics[exp_key]['forgetting'].append({
                        'task': task,
                        'iteration': len(self.metrics[exp_key]['episode_rewards'])
                    })
    
    def run_episode(self, agent: torch.nn.Module, task: Any, algorithm: str) -> Tuple[float, int]:
        """Run a single episode with parameter budget tracking."""
        total_reward = 0
        steps = 0
        
        # Get task state
        state = task.reset()
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, _ = task.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check if we need to enforce budget
            if steps % 100 == 0:  # Check periodically
                agent = self.parameter_budget.enforce_budget(agent, agent.hidden_size)
        
        return total_reward, steps
    
    def evaluate_agent(self, agent: torch.nn.Module, task: Any, num_episodes: int = None) -> Tuple[float, float, float, float]:
        """Evaluate agent performance with parameter budget tracking."""
        if num_episodes is None:
            num_episodes = self.config['curriculum']['evaluation_episodes']
        
        rewards = []
        steps = []
        
        for _ in range(num_episodes):
            reward, step = self.run_episode(agent, task, None)  # No algorithm needed for evaluation
            rewards.append(reward)
            steps.append(step)
        
        # Get budget stats
        budget_stats = self.parameter_budget.get_budget_stats(agent, agent.hidden_size)
        
        return (
            np.mean(rewards),
            np.std(rewards),
            np.mean(steps),
            np.std(steps),
            budget_stats
        )
    
    def replay_task_memory(self, agent: torch.nn.Module, algorithm: str) -> None:
        """Replay experiences from task memory with parameter budget tracking."""
        if not self.task_memory[algorithm]:
            return
        
        for task in self.task_memory[algorithm]:
            # Run a few episodes on each remembered task
            for _ in range(5):  # Replay 5 episodes per task
                self.run_episode(agent, task, algorithm)
            
            # Enforce budget after replay
            agent = self.parameter_budget.enforce_budget(agent, agent.hidden_size)
    
    def test_retention(self, agent: torch.nn.Module, task: Any, baseline_performance: float) -> Dict[str, float]:
        """Test retention of knowledge for a specific task."""
        retention_results = self.evaluate_agent(
            agent, 
            task, 
            self.config['forgetting_test']['retention_episodes']
        )
        
        # Calculate retention metrics
        retention_ratio = retention_results[0] / baseline_performance
        is_retained = retention_ratio >= self.config['forgetting_test']['retention_threshold']
        is_forgotten = retention_ratio <= self.config['forgetting_test']['forgetting_threshold']
        
        return {
            'mean_reward': retention_results[0],
            'std_reward': retention_results[1],
            'retention_ratio': retention_ratio,
            'is_retained': is_retained,
            'is_forgotten': is_forgotten
        }
    
    def run_experiment(self) -> Dict[str, Dict[str, Any]]:
        """Run curriculum learning experiment for all algorithms with parameter budget tracking."""
        results = {}
        
        for algorithm in ['ppo', 'a2c', 'sac']:
            self.logger.info(f"Running curriculum for {algorithm}")
            results[algorithm] = self.run_curriculum(algorithm)
        
        return results 