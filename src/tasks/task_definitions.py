import numpy as np
from typing import Tuple, Dict, Any
from .rl_tasks import RLTaskGenerator, RLTaskEvaluator, RLTaskConfig

class TaskGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
        self.rl_generator = RLTaskGenerator(seed)
    
    def generate_cartpole_task(self) -> Tuple[Any, Any]:
        """Generate CartPole-v1 environment and config."""
        return self.rl_generator.generate_cartpole_task()
    
    def generate_mountain_car_task(self) -> Tuple[Any, Any]:
        """Generate MountainCar-v0 environment and config."""
        return self.rl_generator.generate_mountain_car_task()
    
    def generate_acrobot_task(self) -> Tuple[Any, Any]:
        """Generate Acrobot-v1 environment and config."""
        return self.rl_generator.generate_acrobot_task()

class TaskEvaluator:
    def __init__(self):
        self.rl_evaluator = RLTaskEvaluator()
    
    def evaluate_cartpole(self, env, agent, config: RLTaskConfig) -> Dict[str, float]:
        """Evaluate CartPole-v1 performance."""
        return self.rl_evaluator.evaluate_episodes(env, agent, config)
    
    def evaluate_mountain_car(self, env, agent, config: RLTaskConfig) -> Dict[str, float]:
        """Evaluate MountainCar-v0 performance."""
        return self.rl_evaluator.evaluate_episodes(env, agent, config)
    
    def evaluate_acrobot(self, env, agent, config: RLTaskConfig) -> Dict[str, float]:
        """Evaluate Acrobot-v1 performance."""
        return self.rl_evaluator.evaluate_episodes(env, agent, config) 