import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from torch.distributions import Categorical, Normal
import gymnasium as gym

@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    learning_rate: float = 0.001
    gamma: float = 0.99
    buffer_size: int = 10000
    batch_size: int = 64
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_update_freq: int = 1000
    tau: float = 0.005

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO and A2C."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.actor(state), self.critic(state)

class SACNetwork(nn.Module):
    """Network for SAC."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.target_critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.target_critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

class ReplayBuffer:
    """Replay buffer for storing transitions."""
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return (torch.FloatTensor(states), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(next_states),
                torch.FloatTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class PPOAgent:
    """PPO agent implementation."""
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs, _ = self.network(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def update(self):
        if len(self.memory) < self.config.batch_size:
            return
        
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.config.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            probs, values = self.network(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs)
            
            # Calculate advantages
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            entropy_loss = -0.01 * dist.entropy().mean()
            
            total_loss = actor_loss + self.config.value_coef * critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.memory = []

class A2CAgent:
    """A2C agent implementation."""
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.memory = []
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs, _ = self.network(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def update(self):
        if len(self.memory) < self.config.batch_size:
            return
        
        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Calculate returns
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.config.gamma * R * (1 - done)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # A2C update
        probs, values = self.network(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        advantages = returns - values.squeeze()
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        
        total_loss = actor_loss + self.config.value_coef * critic_loss - self.config.entropy_coef * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.memory = []

class SACAgent:
    """SAC agent implementation."""
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig):
        self.config = config
        self.network = SACNetwork(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.network.actor.parameters(), lr=config.learning_rate)
        self.optimizer_critic = optim.Adam(
            list(self.network.critic1.parameters()) + list(self.network.critic2.parameters()),
            lr=config.learning_rate
        )
        self.buffer = ReplayBuffer(config.buffer_size)
        self.total_it = 0
    
    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.network.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()
    
    def update(self):
        if len(self.buffer) < self.config.batch_size:
            return
        
        self.total_it += 1
        
        # Sample from buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        
        # Update critic
        with torch.no_grad():
            next_probs = self.network.actor(next_states)
            next_dist = Categorical(next_probs)
            next_actions = next_dist.sample()
            next_q1 = self.network.target_critic1(torch.cat([next_states, next_actions.float()], dim=1))
            next_q2 = self.network.target_critic2(torch.cat([next_states, next_actions.float()], dim=1))
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1 - dones) * self.config.gamma * next_q
        
        current_q1 = self.network.critic1(torch.cat([states, actions.float()], dim=1))
        current_q2 = self.network.critic2(torch.cat([states, actions.float()], dim=1))
        critic_loss = 0.5 * (current_q1 - target_q).pow(2).mean() + 0.5 * (current_q2 - target_q).pow(2).mean()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # Update actor
        probs = self.network.actor(states)
        dist = Categorical(probs)
        new_actions = dist.sample()
        q1 = self.network.critic1(torch.cat([states, new_actions.float()], dim=1))
        q2 = self.network.critic2(torch.cat([states, new_actions.float()], dim=1))
        q = torch.min(q1, q2)
        actor_loss = -q.mean() + self.config.entropy_coef * dist.entropy().mean()
        
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # Update target networks
        if self.total_it % self.config.target_update_freq == 0:
            for param, target_param in zip(self.network.critic1.parameters(), self.network.target_critic1.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(self.network.critic2.parameters(), self.network.target_critic2.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data) 