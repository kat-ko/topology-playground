import numpy as np
import torch
from typing import Dict, Any
from ..tasks.rl_tasks import RLTaskConfig

class NetworkAgent:
    """Agent wrapper for our network architectures."""
    
    def __init__(self, network, config: RLTaskConfig):
        """Initialize the agent with a network and configuration."""
        self.network = network
        self.config = config
        
        # Get max dimensions from config
        self.max_state_dim = config.state_dim  # Use actual state dimension from config
        
        # Task-specific routing table
        self.action_dim = config.action_dim
        assert len(self.network.output_nodes) >= self.action_dim, \
            f"Need at least {self.action_dim} output nodes, got {len(self.network.output_nodes)}"
        
        # Use first-k nodes for actions
        self.active_outputs = self.network.output_nodes[:self.action_dim]
        
        # Create output mask (1 for active, 0 for inactive)
        self.output_mask = np.zeros(len(self.network.output_nodes))
        self.output_mask[:self.action_dim] = 1
        
        # Store mapping for logging
        self.action_mapping = {
            env_idx: node_id 
            for env_idx, node_id in enumerate(self.active_outputs)
        }
    
    def select_action(self, state):
        """Select action using the network."""
        # Convert state to numpy if needed
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        
        # Pad state to max dimension
        padded_state = np.zeros(self.max_state_dim)
        padded_state[:len(state)] = state
        
        # Create input dictionary with zero-padding
        inputs = {}
        for j, node in enumerate(self.network.input_nodes):
            inputs[node] = padded_state[j] if j < len(state) else 0.0
        
        # Get network output
        with torch.no_grad():
            outputs = self.network.forward(inputs)
        
        # Convert output dictionary to array and apply masking
        raw_outputs = np.array([outputs[node] for node in self.network.output_nodes])
        masked_outputs = raw_outputs * self.output_mask
        masked_outputs = masked_outputs[:self.action_dim]  # Keep only active outputs
        
        # Add small epsilon to avoid numerical issues
        masked_outputs = masked_outputs + 1e-10
        
        # Normalize probabilities
        action_probs = np.exp(masked_outputs) / np.sum(np.exp(masked_outputs))
        
        # Sample action
        action = np.random.choice(self.action_dim, p=action_probs)
        
        return action 