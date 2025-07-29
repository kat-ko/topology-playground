# 🔧 Enhanced Metrics Collection Plan

## **📊 Current State Analysis**

### **✅ Already Implemented:**
- Basic graph metrics (density, clustering, path length, diameter)
- Training curves and learning progress
- Network visualizations
- Parameter counts and network statistics

### **❌ Missing for Requirements:**
- Real-time graph metrics during training
- Hyperparameter correlation with emergent structure
- Graph-theoretic depth analysis
- Sample efficiency tracking
- Topology-specific learning dynamics

---

## **🎯 Implementation Strategy**

### **Phase 1: Enhanced Graph Metrics Collection**

**1.1 Real-Time Graph Metrics During Training**
```python
# Add to EnhancedDebugCallback
def _log_graph_metrics(self):
    """Log real-time graph metrics during training."""
    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
        actor_G = self.model.policy.actor_topology.topology
        critic_G = self.model.policy.critic_topology.topology
        
        # Calculate graph metrics
        actor_metrics = self._calculate_graph_metrics(actor_G, 'actor')
        critic_metrics = self._calculate_graph_metrics(critic_G, 'critic')
        
        # Log with timestep correlation
        metrics = {
            **actor_metrics,
            **critic_metrics,
            'timestep': self.num_timesteps,
            'topology_type': self.model.policy.topology_type
        }
        
        self.wandb_run.log(metrics, step=self.num_timesteps)
```

**1.2 Comprehensive Graph Metrics**
```python
def _calculate_graph_metrics(self, G, network_type):
    """Calculate comprehensive graph metrics."""
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    # Basic metrics
    metrics = {
        f'{network_type}/nodes': len(G.nodes()),
        f'{network_type}/edges': len(G.edges()),
        f'{network_type}/density': nx.density(G),
        f'{network_type}/avg_degree': sum(dict(G.degree()).values()) / len(G.nodes()),
    }
    
    # Connectivity-dependent metrics
    if nx.is_connected(G_undirected):
        metrics.update({
            f'{network_type}/diameter': nx.diameter(G_undirected),
            f'{network_type}/avg_path_length': nx.average_shortest_path_length(G_undirected),
            f'{network_type}/clustering_coefficient': nx.average_clustering(G_undirected),
        })
    else:
        # Use largest connected component
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        largest_cc_graph = G_undirected.subgraph(largest_cc)
        metrics.update({
            f'{network_type}/diameter_largest_cc': nx.diameter(largest_cc_graph),
            f'{network_type}/avg_path_length_largest_cc': nx.average_shortest_path_length(largest_cc_graph),
            f'{network_type}/clustering_coefficient': nx.average_clustering(G_undirected),
            f'{network_type}/connected_components': nx.number_connected_components(G_undirected),
            f'{network_type}/largest_cc_size': len(largest_cc),
        })
    
    return metrics
```

### **Phase 2: Graph-Theoretic Depth Analysis**

**2.1 Depth Metrics Correlation**
```python
def _log_depth_analysis(self):
    """Log depth analysis correlating graph structure with performance."""
    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
        actor_G = self.model.policy.actor_topology.topology
        critic_G = self.model.policy.critic_topology.topology
        
        # Calculate depth metrics
        actor_depth = self._calculate_depth_metrics(actor_G, 'actor')
        critic_depth = self._calculate_depth_metrics(critic_G, 'critic')
        
        # Performance correlation
        current_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        
        depth_analysis = {
            **actor_depth,
            **critic_depth,
            'performance/current_reward': current_reward,
            'performance/depth_efficiency': current_reward / (actor_depth['actor/avg_path_length'] + 1e-6),
            'performance/density_efficiency': current_reward / (actor_depth['actor/density'] + 1e-6),
        }
        
        self.wandb_run.log(depth_analysis, step=self.num_timesteps)
```

**2.2 Topology-Specific Learning Rate Analysis**
```python
def _analyze_learning_rate_effectiveness(self):
    """Analyze learning rate effectiveness for different topologies."""
    if hasattr(self.model, 'lr_schedule'):
        current_lr = self.model.lr_schedule(self.num_timesteps)
        
        # Get current performance
        current_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        
        # Get graph depth metrics
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
            actor_G = self.model.policy.actor_topology.topology
            G_undirected = actor_G.to_undirected() if actor_G.is_directed() else actor_G
            
            if nx.is_connected(G_undirected):
                avg_path_length = nx.average_shortest_path_length(G_undirected)
                diameter = nx.diameter(G_undirected)
                density = nx.density(actor_G)
                
                # Learning rate analysis
                lr_analysis = {
                    'learning_rate/current': current_lr,
                    'learning_rate/path_length_ratio': current_lr / (avg_path_length + 1e-6),
                    'learning_rate/diameter_ratio': current_lr / (diameter + 1e-6),
                    'learning_rate/density_ratio': current_lr / (density + 1e-6),
                    'learning_rate/performance_ratio': current_reward / (current_lr + 1e-6),
                }
                
                self.wandb_run.log(lr_analysis, step=self.num_timesteps)
```

### **Phase 3: Enhanced Learning Curves**

**3.1 Sample Efficiency Tracking**
```python
def _log_sample_efficiency(self):
    """Log sample efficiency metrics."""
    if len(self.episode_rewards) > 0:
        # Calculate sample efficiency metrics
        total_timesteps = self.num_timesteps
        total_episodes = len(self.episode_rewards)
        
        # Recent performance (last 100 episodes)
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        recent_mean_reward = np.mean(recent_rewards)
        
        # Sample efficiency metrics
        efficiency_metrics = {
            'efficiency/reward_per_timestep': recent_mean_reward / (total_timesteps + 1e-6),
            'efficiency/reward_per_episode': recent_mean_reward,
            'efficiency/episodes_per_timestep': total_episodes / (total_timesteps + 1e-6),
            'efficiency/timesteps_per_episode': total_timesteps / (total_episodes + 1e-6),
            'efficiency/learning_rate': recent_mean_reward / (total_episodes + 1e-6),  # Reward per episode
        }
        
        self.wandb_run.log(efficiency_metrics, step=self.num_timesteps)
```

**3.2 Asymptotic Analysis**
```python
def _log_asymptotic_analysis(self):
    """Log asymptotic learning analysis."""
    if len(self.episode_rewards) > 50:
        # Split into early, middle, and late phases
        total_episodes = len(self.episode_rewards)
        early_end = total_episodes // 3
        middle_end = 2 * total_episodes // 3
        
        early_rewards = self.episode_rewards[:early_end]
        middle_rewards = self.episode_rewards[early_end:middle_end]
        late_rewards = self.episode_rewards[middle_end:]
        
        asymptotic_metrics = {
            'asymptotic/early_mean': np.mean(early_rewards),
            'asymptotic/middle_mean': np.mean(middle_rewards),
            'asymptotic/late_mean': np.mean(late_rewards),
            'asymptotic/improvement_rate': (np.mean(late_rewards) - np.mean(early_rewards)) / (len(late_rewards) + 1e-6),
            'asymptotic/convergence_stability': np.std(late_rewards),
            'asymptotic/learning_plateau': np.mean(late_rewards) - np.mean(middle_rewards),
        }
        
        self.wandb_run.log(asymptotic_metrics, step=self.num_timesteps)
```

### **Phase 4: Topology-Specific Analysis**

**4.1 Topology Comparison Metrics**
```python
def _log_topology_comparison(self):
    """Log topology-specific comparison metrics."""
    topology_type = self.model.policy.topology_type if hasattr(self.model, 'policy') else 'unknown'
    
    # Get current performance
    current_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
    
    # Get graph metrics
    if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'actor_topology'):
        actor_G = self.model.policy.actor_topology.topology
        G_undirected = actor_G.to_undirected() if actor_G.is_directed() else actor_G
        
        if nx.is_connected(G_undirected):
            avg_path_length = nx.average_shortest_path_length(G_undirected)
            diameter = nx.diameter(G_undirected)
            density = nx.density(actor_G)
            clustering = nx.average_clustering(G_undirected)
            
            topology_metrics = {
                f'topology/{topology_type}/performance': current_reward,
                f'topology/{topology_type}/avg_path_length': avg_path_length,
                f'topology/{topology_type}/diameter': diameter,
                f'topology/{topology_type}/density': density,
                f'topology/{topology_type}/clustering': clustering,
                f'topology/{topology_type}/depth_efficiency': current_reward / (avg_path_length + 1e-6),
                f'topology/{topology_type}/density_efficiency': current_reward / (density + 1e-6),
            }
            
            self.wandb_run.log(topology_metrics, step=self.num_timesteps)
```

**4.2 Hyperparameter Correlation**
```python
def _log_hyperparameter_correlation(self):
    """Log hyperparameter correlation with graph structure and performance."""
    if hasattr(self.model, 'policy'):
        # Get current hyperparameters
        config = wandb.config if wandb.run else {}
        
        # Get current performance
        current_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        
        # Get graph metrics
        if hasattr(self.model.policy, 'actor_topology'):
            actor_G = self.model.policy.actor_topology.topology
            G_undirected = actor_G.to_undirected() if actor_G.is_directed() else actor_G
            
            if nx.is_connected(G_undirected):
                avg_path_length = nx.average_shortest_path_length(G_undirected)
                diameter = nx.diameter(G_undirected)
                density = nx.density(actor_G)
                
                # Hyperparameter correlation
                correlation_metrics = {
                    'correlation/learning_rate': config.get('learning_rate', 0),
                    'correlation/hidden_size': config.get('hidden_size', 0),
                    'correlation/num_layers': config.get('num_layers', 0),
                    'correlation/avg_path_length': avg_path_length,
                    'correlation/diameter': diameter,
                    'correlation/density': density,
                    'correlation/performance': current_reward,
                    'correlation/lr_path_length_ratio': config.get('learning_rate', 0) / (avg_path_length + 1e-6),
                    'correlation/lr_diameter_ratio': config.get('learning_rate', 0) / (diameter + 1e-6),
                }
                
                self.wandb_run.log(correlation_metrics, step=self.num_timesteps)
```

---

## **🔧 Integration Points**

### **1. EnhancedDebugCallback Enhancement**
- Add new methods to existing callback
- Integrate with existing logging frequency
- Maintain backward compatibility

### **2. Training Script Updates**
- Update all training scripts to use enhanced metrics
- Ensure consistent logging across all topology types
- Add topology-specific analysis

### **3. Sweep Configuration Integration**
- Add new metrics to sweep configurations
- Ensure metrics are tracked for all analysis types
- Enable correlation analysis in sweeps

---

## **📈 Expected Benefits**

### **1. Scientific Insights**
- **Graph Structure Impact**: Understand how topology affects learning
- **Hyperparameter Optimization**: Correlate graph metrics with optimal hyperparameters
- **Topology Comparison**: Fair comparison of different network structures

### **2. Research Quality**
- **Reproducibility**: Comprehensive metrics for all experiments
- **Interpretability**: Clear correlation between structure and performance
- **Biological Relevance**: Graph metrics that relate to neural network properties

### **3. Optimization Efficiency**
- **Faster Convergence**: Better hyperparameter selection based on topology
- **Reduced Search Space**: Focus on relevant parameter ranges
- **Better Results**: More informed optimization decisions

---

## **🚀 Implementation Priority**

1. **Phase 1**: Real-time graph metrics (High Priority)
2. **Phase 2**: Depth analysis (High Priority)  
3. **Phase 3**: Enhanced learning curves (Medium Priority)
4. **Phase 4**: Topology-specific analysis (Medium Priority)

This plan addresses all your requirements while building on the existing infrastructure!