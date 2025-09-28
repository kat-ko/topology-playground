# 🧠 **Functional Modularity Analysis Concept**

## **Overview**
This concept outlines a clean, methodologically sound approach to detect functional modularity in trained neural networks, building on the existing `topologies_continual_task_training_normal_modularity.py` infrastructure.

## **Core Hypothesis**
**Modular network topologies will develop stronger functional modularity than non-modular topologies when trained on continual learning tasks with distribution shifts.**

---

## **📋 Step-by-Step Analysis Pipeline**

### **Phase 1: Training with Checkpointing**
**Goal**: Train networks and save model states after adaptation to each difficulty level.

**Implementation**:
1. **Reuse existing training**: Use `topologies_continual_task_training_normal_modularity.py` as-is
2. **Add checkpoint saving**: After training completes, save the final trained model
3. **Level-specific training**: Train separate models for each target level (1, 2, 3, etc.)

**Output**: Trained model checkpoints for each topology × level combination

---

### **Phase 2: Activation Collection**
**Goal**: Record hidden layer activations from frozen, trained networks under the EXACT same conditions they were trained on.

**Method** (following Tanner et al. 2023, Ellefsen 2015):
1. **Load trained model**: Load frozen PPO model from checkpoint
2. **Recreate training sequence**: Set up environment to replay the EXACT same continual learning sequence
   - Level 0 (iterations 0-199): Clean environment (no noise)
   - Level 1 (iterations 200-399): Noise level 1 
   - Level 2 (iterations 400-599): Noise level 2
   - etc.
3. **Run evaluation on FULL sequence**: Test model on the complete sequence it was trained on
4. **Record activations per level**: Capture activations separately for each level the model experienced
   ```python
   # For each level in the training sequence
   activations_level_0 = collect_activations(model, clean_env, episodes)
   activations_level_1 = collect_activations(model, noise_level_1_env, episodes)
   activations_level_2 = collect_activations(model, noise_level_2_env, episodes)
   ```
5. **Analyze modularity per adapted level**: Each level represents how the model adapted to that difficulty

**Output**: Activation matrices for each level the model was actually trained to handle

---

### **Phase 3-5: Complete Modularity Analysis Per Level**
**Goal**: For EACH level the model was trained on, run the complete functional vs structural modularity analysis.

## **🔄 FOR EACH LEVEL (0, 1, 2, ..., num_levels):**

### **Step 1: Level-Specific Activation Collection** 
**EXACT IMPLEMENTATION:**

```python
def collect_level_activations(model, task_name, level, noise_vector, num_episodes=100):
    """
    Collect hidden activations for a specific training level.
    Uses fixed seeds for reproducible evaluation set.
    """
    # 1. Create environment with EXACT noise from training
    if task_name == "CartPole-v1":
        env = gym.make("CartPole-v1")
    elif task_name == "Acrobot-v1": 
        env = gym.make("Acrobot-v1")
    elif task_name == "LunarLander-v2":
        env = gym.make("LunarLander-v2")
    
    # 2. Apply exact noise wrapper
    if noise_vector is not None and np.any(noise_vector != 0):
        env = ExactNoiseWrapper(env, noise_vector)
    
    # 3. Fixed evaluation seeds for reproducibility (per level)
    evaluation_seeds = list(range(1000 + level*100, 1000 + level*100 + num_episodes))
    # Level 0: seeds 1000-1099, Level 1: seeds 1100-1199, etc.
    
    activations_list = []
    
    # 4. Run episodes with fixed seeds
    for episode_idx, seed in enumerate(evaluation_seeds):
        env.reset(seed=seed)  # Deterministic episode
        obs, _ = env.reset()
        episode_activations = []
        
        done = False
        step = 0
        max_steps = 500
        
        while not done and step < max_steps:
            # 5. Extract hidden layer activations
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                # Get features from policy network
                features = model.policy.features_extractor(obs_tensor)
                # Flatten to 1D array: [hidden_size]
                activation_vector = features.cpu().numpy().flatten()
                episode_activations.append(activation_vector)
            
            # Take deterministic action
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        
        # Add all timesteps from this episode
        activations_list.extend(episode_activations)
    
    env.close()
    
    # 6. Return as matrix X_k ∈ R^(T×N)
    if len(activations_list) == 0:
        return None
    
    activations_matrix = np.array(activations_list)  # Shape: [total_timesteps, num_neurons]
    print(f"   Level {level}: Collected {activations_matrix.shape[0]} timesteps × {activations_matrix.shape[1]} neurons")
    
    return activations_matrix
```

### **Step 2: Level-Specific Functional Connectivity**
**EXACT IMPLEMENTATION:**

```python
def build_functional_connectivity_matrix(activations, correlation_threshold=0.05):
    """
    Build FC matrix from activation time series using Pearson correlations.
    Following Tanner et al. 2023 protocol.
    """
    from scipy.stats import pearsonr
    
    num_timesteps, num_neurons = activations.shape
    fc_matrix = np.zeros((num_neurons, num_neurons))
    
    print(f"   Computing {num_neurons}×{num_neurons} correlation matrix from {num_timesteps} timesteps...")
    
    # 1. Compute pairwise Pearson correlations
    for i in range(num_neurons):
        for j in range(i, num_neurons):
            if i == j:
                # Self-correlation is always 1.0
                fc_matrix[i, j] = 1.0
            else:
                # Compute correlation between neuron i and neuron j time series
                neuron_i_timeseries = activations[:, i]  # Shape: [timesteps]
                neuron_j_timeseries = activations[:, j]  # Shape: [timesteps]
                
                # Pearson correlation coefficient
                corr_coeff, p_value = pearsonr(neuron_i_timeseries, neuron_j_timeseries)
                
                # Handle NaN values (constant activations)
                if np.isnan(corr_coeff):
                    corr_coeff = 0.0
                
                # 2. Apply correlation threshold (Tanner et al. protocol)
                if abs(corr_coeff) < correlation_threshold:
                    corr_coeff = 0.0
                
                # Fill symmetric matrix
                fc_matrix[i, j] = corr_coeff
                fc_matrix[j, i] = corr_coeff
    
    # 3. Calculate sparsity
    num_total_connections = num_neurons * (num_neurons - 1) / 2  # Exclude diagonal
    num_zero_connections = np.sum(fc_matrix == 0) / 2  # Count off-diagonal zeros
    sparsity = num_zero_connections / num_total_connections
    
    print(f"   FC Matrix: {fc_matrix.shape}, Sparsity: {sparsity:.2f}")
    print(f"   Correlation range: [{np.min(fc_matrix):.3f}, {np.max(fc_matrix):.3f}]")
    
    return fc_matrix
```

### **Step 3: Level-Specific Functional Community Detection**
**EXACT IMPLEMENTATION:**

```python
def detect_functional_communities(fc_matrix, graph_threshold=0.1):
    """
    Detect functional modules using Louvain algorithm.
    Following Tanner et al. 2023 methodology.
    """
    import networkx as nx
    import community as community_louvain
    
    num_neurons = fc_matrix.shape[0]
    
    # 1. Build NetworkX graph from FC matrix
    G = nx.Graph()
    
    # Add all neurons as nodes
    for i in range(num_neurons):
        G.add_node(i)
    
    # Add edges for correlations above threshold
    edges_added = 0
    for i in range(num_neurons):
        for j in range(i + 1, num_neurons):  # Only upper triangle (undirected graph)
            correlation = fc_matrix[i, j]
            
            # Add edge if correlation exceeds threshold
            if abs(correlation) > graph_threshold:
                G.add_edge(i, j, weight=abs(correlation))
                edges_added += 1
    
    print(f"   Graph: {num_neurons} nodes, {edges_added} edges (threshold={graph_threshold})")
    
    # 2. Handle edge case: no edges above threshold
    if G.number_of_edges() == 0:
        print(f"   ⚠️  No edges above threshold {graph_threshold}, creating single community")
        # Single community containing all neurons
        communities = {i: 0 for i in range(num_neurons)}
        modularity_score = 0.0
        num_communities = 1
    else:
        # 3. Apply Louvain community detection algorithm
        communities = community_louvain.best_partition(G)
        
        # 4. Calculate modularity score Q
        modularity_score = community_louvain.modularity(communities, G)
        
        # Count number of unique communities
        num_communities = len(set(communities.values()))
    
    print(f"   Functional Communities: {num_communities}")
    print(f"   Modularity Score Q_functional: {modularity_score:.4f}")
    
    # 5. Community size distribution
    community_sizes = {}
    for neuron, community_id in communities.items():
        if community_id not in community_sizes:
            community_sizes[community_id] = 0
        community_sizes[community_id] += 1
    
    print(f"   Community sizes: {list(community_sizes.values())}")
    
    return communities, modularity_score, num_communities, community_sizes
```

### **Step 4: Level-Specific Structural Analysis** *(Same for all levels)*
**EXACT IMPLEMENTATION:**

```python
def analyze_structural_modularity(model):
    """
    Extract structural modularity from network weights.
    Following Ellefsen 2015 methodology.
    """
    print("   🏗️  Extracting structural modularity from network weights...")
    
    # 1. Extract weight matrices from trained PPO policy
    policy_net = model.policy
    structural_weights = []
    
    # Try features_extractor first
    for name, param in policy_net.features_extractor.named_parameters():
        if 'weight' in name and param.dim() == 2:
            structural_weights.append(param.detach().cpu().numpy())
            print(f"      Found weight matrix: {name}, shape: {param.shape}")
    
    # Fallback to policy layers if no features_extractor weights
    if len(structural_weights) == 0:
        for name, param in policy_net.named_parameters():
            if 'weight' in name and param.dim() == 2:
                structural_weights.append(param.detach().cpu().numpy())
                print(f"      Found policy weight: {name}, shape: {param.shape}")
    
    # 2. Build structural adjacency matrix from weights
    if len(structural_weights) == 0:
        print("      ⚠️  No weight matrices found, using minimal structural matrix")
        num_neurons = 4  # Default for small networks
        structural_matrix = np.ones((num_neurons, num_neurons)) * 0.1
        np.fill_diagonal(structural_matrix, 1.0)
    else:
        # Use first weight matrix for structural analysis
        weight_matrix = structural_weights[0]
        
        # Create symmetric adjacency matrix from absolute weights
        if weight_matrix.shape[0] == weight_matrix.shape[1]:
            structural_matrix = np.abs(weight_matrix)
            structural_matrix = (structural_matrix + structural_matrix.T) / 2
        else:
            # Handle non-square matrices
            min_dim = min(weight_matrix.shape)
            if weight_matrix.shape[0] < weight_matrix.shape[1]:
                square_weights = weight_matrix[:, :min_dim]
            else:
                square_weights = weight_matrix[:min_dim, :]
            
            structural_matrix = np.abs(square_weights)
            if square_weights.shape[0] == square_weights.shape[1]:
                structural_matrix = (structural_matrix + structural_matrix.T) / 2
    
    print(f"      Structural matrix shape: {structural_matrix.shape}")
    
    # 3. Apply Louvain to structural matrix (low threshold for weights)
    struct_communities, struct_modularity, struct_num_communities, struct_sizes = \
        detect_functional_communities(structural_matrix, graph_threshold=0.01)
    
    print(f"   Structural Q: {struct_modularity:.4f}")
    print(f"   Structural Communities: {struct_num_communities}")
    
    return {
        'modularity_score': struct_modularity,
        'num_communities': struct_num_communities,
        'communities': struct_communities,
        'community_sizes': struct_sizes,
        'structural_matrix': structural_matrix
    }
```

### **Step 5: Level-Specific Comparison**
**EXACT IMPLEMENTATION:**

```python
def compare_functional_vs_structural(func_communities, struct_communities, func_q, struct_q):
    """
    Compare functional and structural community structures.
    Computes alignment metrics following literature standards.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    
    # 1. Convert community dictionaries to label arrays
    max_neuron = max(max(func_communities.keys()), max(struct_communities.keys()))
    
    func_labels = np.zeros(max_neuron + 1, dtype=int)
    struct_labels = np.zeros(max_neuron + 1, dtype=int)
    
    for neuron, community in func_communities.items():
        func_labels[neuron] = community
        
    for neuron, community in struct_communities.items():
        struct_labels[neuron] = community
    
    # 2. Compute community alignment metrics
    nmi = normalized_mutual_info_score(func_labels, struct_labels)
    ari = adjusted_rand_score(func_labels, struct_labels)
    
    # 3. Modularity score comparisons
    q_difference = func_q - struct_q
    q_ratio = func_q / struct_q if struct_q != 0 else float('inf')
    
    print(f"   📊 Q_functional vs Q_structural: {func_q:.4f} vs {struct_q:.4f}")
    print(f"   📊 ΔQ (func - struct): {q_difference:.4f}")
    print(f"   📊 Community Alignment NMI: {nmi:.3f}")
    print(f"   📊 Community Alignment ARI: {ari:.3f}")
    
    # 4. Interpretation
    if nmi > 0.7:
        alignment_strength = "HIGH"
    elif nmi > 0.3:
        alignment_strength = "MODERATE"
    else:
        alignment_strength = "LOW"
    
    print(f"   📊 Alignment Strength: {alignment_strength}")
    
    return {
        'nmi': nmi,
        'ari': ari,
        'q_functional': func_q,
        'q_structural': struct_q,
        'q_difference': q_difference,
        'q_ratio': q_ratio,
        'alignment_strength': alignment_strength
    }
```

### **Step 6: Level-Specific Visualization**
```python
# Generate heatmap for this level's FC matrix
plot_fc_heatmap(fc_matrix_level_k, title=f"Level {k} FC Matrix")
# Save community assignments for this level
save_community_structure(functional_communities_k, level=k)
```

## **📊 COMPLETE OUTPUT PER LEVEL:**
```json
{
  "level_0": {
    "noise_condition": "clean_baseline",
    "Q_functional": 0.234,
    "Q_structural": 0.156,
    "num_functional_communities": 3,
    "num_structural_communities": 2,
    "community_alignment_nmi": 0.678,
    "community_alignment_ari": 0.543,
    "fc_matrix": [[...], [...], ...],
    "functional_communities": {0: 0, 1: 0, 2: 1, 3: 1, ...},
    "structural_communities": {0: 0, 1: 0, 2: 1, 3: 1, ...}
  },
  "level_1": {
    "noise_condition": "perturbation_0.2",
    "Q_functional": 0.345,
    "Q_structural": 0.156,  // Same structural Q for all levels
    "num_functional_communities": 4,
    // ... complete analysis for level 1
  },
  "level_2": {
    "noise_condition": "perturbation_0.4", 
    "Q_functional": 0.412,
    "Q_structural": 0.156,  // Same structural Q for all levels
    "num_functional_communities": 5,
    // ... complete analysis for level 2
  }
}
```

---

### **Phase 5: Structural vs Functional Comparison** (Optional)
**Goal**: Compare functional modularity with underlying structural topology.

**Method** (following Ellefsen 2015):
1. **Structural graph**: Build graph from weight magnitudes
   ```python
   structural_graph[i,j] = abs(model.weights[i,j])
   ```
2. **Structural modularity**: Apply Louvain to structural graph
3. **Compare**: Functional Q vs Structural Q

**Output**: Structural/functional modularity comparison

---

## **🔬 Expected Results & Validation**

### **Hypothesis Testing**:
1. **Modular > Hybrid > Standard MLP**: Functional modularity should decrease with structural modularity
2. **Continual learning adaptation**: Functional modularity reflects how the model adapted to the full training sequence
3. **Level-specific specialization**: Different levels may show different functional community structures as the model learned to handle increasing difficulty

## **🔍 KEY INSIGHTS FROM PER-LEVEL ANALYSIS:**

### **What Each Level Tells Us:**
- **Level 0 (Clean)**: Baseline functional organization without noise
- **Level 1 (Low Noise)**: How functional communities adapt to mild perturbations
- **Level 2 (Medium Noise)**: Functional specialization under moderate difficulty
- **Level 3+ (High Noise)**: Maximum functional modularity under high difficulty

### **Expected Patterns:**
- **Increasing Q_functional**: `Q_0 < Q_1 < Q_2 < Q_3` as task difficulty increases
- **Community Evolution**: Number of functional communities may increase with difficulty
- **Topology Differences**: Modular networks should show stronger level-dependent functional modularity than Standard MLP

### **Critical Analysis Questions:**
1. **Does functional modularity increase with task difficulty?**
2. **Do modular topologies show stronger functional specialization per level?**
3. **How well do functional and structural communities align at each difficulty level?**
4. **Is there consistent functional organization across levels or does it reorganize?**

### **Key Metrics**:
- **Primary**: Modularity Score (Q) per topology per level
- **Secondary**: Number of communities, community stability
- **Validation**: Structural vs functional modularity correlation

### **Expected Ranges**:
- **High modularity**: Q > 0.3 (strong community structure)
- **Medium modularity**: Q = 0.1-0.3 (moderate structure)  
- **Low modularity**: Q < 0.1 (weak/no structure)

---

## **🛠 Implementation Strategy**

### **File Structure**:
```
functional_modularity_analysis.py    # Main analysis script
├── train_and_checkpoint()          # Training with model saving
├── collect_activations()           # Activation recording
├── compute_functional_graph()      # FC matrix computation
├── detect_communities()            # Louvain clustering
└── compare_topologies()            # Cross-topology analysis
```

### **Usage**:
```bash
# Train and analyze single topology
python functional_modularity_analysis.py --train --analyze --topology modular --task CartPole-v1

# Analyze multiple topologies
python functional_modularity_analysis.py --analyze --compare-topologies --task CartPole-v1
```

---

## **📊 Output Format**

### **Results JSON**:
```json
{
  "topology": "modular",
  "task": "CartPole-v1", 
  "levels": [
    {
      "level": 1,
      "noise": 0.0,
      "modularity_score": 0.45,
      "num_communities": 4,
      "community_sizes": [32, 28, 35, 33]
    }
  ]
}
```

### **Visualization**:
- Modularity progression plots (Q vs level)
- Community structure heatmaps
- Topology comparison charts

---

## **🎯 Success Criteria**

1. **Methodological soundness**: Follows established protocols (Tanner, Ellefsen)
2. **Clear differentiation**: Modular topologies show higher functional modularity
3. **Reproducible results**: Consistent patterns across seeds/runs
4. **Interpretable metrics**: Simple Q scores and community counts

This approach is **clean**, **focused**, and **directly tests our core hypothesis** about topology-dependent functional modularity emergence.