#!/usr/bin/env python3
"""
Script to save analysis outputs from the data analysis notebook to files.
This allows you to access the processed data outside of the notebook.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import the analysis functions from the notebook
def load_training_data(base_path: str = "test_experiments/cartpole/N0002/S256/") -> Dict:
    """Load all training data from local folders."""
    if not os.path.exists(base_path):
        print(f"❌ Directory {base_path} not found")
        return {}
    
    topology_groups = {}
    
    # Scan all directories
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue
            
        # Try to load metadata
        metadata_file = os.path.join(item_path, "run_metadata.json")
        if not os.path.exists(metadata_file):
            continue
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Extract topology type and seed
            if 'training_config' in metadata:
                config = metadata['training_config']
                topology_type = config.get('topology_type', 'unknown')
                seed = config.get('seed', 'unknown')
                task_name = config.get('task_name', 'unknown')
                
                # Create topology group key (excluding seed)
                topology_key = f"{task_name}_{topology_type}"
                
                if topology_key not in topology_groups:
                    topology_groups[topology_key] = {
                        'runs': [],
                        'task_name': task_name,
                        'topology_type': topology_type,
                        'metadata': metadata
                    }
                
                # Load episode data
                episode_file = os.path.join(item_path, "data", "episode_data.csv")
                if os.path.exists(episode_file):
                    episode_data = pd.read_csv(episode_file)
                    
                    topology_groups[topology_key]['runs'].append({
                        'seed': seed,
                        'folder': item,
                        'episode_data': episode_data,
                        'metadata': metadata
                    })
                    
        except Exception as e:
            print(f"⚠️  Error loading {item}: {e}")
            continue
    
    # Sort runs by seed for each topology
    for topology_key in topology_groups:
        topology_groups[topology_key]['runs'].sort(key=lambda x: x['seed'])
    
    print(f"✅ Loaded data for {len(topology_groups)} topology types:")
    for topology_key, group in topology_groups.items():
        print(f"   📊 {topology_key}: {len(group['runs'])} seeds")
    
    return topology_groups

def group_episodes_by_iteration(episode_data: pd.DataFrame) -> pd.DataFrame:
    """Group episodes by iteration and calculate mean episode reward per iteration."""
    if 'episode_return_raw' not in episode_data.columns:
        print("⚠️  No episode_return_raw column found")
        return pd.DataFrame()
    
    episode_data = episode_data.copy()
    episode_data['iteration'] = (episode_data.index // 2) + 1
    
    iteration_data = episode_data.groupby('iteration').agg({
        'episode_return_raw': 'mean',
        'episode_length': 'mean',
        'global_step_end': 'max'
    }).reset_index()
    
    return iteration_data

def apply_window_smoothing(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply window smoothing to data."""
    if len(data) < window:
        return data
    
    smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
    padding = np.full(window-1, smoothed[0])
    return np.concatenate([padding, smoothed])

def calculate_confidence_intervals(data_arrays: List[np.ndarray], confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mean and confidence intervals across multiple arrays."""
    if not data_arrays:
        return np.array([]), np.array([]), np.array([])
    
    stacked = np.vstack(data_arrays)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    
    n = len(data_arrays)
    if n > 1:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * std / np.sqrt(n)
    else:
        margin = np.zeros_like(std)
    
    lower_bound = mean - margin
    upper_bound = mean + margin
    
    return mean, lower_bound, upper_bound

def prepare_figure6_data(topology_groups: Dict) -> Dict:
    """Prepare data for Figure6-style plots with multi-seed aggregation."""
    figure6_data = {}
    
    for topology_key, group in topology_groups.items():
        if len(group['runs']) < 2:
            print(f"⚠️  Skipping {topology_key}: Need at least 2 seeds for aggregation")
            continue
        
        seed_rewards = []
        max_iterations = 0
        
        for run in group['runs']:
            episode_data = run['episode_data']
            iteration_data = group_episodes_by_iteration(episode_data)
            
            if not iteration_data.empty and 'episode_return_raw' in iteration_data.columns:
                rewards = iteration_data['episode_return_raw'].values
                seed_rewards.append(rewards)
                max_iterations = max(max_iterations, len(rewards))
            else:
                print(f"⚠️  No valid iteration data in {run['folder']}")
        
        if not seed_rewards:
            continue
        
        # Pad arrays to same length and apply smoothing
        padded_rewards = []
        for rewards in seed_rewards:
            if len(rewards) < max_iterations:
                padded = np.pad(rewards, (0, max_iterations - len(rewards)), mode='edge')
            else:
                padded = rewards[:max_iterations]
            
            smoothed = apply_window_smoothing(padded, window=5)
            padded_rewards.append(smoothed)
        
        # Calculate confidence intervals
        mean_rewards, lower_bound, upper_bound = calculate_confidence_intervals(padded_rewards)
        
        figure6_data[topology_key] = {
            'iterations': np.arange(1, max_iterations + 1),
            'mean_rewards': mean_rewards,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'num_seeds': len(padded_rewards),
            'topology_type': group['topology_type'],
            'task_name': group['task_name']
        }
    
    return figure6_data

def calculate_cumulative_rewards(figure6_data: Dict) -> pd.DataFrame:
    """Calculate cumulative rewards for each topology and create comparison table."""
    cumulative_data = []
    
    for topology_key, data in figure6_data.items():
        iterations = data['iterations']
        mean_rewards = data['mean_rewards']
        
        cumulative_rewards = np.cumsum(mean_rewards)
        
        milestones = [50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        milestone_data = {}
        
        for milestone in milestones:
            if milestone <= len(iterations):
                idx = milestone - 1
                milestone_data[f'iteration_{milestone}'] = cumulative_rewards[idx]
            else:
                milestone_data[f'iteration_{milestone}'] = np.nan
        
        final_cumulative = cumulative_rewards[-1]
        
        row_data = {
            'Topology': data['topology_type'],
            'Task': data['task_name'],
            'Seeds': data['num_seeds'],
            'Total Iterations': len(iterations),
            'Final Cumulative Reward': final_cumulative,
            **milestone_data
        }
        
        cumulative_data.append(row_data)
    
    df = pd.DataFrame(cumulative_data)
    
    base_cols = ['Topology', 'Task', 'Seeds', 'Total Iterations', 'Final Cumulative Reward']
    milestone_cols = [col for col in df.columns if col.startswith('iteration_')]
    milestone_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    df = df[base_cols + milestone_cols]
    return df

def save_analysis_outputs(output_dir: str = "analysis_outputs"):
    """Save all analysis outputs to files."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔄 Loading and processing data...")
    
    # Load data
    topology_groups = load_training_data()
    if not topology_groups:
        print("❌ No data found")
        return
    
    # Process data
    figure6_data = prepare_figure6_data(topology_groups)
    cumulative_df = calculate_cumulative_rewards(figure6_data)
    
    print("💾 Saving analysis outputs...")
    
    # 1. Save raw topology groups data
    raw_data_file = os.path.join(output_dir, "raw_topology_groups.json")
    with open(raw_data_file, 'w') as f:
        # Convert DataFrames to dict for JSON serialization
        serializable_groups = {}
        for key, group in topology_groups.items():
            serializable_groups[key] = {
                'task_name': group['task_name'],
                'topology_type': group['topology_type'],
                'runs': []
            }
            
            for run in group['runs']:
                run_data = {
                    'seed': run['seed'],
                    'folder': run['folder'],
                    'episode_data': run['episode_data'].to_dict('records'),
                    'metadata': run['metadata']
                }
                serializable_groups[key]['runs'].append(run_data)
        
        json.dump(serializable_groups, f, indent=2)
    
    print(f"✅ Raw data saved to: {raw_data_file}")
    
    # 2. Save Figure6 processed data
    figure6_file = os.path.join(output_dir, "figure6_processed_data.json")
    with open(figure6_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_figure6 = {}
        for key, data in figure6_data.items():
            serializable_figure6[key] = {
                'iterations': data['iterations'].tolist(),
                'mean_rewards': data['mean_rewards'].tolist(),
                'lower_bound': data['lower_bound'].tolist(),
                'upper_bound': data['upper_bound'].tolist(),
                'num_seeds': data['num_seeds'],
                'topology_type': data['topology_type'],
                'task_name': data['task_name']
            }
        
        json.dump(serializable_figure6, f, indent=2)
    
    print(f"✅ Figure6 data saved to: {figure6_file}")
    
    # 3. Save cumulative rewards table
    cumulative_file = os.path.join(output_dir, "cumulative_rewards_table.csv")
    cumulative_df.to_csv(cumulative_file, index=False)
    print(f"✅ Cumulative rewards table saved to: {cumulative_file}")
    
    # 4. Save individual seed data for each topology
    seeds_dir = os.path.join(output_dir, "individual_seeds")
    os.makedirs(seeds_dir, exist_ok=True)
    
    for topology_key, group in topology_groups.items():
        topology_seeds_dir = os.path.join(seeds_dir, topology_key.replace(' ', '_'))
        os.makedirs(topology_seeds_dir, exist_ok=True)
        
        for run in group['runs']:
            seed_file = os.path.join(topology_seeds_dir, f"seed_{run['seed']}.csv")
            
            # Save iteration-based data
            iteration_data = group_episodes_by_iteration(run['episode_data'])
            if not iteration_data.empty:
                iteration_data.to_csv(seed_file, index=False)
        
        print(f"✅ Individual seed data saved to: {topology_seeds_dir}")
    
    # 5. Create summary report
    summary_file = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("ANALYSIS OUTPUT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Topologies: {len(topology_groups)}\n")
        f.write(f"Total Seeds: {sum(len(group['runs']) for group in topology_groups.values())}\n\n")
        
        f.write("Topology Performance Rankings:\n")
        if not cumulative_df.empty:
            rankings = cumulative_df.sort_values('Final Cumulative Reward', ascending=False)
            for i, (_, row) in enumerate(rankings.iterrows()):
                f.write(f"{i+1}. {row['Topology']}: {row['Final Cumulative Reward']:.2f}\n")
        
        f.write(f"\nFiles Generated:\n")
        f.write(f"- {raw_data_file}\n")
        f.write(f"- {figure6_file}\n")
        f.write(f"- {cumulative_file}\n")
        f.write(f"- {seeds_dir}/ (individual seed data)\n")
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    print(f"\n🎉 All analysis outputs saved to: {output_dir}/")
    print("\n📁 File Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── raw_topology_groups.json          # Raw loaded data")
    print(f"   ├── figure6_processed_data.json       # Processed iteration data")
    print(f"   ├── cumulative_rewards_table.csv      # Performance comparison table")
    print(f"   ├── analysis_summary.txt              # Summary report")
    print(f"   └── individual_seeds/                 # Per-seed data")
    print(f"       ├── CartPole-v1_fully_connected/")
    print(f"       ├── CartPole-v1_small_world/")
    print(f"       └── ...")

if __name__ == "__main__":
    save_analysis_outputs()
