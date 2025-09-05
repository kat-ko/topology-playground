#!/usr/bin/env python3
"""
Script to check data quality and investigate the high variance in results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_episode_data_quality(base_path: str = "test_experiments/cartpole/N0002/S256/"):
    """Check the quality of episode data to understand high variance."""
    
    print("🔍 DATA QUALITY INVESTIGATION")
    print("=" * 50)
    
    if not os.path.exists(base_path):
        print(f"❌ Directory {base_path} not found")
        return
    
    # Check a few sample runs
    sample_runs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            episode_file = os.path.join(item_path, "data", "episode_data.csv")
            if os.path.exists(episode_file):
                sample_runs.append((item, episode_file))
                if len(sample_runs) >= 3:  # Check first 3 runs
                    break
    
    for run_name, episode_file in sample_runs:
        print(f"\n📊 Analyzing: {run_name}")
        print("-" * 40)
        
        try:
            df = pd.read_csv(episode_file)
            print(f"   Total episodes: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            if 'episode_return_raw' in df.columns:
                rewards = df['episode_return_raw']
                print(f"   Reward range: {rewards.min():.2f} to {rewards.max():.2f}")
                print(f"   Reward mean: {rewards.mean():.2f}")
                print(f"   Reward std: {rewards.std():.2f}")
                print(f"   Reward CV: {rewards.std()/rewards.mean():.3f}")
                
                # Check for outliers
                q1, q3 = rewards.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = rewards[(rewards < q1 - 1.5*iqr) | (rewards > q3 + 1.5*iqr)]
                print(f"   Outliers: {len(outliers)} ({len(outliers)/len(rewards)*100:.1f}%)")
                
                # Check first and last 10 episodes
                print(f"   First 10 episodes: {rewards.head(10).tolist()}")
                print(f"   Last 10 episodes: {rewards.tail(10).tolist()}")
                
                # Check if there are any obvious patterns
                if len(rewards) > 100:
                    early_mean = rewards.head(100).mean()
                    late_mean = rewards.tail(100).mean()
                    print(f"   Early episodes (1-100) mean: {early_mean:.2f}")
                    print(f"   Late episodes (last 100) mean: {late_mean:.2f}")
                    print(f"   Learning improvement: {late_mean - early_mean:.2f}")
            
            if 'episode_length' in df.columns:
                lengths = df['episode_length']
                print(f"   Episode length range: {lengths.min()} to {lengths.max()}")
                print(f"   Episode length mean: {lengths.mean():.2f}")
                
        except Exception as e:
            print(f"   ❌ Error reading {episode_file}: {e}")

def check_iteration_grouping(base_path: str = "test_experiments/cartpole/N0002/S256/"):
    """Check how iteration grouping affects the results."""
    
    print("\n🔄 ITERATION GROUPING INVESTIGATION")
    print("=" * 50)
    
    def group_episodes_by_iteration(episode_data: pd.DataFrame) -> pd.DataFrame:
        if 'episode_return_raw' not in episode_data.columns:
            return pd.DataFrame()
        
        episode_data = episode_data.copy()
        episode_data['iteration'] = (episode_data.index // 2) + 1
        
        iteration_data = episode_data.groupby('iteration').agg({
            'episode_return_raw': 'mean',
            'episode_length': 'mean',
            'global_step_end': 'max'
        }).reset_index()
        
        return iteration_data
    
    # Check one sample run
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            episode_file = os.path.join(item_path, "data", "episode_data.csv")
            if os.path.exists(episode_file):
                print(f"\n📊 Sample run: {item}")
                print("-" * 30)
                
                df = pd.read_csv(episode_file)
                iteration_data = group_episodes_by_iteration(df)
                
                if not iteration_data.empty:
                    print(f"   Total iterations: {len(iteration_data)}")
                    print(f"   Iteration rewards range: {iteration_data['episode_return_raw'].min():.2f} to {iteration_data['episode_return_raw'].max():.2f}")
                    print(f"   Iteration rewards mean: {iteration_data['episode_return_raw'].mean():.2f}")
                    print(f"   Iteration rewards std: {iteration_data['episode_return_raw'].std():.2f}")
                    print(f"   Iteration rewards CV: {iteration_data['episode_return_raw'].std()/iteration_data['episode_return_raw'].mean():.3f}")
                    
                    # Show first and last iterations
                    print(f"   First 5 iterations: {iteration_data['episode_return_raw'].head(5).tolist()}")
                    print(f"   Last 5 iterations: {iteration_data['episode_return_raw'].tail(5).tolist()}")
                
                break  # Just check one sample

def check_metadata_consistency(base_path: str = "test_experiments/cartpole/N0002/S256/"):
    """Check if metadata is consistent across runs."""
    
    print("\n📋 METADATA CONSISTENCY CHECK")
    print("=" * 50)
    
    metadata_samples = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            metadata_file = os.path.join(item_path, "run_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata_samples.append((item, metadata))
                    if len(metadata_samples) >= 3:
                        break
                except Exception as e:
                    print(f"   ❌ Error reading {metadata_file}: {e}")
    
    for run_name, metadata in metadata_samples:
        print(f"\n📊 {run_name}")
        print("-" * 30)
        
        if 'training_config' in metadata:
            config = metadata['training_config']
            print(f"   Task: {config.get('task_name', 'unknown')}")
            print(f"   Topology: {config.get('topology_type', 'unknown')}")
            print(f"   Seed: {config.get('seed', 'unknown')}")
            print(f"   Iterations: {config.get('iterations', 'unknown')}")
            print(f"   Episode cap: {config.get('episode_cap', 'unknown')}")

if __name__ == "__main__":
    check_episode_data_quality()
    check_iteration_grouping()
    check_metadata_consistency()
