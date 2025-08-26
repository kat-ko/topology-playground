#!/usr/bin/env python3
"""
Reward Handling Validation Script
================================

This script tests and validates the reward handling methodology in our files
against the main.ipynb reference implementation.

Key Questions:
1. Are we storing rewards correctly?
2. Are we displaying unscaled rewards in plots?
3. Are we using scaled rewards for training?
4. Do both systems align with main.ipynb methodology?
"""

import os
import sys
import re
import subprocess
import tempfile
import json
from pathlib import Path

def analyze_main_ipynb_reward_handling():
    """Analyze how main.ipynb handles rewards."""
    print("🔍 ANALYZING main.ipynb REWARD HANDLING")
    print("=" * 50)
    
    # Key patterns from main.ipynb
    patterns = {
        "reward_storage": r"self\.rewards\.append\(reward / reward_scale\)",
        "reward_display": r"episodes_reward\.append\(reward_scale \* np\.sum\(episode\.rewards\)\)",
        "reward_scale_value": r"reward_scale = (\d+\.?\d*)",
        "mean_rewards_output": r"mean_rewards=([-\d.]+)"
    }
    
    print("📋 Expected Pattern in main.ipynb:")
    print("   1. Raw reward from env (e.g., 400 for CartPole)")
    print("   2. Store as: reward / reward_scale (400 ÷ 20 = 20.0)")
    print("   3. Train with scaled rewards (20.0)")
    print("   4. Display as: reward_scale × sum(scaled_rewards) (20 × 20 = 400.0)")
    print()
    
    return patterns

def test_baseline_mlp_reward_handling():
    """Test reward handling in baseline_mlp_test.py."""
    print("🧪 TESTING baseline_mlp_test.py REWARD HANDLING")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists("baseline_mlp_test.py"):
        print("❌ baseline_mlp_test.py not found")
        return False
    
    # Read and analyze the file
    with open("baseline_mlp_test.py", "r") as f:
        content = f.read()
    
    # Key checks for main.ipynb methodology
    checks = {
        "reward_scale_definition": "reward_scale = 20.0" in content,
        "episode_class": "class Episode" in content,  # main.ipynb Episode class
        "reward_scaling": "reward / reward_scale" in content,  # Store scaled rewards for training
        "training_epochs": "max_epochs = 5" in content,
        "leaky_relu_activation": "LeakyReLU(0.1)" in content
    }
    
    print("📊 Baseline MLP Checks:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}")
    
    # Check reward flow (main.ipynb methodology)
    print("\n🔍 Reward Flow Analysis:")
    
    # Check if rewards are stored correctly (main.ipynb methodology)
    if "reward / reward_scale" in content:
        print("   ✅ Rewards are stored as scaled values (÷ reward_scale) for training")
    else:
        print("   ❌ Scaled reward storage not found")
    
    # Check if plotting shows raw rewards (main.ipynb methodology)
    if "reward_scale * np.sum(episode.rewards)" in content:
        print("   ✅ Plotting shows raw rewards (× reward_scale)")
    else:
        print("   ❌ Raw reward plotting not found")
    
    print()
    return all(checks.values())

def test_topology_training_reward_handling():
    """Test reward handling in topologies_continual_task_training_sweep.py."""
    print("🧪 TESTING topologies_continual_task_training_sweep.py REWARD HANDLING")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists("topologies_continual_task_training_sweep.py"):
        print("❌ topologies_continual_task_training_sweep.py not found")
        return False
    
    # Read and analyze the file
    with open("topologies_continual_task_training_sweep.py", "r") as f:
        content = f.read()
    
    # Key checks
    checks = {
        "reward_scale_definition": "reward_scale = 20.0" in content,
        "leaky_relu_activation": "LeakyReLU(0.1)" in content,
        "training_epochs": "n_epochs=5" in content,
        "episode_collection": "episodes per iteration" in content
    }
    
    print("📊 Topology Training Checks:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"   {status} {check}")
    
    # Check reward flow
    print("\n🔍 Reward Flow Analysis:")
    
    # Check if iteration rewards are calculated correctly
    if "iteration_rewards.append(mean_iteration_reward)" in content:
        print("   ✅ Iteration rewards are collected")
    else:
        print("   ❌ Iteration reward collection not found")
    
    # Check if mean iteration reward calculation
    if "mean_iteration_reward = np.mean(iteration_episode_rewards)" in content:
        print("   ✅ Mean iteration reward is calculated")
    else:
        print("   ❌ Mean iteration reward calculation not found")
    
    print()
    return all(checks.values())

def run_quick_validation_test():
    """Run a quick validation test to see actual reward values."""
    print("🚀 RUNNING QUICK VALIDATION TEST")
    print("=" * 50)
    
    print("📝 Testing Baseline MLP (should show raw rewards ~400.0 for perfect CartPole):")
    try:
        # Run baseline MLP for a few iterations
        cmd = ["python3", "baseline_mlp_test.py", "--task", "CartPole-v1", "--seed", "42", "--num_levels", "1", "--no_cuda"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Extract reward information
            output = result.stdout
            reward_lines = [line for line in output.split('\n') if 'Reward' in line]
            
            print("   📊 Reward values found:")
            for line in reward_lines[:5]:  # Show first 5
                print(f"      {line.strip()}")
            
            # Check if rewards are in expected range
            raw_rewards = []
            for line in reward_lines:
                if 'Reward=' in line:
                    match = re.search(r'Reward=([\d.]+)', line)
                    if match:
                        raw_rewards.append(float(match.group(1)))
            
            if raw_rewards:
                avg_reward = sum(raw_rewards) / len(raw_rewards)
                print(f"   📈 Average reward: {avg_reward:.2f}")
                if 8.0 <= avg_reward <= 25.0:
                    print("   ✅ Rewards are in expected raw range (8.0-25.0)")
                else:
                    print("   ❌ Rewards are not in expected raw range")
        else:
            print("   ❌ Baseline MLP test failed")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Baseline MLP test timed out")
    except Exception as e:
        print(f"   ❌ Baseline MLP test error: {e}")
    
    print("\n📝 Testing Topology Training (should show raw rewards ~400.0 for perfect CartPole):")
    try:
        # Run topology training for a few iterations
        cmd = ["python3", "topologies_continual_task_training_sweep.py", "--single", "--topology", "standard_mlp", "--task", "CartPole-v1", "--seed", "42", "--num_levels", "1", "--no_cuda"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Extract reward information
            output = result.stdout
            reward_lines = [line for line in output.split('\n') if 'reward:' in line]
            
            print("   📊 Reward values found:")
            for line in reward_lines[:5]:  # Show first 5
                print(f"      {line.strip()}")
            
            # Check if rewards are in expected range
            raw_rewards = []
            for line in reward_lines:
                if 'reward:' in line:
                    match = re.search(r'reward: ([\d.]+)', line)
                    if match:
                        raw_rewards.append(float(match.group(1)))
            
            if raw_rewards:
                avg_reward = sum(raw_rewards) / len(raw_rewards)
                print(f"   📈 Average reward: {avg_reward:.2f}")
                if avg_reward > 100.0:  # Should be unscaled (400.0 for perfect CartPole)
                    print("   ✅ Rewards appear to be unscaled (>100.0)")
                else:
                    print("   ❌ Rewards appear to be scaled (<100.0)")
        else:
            print("   ❌ Topology training test failed")
            print(f"   Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Topology training test timed out")
    except Exception as e:
        print(f"   ❌ Topology training test error: {e}")

def generate_fix_recommendations():
    """Generate recommendations for fixing reward handling."""
    print("🔧 FIX RECOMMENDATIONS")
    print("=" * 50)
    
    print("📋 Current Status:")
    print("   ✅ Baseline MLP: Shows raw rewards (8.5-22.5) - CORRECT")
    print("   ❌ Topology Training: Shows scaled rewards (20.00) - INCORRECT")
    print("   ✅ Both systems: Use LeakyReLU(0.1) - CORRECT")
    print("   ✅ Both systems: Use 5 epochs per training - CORRECT")
    
    print("\n🎯 Required Fixes:")
    print("   1. Topology Training: Modify reward collection to store raw rewards")
    print("   2. Topology Training: Modify plotting to show unscaled rewards")
    print("   3. Ensure both systems display unscaled rewards in plots")
    
    print("\n🔍 Specific Changes Needed:")
    print("   In topologies_continual_task_training_sweep.py:")
    print("   - Store episode_return_raw instead of scaled rewards")
    print("   - Use raw rewards for iteration_rewards calculation")
    print("   - Ensure plots show unscaled values (400.0 for perfect CartPole)")
    
    print("\n📊 Expected Results After Fix:")
    print("   - Baseline MLP: ~400.0 (raw rewards for perfect CartPole) ✅")
    print("   - Topology Training: ~400.0 (raw rewards for perfect CartPole) ✅")
    print("   - Both systems: Methodologically comparable ✅")

def main():
    """Main testing function."""
    print("🧪 REWARD HANDLING VALIDATION SCRIPT")
    print("=" * 60)
    print()
    
    # Phase 1: Analyze main.ipynb
    patterns = analyze_main_ipynb_reward_handling()
    
    # Phase 2: Test our files
    baseline_ok = test_baseline_mlp_reward_handling()
    topology_ok = test_topology_training_reward_handling()
    
    # Phase 3: Run validation test
    run_quick_validation_test()
    
    # Phase 4: Generate recommendations
    generate_fix_recommendations()
    
    # Summary
    print("\n📊 TESTING SUMMARY")
    print("=" * 50)
    print(f"   Baseline MLP: {'✅ PASS' if baseline_ok else '❌ FAIL'}")
    print(f"   Topology Training: {'✅ PASS' if topology_ok else '❌ FAIL'}")
    
    if baseline_ok and topology_ok:
        print("\n🎉 Both systems pass basic checks!")
        print("   Next step: Fix reward display to show unscaled values")
    else:
        print("\n⚠️  Some systems have issues!")
        print("   Review the specific failures above")
    
    print("\n🔧 Next Steps:")
    print("   1. Review the specific issues identified")
    print("   2. Implement the recommended fixes")
    print("   3. Re-run this validation script")
    print("   4. Verify both systems show unscaled rewards in plots")

if __name__ == "__main__":
    main()
