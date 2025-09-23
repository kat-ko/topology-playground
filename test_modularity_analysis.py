#!/usr/bin/env python3
"""
Test script for functional modularity analysis
"""

import subprocess
import time
import os

def test_training_and_analysis():
    """Test the complete pipeline: training + analysis"""
    
    print("🧪 Testing Functional Modularity Analysis Pipeline")
    print("=" * 60)
    
    # Test parameters
    topology = "standard_mlp"
    task = "CartPole-v1"
    seed = 42
    num_levels = 3
    
    print(f"📋 Test Configuration:")
    print(f"   Topology: {topology}")
    print(f"   Task: {task}")
    print(f"   Seed: {seed}")
    print(f"   Levels: {num_levels}")
    
    # Step 1: Train with model saving
    print(f"\n🚀 Step 1: Training with model saving...")
    train_cmd = [
        "python", "topologies_continual_task_training_normal_modularity.py",
        "--single", "--save_model", "--no_wandb",
        "--topology", topology,
        "--task", task,
        "--seed", str(seed),
        "--num_levels", str(num_levels)
    ]
    
    print(f"   Command: {' '.join(train_cmd)}")
    
    try:
        result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("   ✅ Training completed successfully!")
            print("   📁 Checking for saved model...")
            
            # Check if model was saved
            model_path = f"modularity_checkpoints/{topology}_{task}_seed{seed}/final_model.zip"
            metadata_path = f"modularity_checkpoints/{topology}_{task}_seed{seed}/training_metadata.json"
            
            if os.path.exists(model_path) and os.path.exists(metadata_path):
                print(f"   ✅ Model saved: {model_path}")
                print(f"   ✅ Metadata saved: {metadata_path}")
            else:
                print(f"   ❌ Model files not found!")
                return False
                
        else:
            print("   ❌ Training failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Training timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Training error: {e}")
        return False
    
    # Step 2: Run analysis
    print(f"\n🧠 Step 2: Running functional modularity analysis...")
    analysis_cmd = [
        "python", "topologies_continual_task_training_normal_modularity.py",
        "--single", "--analysis", "--no_wandb",
        "--topology", topology,
        "--task", task,
        "--seed", str(seed),
        "--num_levels", str(num_levels)
    ]
    
    print(f"   Command: {' '.join(analysis_cmd)}")
    
    try:
        result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        if result.returncode == 0:
            print("   ✅ Analysis completed successfully!")
            
            # Check if analysis results were saved
            analysis_path = f"modularity_checkpoints/{topology}_{task}_seed{seed}/modularity_analysis.json"
            
            if os.path.exists(analysis_path):
                print(f"   ✅ Analysis results saved: {analysis_path}")
                
                # Show a preview of results
                import json
                with open(analysis_path, 'r') as f:
                    results = json.load(f)
                
                print(f"\n📊 RESULTS PREVIEW:")
                print(f"   Levels analyzed: {len(results['level_results'])}")
                for level_result in results['level_results']:
                    print(f"   Level {level_result['level']}: Q={level_result['modularity_score']:.4f}, Communities={level_result['num_communities']}")
                
                return True
            else:
                print(f"   ❌ Analysis results not found!")
                return False
                
        else:
            print("   ❌ Analysis failed!")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   ⏰ Analysis timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False

if __name__ == "__main__":
    success = test_training_and_analysis()
    
    if success:
        print(f"\n🎉 PIPELINE TEST SUCCESSFUL!")
        print(f"   Both training and analysis completed without errors.")
    else:
        print(f"\n❌ PIPELINE TEST FAILED!")
        print(f"   Check the error messages above.")
