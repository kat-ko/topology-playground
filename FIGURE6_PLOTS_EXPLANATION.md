# Figure-6 Style Plots Explanation

## Overview
These plots replicate the continual learning analysis from the reference paper, showing how network topologies adapt to piecewise-constant observation shifts over a long training run.

## **Corrected Understanding: Iteration-Based Training**

### **X-Axis: Environment Steps (0 to ~2,400,000)**
- **Total Scale**: 3,000 iterations × ~800 env-steps = ~2.4M environment steps
- **Perturbation Boundaries**: Vertical markers every **160,000 environment steps**
- **Why 160,000?**: Each perturbation level lasts 200 iterations × ~800 env-steps

### **Y-Axis: Raw Episode Returns**
- **CartPole-v1**: 0-500 (raw environment rewards)
- **Acrobot-v1**: -500 to 0 (raw environment rewards)  
- **LunarLander-v2**: -200 to 200 (raw environment rewards)

### **Key Insight: Iterations ≠ Environment Steps**
- **Outer Loop**: 3,000 iterations (not 3,000 env-steps!)
- **Per Iteration**: ~800 environment steps (2 episodes × 400 max steps)
- **Total**: ~2.4 million environment steps

## **Perturbation Schedule (Corrected)**
- **Level 0 (0-199 iterations)**: Clean baseline, **NO NOISE**
- **Level 1 (200-399 iterations)**: First perturbation applied
- **Level 2 (400-599 iterations)**: Second perturbation applied
- **...and so on...**

**Each level lasts 200 iterations ≈ 160,000 environment steps**

## **Reward Scaling (Corrected)**
- **Training**: Rewards divided by 20 (smaller gradients)
- **Logging**: Raw returns shown (×20 applied back)
- **Plots**: Display actual environment performance, not scaled values
