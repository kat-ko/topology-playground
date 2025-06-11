import torch
import numpy as np
from scipy.optimize import curve_fit
from src.utils.parameter_budget import ParameterBudgetCalculator
from config.test_curriculum_config import TestCurriculumConfig

def analyze_scaling_patterns():
    """Analyze parameter scaling patterns for different topologies and sizes."""
    # Create test configuration
    config = TestCurriculumConfig().to_dict()
    
    # Test sizes
    sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Test topologies
    topologies = ['small_world', 'modular', 'hybrid', 'fully_connected']
    
    # Create calculator
    calculator = ParameterBudgetCalculator(config)
    
    print("\nAnalyzing Parameter Scaling Patterns")
    print("===================================")
    
    for topology in topologies:
        print(f"\nTopology: {topology}")
        print("-" * 50)
        print("Size | Parameters | Scaling Factor")
        print("-" * 50)
        
        # Get parameters for each size
        params_list = []
        for size in sizes:
            # Create network
            if topology == 'small_world':
                network = calculator._create_sample_small_world(size)
            elif topology == 'modular':
                network = calculator._create_sample_modular(size)
            elif topology == 'hybrid':
                network = calculator._create_sample_hybrid(size)
            elif topology == 'fully_connected':
                network = calculator._create_sample_fully_connected(size)
            
            # Count parameters
            params = sum(p.numel() for p in network.parameters() if p.requires_grad)
            params_list.append(params)
            
            # Calculate scaling factor relative to size
            if topology == 'fully_connected':
                # Quadratic scaling
                scaling = params / (size * size)
            elif topology == 'small_world':
                # Linear scaling with k
                k = max(2, size // 10)
                scaling = params / (size * k)
            elif topology == 'modular':
                # Module-based scaling
                num_modules = max(2, size // 20)
                module_size = size // num_modules
                scaling = params / (size * module_size)
            elif topology == 'hybrid':
                # Hybrid scaling
                k = max(2, size // 10)
                num_modules = max(2, size // 20)
                module_size = size // num_modules
                scaling = params / (size * k + k * module_size)
            
            print(f"{size:4d} | {params:10d} | {scaling:.2f}")
            
            # Print network structure
            print("\nNetwork Structure:")
            for name, param in network.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.shape}")
            print()
        
        # Empirical fitting for small_world and hybrid
        if topology in ['small_world', 'hybrid']:
            # Define the function to fit: params = a * size^b
            def power_law(x, a, b):
                return a * np.power(x, b)
            
            # Fit the curve
            popt, pcov = curve_fit(power_law, sizes, params_list)
            a, b = popt
            
            print(f"Empirical fit: params = {a:.2f} * size^{b:.2f}")
            print(f"R-squared: {1 - np.sum((params_list - power_law(sizes, a, b))**2) / np.sum((params_list - np.mean(params_list))**2):.4f}")

if __name__ == "__main__":
    analyze_scaling_patterns() 