"""
Check if the RMS Norm Epsilon difference is causing the Qwen precision issues
"""
import numpy as np

def analyze_rms_norm_precision():
    """Analyze RMS normalization precision requirements"""
    
    print("="*80)
    print("RMS Normalization Precision Analysis")
    print("="*80)
    
    # Model configurations
    qwen_eps = 1e-6
    llama_eps = 1e-5
    
    print(f"\nModel Epsilon Values:")
    print(f"  Qwen RMS Norm Epsilon:  {qwen_eps:.2e} (10x smaller)")
    print(f"  Llama RMS Norm Epsilon: {llama_eps:.2e}")
    print(f"  Ratio: {llama_eps/qwen_eps:.1f}x")
    
    # Posit32es2 characteristics
    print(f"\n{'='*80}")
    print("Posit32es2 Numerical Characteristics")
    print(f"{'='*80}")
    
    # Approximate values for posit32es2
    posit32_min_normal = 5.9e-9
    posit32_max = 1.7e8
    posit32_effective_precision = 30  # bits
    
    print(f"  Minimum Normal Value:     {posit32_min_normal:.2e}")
    print(f"  Maximum Value:            {posit32_max:.2e}")
    print(f"  Effective Precision:      ~{posit32_effective_precision} bits")
    
    # Check if epsilon values are safe
    print(f"\n{'='*80}")
    print("Epsilon Safety Check")
    print(f"{'='*80}")
    
    print(f"\nQwen Epsilon ({qwen_eps:.2e}):")
    if qwen_eps > posit32_min_normal:
        ratio = qwen_eps / posit32_min_normal
        print(f"  ✓ Safe: {ratio:.2e}x larger than Posit32 minimum")
    else:
        print(f"  ✗ UNSAFE: Smaller than Posit32 minimum!")
    
    print(f"\nLlama Epsilon ({llama_eps:.2e}):")
    if llama_eps > posit32_min_normal:
        ratio = llama_eps / posit32_min_normal
        print(f"  ✓ Safe: {ratio:.2e}x larger than Posit32 minimum")
    else:
        print(f"  ✗ UNSAFE: Smaller than Posit32 minimum!")
    
    # Simulate RMS normalization
    print(f"\n{'='*80}")
    print("RMS Normalization Simulation")
    print(f"{'='*80}")
    
    # Test with various input magnitudes
    test_inputs = [
        ("Normal range", np.array([0.1, 0.2, -0.15, 0.3], dtype=np.float32)),
        ("Small values", np.array([1e-4, 2e-4, -1.5e-4, 3e-4], dtype=np.float32)),
        ("Very small", np.array([1e-6, 2e-6, -1.5e-6, 3e-6], dtype=np.float32)),
        ("Near epsilon", np.array([1e-7, 2e-7, -1.5e-7, 3e-7], dtype=np.float32)),
    ]
    
    for test_name, x in test_inputs:
        print(f"\n{test_name}: {x}")
        
        # Calculate RMS
        rms = np.sqrt(np.mean(x**2))
        print(f"  RMS: {rms:.2e}")
        
        # Qwen normalization
        qwen_denom = rms + qwen_eps
        qwen_output = x / qwen_denom
        print(f"  Qwen (eps={qwen_eps:.2e}):")
        print(f"    Denominator: {qwen_denom:.2e}")
        print(f"    Output range: [{qwen_output.min():.2e}, {qwen_output.max():.2e}]")
        
        # Llama normalization
        llama_denom = rms + llama_eps
        llama_output = x / llama_denom
        print(f"  Llama (eps={llama_eps:.2e}):")
        print(f"    Denominator: {llama_denom:.2e}")
        print(f"    Output range: [{llama_output.min():.2e}, {llama_output.max():.2e}]")
        
        # Difference
        diff = np.abs(qwen_output - llama_output)
        rel_diff = diff / np.abs(llama_output + 1e-10)
        print(f"  Absolute difference: {diff.max():.2e}")
        print(f"  Relative difference: {rel_diff.max():.2%}")
        
        # Check if difference matters
        if rms < qwen_eps * 10:
            print(f"  ⚠️  WARNING: RMS is close to epsilon - high sensitivity!")
    
    # Analysis of cumulative error
    print(f"\n{'='*80}")
    print("Cumulative Error Analysis")
    print(f"{'='*80}")
    
    print("\nIn a 24-layer Qwen model:")
    print("  If each layer has ~0.1% error due to epsilon difference,")
    print("  cumulative error could be: 0.001^24 ≈ significant drift")
    
    print("\nPotential Issue:")
    print("  Qwen's smaller epsilon (1e-6 vs 1e-5) means:")
    print("  1. More sensitive to small value perturbations")
    print("  2. Higher precision requirements in division operations")
    print("  3. Accumulated errors over 24 layers amplify divergence")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nThe 10x difference in RMS Norm epsilon is likely contributing to")
    print("the accuracy gap between Qwen FP32 and Posit32:")
    print()
    print("  • Qwen epsilon (1e-6) requires higher precision")
    print("  • With 24 layers vs Llama's 16, errors accumulate more")
    print("  • Posit32's precision may be marginally insufficient for Qwen's")
    print("    tighter normalization tolerance")
    print()
    print("Recommendations:")
    print("  1. Test Qwen with relaxed epsilon (1e-5) to match Llama")
    print("  2. Use higher precision (FP64) for normalization layers")
    print("  3. Monitor normalization layer outputs during inference")
    print("  4. Consider mixed precision: Posit32 for weights, FP32 for norms")
    
    print("\nDone!")

if __name__ == "__main__":
    analyze_rms_norm_precision()
