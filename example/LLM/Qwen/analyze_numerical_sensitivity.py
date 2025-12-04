"""
Analyze numerical sensitivity of Qwen vs Llama models
This script examines key numerical characteristics that may cause precision issues
"""
import numpy as np
from transformers import AutoConfig
import json

def analyze_model_config(model_name, config_name):
    """Analyze model configuration for numerical sensitivity indicators"""
    print(f"\n{'='*80}")
    print(f"Analyzing {model_name} Model Configuration")
    print(f"{'='*80}")
    
    config = AutoConfig.from_pretrained(config_name)
    
    # Key parameters that affect numerical stability
    params = {
        "Model Architecture": model_name,
        "Hidden Size": config.hidden_size,
        "Num Layers": config.num_hidden_layers,
        "Num Attention Heads": config.num_attention_heads,
        "Num Key-Value Heads": config.num_key_value_heads,
        "Head Dim": config.hidden_size // config.num_attention_heads,
        "Intermediate Size": config.intermediate_size,
        "Vocab Size": config.vocab_size,
        "Max Position Embeddings": config.max_position_embeddings,
    }
    
    # Architecture-specific parameters
    if hasattr(config, 'rms_norm_eps'):
        params["RMS Norm Epsilon"] = config.rms_norm_eps
    if hasattr(config, 'layer_norm_epsilon'):
        params["Layer Norm Epsilon"] = config.layer_norm_epsilon
    if hasattr(config, 'rope_theta'):
        params["RoPE Theta"] = config.rope_theta
    if hasattr(config, 'attention_dropout'):
        params["Attention Dropout"] = config.attention_dropout
    if hasattr(config, 'hidden_act'):
        params["Activation Function"] = config.hidden_act
    if hasattr(config, 'tie_word_embeddings'):
        params["Tie Word Embeddings"] = config.tie_word_embeddings
    
    # Print parameters
    for key, value in params.items():
        print(f"  {key:.<40} {value}")
    
    # Calculate derived metrics that indicate numerical sensitivity
    print(f"\n{'-'*80}")
    print("Numerical Sensitivity Indicators:")
    print(f"{'-'*80}")
    
    # 1. Attention scaling factor
    head_dim = config.hidden_size // config.num_attention_heads
    attention_scale = 1.0 / np.sqrt(head_dim)
    print(f"  Attention Scale Factor (1/sqrt(head_dim)):... {attention_scale:.6f}")
    
    # 2. GQA ratio (affects KV cache precision requirements)
    gqa_ratio = config.num_attention_heads / config.num_key_value_heads
    print(f"  GQA Ratio (Q heads / KV heads):............. {gqa_ratio:.2f}")
    
    # 3. FFN expansion ratio
    ffn_ratio = config.intermediate_size / config.hidden_size
    print(f"  FFN Expansion Ratio:........................ {ffn_ratio:.2f}")
    
    # 4. Parameter count estimation
    embedding_params = config.vocab_size * config.hidden_size
    attention_params_per_layer = 4 * config.hidden_size * config.hidden_size
    ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size
    layer_params = config.num_hidden_layers * (attention_params_per_layer + ffn_params_per_layer)
    total_params = embedding_params + layer_params
    print(f"  Estimated Total Parameters:................. {total_params/1e6:.2f}M")
    
    # 5. Normalization epsilon (smaller = more sensitive to numerical errors)
    if hasattr(config, 'rms_norm_eps'):
        norm_eps = config.rms_norm_eps
        print(f"  Normalization Epsilon (RMS):................ {norm_eps:.2e}")
        if norm_eps < 1e-6:
            print(f"    ⚠️  WARNING: Very small epsilon - high sensitivity to precision!")
    
    # 6. RoPE theta (affects positional encoding stability)
    if hasattr(config, 'rope_theta'):
        print(f"  RoPE Theta:................................. {config.rope_theta:.1f}")
        if config.rope_theta != 10000.0:
            print(f"    ℹ️  Non-standard RoPE theta - may affect precision requirements")
    
    return config, params

def compare_numerical_characteristics(config_qwen, config_llama):
    """Compare numerical characteristics between models"""
    print(f"\n{'='*80}")
    print("Comparative Analysis: Qwen vs Llama")
    print(f"{'='*80}")
    
    comparisons = []
    
    # Head dimension comparison
    qwen_head_dim = config_qwen.hidden_size // config_qwen.num_attention_heads
    llama_head_dim = config_llama.hidden_size // config_llama.num_attention_heads
    comparisons.append(("Head Dimension", qwen_head_dim, llama_head_dim))
    
    # Attention scale comparison
    qwen_scale = 1.0 / np.sqrt(qwen_head_dim)
    llama_scale = 1.0 / np.sqrt(llama_head_dim)
    comparisons.append(("Attention Scale", qwen_scale, llama_scale))
    
    # GQA ratio comparison
    qwen_gqa = config_qwen.num_attention_heads / config_qwen.num_key_value_heads
    llama_gqa = config_llama.num_attention_heads / config_llama.num_key_value_heads
    comparisons.append(("GQA Ratio", qwen_gqa, llama_gqa))
    
    # Normalization epsilon
    qwen_eps = config_qwen.rms_norm_eps if hasattr(config_qwen, 'rms_norm_eps') else None
    llama_eps = config_llama.rms_norm_eps if hasattr(config_llama, 'rms_norm_eps') else None
    if qwen_eps and llama_eps:
        comparisons.append(("RMS Norm Epsilon", qwen_eps, llama_eps))
    
    # RoPE theta
    qwen_rope = config_qwen.rope_theta if hasattr(config_qwen, 'rope_theta') else None
    llama_rope = config_llama.rope_theta if hasattr(config_llama, 'rope_theta') else None
    if qwen_rope and llama_rope:
        comparisons.append(("RoPE Theta", qwen_rope, llama_rope))
    
    # Print comparisons
    print(f"\n{'Metric':<30} {'Qwen':<20} {'Llama':<20} {'Difference':<20}")
    print(f"{'-'*90}")
    for metric, qwen_val, llama_val in comparisons:
        if isinstance(qwen_val, float) and qwen_val < 1:
            diff = abs(qwen_val - llama_val)
            diff_pct = (diff / llama_val * 100) if llama_val != 0 else 0
            print(f"{metric:<30} {qwen_val:<20.6f} {llama_val:<20.6f} {diff:.6f} ({diff_pct:.2f}%)")
        else:
            diff = abs(qwen_val - llama_val)
            print(f"{metric:<30} {qwen_val:<20.2f} {llama_val:<20.2f} {diff:.2f}")

def analyze_precision_requirements(config):
    """Analyze precision requirements based on model characteristics"""
    print(f"\n{'-'*80}")
    print("Precision Requirements Analysis")
    print(f"{'-'*80}")
    
    head_dim = config.hidden_size // config.num_attention_heads
    attention_scale = 1.0 / np.sqrt(head_dim)
    
    # Simulate attention score computation
    print("\nAttention Score Range Analysis:")
    print("  Assuming normalized inputs (mean=0, std=1)")
    
    # Typical range of dot products before scaling
    typical_dot_product_range = 3.0 * np.sqrt(head_dim)  # 3-sigma range
    print(f"  Typical dot product range:.................. ±{typical_dot_product_range:.2f}")
    
    # After scaling
    scaled_range = typical_dot_product_range * attention_scale
    print(f"  After attention scaling:.................... ±{scaled_range:.2f}")
    
    # After softmax, we need precision to distinguish between similar values
    # The minimum distinguishable difference in softmax outputs
    min_softmax_diff = np.exp(-scaled_range) / (1 + np.exp(-scaled_range))
    print(f"  Min distinguishable softmax difference:..... {min_softmax_diff:.2e}")
    
    # Required precision bits
    required_bits = -np.log2(min_softmax_diff)
    print(f"  Required precision bits:.................... {required_bits:.1f} bits")
    
    # Check if posit32 is sufficient
    posit32_effective_bits = 30  # Approximate effective precision
    if required_bits > posit32_effective_bits:
        print(f"  ⚠️  WARNING: Required precision ({required_bits:.1f} bits) may exceed")
        print(f"              Posit32 effective precision (~{posit32_effective_bits} bits)")
    else:
        print(f"  ✓ Posit32 effective precision (~{posit32_effective_bits} bits) is sufficient")
    
    # Normalization epsilon analysis
    if hasattr(config, 'rms_norm_eps'):
        eps = config.rms_norm_eps
        print(f"\nNormalization Epsilon Analysis:")
        print(f"  RMS Norm Epsilon:........................... {eps:.2e}")
        
        # Minimum representable value in different formats
        fp32_min = 1.175494e-38
        posit32_min = 5.9e-9  # Approximate for posit32es2
        
        if eps < posit32_min:
            print(f"  ⚠️  WARNING: Epsilon ({eps:.2e}) is smaller than")
            print(f"              Posit32 minimum normal ({posit32_min:.2e})")
            print(f"              This may cause numerical instability!")
        else:
            print(f"  ✓ Epsilon is within Posit32 representable range")

if __name__ == "__main__":
    # Analyze Qwen
    config_qwen, params_qwen = analyze_model_config(
        "Qwen2.5-0.5B",
        "onnx-community/Qwen2.5-0.5B"
    )
    
    # Analyze precision requirements for Qwen
    analyze_precision_requirements(config_qwen)
    
    print("\n" + "="*80)
    
    # Analyze Llama
    config_llama, params_llama = analyze_model_config(
        "Llama-3.2-1B",
        "onnx-community/Llama-3.2-1B"
    )
    
    # Analyze precision requirements for Llama
    analyze_precision_requirements(config_llama)
    
    # Compare the two models
    compare_numerical_characteristics(config_qwen, config_llama)
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print("Summary and Recommendations")
    print(f"{'='*80}")
    
    print("\nKey Findings:")
    print("  1. Check if normalization epsilon is compatible with Posit32 range")
    print("  2. Verify attention scaling factors don't cause underflow/overflow")
    print("  3. Examine if GQA ratio affects KV cache precision requirements")
    print("  4. Compare RoPE theta values for positional encoding stability")
    
    print("\nRecommended Next Steps:")
    print("  1. Run numerical range analysis during inference")
    print("  2. Monitor intermediate activation ranges")
    print("  3. Check for underflow/overflow in attention computations")
    print("  4. Verify normalization layer outputs")
    
    print("\nDone!")
