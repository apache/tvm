import argparse
import os
import numpy as np
import time
import tvm
from transformers import AutoTokenizer, AutoConfig
import sys
import ctypes
from tvm import ffi as _ffi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from BYODT.register import _posit_registered

_posit_registered()

project_root = os.path.abspath(sys.path[0])
os.chdir(project_root)
TAG = "tvm"

processor = AutoTokenizer.from_pretrained("onnx-community/gpt2-ONNX")
config = AutoConfig.from_pretrained("onnx-community/gpt2-ONNX")

# Global posit converters
posit16_converter_func = None
posit32_converter_func = None

def initialize_posit_converters():
    """Initialize posit16 and posit32 converters"""
    global posit16_converter_func, posit32_converter_func
    
    libtvm = ctypes.CDLL("/home/wang/venv/mlc-llm/tvm/build/libtvm.so")
    
    # Posit16es2 converter
    libtvm.Posit16es2ToFloat.restype = ctypes.c_float
    libtvm.Posit16es2ToFloat.argtypes = [ctypes.c_uint16]
    posit16_converter_func = np.vectorize(
        lambda x: libtvm.Posit16es2ToFloat(np.uint16(x)),
        otypes=[np.float32]
    )
    
    # Posit32es2 converter
    libtvm.Posit32es2ToFloat.restype = ctypes.c_float
    libtvm.Posit32es2ToFloat.argtypes = [ctypes.c_uint32]
    posit32_converter_func = np.vectorize(
        lambda x: libtvm.Posit32es2ToFloat(np.uint32(x)),
        otypes=[np.float32]
    )
    
    print("Posit converters initialized (16-bit and 32-bit)")

def convert_posit_to_float(posit_tensor, dtype_str, dev):
    """Convert posit tensor to float numpy array"""
    if "custom[posites2]16" in dtype_str:
        logits_shape = posit_tensor.shape
        logits_uint16 = tvm.runtime.empty(logits_shape, dtype="uint16", device=dev)
        posit_tensor.copyto(logits_uint16)
        logits_uint_np = logits_uint16.numpy()
        return posit16_converter_func(logits_uint_np)
    elif "custom[posites2]32" in dtype_str:
        logits_shape = posit_tensor.shape
        logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
        posit_tensor.copyto(logits_uint32)
        logits_uint_np = logits_uint32.numpy()
        return posit32_converter_func(logits_uint_np)
    else:
        return posit_tensor.numpy()

def benchmark_dual_models(model1_path, model2_path, dtype1, dtype2, input_data):
    """
    Benchmark two models with different data types
    dtype1, dtype2: 'float16', 'float32', 'custom[posites2]16', 'custom[posites2]32'
    """
    global posit16_converter_func, posit32_converter_func

    print("Running Dual Model Comparison Benchmark...")
    print(f"Model 1: {model1_path} (dtype: {dtype1})")
    print(f"Model 2: {model2_path} (dtype: {dtype2})")

    # Load both models
    lib_model1 = tvm.runtime.load_module(model1_path)
    decoder_model1 = tvm.relax.vm.VirtualMachine(lib_model1, tvm.device("cpu"))
    
    lib_model2 = tvm.runtime.load_module(model2_path)
    decoder_model2 = tvm.relax.vm.VirtualMachine(lib_model2, tvm.device("cpu"))
    
    # Extract parameters
    batch_size = 1
    num_layers = config.n_layer
    num_key_value_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    dev = tvm.device("cpu")
    
    # Initialize posit converters if needed
    if "posites2" in dtype1.lower() or "posites2" in dtype2.lower():
        if posit16_converter_func is None or posit32_converter_func is None:
            initialize_posit_converters()

    # Initialize KV caches for both models
    if dtype1.startswith("float"):
        kvcache_model1 = [
            tvm.runtime.tensor(np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32))
            for _ in range(num_layers * 2)
        ]
    else:  # custom posit type
        kvcache_model1 = [
            tvm.runtime.empty(
                (batch_size, num_key_value_heads, 0, head_dim),
                dtype=dtype1,
                device=dev,
            )
            for _ in range(num_layers * 2)
        ]
    
    if dtype2.startswith("float"):
        kvcache_model2 = [
            tvm.runtime.tensor(np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32))
            for _ in range(num_layers * 2)
        ]
    else:  # custom posit type
        kvcache_model2 = [
            tvm.runtime.empty(
                (batch_size, num_key_value_heads, 0, head_dim),
                dtype=dtype2,
                device=dev,
            )
            for _ in range(num_layers * 2)
        ]

    input_ids = processor(input_data).input_ids
    attention_mask = [1]
    position_ids = [0]

    # Performance tracking
    prefill_time_model1 = 0
    prefill_time_model2 = 0
    decode_times_model1 = []
    decode_times_model2 = []
    total_generation_time_model1 = 0
    total_generation_time_model2 = 0
    
    # Error tracking
    logit_errors = []  # MAE between logits
    token_matches = []  # Whether tokens match
    generation_tokens_model1 = []
    generation_tokens_model2 = []
    
    # Prefill phase for both models
    print("\n" + "="*80)
    print("PREFILL PHASE")
    print("="*80)
    
    # Model 1 prefill
    prefill_start_model1 = time.time()
    for input_id in input_ids:
        out_model1 = decoder_model1["main"](
            tvm.runtime.tensor([[input_id]]), 
            *kvcache_model1, 
            tvm.runtime.tensor([attention_mask]), 
            tvm.runtime.tensor([position_ids])
        )
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache_model1 = out_model1[1:]
    prefill_time_model1 = time.time() - prefill_start_model1
    
    # Reset for model2 prefill
    position_ids = [0]
    attention_mask = [1]
    
    # Model 2 prefill
    prefill_start_model2 = time.time()
    for input_id in input_ids:
        out_model2 = decoder_model2["main"](
            tvm.runtime.tensor([[input_id]]), 
            *kvcache_model2, 
            tvm.runtime.tensor([attention_mask]), 
            tvm.runtime.tensor([position_ids])
        )
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache_model2 = out_model2[1:]
    prefill_time_model2 = time.time() - prefill_start_model2
    
    print(f"Model 1 Prefill Time: {prefill_time_model1:.6f} seconds")
    print(f"Model 2 Prefill Time: {prefill_time_model2:.6f} seconds")

    # Get first token from both models
    logits_model1_raw = out_model1[0]
    logits_model1 = convert_posit_to_float(logits_model1_raw, dtype1, dev)[0][-1]
    
    logits_model2_raw = out_model2[0]
    logits_model2 = convert_posit_to_float(logits_model2_raw, dtype2, dev)[0][-1]
    
    # Calculate first logit error
    mae = np.mean(np.abs(logits_model1 - logits_model2))
    logit_errors.append(mae)
    
    print(f"Prefill Logits MAE: {mae:.6f}")
    
    next_token_model1 = np.argmax(logits_model1, axis=-1)
    next_token_model2 = np.argmax(logits_model2, axis=-1)
    
    token_match = (next_token_model1 == next_token_model2)
    token_matches.append(token_match)
    
    generation_tokens_model1.append(next_token_model1)
    generation_tokens_model2.append(next_token_model2)

    # Generation loop
    print("\n" + "="*80)
    print("DECODE PHASE")
    print("="*80)
    
    step = 1
    while len(generation_tokens_model1) < ITER and next_token_model1 != 151643:
        # Model 1
        input_ids_model1 = next_token_model1.reshape(1, 1)
        tvm_inputs0_model1 = [tvm.runtime.tensor(input_ids_model1)]
        tvm_inputs1 = [tvm.runtime.tensor([attention_mask]), tvm.runtime.tensor([position_ids])]
        
        start = time.time()
        out_model1 = decoder_model1["main"](*tvm_inputs0_model1, *kvcache_model1, *tvm_inputs1)
        decode_time_model1 = time.time() - start
        decode_times_model1.append(decode_time_model1)
        
        kvcache_model1 = out_model1[1:]
        logits_model1_raw = out_model1[0]
        logits_model1 = convert_posit_to_float(logits_model1_raw, dtype1, dev)[0][-1]
        
        next_token_model1 = np.argmax(logits_model1, axis=-1)
        generation_tokens_model1.append(next_token_model1)
        
        # Model 2
        input_ids_model2 = next_token_model2.reshape(1, 1)
        tvm_inputs0_model2 = [tvm.runtime.tensor(input_ids_model2)]
        
        start = time.time()
        out_model2 = decoder_model2["main"](*tvm_inputs0_model2, *kvcache_model2, *tvm_inputs1)
        decode_time_model2 = time.time() - start
        decode_times_model2.append(decode_time_model2)
        
        kvcache_model2 = out_model2[1:]
        logits_model2_raw = out_model2[0]
        logits_model2 = convert_posit_to_float(logits_model2_raw, dtype2, dev)[0][-1]
        
        next_token_model2 = np.argmax(logits_model2, axis=-1)
        generation_tokens_model2.append(next_token_model2)
        
        # Calculate error
        mae = np.mean(np.abs(logits_model1 - logits_model2))
        logit_errors.append(mae)
        
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        
        token_match = (next_token_model1 == next_token_model2)
        token_matches.append(token_match)
        
        token_str_model1 = processor.decode([next_token_model1])
        token_str_model2 = processor.decode([next_token_model2])
        
        print(f"\nStep {step}:")
        print(f"  Model1: Token={next_token_model1}, Text='{token_str_model1}', Logit={logits_model1[next_token_model1]:.4f}, Time={decode_time_model1:.6f}s")
        print(f"  Model2: Token={next_token_model2}, Text='{token_str_model2}', Logit={logits_model2[next_token_model2]:.4f}, Time={decode_time_model2:.6f}s")
        print(f"  Logits MAE: {mae:.6f}, Token Match: {token_match}")
        
        step += 1

    # Calculate total generation times (prefill + decode)
    total_generation_time_model1 = prefill_time_model1 + np.sum(decode_times_model1)
    total_generation_time_model2 = prefill_time_model2 + np.sum(decode_times_model2)

    # Print profiling results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nModel 1 ({dtype1}) Performance:")
    print(f"  Prefill Time: {prefill_time_model1:.6f} seconds")
    print(f"  Avg Decode Time: {np.mean(decode_times_model1):.6f} ± {np.std(decode_times_model1):.6f} seconds")
    print(f"  Total Decode Time: {np.sum(decode_times_model1):.6f} seconds")
    print(f"  Total Generation Time: {total_generation_time_model1:.6f} seconds")
    print(f"  Avg Time per Token: {total_generation_time_model1 / len(generation_tokens_model1):.6f} seconds")
    
    print(f"\nModel 2 ({dtype2}) Performance:")
    print(f"  Prefill Time: {prefill_time_model2:.6f} seconds")
    print(f"  Avg Decode Time: {np.mean(decode_times_model2):.6f} ± {np.std(decode_times_model2):.6f} seconds")
    print(f"  Total Decode Time: {np.sum(decode_times_model2):.6f} seconds")
    print(f"  Total Generation Time: {total_generation_time_model2:.6f} seconds")
    print(f"  Avg Time per Token: {total_generation_time_model2 / len(generation_tokens_model2):.6f} seconds")
    
    print(f"\nGeneration Summary:")
    print(f"  Total Tokens Generated: {len(generation_tokens_model1)}")
    
    print(f"\nAccuracy Analysis:")
    print(f"  Avg Logits MAE: {np.mean(logit_errors):.6f} ± {np.std(logit_errors):.6f}")
    print(f"  Max Logits MAE: {np.max(logit_errors):.6f}")
    print(f"  Min Logits MAE: {np.min(logit_errors):.6f}")
    print(f"  Token Match Rate: {np.mean(token_matches)*100:.2f}% ({np.sum(token_matches)}/{len(token_matches)})")

    # Decode final sentences
    output_model1 = processor.decode(generation_tokens_model1, skip_special_tokens=True)
    output_model2 = processor.decode(generation_tokens_model2, skip_special_tokens=True)
    
    print(f"\nModel 1 Output:\n{output_model1}")
    print(f"\nModel 2 Output:\n{output_model2}")
    print(f"\nOutputs Match: {output_model1 == output_model2}")

    # Save to CSV
    import csv
    with open("benchmark_result.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Tag",
                "Model1_Type",
                "Model2_Type",
                "Model1_Prefill_Time",
                "Model2_Prefill_Time",
                "Model1_Avg_Decode_Time",
                "Model2_Avg_Decode_Time",
                "Model1_Total_Decode_Time",
                "Model2_Total_Decode_Time",
                "Model1_Total_Generation_Time",
                "Model2_Total_Generation_Time",
                "Model1_Avg_Time_Per_Token",
                "Model2_Avg_Time_Per_Token",
                "Avg_Logits_MAE",
                "Max_Logits_MAE",
                "Token_Match_Rate",
                "Tokens_Generated",
            ])
        writer.writerow([
            TAG,
            dtype1,
            dtype2,
            round(prefill_time_model1, 6),
            round(prefill_time_model2, 6),
            round(np.mean(decode_times_model1), 6),
            round(np.mean(decode_times_model2), 6),
            round(np.sum(decode_times_model1), 6),
            round(np.sum(decode_times_model2), 6),
            round(total_generation_time_model1, 6),
            round(total_generation_time_model2, 6),
            round(total_generation_time_model1 / len(generation_tokens_model1), 6),
            round(total_generation_time_model2 / len(generation_tokens_model2), 6),
            round(np.mean(logit_errors), 6),
            round(np.max(logit_errors), 6),
            round(np.mean(token_matches)*100, 2),
            len(generation_tokens_model1),
        ])
    
    print(f"\nResults saved to benchmark_result.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Benchmark TVM runtime dual model comparison with flexible dtype support"
    )
    p.add_argument("--tag", required=True, help="tag for the benchmark")
    p.add_argument("--model1", required=True, help="First model .so file path")
    p.add_argument("--model2", required=True, help="Second model .so file path")
    p.add_argument("--dtype1", required=True, 
                   choices=["float16", "float32", "custom[posites2]16", "custom[posites2]32"],
                   help="Data type for first model")
    p.add_argument("--dtype2", required=True,
                   choices=["float16", "float32", "custom[posites2]16", "custom[posites2]32"],
                   help="Data type for second model")
    p.add_argument("--num-threads", type=int, required=True, help="number of threads")
    p.add_argument("--is-config-threadpool", type=bool, default=False, help="config threadpool")
    p.add_argument("--iter", "-n", type=int, default=50, help="number of timed iterations (default: 50)")
    p.add_argument("--prompt", default="Tell me about AI", help="input prompt for generation")

    args = p.parse_args()
    ITER = args.iter
    TAG = args.tag
    IS_CONFIG_THREADPOOL = args.is_config_threadpool
    NUM_THREADS = args.num_threads

    if IS_CONFIG_THREADPOOL:
        config_func = _ffi.get_global_func("runtime.config_threadpool")
        cpus = [str(i) for i in range(NUM_THREADS)]
        config_func(-2, NUM_THREADS, cpus)
    else:
        os.environ["TVM_NUM_THREADS"] = str(NUM_THREADS)
    
    print(f"\nConfiguration:")
    print(f"  Tag: {TAG}")
    print(f"  Model 1: {args.model1} ({args.dtype1})")
    print(f"  Model 2: {args.model2} ({args.dtype2})")
    print(f"  Threads: {NUM_THREADS}")
    print(f"  Iterations: {ITER}")
    print(f"  Prompt: {args.prompt}")
    
    benchmark_dual_models(args.model1, args.model2, args.dtype1, args.dtype2, args.prompt)
    print("\nDone!")