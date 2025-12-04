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

processor = AutoTokenizer.from_pretrained("onnx-community/Qwen2.5-0.5B")
config = AutoConfig.from_pretrained("onnx-community/Qwen2.5-0.5B")

# Global posit converter
posit_converter_func = None


def benchmark_tvmruntime(decoder_path, input_data):
    global posit_converter_func
    
    print("Running TVM Runtime Profiled Benchmark...")

    lib = tvm.runtime.load_module(decoder_path)
    decoder = tvm.relax.vm.VirtualMachine(lib, tvm.device("cpu"))
    
    # Extract parameters
    batch_size = 1
    num_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dev = tvm.device("cpu")
    
    # Auto-detect dtype from model
    use_posit = "posit" in decoder_path.lower()
    kv_dtype = "custom[posites2]32" if use_posit else "float32"
    
    print(f"Using KV cache dtype: {kv_dtype}")
    
    # Initialize posit converter if needed
    if use_posit and posit_converter_func is None:
        libtvm = ctypes.CDLL("/home/wang/venv/mlc-llm/tvm/build/libtvm.so")
        libtvm.Posit32es2ToFloat.restype = ctypes.c_float
        libtvm.Posit32es2ToFloat.argtypes = [ctypes.c_uint32]
        posit_converter_func = np.vectorize(
            lambda x: libtvm.Posit32es2ToFloat(int(x)),
            otypes=[np.float32]
        )
        print("Posit converter initialized")

    # Initialize empty KV cache
    if use_posit:
        kvcache = [
            tvm.runtime.empty(
                (batch_size, num_key_value_heads, 0, head_dim),
                dtype=kv_dtype,
                device=dev,
            )
            for _ in range(num_layers * 2)
        ]
    else:
        kvcache = [
            tvm.runtime.tensor(np.zeros((batch_size, num_key_value_heads, 0, head_dim), dtype=np.float32))
            for _ in range(num_layers * 2)
        ]

    input_ids = processor(input_data).input_ids
    attention_mask = [1]
    position_ids = [0]

    decode_times = []
    total_generation_times = []
    token_generation_times = []

    out1 = None
    generation_tokens = []
    sentence = [[]]

    total_start = time.time()
    
    # Prefill: feed tokens one by one
    for input_id in input_ids:
        out1 = decoder["main"](tvm.runtime.tensor([[input_id]]), tvm.runtime.tensor([attention_mask]), tvm.runtime.tensor([position_ids]), *kvcache)
        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache = out1[1:]
    
    # Convert output to float32 if needed - FIXED VERSION
    if use_posit:
        # Posit tensor can't convert to numpy directly, copy to uint32 first
        logits_shape = out1[0].shape
        logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
        out1[0].copyto(logits_uint32)  # Copy posit bits as uint32
        logits_uint32_np = logits_uint32.numpy()[0][-1]
        out1 = posit_converter_func(logits_uint32_np)  # Convert uint32 -> float32
    else:
        out1 = out1[0].numpy()[0][-1]
    
    print(out1)
    next_token = np.argmax(out1, axis=-1)
    input_ids = next_token.reshape(1, 1)
    sentence = np.concatenate(
        [sentence, next_token.reshape(1, 1)], 
        axis=1,
    )
    generation_tokens.append(next_token)


    while next_token != 151643 and len(generation_tokens) < ITER:
        # Decode
        tvm_inputs = [tvm.runtime.tensor(input_ids), tvm.runtime.tensor([attention_mask]), tvm.runtime.tensor([position_ids])]
        start = time.time()
        out1 = decoder["main"](*tvm_inputs, *kvcache)
        decode_time = time.time() - start
        decode_times.append(decode_time)

        position_ids[0] = position_ids[0] + 1
        attention_mask.append(1)
        kvcache = out1[1:]
        
        # Convert output to float32 if needed - FIXED VERSION
        if use_posit:
            # Posit tensor can't convert to numpy directly, copy to uint32 first
            logits_shape = out1[0].shape
            logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
            out1[0].copyto(logits_uint32)  # Copy posit bits as uint32 (same bits, safe)
            logits_uint32_np = logits_uint32.numpy()[0][-1]
            out1 = posit_converter_func(logits_uint32_np)  # Convert uint32 -> float32
        else:
            out1 = out1[0].numpy()[0][-1]
        print(out1)
        next_token = np.argmax(out1, axis=-1)
        token_str = processor.decode([next_token])
        print(f"Token {len(generation_tokens)+1}: ID={next_token}, Text='{token_str}', Logit={out1[next_token]:.4f}, IsEOS={next_token==151643}")
        input_ids = next_token.reshape(1, 1)
        sentence = np.concatenate(
            [sentence, next_token.reshape(1, 1)], 
            axis=1,
        )
        generation_tokens.append(next_token)

    # Total generation time
    total_generation_time = time.time() - total_start
    total_generation_times.append(total_generation_time)
    token_generation_times.append(total_generation_time / len(generation_tokens))

    # Print profiling results
    print("\nProfiling Results (Average over {} iterations):".format(ITER))
    print("Decode Time:      {:.4f} ± {:.4f} seconds".format(
        np.mean(decode_times), np.std(decode_times)))
    print("Total Generation: {:.4f} ± {:.4f} seconds".format(
        np.mean(total_generation_times), np.std(total_generation_times)))
    print("Avg Token Time:   {:.4f} ± {:.4f} seconds".format(
        np.mean(token_generation_times), np.std(token_generation_times)))

    # Decode final sentence
    output = processor.decode(generation_tokens, skip_special_tokens=True)
    print("\nTranscription:\n", output)

    import csv
    with open("benchmark_result.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Tag",
                "Decode",
                "Total_generation",
                "Token_generation",
            ])
        writer.writerow([
            TAG,
            round(np.mean(decode_times), 6),
            round(np.mean(total_generation_times), 6),
            round(np.mean(token_generation_times), 6),
        ])

    print("Done!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Benchmark TVM runtime shared-object pipelines"
    )
    p.add_argument("--tag",                required=True, help="tag for the benchmark")
    p.add_argument("--decoder",            required=True, help="decoder .so file")
    p.add_argument("--num-threads",        type=int, required=True, help="number of threads")
    p.add_argument(
        "--is-config-threadpool",  
        type=bool,
        default=False,
        help="config threadpool")
    p.add_argument(
        "--iter", "-n",
        type=int,
        default=50,
        help="number of timed iterations (default: 50)"
    )

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
    prompt = "Tell me about AI"
    benchmark_tvmruntime(args.decoder, prompt)
