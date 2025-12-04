import os
import numpy as np
import time
import tvm
from transformers import WhisperProcessor
import sys
import ctypes
from tvm import ffi as _ffi
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from BYODT.register import _posit_registered

_posit_registered()
project_root = os.path.abspath(sys.path[0])
os.chdir(project_root)
WARMUP_ITER = 1
ITER = 32
TAG = "tvm"
IS_CONFIG_THREADPOOL = False
NUM_THREADS = 16


processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
config = {"decoder_start_token_id": 50258,"eos_token_id": 50257}

posit_converter_func = None

def benchmark_tvmruntime(encoder_path, prefill_path, decoder_path, input_data):
    global posit_converter_func

    print("Running TVM Runtime Profiled Benchmark...")

    # Load modules
    lib = tvm.runtime.load_module(encoder_path)
    encoder = tvm.relax.vm.VirtualMachine(lib, tvm.device("cpu"))

    lib = tvm.runtime.load_module(prefill_path)
    decoder = tvm.relax.vm.VirtualMachine(lib, tvm.device("cpu"))

    lib = tvm.runtime.load_module(decoder_path)
    decoder_with_past = tvm.relax.vm.VirtualMachine(lib, tvm.device("cpu"))
    
    # Auto-detect dtype from model
    use_posit = "posit" in decoder_path.lower()
    
    print(f"Using posit mode: {use_posit}")
    
    # Initialize posit converter if needed
    if use_posit and posit_converter_func is None:
        libtvm = ctypes.CDLL("/home/wang/venv/mlc-llm/tvm/build/libtvm.so")
        libtvm.Posit32es2ToFloat.restype = ctypes.c_float
        libtvm.Posit32es2ToFloat.argtypes = [ctypes.c_uint32]
        libtvm.FloatToPosit32es2.restype = ctypes.c_uint32
        libtvm.FloatToPosit32es2.argtypes = [ctypes.c_float]
        posit_converter_func = np.vectorize(
            lambda x: libtvm.Posit32es2ToFloat(int(x)),
            otypes=[np.float32]
        )
        print("Posit converter initialized")
    
    dev = tvm.device("cpu")
    
    # Convert input data to posit if needed
    if use_posit:
        libtvm = ctypes.CDLL("/home/wang/venv/mlc-llm/tvm/build/libtvm.so")
        libtvm.FloatToPosit32es2.restype = ctypes.c_uint32
        libtvm.FloatToPosit32es2.argtypes = [ctypes.c_float]
        float_to_posit = np.vectorize(
            lambda x: libtvm.FloatToPosit32es2(float(x)),
            otypes=[np.uint32]
        )
        # Convert input_data from float32 to posit
        input_data_np = input_data.numpy() if hasattr(input_data, 'numpy') else np.array(input_data)
        input_data_uint32 = float_to_posit(input_data_np)
        input_data_posit = tvm.runtime.empty(input_data_uint32.shape, dtype="custom[posites2]32", device=dev)
        temp_uint32 = tvm.runtime.tensor(input_data_uint32)
        temp_uint32.copyto(input_data_posit)
        input_data = input_data_posit
        print("Input data converted to posit format")
    print("Input data shape:", input_data.shape)
    # Warm-up phase
    print(f"Performing {WARMUP_ITER} warm-up iterations...")
    for _ in range(WARMUP_ITER):
        # Simulate a full inference pass during warm-up
        input_ids = np.array([[config["decoder_start_token_id"]]], dtype=np.int64)
        next_token = config["decoder_start_token_id"]
        
        out = encoder["main"](tvm.runtime.tensor(input_data))
        out1 = decoder["main"](tvm.runtime.tensor(input_ids), out)
        
        kvcache = out1[1:]
        
        # Convert output to float32 if needed
        if use_posit:
            logits_shape = out1[0].shape
            logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
            out1[0].copyto(logits_uint32)
            logits_uint32_np = logits_uint32.numpy()[0][-1]
            out1_converted = posit_converter_func(logits_uint32_np)
            next_token = np.argmax(out1_converted, axis=-1)
        else:
            out1 = out1[0].numpy()[0][-1]
            next_token = np.argmax(out1, axis=-1)
        
        while next_token != config["eos_token_id"]:
            tvm_input_ids = tvm.runtime.tensor(next_token.reshape(1, 1))
            out1 = decoder_with_past["main"](tvm_input_ids, *kvcache)
            
            # Correctly update kvcache
            for i in range(4):
                kvcache[i * 4] = out1[i * 2 + 1]
                kvcache[i * 4 + 1] = out1[i * 2 + 2]
            
            # Convert output to float32 if needed
            if use_posit:
                logits_shape = out1[0].shape
                logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
                out1[0].copyto(logits_uint32)
                logits_uint32_np = logits_uint32.numpy()[0][-1]
                out1_converted = posit_converter_func(logits_uint32_np)
                next_token = np.argmax(out1_converted, axis=-1)
            else:
                out1 = out1[0].numpy()[0][-1]
                next_token = np.argmax(out1, axis=-1)
            
            if next_token == config["eos_token_id"]:
                break

    # Profiling variables
    encode_times = []
    prefill_times = []
    decode_times = []
    total_generation_times = []
    token_generation_times = []

    for _ in range(ITER):
        # Reset for each iteration
        input_ids = np.array(
            [[config["decoder_start_token_id"]]],
            dtype=np.int64,
        )
        next_token = config["decoder_start_token_id"]

        # Encode
        start = time.time()
        out = encoder["main"](tvm.runtime.tensor(input_data))
        encode_time = time.time() - start
        encode_times.append(encode_time)

        # Prefill
        start = time.time()
        tvm_input_ids = tvm.runtime.tensor(input_ids)
        out1 = decoder["main"](tvm_input_ids, out)
        prefill_time = time.time() - start
        prefill_times.append(prefill_time)

        # Prepare initial state
        kvcache = out1[1:]
        
        # Convert output to float32 if needed
        if use_posit:
            logits_shape = out1[0].shape
            logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
            out1[0].copyto(logits_uint32)
            logits_uint32_np = logits_uint32.numpy()[0][-1]
            out1_converted = posit_converter_func(logits_uint32_np)
            next_token = np.argmax(out1_converted, axis=-1)
        else:
            out1 = out1[0].numpy()[0][-1]
            next_token = np.argmax(out1, axis=-1)
        sentence = np.concatenate(
            [input_ids, next_token.reshape(1, 1)], 
            axis=1,
        )
        input_ids = next_token.reshape(1, 1)

        # Start total generation time
        total_start = time.time()
        generation_tokens = []

        while next_token != config["eos_token_id"]:
            # Decode
            start = time.time()
            tvm_input_ids = tvm.runtime.tensor(input_ids)
            out1 = decoder_with_past["main"](tvm_input_ids, *kvcache)
            decode_time = time.time() - start
            decode_times.append(decode_time)

            # Update cache
            for i in range(4):
                kvcache[i * 4] = out1[i * 2 + 1]
                kvcache[i * 4 + 1] = out1[i * 2 + 2]
            
            # Convert output to float32 if needed
            if use_posit:
                logits_shape = out1[0].shape
                logits_uint32 = tvm.runtime.empty(logits_shape, dtype="uint32", device=dev)
                out1[0].copyto(logits_uint32)
                logits_uint32_np = logits_uint32.numpy()[0][-1]
                out1_converted = posit_converter_func(logits_uint32_np)
                next_token = np.argmax(out1_converted, axis=-1)
            else:
                out1 = out1[0].numpy()[0][-1]
                next_token = np.argmax(out1, axis=-1)
            
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
    print("Encode Time:      {:.4f} ± {:.4f} seconds".format(
        np.mean(encode_times), np.std(encode_times)))
    print("Prefill Time:     {:.4f} ± {:.4f} seconds".format(
        np.mean(prefill_times), np.std(prefill_times)))
    print("Decode Time:      {:.4f} ± {:.4f} seconds".format(
        np.mean(decode_times), np.std(decode_times)))
    print("Total Generation: {:.4f} ± {:.4f} seconds".format(
        np.mean(total_generation_times), np.std(total_generation_times)))
    print("Avg Token Time:   {:.4f} ± {:.4f} seconds".format(
        np.mean(token_generation_times), np.std(token_generation_times)))

    # Decode final sentence
    output = processor.decode(sentence[0], skip_special_tokens=True)
    print("\nTranscription:\n", output)
    import csv
    with open("benchmark_result.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Tag",
                "Encode",
                "Prefill",
                "Decode",
                "Total_generation",
                "Token_generation",
            ])
        writer.writerow([
            TAG,
            round(np.mean(encode_times), 6),
            round(np.mean(prefill_times), 6),
            round(np.mean(decode_times), 6),
            round(np.mean(total_generation_times), 6),
            round(np.mean(token_generation_times), 6),
        ])

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Benchmark TVM runtime shared-object pipelines"
    )
    p.add_argument("--tag",                required=True, help="tag for the benchmark")
    p.add_argument("--encoder",            required=True, help="encoder .so file")
    p.add_argument("--decoder",            required=True, help="decoder .so file")
    p.add_argument("--decoder-with-past",  required=True, help="decoder_with_past .so file")
    p.add_argument("--data-path",         required=True, help="path to data")
    p.add_argument("--num-threads",        type=int, required=True, help="number of threads")
    p.add_argument(
        "--is-config-threadpool",  
        type=bool,
        default=False,
        help="config threadpool")
    # add optional arguments with defaults
    p.add_argument(
        "--warmup-iter", "-w",
        type=int,
        default=1,
        help="number of warm-up iterations (default: 1)"
    )
    p.add_argument(
        "--iter", "-n",
        type=int,
        default=50,
        help="number of timed iterations (default: 50)"
    )


    args = p.parse_args()
    WARMUP_ITER = args.warmup_iter
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
    
    data = np.load(args.data_path)
    benchmark_tvmruntime(args.encoder,
                        args.decoder,
                        args.decoder_with_past,
                        tvm.runtime.tensor(data)) 
    print("Done!")
