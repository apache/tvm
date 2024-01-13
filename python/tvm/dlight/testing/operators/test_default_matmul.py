import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.gpu import Matmul
import time

def matmul_nt(
    M, N, K, in_dtype="float16", out_dtype="float16"
):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk].astype(out_dtype) * B[vj, vk].astype(out_dtype)
    return MatmulNT

benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    (matmul_nt, (1024, 1024, 1024, "float16", "float16"), Matmul, Matmul),
    (matmul_nt, (8192, 8192, 8192, "float16", "float16"), Matmul, Matmul),
    (matmul_nt, (16384, 16384, 16384, "float16", "float16"), Matmul, Matmul),
]
benchmark_results = {}
for get_prim_func, input_args, f_schedule, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")

    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()
            
    profile_tensors = []
    for arg in args:
        profile_tensors.append(tvm.nd.array(
            np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype), device=tvm.cuda())
        )
                    
    timer_cuda_mod = mod_default.time_evaluator(
        mod_default.entry_name, tvm.cuda(), number=5)
    t = timer_cuda_mod(*profile_tensors).mean


    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))
    
    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            'default_dlight_tune_time': default_tune_time,
            'default_dlight_latency': t* 1e3,
            
        }
    }
    benchmark_results.update(profile_config)

headers = ["PrimFunc", "Input Arguments", "FastDLight Top20 Tune Time", "FastDLight Top1 Latency",
           "FastDLight Top20 Latency", "DefaultDLight Tune Time", "DefaultDLight Latency"]

col_width = max(len(word) for row in [headers] + list(profile_config.values()) for word in row) + 2  # padding

print("".join(word.ljust(col_width) for word in headers))

print("-" * col_width * len(headers))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f" {str(values['fast_dlight_top20_tune_time'])} s",
        f"{values['fast_dlight_top1_latency']:.3f} ms",
        f"{values['fast_dlight_top20_latency']:.3f} ms",
        str(values['default_dlight_tune_time']),
        f"{values['default_dlight_latency']:.3f} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))            
