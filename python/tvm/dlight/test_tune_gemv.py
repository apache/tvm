import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise, GeneralReduction, GEMV
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel

def gemv(
    M, N, K, dtype="float16"
):
    @tvm.script.ir_module
    class GEMV:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [N, K], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk] * B[vj, vk]
    return GEMV

benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    (gemv, (1, 1024, 1024, "float16"), GEMV, GEMV),
    (gemv, (1, 8192, 8192, "float16"), GEMV, GEMV),
    (gemv, (1, 16384, 16384, "float16"), GEMV, GEMV),
]
benchmark_results = {}
for get_prim_func, input_args, f_schedule, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    configs = policy.emit_config(20)
    rule = f_schedule()
    
    tune_start = time.time()
    cpresults, best = apply_and_build_parallel(func, rule, configs, arch)
    fast_tune_time = time.time() - tune_start
    print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
    print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))

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
            np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype), device=arch.device)
        )
                    
    timer_cuda_mod = mod_default.time_evaluator(
        mod_default.entry_name, arch.device, number=5)
    t = timer_cuda_mod(*profile_tensors).mean


    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))
    
    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            'fast_dlight_top20_tune_time': fast_tune_time,
            'fast_dlight_top1_latency': cpresults[0].latency * 1e3,
            'fast_dlight_top20_latency': best.latency * 1e3,
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
