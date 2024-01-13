import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import GeneralReduction, Matmul
from tvm.dlight.base.utils import apply_and_build

def softmax(M, N, dtype="float16"):
    @tvm.script.ir_module
    class Softmax:
        @T.prim_func
        def main(A: T.Buffer((M, N), dtype), T_softmax_norm: T.Buffer((M, N), dtype)):
            T_softmax_maxelem = T.alloc_buffer((M,))
            T_softmax_exp = T.alloc_buffer((M, N))
            T_softmax_expsum = T.alloc_buffer((M,))
            for i0, k in T.grid(M, N):
                with T.block("T_softmax_maxelem"):
                    v_i0, v_k = T.axis.remap("SR", [i0, k])
                    with T.init():
                        T_softmax_maxelem[v_i0] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[v_i0] = T.max(T_softmax_maxelem[v_i0], A[v_i0, v_k])
            for i0, i1 in T.grid(M, N):
                with T.block("T_softmax_exp"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T_softmax_exp[v_i0, v_i1] = T.exp(A[v_i0, v_i1] - T_softmax_maxelem[v_i0])
            for i0, k in T.grid(M, N):
                with T.block("T_softmax_expsum"):
                    v_i0, v_k = T.axis.remap("SR", [i0, k])
                    with T.init():
                        T_softmax_expsum[v_i0] = T.float32(0)
                    T_softmax_expsum[v_i0] = T_softmax_expsum[v_i0] + T_softmax_exp[v_i0, v_k]
            for i0, i1 in T.grid(M, N):
                with T.block("T_softmax_norm"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T_softmax_norm[v_i0, v_i1] = T_softmax_exp[v_i0, v_i1] / T_softmax_expsum[v_i0]
    return Softmax

benchmark_sets = [
    (softmax, (128, 10, 'float16'), GeneralReduction),
]

benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    configs = policy.emit_config(20)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    fast_tune_time = time.time() - tune_start
    print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
    print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))
    print(best.code)

    # evaluate the performance of the default schedule
    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    print(sch_default.mod)
    mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    profile_tensors = []
    for arg in args:
        profile_tensors.append(
            tvm.nd.array(
                np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )

    timer_cuda_mod = mod_default.time_evaluator(mod_default.entry_name, arch.device, number=5)
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_dlight_top20_tune_time": fast_tune_time,
            "fast_dlight_top1_latency": cpresults[0].latency * 1e3,
            "fast_dlight_top20_latency": best.latency * 1e3,
            "default_dlight_tune_time": default_tune_time,
            "default_dlight_latency": t * 1e3,
        }
    }
    benchmark_results.update(profile_config)

headers = [
    "PrimFunc",
    "Input Arguments",
    "FastDLight Top20 Tune Time",
    "FastDLight Top1 Latency",
    "FastDLight Top20 Latency",
    "DefaultDLight Tune Time",
    "DefaultDLight Latency",
]

col_width = (
    max(len(word) for row in [headers] + list(profile_config.values()) for word in row) + 2
)  # padding

print("".join(word.ljust(col_width) for word in headers))

print("-" * col_width * len(headers))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f"{values['fast_dlight_top20_tune_time']:.3f} s",
        f"{values['fast_dlight_top1_latency']:.3f} ms",
        f"{values['fast_dlight_top20_latency']:.3f} ms",
        f"{values['default_dlight_tune_time']:.3f} s",
        f"{values['default_dlight_latency']:.3f} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))
