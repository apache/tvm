import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import GeneralReduction, Matmul
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel, apply_and_build
from tvm import te, tir


def gemv(M, N, K, dtype="float16"):
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
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    return GEMV


def matmul_nt(M, N, K, in_dtype="float16", out_dtype="float16"):
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
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[vj, vk].astype(
                        out_dtype
                    )

    return MatmulNT


def conv2d_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p, in_dtype="float16", out_dtype="float16"):
    A = te.placeholder((n, h, w, c), name="input", dtype=in_dtype)
    B = te.placeholder((f, kh, kw, c), name="weight", dtype=in_dtype)

    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
        pad_shape,
        lambda n, h, w, c: te.if_then_else(
            tir.all(
                h >= p,
                w >= p,
                h < pad_shape[1] - p,
                w < pad_shape[2] - p,
            ),
            A[n, h - p, w - p, c],
            pad_value,
        ),
        name="pad",
    )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    out_h = (h + 2 * p - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (w + 2 * p - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    out_shape = (n, out_h, out_w, f)
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    c = te.reduce_axis((0, c), name="c")
    C = te.compute(
        out_shape,
        lambda n, h, w, f: te.sum(
            pad[
                n,
                h * stride_h + tir.any(dilation_h),
                w * stride_w + tir.any(dilation_w),
                c,
            ]
            * B[f, kh - 1 - tir.any(dilation_h), kw - 1 - tir.any(dilation_w), c],
            axis=[kh, kw, c],
        ),
        name="C",
    )
    return tvm.ir.IRModule({"main": te.create_prim_func([A, B, C])})


benchmark_sets = [
    # (gemv, (1, 16384, 16384, "float16"), GeneralReduction, Fallback),
    # (matmul_nt, (1024, 1024, 1024, "float16", "float16"), GeneralReduction, Fallback),
    (matmul_nt, (128, 128, 150528, "float32", "float32"), GeneralReduction, Matmul),
    # (conv2d_nhwc_hwnc, (128, 512, 7, 7, 2048, 1, 1, 1, 1, 0, "float16", "float16"), GeneralReduction, Fallback),
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
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    fast_tune_time = time.time() - tune_start
    print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
    print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))
    print(best.code)
    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
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
