from typing import Dict, List
import numpy as np

import tvm
from tvm import ir, te, tir, dlight
from tvm.contrib import nvcc
from tvm.script import tir as T, ir as I

TARGET = tvm.target.Target("nvidia/geforce-rtx-3080")
DEVICE = tvm.cuda(0)


@T.prim_func
def fused_softmax1_cast1(p_lv1613: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv1613 = T.match_buffer(p_lv1613, (T.int64(1), T.int64(32), T.int64(1), n))
    var_compute_intermediate = T.match_buffer(
        p_output0, (T.int64(1), T.int64(32), T.int64(1), n), "float16"
    )
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)))
    var_T_softmax_norm_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1613[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(
                T_softmax_maxelem[v_i0, v_i1, v_i2], lv1613[v_i0, v_i1, v_i2, v_k]
            )
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv1613[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(
                lv1613[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = (
                T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
            )
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3] = (
                T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]
            )
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast(
                "float16", var_T_softmax_norm_intermediate[v_i0, v_i1, v_i2, v_i3]
            )


@T.prim_func
def softmax(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
    T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1)))
    for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_i1, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e38)
            T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
            T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(
                A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1]
            )
    for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1] = (
                T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
            T.block_attr({"axis": 2})
            T_softmax_norm[v_i0, v_i1, v_i2] = (
                T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]
            )


def prepare_args(func: tir.PrimFunc, var_dict: Dict[str, int]):
    args: List[np.ndarray] = []
    analyzer = tvm.arith.Analyzer()
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tir.Var):
                assert dim.name in var_dict
                value = var_dict[dim.name]
                shape.append(value)
                analyzer.bind(dim, value)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        np_array = np.random.uniform(size=shape).astype(buffer.dtype)
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)

    return args


def evaluate(func: tir.PrimFunc, args, run_only: bool = False):
    rt_mod = tvm.build(func, target=TARGET)
    rt_mod(*args)
    ret = args[-1]
    if not run_only:
        DEVICE.sync()
        time_eval = rt_mod.time_evaluator(rt_mod.entry_name, DEVICE, number=1)
        DEVICE.sync()
        time = time_eval(*args).mean * 1e3

        print(f"Time (ms): {time:.3f}", sep="\t")
    return ret


def main():
    for func in [fused_softmax1_cast1, softmax]:
        target = tvm.target.Target("nvidia/geforce-rtx-3080")
        dlight_sch = dlight.gpu.Reduction().apply(func, target, False)
        args = prepare_args(dlight_sch.mod["main"], {"n": 256})
        evaluate(dlight_sch.mod["main"], args)
        dlight_sch = dlight.gpu.reduction_new.Reduction().apply(func, target, False)
        evaluate(dlight_sch.mod["main"], args)
        print(dlight_sch.mod)


if __name__ == "__main__":
    main()
