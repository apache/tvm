from typing import Dict, List
import numpy as np

import tvm
from tvm import ir, te, tir, dlight
from tvm.contrib import nvcc
from tvm.script import tir as T, ir as I

TARGET = tvm.target.Target("nvidia/geforce-rtx-3080")
DEVICE = tvm.cuda(0)


@T.prim_func
def decode_gemv(lv19: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv20: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv19[v_i, v_j // T.int64(8)], lv20[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv19[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv20[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

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
    dlight_sch = dlight.gpu.DecodeGEMV().apply(decode_gemv, TARGET, False)
    args = prepare_args(dlight_sch.mod["main"], {"n": 256})
    evaluate(dlight_sch.mod["main"], args)

if __name__ == "__main__":
    main()