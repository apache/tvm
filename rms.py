from typing import Dict, List
import numpy as np

import tvm
from tvm import ir, te, tir, dlight
from tvm.contrib import nvcc
from tvm.script import tir as T, ir as I

TARGET = tvm.target.Target("nvidia/geforce-rtx-3080")
DEVICE = tvm.cuda(0)



@T.prim_func
def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))

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
    dlight_sch = dlight.gpu.DecodeGEMV().apply(rms_norm, TARGET, False)
    args = prepare_args(dlight_sch.mod["main"], {"n": 256})
    evaluate(dlight_sch.mod["main"], args)
    dlight_sch = dlight.gpu.Reduction().apply(rms_norm, TARGET, False)
    evaluate(dlight_sch.mod["main"], args)

if __name__ == "__main__":
    main()