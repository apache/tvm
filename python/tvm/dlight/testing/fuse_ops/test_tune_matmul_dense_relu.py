import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time


@T.prim_func
def fused_dense_add_relu(lv: T.Buffer((T.int64(128), T.int64(150528)), "float32"), param_0: T.Buffer((T.int64(128), T.int64(150528)), "float32"), param_1: T.Buffer((T.int64(1), T.int64(128)), "float32"), var_T_relu_intermediate: T.Buffer((T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_matmul_NT_intermediate = T.alloc_buffer((T.int64(128), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(128), T.int64(128)))
    for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(150528)):
        with T.block("T_matmul_NT"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv[v_i0, v_k], param_0[v_i1, v_k])
            T.writes(var_T_matmul_NT_intermediate[v_i0, v_i1])
            with T.init():
                var_T_matmul_NT_intermediate[v_i0, v_i1] = T.float32(0)
            var_T_matmul_NT_intermediate[v_i0, v_i1] = var_T_matmul_NT_intermediate[v_i0, v_i1] + lv[v_i0, v_k] * param_0[v_i1, v_k]
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_matmul_NT_intermediate[v_ax0, v_ax1], param_1[T.int64(0), v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_T_matmul_NT_intermediate[v_ax0, v_ax1] + param_1[T.int64(0), v_ax1]
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with T.block("T_relu"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1])
            T.writes(var_T_relu_intermediate[v_ax0, v_ax1])
            var_T_relu_intermediate[v_ax0, v_ax1] = T.max(var_T_add_intermediate[v_ax0, v_ax1], T.float32(0))

@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, [128, 150528], dtype="float32")
    B = T.match_buffer(b, [128, 150528], dtype="float32")
    C = T.match_buffer(c, [128, 128], dtype="float32")
    
    for i, j, k in T.grid(128, 128, 150528):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + \
                A[vi, vk] * B[vj, vk]

func = fused_dense_add_relu
# func = matmul
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(20)
cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
