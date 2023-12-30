import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise, GeneralReduction, GEMV
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel, apply_and_build
from tvm import te
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
import ctypes
_cudart = ctypes.CDLL('libcudart.so')


def profile_start():
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def profile_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

def gemv_i4(M, N, K, dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=dtype)
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C])
    func = func.with_attr("inconsistent", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format':{
                'bits': 16,
                'format': 'float',
            }
        }
    })
    return tvm.IRModule.from_expr(func)

M = 1
N = 16384
K = 16384
benchmark_results = {}
ir_module = gemv_i4(M, N, K, "float16")
func = ir_module["main"]
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(20)
rule = GEMV()

tune_start = time.time()
cpresults, best = apply_and_build(func, rule, configs, arch, parallel_build=False)
fast_tune_time = time.time() - tune_start
print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency * 1e3))
print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency * 1e3))

rule = GEMV()
default_tune_start = time.time()
sch_default = rule.apply(func, target, False)
mod_default = tvm.build(sch_default.mod["main"], target="cuda")
default_tune_time = time.time() - default_tune_start

args = func.buffer_map.values()
        
profile_tensors = []
for arg in args:
    if arg.dtype == "int8":
        profile_tensors.append(
            tvm.nd.array(
                np.random.randint(-127, 128, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )
    else:
        profile_tensors.append(tvm.nd.array(
            np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype), device=arch.device)
        )

tvm_a, tvm_b, tvm_c = profile_tensors
numpy_b = tvm_b.numpy()
def interleave_weight(qweight):
    # reinterpret the data type of qweight to int32
    qweight = qweight.view(np.int32)
     
    new_qweight = np.zeros_like(qweight)
    new_qweight |= (qweight & 0x0000000f)
    new_qweight |= (qweight & 0x000000f0) << 12
    new_qweight |= (qweight & 0x00000f00) >> 4
    new_qweight |= (qweight & 0x0000f000) << 8
    new_qweight |= (qweight & 0x000f0000) >> 8
    new_qweight |= (qweight & 0x00f00000) << 4
    new_qweight |= (qweight & 0x0f000000) >> 12
    new_qweight |= (qweight & 0xf0000000)
    return new_qweight.view(np.int8)

permutated_b = interleave_weight(numpy_b)
tvm_new_b = tvm.nd.array(permutated_b.reshape(tvm_b.shape), device=arch.device)
profile_start()
best.mod(tvm_a, tvm_new_b, tvm_c)
profile_stop()

print(best.code)
print(tvm_c)

int4_tvm_c = tvm.nd.array(np.zeros((M, N), dtype="float16"), device=arch.device)
mod_default(tvm_a, tvm_b, tvm_c)

print(tvm_c)