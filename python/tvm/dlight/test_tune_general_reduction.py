import numpy as np
import tvm
from tvm.script import tir as T
from tvm.tir import IndexMap
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise, GeneralReduction, GEMV
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel
M = 1
N = K = 16384
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, K], dtype="float16")
        B = T.match_buffer(b, [N, K], dtype="float16")
        C = T.match_buffer(c, [M, N], dtype="float16")
        
        for i, j, k in T.grid(M, N, K):
            with T.block("B"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + \
                    A[vi, vk] * B[vj, vk]
                
ir_module = MyModule
func = ir_module["main"]
target = tvm.target.Target("nvidia/nvidia-a100")

arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(20)

rule = GEMV()

best = apply_and_build_parallel(func, rule, configs, arch)

print("[FastDlight] The best latency is {:.3f} ms".format(best.latency * 1e3))

# evaluate the performance of the default schedule
rule = GEMV()
sch_default = rule.apply(func, target, False)

mod_default = tvm.build(sch_default.mod["main"], target="cuda")

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
