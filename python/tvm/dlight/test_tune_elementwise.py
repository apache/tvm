import numpy as np
import tvm
from tvm.script import tir as T
from tvm.tir import IndexMap
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise
from tvm.dlight.gpu import Fallback

M = N = 16384
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [M, N], dtype="float16")
        B = T.match_buffer(b, [M, N], dtype="float16")
        
        for i, j in T.grid(M, N):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                T.writes(B[vi, vj])
                B[vi, vj] = A[vi, vj]
                
ir_module = MyModule
func = ir_module["main"]
target = tvm.target.Target("cuda")
arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(1)
rule = ElementWise()

best_config = None
best_latency = 1e10
best_schedule = None
for config in configs:
    sch = rule.apply_config(func, config)
    mod = tvm.build(sch.mod["main"], target="cuda")
    cuda_a = tvm.nd.array(np.random.uniform(0, 1, [M, N]).astype("float16"), device=arch.device)
    cuda_b = tvm.nd.array(np.empty([M, N]).astype("float16"), device=arch.device)
    mod(cuda_a, cuda_b)
    
    print("Applying schedule with config ", config)
    timer_cuda_mod = mod.time_evaluator(
    mod.entry_name, arch.device, number=5)
    t = timer_cuda_mod(cuda_a, cuda_b).mean
    print("Time cost of this config: {:.3f} ms".format(t * 1e3))
    
    if t < best_latency:
        best_latency = t
        best_config = config
        best_schedule = sch

print("The best latency is {:.3f} ms".format(best_latency * 1e3))

# evaluate the performance of the default schedule
rule = Fallback()
sch_default = rule.apply(func, target, False)

mod_default = tvm.build(sch_default.mod["main"], target="cuda")

timer_cuda_mod = mod_default.time_evaluator(
    mod_default.entry_name, arch.device, number=5)
t = timer_cuda_mod(cuda_a, cuda_b).mean

print("Time cost of default schedule: {:.3f} ms".format(t * 1e3))
