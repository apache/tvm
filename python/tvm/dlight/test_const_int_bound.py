import numpy as np
import tvm
from tvm.script import tir as T
from tvm import tir
from tvm.tir import IndexMap
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import ElementWise
from tvm.dlight.gpu import Fallback
from tvm import arith, tir

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
sch = tir.Schedule(ir_module)
block = sch.get_block("B")
loops = sch.get_loops(block)

loop_stmt = sch.get_sref(loops[-1]).stmt

ana = arith.Analyzer()

print(ana.const_int_bound(loop_stmt.loop_var))

