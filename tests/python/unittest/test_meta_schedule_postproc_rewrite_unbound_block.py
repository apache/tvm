# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import tvm
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import RewriteUnboundBlock
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            RewriteUnboundBlock(),
        ],
        task_name="test",
    )
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
    return ctx


# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks


@tvm.script.ir_module
class Before_cooperative_fetch:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        for i, j in T.grid(512, 512):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + 1.0


@tvm.script.ir_module
class After_cooperative_fetch:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        for i_j_fused_0 in T.thread_binding(0, 8192, thread="blockIdx.x"):
            for i_j_fused_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(512, (i_j_fused_0 * 32 + i_j_fused_1) // 512)
                    vj = T.axis.spatial(512, (i_j_fused_0 * 32 + i_j_fused_1) % 512)
                    B[vi, vj] = A[vi, vj] + 1.0


@tvm.script.ir_module
class Before_norm_bmn:
    @T.prim_func
    def main(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[(1,), "float32"]) -> None:
        C = T.alloc_buffer([1], dtype="float32")
        for i0, i1, i2 in T.grid(1, 256, 256):
            with T.block("C"):
                b, i, j = T.axis.remap("SRR", [i0, i1, i2])
                with T.init():
                    C[b] = T.float32(0)
                C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0 in T.serial(1):
            with T.block("D"):
                b = T.axis.S(1, i0)
                D[b] = T.sqrt(C[b], dtype="float32")


@tvm.script.ir_module
class After_norm_bmn:
    @T.prim_func
    def main(A: T.Buffer[(1, 256, 256), "float32"], D: T.Buffer[(1,), "float32"]) -> None:
        C = T.alloc_buffer([1], dtype="float32")
        for i0_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                for i1, i2 in T.grid(256, 256):
                    with T.block("C"):
                        b = T.axis.S(1, 0)
                        i, j = T.axis.remap("RR", [i1, i2])
                        T.where(i0_fused_1 < 1)
                        with T.init():
                            C[b] = T.float32(0)
                        C[b] = C[b] + A[b, i, j] * A[b, i, j]
        for i0_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for i0_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                with T.block("D"):
                    b = T.axis.S(1, 0)
                    T.where(i0_fused_1 < 1)
                    D[b] = T.sqrt(C[b], dtype="float32")


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_cooperative_fetch():
    mod = Before_cooperative_fetch
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, After_cooperative_fetch)


def test_rewrite_norm_bmn():
    mod = Before_norm_bmn
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, After_norm_bmn)


if __name__ == "__main__":
    test_rewrite_cooperative_fetch()
    test_rewrite_norm_bmn()
