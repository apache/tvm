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
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.RewriteReductionBlock(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Matmul_before_rewrite:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    T.reads([A[v0, v1]])
                                    T.writes([A_shared[v0, v1]])
                                    T.block_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        T.block_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_1 * 32 + i2_2)
                                T.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class Matmul_after_rewrite:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    T.reads([A[v0, v1]])
                                    T.writes([A_shared[v0, v1]])
                                    T.block_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        T.block_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(2, 2, 16, 2):
                            with T.block("C_init"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3_init * 16 + i0_4_init)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3_init * 2 + i1_4_init)
                                T.reads([])
                                T.writes([C_local[i, j]])
                                C_local[i, j] = T.float32(0)
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C_update"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_1 * 32 + i2_2)
                                T.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class Softmax_cross_thread_reduction:
    @T.prim_func
    def main(A: T.Buffer((256, 256), "float32"), T_softmax_norm: T.Buffer((256, 256), "float32")) -> None:
        T_softmax_maxelem_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        T_softmax_expsum_shared = T.alloc_buffer([256], dtype="float32", scope="shared")
        for i0 in T.serial(256):
            for ax0, ax1_0 in T.grid(1, 8):
                for ax1_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block("T_softmax_maxelem"):
                        i0_1 = T.axis.spatial(256, i0)
                        k = T.axis.reduce(256, ax1_0 * 32 + ax1_1)
                        T.reads(T_softmax_maxelem_shared[i0_1], A[i0_1, k])
                        T.writes(T_softmax_maxelem_shared[i0_1])
                        with T.init():
                            T_softmax_maxelem_shared[i0_1] = T.float32(-3.4028234663852886e+38)
                        T_softmax_maxelem_shared[i0_1] = T.max(T_softmax_maxelem_shared[i0_1], A[i0_1, k])
            for ax0, ax1_0 in T.grid(1, 8):
                for ax1_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block("T_softmax_expsum"):
                        i0_2 = T.axis.spatial(256, i0)
                        k = T.axis.reduce(256, ax1_0 * 32 + ax1_1)
                        T.reads(T_softmax_expsum_shared[i0_2], A[i0_2, k], T_softmax_maxelem_shared[i0_2])
                        T.writes(T_softmax_expsum_shared[i0_2])
                        with T.init():
                            T_softmax_expsum_shared[i0_2] = T.float32(0)
                        T_softmax_expsum_shared[i0_2] = T_softmax_expsum_shared[i0_2] + T.exp(A[i0_2, k] - T_softmax_maxelem_shared[i0_2], dtype="float32")
            for i1_0 in T.serial(8):
                for i1_1 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block("T_softmax_norm"):
                        i0_3 = T.axis.spatial(256, i0)
                        i1 = T.axis.spatial(256, i1_0 * 32 + i1_1)
                        T.reads(A[i0_3, i1], T_softmax_maxelem_shared[i0_3], T_softmax_expsum_shared[i0_3])
                        T.writes(T_softmax_norm[i0_3, i1])
                        T.block_attr({"axis":1})
                        T_softmax_norm[i0_3, i1] = T.exp(A[i0_3, i1] - T_softmax_maxelem_shared[i0_3], dtype="float32") / T_softmax_expsum_shared[i0_3]


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_tiled_matmul():
    mod = Matmul_before_rewrite
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, Matmul_after_rewrite)


def test_rewrite_softmax():
    mod = Softmax_cross_thread_reduction
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    # The module should not be rewritten
    tvm.ir.assert_structural_equal(sch.mod, Softmax_cross_thread_reduction)


if __name__ == "__main__":
    test_rewrite_tiled_matmul()
    test_rewrite_softmax()
