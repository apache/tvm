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
import tvm.testing
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.testing import te_workload
from tvm.script import tir as T
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> ms.TuneContext:
    ctx = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=[],
            postprocs=[
                ms.postproc.RewriteCooperativeFetch(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class AfterRewrite0:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        # body
        # with T.block("root")
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
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_0 * 512 + i2_1 * 32 + i2_2)
                                T.reads([A_shared[i, k], B_shared[k, j]])
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
class WarpExecutionAfterRewrite:
    @T.prim_func
    def main(
        A: T.Buffer((512, 512), "float32"),
        B: T.Buffer((512, 512), "float32"),
        C: T.Buffer((512, 512), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.y"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(
                                    0, 32, thread="threadIdx.x"
                                ):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(
                                            512,
                                            (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 32
                                                + ax0_ax1_fused_2
                                            )
                                            // 512,
                                        )
                                        v1 = T.axis.spatial(
                                            512,
                                            (
                                                ax0_ax1_fused_0 * 256
                                                + ax0_ax1_fused_1 * 32
                                                + ax0_ax1_fused_2
                                            )
                                            % 512,
                                        )
                                        T.reads([A[v0, v1]])
                                        T.writes([A_shared[v0, v1]])
                                        A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 32):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(
                                    0, 32, thread="threadIdx.x"
                                ):
                                    for ax0_ax1_fused_3 in T.vectorized(0, 2):
                                        with T.block("B_shared"):
                                            v0 = T.axis.spatial(
                                                512,
                                                (
                                                    ax0_ax1_fused_0 * 512
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                // 32,
                                            )
                                            v1 = T.axis.spatial(
                                                512,
                                                i0_0_i1_0_fused * 32
                                                + (
                                                    ax0_ax1_fused_0 * 512
                                                    + ax0_ax1_fused_1 * 64
                                                    + ax0_ax1_fused_2 * 2
                                                    + ax0_ax1_fused_3
                                                )
                                                % 32,
                                            )
                                            T.reads([B[v0, v1]])
                                            T.writes([B_shared[v0, v1]])
                                            B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(
                                    512,
                                    i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4,
                                )
                                k = T.axis.reduce(512, i2_0 * 512 + i2_1 * 32 + i2_2)
                                T.reads([A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                T.block_attr({"warp_execution": 1})
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(
                                512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1
                            )
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_cooperative_fetch():
    mod = create_prim_func(te_workload.matmul(n=512, m=512, k=512))
    target = _target()
    ctx = _create_context(mod, target)

    sch = tir.Schedule(mod, debug_mask="all")
    # fmt: off
    # pylint: disable=line-too-long,invalid-name
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 16, 1, 2, 16])
    l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 8, 2, 2])
    l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])
    v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64, decision=[1, 16, 32])
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 = sch.fuse(l10, l20)
    sch.bind(loop=l31, thread_axis="blockIdx.x")
    l32 = sch.fuse(l11, l21)
    sch.bind(loop=l32, thread_axis="vthread.x")
    l33 = sch.fuse(l12, l22)
    sch.bind(loop=l33, thread_axis="threadIdx.x")
    b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l39, l40 = sch.get_loops(block=b34)
    l41 = sch.fuse(l39, l40)
    _, v43 = sch.sample_perfect_tile(loop=l41, n=2, max_innermost_factor=4, decision=[262144, 1])
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l49, l50 = sch.get_loops(block=b44)
    l51 = sch.fuse(l49, l50)
    _, v53 = sch.sample_perfect_tile(loop=l51, n=2, max_innermost_factor=4, decision=[8192, 2])
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v53)
    sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=True)
    # pylint: enable=line-too-long,invalid-name
    # fmt: on
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, AfterRewrite0)


def test_rewrite_warp_execution():
    mod = create_prim_func(te_workload.matmul(n=512, m=512, k=512))
    target = _target()
    ctx = _create_context(mod, target)

    sch = tir.Schedule(mod, debug_mask="all")
    # fmt: off
    # pylint: disable=line-too-long,invalid-name
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    l2, l3, l4 = sch.get_loops(block=b0)
    sch.annotate(b0, "warp_execution", 1)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 16, 1, 2, 16])
    l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 8, 2, 2])
    l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])
    v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64, decision=[1, 16, 32])
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 = sch.fuse(l10, l20)
    sch.bind(loop=l31, thread_axis="blockIdx.x")
    l32 = sch.fuse(l11, l21)
    sch.bind(loop=l32, thread_axis="vthread.x")
    l33 = sch.fuse(l12, l22)
    sch.bind(loop=l33, thread_axis="threadIdx.y")
    b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l39, l40 = sch.get_loops(block=b34)
    l41 = sch.fuse(l39, l40)
    _, v43 = sch.sample_perfect_tile(loop=l41, n=2, max_innermost_factor=4, decision=[262144, 1])
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l49, l50 = sch.get_loops(block=b44)
    l51 = sch.fuse(l49, l50)
    _, v53 = sch.sample_perfect_tile(loop=l51, n=2, max_innermost_factor=4, decision=[8192, 2])
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v53)
    sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=True)
    # pylint: enable=line-too-long,invalid-name
    # fmt: on
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, WarpExecutionAfterRewrite)


if __name__ == "__main__":
    tvm.testing.main()
