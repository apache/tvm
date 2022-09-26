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
                ms.postproc.RewriteLayout(),
            ],
            mutator_probs={},
        ),
        task_name="test",
    )
    return ctx


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    def transform(self):
        def inner(mod):
            target = Target("cuda", host="llvm")
            ctx = ms.TuneContext(
                mod=mod,
                target=target,
                space_generator=ms.space_generator.PostOrderApply(
                    sch_rules=[],
                    postprocs=[
                        ms.postproc.RewriteLayout(),
                    ],
                    mutator_probs={},
                ),
                task_name="test",
            )
            sch = tvm.tir.Schedule(mod, debug_mask="all")
            sch.enter_postproc()
            assert ctx.space_generator.postprocs[0].apply(sch)
            return sch.mod

        return inner


class TestTIRMatmul(BaseBeforeAfter):
    """Main functionality test

    A new block should be inserted to transform the layout, with the
    compute block operating on the temporary transformed buffer.
    """

    def before(
        A: T.Buffer[(16, 16), "float32"],
        B: T.Buffer[(16, 16), "float32"],
        C: T.Buffer[(16, 16), "float32"],
    ) -> None:
        T.func_attr({"layout_free_buffers": [1]})
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.S(16, i0 * 4 + i1)
                vj = T.axis.S(16, j)
                vk = T.axis.R(16, k0 * 4 + k1)
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    def expected(
        A: T.Buffer[(16, 16), "float32"],
        B: T.Buffer[(16, 16), "float32"],
        C: T.Buffer[(16, 16), "float32"],
    ) -> None:
        T.func_attr({"layout_free_buffers": [1]})
        B_reindex = T.alloc_buffer([16, 4, 4], dtype="float32")
        for ax0, ax1 in T.grid(16, 16):
            with T.block("layout_rewrite"):
                i0, i1 = T.axis.remap("SS", [ax0, ax1])
                T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
                B_reindex[i1, i0 // 4, i0 % 4] = B[i0, i1]
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.spatial(16, i0 * 4 + i1)
                vj = T.axis.spatial(16, j)
                vk = T.axis.reduce(16, k0 * 4 + k1)
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B_reindex[vj, vk // 4, vk % 4]


class TestRewrittenBuffersMustOccurWithinBlock(BaseBeforeAfter):
    """Buffers must occur within a Block"""

    def before(
        A: T.Buffer[(16, 16), "float32"],
    ) -> None:
        T.func_attr({"layout_free_buffers": [0]})
        for i, j in T.grid(16, 16):
            T.evaluate(A[i, j])

    expected = tvm.TVMError


class TestExtentOne(BaseBeforeAfter):
    """Buffers with dimensions of extent 1 can be transformed

    Regression test for a previous bug, in which the removal of
    trivial variables resulted in an error in `IndexMap::Inverse`.
    """

    def before(
        A: T.Buffer[(16, 1), "float32"],
    ) -> None:
        T.func_attr({"layout_free_buffers": [0]})
        for i, j in T.grid(16, 1):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.evaluate(A[vi, vj])

    def expected(A: T.Buffer[(16, 1), "float32"]):
        T.func_attr({"layout_free_buffers": [0]})

        A_global = T.alloc_buffer([16], dtype="float32")
        for ax0, ax1 in T.grid(16, 1):
            with T.block("A_global"):
                v0, v1 = T.axis.remap("SS", [ax0, ax1])
                T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
                A_global[v0] = A[v0, v1]

        for i, j in T.grid(16, 1):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.evaluate(A_global[vi])


@T.prim_func
def tir_matmul(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"layout_free_buffers": [1]})
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with T.block("matmul"):
            vi = T.axis.S(16, i0 * 4 + i1)
            vj = T.axis.S(16, j)
            vk = T.axis.R(16, k0 * 4 + k1)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def rewritten_tir_matmul(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"layout_free_buffers": [1]})
    B_reindex = T.alloc_buffer([16, 4, 4], dtype="float32")
    for ax0, ax1 in T.grid(16, 16):
        with T.block("layout_rewrite"):
            i0, i1 = T.axis.remap("SS", [ax0, ax1])
            T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
            B_reindex[i1, i0 // 4, i0 % 4] = B[i0, i1]
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with T.block("matmul"):
            vi = T.axis.spatial(16, i0 * 4 + i1)
            vj = T.axis.spatial(16, j)
            vk = T.axis.reduce(16, k0 * 4 + k1)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B_reindex[vj, vk // 4, vk % 4]


def test_layout_rewrite():
    target = _target()
    ctx = _create_context(tir_matmul, target)
    sch = tvm.tir.Schedule(tir_matmul, debug_mask="all")
    sch.enter_postproc()
    assert ctx.space_generator.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod["main"], rewritten_tir_matmul)


if __name__ == "__main__":
    tvm.testing.main()
