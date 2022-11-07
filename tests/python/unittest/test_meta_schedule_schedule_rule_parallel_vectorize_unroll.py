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
from tvm.meta_schedule.testing.space_generation import (
    check_sketches,
    generate_design_space,
)
from tvm.script import tir as T
from tvm.target import Target

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class ParallelizeVectorizeUnroll:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block("root"):
            T.reads([])
            T.writes([])
            T.block_attr({"meta_schedule.parallel": 128, "meta_schedule.vectorize": 16, "meta_schedule.unroll_explicit": 2})
            for i, j, k in T.grid(1024, 1024, 1024):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# from tvm.script import tir as T
@tvm.script.ir_module
class PureSpatial:
    @T.prim_func
    def main(placeholder: T.Buffer[(1, 13, 13, 3, 85), "float32"], placeholder_1: T.Buffer[(1, 26, 26, 3, 85), "float32"], placeholder_2: T.Buffer[(1, 52, 52, 3, 85), "float32"], T_expand_dims: T.Buffer[(1, 80, 10647), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        T_strided_slice_with_axes = T.alloc_buffer([1, 52, 52, 3, 1], dtype="float32")
        T_sigmoid = T.alloc_buffer([1, 52, 52, 3, 1], dtype="float32")
        T_strided_slice_with_axes_1 = T.alloc_buffer([1, 52, 52, 3, 80], dtype="float32")
        T_sigmoid_1 = T.alloc_buffer([1, 52, 52, 3, 80], dtype="float32")
        T_multiply = T.alloc_buffer([1, 52, 52, 3, 80], dtype="float32")
        T_reshape = T.alloc_buffer([8112, 80], dtype="float32")
        T_strided_slice_with_axes_2 = T.alloc_buffer([1, 26, 26, 3, 1], dtype="float32")
        T_sigmoid_2 = T.alloc_buffer([1, 26, 26, 3, 1], dtype="float32")
        T_strided_slice_with_axes_3 = T.alloc_buffer([1, 26, 26, 3, 80], dtype="float32")
        T_sigmoid_3 = T.alloc_buffer([1, 26, 26, 3, 80], dtype="float32")
        T_multiply_1 = T.alloc_buffer([1, 26, 26, 3, 80], dtype="float32")
        T_reshape_1 = T.alloc_buffer([2028, 80], dtype="float32")
        T_strided_slice_with_axes_4 = T.alloc_buffer([1, 13, 13, 3, 1], dtype="float32")
        T_sigmoid_4 = T.alloc_buffer([1, 13, 13, 3, 1], dtype="float32")
        T_strided_slice_with_axes_5 = T.alloc_buffer([1, 13, 13, 3, 80], dtype="float32")
        T_sigmoid_5 = T.alloc_buffer([1, 13, 13, 3, 80], dtype="float32")
        T_multiply_2 = T.alloc_buffer([1, 13, 13, 3, 80], dtype="float32")
        T_reshape_2 = T.alloc_buffer([507, 80], dtype="float32")
        T_concat = T.alloc_buffer([10647, 80], dtype="float32")
        T_transpose = T.alloc_buffer([80, 10647], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 52, 52, 3, 1):
            with T.block("T_strided_slice_with_axes"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder_2[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)])
                T.writes(T_strided_slice_with_axes[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes[ax0, ax1, ax2, ax3, ax4] = placeholder_2[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)]
        for i0, i1, i2, i3, i4 in T.grid(1, 52, 52, 3, 1):
            with T.block("T_sigmoid"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 52, 52, 3, 80):
            with T.block("T_strided_slice_with_axes_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder_2[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)])
                T.writes(T_strided_slice_with_axes_1[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes_1[ax0, ax1, ax2, ax3, ax4] = placeholder_2[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)]
        for i0, i1, i2, i3, i4 in T.grid(1, 52, 52, 3, 80):
            with T.block("T_sigmoid_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid_1[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid_1[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes_1[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 52, 52, 3, 80):
            with T.block("T_multiply"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_sigmoid[ax0, ax1, ax2, ax3, 0], T_sigmoid_1[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply[ax0, ax1, ax2, ax3, ax4])
                T_multiply[ax0, ax1, ax2, ax3, ax4] = T_sigmoid[ax0, ax1, ax2, ax3, 0] * T_sigmoid_1[ax0, ax1, ax2, ax3, ax4]
        for i0, i1 in T.grid(8112, 80):
            with T.block("T_reshape"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_multiply[0, (ax1 // 80 + ax0) % 8112 // 156, (ax1 // 80 + ax0) % 156 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80])
                T.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = T_multiply[0, (ax1 // 80 + ax0) % 8112 // 156, (ax1 // 80 + ax0) % 156 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80]
        for i0, i1, i2, i3, i4 in T.grid(1, 26, 26, 3, 1):
            with T.block("T_strided_slice_with_axes_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder_1[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)])
                T.writes(T_strided_slice_with_axes_2[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes_2[ax0, ax1, ax2, ax3, ax4] = placeholder_1[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)]
        for i0, i1, i2, i3, i4 in T.grid(1, 26, 26, 3, 1):
            with T.block("T_sigmoid_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes_2[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid_2[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid_2[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes_2[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 26, 26, 3, 80):
            with T.block("T_strided_slice_with_axes_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder_1[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)])
                T.writes(T_strided_slice_with_axes_3[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes_3[ax0, ax1, ax2, ax3, ax4] = placeholder_1[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)]
        for i0, i1, i2, i3, i4 in T.grid(1, 26, 26, 3, 80):
            with T.block("T_sigmoid_3"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes_3[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid_3[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid_3[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes_3[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 26, 26, 3, 80):
            with T.block("T_multiply_1"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_sigmoid_2[ax0, ax1, ax2, ax3, 0], T_sigmoid_3[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply_1[ax0, ax1, ax2, ax3, ax4])
                T_multiply_1[ax0, ax1, ax2, ax3, ax4] = T_sigmoid_2[ax0, ax1, ax2, ax3, 0] * T_sigmoid_3[ax0, ax1, ax2, ax3, ax4]
        for i0, i1 in T.grid(2028, 80):
            with T.block("T_reshape_1"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_multiply_1[0, (ax1 // 80 + ax0) % 2028 // 78, (ax1 // 80 + ax0) % 78 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80])
                T.writes(T_reshape_1[ax0, ax1])
                T_reshape_1[ax0, ax1] = T_multiply_1[0, (ax1 // 80 + ax0) % 2028 // 78, (ax1 // 80 + ax0) % 78 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80]
        for i0, i1, i2, i3, i4 in T.grid(1, 13, 13, 3, 1):
            with T.block("T_strided_slice_with_axes_4"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)])
                T.writes(T_strided_slice_with_axes_4[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes_4[ax0, ax1, ax2, ax3, ax4] = placeholder[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(4)]
        for i0, i1, i2, i3, i4 in T.grid(1, 13, 13, 3, 1):
            with T.block("T_sigmoid_4"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes_4[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid_4[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid_4[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes_4[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 13, 13, 3, 80):
            with T.block("T_strided_slice_with_axes_5"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(placeholder[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)])
                T.writes(T_strided_slice_with_axes_5[ax0, ax1, ax2, ax3, ax4])
                T_strided_slice_with_axes_5[ax0, ax1, ax2, ax3, ax4] = placeholder[ax0, ax1, ax2, ax3, T.cast(ax4, "int64") + T.int64(5)]
        for i0, i1, i2, i3, i4 in T.grid(1, 13, 13, 3, 80):
            with T.block("T_sigmoid_5"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_strided_slice_with_axes_5[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_sigmoid_5[ax0, ax1, ax2, ax3, ax4])
                T_sigmoid_5[ax0, ax1, ax2, ax3, ax4] = T.sigmoid(T_strided_slice_with_axes_5[ax0, ax1, ax2, ax3, ax4], dtype="float32")
        for i0, i1, i2, i3, i4 in T.grid(1, 13, 13, 3, 80):
            with T.block("T_multiply_2"):
                ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                T.reads(T_sigmoid_4[ax0, ax1, ax2, ax3, 0], T_sigmoid_5[ax0, ax1, ax2, ax3, ax4])
                T.writes(T_multiply_2[ax0, ax1, ax2, ax3, ax4])
                T_multiply_2[ax0, ax1, ax2, ax3, ax4] = T_sigmoid_4[ax0, ax1, ax2, ax3, 0] * T_sigmoid_5[ax0, ax1, ax2, ax3, ax4]
        for i0, i1 in T.grid(507, 80):
            with T.block("T_reshape_2"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_multiply_2[0, (ax1 // 80 + ax0) % 507 // 39, (ax1 // 80 + ax0) % 39 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80])
                T.writes(T_reshape_2[ax0, ax1])
                T_reshape_2[ax0, ax1] = T_multiply_2[0, (ax1 // 80 + ax0) % 507 // 39, (ax1 // 80 + ax0) % 39 // 3, (ax1 // 80 + ax0) % 3, ax1 % 80]
        for i0, i1 in T.grid(10647, 80):
            with T.block("T_concat"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_reshape[ax0 - 2535, ax1], T_reshape_1[ax0 - 507, ax1], T_reshape_2[ax0, ax1])
                T.writes(T_concat[ax0, ax1])
                T_concat[ax0, ax1] = T.if_then_else(2535 <= ax0, T_reshape[ax0 - 2535, ax1], T.if_then_else(507 <= ax0, T_reshape_1[ax0 - 507, ax1], T_reshape_2[ax0, ax1], dtype="float32"), dtype="float32")
        for i0, i1 in T.grid(80, 10647):
            with T.block("T_transpose"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_concat[ax1, ax0])
                T.writes(T_transpose[ax0, ax1])
                T_transpose[ax0, ax1] = T_concat[ax1, ax0]
        for i0, i1, i2 in T.grid(1, 80, 10647):
            with T.block("T_expand_dims"):
                ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_transpose[ax1, ax2])
                T.writes(T_expand_dims[ax0, ax1, ax2])
                T_expand_dims[ax0, ax1, ax2] = T_transpose[ax1, ax2]


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_parallel_vectorize_unroll():
    @T.prim_func
    def Matmul_0(
        A: T.Buffer[(1024, 1024), "float32"],
        B: T.Buffer[(1024, 1024), "float32"],
        C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main"})
        # body
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr(
                {
                    "meta_schedule.parallel": 512,
                    "meta_schedule.unroll_explicit": 16,
                    "meta_schedule.vectorize": 32,
                }
            )
            for i, j, k in T.grid(1024, 1024, 1024):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    decision_0 = [
        ("SampleCategorical", 1),
    ]

    mod = Matmul
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm --num-cores=32"),
        types=None,
        sch_rules=[
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=16,
                max_vectorize_extent=32,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
        ],
    )
    check_sketches(
        mod,
        sketches=actual,
        expected_mods=[Matmul_0],
        expected_decisions=[decision_0],
    )


def test_parallel_vectorize_unroll_spatial():
    mod = PureSpatial
    actual = generate_design_space(
        kind="llvm",
        mod=mod,
        target=Target("llvm --num-cores=32"),
        types=None,
        sch_rules=[
            ms.schedule_rule.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,
                max_vectorize_extent=-1,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
        ],
    )
    assert len(actual) == 1
    trace = actual[0].trace.simplified(remove_postproc=True)
    assert not trace.insts


if __name__ == "__main__":
    test_parallel_vectorize_unroll()
    test_parallel_vectorize_unroll_spatial()
