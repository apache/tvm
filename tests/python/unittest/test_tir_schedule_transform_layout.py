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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks


def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16


@T.prim_func
def two_elementwise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_intermediate_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((8, 8, 16, 16), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0


@T.prim_func
def two_elementwise_transformed_input_buffer(
    A: T.Buffer[(8, 8, 16, 16), "float32"], C: T.Buffer[(128, 128), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi // 16, vj // 16, vi % 16, vj % 16] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_output_buffer(
    A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(8, 8, 16, 16), "float32"]
) -> None:
    B = T.alloc_buffer((128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi // 16, vj // 16, vi % 16, vj % 16] = B[vi, vj] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on

use_sugared_transform = tvm.testing.parameter(
    by_dict={"transform_layout": False, "transform_layout_sugared": True}
)


def test_two_elementwise_transform_intermediate_buffer(use_sugared_transform):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_sugared_transform:
        sch.transform_layout(
            block="B",
            buffer="B",
            index_map=packed_index_map_func,
        )
    else:
        block = sch.get_block("B")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_intermediate_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_input_buffer(use_sugared_transform):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_sugared_transform:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="B",
            buffer="A",
        )
    else:
        block = sch.get_block("B")
        sch.transform_layout(block, ("read", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_input_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_output_buffer(use_sugared_transform):
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    if use_sugared_transform:
        sch.transform_layout(
            index_map=packed_index_map_func,
            block="C",
            buffer="C",
        )
    else:
        block = sch.get_block("C")
        sch.transform_layout(block, ("write", 0), packed_index_map_func)

    tvm.ir.assert_structural_equal(two_elementwise_transformed_output_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_simplify():
    sch = tir.Schedule(two_elementwise, debug_mask="all")

    i, j = sch.get_loops(sch.get_block("C"))
    i, i_inner = sch.split(i, factors=[None, 16])
    j, j_inner = sch.split(j, factors=[None, 16])

    sch.reorder(
        i,
        j,
        i_inner,
        j_inner,
    )

    block_outer = sch.blockize(i_inner)

    B = sch.cache_read(block_outer, 0, "global")
    sch.transform_layout(B, ("write", 0), lambda i, j: (i // 16, j // 16, i % 16, j % 16))

    @T.prim_func
    def ref(B: T.Buffer[(8, 8, 16, 16), "float32"], C: T.Buffer[(128, 128), "float32"]):
        for i_0, j_0 in T.grid(8, 8):
            with T.block("C_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads(B[vi_o, vj_o, 0:16, 0:16])
                T.writes(C[vi_o * 16 : vi_o * 16 + 16, vj_o * 16 : vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("C"):
                        vi, vj = T.axis.remap("SS", [i_1, j_1])
                        T.reads(B[vi_o, vj_o, vi, vj])
                        T.writes(C[vi_o * 16 + vi, vj_o * 16 + vj])
                        C[vi_o * 16 + vi, vj_o * 16 + vj] = B[vi_o, vj_o, vi, vj] + T.float32(1)

                        # Without simplification
                        # T.reads(B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16])
                        # C[...] = B[vi // 16 + vi_o, vj // 16 + vj_o, vi % 16, vj % 16] + T.float32(1)

    tvm.ir.assert_structural_equal(ref.body.block.body, sch.get(sch.get_loops(block_outer)[0]))


def test_var_args_sugar():
    @T.prim_func
    def summation_3d(
        A: T.Buffer[(1024, 1024, 32), "float32"], B: T.Buffer[(1,), "float32"]
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with T.block("compute"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk]

    @T.prim_func
    def summation_3d_split(
        A: T.Buffer[(1024, 1024, 8, 4), "float32"], B: T.Buffer[(1,), "float32"]
    ) -> None:
        B[0] = 0
        for i, j, k in T.grid(1024, 1024, 32):
            with T.block("compute"):
                vi, vj, vk = T.axis.remap("SSS", [i, j, k])
                B[0] = B[0] + A[vi, vj, vk // 4, vk % 4]

    sch = tir.Schedule(summation_3d, debug_mask="all")
    sch.transform_layout(
        index_map=lambda *indices, k: [*indices, k // 4, k % 4], block="compute", buffer="A"
    )
    tvm.ir.assert_structural_equal(summation_3d_split, sch.mod["main"])


if __name__ == "__main__":
    tvm.testing.main()
