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
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks

def packed_index_map_func(m, n):
    return m // 16, n // 16, m % 16, n % 16


@T.prim_func
def two_elementwise(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_intermediate_buffer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((8, 8, 16, 16), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi // 16, vj // 16, vi % 16, vj % 16] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi // 16, vj // 16, vi % 16, vj % 16] + 1.0


@T.prim_func
def two_elementwise_transformed_input_buffer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (8, 8, 16, 16), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (128, 128), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi // 16, vj // 16, vi % 16, vj % 16] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def two_elementwise_transformed_output_buffer(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.match_buffer(c, (8, 8, 16, 16), "float32")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi // 16, vj // 16, vi % 16, vj % 16] = B[vi, vj] + 1.0


@T.prim_func
def permuted_shared_memory(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    A_shared = T.alloc_buffer((128, 128), scope="shared")
    for i0, j0, in T.grid(32, 4):
        for fused_i1_j1 in T.thread_binding(0, 32, 'threadIdx.x'):
            for j2 in T.vectorized(0, 4):
                with T.block("A_shared"):
                    vi = T.axis.S(128, i0 * 4 + fused_i1_j1 // 8)
                    vj = T.axis.S(128, j0 * 32 + fused_i1_j1 % 8 * 4 + j2)
                    A_shared[vi, vj] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A_shared[vi, vj] + 1.0


@T.prim_func
def permuted_shared_memory_transformed(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    C = T.match_buffer(c, (128, 128))
    A_shared = T.alloc_buffer((32, 4, 4, 32), scope="shared")
    for i0, j0, in T.grid(32, 4):
        for fused_i1_j1 in T.thread_binding(0, 32, 'threadIdx.x'):
            for j2 in T.vectorized(0, 4):
                with T.block("A_shared"):
                    vi = T.axis.S(128, i0 * 4 + fused_i1_j1 // 8)
                    vj = T.axis.S(128, j0 * 32 + fused_i1_j1 % 8 * 4 + j2)
                    A_shared[vi // 4, vj // 32, vi % 4, (((vj % 32) // 8) ^ (vi % 4)) + vj % 8] = A[vi, vj]
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A_shared[vi // 4, vj // 32, vi % 4, (((vj % 32) // 8) ^ (vi % 4)) + vj % 8] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks
# fmt: on


def test_two_elementwise_transform_intermediate_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    sch.transform_layout(block, 0, False, packed_index_map_func)
    tvm.ir.assert_structural_equal(two_elementwise_transformed_intermediate_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_input_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("B")
    sch.transform_layout(block, 0, True, packed_index_map_func)
    print(sch.mod["main"].script())
    tvm.ir.assert_structural_equal(two_elementwise_transformed_input_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


def test_two_elementwise_transform_output_buffer():
    sch = tir.Schedule(two_elementwise, debug_mask="all")
    block = sch.get_block("C")
    sch.transform_layout(block, 0, False, packed_index_map_func)
    tvm.ir.assert_structural_equal(two_elementwise_transformed_output_buffer, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=two_elementwise)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
