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
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.tir import IndexMap
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

@T.prim_func
def element_wise(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_set_axis_separator(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32", axis_separators=[1])

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def element_wise_set_axis_separator_input_buffer(A: T.Buffer(shape=(128, 128), dtype="float32", axis_separators=(1,)), C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def element_wise_subregion_match(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer((128, 128), dtype="float32")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion0 = T.match_buffer(B[vi, vj], [], offset_factor=1)
            B_subregion0[()] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion1 = T.match_buffer(B[vi, vj], [], offset_factor=1)
            C[vi, vj] = B_subregion1[()] + 1.0


@T.prim_func
def element_wise_subregion_match_set_axis_separator(A: T.Buffer[(128, 128), "float32"], C: T.Buffer[(128, 128), "float32"]) -> None:
    B = T.alloc_buffer([128, 128], dtype="float32", axis_separators=[1])

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion0 = T.match_buffer(B[vi, vj], [], dtype="float32", offset_factor=1, axis_separators=[1])
            B_subregion0[()] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion1 = T.match_buffer(B[vi, vj], [], dtype="float32", offset_factor=1, axis_separators=[1])
            C[vi, vj] = B_subregion1[()] + T.float32(1)


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

use_sugared_transform = tvm.testing.parameter(
    by_dict={"set_axis_separators": False, "transform_layout_sugared": True}
)

def test_set_axis_separator(use_sugared_transform):
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')

    if use_sugared_transform:
        s.set_axis_separator(s.get_block("B"), ("write",0), [1])
    else:
        s.transform_layout(block='B', buffer='B', index_map=lambda i,j: [i,IndexMap.AXIS_SEPARATOR,j])

    tvm.ir.assert_structural_equal(element_wise_set_axis_separator, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_set_scope_fail_on_index_out_of_bound():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    with pytest.raises(AssertionError):
        s.set_axis_separator(s.get_block("B"), ("write",1),[1])
    with pytest.raises(AssertionError):
        s.set_axis_separator(s.get_block("B"), ("read",-1),[1])


def test_set_axis_separator_input_buffer(use_sugared_transform):
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')

    if use_sugared_transform:
        s.transform_layout(block='B', buffer='A', index_map=lambda i,j: [i,IndexMap.AXIS_SEPARATOR,j])
    else:
        s.set_axis_separator(s.get_block("B"), ("read",0), [1])


    tvm.ir.assert_structural_equal(element_wise_set_axis_separator_input_buffer, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_set_axis_separator_subregion(use_sugared_transform):
    func = element_wise_subregion_match
    s = tir.Schedule(func, debug_mask='all')

    if use_sugared_transform:
        s.transform_layout(block='B', buffer='B', index_map=lambda i,j: [i,IndexMap.AXIS_SEPARATOR,j])
    else:
        s.set_axis_separator(s.get_block("B"), ("write",0), [1])

    tvm.ir.assert_structural_equal(element_wise_subregion_match_set_axis_separator, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


if __name__ == "__main__":
    tvm.testing.main()
