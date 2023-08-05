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
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

@T.prim_func
def element_wise(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")) -> None:
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
def element_wise_set_dtype(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
    B = T.alloc_buffer((128, 128), "float16")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B[vi, vj] = T.cast(A[vi, vj] * 2.0, "float16")
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
            C[vi, vj] = T.cast(B[vi, vj], "float32") + 1.0

@T.prim_func
def element_wise_subregion_match(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")) -> None:
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
def element_wise_subregion_match_set_dtype(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")) -> None:
    B = T.alloc_buffer((128, 128), "float16")
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(A[vi, vj])
            T.writes(B[vi, vj])
            B_subregion0 = T.match_buffer(B[vi, vj], (), "float16", offset_factor=1)
            B_subregion0[()] = T.cast(A[vi, vj] * 2.0, "float16")
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            T.reads(B[vi, vj])
            T.writes(C[vi, vj])
            B_subregion1 = T.match_buffer(B[vi, vj], (), "float16", offset_factor=1)
            C[vi, vj] = T.cast(B_subregion1[()], "float32") + 1.0


use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})

def test_set_dtype(use_block_name):
    func = element_wise
    sch = tir.Schedule(func, debug_mask="all")
    sch.unsafe_set_dtype("B" if use_block_name else sch.get_block("B"), 0, "float16")
    assert_structural_equal_ignore_global_symbol(element_wise_set_dtype, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func)

def test_set_dtype_fail_on_output_buffer(use_block_name):
    func = element_wise
    sch = tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.tir.ScheduleError):
        sch.unsafe_set_dtype('C' if use_block_name else sch.get_block("C"), 0, "float16")

def test_set_dtype_fail_on_index_out_of_bound():
    func = element_wise
    sch = tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.tir.ScheduleError):
        sch.unsafe_set_dtype(sch.get_block("B"), 1, "float64")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.unsafe_set_dtype(sch.get_block("B"), -1, "float64")

def test_set_dtype_subregion():
    func = element_wise_subregion_match
    sch = tir.Schedule(func, debug_mask='all')
    sch.unsafe_set_dtype(sch.get_block("B"), 0, "float16")
    assert_structural_equal_ignore_global_symbol(element_wise_subregion_match_set_dtype, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=func)


if __name__ == "__main__":
    tvm.testing.main()
