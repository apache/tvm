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
def element_wise_set_scope(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")) -> None:
    B_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_shared[vi, vj] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B_shared[vi, vj] + T.float32(1)


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
def element_wise_subregion_match_set_scope(A: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")) -> None:
    B_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared")

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion0_shared = T.match_buffer(B_shared[vi, vj], [], dtype="float32", scope="shared", offset_factor=1)
            B_subregion0_shared[()] = A[vi, vj] * T.float32(2)
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            B_subregion1_shared = T.match_buffer(B_shared[vi, vj], [], dtype="float32", scope="shared", offset_factor=1)
            C[vi, vj] = B_subregion1_shared[()] + T.float32(1)


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

use_block_name = tvm.testing.parameter(by_dict={"block_obj": False, "block_name": True})
use_buffer_name = tvm.testing.parameter(by_dict={"buffer_index": False, "buffer_name": True})

def test_set_scope(use_block_name, use_buffer_name):
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    s.set_scope('B' if use_block_name else s.get_block("B"), 'B' if use_buffer_name else 0, "shared")
    assert_structural_equal_ignore_global_symbol(element_wise_set_scope, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


def test_set_scope_fail_on_output_buffer(use_block_name, use_buffer_name):
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.tir.ScheduleError):
        s.set_scope('C' if use_block_name else s.get_block("C"), 'C' if use_buffer_name else 0, "shared")


def test_set_scope_fail_on_index_out_of_bound():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.tir.ScheduleError):
        s.set_scope(s.get_block("B"), 1, "shared")
    with pytest.raises(tvm.tir.ScheduleError):
        s.set_scope(s.get_block("B"), -1, "shared")


def test_set_scope_fail_on_invalid_scope():
    func = element_wise
    s = tir.Schedule(func, debug_mask='all')
    with pytest.raises(tvm.tir.ScheduleError):
        s.set_scope(s.get_block("B"), 0, "test_scope")


def test_set_scope_subregion():
    func = element_wise_subregion_match
    s = tir.Schedule(func, debug_mask='all')
    s.set_scope(s.get_block("B"), 0, "shared")
    assert_structural_equal_ignore_global_symbol(element_wise_subregion_match_set_scope, s.mod["main"])
    verify_trace_roundtrip(sch=s, mod=func)


if __name__ == "__main__":
    tvm.testing.main()
