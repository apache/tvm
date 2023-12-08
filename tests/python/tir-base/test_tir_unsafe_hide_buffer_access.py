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


@T.prim_func
def indirect_mem_access(a: T.handle, idx_a: T.handle, b: T.handle, idx_b: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    IA = T.match_buffer(idx_a, [10], dtype="int32")
    B = T.match_buffer(b, [128], dtype="float32")
    IB = T.match_buffer(idx_b, [10], dtype="int32")

    for i in range(10):
        with T.block("B"):
            vi = T.axis.spatial(10, i)
            T.reads(A[IA[vi]], IA[vi])
            T.writes(B[IB[vi]], IB[vi])
            B[IB[vi]] = A[IA[vi]]


@T.prim_func
def indirect_mem_access_hide_ia(a: T.handle, idx_a: T.handle, b: T.handle, idx_b: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    IA = T.match_buffer(idx_a, [10], dtype="int32")
    B = T.match_buffer(b, [128], dtype="float32")
    IB = T.match_buffer(idx_b, [10], dtype="int32")

    for i in range(10):
        with T.block("B"):
            vi = T.axis.spatial(10, i)
            T.reads(A[IA[vi]])
            T.writes(B[IB[vi]], IB[vi])
            B[IB[vi]] = A[IA[vi]]


@T.prim_func
def indirect_mem_access_hide_ib(a: T.handle, idx_a: T.handle, b: T.handle, idx_b: T.handle) -> None:
    A = T.match_buffer(a, [128], dtype="float32")
    IA = T.match_buffer(idx_a, [10], dtype="int32")
    B = T.match_buffer(b, [128], dtype="float32")
    IB = T.match_buffer(idx_b, [10], dtype="int32")

    for i in range(10):
        with T.block("B"):
            vi = T.axis.spatial(10, i)
            T.reads(A[IA[vi]], IA[vi])
            T.writes(B[IB[vi]])
            B[IB[vi]] = A[IA[vi]]


def test_hide_buffer_access_read():
    sch = tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_block("B")
    sch.unsafe_hide_buffer_access(block_b, "read", [1])
    assert_structural_equal_ignore_global_symbol(indirect_mem_access_hide_ia, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=indirect_mem_access)


def test_hide_buffer_access_write():
    sch = tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_block("B")
    sch.unsafe_hide_buffer_access(block_b, "write", [1])
    assert_structural_equal_ignore_global_symbol(indirect_mem_access_hide_ib, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=indirect_mem_access)


def test_hide_buffer_access_fail_buffer_type():
    sch = tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.error.TVMError):
        sch.unsafe_hide_buffer_access(block_b, "opaque", [0])


def test_hide_buffer_access_fail_buffer_index():
    sch = tir.Schedule(indirect_mem_access, debug_mask="all")
    block_b = sch.get_block("B")
    with pytest.raises(tvm.error.TVMError):
        sch.unsafe_hide_buffer_access(block_b, "read", [2])


if __name__ == "__main__":
    tvm.testing.main()
