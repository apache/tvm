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

# pylint: disable=no-member,invalid-name,unused-variable

########## Function before schedule ##########


@T.prim_func
def resize(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (1, 3, 40, 40))
    B = T.match_buffer(b, (1, 3, 80, 80))
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with T.block("A"):
            n, c, vi, vj = T.axis.remap("SSSS", [i0, i1, i2, i3])
            B[n, c, vi, vj] = A[n, c, vi // 4 + vj // 4, vj // 2]


@T.prim_func
def resize_cache_index(
    A: T.Buffer[(1, 3, 40, 40), "float32"], B: T.Buffer[(1, 3, 80, 80), "float32"]
) -> None:
    index_var_0 = T.alloc_buffer([80, 80], dtype="int32", strides=[1])
    index_var_1 = T.alloc_buffer([80], dtype="int32", strides=[1])
    for ax0, ax1 in T.grid(80, 80):
        with T.block("index_0"):
            v0 = T.axis.spatial(80, ax0)
            v1 = T.axis.spatial(80, ax1)
            T.reads()
            T.writes(index_var_0[v0, v1])
            index_var_0[v0, v1] = v0 // 4 + v1 // 4
    for ax0 in T.serial(80):
        with T.block("index_1"):
            v0 = T.axis.spatial(80, ax0)
            T.reads()
            T.writes(index_var_1[v0])
            index_var_1[v0] = v0 // 2
    for i0, i1, i2, i3 in T.grid(1, 3, 80, 80):
        with T.block("A"):
            n, c, vi, vj = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[n, c, vi // 4 + vj // 4, vj // 2])
            T.writes(B[n, c, vi, vj])
            B[n, c, vi, vj] = A[n, c, index_var_0[vi, vj], index_var_1[vj]]


def test_inplace_cache_read():
    sch = tvm.tir.Schedule(resize, debug_mask="all")
    block = sch.get_block("A")
    sch.cache_index(block, 0)
    tvm.ir.assert_structural_equal(resize_cache_index, sch.mod["main"])
    verify_trace_roundtrip(sch=sch, mod=resize)


if __name__ == "__main__":
    tvm.testing.main()
