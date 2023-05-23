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
from tvm import tir, te
from tvm.script import tir as T
from tvm.tir.schedule.schedule import ScheduleError
from tvm.tir.schedule.testing import verify_trace_roundtrip
from tvm.meta_schedule.testing import te_workload

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def matmul_before(
    A: T.Buffer((128, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((128, 127), "float32"),
) -> None:
    A_shared = T.alloc_buffer((128, 127), "float32", scope="shared")
    B_shared = T.alloc_buffer((127, 127), "float32", scope="shared")
    C_shared = T.alloc_buffer((128, 127), "float32", scope="shared")
    for i0, i1 in T.grid(128, 127):
        with T.block("A"):
            i, j = T.axis.remap("SS", [i0, i1])
            A_shared[i, j] = A[i, j]
    for i0, i1 in T.grid(127, 127):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0, i1])
            B_shared[i, j] = B[i, j]
    for i0, i1, i2 in T.grid(128, 127, 127):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C_shared[i, j] = T.float32(0)
            C_shared[i, j] = C_shared[i, j] + A_shared[i, k] * B_shared[k, j]
    for i0, i1 in T.grid(128, 127):
        with T.block("C"):
            i, j = T.axis.remap("SS", [i0, i1])
            C[i, j] = C_shared[i, j]


@T.prim_func
def matmul_expected(
    A: T.Buffer((128, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((128, 127), "float32"),
) -> None:
    A_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    B_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    C_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    for i0, i1 in T.grid(128, 128):
        with T.block("A"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(A[i, j])
            T.writes(A_shared_padded[i, j])
            A_shared_padded[i, j] = T.if_then_else(j < 127, A[i, j], T.float32(0), dtype="float32")
    for i0, i1 in T.grid(128, 128):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(B[i, j])
            T.writes(B_shared_padded[i, j])
            B_shared_padded[i, j] = T.if_then_else(
                i < 127 and j < 127, B[i, j], T.float32(0), dtype="float32"
            )
    for i0, i1, i2 in T.grid(128, 128, 128):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(A_shared_padded[i, k], B_shared_padded[k, j])
            T.writes(C_shared_padded[i, j])
            with T.init():
                C_shared_padded[i, j] = T.float32(0)
            C_shared_padded[i, j] = (
                C_shared_padded[i, j] + A_shared_padded[i, k] * B_shared_padded[k, j]
            )
    for i0, i1 in T.grid(128, 127):
        with T.block("C"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(C_shared_padded[i, j])
            T.writes(C[i, j])
            C[i, j] = C_shared_padded[i, j]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_pad_matmul():
    sch = tir.Schedule(matmul_before, debug_mask="all")
    C = sch.get_block("C_shared")
    sch.pad_einsum(C, [0, 1, 1])
    tvm.ir.assert_structural_equal(matmul_expected, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=matmul_before)


def test_pad_matmul_error_non_intermediate_buffer():
    func = te.create_prim_func(te_workload.matmul(128, 127, 127))
    sch = tir.Schedule(func, debug_mask="all")
    C = sch.get_block("C")
    with pytest.raises(ScheduleError):
        sch.pad_einsum(C, [0, 1, 1])


if __name__ == "__main__":
    tvm.testing.main()
