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

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip


@T.prim_func
def matmul(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_after_reorder_block_iter_var(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vk, vj, vi = T.axis.remap("RSS", [k, j, i])
            T.reads(A[vi, vk], B[vj, vk])
            T.writes(C[vi, vj])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reorder_block_iter_var():
    sch = tir.Schedule(matmul, debug_mask="all")
    C = sch.get_block("C")
    sch.reorder_block_iter_var(C, [2, 1, 0])
    tvm.ir.assert_structural_equal(
        matmul_after_reorder_block_iter_var.with_attr("global_symbol", "matmul"), sch.mod["main"]
    )
    verify_trace_roundtrip(sch=sch, mod=matmul)


def test_reorder_block_iter_var_fail_not_full():
    sch = tir.Schedule(matmul, debug_mask="all")
    C = sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder_block_iter_var(C, [2, 1])


def test_reorder_block_iter_var_fail_not_within_bound():
    sch = tir.Schedule(matmul, debug_mask="all")
    C = sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder_block_iter_var(C, [-1, 3, 2])


def test_reorder_block_iter_var_fail_not_unique():
    sch = tir.Schedule(matmul, debug_mask="all")
    C = sch.get_block("C")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.reorder_block_iter_var(C, [0, 0, 2])


if __name__ == "__main__":
    tvm.testing.main()
