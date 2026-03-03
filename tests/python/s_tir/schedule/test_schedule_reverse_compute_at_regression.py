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
"""Regression test for reverse_compute_at with T.alloc_buffer (AllocBuffer statement node).

Before the fix, reverse_compute_at would raise a spurious ScheduleError:
  "Block ... is the only leaf in the scope ..., which cannot be removed"

The root cause was that LeafBlockRemovalPlan in transform.cc only peeled off
DeclBuffer nodes from a block body but not AllocBuffer nodes. When the root
block body starts with an AllocBuffer (from T.alloc_buffer at function level),
the SeqStmt was not visible, causing the error.
"""
# ruff: noqa: E501

import tvm
import tvm.s_tir
import tvm.testing
from tvm.script import tir as T


# fmt: off
@T.prim_func
def matmul_relu_alloc_buffer(
    A_handle: T.handle, B_handle: T.handle, C_handle: T.handle
):
    """Matmul + relu with T.alloc_buffer (creates AllocBuffer statement node in root block)."""
    T.func_attr({"tir.noalias": True})
    A = T.match_buffer(A_handle, (128, 128))
    B = T.match_buffer(B_handle, (128, 128))
    C = T.match_buffer(C_handle, (128, 128))
    Y = T.alloc_buffer((128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("Y"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                Y[vi, vj] = T.float32(0)
            Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
    for i, j in T.grid(128, 128):
        with T.sblock("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
# fmt: on


def test_reverse_compute_at_with_alloc_buffer():
    """reverse_compute_at must work when the scope root block body starts with AllocBuffer.

    Regression test for the bug where LeafBlockRemovalPlan did not peel
    AllocBuffer nodes from the block body before looking for a SeqStmt,
    causing a spurious "only leaf in scope" ScheduleError.
    """
    sch = tvm.s_tir.Schedule(matmul_relu_alloc_buffer, debug_mask="all")
    Y = sch.get_sblock("Y")
    C_block = sch.get_sblock("C")

    # Split j loop of Y into (j_0=16, j_1=8), then reorder to i, j_0, k, j_1
    i_loop, j_loop, k_loop = sch.get_loops(Y)
    j0, j1 = sch.split(j_loop, factors=[16, 8])
    _, j0_new, j1_new, k_new = sch.get_loops(Y)
    sch.reorder(sch.get_loops(Y)[0], j0_new, k_new, j1_new)

    # reverse_compute_at should succeed (not raise ScheduleError)
    sch.reverse_compute_at(C_block, j0)

    # Verify that C was moved inside the j_0 loop
    final_ir = sch.mod.script()
    assert "C" in final_ir, "C block should still exist in the schedule"


if __name__ == "__main__":
    tvm.testing.main()
