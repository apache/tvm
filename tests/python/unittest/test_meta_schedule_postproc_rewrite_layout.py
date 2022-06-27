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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import tvm
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import RewriteLayout
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> TuneContext:
    return TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            RewriteLayout(),
        ],
        task_name="test",
    )


@T.prim_func
def tir_matmul(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"layout_free_buffers": [1]})
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with T.block("matmul"):
            vi = T.axis.S(16, i0 * 4 + i1)
            vj = T.axis.S(16, j)
            vk = T.axis.R(16, k0 * 4 + k1)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def rewritten_tir_matmul(
    A: T.Buffer[(16, 16), "float32"],
    B: T.Buffer[(16, 16), "float32"],
    C: T.Buffer[(16, 16), "float32"],
) -> None:
    T.func_attr({"layout_free_buffers": [1]})
    B_reindex = T.alloc_buffer([16, 4, 4], dtype="float32")
    for ax0, ax1 in T.grid(16, 16):
        with T.block("layout_rewrite"):
            i0, i1 = T.axis.remap("SS", [ax0, ax1])
            T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
            B_reindex[i1, i0 // 4, i0 % 4] = B[i0, i1]
    for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
        with T.block("matmul"):
            vi = T.axis.spatial(16, i0 * 4 + i1)
            vj = T.axis.spatial(16, j)
            vk = T.axis.reduce(16, k0 * 4 + k1)
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B_reindex[vj, vk // 4, vk % 4]


def test_layout_rewrite():
    target = _target()
    ctx = _create_context(tir_matmul, target)
    sch = tvm.tir.Schedule(tir_matmul, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod["main"], rewritten_tir_matmul)


if __name__ == "__main__":
    test_layout_rewrite()
