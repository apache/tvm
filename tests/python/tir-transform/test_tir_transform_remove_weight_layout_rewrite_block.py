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
import sys

import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
from tvm.tir.function import PrimFunc


def _check(before, expect):
    if isinstance(before, PrimFunc):
        before = IRModule({"main": before.with_attr("global_symbol", "main")})
    if isinstance(expect, PrimFunc):
        expect = IRModule({"main": expect.with_attr("global_symbol", "main")})

    mod = tvm.tir.transform.RemoveWeightLayoutRewriteBlock()(before)
    tvm.ir.assert_structural_equal(mod, expect)


def test_matmul():
    @T.prim_func
    def before(
        A: T.Buffer((16, 16), "float32"),
        B: T.Buffer((16, 16), "float32"),
        C: T.Buffer((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_buffers": [1]})
        B_ = T.alloc_buffer([16, 4, 4], dtype="float32")
        for i0_o, i1_o in T.grid(16, 16):
            with T.block("layout_rewrite"):
                i0, i1 = T.axis.remap("SS", [i0_o, i1_o])
                T.reads(B[i0, i1])
                T.writes(B_[i1, i0 // 4, i0 % 4])
                T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
                B_[i1, i0 // 4, i0 % 4] = B[i0, i1]
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.spatial(16, i0 * 4 + i1)
                vj = T.axis.spatial(16, j)
                vk = T.axis.reduce(16, k0 * 4 + k1)
                T.reads(A[vi, vk], B_[vj, vk // 4, vk % 4])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B_[vj, vk // 4, vk % 4]

    @T.prim_func
    def after(
        A: T.Buffer((16, 16), "float32"),
        B: T.Buffer((16, 4, 4), "float32"),
        C: T.Buffer((16, 16), "float32"),
    ) -> None:
        T.func_attr({"layout_free_buffers": [1]})
        for i0_o, i1_o in T.grid(16, 16):
            with T.block("layout_rewrite"):
                i0, i1 = T.axis.remap("SS", [i0_o, i1_o])
                T.reads()
                T.writes()
                T.block_attr({"meta_schedule.layout_rewrite_preproc": True})
                T.evaluate(0)
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):
            with T.block("matmul"):
                vi = T.axis.spatial(16, i0 * 4 + i1)
                vj = T.axis.spatial(16, j)
                vk = T.axis.reduce(16, k0 * 4 + k1)
                T.reads(A[vi, vk], B[vj, vk // 4, vk % 4])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk // 4, vk % 4]

    _check(before, after)


if __name__ == "__main__":
    test_matmul()
