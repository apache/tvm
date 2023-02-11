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
from tvm import tir, relax
from tvm.ir import assert_structural_equal

import tvm.script
from tvm.script import tir as T, relax as R


@tvm.script.ir_module
class Before:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        m = T.var("int64")
        n = T.var("int64")
        k = T.var("int64")
        A = T.match_buffer(x, (m, n))
        B = T.match_buffer(y, (n, k))
        C = T.match_buffer(z, (m, k))

        for i, j, k in T.grid(m, k, n):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")) -> R.Tensor:
        m, n, k = T.var("int64"), T.var("int64"), T.var("int64")
        gv0 = R.call_tir("tir_matmul", (x, w), R.Tensor((m, k), dtype="float32"))
        return gv0


def test_basic():
    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
            T.func_attr({"global_symbol": "tir_matmul"})
            m = T.var("int64")
            n = T.var("int64")
            k = T.var("int64")
            A = T.match_buffer(x, (m, n))
            B = T.match_buffer(y, (n, k))
            C = T.match_buffer(z, (m, k))

            for i, j, k in T.grid(m, k, n):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: R.Tensor(("m", "n"), "float32"), w: R.Tensor(("n", "k"), "float32")
        ) -> R.Tensor:
            R.func_attr({"global_symbol": "main"})
            m, n, k = T.var("int64"), T.var("int64"), T.var("int64")
            gv0 = R.call_tir("tir_matmul", (x, w), R.Tensor((m, k), dtype="float32"))
            return gv0

    before = Before
    expected = Expected
    after = relax.transform.AttachGlobalSymbol()(before)
    assert_structural_equal(after, expected)


if __name__ == "__main__":
    pytest.main([__file__])
