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

import tvm
from tvm.script import tir as T


def test_meta_programming_matmul():
    def matmul_generator(M: int, N: int, K: int, dtype: str):
        @T.prim_func
        def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [N, K], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block():
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        return matmul

    @T.prim_func
    def matmul_128_128_128_fp16(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128], dtype="float16")
        B = T.match_buffer(b, [128, 128], dtype="float16")
        C = T.match_buffer(c, [128, 128], dtype="float16")

        for i, j, k in T.grid(128, 128, 128):
            with T.block():
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    f = matmul_generator(128, 128, 128, "float16").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f, matmul_128_128_128_fp16.with_attr("global_symbol", "main"))


def test_meta_programming_uncaptured_var():
    def generate_erf(dtype):
        @T.prim_func
        def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
            for i in range(1):
                with T.block("C"):
                    C[i] = T.erf(A[i])

        return main

    @T.prim_func
    def fp32(A: T.Buffer((1,), "float32"), C: T.Buffer((1,), "float32")):
        for i in range(1):
            with T.block("C"):
                C[i] = T.erf(A[i])

    @T.prim_func
    def fp16(A: T.Buffer((1,), "float16"), C: T.Buffer((1,), "float16")):
        for i in range(1):
            with T.block("C"):
                C[i] = T.erf(A[i])

    tvm.ir.assert_structural_equal(fp16.with_attr("global_symbol", "main"), generate_erf("float16"))
    tvm.ir.assert_structural_equal(fp32.with_attr("global_symbol", "main"), generate_erf("float32"))


if __name__ == "__main__":
    test_meta_programming_matmul()
    test_meta_programming_uncaptured_var()
