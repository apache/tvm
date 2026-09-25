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
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


def test_meta_programming_matmul():
    def matmul_generator(M: int, N: int, K: int, dtype: str):
        @Ts.prim_func
        def matmul(
            A: T.Buffer([M, K], dtype=dtype),
            B: T.Buffer([N, K], dtype=dtype),
            C: T.Buffer([M, N], dtype=dtype),
        ) -> None:
            for i, j, k in T.grid(M, N, K):
                with Ts.sblock():
                    vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                    with Ts.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

        return matmul

    @Ts.prim_func
    def matmul_128_128_128_fp16(
        A: T.Buffer([128, 128], dtype="float16"),
        B: T.Buffer([128, 128], dtype="float16"),
        C: T.Buffer([128, 128], dtype="float16"),
    ) -> None:
        for i, j, k in T.grid(128, 128, 128):
            with Ts.sblock():
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                with Ts.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    f = matmul_generator(128, 128, 128, "float16").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f, matmul_128_128_128_fp16.with_attr("global_symbol", "main"))


def test_meta_programming_uncaptured_var():
    def generate_erf(dtype):
        @Ts.prim_func
        def main(A: T.Buffer((1,), dtype), C: T.Buffer((1,), dtype)):
            for i in range(1):
                with Ts.sblock("C"):
                    C[i] = T.erf(A[i])

        return main

    @Ts.prim_func
    def fp32(A: T.Buffer((1,), "float32"), C: T.Buffer((1,), "float32")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    @Ts.prim_func
    def fp16(A: T.Buffer((1,), "float16"), C: T.Buffer((1,), "float16")):
        for i in range(1):
            with Ts.sblock("C"):
                C[i] = T.erf(A[i])

    f1 = generate_erf("float32").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f1, fp32.with_attr("global_symbol", "main"))
    f2 = generate_erf("float16").with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(f2, fp16.with_attr("global_symbol", "main"))


if __name__ == "__main__":
    test_meta_programming_matmul()
    test_meta_programming_uncaptured_var()
