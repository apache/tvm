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
"""Unittests for tvm.script.parser.tir"""

import tvm.testing
from tvm.script.parser import tir as T
from tvm import ir, tir


def test_tir_buffer_proxy():
    buffer_0 = T.Buffer((128, 128), "float32")
    assert (
        isinstance(buffer_0, tir.Buffer)
        and list(buffer_0.shape) == [128, 128]
        and buffer_0.dtype == "float32"
    )

    buffer_1 = T.Buffer((64, 64, 64), "int32")
    assert (
        isinstance(buffer_1, tir.Buffer)
        and list(buffer_1.shape) == [64, 64, 64]
        and buffer_1.dtype == "int32"
    )


def test_tir_ptr_proxy():
    ptr_0 = T.handle("int32", "global")
    assert (
        isinstance(ptr_0, tir.Var)
        and ptr_0.dtype == "handle"
        and isinstance(ptr_0.type_annotation, ir.PointerType)
        and ptr_0.type_annotation.element_type == ir.PrimType("int32")
        and ptr_0.type_annotation.storage_scope == "global"
    )

    ptr_1 = T.handle("float32", "shared")
    assert (
        isinstance(ptr_1, tir.Var)
        and ptr_1.dtype == "handle"
        and isinstance(ptr_1.type_annotation, ir.PointerType)
        and ptr_1.type_annotation.element_type == ir.PrimType("float32")
        and ptr_1.type_annotation.storage_scope == "shared"
    )


def test_tir_func_name():
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    assert matmul.__name__ == "matmul"


def test_tir_macro():
    @T.macro
    def assign(i, *args, t1, **kwargs):
        vi, vj, vk = T.axis.remap("SSR", [i, args[0], args[1]])
        kwargs["t3"][vi, vj] = kwargs["t3"][vi, vj] + t1[vi, vk] * kwargs["t2"][vj, vk]

    @T.prim_func
    def matmul_w_macro(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                T.insert(assign, i, j, k, t1=A, t2=B, t3=C)

    @T.prim_func
    def matmul_no_macro(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128])
        B = T.match_buffer(b, [128, 128])
        C = T.match_buffer(c, [128, 128])
        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    tvm.ir.assert_structural_equal(matmul_no_macro, matmul_w_macro)


if __name__ == "__main__":
    tvm.testing.main()
