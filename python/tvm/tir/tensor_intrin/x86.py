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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for x86 tensorization."""
from tvm.script import tir as T
from .. import TensorIntrin


# Tensorized intrinsic description and VNNI-specific implementation.
# Equivalent to the ones in topi/x86/tensor_intrin.py


@T.prim_func
def dot_product_16x4_u8i8i32_desc(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((16, 4), "int8", offset_factor=1),
    C: T.Buffer((16,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])
        for i in T.serial(0, 16):
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_product_16x4_u8i8i32_vnni(
    A: T.Buffer((4,), "uint8", offset_factor=1),
    B: T.Buffer((16, 4), "int8", offset_factor=1),
    C: T.Buffer((16,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:16], A[0:4], B[0:16, 0:4])
        T.writes(C[0:16])

        A_u8x4 = A.vload([0], "uint8x4")
        A_i32 = T.reinterpret(A_u8x4, dtype="int32")

        B_i8x64 = B.vload([0, 0], dtype="int8x64")
        B_i32x16 = T.reinterpret(B_i8x64, dtype="int32x16")
        C_i32x16 = C.vload([0], dtype="int32x16")

        C[T.ramp(T.int32(0), 1, 16)] = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.x86.avx512.vpdpbusd.512"),
            T.uint32(0),
            C_i32x16,
            T.broadcast(A_i32, 16),
            B_i32x16,
            dtype="int32x16",
        )


VNNI_DOT_16x4_INTRIN = "dot_16x4_vnni"

TensorIntrin.register(
    VNNI_DOT_16x4_INTRIN, dot_product_16x4_u8i8i32_desc, dot_product_16x4_u8i8i32_vnni
)
