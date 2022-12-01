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
"""Intrinsics for Hexagon tensorization."""
from tvm.script import tir as T
from .. import TensorIntrin


def generate_dot_product_32x4_u8u8i32(mem_scope="global"):
    @T.prim_func
    def dot_product_32x4_u8u8i32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

    @T.prim_func
    def dot_product_32x4_u8u8i32_vrmpy(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "uint8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])

            A_u8x4 = A.vload([0], "uint8x4")
            A_i32 = T.reinterpret(A_u8x4, dtype="int32")

            B_i8x128 = B.vload([0, 0], dtype="uint8x128")
            B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpyub.acc.128B"),
                T.uint32(3),
                C[T.ramp(T.int32(0), 1, 32)],
                B_i32x32,
                A_i32,
                dtype="int32x32",
            )

    return dot_product_32x4_u8u8i32_desc, dot_product_32x4_u8u8i32_vrmpy


def generate_dot_product_32x4_u8i8i32(mem_scope="global"):
    @T.prim_func
    def dot_product_32x4_u8i8i32_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "int8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])
            for i in T.serial(0, 32):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vk = T.axis.remap("SR", [i, k])
                        C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")

    @T.prim_func
    def dot_product_32x4_u8i8i32_vrmpy(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, (4,), "uint8", offset_factor=1, scope=mem_scope)
        B = T.match_buffer(b, (32, 4), "int8", offset_factor=1, scope=mem_scope)
        C = T.match_buffer(c, (32,), "int32", offset_factor=1, scope=mem_scope)
        with T.block("root"):
            T.reads(C[0:32], A[0:4], B[0:32, 0:4])
            T.writes(C[0:32])

            A_u8x4 = A.vload([0], "uint8x4")
            A_i32 = T.reinterpret(A_u8x4, dtype="int32")

            B_i8x128 = B.vload([0, 0], dtype="int8x128")
            B_i32x32 = T.reinterpret(B_i8x128, dtype="int32x32")

            C[T.ramp(T.int32(0), 1, 32)] = T.call_llvm_pure_intrin(
                T.llvm_lookup_intrinsic_id("llvm.hexagon.V6.vrmpybusv.acc.128B"),
                T.uint32(3),
                C[T.ramp(T.int32(0), 1, 32)],
                T.broadcast(A_i32, 32),
                B_i32x32,
                dtype="int32x32",
            )

    return dot_product_32x4_u8i8i32_desc, dot_product_32x4_u8i8i32_vrmpy


VRMPY_u8u8i32_INTRIN = "dot_32x4_u8u8i32_vrmpy"

TensorIntrin.register(VRMPY_u8u8i32_INTRIN, *generate_dot_product_32x4_u8u8i32())

VRMPY_u8i8i32_INTRIN = "dot_32x4_u8i8i32_vrmpy"

TensorIntrin.register(VRMPY_u8i8i32_INTRIN, *generate_dot_product_32x4_u8i8i32())

VRMPY_u8u8i32_VTCM_INTRIN = "dot_32x4_u8u8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8u8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8u8i32("global.vtcm"))

VRMPY_u8i8i32_VTCM_INTRIN = "dot_32x4_u8i8i32_vtcm_vrmpy"
TensorIntrin.register(VRMPY_u8i8i32_VTCM_INTRIN, *generate_dot_product_32x4_u8i8i32("global.vtcm"))
