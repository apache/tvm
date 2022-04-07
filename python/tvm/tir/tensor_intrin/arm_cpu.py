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
from .. import TensorIntrin
from tvm.script import tir as T


@T.prim_func
def dot_product_4x4_i8i8i32_desc(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])
        for i in T.serial(0, 4):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")


@T.prim_func
def dot_product_4x4_i8i8i32_neon(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_int8 = A.vload([0], "int8x4")
        re_int32 = T.reinterpret(A_int8, dtype="int32")
        vec_ai32 = T.broadcast(re_int32, 2)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x8")

        vec_b = B.vload([0, 0], dtype="int8x8")

        multiply = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b,
            dtype="int16x8",
        )

        pair1 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply,
            dtype="int32x4",
        )

        vec_b_2 = B.vload([2, 0], dtype="int8x8")

        multiply_2 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_2,
            dtype="int16x8",
        )

        pair2 = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_2,
            dtype="int32x4",
        )

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.addp.v4i32"),
            T.uint32(2),
            pair1,
            pair2,
            dtype="int32x4",
        )


@T.prim_func
def dot_product_4x4_i8i8i32_sdot(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_i8x4 = A.vload([0], "int8x4")
        A_i32 = T.reinterpret(A_i8x4, dtype="int32")
        vec_ai32 = T.broadcast(A_i32, 4)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x16")

        vec_b = B.vload([0, 0], dtype="int8x16")

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            T.uint32(3),
            T.int32x4(0),
            vec_a,
            vec_b,
            dtype="int32x4",
        )


ARM_DOT_4x4_i8_NEON_INTRIN = "dot_4x4_i8i8s32_neon"
ARM_DOT_4x4_i8_SDOT_INTRIN = "dot_4x4_i8i8s32_sdot"

TensorIntrin.register(
    ARM_DOT_4x4_i8_NEON_INTRIN, dot_product_4x4_i8i8i32_desc, dot_product_4x4_i8i8i32_neon
)

TensorIntrin.register(
    ARM_DOT_4x4_i8_SDOT_INTRIN, dot_product_4x4_i8i8i32_desc, dot_product_4x4_i8i8i32_sdot
)
