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
"""Dot product related intrinsics."""
from tvm.script import tir as T
from .. import TensorIntrin


def get_dp4a_intrin(dtype_a, dtype_b, dtype_c):
    if dtype_c == "uint32":
        assert dtype_a == dtype_b == "uint8"
    vec_type_a = "int8x4" if dtype_a == "int8" else "uint8x4"
    vec_type_b = "int8x4" if dtype_b == "int8" else "uint8x4"

    @T.prim_func
    def dp4a_desc(
        A: T.Buffer((4,), dtype_a, offset_factor=1, align=4, scope="shared"),
        B: T.Buffer((4,), dtype_b, offset_factor=1, align=4, scope="shared"),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4, scope="local"),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:4], B[0:4])
            T.writes(C[0])
            for i in range(0, 4):
                with T.block("update"):
                    vi = T.axis.remap("R", [i])
                    C[0] = C[0] + T.cast(A[vi], dtype_c) * T.cast(B[vi], dtype_c)

    @T.prim_func
    def dp4a_impl(
        A: T.Buffer((4,), dtype_a, offset_factor=1, align=4, scope="shared"),
        B: T.Buffer((4,), dtype_b, offset_factor=1, align=4, scope="shared"),
        C: T.Buffer((1,), dtype_c, offset_factor=1, align=4, scope="local"),
    ) -> None:
        with T.block("root"):
            T.reads(C[0], A[0:4], B[0:4])
            T.writes(C[0])

            C[0] += T.call_pure_extern(
                "__dp4a",
                A.vload([0], vec_type_a),
                B.vload([0], vec_type_b),
                T.uint32(0) if dtype_c == "uint32" else T.int32(0),
                dtype=dtype_c,
            )

    return dp4a_desc, dp4a_impl


DP4A_S8S8S32_INTRIN = "dp4a_s8s8s32"
TensorIntrin.register(DP4A_S8S8S32_INTRIN, *get_dp4a_intrin("int8", "int8", "int32"))
DP4A_U8S8S32_INTRIN = "dp4a_u8s8s32"
TensorIntrin.register(DP4A_U8S8S32_INTRIN, *get_dp4a_intrin("uint8", "int8", "int32"))
DP4A_S8U8S32_INTRIN = "dp4a_s8u8s32"
TensorIntrin.register(DP4A_S8U8S32_INTRIN, *get_dp4a_intrin("int8", "uint8", "int32"))
DP4A_U8U8U32_INTRIN = "dp4a_u8u8u32"
TensorIntrin.register(DP4A_U8U8U32_INTRIN, *get_dp4a_intrin("uint8", "uint8", "uint32"))
