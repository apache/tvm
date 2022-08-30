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


@T.prim_func
def dp4a_desc(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])
        for i in range(0, 4):
            with T.block("update"):
                vi = T.axis.remap("R", [i])
                C[0] = C[0] + T.cast(A[vi], "int32") * T.cast(B[vi], "int32")


@T.prim_func
def dp4a_impl(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])

        C[0] += T.call_pure_extern(
            "__dp4a", A.vload([0], "int8x4"), B.vload([0], "int8x4"), T.int32(0), dtype="int32"
        )


DP4A_INTRIN = "dp4a"

TensorIntrin.register(DP4A_INTRIN, dp4a_desc, dp4a_impl)
