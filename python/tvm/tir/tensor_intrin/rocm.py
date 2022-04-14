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
"""Intrinsics for AMDGPU tensorization."""
from tvm.script import tir as T
from .. import TensorIntrin
from .dot_product_common import dp4a_desc


@T.prim_func
def sdot4(
    A: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    B: T.Buffer((4,), "int8", offset_factor=1, align=4, scope="shared"),
    C: T.Buffer((1,), "int32", offset_factor=1, align=4, scope="local"),
) -> None:
    with T.block("root"):
        T.reads(C[0], A[0:4], B[0:4])
        T.writes(C[0])

        C[0] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.amdgcn.sdot4"),
            T.uint32(4),
            T.reinterpret(A.vload([0], "int8x4"), dtype="int32"),
            T.reinterpret(B.vload([0], "int8x4"), dtype="int32"),
            T.int32(0),
            T.bool(1),
            dtype="int32",
        )


AMDGPU_SDOT4_INTRIN = "sdot4"

TensorIntrin.register(AMDGPU_SDOT4_INTRIN, dp4a_desc, sdot4)
