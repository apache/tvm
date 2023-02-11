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

from typing import List, Set, Union

import tvm
import tvm.testing
from tvm import tir
from tvm import relax as rx
from tvm.relax.analysis import has_reshape_pattern
from tvm.script import relax as R, tir as T


def test_reshape_pattern_reshape():
    @T.prim_func
    def reshape(
        rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
        T_reshape: T.Buffer[(8, 3), "float32"],
    ):
        for i0, i1 in T.grid(8, 3):
            with T.block("T_reshape"):
                ax0, ax1 = T.axis.remap("SS", [i0, i1])
                T.reads(
                    rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]
                )
                T.writes(T_reshape[ax0, ax1])
                T_reshape[ax0, ax1] = rxplaceholder[
                    (ax0 * 3 + ax1) // 24,
                    (ax0 * 3 + ax1) % 24 // 12,
                    (ax0 * 3 + ax1) % 12 // 4,
                    (ax0 * 3 + ax1) % 4,
                ]

    assert has_reshape_pattern(reshape)


def test_reshape_pattern_reshape_scheduled():
    @T.prim_func
    def reshape_scheduled(
        rxplaceholder: T.Buffer[(1, 2, 3, 4), "float32"],
        T_reshape: T.Buffer[(8, 3), "float32"],
    ):
        for i0_i1_fused_0 in T.thread_binding(1, thread="blockIdx.x"):
            for i0_i1_fused_1 in T.thread_binding(24, thread="threadIdx.x"):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(8, (i0_i1_fused_0 * 24 + i0_i1_fused_1) // 3)
                    ax1 = T.axis.spatial(3, (i0_i1_fused_0 * 24 + i0_i1_fused_1) % 3)
                    T.reads(
                        rxplaceholder[
                            (ax0 * 3 + ax1) // 24,
                            (ax0 * 3 + ax1) % 24 // 12,
                            (ax0 * 3 + ax1) % 12 // 4,
                            (ax0 * 3 + ax1) % 4,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * 3 + ax1) // 24,
                        (ax0 * 3 + ax1) % 24 // 12,
                        (ax0 * 3 + ax1) % 12 // 4,
                        (ax0 * 3 + ax1) % 4,
                    ]

    assert has_reshape_pattern(reshape_scheduled)


def test_reshape_pattern_expand_dims():
    @T.prim_func
    def expand_dims(
        rxplaceholder: T.Buffer[(2, 3, 4), "float32"],
        expand_dims: T.Buffer[(2, 1, 1, 1, 3, 1, 4, 1), "float32"],
    ):
        T.func_attr({"tir.noalias": True})
        for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(2, 1, 1, 1, 3, 1, 4, 1):
            with T.block("expand_dims"):
                i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = T.axis.remap(
                    "SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7]
                )
                T.reads(rxplaceholder[i0_1, i4_1, i6_1])
                T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[
                    i0_1, i4_1, i6_1
                ]

    assert has_reshape_pattern(expand_dims)


def test_reshape_pattern_with_raggedness():
    @T.prim_func
    def reshape_raggedness(
        A: T.Buffer[(100, 768), "float32"],
        src_indptr: T.Buffer[(9,), "int32"],
        B: T.Buffer[(100, 12, 64), "float32"],
    ):
        for b in T.serial(8):
            with T.block("block0"):
                vb = T.axis.spatial(8, b)
                for i in T.serial(src_indptr[vb + 1] - src_indptr[vb]):
                    for h in T.serial(12):
                        for f in T.serial(64):
                            with T.block("block1"):
                                vi, vh, vf = T.axis.remap("SSS", [i, h, f])
                                B[src_indptr[vb] + vi, vh, vf] = A[
                                    src_indptr[vb] + vi, vh * 64 + vf
                                ]

    assert has_reshape_pattern(reshape_raggedness)


def test_reshape_pattern_reject_seqstmt():
    @T.prim_func
    def identity_bias(A: T.Buffer[(4, 4), "float32"], B: T.Buffer[(4, 4), "float32"]):
        C = T.alloc_buffer((128, 128), "float32")
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                C[vi0, vi1] = A[vi0, vi1]
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                B[vi0, vi1] = C[vi0, vi1] + T.float32(1)

    @T.prim_func
    def identity_identity(A: T.Buffer[(4, 4), "float32"], B: T.Buffer[(4, 4), "float32"]):
        C = T.alloc_buffer((128, 128), "float32")
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                C[vi0, vi1] = A[vi0, vi1]
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SS", [i0, i1])
                B[vi0, vi1] = C[vi0, vi1]

    assert not has_reshape_pattern(identity_bias)
    assert not has_reshape_pattern(identity_identity)


def test_reshape_pattern_reject_reduction():
    @T.prim_func
    def reduction(A: T.Buffer[(4, 4), "float32"], B: T.Buffer[(4,), "float32"]):
        for i0, i1 in T.grid(4, 4):
            with T.block("identity"):
                vi0, vi1 = T.axis.remap("SR", [i0, i1])
                with T.init():
                    B[vi0] = T.float32(0)
                B[vi0] = B[vi0] + A[vi0, vi1]

    assert not has_reshape_pattern(reduction)


if __name__ == "__main__":
    tvm.testing.main()
