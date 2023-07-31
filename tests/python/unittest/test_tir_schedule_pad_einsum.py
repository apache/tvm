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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    verify_trace_roundtrip,
    assert_structural_equal_ignore_global_symbol,
)

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def matmul_before(
    A: T.Buffer((128, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((128, 127), "float32"),
) -> None:
    A_shared = T.alloc_buffer((128, 127), "float32", scope="shared")
    B_shared = T.alloc_buffer((127, 127), "float32", scope="shared")
    C_shared = T.alloc_buffer((128, 127), "float32", scope="shared")
    for i0, i1 in T.grid(128, 127):
        with T.block("A"):
            i, j = T.axis.remap("SS", [i0, i1])
            A_shared[i, j] = A[i, j]
    for i0, i1 in T.grid(127, 127):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0, i1])
            B_shared[i, j] = B[i, j]
    for i0, i1, i2 in T.grid(128, 127, 127):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            with T.init():
                C_shared[i, j] = T.float32(0)
            C_shared[i, j] = C_shared[i, j] + A_shared[i, k] * B_shared[k, j]
    for i0, i1 in T.grid(128, 127):
        with T.block("C"):
            i, j = T.axis.remap("SS", [i0, i1])
            C[i, j] = C_shared[i, j]


@T.prim_func
def matmul_expected(
    A: T.Buffer((128, 127), "float32"),
    B: T.Buffer((127, 127), "float32"),
    C: T.Buffer((128, 127), "float32"),
) -> None:
    A_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    B_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    C_shared_padded = T.alloc_buffer([128, 128], dtype="float32", scope="shared")
    for i0, i1 in T.grid(128, 128):
        with T.block("A"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(A[i, j])
            T.writes(A_shared_padded[i, j])
            A_shared_padded[i, j] = T.if_then_else(j < 127, A[i, j], T.float32(0), dtype="float32")
    for i0, i1 in T.grid(128, 128):
        with T.block("B"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(B[i, j])
            T.writes(B_shared_padded[i, j])
            B_shared_padded[i, j] = T.if_then_else(
                i < 127 and j < 127, B[i, j], T.float32(0), dtype="float32"
            )
    for i0, i1, i2 in T.grid(128, 128, 128):
        with T.block("C_shared"):
            i, j, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(A_shared_padded[i, k], B_shared_padded[k, j])
            T.writes(C_shared_padded[i, j])
            with T.init():
                C_shared_padded[i, j] = T.float32(0)
            C_shared_padded[i, j] = (
                C_shared_padded[i, j] + A_shared_padded[i, k] * B_shared_padded[k, j]
            )
    for i0, i1 in T.grid(128, 127):
        with T.block("C"):
            i, j = T.axis.remap("SS", [i0, i1])
            T.reads(C_shared_padded[i, j])
            T.writes(C[i, j])
            C[i, j] = C_shared_padded[i, j]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_pad_matmul():
    # pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg

    @T.prim_func
    def matmul_before(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:
        n = T.int32()
        A = T.match_buffer(a, (128, 128), "float32")
        B = T.match_buffer(b, (n, 128), "float32")
        C = T.match_buffer(c, (128, n), "float32")
        for i0, i1, i2 in T.grid(128, n, 128):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[j, k]

    @T.prim_func
    def matmul_after(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ):
        n = T.int32()
        A = T.match_buffer(a, (128, 128), "float32")
        B = T.match_buffer(b, (n, 128), "float32")
        C = T.match_buffer(c, (128, n), "float32")
        B_pad = T.alloc_buffer(((n + 31) // 32 * 32, 128))
        C_pad = T.alloc_buffer((128, (n + 31) // 32 * 32))
        for i0, i1 in T.grid((n + 31) // 32 * 32, 128):
            with T.block("B_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                B_pad[v0, v1] = T.if_then_else(v0 < n, B[v0, v1], T.float32(0))
        for i0, i1, i2 in T.grid(128, (n + 31) // 32 * 32, 128):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(A[i, k], B_pad[j, k])
                T.writes(C_pad[i, j])
                with T.init():
                    C_pad[i, j] = T.float32(0)
                C_pad[i, j] = C_pad[i, j] + A[i, k] * B_pad[j, k]
        for i0, i1 in T.grid(128, n):
            with T.block("C_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                C[v0, v1] = C_pad[v0, v1]

    sch = tir.Schedule(matmul_before, debug_mask="all")
    C = sch.get_block("C")
    sch.pad_einsum(C, [32, 32, 32])
    assert_structural_equal_ignore_global_symbol(matmul_after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=matmul_before)


def test_pad_matmul_2():
    @T.prim_func
    def before(
        a: T.handle,
        b: T.handle,
        m: T.handle,
        d: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(a, (1, n, 4096))
        B = T.match_buffer(b, (11008, 4096))
        M = T.match_buffer(m, (1, n, 11008))
        D = T.match_buffer(d, (1, n, 11008))
        C = T.alloc_buffer((1, n, 11008))
        for i0, i1, i2, k in T.grid(1, n, 11008, 4096):
            with T.block("C"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(C[v_i0, v_i1, v_i2])
                with T.init():
                    C[v_i0, v_i1, v_i2] = T.float32(0)
                C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(1, n, 11008):
            with T.block("D"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                D[v_ax0, v_ax1, v_ax2] = M[v_ax0, v_ax1, v_ax2] * C[v_ax0, v_ax1, v_ax2]

    @T.prim_func
    def after(a: T.handle, b: T.handle, m: T.handle, d: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(a, (1, n, 4096))
        B = T.match_buffer(b, (11008, 4096))
        M = T.match_buffer(m, (1, n, 11008))
        D = T.match_buffer(d, (1, n, 11008))
        # with T.block("root"):
        C = T.alloc_buffer((1, n, 11008))
        A_pad = T.alloc_buffer((1, (n + 31) // 32 * 32, 4096))
        C_pad = T.alloc_buffer((1, (n + 31) // 32 * 32, 11008))
        for i0, i1, i2 in T.grid(1, (n + 31) // 32 * 32, 4096):
            with T.block("A_pad"):
                v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
                A_pad[v0, v1, v2] = T.if_then_else(v1 < n, A[v0, v1, v2], T.float32(0))
        for i0, i1, i2, k in T.grid(1, (n + 31) // 32 * 32, 11008, 4096):
            with T.block("C"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(A_pad[v_i0, v_i1, v_k], B[v_i2, v_k])
                T.writes(C_pad[v_i0, v_i1, v_i2])
                with T.init():
                    C_pad[v_i0, v_i1, v_i2] = T.float32(0)
                C_pad[v_i0, v_i1, v_i2] = (
                    C_pad[v_i0, v_i1, v_i2] + A_pad[v_i0, v_i1, v_k] * B[v_i2, v_k]
                )
        for i0, i1, i2 in T.grid(1, n, 11008):
            with T.block("C_pad"):
                v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
                C[v0, v1, v2] = C_pad[v0, v1, v2]
        for ax0, ax1, ax2 in T.grid(1, n, 11008):
            with T.block("D"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                D[v_ax0, v_ax1, v_ax2] = M[v_ax0, v_ax1, v_ax2] * C[v_ax0, v_ax1, v_ax2]

    sch = tir.Schedule(before, debug_mask="all")
    C = sch.get_block("C")
    sch.pad_einsum(C, [1, 32, 32, 32])
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=before)


def test_pad_rms():
    @T.prim_func
    def before(
        a: T.handle,
        w: T.handle,
        r: T.handle,
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(a, (1, n, 4096))
        W = T.match_buffer(w, (4096,), "float32")
        R = T.match_buffer(r, (1, n, 4096), "float32")
        S = T.alloc_buffer((1, n), "float32")
        for bsz, i, k in T.grid(1, n, 4096):
            with T.block("S"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(S[v_bsz, v_i])
                with T.init():
                    S[v_bsz, v_i] = T.float32(0)
                S[v_bsz, v_i] = S[v_bsz, v_i] + A[v_bsz, v_i, v_k] * A[v_bsz, v_i, v_k]
        for bsz, i, k in T.grid(1, n, 4096):
            with T.block("R"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                R[v_bsz, v_i, v_k] = W[v_k] * (
                    A[v_bsz, v_i, v_k]
                    / T.sqrt(S[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1e-6))
                )

    @T.prim_func
    def after(a: T.handle, w: T.handle, r: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int32()
        A = T.match_buffer(a, (1, n, 4096))
        W = T.match_buffer(w, (4096,), "float32")
        R = T.match_buffer(r, (1, n, 4096))
        S = T.alloc_buffer((1, n))
        A_pad = T.alloc_buffer((1, (n + 31) // 32 * 32, 4096))
        S_pad = T.alloc_buffer((1, (n + 31) // 32 * 32))
        for i0, i1, i2 in T.grid(1, (n + 31) // 32 * 32, 4096):
            with T.block("A_pad"):
                v0, v1, v2 = T.axis.remap("SSS", [i0, i1, i2])
                A_pad[v0, v1, v2] = T.if_then_else(v1 < n, A[v0, v1, v2], T.float32(0))
        for bsz, i, k in T.grid(1, (n + 31) // 32 * 32, 4096):
            with T.block("S"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A_pad[v_bsz, v_i, v_k])
                T.writes(S_pad[v_bsz, v_i])
                with T.init():
                    S_pad[v_bsz, v_i] = T.float32(0)
                S_pad[v_bsz, v_i] = (
                    S_pad[v_bsz, v_i] + A_pad[v_bsz, v_i, v_k] * A_pad[v_bsz, v_i, v_k]
                )
        for i0, i1 in T.grid(1, n):
            with T.block("S_pad"):
                v0, v1 = T.axis.remap("SS", [i0, i1])
                S[v0, v1] = S_pad[v0, v1]
        for bsz, i, k in T.grid(1, n, 4096):
            with T.block("R"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                R[v_bsz, v_i, v_k] = W[v_k] * (
                    A[v_bsz, v_i, v_k]
                    / T.sqrt(S[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1e-6))
                )

    sch = tir.Schedule(before, debug_mask="all")
    C = sch.get_block("S")
    sch.pad_einsum(C, [1, 32, 1])
    assert_structural_equal_ignore_global_symbol(after, sch.mod["main"])
    verify_trace_roundtrip(sch, mod=before)


if __name__ == "__main__":
    test_pad_matmul()
    test_pad_matmul_2()
    test_pad_rms()
