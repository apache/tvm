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
# pylint: disable=invalid-name,missing-function-docstring,chained-comparison
"""TIR Patterns"""
from typing import List

import tvm
from tvm.runtime import Object
import tvm._ffi

from tvm.script import tir as T


@tvm._ffi.register_object("relax.MatchResult")
class MatchResult(Object):
    """The match result of a TIR pattern."""

    def __init__(self, pattern, symbol_values, matched_buffers):
        self.__init_handle_by_constructor__(
            tvm._ffi.MatchResult, pattern, symbol_values, matched_buffers
        )


@T.prim_func
def matmul_rrr_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_matmul: T.handle,
    M: T.int64,
    N: T.int64,
    K: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [M, K], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [K, N], dtype="float16")
    matmul = T.match_buffer(var_matmul, [M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(M, N, K):
        with T.block("matmul"):
            i0_1, i1_1, k = T.axis.remap("SSR", [i0, i1, i2])
            T.reads(rxplaceholder[i0_1, k], rxplaceholder_1[k, i1_1])
            T.writes(matmul[i0_1, i1_1])
            with T.init():
                matmul[i0_1, i1_1] = T.float16(0)
            matmul[i0_1, i1_1] = (
                matmul[i0_1, i1_1] + rxplaceholder[i0_1, k] * rxplaceholder_1[k, i1_1]
            )


@T.prim_func
def bias_row_2d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_T_add: T.handle,
    M: T.int64,
    N: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [M, N], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [T.int64(1), N], dtype="float16")
    T_add = T.match_buffer(var_T_add, [M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1 in T.grid(M, N):
        with T.block("T_add"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[T.int64(0), ax1])
            T.writes(T_add[ax0, ax1])
            T_add[ax0, ax1] = rxplaceholder[ax0, ax1] + rxplaceholder_1[T.int64(0), ax1]


@T.prim_func
def bias_row_1d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_T_add: T.handle,
    M: T.int64,
    N: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [M, N], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [N], dtype="float16")
    T_add = T.match_buffer(var_T_add, [M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1 in T.grid(M, N):
        with T.block("T_add"):
            ax0, ax1 = T.axis.remap("SS", [i0, i1])
            T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax1])
            T.writes(T_add[ax0, ax1])
            T_add[ax0, ax1] = rxplaceholder[ax0, ax1] + rxplaceholder_1[ax1]


@T.prim_func
def batch_bias_row_2d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_T_add: T.handle,
    batch: T.int64,
    M: T.int64,
    N: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [batch, M, N], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [T.int64(1), N], dtype="float16")
    T_add = T.match_buffer(var_T_add, [batch, M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(batch, M, N):
        with T.block("T_add"):
            ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(rxplaceholder[ax0, ax1, ax2], rxplaceholder_1[T.int64(0), ax2])
            T.writes(T_add[ax0, ax1, ax2])
            T_add[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2] + rxplaceholder_1[T.int64(0), ax2]


@T.prim_func
def batch_bias_row_1d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_T_add: T.handle,
    batch: T.int64,
    M: T.int64,
    N: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [batch, M, N], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [N], dtype="float16")
    T_add = T.match_buffer(var_T_add, [batch, M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1, i2 in T.grid(batch, M, N):
        with T.block("T_add"):
            ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(rxplaceholder[ax0, ax1, ax2], rxplaceholder_1[ax2])
            T.writes(T_add[ax0, ax1, ax2])
            T_add[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2] + rxplaceholder_1[ax2]


@T.prim_func
def relu_fp16(var_rxplaceholder: T.handle, var_compute: T.handle, M: T.int64, N: T.int64) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [M, N], dtype="float16")
    compute = T.match_buffer(var_compute, [M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1 in T.grid(M, N):
        with T.block("compute"):
            i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
            T.reads(rxplaceholder[i0_1, i1_1])
            T.writes(compute[i0_1, i1_1])
            compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float16(0))


@T.prim_func
def batch_matmul_rrr_2d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_matmul: T.handle,
    batch: T.int64,
    M: T.int64,
    N: T.int64,
    K: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [batch, M, K], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [K, N], dtype="float16")
    matmul = T.match_buffer(var_matmul, [batch, M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1, i2, i3 in T.grid(batch, M, N, K):
        with T.block("matmul"):
            i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
            T.reads(rxplaceholder[i0_1, i1_1, k], rxplaceholder_1[k, i2_1])
            T.writes(matmul[i0_1, i1_1, i2_1])
            with T.init():
                matmul[i0_1, i1_1, i2_1] = T.float16(0)
            matmul[i0_1, i1_1, i2_1] = (
                matmul[i0_1, i1_1, i2_1] + rxplaceholder[i0_1, i1_1, k] * rxplaceholder_1[k, i2_1]
            )


@T.prim_func
def batch_matmul_rrr_3d_fp16(
    var_rxplaceholder: T.handle,
    var_rxplaceholder_1: T.handle,
    var_matmul: T.handle,
    batch: T.int64,
    M: T.int64,
    N: T.int64,
    K: T.int64,
) -> None:
    # function attr dict
    T.func_attr({"tir.noalias": True})
    rxplaceholder = T.match_buffer(var_rxplaceholder, [batch, M, K], dtype="float16")
    rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [batch, K, N], dtype="float16")
    matmul = T.match_buffer(var_matmul, [batch, M, N], dtype="float16")
    # body
    # with T.block("root")
    for i0, i1, i2, i3 in T.grid(batch, M, N, K):
        with T.block("matmul"):
            i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
            T.reads(rxplaceholder[i0_1, i1_1, k], rxplaceholder_1[i0_1, k, i2_1])
            T.writes(matmul[i0_1, i1_1, i2_1])
            with T.init():
                matmul[i0_1, i1_1, i2_1] = T.float16(0)
            matmul[i0_1, i1_1, i2_1] = (
                matmul[i0_1, i1_1, i2_1]
                + rxplaceholder[i0_1, i1_1, k] * rxplaceholder_1[i0_1, k, i2_1]
            )


@T.prim_func
def copy_4d_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    N: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
) -> None:
    A = T.match_buffer(A_handle, [N, H, W, C], dtype="float16")
    B = T.match_buffer(B_handle, [N, H, W, C], dtype="float16")
    # body
    # with T.block("root")
    for n, h, w, c in T.grid(N, H, W, C):
        with T.block("copy"):
            vn, vh, vw, vc = T.axis.remap("SSSS", [n, h, w, c])
            T.reads(A[vn, vh, vw, vc])
            T.writes(B[vn, vh, vw, vc])
            B[vn, vh, vw, vc] = A[vn, vh, vw, vc]


@T.prim_func
def padding_2d_nhwc_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    N: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
    pH: T.int64,
    pW: T.int64,
    lH: T.int64,
    lW: T.int64,
    rH: T.int64,
    rW: T.int64,
) -> None:
    A = T.match_buffer(A_handle, [N, H, W, C], dtype="float16")
    B = T.match_buffer(B_handle, [N, pH, pW, C], dtype="float16")
    # body
    # with T.block("root")
    for v, v_1, v_2, v_3 in T.grid(N, pH, pW, C):
        with T.block("copy"):
            v_4, v_5, v_6, v_7 = T.axis.remap("SSSS", [v, v_1, v_2, v_3])
            T.reads(A[v_4, v_5 - lH, v_6 - lW, v_7])
            T.writes(B[v_4, v_5, v_6, v_7])
            B[v_4, v_5, v_6, v_7] = T.if_then_else(
                lH <= v_5 and v_5 < rH and lW <= v_6 and v_6 < rW,
                A[v_4, v_5 - lH, v_6 - lW, v_7],
                T.float16(0),
                dtype="float16",
            )


@T.prim_func
def conv2d_nhwc_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    out_handle: T.handle,
    N: T.int64,
    pH: T.int64,
    pW: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
    O: T.int64,
    KH: T.int64,
    KW: T.int64,
    StrideH: T.int64,
    StrideW: T.int64,
    DilateH: T.int64,
    DilateW: T.int64,
) -> None:
    A = T.match_buffer(A_handle, [N, pH, pW, C], dtype="float16")
    B = T.match_buffer(B_handle, [O, KH, KW, C], dtype="float16")
    out = T.match_buffer(out_handle, [N, H, W, O], dtype="float16")
    # body
    # with T.block("root")
    for v, v_1, v_2, v_3, v_4, v_5, v_6 in T.grid(N, H, W, O, KH, KW, C):
        with T.block("conv"):
            v_7, v_8, v_9, v_10, v_11, v_12, v_13 = T.axis.remap(
                "SSSSRRR", [v, v_1, v_2, v_3, v_4, v_5, v_6]
            )
            T.reads(
                A[v_7, v_11 * DilateH + v_8 * StrideH, v_12 * DilateW + v_9 * StrideW, v_13],
                B[v_10, v_11, v_12, v_13],
            )
            T.writes(out[v_7, v_8, v_9, v_10])
            with T.init():
                out[v_7, v_8, v_9, v_10] = T.float16(0)
            out[v_7, v_8, v_9, v_10] = (
                out[v_7, v_8, v_9, v_10]
                + A[v_7, v_11 * DilateH + v_8 * StrideH, v_12 * DilateW + v_9 * StrideW, v_13]
                * B[v_10, v_11, v_12, v_13]
            )


@T.prim_func
def bias_add_nhwc_2d_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    out_handle: T.handle,
    N: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
):
    A = T.match_buffer(A_handle, [N, H, W, C], dtype="float16")
    B = T.match_buffer(B_handle, [1, 1, 1, C], dtype="float16")
    out = T.match_buffer(out_handle, [N, H, W, C], dtype="float16")
    for ax0, ax1, ax2, ax3 in T.grid(N, H, W, C):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, T.int64(0), T.int64(0), v_ax3])
            T.writes(out[v_ax0, v_ax1, v_ax2, v_ax3])
            out[v_ax0, v_ax1, v_ax2, v_ax3] = (
                A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, T.int64(0), T.int64(0), v_ax3]
            )


@T.prim_func
def bias_add_nhwc_1d_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    out_handle: T.handle,
    N: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
):
    A = T.match_buffer(A_handle, [N, H, W, C], dtype="float16")
    B = T.match_buffer(B_handle, [1, 1, 1, C], dtype="float16")
    out = T.match_buffer(out_handle, [N, H, W, C], dtype="float16")
    for ax0, ax1, ax2, ax3 in T.grid(N, H, W, C):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[T.int64(0), T.int64(0), T.int64(0), v_ax3])
            T.writes(out[v_ax0, v_ax1, v_ax2, v_ax3])
            out[v_ax0, v_ax1, v_ax2, v_ax3] = (
                A[v_ax0, v_ax1, v_ax2, v_ax3] + B[T.int64(0), T.int64(0), T.int64(0), v_ax3]
            )


@T.prim_func
def elem_add_2d_fp16(
    in0_handle: T.handle,
    in1_handle: T.handle,
    out_handle: T.handle,
    N: T.int64,
    M: T.int64,
):
    in0 = T.match_buffer(in0_handle, [N, M], dtype="float16")
    in1 = T.match_buffer(in1_handle, [N, M], dtype="float16")
    out = T.match_buffer(out_handle, [N, M], dtype="float16")
    for ax0, ax1 in T.grid(N, M):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(in0[v_ax0, v_ax1], in1[v_ax0, v_ax1])
            T.writes(out[v_ax0, v_ax1])
            out[v_ax0, v_ax1] = in0[v_ax0, v_ax1] + in1[v_ax0, v_ax1]


@T.prim_func
def elem_add_3d_fp16(
    in0_handle: T.handle,
    in1_handle: T.handle,
    out_handle: T.handle,
    B: T.int64,
    N: T.int64,
    M: T.int64,
):
    in0 = T.match_buffer(in0_handle, [B, N, M], dtype="float16")
    in1 = T.match_buffer(in1_handle, [B, N, M], dtype="float16")
    out = T.match_buffer(out_handle, [B, N, M], dtype="float16")
    for ax0, ax1, ax2 in T.grid(B, N, M):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(in0[v_ax0, v_ax1, v_ax2], in1[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = in0[v_ax0, v_ax1, v_ax2] + in1[v_ax0, v_ax1, v_ax2]


@T.prim_func
def elem_add_4d_fp16(
    A_handle: T.handle,
    B_handle: T.handle,
    out_handle: T.handle,
    N: T.int64,
    H: T.int64,
    W: T.int64,
    C: T.int64,
):
    A = T.match_buffer(A_handle, [N, H, W, C], dtype="float16")
    B = T.match_buffer(B_handle, [N, H, W, C], dtype="float16")
    out = T.match_buffer(out_handle, [N, H, W, C], dtype="float16")
    for ax0, ax1, ax2, ax3 in T.grid(N, H, W, C):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(out[v_ax0, v_ax1, v_ax2, v_ax3])
            out[v_ax0, v_ax1, v_ax2, v_ax3] = (
                A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]
            )


@T.prim_func
def scalar_mul_3d_fp16(
    inp0_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
    scalar: T.float16,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float16")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float16")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_mul"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = inp0[v_ax0, v_ax1, v_ax2] * scalar


@T.prim_func
def erf_3d_fp32(
    inp0_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float32")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float32")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_erf"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = T.erf(inp0[v_ax0, v_ax1, v_ax2])


@T.prim_func
def scalar_add_3d_fp16(
    inp0_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
    scalar: T.float16,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float16")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float16")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = scalar + inp0[v_ax0, v_ax1, v_ax2]


@T.prim_func
def elem_mul_3d_fp16(
    inp0_handle: T.handle,
    inp1_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float16")
    inp1 = T.match_buffer(inp1_handle, [D1, D2, D3], dtype="float16")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float16")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_mul"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2], inp1[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = inp0[v_ax0, v_ax1, v_ax2] * inp1[v_ax0, v_ax1, v_ax2]


@T.prim_func
def cast_3d_fp16(
    inp0_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float32")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float16")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_cast"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = T.Cast("float16", inp0[v_ax0, v_ax1, v_ax2])


@T.prim_func
def cast_3d_fp32(
    inp0_handle: T.handle,
    out_handle: T.handle,
    D1: T.int64,
    D2: T.int64,
    D3: T.int64,
):
    inp0 = T.match_buffer(inp0_handle, [D1, D2, D3], dtype="float16")
    out = T.match_buffer(out_handle, [D1, D2, D3], dtype="float32")
    for ax0, ax1, ax2 in T.grid(D1, D2, D3):
        with T.block("T_cast"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inp0[v_ax0, v_ax1, v_ax2])
            T.writes(out[v_ax0, v_ax1, v_ax2])
            out[v_ax0, v_ax1, v_ax2] = T.Cast("float32", inp0[v_ax0, v_ax1, v_ax2])


def get_tir_pattern() -> List[tvm.tir.PrimFunc]:
    """Get the tir patterns for backend dispatch."""
    return [
        matmul_rrr_fp16,
        bias_row_2d_fp16,
        bias_row_1d_fp16,
        batch_bias_row_2d_fp16,
        batch_bias_row_1d_fp16,
        relu_fp16,
        erf_3d_fp32,
        batch_matmul_rrr_2d_fp16,
        batch_matmul_rrr_3d_fp16,
        copy_4d_fp16,
        padding_2d_nhwc_fp16,
        conv2d_nhwc_fp16,
        bias_add_nhwc_2d_fp16,
        bias_add_nhwc_1d_fp16,
        elem_add_2d_fp16,
        elem_add_3d_fp16,
        elem_add_4d_fp16,
        elem_mul_3d_fp16,
        scalar_add_3d_fp16,
        scalar_mul_3d_fp16,
        cast_3d_fp16,
        cast_3d_fp32,
    ]
