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
import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip
import pytest


def check_rolling_buffer(
    sch: tir.Schedule, origin: tir.PrimFunc, expected: tir.PrimFunc, check_run=False
):
    scheduled = sch.mod["main"]
    tvm.ir.assert_structural_equal(scheduled, expected)
    verify_trace_roundtrip(sch, origin)
    if check_run:
        in_buffer = origin.buffer_map[origin.params[0]]
        out_buffer = origin.buffer_map[origin.params[1]]
        in_shape = [int(_) for _ in in_buffer.shape]
        out_shape = [int(_) for _ in out_buffer.shape]
        x = tvm.nd.array(np.random.uniform(0, 64, in_shape).astype(in_buffer.dtype))
        y0 = tvm.nd.array(np.zeros(out_shape).astype(out_buffer.dtype))
        y1 = tvm.nd.array(np.zeros(out_shape).astype(out_buffer.dtype))
        f_origin = tvm.build(origin)
        f_scheduled = tvm.build(scheduled)
        f_origin(x, y0)
        f_scheduled(x, y1)
        tvm.testing.assert_allclose(y0.numpy(), y1.numpy())


def _tile_nd(s, tile, block_name):
    outer_indices = []
    inner_indices = []
    block = s.get_block(block_name)
    loops = s.get_loops(block)
    for i, size in enumerate(tile):
        outer, inner = s.split(loops[i], [None, size])
        outer_indices.append(outer)
        inner_indices.append(inner)

    s.reorder(*outer_indices, *inner_indices)
    return outer_indices, inner_indices


def test_1d_rolling_buffer():
    @T.prim_func
    def before(A: T.Buffer[(4, 12), "int32"], C: T.Buffer[(4, 8), "int32"]):
        B = T.alloc_buffer((4, 10), "int32")
        for c in T.serial(4):
            for i in T.serial(0, 10):
                for k in T.serial(3):
                    with T.block("B"):
                        cc, vi, vk = T.axis.remap("SSR", [c, i, k])
                        with T.init():
                            B[cc, vi] = 0
                        B[cc, vi] = B[cc, vi] + A[cc, vi + vk]
            for i in T.serial(0, 8):
                for k in T.serial(3):
                    with T.block("C"):
                        cc, vi, vk = T.axis.remap("SSR", [c, i, k])
                        with T.init():
                            C[cc, vi] = 0
                        C[cc, vi] = C[cc, vi] + B[cc, vi + vk]

    @T.prim_func
    def expected(A: T.Buffer[(4, 12), "int32"], C: T.Buffer[(4, 8), "int32"]):
        B = T.alloc_buffer([4, 6], dtype="int32")
        for c, i_0 in T.grid(4, 2):
            for ax0, ax1 in T.grid(6, 3):
                with T.block("B"):
                    T.where(i_0 < 1 or 2 <= ax0)
                    cc = T.axis.spatial(4, c)
                    vi = T.axis.opaque(10, i_0 * 4 + ax0)
                    vk = T.axis.reduce(3, ax1)
                    T.reads(A[cc, vi + vk])
                    T.writes(B[cc, vi % 6])
                    with T.init():
                        B[cc, vi % 6] = 0
                    B[cc, vi % 6] = B[cc, vi % 6] + A[cc, vi + vk]
            for i_1, k in T.grid(4, 3):
                with T.block("C"):
                    cc = T.axis.spatial(4, c)
                    vi = T.axis.opaque(8, i_0 * 4 + i_1)
                    vk = T.axis.reduce(3, k)
                    T.reads(B[cc, (vi + vk) % 6])
                    T.writes(C[cc, vi])
                    with T.init():
                        C[cc, vi] = 0
                    C[cc, vi] = C[cc, vi] + B[cc, (vi + vk) % 6]

    sch = tir.Schedule(before, debug_mask="all")
    _, i, _ = sch.get_loops(sch.get_block("C"))
    io, _ = sch.split(i, [2, 4])
    sch.compute_at(sch.get_block("B"), io)
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, before, expected, check_run=True)


@T.prim_func
def cascade_2_max_pool2d(A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]):
    B = T.alloc_buffer([1, 10, 10, 16], dtype="int8")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 10, 10, 16, 3, 3):
        with T.block("B"):
            ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with T.init():
                B[ax0, ax1, ax2, ax3] = T.int8(-128)
            B[ax0, ax1, ax2, ax3] = T.max(B[ax0, ax1, ax2, ax3], A[ax0, ax1 + rv0, ax2 + rv1, ax3])
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 8, 8, 16, 3, 3):
        with T.block("C"):
            ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with T.init():
                C[ax0, ax1, ax2, ax3] = T.int8(-128)
            C[ax0, ax1, ax2, ax3] = T.max(C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, ax2 + rv1, ax3])


@T.prim_func
def cascade_3_max_pool2d_with_stride(
    A: T.Buffer[(1, 24, 24, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]
):
    B_0 = T.alloc_buffer([1, 22, 22, 16], dtype="int8")
    B_1 = T.alloc_buffer([1, 10, 10, 16], dtype="int8")
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 22, 22, 16, 3, 3):
        with T.block("B_0"):
            ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with T.init():
                B_0[ax0, ax1, ax2, ax3] = T.int8(-128)
            B_0[ax0, ax1, ax2, ax3] = T.max(
                B_0[ax0, ax1, ax2, ax3], A[ax0, ax1 + rv0, ax2 + rv1, ax3]
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 10, 10, 16, 3, 3):
        with T.block("B_1"):
            ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with T.init():
                B_1[ax0, ax1, ax2, ax3] = T.int8(-128)
            B_1[ax0, ax1, ax2, ax3] = T.max(
                B_1[ax0, ax1, ax2, ax3], B_0[ax0, ax1 * 2 + rv0, ax2 * 2 + rv1, ax3]
            )
    for i0, i1, i2, i3, i4, i5 in T.grid(1, 8, 8, 16, 3, 3):
        with T.block("C"):
            ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
            with T.init():
                C[ax0, ax1, ax2, ax3] = T.int8(-128)
            C[ax0, ax1, ax2, ax3] = T.max(
                C[ax0, ax1, ax2, ax3], B_1[ax0, ax1 + rv0, ax2 + rv1, ax3]
            )


def test_cascade_max_pool2d_w_tiled():
    @T.prim_func
    def expected(A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]):
        B = T.alloc_buffer([1, 10, 6, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 1, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(10, 6, 16, 3, 3):
                with T.block("B"):
                    T.where(i2_0 < 1 or 2 <= ax1)
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.spatial(10, ax0)
                    ax2_1 = T.axis.opaque(10, i2_0 * 4 + ax1)
                    ax3_1, rv0, rv1 = T.axis.remap("SRR", [ax2, ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1, ax2_1 % 6, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1, ax2_1 % 6, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1 % 6, ax3_1] = T.max(
                        B[ax0_1, ax1_1, ax2_1 % 6, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 8, 4, 16, 3, 3):
                with T.block("C"):
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.spatial(8, i1_0 * 8 + i1_1)
                    ax2 = T.axis.opaque(8, i2_0 * 4 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, ax1 + rv0, (ax2 + rv1) % 6, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, (ax2 + rv1) % 6, ax3]
                    )

    sch = tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    oi, _ = _tile_nd(sch, [1, 8, 4, 16], "C")
    sch.compute_at(sch.get_block("B"), oi[-1])
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_h_tiled():
    @T.prim_func
    def expected(A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]):
        B = T.alloc_buffer([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 1, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 10, 16, 3, 3):
                with T.block("B"):
                    T.where(i1_0 < 1 or 2 <= ax0)
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_1 = T.axis.spatial(10, ax1)
                    ax3_1, rv0, rv1 = T.axis.remap("SRR", [ax2, ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1 % 6, ax2_1, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 8, 16, 3, 3):
                with T.block("C"):
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2 = T.axis.spatial(8, i2_0 * 8 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3]
                    )

    sch = tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 8, 16], "C")
    sch.compute_at(sch.get_block("B"), io[-1])
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_h_w_c_tiled():
    @T.prim_func
    def expected(A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]):
        B = T.alloc_buffer([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 2):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 6, 8, 3, 3):
                with T.block("B"):
                    T.where((i1_0 < 1 or 2 <= ax0) and (i2_0 < 1 or 2 <= ax1))
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_1 = T.axis.spatial(10, i2_0 * 4 + ax1)
                    ax3_1 = T.axis.spatial(16, i3_0 * 8 + ax2)
                    rv0, rv1 = T.axis.remap("RR", [ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1 % 6, ax2_1, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 6, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 6, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 8, 3, 3):
                with T.block("C"):
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2 = T.axis.spatial(8, i2_0 * 4 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 8 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 6, ax2 + rv1, ax3]
                    )

    sch = tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 4, 8], "C")
    sch.compute_at(sch.get_block("B"), io[-1])
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_max_pool2d_non_perfect_tiled():
    @T.prim_func
    def expected(A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]) -> None:
        B = T.alloc_buffer([1, 8, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(8, 8, 16, 3, 3):
                with T.block("B"):
                    T.where(
                        i1_0 * 6 + ax0 < 10
                        and i2_0 * 6 + ax1 < 10
                        and (i1_0 < 1 or 2 <= ax0)
                        and (i2_0 < 1 or 2 <= ax1)
                    )
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.opaque(10, i1_0 * 6 + ax0)
                    ax2_1 = T.axis.spatial(10, i2_0 * 6 + ax1)
                    ax3_1, rv0, rv1 = T.axis.remap("SRR", [ax2, ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1 % 8, ax2_1, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1 % 8, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 8, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 8, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 6, 6, 16, 3, 3):
                with T.block("C"):
                    T.where(i1_0 * 6 + i1_1 < 8 and i2_0 * 6 + i2_1 < 8)
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.opaque(8, i1_0 * 6 + i1_1)
                    ax2 = T.axis.spatial(8, i2_0 * 6 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, (ax1 + rv0) % 8, ax2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 + rv0) % 8, ax2 + rv1, ax3]
                    )

    sch = tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 6, 6, 16], "C")
    sch.compute_at(sch.get_block("B"), io[-1])
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, cascade_2_max_pool2d, expected, check_run=True)


def test_cascade_3_max_pool2d_with_stride():
    @T.prim_func
    def expected(A: T.Buffer[(1, 24, 24, 16), "int8"], C: T.Buffer[(1, 8, 8, 16), "int8"]) -> None:
        B_0 = T.alloc_buffer([1, 13, 22, 16], dtype="int8")
        B_1 = T.alloc_buffer([1, 6, 10, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 2, 2, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(13, 13, 16, 3, 3):
                with T.block("B_0"):
                    T.where((i1_0 < 1 or 5 <= ax0) and (i2_0 < 1 or 5 <= ax1))
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.opaque(22, i1_0 * 8 + ax0)
                    ax2_1 = T.axis.spatial(22, i2_0 * 8 + ax1)
                    ax3_1, rv0, rv1 = T.axis.remap("SRR", [ax2, ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1])
                    with T.init():
                        B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1] = T.int8(-128)
                    B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1] = T.max(
                        B_0[ax0_1, ax1_1 % 13, ax2_1, ax3_1],
                        A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1],
                    )
            for ax0, ax1, ax2, ax3, ax4 in T.grid(6, 6, 16, 3, 3):
                with T.block("B_1"):
                    T.where((i1_0 < 1 or 2 <= ax0) and (i2_0 < 1 or 2 <= ax1))
                    ax0_2 = T.axis.spatial(1, 0)
                    ax1_2 = T.axis.opaque(10, i1_0 * 4 + ax0)
                    ax2_2 = T.axis.spatial(10, i2_0 * 4 + ax1)
                    ax3_2, rv0, rv1 = T.axis.remap("SRR", [ax2, ax3, ax4])
                    T.reads(B_0[ax0_2, (ax1_2 * 2 + rv0) % 13, ax2_2 * 2 + rv1, ax3_2])
                    T.writes(B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2])
                    with T.init():
                        B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2] = T.int8(-128)
                    B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2] = T.max(
                        B_1[ax0_2, ax1_2 % 6, ax2_2, ax3_2],
                        B_0[ax0_2, (ax1_2 * 2 + rv0) % 13, ax2_2 * 2 + rv1, ax3_2],
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 16, 3, 3):
                with T.block("C"):
                    ax0_3 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1_3 = T.axis.opaque(8, i1_0 * 4 + i1_1)
                    ax2_3 = T.axis.spatial(8, i2_0 * 4 + i2_1)
                    ax3_3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B_1[ax0_3, (ax1_3 + rv0) % 6, ax2_3 + rv1, ax3_3])
                    T.writes(C[ax0_3, ax1_3, ax2_3, ax3_3])
                    with T.init():
                        C[ax0_3, ax1_3, ax2_3, ax3_3] = T.int8(-128)
                    C[ax0_3, ax1_3, ax2_3, ax3_3] = T.max(
                        C[ax0_3, ax1_3, ax2_3, ax3_3],
                        B_1[ax0_3, (ax1_3 + rv0) % 6, ax2_3 + rv1, ax3_3],
                    )

    sch = tir.Schedule(cascade_3_max_pool2d_with_stride, debug_mask="all")
    io, _ = _tile_nd(sch, [1, 4, 4, 16], "C")
    sch.compute_at(sch.get_block("B_1"), io[-1])
    sch.compute_at(sch.get_block("B_0"), io[-1])
    sch.rolling_buffer(sch.get_block("B_0"), 0)
    sch.rolling_buffer(sch.get_block("B_1"), 0)
    check_rolling_buffer(sch, cascade_3_max_pool2d_with_stride, expected, check_run=True)


def test_upscale():
    @T.prim_func
    def before(A: T.Buffer[(1, 16, 16, 16), "int8"], C: T.Buffer[(1, 24, 24, 16), "int8"]) -> None:
        B = T.alloc_buffer([1, 14, 14, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 5, 5, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(5, 5, 16, 3, 3):
                with T.block("B"):
                    T.where(i1_0 * 5 // 2 + ax0 < 14 and i2_0 * 5 // 2 + ax1 < 14)
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.spatial(14, i1_0 * 5 // 2 + ax0)
                    ax2_1 = T.axis.spatial(14, i2_0 * 5 // 2 + ax1)
                    ax3_1 = T.axis.spatial(16, ax2)
                    rv0, rv1 = T.axis.remap("RR", [ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1, ax2_1, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 5, 5, 16, 3, 3):
                with T.block("C"):
                    T.where(i1_0 * 5 + i1_1 < 24 and i2_0 * 5 + i2_1 < 24)
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.spatial(24, i1_0 * 5 + i1_1)
                    ax2 = T.axis.spatial(24, i2_0 * 5 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, ax1 // 2 + rv0, ax2 // 2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 // 2 + rv0, ax2 // 2 + rv1, ax3]
                    )

    @T.prim_func
    def expected(
        A: T.Buffer[(1, 16, 16, 16), "int8"], C: T.Buffer[(1, 24, 24, 16), "int8"]
    ) -> None:
        B = T.alloc_buffer([1, 5, 14, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 5, 5, 1):
            for ax0, ax1, ax2, ax3, ax4 in T.grid(5, 5, 16, 3, 3):
                with T.block("B"):
                    T.where(
                        i1_0 * 5 // 2 + ax0 < 14
                        and i2_0 * 5 // 2 + ax1 < 14
                        and (i1_0 < 1 or 2 <= ax0)
                        and (i2_0 < 1 or 2 <= ax1)
                    )
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.opaque(14, i1_0 * 5 // 2 + ax0)
                    ax2_1 = T.axis.spatial(14, i2_0 * 5 // 2 + ax1)
                    ax3_1 = T.axis.spatial(16, ax2)
                    rv0, rv1 = T.axis.remap("RR", [ax3, ax4])
                    T.reads(A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1])
                    T.writes(B[ax0_1, ax1_1 % 5, ax2_1, ax3_1])
                    with T.init():
                        B[ax0_1, ax1_1 % 5, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1 % 5, ax2_1, ax3_1] = T.max(
                        B[ax0_1, ax1_1 % 5, ax2_1, ax3_1], A[ax0_1, ax1_1 + rv0, ax2_1 + rv1, ax3_1]
                    )
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 5, 5, 16, 3, 3):
                with T.block("C"):
                    T.where(i1_0 * 5 + i1_1 < 24 and i2_0 * 5 + i2_1 < 24)
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.opaque(24, i1_0 * 5 + i1_1)
                    ax2 = T.axis.spatial(24, i2_0 * 5 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, (ax1 // 2 + rv0) % 5, ax2 // 2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, (ax1 // 2 + rv0) % 5, ax2 // 2 + rv1, ax3]
                    )

    sch = tir.Schedule(before, debug_mask="all")
    sch.rolling_buffer(sch.get_block("B"), 0)
    check_rolling_buffer(sch, before, expected, check_run=True)


def test_fail_rolling_buffer_multi_writers():
    @T.prim_func
    def func_multi_writers(
        A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 12, 12, 16), "int8"]
    ):
        B = T.alloc_buffer([1, 12, 12, 16], dtype="int8")
        for i0, i1, i2, i3 in T.grid(1, 3, 3, 1):
            for ax0, ax1, ax2 in T.grid(6, 6, 16):
                with T.block("B_writer_0"):
                    ax0_1 = T.axis.spatial(1, i0)
                    ax1_1 = T.axis.spatial(12, i1 * 4 + ax0)
                    ax2_1 = T.axis.spatial(12, i2 * 4 + ax1)
                    ax3_1 = T.axis.spatial(16, ax2)
                    with T.init():
                        B[ax0_1, ax1_1, ax2_1, ax3_1] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3_1] = A[ax0_1, ax1_1, ax2_1, ax3_1] + T.int8(1)
            for ax0, ax1, ax2 in T.grid(6, 6, 16):
                with T.block("B_writer_1"):
                    ax0_2 = T.axis.spatial(1, i0)
                    ax1_2 = T.axis.spatial(12, i1 * 4 + ax0)
                    ax2_2 = T.axis.spatial(12, i2 * 4 + ax1)
                    ax3_2 = T.axis.spatial(16, ax2)
                    with T.init():
                        B[ax0_2, ax1_2, ax2_2, ax3_2] = T.int8(-128)
                    B[ax0_2, ax1_2, ax2_2, ax3_2] = B[ax0_2, ax1_2, ax2_2, ax3_2] + A[
                        ax0_2, ax1_2, ax2_2, ax3_2
                    ] * T.int8(2)
            for ax0, ax1, ax2, ax3, ax4, ax5 in T.grid(1, 4, 4, 16, 3, 3):
                with T.block("C"):
                    ax0_3 = T.axis.spatial(1, i0 + ax0)
                    ax1_3 = T.axis.spatial(12, i1 * 4 + ax1)
                    ax2_3 = T.axis.spatial(12, i2 * 4 + ax2)
                    ax3_3 = T.axis.spatial(16, i3 * 16 + ax3)
                    rv0, rv1 = T.axis.remap("RR", [ax4, ax5])
                    with T.init():
                        C[ax0_3, ax1_3, ax2_3, ax3_3] = T.int8(-128)
                    C[ax0_3, ax1_3, ax2_3, ax3_3] = T.max(
                        C[ax0_3, ax1_3, ax2_3, ax3_3], B[ax0_3, ax1_3 + rv0, ax2_3 + rv1, ax3_3]
                    )

    sch = tir.Schedule(func_multi_writers, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.rolling_buffer(sch.get_block("B_writer_0"), 0)


def test_fail_rolling_buffer_not_match():
    @T.prim_func
    def func_non_overlap(
        A: T.Buffer[(1, 12, 12, 16), "int8"], C: T.Buffer[(1, 12, 12, 16), "int8"]
    ):
        B = T.alloc_buffer([1, 12, 12, 16], dtype="int8")
        for i0_0, i1_0, i2_0, i3_0 in T.grid(1, 3, 3, 1):
            for ax0, ax1, ax2 in T.grid(4, 4, 16):
                with T.block("B"):
                    ax0_1 = T.axis.spatial(1, 0)
                    ax1_1 = T.axis.spatial(12, i1_0 * 4 + ax0)
                    ax2_1 = T.axis.spatial(12, i2_0 * 4 + ax1)
                    ax3 = T.axis.spatial(16, ax2)
                    T.reads(A[ax0_1, ax1_1, ax2_1, ax3])
                    T.writes(B[ax0_1, ax1_1, ax2_1, ax3])
                    with T.init():
                        B[ax0_1, ax1_1, ax2_1, ax3] = T.int8(-128)
                    B[ax0_1, ax1_1, ax2_1, ax3] = A[ax0_1, ax1_1, ax2_1, ax3]
            for i0_1, i1_1, i2_1, i3_1, i4, i5 in T.grid(1, 4, 4, 16, 1, 1):
                with T.block("C"):
                    ax0 = T.axis.spatial(1, i0_0 + i0_1)
                    ax1 = T.axis.spatial(12, i1_0 * 4 + i1_1)
                    ax2 = T.axis.spatial(12, i2_0 * 4 + i2_1)
                    ax3 = T.axis.spatial(16, i3_0 * 16 + i3_1)
                    rv0, rv1 = T.axis.remap("RR", [i4, i5])
                    T.reads(B[ax0, ax1 + rv0, ax2 + rv1, ax3])
                    T.writes(C[ax0, ax1, ax2, ax3])
                    with T.init():
                        C[ax0, ax1, ax2, ax3] = T.int8(-128)
                    C[ax0, ax1, ax2, ax3] = T.max(
                        C[ax0, ax1, ax2, ax3], B[ax0, ax1 + rv0, ax2 + rv1, ax3]
                    )

    sch = tir.Schedule(func_non_overlap, debug_mask="all")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.rolling_buffer(sch.get_block("B"), 0)


def test_fail_rolling_buffer_injection_invalid():
    sch = tir.Schedule(cascade_2_max_pool2d, debug_mask="all")
    # Block B is not compute_at to Block C, so rolling_buffer injection is invalid.
    _, _ = _tile_nd(sch, [1, 4, 8, 16], "C")
    _, _ = _tile_nd(sch, [1, 4, 8, 16], "B")
    with pytest.raises(tvm.tir.ScheduleError):
        sch.rolling_buffer(sch.get_block("B"), 0)


if __name__ == "__main__":
    tvm.testing.main()
