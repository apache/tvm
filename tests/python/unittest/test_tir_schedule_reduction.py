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
import sys

import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def rowsum_blockized(a: T.handle, b: T.handle) -> None:
    B = T.match_buffer(b, [32, 4])
    A = T.match_buffer(a, [32, 4, 128])
    for i0, i2_0 in T.grid(32, 16):
        with T.block([32, T.reduce_axis(0, 16)], "blockized_B") as [io, ko]:
            T.bind(io, i0)
            T.bind(ko, i2_0)
            with T.init():
                for i1 in T.serial(0, 4):
                    with T.block([4], "B_init") as [ii_init]:
                        T.bind(ii_init, i1)
                        B[io, ii_init] = 0.0
            for i1_1, i2_1 in T.grid(4, 8):
                with T.block([4, T.reduce_axis(0, 128)], "B") as [ii, k]:
                    T.bind(ii, i1_1)
                    T.bind(k, ko * 8 + i2_1)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose0(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block([128, 128], "init") as [vi, vj]:
        C[vi, vj] = 0.0

    with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [32, 4, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [32, 4], elem_offset=0, align=128, offset_factor=1)

    for i0 in T.serial(0, 32):
        with T.block([32], "blockized_B_init") as [io]:
            for i1 in T.serial(0, 4):
                with T.block([4], "B_init") as [ii]:
                    B[io, ii] = T.float32(0)
    for i0, i2_o in T.grid(32, 16):
        with T.block([32, T.reduce_axis(0, 16)], "blockized_B_update") as [io, ko]:
            for i1, i2_i in T.grid(4, 8):
                with T.block([4, T.reduce_axis(0, 128)], "B") as [ii, k]:
                    T.bind(ii, i1)
                    T.bind(k, ((ko * 8) + i2_i))
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@T.prim_func
def matmul_decompose2(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)

    for i0, i1 in T.grid(128, 128):
        with T.block([128, 128], "update_init") as [vi_init, vj_init]:
            C[vi_init, vj_init] = T.float32(0)
        for i2 in T.serial(0, 128):
            with T.block([128, 128, T.reduce_axis(0, 128)], "update_update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def matmul_decompose_fail3(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, k, j in T.grid(128, 128, 128):
        with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
            with T.init():
                C[vi, vj] = 0.0
            T.bind(vi, i)
            T.bind(vj, j)
            T.bind(vk, k)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_decompose4(a: T.handle, b: T.handle, c: T.handle) -> None:
    C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = T.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with T.block([], "root"):
        T.reads([])
        T.writes([])
        for i0_0 in T.serial(0, 16):
            for i0_1_init, i1_init in T.grid(8, 128):
                with T.block([128, 128], "update_init") as [vi_init, vj_init]:
                    T.bind(vi_init, ((i0_0 * 8) + i0_1_init))
                    T.bind(vj_init, i1_init)
                    C[vi_init, vj_init] = T.float32(0)
            for i0_1, i1, i2_0, i2_1 in T.grid(8, 128, 19, 7):
                with T.block([128, 128, T.reduce_axis(0, 128)], "update_update") as [
                    vi,
                    vj,
                    vk,
                ]:
                    T.where((((i2_0 * 7) + i2_1) < 128))
                    T.bind(vi, ((i0_0 * 8) + i0_1))
                    T.bind(vj, i1)
                    T.bind(vk, ((i2_0 * 7) + i2_1))
                    C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_reduction_decompose0():
    s = tir.Schedule(matmul, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose1():
    s = tir.Schedule(rowsum_blockized, debug_mask="all")
    blockized_B = s.get_block("blockized_B")
    io, ko = s.get_loops(blockized_B)
    s.decompose_reduction(blockized_B, io)
    tvm.ir.assert_structural_equal(matmul_decompose1, s.mod["main"])
    verify_trace_roundtrip(s, mod=rowsum_blockized)


def test_reduction_decompose2():
    s = tir.Schedule(matmul, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    s.decompose_reduction(C, k)
    tvm.ir.assert_structural_equal(matmul_decompose2, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


def test_reduction_decompose3():
    s = tir.Schedule(matmul_decompose_fail3, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.decompose_reduction(C, k)


def test_reduction_decompose4():
    s = tir.Schedule(matmul, debug_mask="all")
    C = s.get_block("update")
    i, j, k = s.get_loops(C)
    io, ii = s.split(i, factors=[16, 8])
    ko, ki = s.split(k, factors=[19, 7])
    s.decompose_reduction(C, ii)
    tvm.ir.assert_structural_equal(matmul_decompose4, s.mod["main"])
    verify_trace_roundtrip(s, mod=matmul)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
