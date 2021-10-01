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
def transformed_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
            T.bind(vi, i0)
            T.bind(vj, i1)
            T.bind(vk, (((i2_outer * 32) + (i2_inner_outer * 4)) + i2_inner_inner))
            T.reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj]])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def matmul_rfactor(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    C_rf = T.alloc_buffer([4, 128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block([4, 128, 128, T.reduce_axis(0, 4), T.reduce_axis(0, 8)], "update_rf") as [
            vi2_inner_inner,
            vi,
            vj,
            vi2_outer,
            vi2_inner_outer,
        ]:
            T.bind(vi2_inner_inner, i2_inner_inner)
            T.bind(vi, i0)
            T.bind(vj, i1)
            T.bind(vi2_outer, i2_outer)
            T.bind(vi2_inner_outer, i2_inner_outer)
            with T.init():
                C_rf[vi2_inner_inner, vi, vj] = 0.0
            C_rf[vi2_inner_inner, vi, vj] = C_rf[vi2_inner_inner, vi, vj] + (
                A[vi, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
                * B[vj, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
            )

    for i0_1, i1_1, i2_inner_inner_1 in T.grid(128, 128, 4):
        with T.block([T.reduce_axis(0, 4), 128, 128], "update") as [
            vi2_inner_inner_1,
            vi_1,
            vj_1,
        ]:
            T.bind(vi2_inner_inner_1, i2_inner_inner_1)
            T.bind(vi_1, i0_1)
            T.bind(vj_1, i1_1)
            with T.init():
                C[vi_1, vj_1] = 0.0
            C[vi_1, vj_1] = C[vi_1, vj_1] + C_rf[vi2_inner_inner_1, vi_1, vj_1]


@T.prim_func
def matmul_not_stage_pipeline(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [256, 256])
    B = T.match_buffer(b, [256, 256])
    D = T.match_buffer(d, [256, 256])
    C = T.alloc_buffer([256, 256])

    with T.block([128, 128, T.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    with T.block([256, 256], "D") as [vi, vj]:
        D[vi, vj] = C[vi, vj]


@T.prim_func
def matmul_not_same_buffer_access(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    with T.block([128, 128, T.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vj, vi] = C[vj, vi] + A[vi, vk] * B[vk, vj]


@T.prim_func
def matmul_loop_multiple_children(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    D = T.match_buffer(d, [128, 128])

    for k, i, j in T.grid(128, 128, 128):
        with T.block([T.reduce_axis(0, 128), 128, 128], "C") as [ck, ci, cj]:
            T.bind(ck, k)
            T.bind(ci, i)
            T.bind(cj, j)
            with T.init():
                C[ci, cj] = 0.0
            C[ci, cj] = C[ci, cj] + A[ci, ck] * B[ck, cj]
        with T.block([T.reduce_axis(0, 128), 128, 128], "D") as [dk, di, dj]:
            T.bind(dk, k)
            T.bind(di, i)
            T.bind(dj, j)
            with T.init():
                D[di, dj] = 0.0
            D[di, dj] = D[di, dj] + B[di, dk] * A[dk, dj]


@T.prim_func
def square_sum(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])

    with T.block([16, T.reduce_axis(0, 256), T.reduce_axis(0, 256)], "C") as [b, i, j]:
        with T.init():
            C[b] = 0.0
        C[b] = C[b] + A[b, i, j] * A[b, i, j]


@T.prim_func
def square_sum_rfactor(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])
    C_rf = T.alloc_buffer([16, 256])

    for i0, i1, i2 in T.grid(16, 256, 256):
        with T.block([256, 16, T.reduce_axis(0, 256)], "C_rf") as [vi2, b, i]:
            T.bind(vi2, i2)
            T.bind(b, i0)
            T.bind(i, i1)
            with T.init():
                C_rf[b, vi2] = 0.0
            C_rf[b, vi2] = C_rf[b, vi2] + (A[b, i, vi2] * A[b, i, vi2])

    for i0_1, i2_1 in T.grid(16, 256):
        with T.block([T.reduce_axis(0, 256), 16], "C") as [vi2_1, b_1]:
            T.bind(vi2_1, i2_1)
            T.bind(b_1, i0_1)
            with T.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[b_1, vi2_1]


@T.prim_func
def transformed_square_sum_square_root(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block([16, T.reduce_axis(0, 256), T.reduce_axis(0, 256)], "C") as [b, i, j]:
            T.bind(b, i0)
            T.bind(i, T.floordiv(i1_i2_fused_outer, 256))
            T.bind(j, T.floormod(i1_i2_fused_outer, 256))
            T.reads([C[b], A[b, i, j]])
            T.writes([C[b]])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block([16], "D") as [b_1]:
            T.bind(b_1, i0_1)
            T.reads([C[b_1]])
            T.writes([D[b_1]])
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_rfactor(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])
    C_rf = T.alloc_buffer([1, 16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block([1, 16, T.reduce_axis(0, 256), T.reduce_axis(0, 256)], "C_rf") as [
            vi1_i2_fused_inner,
            b,
            i,
            j,
        ]:
            T.bind(vi1_i2_fused_inner, i1_i2_fused_inner)
            T.bind(b, i0)
            T.bind(i, T.floordiv(i1_i2_fused_outer, 256))
            T.bind(j, T.floormod(i1_i2_fused_outer, 256))
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = 0.0
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + (A[b, i, j] * A[b, i, j])

    for i0_1, i1_i2_fused_inner_1 in T.grid(16, 1):
        with T.block([T.reduce_axis(0, 1), 16], "C") as [vi1_i2_fused_inner_1, b_1]:
            T.bind(vi1_i2_fused_inner_1, i1_i2_fused_inner_1)
            T.bind(b_1, i0_1)
            with T.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[vi1_i2_fused_inner_1, b_1]

    for i0_2 in T.serial(0, 16):
        with T.block([16], "D") as [b_2]:
            T.bind(b_2, i0_2)
            D[b_2] = T.sqrt(C[b_2], dtype="float32")


@T.prim_func
def element_wise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    with T.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def rowsum(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_quasi_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 16):
        with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
            T.bind(vi, i)
            T.bind(vk, T.floordiv(k * k, 2))
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_dominant(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vi, vk] = 0.0
        B[vi, vk] = B[vi, vk] + A[vi, vk]


@T.prim_func
def rowsum_not_serial(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i in T.serial(0, 128):
        for k in T.parallel(0, 128):
            with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
                T.bind(vi, i)
                T.bind(vk, k)
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vi] = 1.0
        B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern2(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] - A[vi, vk]


@T.prim_func
def rowsum_transformed(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for io, ii_ko_fused, ki in T.grid(32, 128, 4):
        with T.block([128, T.reduce_axis(0, 128)], "B") as [vi, vk]:
            T.bind(vi, io * 4 + T.floordiv(ii_ko_fused, 32))
            T.bind(vk, T.floormod(ii_ko_fused, 32) * 4 + ki)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_zero_dim(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128])
    B = T.match_buffer(b, [])

    with T.block([T.reduce_axis(0, 128)], "B") as [k]:
        with T.init():
            B[()] = 0.0
        B[()] = B[()] + A[k]


@T.prim_func
def rowsum_zero_dim_rfactor(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128])
    B = T.match_buffer(b, [])
    B_rf = T.alloc_buffer([128])

    with T.block([128], "B_rf") as [vi0]:
        with T.init():
            B_rf[vi0] = 0.0
        B_rf[vi0] = B_rf[vi0] + A[vi0]

    with T.block([T.reduce_axis(0, 128)], "B") as [vi0_1]:
        with T.init():
            B[()] = 0.0
        B[()] = B[()] + B_rf[vi0_1]


@T.prim_func
def multiple_reduction_blocks(a: T.handle, f: T.handle) -> None:
    A = T.match_buffer(a, (16, 16, 16))
    C = T.alloc_buffer((16, 16))
    D = T.alloc_buffer((16, 16))
    E = T.alloc_buffer((16, 16))
    F = T.match_buffer(f, (16, 16))

    for i in T.serial(0, 16):
        for j1 in T.serial(0, 16):
            for k1o, k1i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "C") as [ci, cj, ck]:
                    T.bind(ci, i)
                    T.bind(cj, j1)
                    T.bind(ck, k1o * 4 + k1i)
                    with T.init():
                        C[ci, cj] = 0.0
                    C[ci, cj] = C[ci, cj] + A[ci, cj, ck]
            for k2o, k2i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "D") as [di, dj, dk]:
                    T.bind(di, i)
                    T.bind(dj, j1)
                    T.bind(dk, k2o * 4 + k2i)
                    with T.init():
                        D[di, dj] = 0.0
                    D[di, dj] = D[di, dj] + A[di, dj, dk] + C[di, dj]
        for j2 in T.serial(0, 16):
            for k3o, k3i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "E") as [ei, ej, ek]:
                    T.bind(ei, i)
                    T.bind(ej, j2)
                    T.bind(ek, k3o * 4 + k3i)
                    with T.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = E[ei, ej] + A[ei, ej, ek] + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "F") as [fi, fj, fk]:
                    T.bind(fi, i)
                    T.bind(fj, j2)
                    T.bind(fk, k4o * 4 + k4i)
                    with T.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = F[fi, fj] + A[fi, fj, fk] + E[fi, fj]


@T.prim_func
def multiple_reduction_blocks_rfactor(a: T.handle, f: T.handle) -> None:
    A = T.match_buffer(a, [16, 16, 16])
    C = T.alloc_buffer([16, 16])
    D = T.alloc_buffer([16, 16])
    E = T.alloc_buffer([16, 16])
    F = T.match_buffer(f, [16, 16])
    C_rf = T.alloc_buffer([16, 16, 4])

    for i, j1, k1o, k1i in T.grid(16, 16, 4, 4):
        with T.block([4, 16, 16, T.reduce_axis(0, 4)], "C_rf") as [vk1o, ci, cj, vk1i]:
            T.bind(vk1o, k1o)
            T.bind(ci, i)
            T.bind(cj, j1)
            T.bind(vk1i, k1i)
            with T.init():
                C_rf[ci, cj, vk1o] = 0.0
            C_rf[ci, cj, vk1o] = C_rf[ci, cj, vk1o] + A[ci, cj, ((vk1o * 4) + vk1i)]
    for i_1 in T.serial(0, 16):
        for j1_1 in T.serial(0, 16):
            for k1o_1 in T.serial(0, 4):
                with T.block([T.reduce_axis(0, 4), 16, 16], "C") as [vk1o_1, ci_1, cj_1]:
                    T.bind(vk1o_1, k1o_1)
                    T.bind(ci_1, i_1)
                    T.bind(cj_1, j1_1)
                    with T.init():
                        C[ci_1, cj_1] = 0.0
                    C[ci_1, cj_1] = C[ci_1, cj_1] + C_rf[ci_1, cj_1, vk1o_1]
            for k2o, k2i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "D") as [di, dj, dk]:
                    T.bind(di, i_1)
                    T.bind(dj, j1_1)
                    T.bind(dk, (k2o * 4) + k2i)
                    with T.init():
                        D[di, dj] = 0.0
                    D[di, dj] = (D[di, dj] + A[di, dj, dk]) + C[di, dj]
        for j2 in T.serial(0, 16):
            for k3o, k3i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "E") as [ei, ej, ek]:
                    T.bind(ei, i_1)
                    T.bind(ej, j2)
                    T.bind(ek, (k3o * 4) + k3i)
                    with T.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = (E[ei, ej] + A[ei, ej, ek]) + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block([16, 16, T.reduce_axis(0, 16)], "F") as [fi, fj, fk]:
                    T.bind(fi, i_1)
                    T.bind(fj, j2)
                    T.bind(fk, (k4o * 4) + k4i)
                    with T.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = (F[fi, fj] + A[fi, fj, fk]) + E[fi, fj]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_reduction_rfactor_matmul():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul)


def test_reduction_rfactor_square_sum():
    s = tir.Schedule(square_sum, debug_mask="all")
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=square_sum)


def test_reduction_rfactor_square_sum_square_root():
    s = tir.Schedule(transformed_square_sum_square_root, debug_mask="all")
    C = s.get_block("C")
    _, _, f_i = s.get_loops(C)
    rf_block = s.rfactor(f_i, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=transformed_square_sum_square_root)


def test_reduction_rfactor_loop_multiple_children():
    s = tir.Schedule(matmul_loop_multiple_children, debug_mask="all")
    k, _, _ = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_stage_pipeline():
    s = tir.Schedule(matmul_not_stage_pipeline, debug_mask="all")
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block1():
    s = tir.Schedule(element_wise, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_not_reduction_block2():
    s = tir.Schedule(rowsum_not_quasi_affine, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block3():
    s = tir.Schedule(rowsum_not_dominant, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_serial_loop():
    s = tir.Schedule(rowsum_not_serial, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_same_buffer_access():
    s = tir.Schedule(matmul_not_same_buffer_access, debug_mask="all")
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_factor_axis_range_fail():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    _, _, _, _, kii = s.get_loops(s.get_block("update"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, 3)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, -4)


def test_reduction_rfactor_factor_axis_range():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, -3)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul)


def test_reduction_rfactor_wrong_reduce_pattern1():
    s = tir.Schedule(rowsum_wrong_reduce_pattern1, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_reduce_pattern2():
    s = tir.Schedule(rowsum_wrong_reduce_pattern2, debug_mask="all")
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_loops1():
    s = tir.Schedule(rowsum, debug_mask="all")
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_wrong_loops2():
    s = tir.Schedule(rowsum_transformed, debug_mask="all")
    _, _, k_i = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k_i, 0)


def test_reduction_rfactor_zero_dim():
    s = tir.Schedule(rowsum_zero_dim, debug_mask="all")
    B = s.get_block("B")
    (k,) = s.get_loops(B)
    rf_block = s.rfactor(k, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_zero_dim_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("B_rf")))
    assert s.get(B).same_as(s.get(s.get_block("B")))
    verify_trace_roundtrip(s, mod=rowsum_zero_dim)


def test_reduction_rfactor_outermost_loop_multiple_children_fail():  # pylint: disable=invalid-name
    s = tir.Schedule(multiple_reduction_blocks, debug_mask="all")
    _, _, k2o, k2i = s.get_loops(s.get_block("D"))
    _, _, k3o, k3i = s.get_loops(s.get_block("E"))
    _, _, k4o, k4i = s.get_loops(s.get_block("F"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k2o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k2i, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k3o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k3i, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k4o, 0)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k4i, 0)


def test_reduction_rfactor_outermost_loop_multiple_children():  # pylint: disable=invalid-name
    s = tir.Schedule(multiple_reduction_blocks, debug_mask="all")
    C = s.get_block("C")
    _, _, k1o, _ = s.get_loops(C)
    rf_block = s.rfactor(k1o, 2)
    tvm.ir.assert_structural_equal(s.mod["main"], multiple_reduction_blocks_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=multiple_reduction_blocks)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
