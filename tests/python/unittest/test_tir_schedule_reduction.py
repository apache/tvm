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
import pytest

import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable,missing-function-docstring,missing-module-docstring


@tvm.script.tir
def transformed_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in tir.grid(128, 128, 4, 8, 4):
        with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
            tir.bind(vi, i0)
            tir.bind(vj, i1)
            tir.bind(vk, (((i2_outer * 32) + (i2_inner_outer * 4)) + i2_inner_inner))
            tir.reads([C[vi, vj], A[vi, vk], B[vj, vk]])
            tir.writes([C[vi, vj]])
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@tvm.script.tir
def matmul_rfactor(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    C_rf = tir.alloc_buffer([4, 128, 128])

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in tir.grid(128, 128, 4, 8, 4):
        with tir.block(
            [4, 128, 128, tir.reduce_axis(0, 4), tir.reduce_axis(0, 8)], "update_rf"
        ) as [vi2_inner_inner, vi, vj, vi2_outer, vi2_inner_outer]:
            tir.bind(vi2_inner_inner, i2_inner_inner)
            tir.bind(vi, i0)
            tir.bind(vj, i1)
            tir.bind(vi2_outer, i2_outer)
            tir.bind(vi2_inner_outer, i2_inner_outer)
            with tir.init():
                C_rf[vi2_inner_inner, vi, vj] = 0.0
            C_rf[vi2_inner_inner, vi, vj] = C_rf[vi2_inner_inner, vi, vj] + (
                A[vi, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
                * B[vj, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
            )

    for i0_1, i1_1, i2_inner_inner_1 in tir.grid(128, 128, 4):
        with tir.block([tir.reduce_axis(0, 4), 128, 128], "update") as [
            vi2_inner_inner_1,
            vi_1,
            vj_1,
        ]:
            tir.bind(vi2_inner_inner_1, i2_inner_inner_1)
            tir.bind(vi_1, i0_1)
            tir.bind(vj_1, i1_1)
            with tir.init():
                C[vi_1, vj_1] = 0.0
            C[vi_1, vj_1] = C[vi_1, vj_1] + C_rf[vi2_inner_inner_1, vi_1, vj_1]


@tvm.script.tir
def matmul_not_stage_pipeline(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [256, 256])
    B = tir.match_buffer(b, [256, 256])
    D = tir.match_buffer(d, [256, 256])
    C = tir.alloc_buffer([256, 256])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    with tir.block([256, 256], "D") as [vi, vj]:
        D[vi, vj] = C[vi, vj]


@tvm.script.tir
def matmul_not_same_buffer_access(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vj, vi] = C[vj, vi] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def matmul_loop_multiple_children(a: ty.handle, b: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    D = tir.match_buffer(d, [128, 128])

    for k, i, j in tir.grid(128, 128, 128):
        with tir.block([tir.reduce_axis(0, 128), 128, 128], "C") as [ck, ci, cj]:
            tir.bind(ck, k)
            tir.bind(ci, i)
            tir.bind(cj, j)
            with tir.init():
                C[ci, cj] = 0.0
            C[ci, cj] = C[ci, cj] + A[ci, ck] * B[ck, cj]
        with tir.block([tir.reduce_axis(0, 128), 128, 128], "D") as [dk, di, dj]:
            tir.bind(dk, k)
            tir.bind(di, i)
            tir.bind(dj, j)
            with tir.init():
                D[di, dj] = 0.0
            D[di, dj] = D[di, dj] + B[di, dk] * A[dk, dj]


@tvm.script.tir
def square_sum(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    C = tir.match_buffer(c, [16])

    with tir.block([16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C") as [b, i, j]:
        with tir.init():
            C[b] = 0.0
        C[b] = C[b] + A[b, i, j] * A[b, i, j]


@tvm.script.tir
def square_sum_rfactor(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    C = tir.match_buffer(c, [16])
    C_rf = tir.alloc_buffer([16, 256])

    for i0, i1, i2 in tir.grid(16, 256, 256):
        with tir.block([256, 16, tir.reduce_axis(0, 256)], "C_rf") as [vi2, b, i]:
            tir.bind(vi2, i2)
            tir.bind(b, i0)
            tir.bind(i, i1)
            with tir.init():
                C_rf[b, vi2] = 0.0
            C_rf[b, vi2] = C_rf[b, vi2] + (A[b, i, vi2] * A[b, i, vi2])

    for i0_1, i2_1 in tir.grid(16, 256):
        with tir.block([tir.reduce_axis(0, 256), 16], "C") as [vi2_1, b_1]:
            tir.bind(vi2_1, i2_1)
            tir.bind(b_1, i0_1)
            with tir.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[b_1, vi2_1]


@tvm.script.tir
def transformed_square_sum_square_root(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    D = tir.match_buffer(d, [16])
    C = tir.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in tir.grid(16, 65536, 1):
        with tir.block([16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C") as [b, i, j]:
            tir.bind(b, i0)
            tir.bind(i, tir.floordiv(i1_i2_fused_outer, 256))
            tir.bind(j, tir.floormod(i1_i2_fused_outer, 256))
            tir.reads([C[b], A[b, i, j]])
            tir.writes([C[b]])
            with tir.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in tir.serial(0, 16):
        with tir.block([16], "D") as [b_1]:
            tir.bind(b_1, i0_1)
            tir.reads([C[b_1]])
            tir.writes([D[b_1]])
            D[b_1] = tir.sqrt(C[b_1], dtype="float32")


@tvm.script.tir
def square_sum_square_root_rfactor(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    D = tir.match_buffer(d, [16])
    C = tir.alloc_buffer([16])
    C_rf = tir.alloc_buffer([1, 16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in tir.grid(16, 65536, 1):
        with tir.block([1, 16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C_rf") as [
            vi1_i2_fused_inner,
            b,
            i,
            j,
        ]:
            tir.bind(vi1_i2_fused_inner, i1_i2_fused_inner)
            tir.bind(b, i0)
            tir.bind(i, tir.floordiv(i1_i2_fused_outer, 256))
            tir.bind(j, tir.floormod(i1_i2_fused_outer, 256))
            with tir.init():
                C_rf[vi1_i2_fused_inner, b] = 0.0
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + (A[b, i, j] * A[b, i, j])

    for i0_1, i1_i2_fused_inner_1 in tir.grid(16, 1):
        with tir.block([tir.reduce_axis(0, 1), 16], "C") as [vi1_i2_fused_inner_1, b_1]:
            tir.bind(vi1_i2_fused_inner_1, i1_i2_fused_inner_1)
            tir.bind(b_1, i0_1)
            with tir.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[vi1_i2_fused_inner_1, b_1]

    for i0_2 in tir.serial(0, 16):
        with tir.block([16], "D") as [b_2]:
            tir.bind(b_2, i0_2)
            D[b_2] = tir.sqrt(C[b_2], dtype="float32")


@tvm.script.tir
def element_wise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))

    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def rowsum(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_not_quasi_affine(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    for i, k in tir.grid(128, 16):
        with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
            tir.bind(vi, i)
            tir.bind(vk, tir.floordiv(k * k, 2))
            with tir.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_not_dominant(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vi, vk] = 0.0
        B[vi, vk] = B[vi, vk] + A[vi, vk]


@tvm.script.tir
def rowsum_not_serial(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    for i in tir.serial(0, 128):
        for k in tir.parallel(0, 128):
            with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
                tir.bind(vi, i)
                tir.bind(vk, k)
                with tir.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_wrong_reduce_pattern1(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vi] = 1.0
        B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_wrong_reduce_pattern2(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vi] = 0.0
        B[vi] = B[vi] - A[vi, vk]


@tvm.script.tir
def rowsum_transformed(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    for io, ii_ko_fused, ki in tir.grid(32, 128, 4):
        with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
            tir.bind(vi, io * 4 + tir.floordiv(ii_ko_fused, 32))
            tir.bind(vk, tir.floormod(ii_ko_fused, 32) * 4 + ki)
            with tir.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_zero_dim(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128])
    B = tir.match_buffer(b, [])

    with tir.block([tir.reduce_axis(0, 128)], "B") as [k]:
        with tir.init():
            B[()] = 0.0
        B[()] = B[()] + A[k]


@tvm.script.tir
def rowsum_zero_dim_rfactor(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128])
    B = tir.match_buffer(b, [])
    B_rf = tir.alloc_buffer([128])

    with tir.block([128], "B_rf") as [vi0]:
        with tir.init():
            B_rf[vi0] = 0.0
        B_rf[vi0] = B_rf[vi0] + A[vi0]

    with tir.block([tir.reduce_axis(0, 128)], "B") as [vi0_1]:
        with tir.init():
            B[()] = 0.0
        B[()] = B[()] + B_rf[vi0_1]


@tvm.script.tir
def multiple_reduction_blocks(a: ty.handle, f: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16))
    C = tir.alloc_buffer((16, 16))
    D = tir.alloc_buffer((16, 16))
    E = tir.alloc_buffer((16, 16))
    F = tir.match_buffer(f, (16, 16))

    for i in tir.serial(0, 16):
        for j1 in tir.serial(0, 16):
            for k1o, k1i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "C") as [ci, cj, ck]:
                    tir.bind(ci, i)
                    tir.bind(cj, j1)
                    tir.bind(ck, k1o * 4 + k1i)
                    with tir.init():
                        C[ci, cj] = 0.0
                    C[ci, cj] = C[ci, cj] + A[ci, cj, ck]
            for k2o, k2i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "D") as [di, dj, dk]:
                    tir.bind(di, i)
                    tir.bind(dj, j1)
                    tir.bind(dk, k2o * 4 + k2i)
                    with tir.init():
                        D[di, dj] = 0.0
                    D[di, dj] = D[di, dj] + A[di, dj, dk] + C[di, dj]
        for j2 in tir.serial(0, 16):
            for k3o, k3i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "E") as [ei, ej, ek]:
                    tir.bind(ei, i)
                    tir.bind(ej, j2)
                    tir.bind(ek, k3o * 4 + k3i)
                    with tir.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = E[ei, ej] + A[ei, ej, ek] + D[ei, ej]
            for k4o, k4i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "F") as [fi, fj, fk]:
                    tir.bind(fi, i)
                    tir.bind(fj, j2)
                    tir.bind(fk, k4o * 4 + k4i)
                    with tir.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = F[fi, fj] + A[fi, fj, fk] + E[fi, fj]


@tvm.script.tir
def multiple_reduction_blocks_rfactor(a: ty.handle, f: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16, 16])
    C = tir.alloc_buffer([16, 16])
    D = tir.alloc_buffer([16, 16])
    E = tir.alloc_buffer([16, 16])
    F = tir.match_buffer(f, [16, 16])
    C_rf = tir.alloc_buffer([16, 16, 4])

    for i, j1, k1o, k1i in tir.grid(16, 16, 4, 4):
        with tir.block([4, 16, 16, tir.reduce_axis(0, 4)], "C_rf") as [vk1o, ci, cj, vk1i]:
            tir.bind(vk1o, k1o)
            tir.bind(ci, i)
            tir.bind(cj, j1)
            tir.bind(vk1i, k1i)
            with tir.init():
                C_rf[ci, cj, vk1o] = 0.0
            C_rf[ci, cj, vk1o] = C_rf[ci, cj, vk1o] + A[ci, cj, ((vk1o * 4) + vk1i)]
    for i_1 in tir.serial(0, 16):
        for j1_1 in tir.serial(0, 16):
            for k1o_1 in tir.serial(0, 4):
                with tir.block([tir.reduce_axis(0, 4), 16, 16], "C") as [vk1o_1, ci_1, cj_1]:
                    tir.bind(vk1o_1, k1o_1)
                    tir.bind(ci_1, i_1)
                    tir.bind(cj_1, j1_1)
                    with tir.init():
                        C[ci_1, cj_1] = 0.0
                    C[ci_1, cj_1] = C[ci_1, cj_1] + C_rf[ci_1, cj_1, vk1o_1]
            for k2o, k2i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "D") as [di, dj, dk]:
                    tir.bind(di, i_1)
                    tir.bind(dj, j1_1)
                    tir.bind(dk, (k2o * 4) + k2i)
                    with tir.init():
                        D[di, dj] = 0.0
                    D[di, dj] = (D[di, dj] + A[di, dj, dk]) + C[di, dj]
        for j2 in tir.serial(0, 16):
            for k3o, k3i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "E") as [ei, ej, ek]:
                    tir.bind(ei, i_1)
                    tir.bind(ej, j2)
                    tir.bind(ek, (k3o * 4) + k3i)
                    with tir.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = (E[ei, ej] + A[ei, ej, ek]) + D[ei, ej]
            for k4o, k4i in tir.grid(4, 4):
                with tir.block([16, 16, tir.reduce_axis(0, 16)], "F") as [fi, fj, fk]:
                    tir.bind(fi, i_1)
                    tir.bind(fj, j2)
                    tir.bind(fk, (k4o * 4) + k4i)
                    with tir.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = (F[fi, fj] + A[fi, fj, fk]) + E[fi, fj]


# pylint: enable=no-member,invalid-name,unused-variable


def test_reduction_rfactor_matmul():
    s = tir.Schedule(transformed_matmul, debug_mode=True)
    C = s.get_block("update")
    _, _, _, _, kii = s.get_loops(C)
    rf_block = s.rfactor(kii, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128), dtype="float32"))
    func(a, b, c)
    c_np = np.matmul(a_np, b_np.T)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-4, atol=1e-4)


def test_reduction_rfactor_square_sum():
    s = tir.Schedule(square_sum, debug_mode=True)
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(16, 256, 256)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((16,), dtype="float32"))
    func(a, c)
    c_np = np.sum(a_np * a_np, axis=(1, 2))
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-4, atol=1e-4)


def test_reduction_rfactor_square_sum_square_root():
    s = tir.Schedule(transformed_square_sum_square_root, debug_mode=True)
    C = s.get_block("C")
    _, _, fi = s.get_loops(C)
    rf_block = s.rfactor(fi, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(16, 256, 256)).astype("float32")
    a = tvm.nd.array(a_np)
    d = tvm.nd.array(np.zeros((16,), dtype="float32"))
    func(a, d)
    d_np = np.sqrt(np.sum(a_np * a_np, axis=(1, 2)))
    tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-4, atol=1e-4)


def test_reduction_rfactor_loop_multiple_children():
    s = tir.Schedule(matmul_loop_multiple_children, debug_mode=True)
    C = s.get_block("C")
    k, _, _ = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_stage_pipeline():
    s = tir.Schedule(matmul_not_stage_pipeline, debug_mode=True)
    C = s.get_block("C")
    _, _, k = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block1():
    s = tir.Schedule(element_wise, debug_mode=True)
    B = s.get_block("B")
    i, _ = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_not_reduction_block2():
    s = tir.Schedule(rowsum_not_quasi_affine, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_reduction_block3():
    s = tir.Schedule(rowsum_not_dominant, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_serial_loop():
    s = tir.Schedule(rowsum_not_serial, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_not_same_buffer_access():
    s = tir.Schedule(matmul_not_same_buffer_access, debug_mode=True)
    C = s.get_block("C")
    _, _, k = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_factor_axis_range():
    s = tir.Schedule(transformed_matmul, debug_mode=True)
    C = s.get_block("update")
    _, _, _, _, kii = s.get_loops(C)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, 3)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(kii, -4)

    rf_block = s.rfactor(kii, -3)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128), dtype="float32"))
    func(a, b, c)
    c_np = np.matmul(a_np, b_np.T)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-4, atol=1e-4)


def test_reduction_rfactor_wrong_reduce_pattern1():
    s = tir.Schedule(rowsum_wrong_reduce_pattern1, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_reduce_pattern2():
    s = tir.Schedule(rowsum_wrong_reduce_pattern2, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(k, 0)


def test_reduction_rfactor_wrong_loops1():
    s = tir.Schedule(rowsum, debug_mode=True)
    B = s.get_block("B")
    i, _ = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(i, 0)


def test_reduction_rfactor_wrong_loops2():
    s = tir.Schedule(rowsum_transformed, debug_mode=True)
    B = s.get_block("B")
    _, _, ki = s.get_loops(B)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 0)


def test_reduction_rfactor_zero_dim():
    s = tir.Schedule(rowsum_zero_dim, debug_mode=True)
    B = s.get_block("B")
    (k,) = s.get_loops(B)
    s.rfactor(k, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_zero_dim_rfactor)

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(128,)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(np.array(1, dtype="float32"))
    func(a, b)
    b_np = np.array(np.sum(a_np))
    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-4, atol=1e-4)


def test_reduction_rfactor_outermost_loop_multiple_children():
    s = tir.Schedule(multiple_reduction_blocks, debug_mode=True)
    D = s.get_block("D")
    E = s.get_block("E")
    F = s.get_block("F")
    _, _, k2o, k2i = s.get_loops(D)
    _, _, k3o, k3i = s.get_loops(E)
    _, _, k4o, k4i = s.get_loops(F)
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

    C = s.get_block("C")
    i, j1, k1o, k1i = s.get_loops(C)
    s.rfactor(k1o, 2)
    tvm.ir.assert_structural_equal(s.mod["main"], multiple_reduction_blocks_rfactor)

    func = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np)
    f = tvm.nd.array(np.zeros((16, 16), dtype="float32"))
    func(a, f)
    f_np = np.sum(a_np, axis=2) * 4369
    tvm.testing.assert_allclose(f.numpy(), f_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
