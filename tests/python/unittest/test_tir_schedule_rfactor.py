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
from tvm import te, tir, topi
from tvm.script import tir as T
from tvm.tir.schedule.testing import (
    assert_structural_equal_ignore_global_symbol,
    verify_trace_roundtrip,
)

# pylint: disable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


@T.prim_func
def transformed_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128, 128], dtype="float32")
    C = T.match_buffer(c, [128, 128], dtype="float32")

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads([A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj]])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@T.prim_func
def transformed_matmul_with_let(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128, 128], dtype="float32")
    C = T.match_buffer(c, [128, 128], dtype="float32")

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update"):
            vi, vj = T.axis.remap("SS", [i0, i1])
            vk = T.axis.R(128, i2_outer * 32 + i2_inner_outer * 4 + i2_inner_inner)
            T.reads([A[vi, vk], B[vj, vk]])
            T.writes([C[vi, vj]])
            with T.init():
                C[vi, vj] = 0.0
            v_C: T.float32 = C[vi, vj] + (A[vi, vk] * B[vj, vk])
            C[vi, vj] = v_C


@T.prim_func
def matmul_rfactor(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128, 128], dtype="float32")
    C = T.match_buffer(c, [128, 128], dtype="float32")
    C_rf = T.alloc_buffer([4, 128, 128], dtype="float32")

    for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(128, 128, 4, 8, 4):
        with T.block("update_rf"):
            vi2_inner_inner = T.axis.S(4, i2_inner_inner)
            vi = T.axis.S(128, i0)
            vj = T.axis.S(128, i1)
            vi2_outer = T.axis.R(4, i2_outer)
            vi2_inner_outer = T.axis.R(8, i2_inner_outer)
            with T.init():
                C_rf[vi2_inner_inner, vi, vj] = 0.0
            C_rf[vi2_inner_inner, vi, vj] = C_rf[vi2_inner_inner, vi, vj] + (
                A[vi, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
                * B[vj, (((vi2_outer * 32) + (vi2_inner_outer * 4)) + vi2_inner_inner)]
            )

    for i0_1, i1_1, i2_inner_inner_1 in T.grid(128, 128, 4):
        with T.block("update"):
            vi2_inner_inner_1, vi_1, vj_1 = T.axis.remap("RSS", [i2_inner_inner_1, i0_1, i1_1])
            with T.init():
                C[vi_1, vj_1] = 0.0
            C[vi_1, vj_1] = C[vi_1, vj_1] + C_rf[vi2_inner_inner_1, vi_1, vj_1]


@T.prim_func
def matmul_not_stage_pipeline(a: T.handle, b: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [256, 256])
    B = T.match_buffer(b, [256, 256])
    D = T.match_buffer(d, [256, 256])
    C = T.alloc_buffer([256, 256])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    for i, j in T.grid(256, 256):
        with T.block("D"):
            vi, vj = T.axis.remap("SS", [i, j])
            D[vi, vj] = C[vi, vj]


@T.prim_func
def matmul_not_same_buffer_access(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    C = T.match_buffer(c, (128, 128))

    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
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
        with T.block("C"):
            ck, ci, cj = T.axis.remap("RSS", [k, i, j])
            with T.init():
                C[ci, cj] = 0.0
            C[ci, cj] = C[ci, cj] + A[ci, ck] * B[ck, cj]
        with T.block("D"):
            dk, di, dj = T.axis.remap("RSS", [k, i, j])
            with T.init():
                D[di, dj] = 0.0
            D[di, dj] = D[di, dj] + B[di, dk] * A[dk, dj]


@T.prim_func
def square_sum(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])

    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C"):
            b, i, j = T.axis.remap("SRR", [b0, i0, j0])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + A[b, i, j] * A[b, i, j]


@T.prim_func
def square_sum_rfactor(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])
    C_rf = T.alloc_buffer([16, 256])

    for i0, i1, i2 in T.grid(16, 256, 256):
        with T.block("C_rf"):
            vi2, b, i = T.axis.remap("SSR", [i2, i0, i1])
            with T.init():
                C_rf[b, vi2] = 0.0
            C_rf[b, vi2] = C_rf[b, vi2] + (A[b, i, vi2] * A[b, i, vi2])

    for i0_1, i2_1 in T.grid(16, 256):
        with T.block("C"):
            vi2_1, b_1 = T.axis.remap("RS", [i2_1, i0_1])
            with T.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[b_1, vi2_1]


@T.prim_func
def transformed_square_sum_square_root(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            T.reads([A[b, i, j]])
            T.writes([C[b]])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
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
        with T.block("C_rf"):
            vi1_i2_fused_inner, b = T.axis.remap("SS", [i1_i2_fused_inner, i0])
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = 0.0
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + (A[b, i, j] * A[b, i, j])

    for i0_1, i1_i2_fused_inner_1 in T.grid(16, 1):
        with T.block("C"):
            vi1_i2_fused_inner_1, b_1 = T.axis.remap("RS", [i1_i2_fused_inner_1, i0_1])
            with T.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[vi1_i2_fused_inner_1, b_1]

    for i0_2 in T.serial(0, 16):
        with T.block("D"):
            b_2 = T.axis.S(16, i0_2)
            D[b_2] = T.sqrt(C[b_2], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_1(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_outer, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_outer, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_1_rfactor(
    A: T.Buffer((16, 256, 256), "float32"), D: T.Buffer((16,), "float32")
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([1, 16], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 65536, 1):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_outer // 256)
            j = T.axis.reduce(256, i1_i2_fused_outer % 256)
            vi1_i2_fused_inner = T.axis.spatial(1, i1_i2_fused_inner)
            with T.init():
                C_rf[vi1_i2_fused_inner, b] = T.float32(0)
            C_rf[vi1_i2_fused_inner, b] = C_rf[vi1_i2_fused_inner, b] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_inner in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_inner = T.axis.remap("SR", [i0, i1_i2_fused_inner])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[vi1_i2_fused_inner, b]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def transformed_square_sum_square_root_factor_one_2(a: T.handle, d: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    D = T.match_buffer(d, [16])
    C = T.alloc_buffer([16])

    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C"):
            b = T.axis.S(16, i0)
            i = T.axis.R(256, T.floordiv(i1_i2_fused_inner, 256))
            j = T.axis.R(256, T.floormod(i1_i2_fused_inner, 256))
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + (A[b, i, j] * A[b, i, j])
    for i0_1 in T.serial(0, 16):
        with T.block("D"):
            b_1 = T.axis.S(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_square_root_factor_one_2_rfactor(
    A: T.Buffer((16, 256, 256), "float32"), D: T.Buffer((16,), "float32")
) -> None:
    C = T.alloc_buffer([16], dtype="float32")
    C_rf = T.alloc_buffer([16, 1], dtype="float32")
    for i0, i1_i2_fused_outer, i1_i2_fused_inner in T.grid(16, 1, 65536):
        with T.block("C_rf"):
            b = T.axis.spatial(16, i0)
            i = T.axis.reduce(256, i1_i2_fused_inner // 256)
            j = T.axis.reduce(256, i1_i2_fused_inner % 256)
            vi1_i2_fused_outer = T.axis.spatial(1, i1_i2_fused_outer)
            with T.init():
                C_rf[b, vi1_i2_fused_outer] = T.float32(0)
            C_rf[b, vi1_i2_fused_outer] = C_rf[b, vi1_i2_fused_outer] + A[b, i, j] * A[b, i, j]
    for i0, i1_i2_fused_outer in T.grid(16, 1):
        with T.block("C"):
            b, vi1_i2_fused_outer = T.axis.remap("SR", [i0, i1_i2_fused_outer])
            with T.init():
                C[b] = T.float32(0)
            C[b] = C[b] + C_rf[b, vi1_i2_fused_outer]
    for i0_1 in T.serial(16):
        with T.block("D"):
            b_1 = T.axis.spatial(16, i0_1)
            D[b_1] = T.sqrt(C[b_1], dtype="float32")


@T.prim_func
def square_sum_with_annotation(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])

    for b0, i0, j0 in T.grid(16, 256, 256):
        with T.block("C"):
            T.block_attr({"test_annotation": 1})
            b, i, j = T.axis.remap("SRR", [b0, i0, j0])
            with T.init():
                C[b] = 0.0
            C[b] = C[b] + A[b, i, j] * A[b, i, j]


@T.prim_func
def square_sum_with_annotation_rfactor(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 256, 256])
    C = T.match_buffer(c, [16])
    C_rf = T.alloc_buffer([16, 256])

    for i0, i1, i2 in T.grid(16, 256, 256):
        with T.block("C_rf"):
            T.block_attr({"test_annotation": 1})
            vi2, b, i = T.axis.remap("SSR", [i2, i0, i1])
            with T.init():
                C_rf[b, vi2] = 0.0
            C_rf[b, vi2] = C_rf[b, vi2] + (A[b, i, vi2] * A[b, i, vi2])

    for i0_1, i2_1 in T.grid(16, 256):
        with T.block("C"):
            T.block_attr({"test_annotation": 1})
            vi2_1, b_1 = T.axis.remap("RS", [i2_1, i0_1])
            with T.init():
                C[b_1] = 0.0
            C[b_1] = C[b_1] + C_rf[b_1, vi2_1]


@T.prim_func
def element_wise(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


@T.prim_func
def rowsum(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_quasi_affine(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 16):
        with T.block("B"):
            vi = T.axis.S(128, i)
            vk = T.axis.R(128, T.floordiv(k * k, 2))
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_not_dominant(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi, vk] = 0.0
            B[vi, vk] = B[vi, vk] + A[vi, vk]


@T.prim_func
def rowsum_not_serial(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i in T.serial(0, 128):
        for k in T.parallel(0, 128):
            with T.block("B"):
                vi, vk = T.axis.remap("SR", [i, k])
                with T.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern1(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 1.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_wrong_reduce_pattern2(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] - A[vi, vk]


@T.prim_func
def rowsum_init_not_bufferstore(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for i, k in T.grid(128, 128):
        with T.block("B"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                v_init: T.float32 = T.float32(0)
                B[vi] = v_init
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_transformed(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128,))

    for io, ii_ko_fused, ki in T.grid(32, 128, 4):
        with T.block("B"):
            vi = T.axis.S(128, io * 4 + T.floordiv(ii_ko_fused, 32))
            vk = T.axis.R(128, T.floormod(ii_ko_fused, 32) * 4 + ki)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_zero_dim(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128])
    B = T.match_buffer(b, [])

    for k0 in range(128):
        with T.block("B"):
            k = T.axis.R(128, k0)
            with T.init():
                B[()] = 0.0
            B[()] = B[()] + A[k]


@T.prim_func
def rowsum_zero_dim_rfactor(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128])
    B = T.match_buffer(b, [])
    B_rf = T.alloc_buffer([128], elem_offset=T.int64(0))

    for i in range(128):
        with T.block("B_rf"):
            vi0 = T.axis.S(128, i)
            B_rf[vi0] = A[vi0]

    for i in range(128):
        with T.block("B"):
            vi0_1 = T.axis.R(128, i)
            with T.init():
                B[()] = 0.0
            B[()] = B[()] + B_rf[vi0_1]


@T.prim_func
def rowsum_predicate(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    for i, k_0, k_1 in T.grid(128, 13, 10):
        with T.block("B"):
            T.where(k_0 * 10 + k_1 < 128)
            vi = T.axis.S(128, i)
            vk = T.axis.R(128, k_0 * 10 + k_1)
            with T.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@T.prim_func
def rowsum_predicate_rfactor(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [128, 128], dtype="float32")
    B = T.match_buffer(b, [128], dtype="float32")
    B_rf = T.alloc_buffer([128, 13], dtype="float32")
    for i, k_0, k_1 in T.grid(128, 13, 10):
        with T.block("B_rf"):
            vk_0, vi, vk_1 = T.axis.remap("SSR", [k_0, i, k_1])
            T.where(k_0 * 10 + k_1 < 128)
            with T.init():
                B_rf[vi, vk_0] = T.float32(0)
            B_rf[vi, vk_0] = B_rf[vi, vk_0] + A[vi, vk_0 * 10 + vk_1]
    for i, k_0 in T.grid(128, 13):
        with T.block("B"):
            vk_0, vi = T.axis.remap("RS", [k_0, i])
            with T.init():
                B[vi] = T.float32(0)
            B[vi] = B[vi] + B_rf[vi, vk_0]


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
                with T.block("C"):
                    ci, cj = T.axis.remap("SS", [i, j1])
                    ck = T.axis.R(16, k1o * 4 + k1i)
                    with T.init():
                        C[ci, cj] = 0.0
                    C[ci, cj] = C[ci, cj] + A[ci, cj, ck]
            for k2o, k2i in T.grid(4, 4):
                with T.block("D"):
                    di, dj = T.axis.remap("SS", [i, j1])
                    dk = T.axis.R(16, k2o * 4 + k2i)
                    with T.init():
                        D[di, dj] = 0.0
                    D[di, dj] = D[di, dj] + A[di, dj, dk] + C[di, dj]
        for j2 in T.serial(0, 16):
            for k3o, k3i in T.grid(4, 4):
                with T.block("E"):
                    ei, ej = T.axis.remap("SS", [i, j2])
                    ek = T.axis.R(16, k3o * 4 + k3i)
                    with T.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = E[ei, ej] + A[ei, ej, ek] + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block("F"):
                    fi, fj = T.axis.remap("SS", [i, j2])
                    fk = T.axis.R(16, k4o * 4 + k4i)
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
        with T.block("C_rf"):
            vk1o, ci, cj, vk1i = T.axis.remap("SSSR", [k1o, i, j1, k1i])
            with T.init():
                C_rf[ci, cj, vk1o] = 0.0
            C_rf[ci, cj, vk1o] = C_rf[ci, cj, vk1o] + A[ci, cj, ((vk1o * 4) + vk1i)]
    for i_1 in T.serial(0, 16):
        for j1_1 in T.serial(0, 16):
            for k1o_1 in T.serial(0, 4):
                with T.block("C"):
                    vk1o_1, ci_1, cj_1 = T.axis.remap("RSS", [k1o_1, i_1, j1_1])
                    with T.init():
                        C[ci_1, cj_1] = 0.0
                    C[ci_1, cj_1] = C[ci_1, cj_1] + C_rf[ci_1, cj_1, vk1o_1]
            for k2o, k2i in T.grid(4, 4):
                with T.block("D"):
                    di, dj = T.axis.remap("SS", [i_1, j1_1])
                    dk = T.axis.R(16, k2o * 4 + k2i)
                    with T.init():
                        D[di, dj] = 0.0
                    D[di, dj] = (D[di, dj] + A[di, dj, dk]) + C[di, dj]
        for j2 in T.serial(0, 16):
            for k3o, k3i in T.grid(4, 4):
                with T.block("E"):
                    ei, ej = T.axis.remap("SS", [i_1, j2])
                    ek = T.axis.R(16, k3o * 4 + k3i)
                    with T.init():
                        E[ei, ej] = 0.0
                    E[ei, ej] = (E[ei, ej] + A[ei, ej, ek]) + D[ei, ej]
            for k4o, k4i in T.grid(4, 4):
                with T.block("F"):
                    fi, fj = T.axis.remap("SS", [i_1, j2])
                    fk = T.axis.R(16, k4o * 4 + k4i)
                    with T.init():
                        F[fi, fj] = 0.0
                    F[fi, fj] = (F[fi, fj] + A[fi, fj, fk]) + E[fi, fj]


@T.prim_func
def rfactor_spatial_only(
    A: T.Buffer((1, 512, 7, 7), "float32"),
    B: T.Buffer((1, 512, 1, 1), "float32"),
) -> None:
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc"):
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            rv0 = T.axis.reduce(7, i4 // 7)
            rv1 = T.axis.reduce(7, i4 % 7)
            T.reads(A[ax0, ax1, ax2 * 7 + rv0, ax3 * 7 + rv1])
            T.writes(B[ax0, ax1, ax2, ax3])
            with T.init():
                B[ax0, ax1, ax2, ax3] = T.float32(0)
            B[ax0, ax1, ax2, ax3] = (
                B[ax0, ax1, ax2, ax3] + A[ax0, ax1, ax2 * 7 + rv0, ax3 * 7 + rv1]
            )


@T.prim_func
def rfactor_spatial_only_after(
    A: T.Buffer((1, 512, 7, 7), "float32"),
    B: T.Buffer((1, 512, 1, 1), "float32"),
) -> None:
    # body
    # with T.block("root")
    B_rf = T.alloc_buffer([1, 512, 1, 1, 49], dtype="float32")
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc_rf"):
            vi4 = T.axis.spatial(49, i4)
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            B_rf[ax0, ax1, ax2, ax3, vi4] = A[ax0, ax1, ax2 * 7 + vi4 // 7, ax3 * 7 + vi4 % 7]
    for _i0, i1, _i2, _i3, i4, _i5 in T.grid(1, 512, 1, 1, 49, 1):
        with T.block("acc"):
            vi4 = T.axis.reduce(49, i4)
            ax0 = T.axis.spatial(1, 0)
            ax1 = T.axis.spatial(512, i1)
            ax2 = T.axis.spatial(1, 0)
            ax3 = T.axis.spatial(1, 0)
            with T.init():
                B[ax0, ax1, ax2, ax3] = T.float32(0)
            B[ax0, ax1, ax2, ax3] = B[ax0, ax1, ax2, ax3] + B_rf[ax0, ax1, ax2, ax3, vi4]


@T.prim_func
def argmax_split(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmin_split_init_update_reordered(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmin_v0: T.Buffer((128,), "int32"),
    argmin_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmin"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmin_v0[i], argmin_v1[i])
            with T.init():
                argmin_v1[i] = T.max_value("float32")
                argmin_v0[i] = -1
            v_argmin_v0: T.int32 = T.Select(argmin_v1[i] <= val[i, k], argmin_v0[i], idx[i, k])
            v_argmin_v1: T.float32 = T.Select(argmin_v1[i] <= val[i, k], argmin_v1[i], val[i, k])
            argmin_v1[i] = v_argmin_v1
            argmin_v0[i] = v_argmin_v0


@T.prim_func
def argmax_split_different_shape(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((256,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_different_indices(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i + 1] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i + 1] = v_argmax_v1


@T.prim_func
def argmax_split_init_not_bufferstore(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                v1_init: T.float32 = T.min_value("float32")
                argmax_v1[i] = v1_init
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_init_buffer_duplicate(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v0[i] = -1
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_letstmt_fewer_than_init(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])


@T.prim_func
def argmax_split_letstmt_more_than_init(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_let_body_neither_seqstmt_nor_bufferstore(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            T.evaluate(0)


@T.prim_func
def argmax_split_init_update_inconsistent_bufferstore_number(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_body_seq_not_bufferstore(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            T.evaluate(0)


@T.prim_func
def argmax_split_body_bufferstore_value_not_var(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_body_bufferstore_value_unbound_var(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    v_unbound = T.int32()
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_unbound
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_one_let_var_used_multi_times(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "int32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "int32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("int32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v0


@T.prim_func
def argmax_split_body_one_buffer_updated_multi_times(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "int32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "int32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("int32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v0[i] = v_argmax_v1


@T.prim_func
def argmax_split_init_buffer_not_match(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v0_1: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax"):
            i = T.axis.spatial(128, i0)
            k = T.axis.reduce(128, i1_0 * 32 + i1_1)
            T.reads(idx[i, k], val[i, k])
            T.writes(argmax_v0[i], argmax_v0_1[i], argmax_v1[i])
            with T.init():
                argmax_v0_1[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v0[i], idx[i, k])
            v_argmax_v1: T.float32 = T.Select(argmax_v1[i] >= val[i, k], argmax_v1[i], val[i, k])
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmax_split_rfactor(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmax_v0: T.Buffer((128,), "int32"),
    argmax_v1: T.Buffer((128,), "float32"),
) -> None:
    argmax_v0_rf = T.alloc_buffer([128, 32], dtype="int32")
    argmax_v1_rf = T.alloc_buffer([128, 32], dtype="float32")
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmax_rf"):
            vi1_1, i, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
            T.reads(idx[i, vi1_0 * 32 + vi1_1], val[i, vi1_0 * 32 + vi1_1])
            T.writes(argmax_v0_rf[i, vi1_1], argmax_v1_rf[i, vi1_1])
            with T.init():
                argmax_v0_rf[i, vi1_1] = -1
                argmax_v1_rf[i, vi1_1] = T.min_value("float32")
            v_argmax_v0_rf: T.int32 = T.Select(
                argmax_v1_rf[i, vi1_1] >= val[i, vi1_0 * 32 + vi1_1],
                argmax_v0_rf[i, vi1_1],
                idx[i, vi1_0 * 32 + vi1_1],
            )
            v_argmax_v1_rf: T.float32 = T.Select(
                argmax_v1_rf[i, vi1_1] >= val[i, vi1_0 * 32 + vi1_1],
                argmax_v1_rf[i, vi1_1],
                val[i, vi1_0 * 32 + vi1_1],
            )
            argmax_v0_rf[i, vi1_1] = v_argmax_v0_rf
            argmax_v1_rf[i, vi1_1] = v_argmax_v1_rf
    for i0, i1_1 in T.grid(128, 32):
        with T.block("argmax"):
            vi1_1, i = T.axis.remap("RS", [i1_1, i0])
            T.reads(argmax_v0_rf[i, vi1_1], argmax_v1_rf[i, vi1_1])
            T.writes(argmax_v0[i], argmax_v1[i])
            with T.init():
                argmax_v0[i] = -1
                argmax_v1[i] = T.min_value("float32")
            v_argmax_v0: T.int32 = T.Select(
                argmax_v1[i] >= argmax_v1_rf[i, vi1_1], argmax_v0[i], argmax_v0_rf[i, vi1_1]
            )
            v_argmax_v1: T.float32 = T.Select(
                argmax_v1[i] >= argmax_v1_rf[i, vi1_1], argmax_v1[i], argmax_v1_rf[i, vi1_1]
            )
            argmax_v0[i] = v_argmax_v0
            argmax_v1[i] = v_argmax_v1


@T.prim_func
def argmin_split_rfactor(
    idx: T.Buffer((128, 128), "int32"),
    val: T.Buffer((128, 128), "float32"),
    argmin_v0: T.Buffer((128,), "int32"),
    argmin_v1: T.Buffer((128,), "float32"),
) -> None:
    argmin_v0_rf = T.alloc_buffer([128, 32], dtype="int32")
    argmin_v1_rf = T.alloc_buffer([128, 32], dtype="float32")
    for i0, i1_0, i1_1 in T.grid(128, 4, 32):
        with T.block("argmin_rf"):
            vi1_1, i, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
            T.reads(idx[i, vi1_0 * 32 + vi1_1], val[i, vi1_0 * 32 + vi1_1])
            T.writes(argmin_v0_rf[i, vi1_1], argmin_v1_rf[i, vi1_1])
            with T.init():
                argmin_v0_rf[i, vi1_1] = -1
                argmin_v1_rf[i, vi1_1] = T.max_value("float32")
            v_argmin_v0_rf: T.int32 = T.Select(
                argmin_v1_rf[i, vi1_1] <= val[i, vi1_0 * 32 + vi1_1],
                argmin_v0_rf[i, vi1_1],
                idx[i, vi1_0 * 32 + vi1_1],
            )
            v_argmin_v1_rf: T.float32 = T.Select(
                argmin_v1_rf[i, vi1_1] <= val[i, vi1_0 * 32 + vi1_1],
                argmin_v1_rf[i, vi1_1],
                val[i, vi1_0 * 32 + vi1_1],
            )
            argmin_v0_rf[i, vi1_1] = v_argmin_v0_rf
            argmin_v1_rf[i, vi1_1] = v_argmin_v1_rf
    for i0, i1_1 in T.grid(128, 32):
        with T.block("argmin"):
            vi1_1, i = T.axis.remap("RS", [i1_1, i0])
            T.reads(argmin_v0_rf[i, vi1_1], argmin_v1_rf[i, vi1_1])
            T.writes(argmin_v0[i], argmin_v1[i])
            with T.init():
                argmin_v0[i] = -1
                argmin_v1[i] = T.max_value("float32")
            v_argmin_v0: T.int32 = T.Select(
                argmin_v1[i] <= argmin_v1_rf[i, vi1_1], argmin_v0[i], argmin_v0_rf[i, vi1_1]
            )
            v_argmin_v1: T.float32 = T.Select(
                argmin_v1[i] <= argmin_v1_rf[i, vi1_1], argmin_v1[i], argmin_v1_rf[i, vi1_1]
            )
            argmin_v0[i] = v_argmin_v0
            argmin_v1[i] = v_argmin_v1


@T.prim_func
def argmax_topi_rfactor(
    placeholder: T.Buffer((1, 32), "int32"), placeholder_red: T.Buffer(1, "int32")
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    placeholder_red_temp_v0 = T.alloc_buffer([1], dtype="int32")
    placeholder_red_temp_v1 = T.alloc_buffer([1], dtype="int32")
    placeholder_red_temp_v0_rf = T.alloc_buffer([1, 8], dtype="int32")
    placeholder_red_temp_v1_rf = T.alloc_buffer([1, 8], dtype="int32")
    for i0, i1_0, i1_1 in T.grid(1, 4, 8):
        with T.block("placeholder_red_temp_rf"):
            vi1_1, ax0, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
            T.reads(placeholder[ax0, vi1_0 * 8 + vi1_1])
            T.writes(placeholder_red_temp_v0_rf[ax0, vi1_1], placeholder_red_temp_v1_rf[ax0, vi1_1])
            with T.init():
                placeholder_red_temp_v0_rf[ax0, vi1_1] = -1
                placeholder_red_temp_v1_rf[ax0, vi1_1] = -2147483648
            v_placeholder_red_temp_v0_rf: T.int32 = T.Select(
                placeholder_red_temp_v1_rf[ax0, vi1_1] > placeholder[ax0, vi1_0 * 8 + vi1_1]
                or placeholder_red_temp_v1_rf[ax0, vi1_1] == placeholder[ax0, vi1_0 * 8 + vi1_1]
                and placeholder_red_temp_v0_rf[ax0, vi1_1] < vi1_0 * 8 + vi1_1,
                placeholder_red_temp_v0_rf[ax0, vi1_1],
                vi1_0 * 8 + vi1_1,
            )
            v_placeholder_red_temp_v1_rf: T.int32 = T.Select(
                placeholder_red_temp_v1_rf[ax0, vi1_1] > placeholder[ax0, vi1_0 * 8 + vi1_1],
                placeholder_red_temp_v1_rf[ax0, vi1_1],
                placeholder[ax0, vi1_0 * 8 + vi1_1],
            )
            placeholder_red_temp_v0_rf[ax0, vi1_1] = v_placeholder_red_temp_v0_rf
            placeholder_red_temp_v1_rf[ax0, vi1_1] = v_placeholder_red_temp_v1_rf
    for i0, i1_1 in T.grid(1, 8):
        with T.block("placeholder_red_temp"):
            vi1_1, ax0 = T.axis.remap("RS", [i1_1, i0])
            T.reads(placeholder_red_temp_v0_rf[ax0, vi1_1], placeholder_red_temp_v1_rf[ax0, vi1_1])
            T.writes(placeholder_red_temp_v0[ax0], placeholder_red_temp_v1[ax0])
            with T.init():
                placeholder_red_temp_v0[ax0] = -1
                placeholder_red_temp_v1[ax0] = -2147483648
            v_placeholder_red_temp_v0: T.int32 = T.Select(
                placeholder_red_temp_v1[ax0] > placeholder_red_temp_v1_rf[ax0, vi1_1]
                or placeholder_red_temp_v1[ax0] == placeholder_red_temp_v1_rf[ax0, vi1_1]
                and placeholder_red_temp_v0[ax0] < placeholder_red_temp_v0_rf[ax0, vi1_1],
                placeholder_red_temp_v0[ax0],
                placeholder_red_temp_v0_rf[ax0, vi1_1],
            )
            v_placeholder_red_temp_v1: T.int32 = T.Select(
                placeholder_red_temp_v1[ax0] > placeholder_red_temp_v1_rf[ax0, vi1_1],
                placeholder_red_temp_v1[ax0],
                placeholder_red_temp_v1_rf[ax0, vi1_1],
            )
            placeholder_red_temp_v0[ax0] = v_placeholder_red_temp_v0
            placeholder_red_temp_v1[ax0] = v_placeholder_red_temp_v1
    for i0 in T.serial(1):
        with T.block("placeholder_red"):
            ax0 = T.axis.spatial(1, i0)
            T.reads(placeholder_red_temp_v0[ax0])
            T.writes(placeholder_red[ax0])
            placeholder_red[ax0] = placeholder_red_temp_v0[ax0]


@T.prim_func
def argmin_topi_rfactor(
    placeholder: T.Buffer((1, 32), "int32"), placeholder_red: T.Buffer(1, "int32")
) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    placeholder_red_temp_v0 = T.alloc_buffer([1], dtype="int32")
    placeholder_red_temp_v1 = T.alloc_buffer([1], dtype="int32")
    placeholder_red_temp_v0_rf = T.alloc_buffer([1, 8], dtype="int32")
    placeholder_red_temp_v1_rf = T.alloc_buffer([1, 8], dtype="int32")
    for i0, i1_0, i1_1 in T.grid(1, 4, 8):
        with T.block("placeholder_red_temp_rf"):
            vi1_1, ax0, vi1_0 = T.axis.remap("SSR", [i1_1, i0, i1_0])
            T.reads(placeholder[ax0, vi1_0 * 8 + vi1_1])
            T.writes(placeholder_red_temp_v0_rf[ax0, vi1_1], placeholder_red_temp_v1_rf[ax0, vi1_1])
            with T.init():
                placeholder_red_temp_v0_rf[ax0, vi1_1] = -1
                placeholder_red_temp_v1_rf[ax0, vi1_1] = 2147483647
            v_placeholder_red_temp_v0_rf: T.int32 = T.Select(
                placeholder_red_temp_v1_rf[ax0, vi1_1] < placeholder[ax0, vi1_0 * 8 + vi1_1]
                or placeholder_red_temp_v1_rf[ax0, vi1_1] == placeholder[ax0, vi1_0 * 8 + vi1_1]
                and placeholder_red_temp_v0_rf[ax0, vi1_1] < vi1_0 * 8 + vi1_1,
                placeholder_red_temp_v0_rf[ax0, vi1_1],
                vi1_0 * 8 + vi1_1,
            )
            v_placeholder_red_temp_v1_rf: T.int32 = T.Select(
                placeholder_red_temp_v1_rf[ax0, vi1_1] < placeholder[ax0, vi1_0 * 8 + vi1_1],
                placeholder_red_temp_v1_rf[ax0, vi1_1],
                placeholder[ax0, vi1_0 * 8 + vi1_1],
            )
            placeholder_red_temp_v0_rf[ax0, vi1_1] = v_placeholder_red_temp_v0_rf
            placeholder_red_temp_v1_rf[ax0, vi1_1] = v_placeholder_red_temp_v1_rf
    for i0, i1_1 in T.grid(1, 8):
        with T.block("placeholder_red_temp"):
            vi1_1, ax0 = T.axis.remap("RS", [i1_1, i0])
            T.reads(placeholder_red_temp_v0_rf[ax0, vi1_1], placeholder_red_temp_v1_rf[ax0, vi1_1])
            T.writes(placeholder_red_temp_v0[ax0], placeholder_red_temp_v1[ax0])
            with T.init():
                placeholder_red_temp_v0[ax0] = -1
                placeholder_red_temp_v1[ax0] = 2147483647
            v_placeholder_red_temp_v0: T.int32 = T.Select(
                placeholder_red_temp_v1[ax0] < placeholder_red_temp_v1_rf[ax0, vi1_1]
                or placeholder_red_temp_v1[ax0] == placeholder_red_temp_v1_rf[ax0, vi1_1]
                and placeholder_red_temp_v0[ax0] < placeholder_red_temp_v0_rf[ax0, vi1_1],
                placeholder_red_temp_v0[ax0],
                placeholder_red_temp_v0_rf[ax0, vi1_1],
            )
            v_placeholder_red_temp_v1: T.int32 = T.Select(
                placeholder_red_temp_v1[ax0] < placeholder_red_temp_v1_rf[ax0, vi1_1],
                placeholder_red_temp_v1[ax0],
                placeholder_red_temp_v1_rf[ax0, vi1_1],
            )
            placeholder_red_temp_v0[ax0] = v_placeholder_red_temp_v0
            placeholder_red_temp_v1[ax0] = v_placeholder_red_temp_v1
    for i0 in T.serial(1):
        with T.block("placeholder_red"):
            ax0 = T.axis.spatial(1, i0)
            T.reads(placeholder_red_temp_v0[ax0])
            T.writes(placeholder_red[ax0])
            placeholder_red[ax0] = placeholder_red_temp_v0[ax0]


# pylint: enable=no-member,invalid-name,unused-variable,unexpected-keyword-arg


def test_reduction_rfactor_matmul():
    s = tir.Schedule(transformed_matmul, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, 0)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul)


def test_reduction_rfactor_matmul_with_let():
    s = tir.Schedule(transformed_matmul_with_let, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, 0)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], matmul_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=transformed_matmul_with_let)


def test_reduction_rfactor_square_sum():
    s = tir.Schedule(square_sum, debug_mask="all")
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], square_sum_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=square_sum)


def test_reduction_rfactor_square_sum_square_root():
    s = tir.Schedule(transformed_square_sum_square_root, debug_mask="all")
    C = s.get_block("C")
    _, _, f_i = s.get_loops(C)
    rf_block = s.rfactor(f_i, 0)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], square_sum_square_root_rfactor)
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
    assert_structural_equal_ignore_global_symbol(s.mod["main"], matmul_rfactor)
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


def test_reduction_rfactor_init_not_bufferstore():
    s = tir.Schedule(rowsum_init_not_bufferstore, debug_mask="all")
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
    assert_structural_equal_ignore_global_symbol(s.mod["main"], rowsum_zero_dim_rfactor)
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
    assert_structural_equal_ignore_global_symbol(s.mod["main"], multiple_reduction_blocks_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=multiple_reduction_blocks)


def test_reduction_rfactor_predicate():  # pylint: disable=invalid-name
    s = tir.Schedule(rowsum_predicate, debug_mask="all")
    B = s.get_block("B")
    _, ko, _ = s.get_loops(B)
    # TODO: should be a tvm.tir.ScheduleError
    with pytest.raises(tvm.TVMError):
        rf_block = s.rfactor(ko, 1)


def test_reduction_rfactor_with_annotation():
    s = tir.Schedule(square_sum_with_annotation, debug_mask="all")
    C = s.get_block("C")
    _, _, j = s.get_loops(C)
    rf_block = s.rfactor(j, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], square_sum_with_annotation_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("C_rf")))
    assert s.get(C).same_as(s.get(s.get_block("C")))
    verify_trace_roundtrip(s, mod=square_sum_with_annotation)


def test_reduction_rfactor_spatial_only():
    s = tir.Schedule(rfactor_spatial_only, debug_mask="all")
    block = s.get_block(name="acc", func_name="main")
    _, _, _, _, loop, _ = s.get_loops(block)
    rf_block = s.rfactor(loop=loop, factor_axis=4)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], rfactor_spatial_only_after)
    assert s.get(rf_block).same_as(s.get(s.get_block("acc_rf")))
    assert s.get(block).same_as(s.get(s.get_block("acc")))
    verify_trace_roundtrip(s, mod=rfactor_spatial_only)


def test_reduction_rfactor_argmax():
    s = tir.Schedule(argmax_split, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    rf_block = s.rfactor(ki, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], argmax_split_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("argmax_rf")))
    assert s.get(argmax).same_as(s.get(s.get_block("argmax")))
    verify_trace_roundtrip(s, mod=argmax_split)


def test_reduction_rfactor_argmin_init_update_reordeded():
    s = tir.Schedule(argmin_split_init_update_reordered, debug_mask="all")
    argmin = s.get_block("argmin")
    _, _, ki = s.get_loops(argmin)
    rf_block = s.rfactor(ki, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], argmin_split_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("argmin_rf")))
    assert s.get(argmin).same_as(s.get(s.get_block("argmin")))
    verify_trace_roundtrip(s, mod=argmin_split_init_update_reordered)


def test_reduction_rfactor_argmax_reduction_buffer_different_shape():
    s = tir.Schedule(argmax_split_different_shape, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_different_access_indices():
    s = tir.Schedule(argmax_split_different_indices, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_init_not_bufferstore():
    s = tir.Schedule(argmax_split_init_not_bufferstore, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_init_buffer_duplicate():
    s = tir.Schedule(argmax_split_init_buffer_duplicate, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_letstmt_fewer_than_init():
    s = tir.Schedule(argmax_split_letstmt_fewer_than_init, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_letstmt_more_than_init():
    s = tir.Schedule(argmax_split_letstmt_more_than_init, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_let_body_neither_seqstmt_nor_bufferstore():
    s = tir.Schedule(argmax_split_let_body_neither_seqstmt_nor_bufferstore, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_init_update_inconsistent_bufferstore_number():
    s = tir.Schedule(argmax_split_init_update_inconsistent_bufferstore_number, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_body_seq_not_bufferstore():
    s = tir.Schedule(argmax_split_body_seq_not_bufferstore, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_body_bufferstore_value_not_var():
    s = tir.Schedule(argmax_split_body_bufferstore_value_not_var, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_body_bufferstore_value_unbound_var():
    s = tir.Schedule(argmax_split_body_bufferstore_value_unbound_var, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_one_let_var_used_multi_times():
    s = tir.Schedule(argmax_split_one_let_var_used_multi_times, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_body_one_buffer_updated_multi_times():
    s = tir.Schedule(argmax_split_body_one_buffer_updated_multi_times, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_argmax_init_buffer_not_match():
    s = tir.Schedule(argmax_split_init_buffer_not_match, debug_mask="all")
    argmax = s.get_block("argmax")
    _, _, ki = s.get_loops(argmax)
    with pytest.raises(tvm.tir.ScheduleError):
        s.rfactor(ki, 1)


def test_reduction_rfactor_topi_argmax():
    A = te.placeholder((1, 32), dtype="int32")
    B = topi.argmax(A, axis=1)
    argmax_topi = te.create_prim_func([A, B])
    s = tir.Schedule(argmax_topi, debug_mask="all")
    argmax = s.get_block("placeholder_red_temp")
    _, k = s.get_loops(argmax)
    _, ki = s.split(k, [None, 8])
    rf_block = s.rfactor(ki, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], argmax_topi_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("placeholder_red_temp_rf")))
    assert s.get(argmax).same_as(s.get(s.get_block("placeholder_red_temp")))
    verify_trace_roundtrip(s, mod=argmax_topi)


def test_reduction_rfactor_topi_argmin():
    A = te.placeholder((1, 32), dtype="int32")
    B = topi.argmin(A, axis=1)
    argmin_topi = te.create_prim_func([A, B])
    s = tir.Schedule(argmin_topi, debug_mask="all")
    argmin = s.get_block("placeholder_red_temp")
    _, k = s.get_loops(argmin)
    _, ki = s.split(k, [None, 8])
    rf_block = s.rfactor(ki, 1)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], argmin_topi_rfactor)
    assert s.get(rf_block).same_as(s.get(s.get_block("placeholder_red_temp_rf")))
    assert s.get(argmin).same_as(s.get(s.get_block("placeholder_red_temp")))
    verify_trace_roundtrip(s, mod=argmin_topi)


def test_reduction_rfactor_int64():
    # fmt: off
    @T.prim_func
    def before(
        A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        C: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ):
        for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(
            T.int64(128), T.int64(128), T.int64(4), T.int64(8), T.int64(4)
        ):
            with T.block("update"):
                vi, vj = T.axis.remap("SS", [i0, i1])
                vk = T.axis.R(
                    T.int64(128),
                    i2_outer * T.int64(32) + i2_inner_outer * T.int64(4) + i2_inner_inner,
                )
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])

    @T.prim_func
    def expected(A: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        B: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        C: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ):
        C_rf = T.alloc_buffer((T.int64(4), T.int64(128), T.int64(128)), "float32")

        for i0, i1, i2_outer, i2_inner_outer, i2_inner_inner in T.grid(T.int64(128), T.int64(128), T.int64(4), T.int64(8), T.int64(4)):
            with T.block("update_rf"):
                vi2_inner_inner, vi, vj, vi2_outer, vi2_inner_outer= T.axis.remap("SSSRR", [i2_inner_inner, i0, i1, i2_outer, i2_inner_outer])
                with T.init():
                    C_rf[vi2_inner_inner, vi, vj] = 0.0
                C_rf[vi2_inner_inner, vi, vj] = C_rf[vi2_inner_inner, vi, vj] + (
                    A[vi, (((vi2_outer * T.int64(32)) + (vi2_inner_outer * T.int64(4))) + vi2_inner_inner)]
                    * B[vj, (((vi2_outer * T.int64(32)) + (vi2_inner_outer * T.int64(4))) + vi2_inner_inner)]
                )

        for i0_1, i1_1, i2_inner_inner_1 in T.grid(T.int64(128), T.int64(128), T.int64(4)):
            with T.block("update"):
                vi2_inner_inner_1, vi_1, vj_1 = T.axis.remap("RSS", [i2_inner_inner_1, i0_1, i1_1])
                with T.init():
                    C[vi_1, vj_1] = 0.0
                C[vi_1, vj_1] = C[vi_1, vj_1] + C_rf[vi2_inner_inner_1, vi_1, vj_1]
    # fmt: on

    s = tir.Schedule(before, debug_mask="all")
    update = s.get_block("update")
    _, _, _, _, kii = s.get_loops(update)
    rf_block = s.rfactor(kii, 0)
    assert_structural_equal_ignore_global_symbol(s.mod["main"], expected)
    assert s.get(rf_block).same_as(s.get(s.get_block("update_rf")))
    assert s.get(update).same_as(s.get(s.get_block("update")))
    verify_trace_roundtrip(s, mod=before)


if __name__ == "__main__":
    tvm.testing.main()
