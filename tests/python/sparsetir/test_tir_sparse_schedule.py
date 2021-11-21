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
from os import replace
from numpy.core.fromnumeric import size
from scipy.sparse import bsr
import pytest
import tvm
import tvm.testing
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T


@T.prim_func
def csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), K), m * k, "float32")
    C = T.match_sparse_buffer(c, (I, K), n * k, "float32")
    with T.iter([T.cord(I), T.pos(J), T.cord(K)], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def lowered_csrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    k: T.int32,
    nnz: T.int32,
) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [m * k], dtype="float32")
    C_data = T.match_buffer(c, [n * k], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, n):
        for v_vj, v_vk in T.grid(J_indptr[v_vi + 1] - J_indptr[v_vi], k):
            with T.block("csrmm"):
                vi, vj, vk = T.axis.remap("SRS", [v_vi, v_vj, v_vk])
                T.reads(
                    [
                        J_indptr[0 : n + 1],
                        J_indices[0:nnz],
                        A_data[0:nnz],
                        B_data[0 : m * k],
                        C_data[0 : n * k],
                    ]
                )
                T.writes([C_data[0 : n * k]])
                with T.init():
                    C_data[vi * k + vk] = T.float32(0)
                C_data[vi * k + vk] = (
                    C_data[vi * k + vk]
                    + A_data[J_indptr[vi] + vj] * B_data[J_indices[J_indptr[vi] + vj] * k + vk]
                )


@T.prim_func
def csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(n)
    J = T.sparse_variable((m, n + 1, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (I,), n, "float32")
    with T.iter([T.cord(I), T.pos(J)], "SR", "csr_reduce") as [vi, vj]:
        with T.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vj]


@T.prim_func
def lowered_csr_reduce(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    n: T.int32,
    m: T.int32,
    nnz: T.int32,
) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [n], dtype="float32")
    J_indptr = T.match_buffer(indptr, [n + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, n):
        for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
            with T.block("csr_reduce"):
                vi, vj = T.axis.remap("SR", [v_vi, v_vj])
                T.reads([J_indptr[0 : n + 1], J_indices[0:nnz], A_data[0:nnz], B_data[0:n]])
                T.writes([B_data[0:n]])
                with T.init():
                    B_data[vi] = T.float32(0)
                B_data[vi] = B_data[vi] + A_data[J_indptr[vi] + vj]


@T.prim_func
def bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    I = T.dense_fixed(nb)
    J = T.sparse_variable((mb, nb + 1, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), nnzb * blk * blk, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), BJ, F), mb * blk * feat_size, "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), nb * blk * feat_size, "float32")

    with T.iter([T.cord(I), T.pos(J), T.cord(BI), T.cord(BJ), T.cord(F)], "SRSRS", "bsrmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def lowered_bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    A_data = T.match_buffer(a, [nnzb * blk * blk], dtype="float32")
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32")
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32")
    J_indptr = T.match_buffer(indptr, [nb + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnzb], dtype="int32")
    for v_vi in T.serial(0, nb):
        for v_vj, v_vbi, v_vbj, v_vf in T.grid(
            J_indptr[v_vi + 1] - J_indptr[v_vi], blk, blk, feat_size
        ):
            with T.block("bsrmm"):
                vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [v_vi, v_vj, v_vbi, v_vbj, v_vf])
                T.reads(
                    [
                        J_indptr[0 : nb + 1],
                        J_indices[0:nnzb],
                        A_data[0 : nnzb * blk * blk],
                        B_data[0 : mb * blk * feat_size],
                        C_data[0 : nb * blk * feat_size],
                    ]
                )
                T.writes([C_data[0 : nb * blk * feat_size]])
                with T.init():
                    C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
                C_data[(vi * blk + vbi) * feat_size + vf] = (
                    C_data[(vi * blk + vbi) * feat_size + vf]
                    + A_data[((J_indptr[vi] + vj) * blk + vbi) * blk + vbj]
                    * B_data[(J_indices[J_indptr[vi] + vj] * blk + vbj) * feat_size + vf]
                )


@T.prim_func
def ellpack_mm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    I = T.dense_fixed(nb)
    J = T.sparse_fixed((mb, nnz, col), indices, "int32")
    F = T.dense_fixed(feat_size)
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), nnz * blk * blk, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), BJ, F), mb * blk * feat_size, "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), nb * blk * feat_size, "float32")

    with T.iter([T.cord(I), T.pos(J), T.cord(BI), T.cord(BJ), T.cord(F)], "SRSRS", "bsrmm") as [
        vi,
        vj,
        vbi,
        vbj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = 0.0
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


@T.prim_func
def lowered_ellpack_mm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    feat_size: T.int32,
    nnz: T.int32,
    col: T.int32,
    blk: T.int32,
) -> None:
    A_data = T.match_buffer(a, [nnz * blk * blk], dtype="float32")
    B_data = T.match_buffer(b, [mb * blk * feat_size], dtype="float32")
    C_data = T.match_buffer(c, [nb * blk * feat_size], dtype="float32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi, v_vj, v_vbi, v_vbj, v_vf in T.grid(nb, col, blk, blk, feat_size):
        with T.block("bsrmm"):
            vi, vj, vbi, vbj, vf = T.axis.remap("SRSRS", [v_vi, v_vj, v_vbi, v_vbj, v_vf])
            T.reads(
                [
                    J_indices[0:nnz],
                    A_data[0 : nnz * blk * blk],
                    B_data[0 : mb * blk * feat_size],
                    C_data[0 : nb * blk * feat_size],
                ]
            )
            T.writes([C_data[0 : nb * blk * feat_size]])
            with T.init():
                C_data[(vi * blk + vbi) * feat_size + vf] = T.float32(0)
            C_data[(vi * blk + vbi) * feat_size + vf] = (
                C_data[(vi * blk + vbi) * feat_size + vf]
                + A_data[((vi * col + vj) * blk + vbi) * blk + vbj]
                * B_data[(J_indices[vi * col + vj] * blk + vbj) * feat_size + vf]
            )


@T.prim_func
def batch_mm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    i_indptr: T.handle,
    j_a_indptr: T.handle,
    j_b_indptr: T.handle,
    k_b_indptr: T.handle,
    k_c_indptr: T.handle,
    batch: T.int32,
    n_max: T.int32,
    m_max: T.int32,
    k_max: T.int32,
    nnz_ac1: T.int32,
    nnz_b1: T.int32,
    nnz_a2: T.int32,
    nnz_b2: T.int32,
    nnz_c2: T.int32,
) -> None:
    Batch = T.dense_fixed(batch)
    I = T.dense_variable((n_max, batch + 1), i_indptr, "int32")
    J_a = T.dense_variable((m_max, nnz_ac1 + 1), j_a_indptr, "int32")
    J_b = T.dense_variable((m_max, batch + 1), j_b_indptr, "int32")
    K_b = T.dense_variable((k_max, nnz_b1 + 1), k_b_indptr, "int32")
    K_c = T.dense_variable((k_max, nnz_ac1 + 1), k_c_indptr, "int32")
    A = T.match_sparse_buffer(a, (Batch, I, J_a), nnz_a2, "float32")
    B = T.match_sparse_buffer(b, (Batch, J_b, K_b), nnz_b2, "float32")
    C = T.match_sparse_buffer(c, (Batch, I, K_c), nnz_c2, "float32")

    with T.iter([T.cord(Batch), T.cord(I), T.cord(J_a), T.cord(K_b)], "SSSR", "batch_mm") as [
        vb,
        vi,
        vj,
        vk,
    ]:
        with T.init():
            C[vb, vi, vk] = 0.0
        C[vb, vi, vk] = C[vb, vi, vk] + A[vb, vi, vj] * B[vb, vj, vk]


@T.prim_func
def csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable((n, m + 1, nnz), (indptr, indices), "int32")
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (I, J), nnz, "float32")

    with T.iter([T.cord(I), T.pos(J)], "SS", "csr_element_wise") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.5


@T.prim_func
def lowered_csr_element_wise(
    a: T.handle,
    b: T.handle,
    indptr: T.handle,
    indices: T.handle,
    m: T.int32,
    n: T.int32,
    nnz: T.int32,
) -> None:
    A_data = T.match_buffer(a, [nnz], dtype="float32")
    B_data = T.match_buffer(b, [nnz], dtype="float32")
    J_indptr = T.match_buffer(indptr, [m + 1], dtype="int32")
    J_indices = T.match_buffer(indices, [nnz], dtype="int32")
    for v_vi in T.serial(0, m):
        for v_vj in T.serial(0, J_indptr[v_vi + 1] - J_indptr[v_vi]):
            with T.block("csr_element_wise"):
                vi, vj = T.axis.remap("SS", [v_vi, v_vj])
                T.reads([J_indptr[0 : m + 1], J_indices[0:nnz], A_data[0:nnz]])
                T.writes([B_data[0:nnz]])
                B_data[J_indptr[vi] + vj] = A_data[J_indptr[vi] + vj] * T.float32(2.5)


@T.prim_func
def reordered_bsrmm(
    a: T.handle,
    b: T.handle,
    c: T.handle,
    indptr: T.handle,
    indices: T.handle,
    nb: T.int32,
    mb: T.int32,
    nnzb: T.int32,
    blk: T.int32,
    feat_size: T.int32,
) -> None:
    I = T.dense_fixed(nb)
    J = T.sparse_variable((mb, nb + 1, nnzb), (indptr, indices), "int32")
    BI = T.dense_fixed(blk)
    BJ = T.dense_fixed(blk)
    F = T.dense_fixed(feat_size)
    A = T.match_sparse_buffer(a, (I, J, BI, BJ), nnzb * blk * blk, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), BJ, F), mb * blk * feat_size, "float32")
    C = T.match_sparse_buffer(c, (I, BI, F), nb * blk * feat_size, "float32")
    # body
    with T.iter([T.cord(BI), T.cord(BJ), T.cord(I), T.pos(J), T.cord(F)], "SRSRS", "bsrmm") as [
        vbi,
        vbj,
        vi,
        vj,
        vf,
    ]:
        with T.init():
            C[vi, vbi, vf] = T.float32(0)
        C[vi, vbi, vf] = C[vi, vbi, vf] + A[vi, vj, vbi, vbj] * B[vj, vbj, vf]


def test_get_sparse_block():
    sch = tir.Schedule(csrmm, debug_mask="all")
    block_rv = sch.get_sparse_block("csrmm")
    block = sch.get(block_rv)
    assert block.name == "csrmm"
    assert block.same_as(csrmm.body)


def test_reorder():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block_rv = sch.get_sparse_block("bsrmm")
    block = sch.get(block_rv)
    i, j, bi, bj, f = block.sp_iter_vars
    sch.sparse_reorder(block_rv, [bi, bj, i, j, f])
    tvm.ir.assert_structural_equal(sch.mod["main"], reordered_bsrmm, True)


def test_reorder_fail_on_dependency():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block_rv = sch.get_sparse_block("bsrmm")
    block = sch.get(block_rv)
    i, j, bi, bj, f = block.sp_iter_vars
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block_rv, [bi, bj, j, i, f])


def test_reorder_fail_on_new_order_length():
    sch = tir.Schedule(bsrmm, debug_mask="all")
    block_rv = sch.get_sparse_block("bsrmm")
    block = sch.get(block_rv)
    i, j, bi, bj, f = block.sp_iter_vars
    with pytest.raises(tvm.tir.ScheduleError):
        sch.sparse_reorder(block_rv, [bi, bj, i, j])


if __name__ == "__main__":
    test_get_sparse_block()
    test_reorder()
    test_reorder_fail_on_dependency()
    test_reorder_fail_on_new_order_length()
