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
import tvm
from tvm.runtime.ndarray import device
import tvm.tir as tir
import scipy.sparse as sp
import numpy as np
from tvm.script import tir as T


@T.prim_func
def csrmm(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, m: T.int32, n: T.int32, k: T.int32, nnz: T.int32) -> None:
    I = T.dense_fixed(m)
    J = T.sparse_variable((n, m + 1, nnz), (indptr, indices), "int32")
    K = T.dense_fixed(k)
    A = T.match_sparse_buffer(a, (I, J), nnz, "float32")
    B = T.match_sparse_buffer(b, (T.to_dense(J), K), n * k, "float32")
    C = T.match_sparse_buffer(c, (I, K), m * k, "float32")
    with T.iter([T.cord(I), T.cord(J), T.cord(K)], "SRS", "csrmm") as [vi, vj, vk]:
        with T.init():
            C[vi, vk] = 0.0
        C[vi, vk] = C[vi, vk] + A[vi, vj] * B[vj, vk]


@T.prim_func
def csrmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, NNZ: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZ,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZ,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.
            for j in T.serial(0, A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    vj = T.axis.R(NNZ, j + A_indptr[vi])
                    C[vi * K + vk] = C[vi * K + vk] + \
                        A_data[vj] * B[A_indices[vj] * K + vk]


@T.prim_func
def bsrmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, MB: T.int32, NB: T.int32, K: T.int32, BLOCK_SIZE: T.int32, NNZB: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (NNZB * BLOCK_SIZE * BLOCK_SIZE), "float32")
    B = T.match_buffer(b, (NB * BLOCK_SIZE * K,), "float32")
    C = T.match_buffer(c, (MB * BLOCK_SIZE * K,), "float32")
    A_indptr = T.match_buffer(indptr, (MB + 1,), "int32")
    A_indices = T.match_buffer(indices, (NNZB,), "int32")
    for io, ii, ji, k in T.grid(MB, BLOCK_SIZE, BLOCK_SIZE, K):
        with T.block("spmm_outer"):
            vio, vii, vji, vk = T.axis.remap("SSSS", [io, ii, ji, k])
            with T.init():
                C[(vio * BLOCK_SIZE + vii) * K + vk] = 0.
            for jo in T.serial(0, A_indptr[vio + 1] - A_indptr[vio]):
                with T.block("spmm_inner"):
                    vjo = T.axis.R(NNZB, jo + A_indptr[vio])
                    C[(vio * BLOCK_SIZE + vii) * K + vk] = C[(vio * BLOCK_SIZE + vii) * K + vk] + A_data[(
                        vjo * BLOCK_SIZE + vii) * BLOCK_SIZE + vji] * B[(A_indices[vjo] * BLOCK_SIZE + vji) * K + vk]


@T.prim_func
def ellmm_tir(a: T.handle, b: T.handle, c: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, NNZ_COLS: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (M * NNZ_COLS,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indices = T.match_buffer(indices, (M * NNZ_COLS,), "int32")
    for i, j, k in T.grid(M, NNZ_COLS, K):
        with T.block("spmm"):
            vi, vj, vk = T.axis.remap("SRS", [i, j, k])
            with T.init():
                C[vi * K + vk] = 0.
            C[vi * K + vk] = C[vi * K + vk] + A_data[vi * NNZ_COLS + vj] * \
                B[A_indices[vi * NNZ_COLS + vj] * K + vk]


def test_csrmm():
    # generate random input
    m = 4096
    n = 4096
    k = 256
    A = sp.random(m, n, dtype="float32", density=0.0125, format='csr')
    nnz = A.nnz
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, M, N, K, NNZ = csrmm_tir.params
    sch = tir.Schedule(
        csrmm_tir.specialize(
            {M: m, N: n, K: k, NNZ: nnz}
        )
    )
    blk_outer = sch.get_block("spmm_outer")
    i, k = sch.get_loops(blk_outer)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(A.indptr.astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(A.indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(A.data.astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target='cuda')
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)

    # assertion
    assert np.allclose(y_ground_truth.reshape(-1), Y_nd.numpy())


def test_bsrmm():
    # generate random input
    block_size = 1
    mb = 64
    nb = 64
    k = 256
    m = mb * block_size
    n = nb * block_size
    A_block = sp.random(mb, nb, dtype="float32", density=0.05, format='csr')
    indptr = A_block.indptr
    indices = A_block.indices
    nnzb = A_block.nnz
    data = np.random.rand(nnzb, block_size, block_size)
    A = sp.bsr_matrix((data, indices, indptr), shape=(m, n))
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")

    # specialize function
    _, _, _, _, _, MB, NB, K, BLOCK_SIZE, NNZB = bsrmm_tir.params
    sch = tir.Schedule(
        bsrmm_tir.specialize(
            {MB: mb, NB: nb, K: k, BLOCK_SIZE: block_size, NNZB: nnzb}
        )
    )
    blk_outer = sch.get_block("spmm_outer")
    io, ii, ji, k = sch.get_loops(blk_outer)
    sch.unroll(ii)
    sch.unroll(ji)
    sch.bind(io, "blockIdx.x")
    sch.bind(k, "threadIdx.x")

    # convert numpy tensor to tvm ndarray
    A_indptr = tvm.nd.array(indptr.astype("int32"), device=tvm.cuda(0))
    A_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(
        data.reshape(-1).astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)

    # assertion
    assert np.allclose(y_ground_truth.reshape(-1), Y_nd.numpy())


def test_ellmm():
    # generate random input
    nnz_cols = 64
    m = 4096
    n = 4096
    k = 256
    nnz = nnz_cols * m
    indptr = np.arange(0, (m + 1) * nnz_cols, nnz_cols)
    indices = np.random.randint(0, n, size=(nnz,))
    data = np.random.rand(nnz)
    A = sp.csr_matrix((data, indices, indptr), shape=(m, n))
    x = np.random.rand(n, k).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((m * k,)).astype("float32")
    # specialize function
    _, _, _, _, M, N, K, NNZ_COLS = ellmm_tir.params
    sch = tir.Schedule(
        ellmm_tir.specialize(
            {M: m, N: n, K: k, NNZ_COLS: nnz_cols}
        )
    )
    blk = sch.get_block("spmm")
    i, j, k = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(k, "threadIdx.x")
    sch.unroll(j)

    # convert numpy tensor to tvm ndarray
    A_indices = tvm.nd.array(indices.astype("int32"), device=tvm.cuda(0))
    A_data = tvm.nd.array(data.astype("float32"), device=tvm.cuda(0))
    X_nd = tvm.nd.array(x.reshape(-1), device=tvm.cuda(0))
    Y_nd = tvm.nd.array(y, device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target="cuda")
    f(A_data, X_nd, Y_nd, A_indices)

    # assertion
    assert np.allclose(y_ground_truth.reshape(-1), Y_nd.numpy())


def test_bmm():
    # TODO(zihao)
    pass


if __name__ == "__main__":
    test_csrmm()
    test_bsrmm()
    test_ellmm()
    test_bmm()
