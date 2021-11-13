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
def csrmm_tir(a: T.handle, b: T.handle, c: T.handle, indptr: T.handle, indices: T.handle, M: T.int32, N: T.int32, K: T.int32, nnz: T.int32) -> None:
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A_data = T.match_buffer(a, (nnz,), "float32")
    B = T.match_buffer(b, (N * K,), "float32")
    C = T.match_buffer(c, (M * K,), "float32")
    A_indptr = T.match_buffer(indptr, (M + 1,), "int32")
    A_indices = T.match_buffer(indices, (nnz,), "int32")
    for i, k in T.grid(M, K):
        with T.block("spmm_outer"):
            vi, vk = T.axis.remap("SS", [i, k])
            with T.init():
                C[vi * K + vk] = 0.
            for j in T.serial(0, A_indptr[vi + 1] - A_indptr[vi]):
                with T.block("spmm_inner"):
                    vj = T.axis.R(N, j + A_indptr[vi]) 
                    C[vi * K + vk] = C[vi * K + vk] + A_data[vj] * B[A_indices[vj] * K + vk]


def test_csrmm():
    # generate random input
    A = sp.random(4096, 4096, dtype="float32", density=0.0125, format='csr')
    x = np.random.rand(4096, 256).astype("float32")
    y_ground_truth = A * x
    y = np.zeros((4096, 256)).astype("float32")

    # specialize function
    _, _, _, _, _, m, n, k, nnz = csrmm_tir.params
    sch = tir.Schedule(
        csrmm_tir.specialize(
            {m: 4096, n: 4096, k: 256, nnz: A.nnz}
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
    Y_nd = tvm.nd.array(y.reshape(-1), device=tvm.cuda(0))

    # build function
    f = tvm.build(sch.mod, target='cuda')
    f(A_data, X_nd, Y_nd, A_indptr, A_indices)

    # assertion
    assert np.allclose(y_ground_truth.reshape(-1), Y_nd.numpy())


if __name__ == "__main__":
    test_csrmm()