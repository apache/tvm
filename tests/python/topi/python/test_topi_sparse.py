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
"""Test code for sparse operator"""
import numpy as np
import tvm
from tvm import te
from tvm import topi
import tvm.topi.testing
from tvm.topi.util import get_const_tuple
import tvm.contrib.sparse as tvmsp
from collections import namedtuple
import time
import scipy.sparse as sp
import tvm.testing

_sparse_dense_implement = {
    "generic": (topi.nn.sparse_dense, topi.generic.schedule_sparse_dense),
    "cuda": (topi.cuda.sparse_dense, topi.cuda.schedule_sparse_dense),
    "x86": (topi.nn.sparse_dense, topi.x86.schedule_sparse_dense),
}


def verify_dynamic_csrmv(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = te.var("nr"), te.var("nc"), te.var("n")
    dtype = "float32"
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name="A")
    B = te.placeholder((in_dim, 1), name="B")
    C = te.placeholder((nr,), name="C")
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype) - 0.5, 0.0)
        b_np = np.random.uniform(size=(in_dim, 1)).astype(dtype) - 0.5
        c_np = np.random.uniform(size=(batch,)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0] - 1
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros((_nr, 1), dtype=dtype), ctx)
        assert a.data.dtype == A.data.dtype
        assert a.indices.dtype == A.indices.dtype
        assert a.indptr.dtype == A.indptr.dtype
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmv")
        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    for device in ["llvm"]:
        check_device(device)


def verify_dynamic_csrmm(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = te.var("nr"), te.var("nc"), te.var("n")
    dtype = "float32"
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name="A")
    B = te.placeholder((in_dim, out_dim), name="B")
    C = te.placeholder((nr,), name="C")
    D = topi.sparse.csrmm(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype) - 0.5, 0.0)
        b_np = np.random.uniform(size=(in_dim, out_dim)).astype(dtype) - 0.5
        c_np = np.random.uniform(size=(batch,)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0] - 1
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros((_nr, out_dim), dtype=dtype), ctx)
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmm")

        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-2, atol=1e-2)

    for device in ["llvm"]:
        check_device(device)


def verify_dense_si(batch, in_dim, out_dim, use_bias=True, dtype="float32"):
    nonzeros = te.var("nonzeros")
    A = tvmsp.placeholder(shape=(batch, in_dim), nonzeros=nonzeros, dtype=dtype, name="A")
    B = te.placeholder((out_dim, in_dim), dtype=dtype, name="B")
    C = te.placeholder((out_dim,), dtype=dtype, name="C")
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.0
        a_np = np.maximum(
            mag * (np.random.uniform(size=(batch, in_dim)).astype("float32") - 0.5), 0.0
        ).astype(dtype)
        b_np = (mag * (np.random.uniform(size=(out_dim, in_dim)).astype("float32") - 0.5)).astype(
            dtype
        )
        c_np = (mag * (np.random.uniform(size=(out_dim,)).astype("float32") - 0.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A.data, A.indices, A.indptr, B, C, D], device, name="dense")
        f(a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    check_device("llvm")


def verify_dense_sw(batch, in_dim, out_dim, use_bias=True, dtype="float32"):
    nonzeros = te.var("nonzeros")
    A = te.placeholder((batch, in_dim), dtype=dtype, name="A")
    B = tvmsp.placeholder(shape=(out_dim, in_dim), nonzeros=nonzeros, dtype=dtype, name="B")
    C = te.placeholder((out_dim,), dtype=dtype, name="C")
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.0
        a_np = (mag * (np.random.uniform(size=(batch, in_dim)).astype("float32") - 0.5)).astype(
            dtype
        )
        b_np = np.maximum(
            mag * (np.random.uniform(size=(out_dim, in_dim)).astype("float32") - 0.5), 0.0
        ).astype(dtype)
        c_np = (mag * (np.random.uniform(size=(out_dim,)).astype("float32") - 0.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvm.nd.array(a_np, ctx)
        b = tvmsp.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
        f = tvm.build(s, [A, B.data, B.indices, B.indptr, C, D], device, name="dense")
        f(a, b.data, b.indices, b.indptr, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4, atol=1e-4)

    check_device("llvm")


def test_csrmv():
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=False)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=True)


def test_csrmm():
    M, K, N = 5, 7, 2
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=False)
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=True)


def test_dense_si():
    M, K, N = 3, 5, 2
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="float32")
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="float32")
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="int32")
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="int32")
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="int16")
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="int16")


def test_dense_sw():
    M, K, N = 3, 5, 2
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="float32")
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="float32")
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="int32")
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="int32")
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype="int16")
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype="int16")


def test_dense():
    test_dense_si()
    test_dense_sw()


def test_sparse_dense_csr():
    M, N, K, density = 1, 17, 47, 0.2
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = sp.random(N, K, density=density, format="csr", dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr)
    s = te.create_schedule(Y.op)
    func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
    Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype))
    func(
        tvm.nd.array(X_np),
        tvm.nd.array(W_sp_np.data),
        tvm.nd.array(W_sp_np.indices),
        tvm.nd.array(W_sp_np.indptr),
        Y_tvm,
    )
    tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)


def test_sparse_transpose_csr():
    N, density = 1023, 0.3

    X_sp = sp.random(N, N, density=density, format="csr", dtype="float32")

    X_sp_T = X_sp.transpose()
    X_np_T = X_sp_T.todense()

    X_data = te.placeholder(shape=X_sp.data.shape, dtype=str(X_sp.data.dtype))
    X_indices = te.placeholder(shape=X_sp.indices.shape, dtype=str(X_sp.indices.dtype))
    X_indptr = te.placeholder(shape=X_sp.indptr.shape, dtype=str(X_sp.indptr.dtype))

    X_T_data, X_T_indices, X_T_indptr = topi.nn.sparse_transpose(X_data, X_indices, X_indptr)
    s = te.create_schedule([X_T_data.op, X_T_indices.op, X_T_indptr.op])
    func = tvm.build(s, [X_data, X_indices, X_indptr, X_T_data, X_T_indices, X_T_indptr])

    X_T_data_tvm = tvm.nd.array(np.zeros(X_sp_T.data.shape, dtype=X_sp_T.data.dtype))
    X_T_indices_tvm = tvm.nd.array(np.zeros(X_sp_T.indices.shape, dtype=X_sp_T.indices.dtype))
    X_T_indptr_tvm = tvm.nd.array(np.zeros(X_sp_T.indptr.shape, dtype=X_sp_T.indptr.dtype))

    func(
        tvm.nd.array(X_sp.data),
        tvm.nd.array(X_sp.indices),
        tvm.nd.array(X_sp.indptr),
        X_T_data_tvm,
        X_T_indices_tvm,
        X_T_indptr_tvm,
    )

    X_T_out = sp.csr_matrix(
        (X_T_data_tvm.asnumpy(), X_T_indices_tvm.asnumpy(), X_T_indptr_tvm.asnumpy()), shape=(N, N)
    ).todense()
    tvm.testing.assert_allclose(X_np_T, X_T_out, atol=1e-4, rtol=1e-4)


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools

    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s


def verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu):
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np.dot(W_np.T)
    if use_relu:
        Y_np = np.maximum(Y_np, 0.0)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        fcompute, fschedule = tvm.topi.testing.dispatch(device, _sparse_dense_implement)
        with tvm.target.Target(device):
            Y = fcompute(X, W_data, W_indices, W_indptr)
            if use_relu:
                Y = topi.nn.relu(Y)
            s = fschedule([Y])
            func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
            Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
            func(
                tvm.nd.array(X_np, ctx=ctx),
                tvm.nd.array(W_sp_np.data, ctx=ctx),
                tvm.nd.array(W_sp_np.indices, ctx=ctx),
                tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                Y_tvm,
            )
            tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)

    for device in ["llvm", "cuda"]:
        check_device(device)


@tvm.testing.uses_gpu
def test_sparse_dense_bsr():
    M, N, K, BS_R, BS_C, density = 1, 64, 128, 8, 16, 0.9
    verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu=True)
    verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu=False)


@tvm.testing.uses_gpu
def test_sparse_dense_bsr_randomized():
    for _ in range(20):
        BS_R = np.random.randint(1, 16)
        BS_C = np.random.randint(1, 16)
        M = np.random.randint(1, 32)
        N = int(np.random.randint(1, 16) * BS_R)
        K = int(np.random.randint(1, 16) * BS_C)
        density = np.clip(np.random.random(), 0.1, 0.9)
        X_np = np.random.randn(M, K).astype("float32")
        W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")

        W_np = W_sp_np.todense()
        Y_np = np.array(X_np.dot(W_np.T))

        W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
        W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
        W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
        X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not tvm.testing.device_enabled(device):
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _sparse_dense_implement)
            with tvm.target.Target(device):
                Y = fcompute(X, W_data, W_indices, W_indptr)
                s = fschedule([Y])
                func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
                Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
                func(
                    tvm.nd.array(X_np, ctx=ctx),
                    tvm.nd.array(W_sp_np.data, ctx=ctx),
                    tvm.nd.array(W_sp_np.indices, ctx=ctx),
                    tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                    Y_tvm,
                )
                tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-5, rtol=1e-5)

        for device in ["llvm", "cuda"]:
            check_device(device)


if __name__ == "__main__":
    test_csrmv()
    test_csrmm()
    test_dense()
    test_sparse_dense_csr()
    test_sparse_dense_bsr()
    test_sparse_dense_bsr_randomized()
    test_sparse_transpose_csr()
