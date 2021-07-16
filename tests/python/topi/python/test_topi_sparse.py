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
from tvm import relay
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
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


def verify_dynamic_csrmv(batch, in_dim, out_dim, dtype, use_bias=True):
    nr, nc, n = te.var("nr"), te.var("nc"), te.var("n")
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name="A")
    B = te.placeholder((in_dim, 1), dtype=dtype, name="B")
    C = te.placeholder((nr,), dtype=dtype, name="C")
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim), high=100).astype(dtype)
        b_np = np.random.uniform(size=(in_dim, 1), high=100).astype(dtype)
        c_np = np.random.uniform(size=(batch,), high=100).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, dev)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0] - 1
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(c_np, dev)
        d = tvm.nd.array(np.zeros((_nr, 1), dtype=dtype), dev)
        assert a.data.dtype == A.data.dtype
        assert a.indices.dtype == A.indices.dtype
        assert a.indptr.dtype == A.indptr.dtype
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmv")
        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-4, atol=1e-4)

    for device in ["llvm"]:
        check_device(device)


def verify_dynamic_csrmm(batch, in_dim, out_dim, dtype, use_bias=True):
    nr, nc, n = te.var("nr"), te.var("nc"), te.var("n")
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name="A")
    B = te.placeholder((in_dim, out_dim), dtype=dtype, name="B")
    C = te.placeholder((nr,), dtype=dtype, name="C")
    D = topi.sparse.csrmm(A, B, C if use_bias else None)
    s = te.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim), high=100).astype(dtype)
        b_np = np.random.uniform(size=(in_dim, out_dim), high=100).astype(dtype)
        c_np = np.random.uniform(size=(batch,), high=100).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)

    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, dev)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0] - 1
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(c_np, dev)
        d = tvm.nd.array(np.zeros((_nr, out_dim), dtype=dtype), dev)
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmm")

        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-2, atol=1e-2)

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
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(c_np, dev)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), dev)
        f = tvm.build(s, [A.data, A.indices, A.indptr, B, C, D], device, name="dense")
        f(a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-4, atol=1e-4)

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
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvm.nd.array(a_np, dev)
        b = tvmsp.array(b_np, dev)
        c = tvm.nd.array(c_np, dev)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), dev)
        f = tvm.build(s, [A, B.data, B.indices, B.indptr, C, D], device, name="dense")
        f(a, b.data, b.indices, b.indptr, c, d)
        tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-4, atol=1e-4)

    check_device("llvm")


def test_csrmv():
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, dtype="float32", use_bias=False)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, dtype="float64", use_bias=True)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, dtype="int32", use_bias=True)


def test_csrmm():
    M, K, N = 5, 7, 2
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, dtype="int64", use_bias=False)
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, dtype="float64", use_bias=True)


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
    tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-4, rtol=1e-4)


def test_sparse_dense_csr_reverse():
    M, N, K, density = 1, 17, 47, 0.2
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = sp.random(N, K, density=density, format="csr", dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = W_np.dot(X_np.T)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr, sparse_lhs=True)
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
    tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-4, rtol=1e-4)


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
        (X_T_data_tvm.numpy(), X_T_indices_tvm.numpy(), X_T_indptr_tvm.numpy()), shape=(N, N)
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


def verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, use_relu, device, target):
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = X_np @ W_np.T
    if use_relu:
        Y_np = np.maximum(Y_np, 0.0)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    fcompute, fschedule = tvm.topi.testing.dispatch(target, _sparse_dense_implement)
    with tvm.target.Target(target):
        Y = fcompute(X, W_data, W_indices, W_indptr)
        if use_relu:
            Y = topi.nn.relu(Y)
        s = fschedule([Y])
        func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
        Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), device=device)
        func(
            tvm.nd.array(X_np, device=device),
            tvm.nd.array(W_sp_np.data, device=device),
            tvm.nd.array(W_sp_np.indices, device=device),
            tvm.nd.array(W_sp_np.indptr, device=device),
            Y_tvm,
        )
        tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-4, rtol=1e-4)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_sparse_dense_bsr_relu(dev, target):
    M, N, K, BS_R, BS_C, density = 1, 64, 128, 8, 16, 0.9
    verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, True, dev, target)
    verify_sparse_dense_bsr(M, N, K, BS_R, BS_C, density, False, dev, target)


def test_sparse_dense_bsr_reverse():
    M, N, K, BS_R, BS_C, density = 1, 64, 128, 8, 16, 0.9
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    Y_np = W_np.dot(X_np.T)

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    Y = topi.nn.sparse_dense(X, W_data, W_indices, W_indptr, sparse_lhs=True)
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
    tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-4, rtol=1e-4)


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
            dev = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            fcompute, fschedule = tvm.topi.testing.dispatch(device, _sparse_dense_implement)
            with tvm.target.Target(device):
                Y = fcompute(X, W_data, W_indices, W_indptr)
                s = fschedule([Y])
                func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
                Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), device=dev)
                func(
                    tvm.nd.array(X_np, device=dev),
                    tvm.nd.array(W_sp_np.data, device=dev),
                    tvm.nd.array(W_sp_np.indices, device=dev),
                    tvm.nd.array(W_sp_np.indptr, device=dev),
                    Y_tvm,
                )
                tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-5, rtol=1e-5)

        for device in ["llvm", "cuda"]:
            check_device(device)


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_sparse_dense_padded_gpu(target, dev):
    M = 128
    N = 1280
    K = 128
    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, 1, 1, density=0.01, dtype="float32")
    W_sp_np_padded = tvm.topi.cuda.pad_sparse_matrix(W_sp_np, 32)

    W_np = W_sp_np.todense()
    Y_np = X_np @ W_sp_np.T

    W_data = te.placeholder(shape=W_sp_np_padded.data.shape, dtype=str(W_sp_np_padded.data.dtype))
    W_indices = te.placeholder(
        shape=W_sp_np_padded.indices.shape, dtype=str(W_sp_np_padded.indices.dtype)
    )
    W_indptr = te.placeholder(
        shape=W_sp_np_padded.indptr.shape, dtype=str(W_sp_np_padded.indptr.dtype)
    )
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))
    with tvm.target.Target(target):
        Y = topi.cuda.sparse_dense_padded(X, W_data, W_indices, W_indptr)
        s = topi.cuda.schedule_sparse_dense_padded([Y])
        func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
        Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), device=dev)
        func(
            tvm.nd.array(X_np, device=dev),
            tvm.nd.array(W_sp_np_padded.data, device=dev),
            tvm.nd.array(W_sp_np_padded.indices, device=dev),
            tvm.nd.array(W_sp_np_padded.indptr, device=dev),
            Y_tvm,
        )
        tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np, atol=1e-5, rtol=1e-5)


@tvm.testing.parametrize_targets("cuda", "rocm")
def test_sparse_dense_padded_alter_op(target, dev):
    with tvm.target.Target(target):
        M = 128
        N = 16
        K = 128
        X_np = np.random.randn(M, K).astype("float32")
        W_sp_np = random_bsr_matrix(N, K, 2, 2, density=0.01, dtype="float32")
        x = relay.var("x", relay.TensorType(X_np.shape, "float32"))
        mult = relay.op.nn.sparse_dense(
            x,
            (
                relay.Constant(tvm.nd.array(W_sp_np.data)),
                relay.Constant(tvm.nd.array(W_sp_np.indices)),
                relay.Constant(tvm.nd.array(W_sp_np.indptr)),
            ),
        )
        f = relay.Function([x], mult)
        f_ = relay.transform.InferType()(tvm.IRModule.from_expr(f))
        f_ = relay.transform.AlterOpLayout()(f_)
        assert f_["main"].body.op.name == "nn.internal.sparse_dense_padded"

        # build with cuda and AlterOpLayout to ensure that sparse_dense_padded is in action
        with tvm.transform.PassContext(opt_level=3, required_pass="AlterOpLayout"):
            x = relay.build(tvm.IRModule.from_expr(f), target=target)


def test_sparse_add_csr():
    for indices_dtype in ["int32", "int64"]:
        for data_dtype in ["float32", "float64"]:
            M, K, density = 3, 49, 0.2
            X_np = np.random.randn(M, K).astype(data_dtype)
            Y_sp_np = sp.random(M, K, density=density, format="csr", dtype=data_dtype)
            Y_np = Y_sp_np.todense()
            Z_np = X_np + Y_np

            Y_data = te.placeholder(shape=Y_sp_np.data.shape, dtype=data_dtype)
            Y_indices = te.placeholder(shape=Y_sp_np.indices.shape, dtype=indices_dtype)
            Y_indptr = te.placeholder(shape=Y_sp_np.indptr.shape, dtype=indices_dtype)
            X = te.placeholder(shape=X_np.shape, dtype=data_dtype)
            Z = topi.nn.sparse_add(X, Y_data, Y_indices, Y_indptr)
            s = te.create_schedule(Z.op)
            func = tvm.build(s, [X, Y_data, Y_indices, Y_indptr, Z])
            Z_tvm = tvm.nd.array(np.zeros(Z_np.shape, dtype=Z_np.dtype))
            func(
                tvm.nd.array(X_np.astype(data_dtype)),
                tvm.nd.array(Y_sp_np.data.astype(data_dtype)),
                tvm.nd.array(Y_sp_np.indices.astype(indices_dtype)),
                tvm.nd.array(Y_sp_np.indptr.astype(indices_dtype)),
                Z_tvm,
            )
            tvm.testing.assert_allclose(Z_tvm.numpy(), Z_np, atol=1e-4, rtol=1e-4)


def verify_sparse_conv2d_bsr(M, H, W, N, K, BS_R, BS_C, density, layout):
    if layout == "NHWC":
        X_np = np.random.randn(M, H, W, K).astype("float32")
    elif layout == "NCHW":
        X_np = np.random.randn(M, K, H, W).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_np = W_sp_np.todense()
    if layout == "NHWC":
        Y_np = tvm.topi.testing.conv2d_nhwc_python(X_np, np.array(W_np).T.reshape(1, 1, K, N), 1, 0)
    elif layout == "NCHW":
        Y_np = tvm.topi.testing.conv2d_nchw_python(X_np, np.array(W_np).reshape(N, K, 1, 1), 1, 0)

    if BS_C == 1:
        W_data = te.placeholder(shape=W_sp_np.data.shape[:-1], dtype=str(W_sp_np.data.dtype))
        W_sp_np_data = W_sp_np.data.reshape(W_sp_np.data.shape[0], BS_R)
    else:
        W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
        W_sp_np_data = W_sp_np.data
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    Y = topi.nn.sparse_conv2d(X, W_data, W_indices, W_indptr, layout)
    s = te.create_schedule(Y.op)

    def check_device(device):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
        Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype="float32"))
        func(
            tvm.nd.array(X_np, dev),
            tvm.nd.array(W_sp_np_data, dev),
            tvm.nd.array(W_sp_np.indices, dev),
            tvm.nd.array(W_sp_np.indptr, dev),
            Y_tvm,
        )
        tvm.testing.assert_allclose(Y_tvm.numpy(), Y_np.astype("float32"), atol=1e-4, rtol=1e-4)

    check_device("llvm")


def test_sparse_conv2d_bsr():
    M, H, W, N, K, BS_R, BS_C, density = 1, 32, 32, 128, 64, 8, 16, 0.9
    verify_sparse_conv2d_bsr(M, H, W, N, K, BS_R, BS_C, density, "NHWC")
    verify_sparse_conv2d_bsr(M, H, W, N, K, BS_R, BS_C, density, "NCHW")
    verify_sparse_conv2d_bsr(M, H, W, N, K, BS_R, 1, density, "NHWC")


if __name__ == "__main__":
    # test_csrmv()
    # test_csrmm()
    # test_dense()
    # test_sparse_dense_csr()
    # test_sparse_dense_bsr_randomized()
    # test_sparse_transpose_csr()
    # test_sparse_dense_padded_cuda()
    # test_sparse_dense_padded_alter_op()
    # test_sparse_dense_csr_reverse()
    # test_sparse_dense_bsr_reverse()
    # test_sparse_add_csr()
    test_sparse_conv2d()
