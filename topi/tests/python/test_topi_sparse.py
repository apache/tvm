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
import topi
import topi.testing
from topi.util import get_const_tuple
import tvm.contrib.sparse as tvmsp
from collections import namedtuple
import time

def verify_dynamic_csrmv(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = tvm.var("nr"), tvm.var("nc"), tvm.var("n")
    dtype = 'float32'
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name='A')
    B = tvm.placeholder((in_dim, 1), name='B')
    C = tvm.placeholder((nr,), name='C')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, 1)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, )).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0]-1
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
    nr, nc, n = tvm.var("nr"), tvm.var("nc"), tvm.var("n")
    dtype = 'float32'
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name='A')
    B = tvm.placeholder((in_dim, out_dim), name='B')
    C = tvm.placeholder((nr,), name='C')
    D = topi.sparse.csrmm(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, out_dim)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, )).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np.reshape((batch, 1))
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        a = tvmsp.array(a_np, ctx)
        _nr, _nc, _n = a.shape[0], a.shape[1], a.data.shape[0]
        assert a.shape[0] == a.indptr.shape[0]-1
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros((_nr, out_dim), dtype=dtype), ctx)
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmm")

        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-2, atol=1e-2)

    for device in ["llvm"]:
        check_device(device)

def verify_dense_si(batch, in_dim, out_dim, use_bias=True, dtype='float32'):
    nonzeros = tvm.var('nonzeros')
    A = tvmsp.placeholder(shape=(batch, in_dim), nonzeros=nonzeros, dtype=dtype, name='A')
    B = tvm.placeholder((out_dim, in_dim), dtype=dtype, name='B')
    C = tvm.placeholder((out_dim,), dtype=dtype, name='C')
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.
        a_np = np.maximum(mag*(np.random.uniform(size=(batch, in_dim)).astype('float32')-0.5), 0.).astype(dtype)
        b_np = (mag*(np.random.uniform(size=(out_dim, in_dim)).astype('float32')-.5)).astype(dtype)
        c_np = (mag*(np.random.uniform(size=(out_dim,)).astype('float32')-.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
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

    check_device('llvm')

def verify_dense_sw(batch, in_dim, out_dim, use_bias=True, dtype='float32'):
    nonzeros = tvm.var('nonzeros')
    A = tvm.placeholder((batch, in_dim), dtype=dtype, name='A')
    B = tvmsp.placeholder(shape=(out_dim, in_dim), nonzeros=nonzeros, dtype=dtype, name='B')
    C = tvm.placeholder((out_dim,), dtype=dtype, name='C')
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)

    # get the test data
    def get_ref_data():
        mag = 10.
        a_np = (mag*(np.random.uniform(size=(batch, in_dim)).astype('float32')-.5)).astype(dtype)
        b_np = np.maximum(mag*(np.random.uniform(size=(out_dim, in_dim)).astype('float32')-0.5), 0.).astype(dtype)
        c_np = (mag*(np.random.uniform(size=(out_dim,)).astype('float32')-.5)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np.T) + c_np
        else:
            d_np = np.dot(a_np, b_np.T)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
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

    check_device('llvm')

def test_csrmv():
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=False)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=True)

def test_csrmm():
    M, K, N = 5, 7, 2
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=False)
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=True)

def test_dense_si():
    M, K, N = 3, 5, 2
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='float32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='float32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int32')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int16')
    verify_dense_si(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int16')

def test_dense_sw():
    M, K, N = 3, 5, 2
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='float32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='float32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int32')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=False, dtype='int16')
    verify_dense_sw(batch=M, in_dim=K, out_dim=N, use_bias=True, dtype='int16')

def test_dense():
    test_dense_si()
    test_dense_sw()

if __name__ == "__main__":
    test_csrmv()
    test_csrmm()
    test_dense()
