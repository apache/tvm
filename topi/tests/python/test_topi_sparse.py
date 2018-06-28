"""Test code for sparse operator"""
import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))
sys.path.insert(0, os.path.join(thisdir, '../../python'))

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
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmv")
        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        print(d.asnumpy().T)
        print(d_np.T)
        np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-4)

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
        print(d.asnumpy().T)
        print(d_np.T)
        np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-2)

    for device in ["llvm"]:
        check_device(device)

def test_csrmv():
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=False)
    verify_dynamic_csrmv(batch=5, in_dim=7, out_dim=1, use_bias=True)

def test_csrmm():
    M, K, N = 5, 7, 2
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=False)
    verify_dynamic_csrmm(batch=M, in_dim=K, out_dim=N, use_bias=True)


def verify_dense(batch, in_dim, out_dim, use_bias=True):
    A = tvmsp.placeholder((batch, in_dim), name='A')
    B = tvm.placeholder((out_dim, in_dim), name='B')
    C = tvm.placeholder((out_dim,), name='C')
    D = topi.sparse.dense(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
        c_np = np.random.uniform(size=(out_dim,)).astype(dtype)
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
        np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    check_device('llvm')

def test_dense():
    M, K, N = 5, 7, 2
    verify_dense(batch=M, in_dim=K, out_dim=N, use_bias=False)
    verify_dense(batch=M, in_dim=K, out_dim=N, use_bias=True)

if __name__ == "__main__":
    test_csrmv()
    test_csrmm()
    test_dense()
