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

def verify_static_csrmv(batch, in_dim, out_dim, use_bias=True):
    A = tvmsp.placeholder((batch, in_dim), name='A')
    B = tvm.placeholder((in_dim, 1), name='B')
    C = tvm.placeholder((batch, 1), name='C')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, 1)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, 1)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np
        else:
            d_np = np.dot(a_np, b_np)
        return (a_np, b_np, c_np, d_np)
    a_np, b_np, c_np, d_np = get_ref_data()

    device = 'llvm'
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    print("Running on target: %s" % device)
    a = tvmsp.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)
    d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
    A.data = tvm.placeholder(shape=a.data.shape, dtype='float32', name='A_data')
    A.indices = tvm.placeholder(shape=a.indices.shape, dtype='int32', name='A_indices')
    A.indptr = tvm.placeholder(shape=a.indptr.shape, dtype='int32', name='A_indptr')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    Ab = namedtuple('CSRBuffer', ['data','indices','indptr'])
    Ab.data = tvm.decl_buffer(A.data.shape, A.data.dtype, name='A_data')
    Ab.indices = tvm.decl_buffer(A.indices.shape, A.indices.dtype, name='A_indices')
    Ab.indptr = tvm.decl_buffer(A.indptr.shape, A.indptr.dtype, name='A_indptr')
    binds = {A.data: Ab.data, A.indices: Ab.indices, A.indptr: Ab.indptr}
    f = tvm.build(s, [A.data, A.indices, A.indptr, B, C, D], device, name="csrmv", binds=binds)
    f(a.data, a.indices, a.indptr, b, c, d)
    print(d.asnumpy().T)
    print(d_np.T)
    np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

def verify_dynamic_csrmv(batch, in_dim, out_dim, use_bias=True):
    nr, nc, n = tvm.var("nr"), tvm.var("nc"), tvm.var("n")
    dtype = 'float32'
    A = tvmsp.placeholder(shape=(nr, nc), nonzeros=n, dtype=dtype, name='A')
    B = tvm.placeholder((in_dim, 1), name='B')
    C = tvm.placeholder((batch, 1), name='C')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    dtype = A.dtype

    # get the test data
    def get_ref_data():
        a_np = np.maximum(np.random.uniform(size=(batch, in_dim)).astype(dtype)-0.5, 0.)
        b_np = np.random.uniform(size=(in_dim, 1)).astype(dtype)-0.5
        c_np = np.random.uniform(size=(batch, 1)).astype(dtype)
        if use_bias:
            d_np = np.dot(a_np, b_np) + c_np
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
        Ab = namedtuple('CSRBuffer', ['data','indices','indptr'])
        Ab.data = tvm.decl_buffer(A.data.shape, A.data.dtype, name='A_data')
        Ab.indices = tvm.decl_buffer(A.indices.shape, A.indices.dtype, name='A_indices')
        Ab.indptr = tvm.decl_buffer(A.indptr.shape, A.indptr.dtype, name='A_indptr')
        binds = {A.data: Ab.data, A.indices: Ab.indices, A.indptr: Ab.indptr}
        f = tvm.build(s, [nr, A.data, A.indices, A.indptr, B, C, D], device, name="csrmv", binds=binds)
        f(_nr, a.data, a.indices, a.indptr, b, c, d)
        print(d.asnumpy().T)
        print(d_np.T)
        np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in ["llvm"]:
        check_device(device)

def test_csrmv():
    verify_dynamic_csrmv(batch=3, in_dim=5, out_dim=1, use_bias=False)
    verify_static_csrmv(batch=3, in_dim=5, out_dim=1, use_bias=False)

if __name__ == "__main__":
    test_csrmv()
