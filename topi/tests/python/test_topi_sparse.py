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
    print(a_np)
    print(a.data)
    print(a.indices)
    print(a.indptr)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)
    d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=dtype), ctx)
    A.data = tvm.placeholder(shape=a.data.shape, dtype='float32', name='A_data')
    A.indices = tvm.placeholder(shape=a.indices.shape, dtype='int32', name='A_indices')
    A.indptr = tvm.placeholder(shape=a.indptr.shape, dtype='int32', name='A_indptr')
    D = topi.sparse.csrmv(A, B, C if use_bias else None)
    s = tvm.create_schedule(D.op)
    Ab = namedtuple('CSRBuffer', ['data','indices','indptr'])
    print('A_data', A.data.shape, A.data.dtype)
    print('A_indices', A.indices.shape, A.indices.dtype)
    print('A_indptr', A.indptr.shape, A.indptr.dtype)
    print('B:', B.shape, B.dtype)
    print('C:', C.shape, C.dtype)
    print('D:', D.shape, D.dtype)
    print('b:', b.shape, b.dtype)
    print('c:', c.shape, c.dtype)
    print('d:', d.shape, d.dtype)
    Ab.data = tvm.decl_buffer(A.data.shape, A.data.dtype, name='A_data')
    Ab.indices = tvm.decl_buffer(A.indices.shape, A.indices.dtype, name='A_indices')
    Ab.indptr = tvm.decl_buffer(A.indptr.shape, A.indptr.dtype, name='A_indptr')
    binds = {A.data: Ab.data, A.indices: Ab.indices, A.indptr: Ab.indptr}
    f = tvm.build(s, [A.data, A.indices, A.indptr, B, C, D], device, name="csrmv", binds=binds)
    f(a.data, a.indices, a.indptr, b, c, d)
    print(d.asnumpy().T)
    print(d_np.T)
    np.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

def test_csrmv():
    verify_static_csrmv(batch=3, in_dim=5, out_dim=1, use_bias=False)

if __name__ == "__main__":
    test_csrmv()
