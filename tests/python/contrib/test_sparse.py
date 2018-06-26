import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))

import tvm
import tvm.contrib.sparse as tvmsp
import numpy as np
from collections import namedtuple

def test_static_tensor():
    dtype = 'float32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvmsp.placeholder(shape=(m, n), name='A', dtype=dtype)
    print(vars(A))
    assert(A.stype == 'csr')
    n = 3
    a = np.maximum(np.random.uniform(size=(n,n)).astype(dtype)-.6, 0.)
    a = tvmsp.array(a, ctx)
    print(a.data.shape)
    A.data = tvm.placeholder(a.data.shape, dtype, name='A_data')
    Ab = tvm.decl_buffer(a.data.shape, dtype, name='A_data')
    binds = {A.data: Ab}
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    print(C.shape)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A.data, C], target, binds=binds)
    print(a)
    c = tvmsp.array(np.zeros((n,n), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(a.data, c.data)
    print('==== output ====')
    print(a.asnumpy()*2.)
    print(c.asnumpy())
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

def test_dynamic_tensor():
    dtype = 'float32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    m = tvm.var('m')
    n = tvm.var('n')
    nr, nc = 3, 5
    A = tvmsp.placeholder(shape=(m, n), name='A', dtype=dtype)
    print(vars(A))
    assert(A.stype == 'csr')
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    print(C.shape)
    s = tvm.create_schedule(C.op)
    a = np.maximum(np.random.uniform(size=(nr, nc)).astype(dtype)-.6, 0.)
    a = tvmsp.array(a, ctx)
    print(a.data.shape)
    Ab = namedtuple('CSRBuffer', ['data', 'indices', 'indptr'])
    Ab.data = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_data')
    Ab.indices = tvm.decl_buffer(a.data.shape, a.data.dtype, name='A_indices')
    binds = {A.data: Ab.data, A.indices: Ab.indices}
    f = tvm.build(s, [m, n, A.data, C], target, binds=binds)
    print(a)
    c = tvmsp.array(np.zeros((nr, nc), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    f(nr, a.data.shape[0], a.data, c.data)
    print('==== output ====')
    print(a.asnumpy()*2.)
    print(c.asnumpy())
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

if __name__ == "__main__":
    test_static_tensor()
    test_dynamic_tensor()

