import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))

import tvm
import tvm.contrib.sparse as tvmsp
import numpy as np

def test_tensor():
    dtype = 'float32'
    stype = 'csr'
    target = 'llvm'
    ctx = tvm.context(target, 0)
    m = tvm.var('m')
    A = tvmsp.placeholder(shape=(m, ), name='A', dtype=dtype)
    print(vars(A))
    assert(A.stype == 'csr')
    C = tvm.compute(A.data.shape, lambda i: A.data[i] * 2., tag='cs_scatter')
    print(C.shape)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A.data, C], target)
    n = 3
    a = np.maximum(np.random.uniform(size=(n,n)).astype(dtype)-.6, 0.)
    print(a)
    a = tvmsp.array(a, ctx)
    c = tvmsp.array(np.zeros((n,n), dtype), ctx)
    c.data = tvm.nd.empty(a.data.shape, dtype)
    c.indices = a.indices
    c.indptr = a.indptr
    print('==== a ====')
    print(a.data)
    print(a.indices)
    print(a.indptr)
    print('==== c ====')
    print(c.data)
    print(c.indices)
    print(c.indptr)
    f(a.data, c.data)
    print('==== output ====')
    print(a.asnumpy())
    print(c.asnumpy())
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2., rtol=1e-5)

if __name__ == "__main__":
    test_tensor()

