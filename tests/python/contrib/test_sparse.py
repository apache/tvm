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
    A = tvmsp.CSRTensor(shape=(m, ), name='A', dtype=dtype)
    print(vars(A))
    assert(A.stype == 'csr')
    C = tvm.compute(A.data.shape, lambda i: A.data[i] + 1., tag='cs_scatter')
    print(C.shape)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A.data, C], target)
    n = 5
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
    f(a, c)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1.)
    print(a.asnumpy())
    print(c.asnumpy())

if __name__ == "__main__":
    test_tensor()

