import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))

import tvm
import tvm.contrib.sparse as tvmsp

def test_tensor():
    dtype = 'float32'
    stype = 'csr'
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
    A = tvmsp.placeholder((m, ), name='A', stype=stype, dtype=dtype)
    B = tvmsp.placeholder((n, ), name='B', stype=stype, dtype=dtype)
    print(A)
    assert(A.stype == 'csr')
    assert(B.stype == 'csr')
    assert(A.data.shape == (0,))
    assert(A.indices.shape == (0,))
    assert(A.indptr.shape == (0,))
    # T = tvmsp.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])

if __name__ == "__main__":
    test_tensor()

