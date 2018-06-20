import os, sys
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(thisdir, '../../../python'))

import tvm
import tvm.contrib.sparse as tvmsp

def test_tensor():
    dtype = 'float32'
    stype = 'csr'
    m = tvm.var('m')
    A = tvmsp.placeholder((m, ), name='A', stype=stype, dtype=dtype)
    B = tvmsp.placeholder((m, ), name='B', stype=stype, dtype=dtype)
    print(vars(A))
    assert(A.stype == 'csr')
    assert(B.stype == 'csr')
    shape = [0]
    assert(str(A.data.shape) == str(shape))

if __name__ == "__main__":
    test_tensor()

