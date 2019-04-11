import tvm
import numpy as np
from tsim.load import load_driver, load_tsim

def test_tsim(i):
    rmin = 1 # min vector size of 1
    rmax = 64
    n = np.random.randint(rmin, rmax)
    ctx = tvm.cpu(0)
    a = tvm.nd.array(np.random.randint(rmax, size=n).astype("uint64"), ctx)
    b = tvm.nd.array(np.zeros(n).astype("uint64"), ctx)
    tsim = load_tsim()
    f = load_driver()
    f(tsim, a, b)
    emsg = "[FAIL] test number:{} n:{}".format(i, n)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1, err_msg=emsg)
    print("[PASS] test number:{} n:{}".format(i, n))

if __name__ == "__main__":
    for i in range(10):
        test_tsim(i)
