import tvm
import numpy as np
from vsim.load import load_driver, load_vsim

def test_vsim(i):
    rmin = 1 # min vector size of 1
    rmax = 64
    n = np.random.randint(rmin, rmax)
    ctx = tvm.cpu(0)
    a = tvm.nd.array(np.random.randint(rmax, size=n).astype("uint64"), ctx)
    b = tvm.nd.array(np.zeros(n).astype("uint64"), ctx)
    vsim = load_vsim()
    f = load_driver()
    f(vsim, a, b)
    emsg = "[FAIL] test number:{} n:{}".format(i, n)
    np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1, err_msg=emsg)
    print("[PASS] test number:{} n:{}".format(i, n))

if __name__ == "__main__":
    for i in range(10):
        test_vsim(i)
