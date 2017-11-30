import tvm
import numpy as np
from tvm.contrib import random

def test_matmul_add():
    m = 256
    n = 256
    A = random.randint(-127, 128, size=(m, n))
    s = tvm.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.randint", True):
            print("skip because extern function is not avalable")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        print(na)
        print(np.max(na))
        print(np.min(na))
    verify()


if __name__ == "__main__":
    test_matmul_add()
