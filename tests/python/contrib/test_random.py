import tvm
import numpy as np
from tvm.contrib import random

def test_randint():
    m = 1024
    n = 1024
    A = random.randint(-127, 128, size=(m, n), dtype='int32')
    s = tvm.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.randint", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na)) < 0.2
        assert np.min(na) == -127
        assert np.max(na) == 127
    verify()


def test_uniform():
    m = 1024
    n = 1024
    A = random.uniform(0, 1, size=(m, n))
    s = tvm.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.uniform", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na) - 0.5) < 1e-2
        assert abs(np.min(na) - 0.0) < 1e-3
        assert abs(np.max(na) - 1.0) < 1e-3
    verify()


def test_normal():
    m = 1024
    n = 1024
    A = random.normal(3, 4, size=(m, n))
    s = tvm.create_schedule(A.op)

    def verify(target="llvm"):
        if not tvm.module.enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.normal", True):
            print("skip because extern function is not available")
            return
        ctx = tvm.cpu(0)
        f = tvm.build(s, [A], target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), ctx)
        f(a)
        na = a.asnumpy()
        assert abs(np.mean(na) - 3) < 1e-2
        assert abs(np.std(na) - 4) < 1e-2
    verify()


if __name__ == "__main__":
    test_randint()
    test_uniform()
    test_normal()
