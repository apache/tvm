import tvm
import os
import logging
import time

import numpy as np
from tvm.contrib import util
import tvm.micro as micro


# adds two arrays and stores result into third array
def test_micro_add():
    nn = 1024
    n = tvm.convert(nn)
    """
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i), name='C')
    """
    B = tvm.placeholder((n,), name='B')
    A = tvm.compute(B.shape, lambda *i: B(*i) + 1, name='A')
    C = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='C')
    s = tvm.create_schedule(C.op)

    def verify():
        init_path = micro.get_init_lib("../../../src/runtime/micro/device/utvm_runtime.cc")
        micro.micro_init("host", init_path)
        m = tvm.module.load("test.obj", "micro_dev")
        ctx = tvm.micro_dev(0)
        fadd = m['fadd']
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        print(a)
        print(b)
        print(c)
        fadd(a, b, c)
        print(c)
        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + b.asnumpy())
    verify()


if __name__ == "__main__":
    test_micro_add()
