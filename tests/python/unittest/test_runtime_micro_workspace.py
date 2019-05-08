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
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    B = tvm.compute(B.shape, lambda *i: A(*i) + 1, name='B')
    C = tvm.compute(A.shape, lambda *i: B(*i) + 1, name='C')
    s = tvm.create_schedule(C.op)

    def verify():
        init_lib_path = micro.get_init_lib()
        micro.init("host", init_lib_path)
        m = tvm.module.load("fadd_workspace.obj", "micro_dev")
        ctx = tvm.micro_dev(0)
        fadd_workspace = m['fadd_workspace']
        n = nn
        a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
        c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
        print(a)
        print(c)
        fadd_workspace(a, c)
        print(a)
        print(c)
        # import struct
        # ba = bytearray(struct.pack('f', c.asnumpy()[0]))
        # print(ba)

        tvm.testing.assert_allclose(
            c.asnumpy(), a.asnumpy() + 2.0)
    verify()


if __name__ == "__main__":
    test_micro_add()
