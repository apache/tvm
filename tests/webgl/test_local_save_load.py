import numpy as np
import tvm
from tvm import rpc
from tvm.contrib import util, emscripten

def test_local_save_load():
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    n = tvm.var("n")
    A = tvm.placeholder((n,), name='A', dtype='int32')
    B = tvm.placeholder((n,), name='B', dtype='int32')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = tvm.create_schedule(C.op)
    s[C].opengl()

    f = tvm.build(s, [A, B, C], "opengl", target_host="llvm", name="myadd")

    ctx = tvm.opengl(0)
    n = 10
    a = tvm.nd.array(np.random.uniform(high=10, size=(n)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(high=10, size=(n)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.zeros((n), dtype=C.dtype), ctx)
    f(a, b, c)

    temp = util.tempdir()
    path_so = temp.relpath("myadd.so")
    f.export_library(path_so)
    f1 = tvm.module.load(path_so)
    f1(a, b, c)
    np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

if __name__ == "__main__":
    test_local_save_load()
