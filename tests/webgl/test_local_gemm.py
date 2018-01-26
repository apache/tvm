import tvm
import numpy as np

def test_local_gemm():
    if not tvm.module.enabled("opengl"):
        return
    if not tvm.module.enabled("llvm"):
        return

    nn = 1024
    n = tvm.var('n')
    n = tvm.convert(nn)
    m = n
    l = n
    A = tvm.placeholder((n, l), name='A', dtype='int32')
    B = tvm.placeholder((m, l), name='B', dtype='int32')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute((n, m), lambda ii, jj: tvm.sum(A[ii, k] * B[jj, k], axis=k),
                    name='CC')

    s = tvm.create_schedule(C.op)
    s[C].opengl()
    print(tvm.lower(s, [A, B, C], simple_mode=True))

    f = tvm.build(s, [A, B, C], "opengl", name="gemm")
    print("------opengl code------")
    print(f.imported_modules[0].get_source(fmt="gl"))

    ctx = tvm.opengl()
    n, m, l = nn, nn, nn
    a_np = np.random.uniform(low=0, high=10, size=(n, l)).astype(A.dtype)
    b_np = np.random.uniform(low=0, high=10, size=(m, l)).astype(B.dtype)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
    f(a, b, c)

    np.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.T))

if __name__ == "__main__":
    test_local_gemm()
