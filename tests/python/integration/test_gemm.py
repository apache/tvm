import tvm
from tvm.addon import nvcc_compiler
import numpy as np

def test_gemm():
    # graph
    nn = 1024
    n = tvm.Var('n')
    n = tvm.convert(nn)
    m = n
    l = n
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (n, m),
        lambda ii, jj: tvm.sum(A[ii, k] * B[jj, k], axis=k),
        name='CC')
    # schedule
    s = tvm.Schedule(C.op)
    xtile, ytile = 32, 32
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis(None, "blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    block_y = tvm.thread_axis(None, "blockIdx.y")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

    CC = s.cache_write(C, "local")
    AA = s.cache_read(A, "shared", [CC])
    BB = s.cache_read(B, "shared", [CC])
    _, yi = s[C].split(C.op.axis[0], factor=block_factor, outer=block_y)
    _, xi = s[C].split(C.op.axis[1], factor=block_factor, outer=block_x)
    s[C].reorder(block_y, block_x, yi, xi)
    _, yi = s[C].split(yi, outer=thread_y)
    _, xi = s[C].split(xi, outer=thread_x)
    s[C].reorder(thread_y, thread_x, yi, xi)
    yo, xo = CC.op.axis
    s[CC].reorder(k, yo, xo)

    s[CC].compute_at(s[C], thread_x)
    s[AA].compute_at(s[CC], k)
    s[BB].compute_at(s[CC], k)

    _, xi = s[AA].split(s[AA].op.axis[0], outer=thread_y)
    _, xi = s[AA].split(xi, outer=thread_x)
    _, xi = s[BB].split(s[BB].op.axis[0], outer=thread_y)
    _, xi = s[BB].split(xi, outer=thread_x)

    max_auto_unroll_step = 0
    # lowering test
    s.normalize()

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.codegen.enabled(host):
            return
        if not tvm.codegen.enabled(device):
            return

        f = tvm.build(s, [A, B, C], device, host,
                      max_auto_unroll_step=max_auto_unroll_step)
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel.
        n = nn
        m = n
        l = n
        a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        for i in range(4):
            f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(a_np, b_np.T), rtol=1e-5)

    if tvm.module.enabled("opencl"):
        tvm.module.init_opencl()
    check_device("cuda")
    check_device("opencl")

if __name__ == "__main__":
    test_gemm()
