import tvm
import numpy as np
from tvm.contrib.nvcc import have_fp16

def test_cuda_vectorize_add():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.module.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("skip because gpu does not support fp16")
            return
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i]+tvm.const(1, A.dtype), name='B')
        s = tvm.create_schedule(B.op)
        xo, xi = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
        s[B].bind(xi, tvm.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, B], "cuda")
        ctx = tvm.gpu(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(
            np.random.uniform(size=(n, lanes)))
        c = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a, c)
        np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + 1)
        
    check_cuda("float32", 64, 2)
    check_cuda("float16", 64, 2)

if __name__ == "__main__":
    test_cuda_vectorize_add()
