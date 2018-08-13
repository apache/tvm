import tvm
import numpy as np
from tvm.contrib.nvcc import have_fp16, have_int8
from tvm.contrib import nvcc

def test_cuda_vectorize_add():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.module.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "float16" and not have_fp16(tvm.gpu(0).compute_version):
            print("skip because gpu does not support fp16")
            return
        if dtype == "int8" and not have_int8(tvm.gpu(0).compute_version):
            print("skip because gpu does not support int8")
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


def test_cuda_multiply_add():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.module.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        if dtype == "int8" and not have_int8(tvm.gpu(0).compute_version):
            print("skip because gpu does not support int8")
            return
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.placeholder((n,), name='B', dtype="%sx%d" % (dtype, lanes))
        C = tvm.placeholder((n,), name='C', dtype="int32")        
        D = tvm.compute((n,),
                        lambda i: tvm.call_pure_extern("int32", "__dp4a", A[i], B[i], C[i]), name='D')
        s = tvm.create_schedule(D.op)
        xo, xi = s[D].split(D.op.axis[0], factor=num_thread)
        s[D].bind(xo, tvm.thread_axis("blockIdx.x"))
        s[D].bind(xi, tvm.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, B, C, D], "cuda")
        np_a = np.random.randint(low=-128, high=127, size=(n,lanes))
        np_b = np.random.randint(low=-128, high=127, size=(n,lanes))
        np_c = np.random.randint(low=0, high=127, size=(n,))
        np_d = [sum(x * y) + z for x, y, z in zip(np_a, np_b, np_c)]
        ctx = tvm.gpu(0)
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, ctx).copyfrom(np_b)
        c = tvm.nd.empty((n,), C.dtype, ctx).copyfrom(np_c)
        d = tvm.nd.empty((n,), D.dtype, ctx)
        fun(a, b, c, d)
        np.testing.assert_allclose(d.asnumpy(), np_d)
    check_cuda("int8", 64, 4)

def test_cuda_vectorize_load():
    num_thread = 8
    def check_cuda(dtype, n, lanes):
        if not tvm.gpu(0).exist or not tvm.module.enabled("cuda"):
            print("skip because cuda is not enabled..")
            return
        ctx = tvm.gpu(0)
        A = tvm.placeholder((n,), name='A', dtype="%sx%d" % (dtype, lanes))
        B = tvm.compute((n,), lambda i: A[i], name='B')
        s = tvm.create_schedule(B.op)
        bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
        s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
        fun = tvm.build(s, [A, B], "cuda", name="vector_load")
        np_a = np.random.randint(low=-128, high=127, size=(n,lanes))
        a = tvm.nd.empty((n,), A.dtype, ctx).copyfrom(np_a)
        b = tvm.nd.empty((n,), B.dtype, ctx)
        fun(a,b)
        np.testing.assert_allclose(a.asnumpy(), b.asnumpy())
    check_cuda("int8", 64, 8)
    check_cuda("int8", 64, 16)

if __name__ == "__main__":
    test_cuda_vectorize_add()
    test_cuda_multiply_add()
    test_cuda_vectorize_load()
