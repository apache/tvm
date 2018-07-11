import tvm
import numpy as np
from tvm.contrib.cuda import have_fp16

def test_fp16_sum():
    if not tvm.gpu(0).exist or not have_fp16(tvm.gpu(0).compute_version):
        print("skip because gpu does not exist or does not support fp16")
        return
    h, w = (64, 128)
    n = tvm.var("n")
    m = tvm.var("m")
    mysum = tvm.comm_reducer(lambda x, y: x+y,
                lambda t: tvm.const(0, dtype=t), name="mysum")
    A = tvm.placeholder((n, m), dtype="float16", name='A')
    k = tvm.reduce_axis((0, m), name='k')
    B = tvm.compute((n,), lambda i: mysum(A[i, k], axis=k), name='B')

    s = tvm.create_schedule(B.op)
    bx, tx = s[B].split(B.op.axis[0], factor=64)
    s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
    
    fun = tvm.build(s, [A, B], "cuda")

    ctx = tvm.context("cuda", 0)

    np_a = np.random.uniform(size=(h,w)).astype("float16")
    np_b = np.sum(np_a, axis=1)
    print(np_b.shape)

    a = tvm.nd.array(np_a, ctx)
    b = tvm.nd.array(np.zeros((h,), B.dtype), ctx)
    fun(a, b)
    
    np.testing.assert_allclose(b.asnumpy(), np_b, rtol=1e-2)

if __name__ == "__main__":
    test_fp16_sum()
