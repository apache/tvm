import tvm

def test_lower_rfactor():
    n = tvm.var("n")
    m = tvm.var("m")
    A = tvm.placeholder((n, m), name='A')
    k = tvm.reduce_axis((0, m), "k")
    B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name="B")
    s = tvm.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B.op].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[B.op].bind(xi, tvm.thread_axis("threadIdx.y"))
    s[B].bind(s[B].op.reduce_axis[0], tvm.thread_axis("threadIdx.x"))
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    fapi = tvm.lower(s, [A, B])

if __name__ == "__main__":
    test_lower_rfactor()
