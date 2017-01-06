import tvm

def test_tensor():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == (m, n, l))
    assert(A.op is None)

def test_tensor_reduce():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = tvm.IterVar((0, A.shape[1]), name="k")
    C = tvm.compute((m, n), lambda i, j: tvm.sum(T(i, j, rv+1), rdom=rv))

    print(C.op.body)

if __name__ == "__main__":
    test_tensor()
    test_tensor_reduce()
