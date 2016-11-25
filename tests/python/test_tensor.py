import tvm

def test_tensor():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A(i, k) * B(j, k))

    print(T.source)
    assert(tuple(T.shape) == (m, n, l))
    assert(A.source is None)

def test_tensor_reduce():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A(i, k) * B(j, k))
    rd = tvm.RDomain(tvm.Range(A.shape[1]))
    C = tvm.compute((m, n), lambda i, j: tvm.sum(T(i, j, rd.index[0]), rdom=rd))
    print(C.source)

if __name__ == "__main__":
    test_tensor()
    test_tensor_reduce()
