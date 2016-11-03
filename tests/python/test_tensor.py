import tvm

def test_tensor():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.Tensor((m, l), name='A')
    B = tvm.Tensor((n, l), name='B')
    T = tvm.Tensor((m, n, l), lambda i, j, k: A(i, k) * B(j, k))
    print(tvm.format_str(T.source))
    assert(tuple(T.shape) == (m, n, l))
    assert(A.source is None)

def test_tensor_reduce():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.Tensor((m, l), name='A')
    B = tvm.Tensor((n, l), name='B')
    T = tvm.Tensor((m, n, l), lambda i, j, k: A(i, k) * B(j, k))
    rd = tvm.RDomain(tvm.Range(A.shape[1]))
    C = tvm.Tensor((m, n), lambda i, j: tvm.sum(T(i, j, rd.index[0]), rdom=rd))
    print(tvm.format_str(C.source))

if __name__ == "__main__":
    test_tensor()
    test_tensor_reduce()
