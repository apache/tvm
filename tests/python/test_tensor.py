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


if __name__ == "__main__":
    test_tensor()
