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
    assert(isinstance(A.op, tvm.tensor.PlaceholderOp))
    assert(A == A)
    assert(T.op.output(0) == T)
    assert(T.op.output(0).__hash__() == T.__hash__())
    d = {T.op.output(0) : 1}
    assert(d[T] == 1)


def test_tensor_reduce():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = tvm.IterVar((0, A.shape[1]), name="k")
    C = tvm.compute((m, n), lambda i, j: tvm.sum(T(i, j, rv+1), axis=rv))
    # json load save
    C_json = tvm.save_json(C)
    C_loaded = tvm.load_json(C_json)
    assert(isinstance(C_loaded, tvm.tensor.Tensor))
    assert(str(C_loaded) == str(C))


def test_tensor_scan():
    m = tvm.Var("m")
    n = tvm.Var("n")
    t = tvm.IterVar((1, m), "t")
    x = tvm.placeholder((m, n))
    s = tvm.placeholder((m, n))
    res = tvm.scan(tvm.compute((1, n), lambda _, i: x[0, i]),
                   tvm.compute((m, n), lambda t, i: s[t-1, i] + x[t, i]),
                   s)
    assert tuple(res.shape) == (m, n)


if __name__ == "__main__":
    test_tensor()
    test_tensor_reduce()
    test_tensor_scan()
