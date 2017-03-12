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


def test_conv1d():
    n = tvm.Var('n')
    A = tvm.placeholder((n+2), name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB)


def test_tensor_slice():
    n = tvm.Var('n')
    A = tvm.compute((n, n), lambda i, j: 1)
    B = tvm.compute((n,), lambda i: A[0][i] + A[0][i])


def test_tensor_reduce():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = tvm.reduce_axis((0, A.shape[1]), "k")
    C = tvm.compute((m, n), lambda i, j: tvm.sum(T(i, j, rv+1), axis=rv))
    # json load save
    C_json = tvm.save_json(C)
    C_loaded = tvm.load_json(C_json)
    assert(isinstance(C_loaded, tvm.tensor.Tensor))
    assert(str(C_loaded) == str(C))


def test_tensor_scan():
    m = tvm.Var("m")
    n = tvm.Var("n")
    x = tvm.placeholder((m, n))
    s = tvm.placeholder((m, n))
    res = tvm.scan(tvm.compute((1, n), lambda _, i: x[0, i]),
                   tvm.compute((m, n), lambda t, i: s[t-1, i] + x[t, i]),
                   s)
    assert tuple(res.shape) == (m, n)

def test_scan_multi_out():
    m = tvm.Var("m")
    n = tvm.Var("n")
    x1 = tvm.placeholder((m, n))
    s1 = tvm.placeholder((m, n))
    x2 = tvm.placeholder((m, n))
    s2 = tvm.placeholder((m, n))
    s1_init = tvm.compute((1, n), lambda _, i: x1[0, i])
    s2_init = tvm.compute((1, n), lambda _, i: x2[0, i])
    s1_update = tvm.compute((m, n), lambda t, i: s1[t-1, i] + s2[t-1, i] + x1[t, i])
    s2_update = tvm.compute((m, n), lambda t, i: x2[t, i] + s2[t-1,i])

    r0, r1 = tvm.scan([s1_init, s2_init],
                      [s1_update, s2_update],
                      [s1, s2])
    assert(r0.value_index == 0)
    assert(r1.value_index == 1)
    json_str = tvm.save_json(r0.op)
    zz = tvm.load_json(json_str)
    assert isinstance(zz, tvm.tensor.ScanOp)

def test_extern():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed("myadd", ins[0].data, outs[0].data, m)
    B = tvm.extern((m,), [A], extern_func)
    assert(tuple(B.shape) == (m,))


def test_extern_multi_out():
    m = tvm.Var('m')
    A = tvm.placeholder((m,), name='A')
    B = tvm.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed(
            "myadd", ins[0].data, outs[0].data, outs[1].data, m)
    res = tvm.extern([A.shape, A.shape], [A, B], extern_func)
    assert(len(res) == 2)
    assert(res[1].value_index == 1)


if __name__ == "__main__":
    test_conv1d()
    test_tensor_slice()
    test_tensor()
    test_tensor_reduce()
    test_tensor_scan()
    test_scan_multi_out()
    test_extern()
    test_extern_multi_out()
