import tvm

def test_tensor():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
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
    assert(T[0][0][0].astype('float16').dtype == 'float16')


def test_rank_zero():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    scale = tvm.placeholder((), name='s')
    k = tvm.reduce_axis((0, m), name="k")
    T = tvm.compute((), lambda : tvm.sum(A[k] * scale(), axis=k))
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == ())


def test_conv1d():
    n = tvm.var('n')
    A = tvm.placeholder((n+2), name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB)


def test_tensor_slice():
    n = tvm.var('n')
    A = tvm.compute((n, n), lambda i, j: 1)
    B = tvm.compute((n,), lambda i: A[0][i] + A[0][i])


def test_tensor_reduce_multi_axis():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    k1 = tvm.reduce_axis((0, n), "k")
    k2 = tvm.reduce_axis((0, m), "k")
    C = tvm.compute((1,), lambda _: tvm.sum(A[k1, k2], axis=(k1, k2)))
    C = tvm.compute((1,), lambda _: tvm.sum(A[k1, k2], axis=[k1, k2]))


def test_tensor_comm_reducer():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    k = tvm.reduce_axis((0, n), "k")
    mysum = tvm.comm_reducer(lambda x, y: x+y, lambda t: tvm.const(0, dtype=t))
    C = tvm.compute((m,), lambda i: mysum(A[i, k], axis=k))

def test_tensor_comm_reducer_overload():
    m = tvm.var('m')
    n = tvm.var('n')
    mysum = tvm.comm_reducer(lambda x, y: x+y, lambda t: tvm.const(0, dtype=t))
    sum_res = mysum(m, n)

def test_tensor_reduce():
    m = tvm.var('m')
    n = tvm.var('n')
    l = tvm.var('l')
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
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.placeholder((m, n))
    s = tvm.placeholder((m, n))
    res = tvm.scan(tvm.compute((1, n), lambda _, i: x[0, i]),
                   tvm.compute((m, n), lambda t, i: s[t-1, i] + x[t, i]),
                   s)
    assert tuple(res.shape) == (m, n)

def test_scan_multi_out():
    m = tvm.var("m")
    n = tvm.var("n")
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
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed("myadd", ins[0].data, outs[0].data, m)
    B = tvm.extern((m,), [A], extern_func)
    assert(tuple(B.shape) == (m,))


def test_extern_multi_out():
    m = tvm.var('m')
    A = tvm.placeholder((m,), name='A')
    B = tvm.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.schedule.Buffer))
        return tvm.call_packed(
            "myadd", ins[0].data, outs[0].data, outs[1].data, m)
    res = tvm.extern([A.shape, A.shape], [A, B], extern_func)
    assert(len(res) == 2)
    assert(res[1].value_index == 1)

def test_tuple_inputs():
    m = tvm.var('m')
    n = tvm.var('n')
    A0 = tvm.placeholder((m, n), name='A0')
    A1 = tvm.placeholder((m, n), name='A1')
    T0, T1 = tvm.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='T')
    s = tvm.create_schedule(T0.op)

    for i in range(len(T0.shape)):
      assert(T0.shape[i] == T1.shape[i])
    assert(T0.op == T1.op)
    assert(T0.value_index == 0)
    assert(T1.value_index == 1)

def test_tuple_with_different_deps():
    m = tvm.var('m')
    n = tvm.var('n')
    A0 = tvm.placeholder((m, n), name='A1')
    A1 = tvm.placeholder((m, n), name='A2')
    B0, B1 = tvm.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='B')
    C = tvm.compute((m, n), lambda i, j: B0[i, j] + 4, name='C')

    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=10)
    s[B0.op].compute_at(s[C], xo)
    sch = s.normalize()
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds)

    def get_B1_realize(x):
        if isinstance(x, tvm.stmt.Realize) and \
           x.func == B1.op and x.value_index == 1:
            ret.append(x)
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, get_B1_realize)

    assert stmt.node == C.op and len(ret) == 1


def test_tensor_inputs():
    x = tvm.placeholder((1,), name='x')
    y = tvm.compute(x.shape, lambda i: x[i] + x[i])
    assert tuple(y.op.input_tensors) == (x,)


if __name__ == "__main__":
    test_rank_zero()
    test_tensor_inputs()
    test_tensor_reduce_multi_axis()
    test_conv1d()
    test_tensor_slice()
    test_tensor()
    test_tensor_reduce()
    test_tensor_scan()
    test_scan_multi_out()
    test_extern()
    test_extern_multi_out()
    test_tuple_inputs()
    test_tuple_with_different_deps()
