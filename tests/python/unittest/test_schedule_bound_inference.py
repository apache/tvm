import tvm

def test_bound1():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.Schedule([A2.op])
    xo, xi = s[A2].split(s[A2].op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)

def test_bound2():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    s = tvm.Schedule(A2.op)
    xo, yo, xi, yi = s[A2].tile(A2.op.axis[0], A2.op.axis[1], 8, 8)
    s[A1].compute_at(s[A2], yo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)
    assert(bounds[A1.op.axis[1]].extent.value == 8)

def test_bound3():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.Schedule(A2.op)

    s[A1].set_scope("shared")
    thread_x = tvm.thread_axis((0, 16), "threadIdx.x")
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, outer=thread_x)
    yo, yi = s[A2].split(A2.op.axis[1], 16)
    s[A2].reorder(xo, xi0, yo, xi1, yi)
    s[A1].compute_at(s[A2], yo)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    assert(bounds[A1.op.axis[0]].extent.value==32)
    assert(bounds[A1.op.axis[1]].extent.value==16)

def test_bound_scan():
    m = tvm.Var("m")
    n = tvm.Var("n")
    X = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    s_scan = tvm.scan(s_init, s_update, s_state)

    assert tuple(s_scan.shape) == (m, n)

    s = tvm.Schedule(s_scan.op)
    XX = s.cache_read(X, "local", s_update)
    xo, xi = s[s_update].split(s_update.op.axis[1], factor=4)
    s[XX].compute_at(s[s_update], xo)

    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    assert bounds[XX.op.axis[1]].extent.value == 4


if __name__ == "__main__":
    test_bound_scan()
    test_bound3()
    test_bound1()
    test_bound2()
