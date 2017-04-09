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
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, nparts=16)
    s[A2].bind(xi0, tvm.thread_axis("threadIdx.x"))
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

def test_bound_conv1d():
    n = tvm.Var('n')
    A = tvm.compute((n+2), lambda i: 1,  name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB, name='B')
    s = tvm.Schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[0])
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)

def test_bound_blur():
    n = tvm.convert(12)
    A = tvm.compute((n, n), lambda i, j: 1, name='A')
    def computeB(ii, jj):
        # set the correct center
        i = ii + 1
        j = jj + 1
        return A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j+1] + A[i][j-1]
    B = tvm.compute((n-2, n-2), computeB, name='B')
    s = tvm.Schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[1])
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)
    assert(bounds[A.op.axis[1]].extent.value == 3)

def test_bound_rfactor():
    n = tvm.Var('n')
    A = tvm.placeholder((n,), name='A')
    k = tvm.reduce_axis((0, n))
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k, where=(i>1)), name='B')
    # schedule
    s = tvm.Schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s.normalize()
    bounds = tvm.schedule.InferBound(s)

    assert(bounds[BF.op.axis[0]].extent.value == 4)
    assert(bounds[BF.op.axis[1]].extent.value == 1)

def test_bound_group_schedule():
    m = tvm.Var("m")
    n = tvm.Var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.Schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    g.compute_at(s[x2], x2.op.axis[0])
    assert s[x1].group == g
    assert s[x].group == g
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent == n

def test_bound_nest_group():
    m = tvm.Var("m")
    n = tvm.Var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.Schedule(x2.op)
    g1 = s.create_group(outputs=x, inputs=x, include_inputs=True)
    g2 = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x].group == g1
    assert s[x1].group == g2
    g2.compute_at(s[x2], x2.op.axis[0])
    g1.compute_at(s[x1], s[x1].op.axis[1])
    s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent.value == 1
    assert bounds[x1.op.axis[0]].extent.value == 1
    assert bounds[x1.op.axis[1]].extent == n

if __name__ == "__main__":
    test_bound_nest_group()
    test_bound_group_schedule()
    test_bound_scan()
    test_bound3()
    test_bound_rfactor()
    test_bound_blur()
    test_bound_conv1d()
    test_bound1()
    test_bound2()
