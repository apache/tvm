import tvm

def test_flatten2():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.Schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    print(stmt)
    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    A2b = tvm.Buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b})
    stmt = tvm.ir_pass.Simplify(stmt)
    print(stmt)

if __name__ == "__main__":
    test_flatten2()
