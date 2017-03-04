import tvm

def test_virtual_thread():
    m = tvm.Var('m')
    A = tvm.placeholder((m, ), name='A')
    A1 = tvm.compute((m,), lambda i: A[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A1[i] + 3, name='A2')

    s = tvm.Schedule(A2.op)
    vx = tvm.thread_axis((0, 2), "vthread", name="vx")
    xo, xi = s[A2].split(A2.op.axis[0], outer=vx)
    xo, xi = s[A2].split(xi, 8)
    s[A1].compute_at(s[A2], xo)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.Buffer(A.shape, A.dtype, name='A')
    A2b = tvm.Buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b})
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.InjectVirtualThread(stmt)
    print(stmt)

if __name__ == "__main__":
    test_virtual_thread()
