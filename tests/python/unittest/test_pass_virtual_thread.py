import tvm

def test_virtual_thread():
    m = tvm.var('m')
    A = tvm.placeholder((m, ), name='A')
    A1 = tvm.compute((m,), lambda i: A[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A1[i] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    vx = tvm.thread_axis("vthread", name="vx")
    xo, xi = s[A2].split(A2.op.axis[0], nparts=2)
    s[A2].bind(xo, vx)
    xo, xi = s[A2].split(xi, 8)
    s[A1].compute_at(s[A2], xo)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    stmt = tvm.ir_pass.InjectVirtualThread(stmt)
    print(stmt)

if __name__ == "__main__":
    test_virtual_thread()
