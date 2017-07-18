import tvm, numpy

def test_flatten2():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    xo, xi = s[A2].split(A2.op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)

def test_flatten_prefetch():
    A = tvm.placeholder((25, 100, 4), name = 'A')
    _A= tvm.decl_buffer(A.shape, A.dtype, name = 'A');
    i = tvm.var('i')
    j = tvm.var('j')
    region = map(lambda a: tvm.make.range_by_min_extent(a[0], a[1]), [(i, 2), (j, 8), (0, 4)])
    stmt = tvm.make.Prefetch(A.op, 0, A.dtype, region)
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: _A}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert str(stmt.extent) == "2"
    assert str(stmt.body.extent) == "2"

if __name__ == "__main__":
    test_flatten2()
    test_flatten_prefetch()
