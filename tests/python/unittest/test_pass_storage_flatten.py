import tvm

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
    region = [tvm.make.range_by_min_extent(i[0], i[1]) for i in [(i, 2), (j, 8), (0, 4)]]
    stmt = tvm.make.Prefetch(A.op, 0, A.dtype, region)
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: _A}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert stmt.extent.value == 2
    assert isinstance(stmt.body, tvm.stmt.For)
    assert stmt.body.extent.value == 2


def test_flatten_storage_align():
    m = 8
    l = 16
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    s[A1].storage_align(A1.op.axis[0], 2, 1)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    Ab = tvm.decl_buffer(A.shape, A.dtype, name='A')
    A2b = tvm.decl_buffer(A2.shape, A2.dtype, name='A2')
    stmt = tvm.ir_pass.StorageFlatten(stmt, {A: Ab, A2: A2b}, 64)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(stmt.body.extents[0].value == 17 * 8)


if __name__ == "__main__":
    test_flatten_storage_align()
    test_flatten2()
    test_flatten_prefetch()
