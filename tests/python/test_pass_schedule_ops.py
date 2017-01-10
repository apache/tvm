import tvm


def test_schedule0():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    sA1 = tvm.Schedule(A1.op)
    bounds = tvm.schedule.InferBound(sA1)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.ir_pass.ScheduleOps(sA1, bounds)
    print(stmt)

def test_schedule1():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    sA1 = tvm.Schedule(A1.op)
    xo, xi = sA1.split(A1.op.axis[0], 8)
    bounds = tvm.schedule.InferBound(sA1)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.ir_pass.ScheduleOps(sA1, bounds)
    print(stmt)

def test_schedule2():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    sA1 = tvm.Schedule(A1.op)
    sA2 = tvm.Schedule(A2.op)
    xo, xi = sA2.split(A2.op.axis[0], 8)
    sA1.compute_at(sA2, xo)
    bounds = tvm.schedule.InferBound(sA2)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.ir_pass.ScheduleOps(sA2, bounds)
    print(stmt)


if __name__ == "__main__":
    test_schedule0()
    test_schedule1()
    test_schedule2()
