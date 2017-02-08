import tvm


def test_schedule0():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')

    s = tvm.Schedule(A1.op)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)

def test_schedule1():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')

    s = tvm.Schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.collections.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    print(stmt)

def test_schedule2():
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

def test_fusion():
  m = tvm.Var('m')
  n = tvm.Var('n')
  A = tvm.placeholder((m, n), name='A')
  B = tvm.placeholder((m, n), name='B')
  C = tvm.placeholder((m, n), name='C')
  T1 = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='T1')
  T2 = tvm.compute((m, n), lambda i, j: T1(i, j) + C(i, j), name='T2')

  s = tvm.Schedule(T2.op)
  tvm.schedule.AutoFuseEwise(s)
  bounds = tvm.schedule.InferBound(s)
  stmt = tvm.schedule.ScheduleOps(s, bounds)
  print(stmt)


if __name__ == "__main__":
    test_schedule0()
    test_schedule1()
    test_schedule2()
    test_fusion()
