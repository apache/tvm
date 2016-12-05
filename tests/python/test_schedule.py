import tvm

def test_schedule_create():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A(i, k) * B(j, k))

    Tsch = tvm.Schedule(T.op, scope="shared")
    Asch = tvm.Schedule(A.op)

    T.op.


    xo, xi = sch.split(sch.dim_var[0], factor)
    Asch.compute_at(Tsch, xi)

    xf = sch.fuse(xo, xi)


    tk1 = tvm.Split(T.op.dim_var[0], 10)
    assert isinstance(sch, tvm.schedule.Schedule)
    assert isinstance(tk1, tvm.schedule.DimSplit)

    print(tk1.var)
    print(sch.scope)
    print(sch.attachs)


if __name__ == "__main__":
    test_schedule_create()
