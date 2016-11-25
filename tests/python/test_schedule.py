import tvm

def test_schedule_create():
    m = tvm.Var('m')
    n = tvm.Var('n')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.placeholder((n, l), name='B')
    T = tvm.compute((m, n, l), lambda i, j, k: A(i, k) * B(j, k))

    sch = tvm.Schedule(T, scope="shared")
    tk1 = tvm.Split(0, 10)
    assert isinstance(sch, tvm.schedule.Schedule)
    assert isinstance(tk1, tvm.schedule.DimSplit)

    print(sch.scope)
    print(sch.attachs)


if __name__ == "__main__":
    test_schedule_create()
