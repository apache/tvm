import tvm

def test_basic():
    n = tvm.Var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.Schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt)
    assert('if' not in str(stmt.body.body.body.first))
    print(stmt)

def test_multi_loop():
    i = tvm.Var('i')
    j = tvm.Var('j')
    k = tvm.Var('k')
    m = tvm.Var('m')
    n = tvm.Var('n')
    stmt = tvm.make.For(
        i, 0, 4, 0, 0,
        tvm.make.For(
            j, 0, n, 0, 0,
            tvm.make.For(
                k, 0, m, 0, 0,
                tvm.make.IfThenElse(
                    (i*m+j+k < n), tvm.make.Evaluate(m), tvm.make.Evaluate(n)))))
    stmt = tvm.ir_pass.LoopPartition(stmt)
    assert('if' not in str(stmt.body.first))
    print(stmt)


if __name__ == "__main__":
    test_basic()
    test_multi_loop()
