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

def test_multi_if():
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
                tvm.make.Block(
                    tvm.make.IfThenElse((i*m+j+k < n), tvm.make.Evaluate(m), tvm.make.Evaluate(n)),
                    tvm.make.IfThenElse((i*m+j-k < n), tvm.make.Evaluate(m), tvm.make.Evaluate(n))
                    ))))
    stmt = tvm.ir_pass.LoopPartition(stmt)
    assert('if' not in str(stmt.body.first))
    print(stmt)

def test_thread_axis():
    m = tvm.Var('m')
    l = tvm.Var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.compute((m, l), lambda i, j: A[i, j] + 3, name='B')

    s = tvm.Schedule(B.op)

    s[B].set_scope("shared")
    num_thread = 16
    xo, xi = s[B].split(B.op.axis[0], 32)
    xi0, xi1 = s[B].split(xi, nparts=num_thread)
    s[B].bind(xi0, tvm.thread_axis("threadIdx.x"))

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt_ = tvm.ir_pass.LoopPartition(stmt)
    assert('if' not in str(stmt_.body.body.body.first))
    print(stmt_)

if __name__ == "__main__":
    test_basic()
    test_multi_loop()
    test_multi_if()
    test_thread_axis()
