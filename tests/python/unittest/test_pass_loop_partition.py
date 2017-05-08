import tvm

def collect_visit(stmt, f):
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, lambda x : ret.append(f(x)))
    return ret

def test_basic():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.body.body.first))
    print(stmt)

def test_multi_loop():
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i*m+j+k < n)):
                    ib.emit(tvm.make.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.make.Evaluate(n))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    print(stmt)
    assert(not any(collect_visit(stmt.body.first,
                                 lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_multi_if():
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, 4, 'i') as i:
        with ib.for_range(0, n, 'j') as j:
            with ib.for_range(0, m, 'k') as k:
                with ib.if_scope(ib.likely(i*m+j+k < n)):
                    ib.emit(tvm.make.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.make.Evaluate(n))
                with ib.if_scope(ib.likely(i*m+j-k < n)):
                    ib.emit(tvm.make.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.make.Evaluate(n))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.first))
    print(stmt)

def test_thread_axis():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    B = tvm.compute((m, l), lambda i, j: A[i, j] + 3, name='B')

    s = tvm.create_schedule(B.op)

    s[B].set_scope("shared")
    num_thread = 16
    xo, xi = s[B].split(B.op.axis[0], 32)
    xi0, xi1 = s[B].split(xi, nparts=num_thread)
    s[B].bind(xi0, tvm.thread_axis("threadIdx.x"))

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.body.body.first))
    print(stmt)

if __name__ == "__main__":
    test_basic()
    test_multi_loop()
    test_multi_if()
    test_thread_axis()
