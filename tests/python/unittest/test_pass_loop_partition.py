import tvm

def collect_visit(stmt, f):
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, lambda x : ret.append(f(x)))
    return ret

def lower(sch, args):
    binds = {}
    arg_list = []
    for x in args:
        if isinstance(x, tvm.tensor.Tensor):
            buf = tvm.decl_buffer(x.shape, dtype=x.dtype, name=x.name)
            assert x not in binds
            binds[x] = buf
            arg_list.append(buf)
        else:
            raise ValueError("args must be Tensor, Buffer or Var")
    sch = sch.normalize()
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.StorageFlatten(stmt, binds, 64)
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt = tvm.ir_pass.VectorizeLoop(stmt)
    stmt = tvm.ir_pass.Simplify(stmt)
    return stmt

def test_basic():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.body.body.first))

def test_const_loop():
    n = 21
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.body.body.first))

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
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt.body.first, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

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
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.first))

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
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert('if' not in str(stmt.body.body.body.first))

def test_vectorize():
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    bias = tvm.var("bias", dtype="float32")
    scale = tvm.var("scale", dtype="float32")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i) * scale + bias, name='C')
    # schedule
    s = tvm.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=num_thread*4)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=4)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    s[C].vectorize(x)
    stmt = lower(s, [A, B])
    body = stmt.body.body.body.body.body
    assert(x.var.name not in str(body.condition))
    assert(any(collect_visit(body.then_case, lambda x: isinstance(x, tvm.expr.Ramp))))

def test_select():
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, ((n+3)/4), 'i') as i:
      with ib.for_range(0, 4, 'j') as j:
        ib.emit(tvm.make.Evaluate(
          tvm.make.Select(ib.likely(i*4+j<n), m, n)))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt.first, lambda x: isinstance(x, tvm.expr.Select))))

def test_thread_axis2():
    n = tvm.convert(4096)
    m = tvm.var('m')
    A = tvm.placeholder((n,), name='A')
    B = tvm.placeholder((n,), name='B')
    C = tvm.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = tvm.create_schedule(C.op)
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=32)
    tx, x = s[C].split(x, nparts=num_thread)
    _,  x = s[C].split(x, factor=m)
    s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[C].bind(tx, tvm.thread_axis("threadIdx.x"))
    stmt = lower(s, [A, B])
    for_body = stmt.body.body.body.body.body.first
    assert('threadIdx' not in str(for_body.extent))

def test_everything_during_deduction():
    m = tvm.var('m')
    n = tvm.var('n')
    ib = tvm.ir_builder.create()
    with ib.for_range(0, n, 'i') as i:
        with ib.for_range(0, 32, 'j') as j:
            with ib.if_scope(ib.likely(i/j < m)):
                # this guard will produce everything during deduction
                ib.emit(tvm.make.Evaluate(m))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(isinstance(stmt.body.body, tvm.stmt.IfThenElse))

def test_single_likely():
    n = 60
    A = tvm.placeholder((n, ), name='A')
    B = tvm.placeholder((n, ), name='B')

    T = tvm.compute((n, ), lambda i: A[i]+B[i])
    s = tvm.create_schedule(T.op)
    x = T.op.axis[0]
    xo, xi = s[T].split(x, factor=16)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_multi_likely():
    n = 94
    m = 62
    A = tvm.placeholder((n, m), name='A')
    B = tvm.placeholder((n, m), name='B')

    T = tvm.compute((n, m), lambda i, j: A[i, j]+B[i, j])
    s = tvm.create_schedule(T.op)
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    x, y = T.op.axis
    xo, xi = s[T].split(x, factor=16)
    yo, yi = s[T].split(y, factor=16)
    s[T].reorder(xo, yo, xi, yi)

    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_oneD_pool():
    m = tvm.var('m')
    ib = tvm.ir_builder.create()
    #data = tvm.placeholder((16,), name = 'data')
    data = ib.pointer("float32", name="A")
    out = ib.pointer("float32", name="A")
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow > 0)):
                with ib.if_scope(ib.likely(ow < 15)):
                    out[ow] = tvm.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow < 1)):
                with ib.if_scope(ib.likely(kw > 0)):
                    out[ow] = tvm.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow > 14)):
                with ib.if_scope(ib.likely(kw < 2)):
                    out[ow] = tvm.max(out[ow], data[ow + kw - 1])

    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

if __name__ == "__main__":
    test_basic()
    test_const_loop()
    test_multi_loop()
    test_multi_if()
    test_thread_axis()
    test_vectorize()
    test_select()
    test_thread_axis2()
    test_everything_during_deduction()
    test_single_likely()
    test_multi_likely()
    test_oneD_pool()
