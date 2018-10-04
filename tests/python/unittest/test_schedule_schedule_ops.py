import tvm


def test_schedule0():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    s = tvm.create_schedule(A1.op)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_schedule1():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')

    s = tvm.create_schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    s[A1].pragma(xo, "auto_unroll_max_step", 10)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule2():
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


def test_schedule_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: x[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + x[t, i])
    res = tvm.scan(s_init, s_update, s_state)

    assert tuple(res.shape) == (m, n)
    s = tvm.create_schedule(res.op)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[res.op.scan_axis].min.value == 1)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_inline_multi_reduce():
    def argmax_comp(x, y):
        idx = tvm.select((x[1] >= y[1]), x[0], y[0])
        val = tvm.select((x[1] >= y[1]), x[1], y[1])
        return idx, val
    def argmax_init(idx_typ, val_typ):
        return tvm.const(-1, idx_typ), tvm.min_value(val_typ)

    argmax = tvm.comm_reducer(argmax_comp, argmax_init, name='argmax')
    m = tvm.var('m')
    n = tvm.var('n')
    val = tvm.placeholder((m, n), name='val', dtype='float32')
    val1 = tvm.compute((m, n), lambda i, j: val[i, j]+1, name='val1')
    val2 = tvm.compute((m, n), lambda i, j: tvm.exp(val1[i, j]), name='val2')
    k = tvm.reduce_axis((0, n), 'k')
    T_idx, T_val = tvm.compute((m, ), lambda i: argmax((k.var, val2[i, k]), axis=k), name='T')
    s = tvm.create_schedule(T_idx.op)
    s[val1].compute_inline()
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)



def test_auto_inline():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')
    C = tvm.placeholder((m, n), name='C')
    T1 = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='T1')
    T2 = tvm.compute((m, n), lambda i, j: T1(i, j) + C(i, j), name='T2')

    s = tvm.create_schedule(T2.op)
    tvm.schedule.AutoInlineElemWise(s)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_schedule_const_bound():
    n = 128
    A = tvm.placeholder((n,), name='A')
    A1 = tvm.compute((n,), lambda i: A[i] + 1, name='A1')
    s = tvm.create_schedule(A1.op)
    xo, xi = s[A1].split(A1.op.axis[0], 8)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_inline_mixed():
    n = tvm.var('n')
    A = tvm.placeholder((n, ), name='A')
    A1 = tvm.compute(A.shape, lambda *i: A(*i) + 1, name='A1')
    A2 = tvm.compute(A.shape, lambda *i: A1(*i) + 2, name='A2')
    C = tvm.compute((n,), lambda i: A2[i] + A1[i], name='C')

    s = tvm.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=8)
    s[A1].compute_at(s[C], xo)
    s[A2].compute_inline()
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    def check(x):
        if isinstance(x, tvm.expr.Call):
            assert x.func != A2
    tvm.ir_pass.PostOrderVisit(s[C].op.body[0], check)


def test_scan_inline1():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state1 = tvm.placeholder((m, n))
    s_state2 = tvm.placeholder((m, n))
    s_init1 = tvm.compute((1, n), lambda _, i: x[0, i])
    s_init2 = tvm.compute((1, n), lambda _, i: x[0, i])
    s_x1 = tvm.compute((m, n), lambda t, i: s_state1[t-1, i] + x[t, i], name="x1")
    s_x2 = tvm.compute((m, n), lambda t, i: s_state2[t-1, i] + 1 , name="x2")
    s_update1 = tvm.compute((m, n), lambda t, i: s_x1[t, i], "u1")
    s_update2 = tvm.compute((m, n), lambda t, i: s_x2[t, i], "u2")
    res1, res2 = tvm.scan([s_init1, s_init2],
                          [s_update1, s_update2],
                          [s_state1, s_state2])
    s = tvm.create_schedule(res1.op)
    s[s_x1].compute_inline()
    stmt = tvm.lower(s, [x, res1, res2])

def test_scan_inline2():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state1 = tvm.placeholder((m, n))
    s_state2 = tvm.placeholder((m, n))
    s_init1 = tvm.compute((1, n), lambda _, i: x[0, i])
    s_init2 = tvm.compute((1, n), lambda _, i: x[0, i])
    s_xx = tvm.compute((m, n), lambda t, i: s_state1[t-1, i] + x[t, i], name="xx")
    s_x1 = tvm.compute((m, n), lambda t, i: s_xx[t, i] + 1, name="x1")
    s_x2 = tvm.compute((m, n), lambda t, i: s_xx[t, i] + s_state2[t-1, 2], name="x2")
    s_update1 = tvm.compute((m, n), lambda t, i: s_x1[t, i], "u1")
    s_update2 = tvm.compute((m, n), lambda t, i: s_x2[t, i], "u2")
    res1, res2 = tvm.scan([s_init1, s_init2],
                          [s_update1, s_update2],
                          [s_state1, s_state2])
    s = tvm.create_schedule(res1.op)
    s[s_xx].compute_inline()
    s[s_x1].compute_inline()
    s[s_x2].compute_inline()
    stmt = tvm.lower(s, [x, res1, res2])


def test_schedule_cache():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')
    C = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')

    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", readers=[C])
    CC = s.cache_write(C, "shared")
    s[AA].compute_at(s[CC], CC.op.axis[0])
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_schedule_middle_cache():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')

    C = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')
    D = tvm.compute((m, n), lambda i, j:  C(i , j) , name='D')

    s = tvm.create_schedule(D.op)
    AA = s.cache_read(A, "local", readers=[C])
    BB = s.cache_read(B, "local", readers=[C])
    CC = s.cache_read(C, "local", readers=[D])
    DD = s.cache_write(D, "local")
    #s[AA].compute_at(s[CC], CC.op.axis[0])
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)



def test_schedule_cache_relayout1():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m, n), name='A')
    B = tvm.placeholder((m, n), name='B')
    C = tvm.compute((m, n), lambda i, j:  A(i, j) * B(i, j), name='C')

    s = tvm.create_schedule(C.op)
    s[C].reorder(C.op.axis[1], C.op.axis[0])
    CC = s.cache_write(C, "global")
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout2():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m*4, n), name='A')
    B = tvm.placeholder((m*4, n), name='B')
    C = tvm.compute(A.shape, lambda i, j:  A(i, j) * B(i, j), name='C')
    s = tvm.create_schedule(C.op)
    x, y = C.op.axis
    xo, xi = s[C].split(x, factor=4)
    s[C].reorder(xo, y, xi)
    CC = s.cache_write(C, "global")
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule_cache_relayout3():
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m*4, n), name='A')
    B = tvm.placeholder((m*4, n), name='B')
    k = tvm.reduce_axis((0, n), "k")
    C = tvm.compute((A.shape[0],),
                    lambda i: tvm.sum(A(i, k) * B(i, k), axis=k), name='C')
    s = tvm.create_schedule(C.op)
    x = C.op.axis[0]
    xo, xi = s[C].split(x, factor=4)
    CC = s.cache_write(C, "global")
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)

def test_schedule_cache_relayout4():
    def _compute(*indice):
        return A(*indice) + 1, B(*indice) / 2
    m = tvm.var('m')
    n = tvm.var('n')
    A = tvm.placeholder((m*4, n), name='A')
    B = tvm.placeholder((m*4, n), name='B')
    C1, C2 = tvm.compute(A.shape, _compute, name='C')
    s = tvm.create_schedule([C1.op, C2.op])
    C1_cache, C2_cache = s.cache_write([C1, C2], "local")
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule_bound_condition():
   A = tvm.placeholder((64,), name='A', dtype="float32")
   Apad = tvm.compute((66,), lambda i: tvm.select(tvm.all(i>0, i < 65), A[i-1], tvm.const(0.)), name='Apad')
   Apad2 = tvm.compute((66,), lambda i: Apad[i]*2, name='Apad2')
   s = tvm.create_schedule(Apad2.op)
   AL1 = s.cache_read(A,"local",[Apad])
   s = s.normalize()
   bounds = tvm.schedule.InferBound(s)
   stmt = tvm.schedule.ScheduleOps(s, bounds)
   stmt = tvm.ir_pass.Simplify(stmt)
   assert (isinstance(stmt.body.body.first.body.body.then_case, tvm.stmt.IfThenElse))


def intrin_gemv(m, n):
    w = tvm.placeholder((m, n), name='w')
    x = tvm.placeholder((n,), name='x')
    k = tvm.reduce_axis((0, n), name='k')
    z = tvm.compute((m,), lambda i:
                    tvm.sum(w[i, k] * x[k], axis=k), name='z')
    Wb = tvm.decl_buffer(w.shape, w.dtype,
                         name="W",
                         offset_factor=16,
                         strides=[tvm.var('ldw'), 1])
    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        ww_ptr = ww.access_ptr("r")
        xx_ptr = xx.access_ptr("r")
        zz_ptr = zz.access_ptr("w")
        body = tvm.call_packed(
            "gemm", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        reset = tvm.call_packed(
            "fill_zero", zz_ptr, n)
        update = tvm.call_packed(
            "gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, reset, update

    with tvm.build_config(data_alignment=16,
                          offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func,
                                      binds={w: Wb})


def test_schedule_tensor_compute1():
    # basic: split, reorder, tile
    M, N, L = 2048, 1024, 512
    factor, rfactor = 16, 16
    A = tvm.placeholder((N//factor, L//rfactor, factor, rfactor), name='A')
    B = tvm.placeholder((M, L//rfactor, rfactor), name='B')
    k = tvm.reduce_axis((0, L//rfactor), name='k')

    gemv = intrin_gemv(factor, rfactor)
    C = tvm.compute((N, M//factor, factor),
        lambda i, j: gemv(A[i, k, 0:factor, 0:factor], B[j, k, 0:rfactor], reduce_axis=k),
        name='C')

    s = tvm.create_schedule(C.op)
    ai, aj, ax = s[C].op.axis
    aio, aii = s[C].split(ai, 16)
    s[C].reorder(aio, aj, aii)
    aioo, ajo, aioi, aji = s[C].tile(aio, aj, 16, 4)

    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def intrin_vadd(n, cache_read=False, cache_write=False):
    scope_ubuf = 'local'
    dtype = 'float32'
    x = tvm.placeholder((n,), dtype=dtype, name='vx')
    y = tvm.placeholder((n,), dtype=dtype, name='vy')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')
    s = tvm.create_schedule(z.op)

    def create_buffer(t):
        return tvm.decl_buffer(t.shape, t.dtype,
                               name='W'+t.name,
                               scope=scope_ubuf,
                               offset_factor=16)

    binds = {}
    if cache_read:
        binds[x] = create_buffer(x)
        binds[y] = create_buffer(y)
    if cache_write:
        binds[z] = create_buffer(z)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()
        ib.emit(tvm.call_extern(outs[0].dtype, 'vadd', ins[0].access_ptr("r"), ins[1].access_ptr('r'), outs[0].access_ptr('wr')))
        return ib.get()

    with tvm.build_config(offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func, binds=binds)


def test_schedule_tensor_compute2():
    # cache_read, cache_write
    M = 1024
    factor = 16
    dtype = 'float32'
    scope_ubuf = 'local'

    A = tvm.placeholder((M//factor, factor), name="A", dtype=dtype)
    B = tvm.placeholder((M//factor, factor), name="B", dtype=dtype)

    vadd = intrin_vadd(factor, True, True)
    C = tvm.compute((M//factor, factor),
        lambda i: vadd(A[i, 0:factor], B[i, 0:factor]), name='C')

    s = tvm.create_schedule(C.op)
    AL = s.cache_read(A, scope_ubuf, C)
    BL = s.cache_read(B, scope_ubuf, C)
    CL = s.cache_write(C, scope_ubuf)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


def test_schedule_tensor_compute3():
    # compute_at
    M = 1024
    factor = 16
    dtype = 'float32'
    A = tvm.placeholder((M//factor, factor), name="A", dtype=dtype)
    B = tvm.placeholder((M//factor, factor), name="B", dtype=dtype)
    Bi = tvm.compute((M//factor, factor), lambda i, j: B[i, j] + 5, name="Bi")

    vadd = intrin_vadd(factor)
    C = tvm.compute((M//factor, factor),
        lambda i: vadd(A[i, 0:factor], Bi[i, 0:factor]), name='C')
    s = tvm.create_schedule(C.op)
    s[Bi].compute_at(s[C], C.op.axis[0])
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)


if __name__ == "__main__":
    test_schedule_middle_cache()
    test_inline_multi_reduce()
    test_schedule_cache_relayout4()
    test_schedule_cache_relayout3()
    test_schedule_cache_relayout2()
    test_schedule_cache_relayout1()
    test_schedule_const_bound()
    test_scan_inline1()
    test_scan_inline2()
    test_inline_mixed()
    test_auto_inline()
    test_schedule_scan()
    test_schedule0()
    test_schedule1()
    test_schedule2()
    test_schedule_cache()
    test_schedule_bound_condition()
    test_schedule_tensor_compute1()
    test_schedule_tensor_compute2()
    test_schedule_tensor_compute3()
