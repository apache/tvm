import tvm

def intrin_vadd(n):
    x = tvm.placeholder((n,), name='vx')
    y = tvm.placeholder((n,), name='vy')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')
    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
        return tvm.call_packed("vadd", xx, yy, zz)
    with tvm.build_config(offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func)

def _argmax_comp(lhs, rhs):
    idx = tvm.make.Select((lhs[1] >= rhs[1]), lhs[0], rhs[0])
    val = tvm.make.Select((lhs[1] >= rhs[1]), lhs[1], rhs[1])
    return idx, val

def _argmax_init(idx_type, val_type):
    return tvm.const(-1, idx_type), tvm.min_value(val_type)

def intrin_vargmax(m, n):
    # (4, 128) tensorized argmax
    _argmax = tvm.comm_reducer(fcombine=_argmax_comp, fidentity=_argmax_init, name='argmax')
    x = tvm.placeholder((m, n), name='x')
    r = tvm.reduce_axis((0, n), name='r')
    idx, val = tvm.compute((m, ), lambda j: _argmax((r.var, x[j][r]), axis=r), name='argmax_op')
    op = idx.op

    x_layout = tvm.decl_buffer(x.shape, x.dtype, 'x_buffer',
        offset_factor=16, data_alignment=16, strides=[1024, 1])
    idx_layout = tvm.decl_buffer(idx.shape, idx.dtype, 'idx_buffer',
        offset_factor=16, data_alignment=16, strides=[1, ])
    val_layout = tvm.decl_buffer(val.shape, val.dtype, 'val_buffer',
        offset_factor=16, data_alignment=16, strides=[1, ])

    def intrin_func(ins, outs):
        xx = ins[0]
        yy, zz = outs

        xx_ptr = xx.access_ptr("r")
        yy_ptr = yy.access_ptr("w")
        zz_ptr = zz.access_ptr("w")

        body = tvm.call_packed(
            "argmax", xx_ptr, yy_ptr, zz_ptr)
        reset = tvm.call_packed(
            "argmax_init", yy_ptr, zz_ptr)
        update = tvm.call_packed(
            "argmax_cmp", xx_ptr, yy_ptr, zz_ptr)
        return body, reset, update

    return tvm.decl_tensor_intrin(op, intrin_func, name='argmax',
            binds={x: x_layout, idx: idx_layout, val: val_layout})

def test_tensorize_argmax():
    _argmax = tvm.comm_reducer(fcombine=_argmax_comp, fidentity=_argmax_init, name='argmax')
    m = 1024
    n = 1024
    k = tvm.reduce_axis((0, n), name='k')
    data = tvm.placeholder((m, n), name='A')
    T_idx, T_val = tvm.compute((m, ), lambda i: _argmax((k.var, data[i][k]), axis=k), name='T')
    T = T_idx.op

    factor = 4
    rfactor = 16
    s = tvm.create_schedule(T)
    xo, xi = s[T].split(T.axis[0], factor=factor)
    ko, ki = s[T].split(k, factor=rfactor)
    s[T].reorder(xo, ko, xi, ki)

    # tensorize
    vargmax = intrin_vargmax(factor, rfactor)
    s[T].tensorize(xi, vargmax)
    s = s.normalize()
    dom_map = tvm.schedule.InferBound(s)
    finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
    out_dom, in_dom = finfer(s[T], dom_map)
    fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
    body = fmatch(s[T], out_dom, in_dom, vargmax)
    assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                             tvm.ir_pass.CanonicalSimplify(vargmax.op.body[0]))
    tvm.lower(s, [data, T_idx, T_val])


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
            "gemv", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        reset = tvm.call_packed(
            "fill_zero", zz_ptr, n)
        update = tvm.call_packed(
            "gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, reset, update

    with tvm.build_config(data_alignment=16,
                          offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func,
                                      binds={w: Wb})

def intrin_gemv_no_reset(m, n):
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
            "gemv", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        update = tvm.call_packed(
            "gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, None, update

    with tvm.build_config(data_alignment=16,
                          offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func,
                                      binds={w: Wb})


def test_tensorize_vadd():
    m = 128
    x = tvm.placeholder((m,), name='x')
    y = tvm.placeholder((m,), name='y')
    z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')

    def check(factor):
        s = tvm.create_schedule(z.op)
        xo, xi = s[z].split(z.op.axis[0], factor=factor)
        vadd = intrin_vadd(factor)
        s[z].tensorize(xi, vadd)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[z], dom_map)
        assert tvm.ir_pass.Equal(out_dom[z.op.axis[0]].extent, factor)
        assert tvm.ir_pass.Equal(out_dom[z.op.axis[0]].min, xo * factor)
        assert tvm.ir_pass.Equal(in_dom.items()[0][1][0].extent, factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[z], out_dom, in_dom, vadd)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(vadd.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [x, y, z])

    check(16)


def test_tensorize_matmul():
    n = 1024
    m = n
    l = n
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute((n, m), lambda i, j:
                    tvm.sum(B[j, k] * A[i, k], axis=k), name='C')

    def check(factor):
        s = tvm.create_schedule(C.op)
        x, y = C.op.axis
        yo, yi = s[C].split(y, factor=factor)
        gemv = intrin_gemv(factor, l)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        assert tvm.ir_pass.Equal(out_dom[x].extent, 1)
        assert tvm.ir_pass.Equal(out_dom[y].extent, factor)
        assert tvm.ir_pass.Equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(gemv.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])


    def check_rfactor(factor, rfactor):
        s = tvm.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        s[C].reorder(yo, ro, yi, ri)
        gemv = intrin_gemv(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        assert tvm.ir_pass.Equal(out_dom[x].extent, 1)
        assert tvm.ir_pass.Equal(out_dom[y].extent, factor)
        assert tvm.ir_pass.Equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(gemv.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    def check_rfactor_no_reset(factor, rfactor):
        s = tvm.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        s[C].reorder(yo, ro, yi, ri)
        gemv = intrin_gemv_no_reset(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        assert tvm.ir_pass.Equal(out_dom[x].extent, 1)
        assert tvm.ir_pass.Equal(out_dom[y].extent, factor)
        assert tvm.ir_pass.Equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(gemv.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    def check_rfactor_no_reset_multi_reduction(factor, rfactor):
        s = tvm.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        roo, roi = s[C].split(ro, factor=2)
        s[C].reorder(yo, roo, roi, yi, ri)
        gemv = intrin_gemv_no_reset(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        assert tvm.ir_pass.Equal(out_dom[x].extent, 1)
        assert tvm.ir_pass.Equal(out_dom[y].extent, factor)
        assert tvm.ir_pass.Equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(body[0]),
                                 tvm.ir_pass.CanonicalSimplify(gemv.op.body[0]))
        stmt = tvm.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    check(16)
    check_rfactor(4, 16)
    check_rfactor_no_reset(16, 16)
    check_rfactor_no_reset_multi_reduction(16, 16)

# This tests whether algorithm and intrinsics expressions are simplified
# as much as possible first and then checked for equality. See Issue #696
def test_tensorize_op():
    def op_intrin():
        bh = 9
        bw = 9
        x = tvm.placeholder((5, 5), name='A')
        y = tvm.compute((bh, bw), lambda i,j: x[j/3 + i%3, j%3+ i/3])

        def intrin_func(ins, outs):
            xx, = ins
            zz = outs[0]
            return tvm.call_packed("op", xx, zz)

        with tvm.build_config(offset_factor=2):
            return tvm.decl_tensor_intrin(y.op, intrin_func)

    A = tvm.placeholder((5, 5), name='A')
    B = tvm.compute((9,9), lambda i, j: A[j/3 + i%3, j%3 + i/3])
    bt = op_intrin()
    s = tvm.create_schedule(B.op)

    x,y = B.op.axis
    s[B].tensorize(x, bt)
    s = s.normalize()
    tvm.lower(s, [A, B])

if __name__ == "__main__":
    test_tensorize_vadd()
    test_tensorize_matmul()
    test_tensorize_op()
    test_tensorize_argmax()
