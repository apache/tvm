# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
    check_rfactor(16, 16)
    check_rfactor_no_reset(16, 16)
    check_rfactor_no_reset_multi_reduction(16, 16)

# This tests whether algorithm and intrinsics expressions are simplified
# as much as possible first and then checked for equality. See Issue #696
def test_tensorize_op():
    idxd = tvm.indexdiv
    idxm = tvm.indexmod

    def op_intrin():
        bh = 9
        bw = 9
        x = tvm.placeholder((5, 5), name='A')
        y = tvm.compute((bh, bw),
                        lambda i, j: x[idxd(j,3) + idxm(i,3), idxm(j,3)+ idxd(i,3)])

        def intrin_func(ins, outs):
            xx, = ins
            zz = outs[0]
            return tvm.call_packed("op", xx, zz)

        with tvm.build_config(offset_factor=2):
            return tvm.decl_tensor_intrin(y.op, intrin_func)

    A = tvm.placeholder((5, 5), name='A')
    B = tvm.compute((9,9), lambda i, j: A[idxd(j,3) + idxm(i,3), idxm(j,3) + idxd(i,3)])
    bt = op_intrin()
    s = tvm.create_schedule(B.op)

    x,y = B.op.axis
    s[B].tensorize(x, bt)
    s = s.normalize()
    tvm.lower(s, [A, B])

# This test asserts that tensorize does not have any effect on
# TensorComputeOp operations
def test_tensorize_tensor_compute_op():
    # an intrinsic called "multivadd" whose definition (pattern)
    # is a loop of another intrinsic called "vadd"
    def intrin_multivadd(n):
        n_a = tvm.var("n_a")
        Ab = tvm.decl_buffer((n, ), tvm.float32, strides=[n_a])

        n_b = tvm.var("n_b")
        Bb = tvm.decl_buffer((n, ), tvm.float32, strides=[n_b])

        n_c = tvm.var("n_c")
        Cb = tvm.decl_buffer((n, ), tvm.float32, strides=[n_c])

        z = tvm.compute((n,), lambda i: tvm.call_extern("float32", 'vadd',
                                                        Ab.access_ptr("w", offset=n_a*i),
                                                        Bb.access_ptr("r", offset=n_b*i),
                                                        Cb.access_ptr("r", offset=n_c*i)))

        # replace the pattern with the multivadd call. I need to figure out
        # how to pass it the right parameters.
        def intrin_func(ins, outs):
            return tvm.call_packed("multivadd")

        with tvm.build_config():
            return tvm.decl_tensor_intrin(z.op, intrin_func, name="multivadd")

    def intrin_vadd(n):
        dtype = 'float32'
        x = tvm.placeholder((n,), dtype=dtype, name='vx')
        y = tvm.placeholder((n,), dtype=dtype, name='vy')
        z = tvm.compute(x.shape, lambda i: x[i] + y[i], name='z')
        s = tvm.create_schedule(z.op)

        def create_buffer(t):
            return tvm.decl_buffer(t.shape, t.dtype,
                                   name='W'+t.name,
                                   offset_factor=16)

        def intrin_func(ins, outs):
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("float32", 'vadd',
                                    ins[0].access_ptr("r"), ins[1].access_ptr('r'),
                                    outs[0].access_ptr('wr')))
            return ib.get()

        with tvm.build_config(offset_factor=16):
            return tvm.decl_tensor_intrin(z.op, intrin_func, binds={x: create_buffer(x),
                                                                    y: create_buffer(y),
                                                                    z: create_buffer(z)})

    # cache_read, cache_write
    M = 1024
    factor = 16
    dtype = 'float32'

    A = tvm.placeholder((M//factor, factor), name="A", dtype=dtype)
    B = tvm.placeholder((M//factor, factor), name="B", dtype=dtype)

    vadd = intrin_vadd(factor)
    C = tvm.compute((M//factor, factor),
                    lambda i: vadd(A[i, 0:factor], B[i, 0:factor]), name='C')

    s = tvm.create_schedule(C.op)
    multivadd = intrin_multivadd(64)
    s[C].tensorize(C.op.axis[0], multivadd)
    s = s.normalize()
    dom_map = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, dom_map)
    # The loop that we tried to tensorize still exists in the code
    # That means tensorize didn't work as expected
    assert isinstance(stmt.body.body.body, tvm.stmt.For)
    assert stmt.body.body.body.loop_var.name == C.op.axis[0].var.name



if __name__ == "__main__":
    test_tensorize_vadd()
    test_tensorize_matmul()
    test_tensorize_op()
    test_tensorize_tensor_compute_op()
