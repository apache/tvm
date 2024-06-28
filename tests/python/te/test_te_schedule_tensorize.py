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
from tvm import te
from tvm.script import tir as T


def intrin_vadd(xo, m, n):
    x = te.placeholder((n,), name="vx")
    y = te.placeholder((n,), name="vy")
    if m % n == 0:
        body = lambda i: x[i] + y[i]
    else:
        body = lambda i: tvm.tir.Select(
            xo * n + i < m, x[i] + y[i], tvm.tir.const(0, dtype=x.dtype)
        )
    z = te.compute(x.shape, body, name="z")

    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
        # special handle needed to tackle tail loop part when m % n != 0
        # here is tvm.min(n, m - xo * n)
        return tvm.tir.call_packed("vadd", xx, yy, zz)

    buffer_params = {"offset_factor": 16}
    return te.decl_tensor_intrin(z.op, intrin_func, default_buffer_params=buffer_params)


def intrin_gemv(m, n):
    w = te.placeholder((m, n), name="w")
    x = te.placeholder((n,), name="x")
    k = te.reduce_axis((0, n), name="k")
    z = te.compute((m,), lambda i: te.sum(w[i, k] * x[k], axis=k), name="z")
    Wb = tvm.tir.decl_buffer(
        w.shape, w.dtype, name="W", offset_factor=16, strides=[te.var("ldw"), 1]
    )

    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        ww_ptr = ww.access_ptr("r")
        xx_ptr = xx.access_ptr("r")
        zz_ptr = zz.access_ptr("w")
        body = tvm.tir.call_packed("gemv", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        reset = tvm.tir.call_packed("fill_zero", zz_ptr, n)
        update = tvm.tir.call_packed("gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, reset, update

    buffer_params = {"offset_factor": 16, "data_alignment": 16}
    return te.decl_tensor_intrin(
        z.op, intrin_func, binds={w: Wb}, default_buffer_params=buffer_params
    )


def intrin_gemv_no_reset(m, n):
    w = te.placeholder((m, n), name="w")
    x = te.placeholder((n,), name="x")
    k = te.reduce_axis((0, n), name="k")
    z = te.compute((m,), lambda i: te.sum(w[i, k] * x[k], axis=k), name="z")
    Wb = tvm.tir.decl_buffer(
        w.shape, w.dtype, name="W", offset_factor=16, strides=[te.var("ldw"), 1]
    )

    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        ww_ptr = ww.access_ptr("r")
        xx_ptr = xx.access_ptr("r")
        zz_ptr = zz.access_ptr("w")
        body = tvm.tir.call_packed("gemv", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        update = tvm.tir.call_packed("gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, None, update

    buffer_params = {"offset_factor": 16, "data_alignment": 16}
    return te.decl_tensor_intrin(
        z.op, intrin_func, binds={w: Wb}, default_buffer_params=buffer_params
    )


def test_tensorize_vadd():
    def add(m):
        x = te.placeholder((m,), name="x")
        y = te.placeholder((m,), name="y")
        z = te.compute(x.shape, lambda i: x[i] + y[i], name="z")
        return x, y, z

    def check(m, factor):
        x, y, z = add(m)
        factor = T.int32(factor)
        s = te.create_schedule(z.op)
        xo, xi = s[z].split(z.op.axis[0], factor=factor)
        vadd = intrin_vadd(xo, m, factor)
        s[z].tensorize(xi, vadd)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[z], dom_map)
        tvm.ir.assert_structural_equal(out_dom[z.op.axis[0]].extent, factor)
        tvm.ir.assert_structural_equal(out_dom[z.op.axis[0]].min, xo * factor)
        tvm.ir.assert_structural_equal(in_dom.items()[0][1][0].extent, factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[z], out_dom, in_dom, vadd)
        ana = tvm.arith.Analyzer()
        tvm.ir.assert_structural_equal(ana.simplify(body[0]), ana.simplify(vadd.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [x, y, z])

    def check_cache_write(m, factor):
        x, y, z = add(m)
        s = te.create_schedule(z.op)
        _, _ = s[z].split(z.op.axis[0], factor=factor)

        z_global = s.cache_write(z, "global")
        xo, xi = z_global.op.axis

        vadd = intrin_vadd(xo, m, factor)
        s[z_global].tensorize(xi, vadd)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[z_global], dom_map)
        # outer loop var will be rebased, so min value is the new loop var and extent is 1
        tvm.ir.assert_structural_equal(out_dom[xo].extent, T.int32(1))
        assert isinstance(out_dom[xo].min, tvm.tir.Var)
        assert xo.var.name == out_dom[xo].min.name

        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[z_global], out_dom, in_dom, vadd)[0]
        ana = tvm.arith.Analyzer()
        vars = tvm.runtime.convert({xo.var: out_dom[xo].min})
        vadd_body = tvm.tir.stmt_functor.substitute(vadd.op.body[0], vars)
        tvm.ir.assert_structural_equal(ana.simplify(body), ana.simplify(vadd_body))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [x, y, z])

    def check_compute_reuse():
        x, y, z = add(32)

        def _intrin_vadd():
            def _intrin_func(ins, outs):
                return tvm.tir.call_packed("vadd", ins[0], ins[1], outs[0])

            return tvm.te.decl_tensor_intrin(z.op, _intrin_func)

        s = tvm.te.create_schedule(z.op)
        s[z].tensorize(z.op.axis[0], _intrin_vadd())
        tvm.lower(s, [x, y, z])

    check(128, 16)
    check_cache_write(129, 16)
    check_compute_reuse()


def test_tensorize_matmul():
    n = 1024
    m = n
    l = n
    A = te.placeholder((n, l), name="A")
    B = te.placeholder((m, l), name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((n, m), lambda i, j: te.sum(B[j, k] * A[i, k], axis=k), name="C")

    def check(factor):
        s = te.create_schedule(C.op)
        x, y = C.op.axis
        yo, yi = s[C].split(y, factor=factor)
        gemv = intrin_gemv(factor, l)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        tvm.ir.assert_structural_equal(out_dom[x].extent, T.int32(1))
        tvm.ir.assert_structural_equal(out_dom[y].extent, factor)
        tvm.ir.assert_structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()

        tvm.ir.assert_structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    def check_rfactor(factor, rfactor):
        s = te.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        s[C].reorder(yo, ro, yi, ri)
        gemv = intrin_gemv(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        tvm.ir.assert_structural_equal(out_dom[x].extent, T.int32(1))
        tvm.ir.assert_structural_equal(out_dom[y].extent, factor)
        tvm.ir.assert_structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        tvm.ir.assert_structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    def check_rfactor_no_reset(factor, rfactor):
        s = te.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        s[C].reorder(yo, ro, yi, ri)
        gemv = intrin_gemv_no_reset(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        tvm.ir.assert_structural_equal(out_dom[x].extent, T.int32(1))
        tvm.ir.assert_structural_equal(out_dom[y].extent, factor)
        tvm.ir.assert_structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        tvm.ir.assert_structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    def check_rfactor_no_reset_multi_reduction(factor, rfactor):
        s = te.create_schedule(C.op)
        x, y = C.op.axis
        rk = C.op.reduce_axis[0]
        yo, yi = s[C].split(y, factor=factor)
        ro, ri = s[C].split(rk, factor=rfactor)
        roo, roi = s[C].split(ro, factor=2)
        s[C].reorder(yo, roo, roi, yi, ri)
        gemv = intrin_gemv_no_reset(factor, rfactor)
        s[C].tensorize(yi, gemv)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[C], dom_map)
        tvm.ir.assert_structural_equal(out_dom[x].extent, T.int32(1))
        tvm.ir.assert_structural_equal(out_dom[y].extent, factor)
        tvm.ir.assert_structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        tvm.ir.assert_structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    check(T.int32(16))
    check_rfactor(T.int32(16), T.int32(16))
    check_rfactor_no_reset(T.int32(16), T.int32(16))
    check_rfactor_no_reset_multi_reduction(T.int32(16), T.int32(16))


# This tests whether algorithm and intrinsics expressions are simplified
# as much as possible first and then checked for equality. See Issue #696
def test_tensorize_op():
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    def op_intrin():
        bh = 9
        bw = 9
        x = te.placeholder((5, 5), name="A")
        y = te.compute((bh, bw), lambda i, j: x[idxd(j, 3) + idxm(i, 3), idxm(j, 3) + idxd(i, 3)])

        def intrin_func(ins, outs):
            (xx,) = ins
            zz = outs[0]
            return tvm.tir.call_packed("op", xx, zz)

        return te.decl_tensor_intrin(y.op, intrin_func, default_buffer_params={"offset_factor": 2})

    A = te.placeholder((5, 5), name="A")
    B = te.compute((9, 9), lambda i, j: A[idxd(j, 3) + idxm(i, 3), idxm(j, 3) + idxd(i, 3)])
    bt = op_intrin()
    s = te.create_schedule(B.op)

    x, y = B.op.axis
    s[B].tensorize(x, bt)
    s = s.normalize()
    tvm.lower(s, [A, B])


# This test asserts that tensorize does not have any effect on
# TensorComputeOp operations
def test_tensorize_tensor_compute_op():
    # an intrinsic called "multivadd" whose definition (pattern)
    # is a loop of another intrinsic called "vadd"
    def intrin_multivadd(n):
        n_a = te.var("n_a")
        Ab = tvm.tir.decl_buffer((n,), "float32", strides=[n_a])

        n_b = te.var("n_b")
        Bb = tvm.tir.decl_buffer((n,), "float32", strides=[n_b])

        n_c = te.var("n_c")
        Cb = tvm.tir.decl_buffer((n,), "float32", strides=[n_c])

        z = te.compute(
            (n,),
            lambda i: tvm.tir.call_extern(
                "float32",
                "vadd",
                Ab.access_ptr("w", offset=n_a * i),
                Bb.access_ptr("r", offset=n_b * i),
                Cb.access_ptr("r", offset=n_c * i),
            ),
        )

        # replace the pattern with the multivadd call. I need to figure out
        # how to pass it the right parameters.
        def intrin_func(ins, outs):
            return tvm.tir.call_packed("multivadd")

        return te.decl_tensor_intrin(z.op, intrin_func, name="multivadd")

    def intrin_vadd(n):
        dtype = "float32"
        x = te.placeholder((n,), dtype=dtype, name="vx")
        y = te.placeholder((n,), dtype=dtype, name="vy")
        z = te.compute(x.shape, lambda i: x[i] + y[i], name="z")
        s = te.create_schedule(z.op)

        def create_buffer(t):
            return tvm.tir.decl_buffer(t.shape, t.dtype, name="W" + t.name, offset_factor=16)

        def intrin_func(ins, outs):
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_extern(
                    "float32",
                    "vadd",
                    ins[0].access_ptr("r"),
                    ins[1].access_ptr("r"),
                    outs[0].access_ptr("wr"),
                )
            )
            return ib.get()

        return te.decl_tensor_intrin(
            z.op, intrin_func, binds={x: create_buffer(x), y: create_buffer(y), z: create_buffer(z)}
        )

    # cache_read, cache_write
    M = 1024
    factor = 16
    dtype = "float32"

    A = te.placeholder((M // factor, factor), name="A", dtype=dtype)
    B = te.placeholder((M // factor, factor), name="B", dtype=dtype)

    vadd = intrin_vadd(factor)
    C = te.compute((M // factor, factor), lambda i: vadd(A[i, 0:factor], B[i, 0:factor]), name="C")

    s = te.create_schedule(C.op)
    multivadd = intrin_multivadd(64)
    s[C].tensorize(C.op.axis[0], multivadd)
    s = s.normalize()
    dom_map = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
    # The loop that we tried to tensorize still exists in the code
    # That means tensorize didn't work as expected
    assert isinstance(stmt.body, tvm.tir.For)
    assert stmt.body.loop_var.name == C.op.axis[0].var.name


if __name__ == "__main__":
    test_tensorize_vadd()
    test_tensorize_matmul()
    test_tensorize_op()
    test_tensorize_tensor_compute_op()
