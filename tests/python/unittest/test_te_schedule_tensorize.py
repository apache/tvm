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


def intrin_vadd(n):
    x = te.placeholder((n,), name="vx")
    y = te.placeholder((n,), name="vy")
    z = te.compute(x.shape, lambda i: x[i] + y[i], name="z")

    def intrin_func(ins, outs):
        xx, yy = ins
        zz = outs[0]
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
    m = 128
    x = te.placeholder((m,), name="x")
    y = te.placeholder((m,), name="y")
    z = te.compute(x.shape, lambda i: x[i] + y[i], name="z")

    def check(factor):
        s = te.create_schedule(z.op)
        xo, xi = s[z].split(z.op.axis[0], factor=factor)
        vadd = intrin_vadd(factor)
        s[z].tensorize(xi, vadd)
        s = s.normalize()
        dom_map = tvm.te.schedule.InferBound(s)
        finfer = tvm.get_global_func("test.op.InferTensorizeRegion")
        out_dom, in_dom = finfer(s[z], dom_map)
        assert tvm.ir.structural_equal(out_dom[z.op.axis[0]].extent, factor)
        assert tvm.ir.structural_equal(out_dom[z.op.axis[0]].min, xo * factor)
        assert tvm.ir.structural_equal(in_dom.items()[0][1][0].extent, factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[z], out_dom, in_dom, vadd)
        ana = tvm.arith.Analyzer()
        assert tvm.ir.structural_equal(ana.simplify(body[0]), ana.simplify(vadd.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [x, y, z])

    check(16)


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
        assert tvm.ir.structural_equal(out_dom[x].extent, 1)
        assert tvm.ir.structural_equal(out_dom[y].extent, factor)
        assert tvm.ir.structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()

        assert tvm.ir.structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
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
        assert tvm.ir.structural_equal(out_dom[x].extent, 1)
        assert tvm.ir.structural_equal(out_dom[y].extent, factor)
        assert tvm.ir.structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        assert tvm.ir.structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
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
        assert tvm.ir.structural_equal(out_dom[x].extent, 1)
        assert tvm.ir.structural_equal(out_dom[y].extent, factor)
        assert tvm.ir.structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        assert tvm.ir.structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
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
        assert tvm.ir.structural_equal(out_dom[x].extent, 1)
        assert tvm.ir.structural_equal(out_dom[y].extent, factor)
        assert tvm.ir.structural_equal(out_dom[y].min, yo * factor)
        fmatch = tvm.get_global_func("test.op.MatchTensorizeBody")
        body = fmatch(s[C], out_dom, in_dom, gemv)
        ana = tvm.arith.Analyzer()
        assert tvm.ir.structural_equal(ana.simplify(body[0]), ana.simplify(gemv.op.body[0]))
        stmt = tvm.te.schedule.ScheduleOps(s, dom_map)
        tvm.lower(s, [A, B, C])

    check(16)
    check_rfactor(16, 16)
    check_rfactor_no_reset(16, 16)
    check_rfactor_no_reset_multi_reduction(16, 16)


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
    assert isinstance(stmt.body.body, tvm.tir.For)
    assert stmt.body.body.loop_var.name == C.op.axis[0].var.name


if __name__ == "__main__":
    test_tensorize_vadd()
    test_tensorize_matmul()
    test_tensorize_op()
    test_tensorize_tensor_compute_op()
