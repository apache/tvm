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
import pytest
import tvm
import tvm.testing
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy


def collect_visit(stmt, f):
    ret = []
    tvm.tir.stmt_functor.post_order_visit(stmt, lambda x: ret.append(f(x)))
    return ret


def test_basic():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"]

    assert not any(collect_visit(stmt.body.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse)))
    assert any(collect_visit(stmt.body.body[1], lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_const_loop():
    n = 21
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_no_unroll_loop():
    n = 21
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(
        config={
            "tir.LoopPartition": {
                "partition_const_loop": True,
                "no_unroll_loop_with_extent_one": True,
            }
        }
    ):
        mod = tvm.tir.transform.LoopPartition()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        stmt = tvm.tir.transform.RemoveNoOp()(mod)["main"].body

    assert sum(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.For))) == 4


def test_multi_loop():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var("m")
    n = te.size_var("n")
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i * m + j + k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n, m], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_multi_if():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var("m")
    n = te.size_var("n")
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i * m + j + k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
                with ib.if_scope(ib.likely(i * m + j - k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_thread_axis():
    m = te.size_var("m")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")
    B = te.compute((m, l), lambda i, j: A[i, j] + 3, name="B")
    s = te.create_schedule(B.op)

    s[B].set_scope("shared")
    num_thread = 16
    xo, xi = s[B].split(B.op.axis[0], 32)
    xi0, xi1 = s[B].split(xi, nparts=num_thread)
    s[B].bind(xi0, te.thread_axis("threadIdx.x"))

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"]

    assert not any(collect_visit(stmt.body.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_vectorize():
    n = te.size_var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    bias = te.size_var("bias", dtype="float32")
    scale = te.size_var("scale", dtype="float32")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i) * scale + bias, name="C")
    # schedule
    s = te.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=num_thread * 4)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=4)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].vectorize(x)
    stmt = tvm.lower(s, [A, B], name="main")["main"]
    body = stmt.body.body.body.body
    assert x.var.name not in str(body.condition)
    assert any(collect_visit(body.then_case, lambda x: isinstance(x, tvm.tir.Ramp)))


def test_condition():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var("m")
    n = te.size_var("n")
    with ib.for_range(0, tvm.tir.truncdiv(n + 3, 4), "i") as i:
        with ib.for_range(0, 4, "j") as j:
            ib.emit(tvm.tir.Evaluate(tvm.tir.Select(ib.likely(i * 4 + j < n), m, n)))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm.tir.Select)))


def test_condition_EQ():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var("m")
    n = te.size_var("n")
    with ib.for_range(0, 10, "i") as i:
        ib.emit(tvm.tir.Evaluate(tvm.tir.Select(ib.likely(tvm.tir.EQ(i, 5)), m, n)))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt).with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm.tir.Select)))


def test_thread_axis2():
    n = tvm.runtime.convert(4096)
    m = te.size_var("m")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    s = te.create_schedule(C.op)
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=32)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=m)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    stmt = tvm.lower(s, [A, B], name="main")["main"]
    for_body = stmt.body.body.body.body[0]
    assert "threadIdx" not in str(for_body.extent)


def test_everything_during_deduction():
    m = te.size_var("m")
    n = te.size_var("n")
    ib = tvm.tir.ir_builder.create()
    with ib.for_range(0, n, "i") as i:
        with ib.for_range(0, 32, "j") as j:
            with ib.if_scope(ib.likely(tvm.tir.truncdiv(i, j) < m)):
                # this guard will produce everything during deduction
                ib.emit(tvm.tir.Evaluate(m))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt).with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.LoopPartition()(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert isinstance(stmt.body.body, tvm.tir.IfThenElse)


def test_single_likely():
    n = 60
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    T = te.compute((n,), lambda i: A[i] + B[i])
    s = te.create_schedule(T.op)
    x = T.op.axis[0]
    xo, xi = s[T].split(x, factor=16)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_multi_likely():
    n = 94
    m = 62
    A = te.placeholder((n, m), name="A")
    B = te.placeholder((n, m), name="B")

    T = te.compute((n, m), lambda i, j: A[i, j] + B[i, j])
    s = te.create_schedule(T.op)
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    x, y = T.op.axis
    xo, xi = s[T].split(x, factor=16)
    yo, yi = s[T].split(y, factor=16)
    s[T].reorder(xo, yo, xi, yi)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_oneD_pool():
    m = te.size_var("m")
    ib = tvm.tir.ir_builder.create()
    # data = te.placeholder((16,), name = 'data')
    data = ib.pointer("float32", name="A")
    out = ib.pointer("float32", name="A")
    with ib.for_range(0, 16, "ow") as ow:
        with ib.for_range(0, 3, "kw") as kw:
            with ib.if_scope(ib.likely(ow > 0)):
                with ib.if_scope(ib.likely(ow < 15)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, "ow") as ow:
        with ib.for_range(0, 3, "kw") as kw:
            with ib.if_scope(ib.likely(ow < 1)):
                with ib.if_scope(ib.likely(kw > 0)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, "ow") as ow:
        with ib.for_range(0, 3, "kw") as kw:
            with ib.if_scope(ib.likely(ow > 14)):
                with ib.if_scope(ib.likely(kw < 2)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([m, data, out], stmt).with_attr("global_symbol", "main")
    )

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_cce_loop_1():
    ib = tvm.tir.ir_builder.create()
    dtype = "float16"
    n = 514
    m = 514
    _A = te.placeholder((n * m,), name="A")
    Ab = tvm.tir.decl_buffer((n * m,), dtype, name="A")
    A = ib.buffer_ptr(Ab)
    _B = te.placeholder((n * m,), name="B")
    Bb = tvm.tir.decl_buffer((n * m,), dtype, name="B")
    B = ib.buffer_ptr(Bb)
    # for i in 0 to n-1:
    with ib.for_range(0, 11, name="i") as i:
        with ib.for_range(0, 160, name="j") as j:
            with ib.if_scope(ib.likely(((i * 160) + j) < 1600)):
                A[(i + 1) * m + j + 1] = (
                    B[(i) * m + j + 1] + B[(i + 1) * m + j + 1] + B[(i + 2) * m + j + 1]
                )
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc([Ab, Bb], stmt).with_attr("global_symbol", "main")
    )
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_cce_loop_2():
    ib = tvm.tir.ir_builder.create()
    len = 112
    tile = 32
    loop = (len + tile - 1) // tile
    with ib.for_range(0, loop, "i") as i:
        head = i * tile
        with ib.if_scope(ib.likely(head + tile > len)):
            tail = len
            ib.emit(tvm.tir.call_extern("float32", "cce_intrisic", head, tail))
        with ib.else_scope():
            tail = head + tile
            ib.emit(tvm.tir.call_extern("float32", "cce_intrisic", head, tail))

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_cce_loop_3():
    ib = tvm.tir.ir_builder.create()
    loop1 = 4
    loop2 = 9998
    tile = 39991
    with ib.for_range(0, loop2, "i") as i:
        with ib.for_range(0, loop1, "j") as j:
            head1 = i
            head2 = j
            with ib.if_scope(ib.likely(head1 * loop1 + head2 < tile)):
                ib.emit(tvm.tir.call_extern("float16", "cce_intrisic", head1))

    stmt = ib.get()
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_conv_tiling():
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = te.placeholder((batch_size, in_channel, in_height, in_width), name="data")
    kernel = te.placeholder((kernel_height, kernel_width, in_channel, out_channel), name="kernel")
    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")
    conv = te.compute(
        (batch_size, out_channel, out_height, out_width),
        lambda n, oc, oh, ow: te.sum(
            data[n, ic, oh * HSTR + kh, ow * WSTR + kw] * kernel[kh, kw, ic, oc], axis=[ic, kh, kw]
        ),
        name="conv2d",
    )
    s = te.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt).with_attr("global_symbol", "main"))
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.LoopPartition()(mod)
        stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse)))


def test_multilevel_splitting_with_indivisble_factors():
    from tvm import topi

    A = te.placeholder((130,), dtype="float32")
    B = topi.nn.relu(A)
    s = te.create_schedule(B.op)
    (y,) = s[B].op.axis
    (yo, yi) = s[B].split(y, factor=8)
    (yoo, yoi) = s[B].split(yo, factor=16)
    s[B].reorder(yoo, yoi, yi)
    s[B].unroll(yi)

    ## But this does the right thing.
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        lowered_body = tvm.lower(s, [A, B], name="x")["x"].body

        def visit_stmt(op):
            return isinstance(op, tvm.tir.Max)

        num_max = collect_visit(lowered_body, visit_stmt)
        assert num_max.count(True) == 10


def test_double_splitting_with_indivisible_factors():
    m = 48
    dtype = "float32"
    A = te.placeholder((m,), name="A", dtype=dtype)
    C = te.compute((m,), lambda i: A[i], name="C")
    D = te.compute((m,), lambda i: C[i], name="D")

    s = te.create_schedule(D.op)
    co, ci = s[C].split(C.op.axis[0], factor=10)
    do, di = s[D].split(D.op.axis[0], 32)
    s[C].compute_at(s[D], do)

    target = "llvm"
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        f = tvm.lower(s, [A, C, D], name="fadd1", simple_mode=False)
        func = tvm.build(f, target=target)

    top_produce = f["fadd1"].body
    assert not any(collect_visit(top_produce, lambda x: isinstance(x, tvm.tir.IfThenElse)))

    # check functional correctness of generated code
    dev = tvm.device(target, 0)
    a = tvm.nd.array(
        numpy.ones(
            m,
        ).astype(dtype),
        dev,
    )
    c = tvm.nd.array(
        numpy.zeros(
            m,
        ).astype(dtype),
        dev,
    )
    d = tvm.nd.array(
        numpy.zeros(
            m,
        ).astype(dtype),
        dev,
    )
    func(a, c, d)
    tvm.testing.assert_allclose(c.numpy(), a.numpy(), rtol=1e-5)
    tvm.testing.assert_allclose(d.numpy(), a.numpy(), rtol=1e-5)


def test_simple_rfactor():
    K = 16 * 4 + 4
    k = te.reduce_axis((0, K), "k")

    A = te.placeholder((1, K), name="A")

    B = te.compute((1,), lambda b: te.sum(A[b, k], axis=k), name="B")

    s = te.create_schedule(B.op)
    ko, _ = s[B].split(s[B].op.reduce_axis[0], 16)
    BF = s.rfactor(B, ko, 0)

    s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt1 = tvm.te.schedule.ScheduleOps(s, bounds)

    mod1 = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt1).with_attr("global_symbol", "main"))
    stmt1 = tvm.tir.transform.Simplify()(mod1)["main"].body

    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod2 = tvm.tir.transform.LoopPartition()(mod1)
        stmt2 = tvm.tir.transform.Simplify()(mod2)["main"].body

    # make sure loop partition actually did something
    assert not tvm.ir.structural_equal(stmt1.body, stmt2.body)


@T.prim_func
def partitioned_concat(
    A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32"), C: T.Buffer((32,), "float32")
) -> None:
    T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
    for i in T.serial(0, 16):
        C[i] = A[i]
    for i in T.serial(0, 16):
        C[i + 16] = B[i + 16]


def test_explicit_partition_hint():
    A = te.placeholder((16,), name="A")
    B = te.placeholder((16,), name="B")
    C = te.compute((32,), lambda i: te.if_then_else(i < 16, A[i], B[i]), name="C")
    s = te.create_schedule(C.op)
    s.normalize()
    s[C].pragma(s[C].op.axis[0], "loop_partition_hint", True)
    mod = tvm.driver.build_module.schedule_to_module(s, [A, B, C], "main", None)
    with tvm.transform.PassContext(config={"tir.LoopPartition": {"partition_const_loop": True}}):
        mod = tvm.tir.transform.StorageFlatten(64)(mod)
        mod = tvm.tir.transform.LoopPartition()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], partitioned_concat)


def partition_from_scheduled_tir(prim_func, pass_cfg, do_flatten=True):
    with tvm.transform.PassContext(config=pass_cfg):
        mod = IRModule.from_expr(prim_func.with_attr("global_symbol", "main"))
        mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
        if do_flatten:
            mod = tvm.tir.transform.FlattenBuffer()(mod)
        mod = tvm.tir.transform.LoopPartition()(mod)
        mod = tvm.tir.transform.Simplify()(mod)
        mod = tvm.tir.transform.RemoveNoOp()(mod)
        return mod


@T.prim_func
def partitioned_concat_3(
    placeholder: T.Buffer((1, 64, 28, 28), "int8"),
    placeholder_1: T.Buffer((1, 32, 28, 28), "int8"),
    placeholder_2: T.Buffer((1, 32, 28, 28), "int8"),
    T_concat: T.Buffer((1, 128, 28, 28), "int8"),
) -> None:
    placeholder_flat = T.Buffer([50176], "int8", data=placeholder.data)
    placeholder_1_flat = T.Buffer([25088], "int8", data=placeholder_1.data)
    placeholder_2_flat = T.Buffer([25088], "int8", data=placeholder_2.data)
    T_concat_flat = T.Buffer([100352], "int8", data=T_concat.data)
    for i1, i2, i3 in T.grid(64, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_flat[i1 * 784 + i2 * 28 + i3]
    for i1, i2, i3 in T.grid(32, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3 + 50176] = placeholder_1_flat[i1 * 784 + i2 * 28 + i3]
    for i1, i2, i3 in T.grid(32, 28, 28):
        T_concat_flat[i1 * 784 + i2 * 28 + i3 + 75264] = placeholder_2_flat[i1 * 784 + i2 * 28 + i3]


@T.prim_func
def concat_func_3(
    placeholder: T.Buffer((1, 64, 28, 28), "int8"),
    placeholder_1: T.Buffer((1, 32, 28, 28), "int8"),
    placeholder_2: T.Buffer((1, 32, 28, 28), "int8"),
    T_concat: T.Buffer((1, 128, 28, 28), "int8"),
) -> None:
    placeholder_flat = T.Buffer([50176], "int8", data=placeholder.data)
    placeholder_1_flat = T.Buffer([25088], "int8", data=placeholder_1.data)
    placeholder_2_flat = T.Buffer([25088], "int8", data=placeholder_2.data)
    T_concat_flat = T.Buffer([100352], "int8", data=T_concat.data)
    for i1 in T.serial(128, annotations={"pragma_loop_partition_hint": 1}):
        for i2, i3 in T.grid(28, 28):
            if 96 <= i1:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_2_flat[
                    i1 * 784 + i2 * 28 + i3 - 75264
                ]
            if 64 <= i1 and i1 < 96:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_1_flat[
                    i1 * 784 + i2 * 28 + i3 - 50176
                ]
            if i1 < 64:
                T_concat_flat[i1 * 784 + i2 * 28 + i3] = placeholder_flat[i1 * 784 + i2 * 28 + i3]


def test_condition_mutually_exclusive():
    mod = partition_from_scheduled_tir(
        concat_func_3, {"tir.LoopPartition": {"partition_const_loop": True}}
    )
    tvm.ir.assert_structural_equal(
        mod["main"], partitioned_concat_3.with_attr("global_symbol", "main")
    )


def test_loop_partition_unroll_hint():
    @T.prim_func
    def main(
        A_arg: T.Buffer((1, 3, 224, 224), "int8"), B_arg: T.Buffer((1, 224, 7, 16), "int8")
    ) -> None:
        A = T.Buffer(150528, "int8", data=A_arg.data)
        B = T.Buffer(25088, "int8", data=B_arg.data)
        for ax0 in T.serial(
            112,
            annotations={"pragma_loop_partition_hint": True},
        ):
            for ax1, ax2, ax3 in T.grid(224, 7, 16):
                if 3 <= ax0 * 2 + ax2 and ax0 * 2 + ax2 < 227 and ax3 < 3:
                    B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax0 * 2 + ax2 - 3]

    @T.prim_func
    def partitioned_main(
        A_arg: T.Buffer((1, 3, 224, 224), "int8"), B_arg: T.Buffer((1, 224, 7, 16), "int8")
    ) -> None:
        A = T.Buffer(150528, dtype="int8", data=A_arg.data)
        B = T.Buffer(25088, dtype="int8", data=B_arg.data)
        # body
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if 3 <= ax2 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 - 3]
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if 1 <= ax2 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 - 1]
        for ax0, ax1, ax2, ax3 in T.grid(109, 224, 7, 16):
            if ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax0 * 2 + ax2 + 1]
        for ax1, ax2, ax3 in T.grid(224, 7, 16):
            if ax2 < 5 and ax3 < 3:
                B[ax1 * 112 + ax2 * 16 + ax3] = A[ax3 * 50176 + ax1 * 224 + ax2 + 219]

    mod = partition_from_scheduled_tir(
        main,
        {
            "tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    mod = tvm.tir.transform.UnrollLoop()(mod)
    mod = tvm.tir.transform.RemoveNoOp()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], partitioned_main.with_attr("global_symbol", "main"))


def test_loop_partition_recursive_unroll_hint():
    @T.prim_func
    def main():
        placeholder_0_dm = T.decl_buffer([1, 32, 32, 16], dtype="int8")
        for i3_0 in T.serial(5, annotations={"pragma_loop_partition_hint": 1}):
            for i2_0 in T.serial(2, annotations={"pragma_loop_partition_hint": 1}):
                pad_temp = T.decl_buffer([1, 16, 16, 16], dtype="int8")
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if (
                        6 <= i2_0 * 4 + ax0
                        and i2_0 * 4 + ax0 < 26
                        and 6 <= i3_0 * 4 + ax1
                        and i3_0 * 4 + ax1 < 26
                    ):
                        pad_temp[
                            0,
                            i2_0 * 4 + ax0 - 6 + 6 - i2_0 * 4,
                            i3_0 * 4 + ax1 - 6 + 6 - i3_0 * 4,
                            ax2,
                        ] = placeholder_0_dm[
                            0,
                            i2_0 * 4 + ax0 - 6 - -6,
                            i3_0 * 4 + ax1 - 6 - -6,
                            ax2,
                        ]

    @T.prim_func
    def partitioned_main():
        placeholder_0_dm = T.allocate([16384], "int8", "global")
        placeholder_0_dm_1 = T.Buffer([16384], dtype="int8", data=placeholder_0_dm)
        for i3_0 in T.unroll(2):
            for i2_0 in T.unroll(2):
                pad_temp = T.allocate([4096], "int8", "global")
                pad_temp_1 = T.Buffer([4096], dtype="int8", data=pad_temp)
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if 6 <= i2_0 * 4 + ax0 and 6 <= i3_0 * 4 + ax1:
                        pad_temp_1[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm_1[
                            i2_0 * 2048 + ax0 * 512 + i3_0 * 64 + ax1 * 16 + ax2
                        ]
        for i2_0 in T.unroll(2):
            pad_temp_2 = T.allocate([4096], "int8", "global")
            pad_temp_3 = T.Buffer([4096], dtype="int8", data=pad_temp_2)
            for ax0, ax1, ax2 in T.grid(16, 16, 16):
                if 6 <= i2_0 * 4 + ax0:
                    pad_temp_3[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm_1[
                        i2_0 * 2048 + ax0 * 512 + ax1 * 16 + ax2 + 128
                    ]
        for i3_0 in T.unroll(2):
            for i2_0 in T.unroll(2):
                pad_temp_4 = T.allocate([4096], "int8", "global")
                pad_temp_5 = T.Buffer([4096], dtype="int8", data=pad_temp_4)
                for ax0, ax1, ax2 in T.grid(16, 16, 16):
                    if 6 <= i2_0 * 4 + ax0 and i3_0 * 4 + ax1 < 14:
                        pad_temp_5[ax0 * 256 + ax1 * 16 + ax2] = placeholder_0_dm_1[
                            i2_0 * 2048 + ax0 * 512 + i3_0 * 64 + ax1 * 16 + ax2 + 192
                        ]

    mod = partition_from_scheduled_tir(
        main,
        {
            "tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    tvm.ir.assert_structural_equal(mod["main"], partitioned_main.with_attr("global_symbol", "main"))


def test_loop_partition_keep_loop_annotations():
    @T.prim_func
    def before(A: T.Buffer(160, "int32"), B: T.Buffer(160, "int32")) -> None:
        for i in T.serial(
            160,
            annotations={"pragma_loop_partition_hint": True, "key": "value"},
        ):
            if i < 10:
                B[i] = A[i] + 1
            elif 10 <= i and i < 150:
                B[i] = A[i] + 2
            else:
                B[i] = A[i] + 3

    @T.prim_func
    def after(A: T.Buffer(160, "int32"), B: T.Buffer(160, "int32")) -> None:
        for i in T.serial(10, annotations={"key": "value"}):
            B[i] = A[i] + 1
        for i in T.serial(140, annotations={"key": "value"}):
            B[i + 10] = A[i + 10] + 2
        for i in T.serial(10, annotations={"key": "value"}):
            B[i + 150] = A[i + 150] + 3

    mod = partition_from_scheduled_tir(
        before,
        {
            "tir.LoopPartition": {
                "partition_const_loop": True,
            }
        },
    )
    tvm.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


def test_loop_partition_with_unit_loop_in_condition():
    @T.prim_func
    def before(
        placeholder: T.Buffer((50176,), "int8"),
        placeholder_1: T.Buffer((25088,), "int8"),
        placeholder_2: T.Buffer((25088,), "int8"),
        T_concat: T.Buffer((100352,), "int8"),
    ) -> None:
        for k in range(1, annotations={"preserve_unit_loop": True}):
            for i1 in T.serial(128, annotations={"pragma_loop_partition_hint": 1}):
                for i2, i3 in T.grid(28, 28):
                    if 96 <= k * 128 + i1:
                        T_concat[k * i1 * 784 + i2 * 28 + i3] = placeholder_2[
                            i1 * 784 + i2 * 28 + i3 - 75264
                        ]
                    if 64 <= k * 128 + i1 and k * 128 + i1 < 96:
                        T_concat[i1 * 784 + i2 * 28 + i3] = placeholder_1[
                            i1 * 784 + i2 * 28 + i3 - 50176
                        ]
                    if k * 128 + i1 < 64:
                        T_concat[i1 * 784 + i2 * 28 + i3] = placeholder[i1 * 784 + i2 * 28 + i3]

    @T.prim_func
    def after(
        placeholder: T.Buffer(50176, "int8"),
        placeholder_1: T.Buffer(25088, "int8"),
        placeholder_2: T.Buffer(25088, "int8"),
        T_concat: T.Buffer(100352, "int8"),
    ) -> None:
        for _ in T.serial(1, annotations={"preserve_unit_loop": True}):
            for i1, i2, i3 in T.grid(64, 28, 28):
                T_concat[i1 * 784 + i2 * 28 + i3] = placeholder[i1 * 784 + i2 * 28 + i3]
            for i1, i2, i3 in T.grid(32, 28, 28):
                T_concat[i1 * 784 + i2 * 28 + i3 + 50176] = placeholder_1[i1 * 784 + i2 * 28 + i3]
            for i1, i2, i3 in T.grid(32, 28, 28):
                T_concat[i2 * 28 + i3] = placeholder_2[i1 * 784 + i2 * 28 + i3]

    mod = partition_from_scheduled_tir(
        before,
        {
            "tir.LoopPartition": {
                "partition_const_loop": True,
            }
        },
    )
    tvm.ir.assert_structural_equal(mod["main"], after.with_attr("global_symbol", "main"))


@T.prim_func
def concat_func_single_point(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in T.serial(128, annotations={"pragma_loop_partition_hint": 1}):
            if i1 > 63:
                T_concat[i0, i1] = placeholder[i0, i1 - 64]
            elif i1 == 63:
                T_concat[i0, i1] = placeholder_1[i0, i1 - 63]
            else:
                T_concat[i0, i1] = placeholder_2[i0, i1]


@T.prim_func
def expected_partitioned_concat_single_point(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
):
    for i0 in range(28):
        T_concat_1 = T.Buffer((3584,), "int8", data=T_concat.data)
        for i1 in range(63):
            placeholder_2_1 = T.Buffer((1764,), "int8", data=placeholder_2.data)
            T_concat_1[i0 * 128 + i1] = placeholder_2_1[i0 * 63 + i1]
        placeholder_1_1 = T.Buffer((28,), "int8", data=placeholder_1.data)
        T_concat_1[i0 * 128 + 63] = placeholder_1_1[i0]
        for i1 in range(64):
            placeholder_3 = T.Buffer((1792,), "int8", data=placeholder.data)
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]


@T.prim_func
def concat_func_start_point_equality(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(128, annotations={"pragma_loop_partition_hint": 1}):
            if i1 == 0:
                # Special case for i1 == 0
                T_concat[i0, i1] = placeholder_1[i0, 0]
            elif i1 < 64:
                # Normal case for i1 in [1, 63]
                T_concat[i0, i1] = placeholder_2[i0, i1]
            else:
                # Case for i1 in [64, 127]
                T_concat[i0, i1] = placeholder[i0, i1 - 64]


@T.prim_func
def concat_func_start_point_equality_expected(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
):
    for i0 in range(28):
        T_concat_1 = T.Buffer((3584,), "int8", data=T_concat.data)
        placeholder_1_1 = T.Buffer((28,), "int8", data=placeholder_1.data)
        T_concat_1[i0 * 128] = placeholder_1_1[i0]
        for i1 in range(63):
            placeholder_2_1 = T.Buffer((1764,), "int8", data=placeholder_2.data)
            T_concat_1[i0 * 128 + i1 + 1] = placeholder_2_1[i0 * 63 + i1 + 1]
        for i1 in range(64):
            placeholder_3 = T.Buffer((1792,), "int8", data=placeholder.data)
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]


@T.prim_func
def concat_func_end_point_equality(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(128, annotations={"pragma_loop_partition_hint": 1}):
            if i1 == 127:
                # Explicit equality check for the end point i1 == 127
                T_concat[i0, i1] = placeholder_1[i0, 0]
            elif i1 >= 64:
                # Case for i1 in [64, 126]
                T_concat[i0, i1] = placeholder[i0, i1 - 64]
            else:
                # Case for i1 in [0, 63]
                T_concat[i0, i1] = placeholder_2[i0, i1]


@T.prim_func
def concat_func_end_point_equality_expected(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 63), "int8"),
    T_concat: T.Buffer((28, 128), "int8"),
):
    for i0 in range(28):
        T_concat_1 = T.Buffer((3584,), "int8", data=T_concat.data)
        for i1 in range(64):
            placeholder_2_1 = T.Buffer((1764,), "int8", data=placeholder_2.data)
            T_concat_1[i0 * 128 + i1] = placeholder_2_1[i0 * 63 + i1]
        for i1 in range(63):
            placeholder_3 = T.Buffer((1792,), "int8", data=placeholder.data)
            T_concat_1[i0 * 128 + i1 + 64] = placeholder_3[i0 * 64 + i1]
        placeholder_1_1 = T.Buffer((28,), "int8", data=placeholder_1.data)
        T_concat_1[i0 * 128 + 127] = placeholder_1_1[i0]


@T.prim_func
def concat_func_edge_equalities(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 1), "int8"),
    T_concat: T.Buffer((28, 66), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(
            66, annotations={"pragma_loop_partition_hint": 1}
        ):  # Loop from 0 to 65 inclusive
            if i1 == 0:
                # Handle equality at the start of the range: i1 == 0
                T_concat[i0, i1] = placeholder_2[i0, 0]
            elif i1 == 65:
                # Handle equality at the end of the range: i1 == 65
                T_concat[i0, i1] = placeholder_1[i0, 0]
            else:
                # Copying from placeholder (from 0 to 63)
                T_concat[i0, i1] = placeholder[i0, i1 - 1]


@T.prim_func
def concat_func_edge_equalities_expected(
    placeholder: T.Buffer((28, 64), "int8"),
    placeholder_1: T.Buffer((28, 1), "int8"),
    placeholder_2: T.Buffer((28, 1), "int8"),
    T_concat: T.Buffer((28, 66), "int8"),
):
    for i0 in range(28):
        T_concat_1 = T.Buffer((1848,), "int8", data=T_concat.data)
        placeholder_2_1 = T.Buffer((28,), "int8", data=placeholder_2.data)
        T_concat_1[i0 * 66] = placeholder_2_1[i0]
        for i1 in range(64):
            placeholder_3 = T.Buffer((1792,), "int8", data=placeholder.data)
            T_concat_1[i0 * 66 + i1 + 1] = placeholder_3[i0 * 64 + i1]
        placeholder_1_1 = T.Buffer((28,), "int8", data=placeholder_1.data)
        T_concat_1[i0 * 66 + 65] = placeholder_1_1[i0]


@T.prim_func
def concat_five_buffers_with_equalities(
    buffer_a: T.Buffer((28, 1), "int8"),  # Used for i1 == 0
    buffer_b: T.Buffer((28, 63), "int8"),  # Fills i1 from 1 to 63
    buffer_c: T.Buffer((28, 1), "int8"),  # Used for i1 == 64
    buffer_d: T.Buffer((28, 63), "int8"),  # Fills i1 from 65 to 128
    buffer_e: T.Buffer((28, 1), "int8"),  # Used for i1 == 129
    T_concat: T.Buffer((28, 129), "int8"),
) -> None:
    for i0 in range(28):
        for i1 in range(130, annotations={"pragma_loop_partition_hint": 1}):
            if i1 == 0:
                T_concat[i0, i1] = buffer_a[i0, 0]
            elif i1 == 64:
                T_concat[i0, i1] = buffer_c[i0, 0]
            elif i1 == 129:
                T_concat[i0, i1] = buffer_e[i0, 0]
            elif i1 < 64:
                T_concat[i0, i1] = buffer_b[i0, i1 - 1]
            else:  # i1 > 64 and i1 < 128
                T_concat[i0, i1] = buffer_d[i0, i1 - 65]


@T.prim_func
def concat_five_buffers_with_equalities_expected(
    buffer_a: T.Buffer((28, 1), "int8"),  # Used for i1 == 0
    buffer_b: T.Buffer((28, 63), "int8"),  # Fills i1 from 1 to 63
    buffer_c: T.Buffer((28, 1), "int8"),  # Used for i1 == 64
    buffer_d: T.Buffer((28, 63), "int8"),  # Fills i1 from 65 to 128
    buffer_e: T.Buffer((28, 1), "int8"),  # Used for i1 == 129
    T_concat: T.Buffer((28, 129), "int8"),
):
    for i0 in range(28):
        T_concat_1 = T.Buffer((3612,), "int8", data=T_concat.data)
        buffer_a_1 = T.Buffer((28,), "int8", data=buffer_a.data)
        T_concat_1[i0 * 129] = buffer_a_1[i0]
        for i1 in range(63):
            buffer_b_1 = T.Buffer((1764,), "int8", data=buffer_b.data)
            T_concat_1[i0 * 129 + i1 + 1] = buffer_b_1[i0 * 63 + i1]
        buffer_c_1 = T.Buffer((28,), "int8", data=buffer_c.data)
        T_concat_1[i0 * 129 + 64] = buffer_c_1[i0]
        for i1 in range(64):
            buffer_d_1 = T.Buffer((1764,), "int8", data=buffer_d.data)
            T_concat_1[i0 * 129 + i1 + 65] = buffer_d_1[i0 * 63 + i1]
        buffer_e_1 = T.Buffer((28,), "int8", data=buffer_e.data)
        T_concat_1[i0 * 129 + 129] = buffer_e_1[i0]


@T.prim_func
def nested_partition_with_single_points(A: T.Buffer[(25,), "int32"]):
    for i in T.serial(5, annotations={"pragma_loop_partition_hint": 1}):
        if i == 1:
            for j in T.serial(5, annotations={"pragma_loop_partition_hint": 1}):
                if j > 2:
                    A[i * 5 + j] = i * 5 + j
        else:
            for j in T.serial(5, annotations={"pragma_loop_partition_hint": 1}):
                if j > 2:
                    A[i * 5 + j] = i * 15 + j


@T.prim_func
def nested_partition_with_single_points_expected(A: T.Buffer[(25,), "int32"]):
    for j in range(2):
        A[j + 3] = j + 3
    for j in range(2):
        A[j + 8] = j + 8
    for i, j in T.grid(3, 2):
        A[i * 5 + j + 13] = i * 15 + j + 33


@pytest.mark.parametrize(
    "origin,expected",
    [
        (concat_func_single_point, expected_partitioned_concat_single_point),
        (concat_func_start_point_equality, concat_func_start_point_equality_expected),
        (concat_func_end_point_equality, concat_func_end_point_equality_expected),
        (concat_func_edge_equalities, concat_func_edge_equalities_expected),
        (concat_five_buffers_with_equalities, concat_five_buffers_with_equalities_expected),
        (nested_partition_with_single_points, nested_partition_with_single_points_expected),
    ],
)
def test_single_point_partition(origin, expected):
    origin = origin.with_attr({"global_symbol": "main"})
    expected = expected.with_attr({"global_symbol": "main"})
    mod = partition_from_scheduled_tir(
        origin,
        {
            "tir.LoopPartition": {
                "partition_const_loop": True,
                "unroll_loop_with_partition_hint_no_interval": True,
            }
        },
    )
    tvm.ir.assert_structural_equal(mod["main"], expected)


def test_equation_on_floordiv():
    @T.prim_func
    def before(A: T.Buffer[(2, 2, 20), "int32"]):
        for i in T.serial(5, annotations={"pragma_loop_partition_hint": 1}):
            if i == 1:
                for vv in T.vectorized(640, annotations={"pragma_loop_partition_hint": 1}):
                    if i * 2 + vv // 320 == 3:
                        A[i - 1, i * 2 + vv // 320 - 3, vv % 320 // 16] = 1

    @T.prim_func
    def expected(A: T.Buffer[(2, 2, 20), "int32"]):
        for vv in T.vectorized(320):
            A[0, 0, vv // 16] = 1

    expected = expected.with_attr({"global_symbol": "main"})
    after = partition_from_scheduled_tir(
        before.with_attr("global_symbol", "main"), {}, do_flatten=False
    )
    tvm.ir.assert_structural_equal(after["main"], expected)


def test_ignore_loop_partition_hint():
    """Skip unroll body and prologue for pipeline case"""

    @T.prim_func
    def before(A: T.Buffer[(10), "float32"], D: T.Buffer[(10), "float32"]):
        B = T.decl_buffer([2], "float32")
        C = T.decl_buffer([2], "float32")
        for i in T.serial(12, annotations={"pragma_loop_partition_hint": 1}):
            if T.ignore_loop_partition(i < 10):
                B[i % 2] = A[i] + 1.0
            if T.ignore_loop_partition(1 <= i and i < 11):
                C[(i - 1) % 2] = B[(i - 1) % 2] + 2.0
            if 2 <= i:
                D[i - 2] = C[i % 2] + 3.0

    @T.prim_func
    def expected(A: T.Buffer[(10), "float32"], D: T.Buffer[(10), "float32"]):
        B = T.decl_buffer([2], "float32")
        C = T.decl_buffer([2], "float32")
        for i in range(2):
            B[i] = A[i] + 1.0
            if i == 1:
                C[i - 1] = B[i - 1] + 2.0
        for i in T.serial(10):
            if i < 8:
                B[i % 2] = A[i + 2] + 1.0
            if i < 9:
                C[(i + 1) % 2] = B[(i + 1) % 2] + 2.0
            D[i] = C[i % 2] + 3.0

    expected = expected.with_attr({"global_symbol": "main"})
    after = partition_from_scheduled_tir(
        before.with_attr({"global_symbol": "main"}), {}, do_flatten=False
    )
    tvm.ir.assert_structural_equal(after["main"], expected)


if __name__ == "__main__":
    tvm.testing.main()
