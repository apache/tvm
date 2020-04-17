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
import numpy

def collect_visit(stmt, f):
    ret = []
    tvm.tir.ir_pass.PostOrderVisit(stmt, lambda x : ret.append(f(x)))
    return ret


def test_basic():
    n = te.size_var('n')
    A = te.placeholder((n, ), name='A')
    B = te.placeholder((n, ), name='B')

    T = te.compute((n, ), lambda i: A[i]+B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(
        collect_visit(stmt.body.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse))))
    assert(any(
        collect_visit(stmt.body.body[1], lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_const_loop():
    n = 21
    A = te.placeholder((n, ), name='A')
    B = te.placeholder((n, ), name='B')

    T = te.compute((n, ), lambda i: A[i]+B[i])
    s = te.create_schedule(T.op)
    xo, xi = s[T].split(T.op.axis[0], factor=4)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))

def test_multi_loop():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var('m')
    n = te.size_var('n')
    with ib.for_range(0, 4, "i") as i:
        with ib.for_range(0, n, "j") as j:
            with ib.for_range(0, m, "k") as k:
                with ib.if_scope(ib.likely(i*m+j+k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([n, m], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse))))

def test_multi_if():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var('m')
    n = te.size_var('n')
    with ib.for_range(0, 4, 'i') as i:
        with ib.for_range(0, n, 'j') as j:
            with ib.for_range(0, m, 'k') as k:
                with ib.if_scope(ib.likely(i*m+j+k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
                with ib.if_scope(ib.likely(i*m+j-k < n)):
                    ib.emit(tvm.tir.Evaluate(m))
                with ib.else_scope():
                    ib.emit(tvm.tir.Evaluate(n))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(
        collect_visit(stmt.body[0], lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_thread_axis():
    m = te.size_var('m')
    l = te.size_var('l')
    A = te.placeholder((m, l), name='A')
    B = te.compute((m, l), lambda i, j: A[i, j] + 3, name='B')
    s = te.create_schedule(B.op)

    s[B].set_scope("shared")
    num_thread = 16
    xo, xi = s[B].split(B.op.axis[0], 32)
    xi0, xi1 = s[B].split(xi, nparts=num_thread)
    s[B].bind(xi0, te.thread_axis("threadIdx.x"))

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(
        collect_visit(stmt.body. body[0], lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_vectorize():
    n = te.size_var('n')
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    bias = te.size_var("bias", dtype="float32")
    scale = te.size_var("scale", dtype="float32")
    C = te.compute(A.shape, lambda *i: A(*i) + B(*i) * scale + bias, name='C')
    # schedule
    s = te.create_schedule(C.op)
    # create iter var and assign them tags.
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=num_thread*4)
    tx, x = s[C].split(x, nparts=num_thread)
    _, x = s[C].split(x, factor=4)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].vectorize(x)
    stmt = tvm.lower(s, [A, B], name="main")["main"].body
    body = stmt.body.body.body.body
    assert(x.var.name not in str(body.condition))
    assert(any(collect_visit(body.then_case, lambda x: isinstance(x, tvm.tir.Ramp))))


def test_condition():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var('m')
    n = te.size_var('n')
    with ib.for_range(0, tvm.tir.truncdiv(n+3,4), 'i') as i:
      with ib.for_range(0, 4, 'j') as j:
        ib.emit(tvm.tir.Evaluate(
          tvm.tir.Select(ib.likely(i*4+j<n), m, n)))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm.tir.Select))))


def test_condition_EQ():
    ib = tvm.tir.ir_builder.create()
    m = te.size_var('m')
    n = te.size_var('n')
    with ib.for_range(0, 10, 'i') as i:
            ib.emit(tvm.tir.Evaluate(
                tvm.tir.Select(ib.likely(tvm.tir.EQ(i, 5)), m, n)))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt[0], lambda x: isinstance(x, tvm.tir.Select))))


def test_thread_axis2():
    n = tvm.runtime.convert(4096)
    m = te.size_var('m')
    A = te.placeholder((n,), name='A')
    B = te.placeholder((n,), name='B')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='C')
    s = te.create_schedule(C.op)
    num_thread = 32
    bx, x = s[C].split(C.op.axis[0], factor=32)
    tx, x = s[C].split(x, nparts=num_thread)
    _,  x = s[C].split(x, factor=m)
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    stmt = tvm.lower(s, [A, B], name="main")["main"].body
    for_body = stmt.body.body.body.body[0]
    assert('threadIdx' not in str(for_body.extent))

def test_everything_during_deduction():
    m = te.size_var('m')
    n = te.size_var('n')
    ib = tvm.tir.ir_builder.create()
    with ib.for_range(0, n, 'i') as i:
        with ib.for_range(0, 32, 'j') as j:
            with ib.if_scope(ib.likely(tvm.tir.truncdiv(i,j) < m)):
                # this guard will produce everything during deduction
                ib.emit(tvm.tir.Evaluate(m))
    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, n], stmt))
    mod = tvm.tir.transform.LoopPartition(False)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body


    assert(isinstance(stmt.body.body, tvm.tir.IfThenElse))

def test_single_likely():
    n = 60
    A = te.placeholder((n, ), name='A')
    B = te.placeholder((n, ), name='B')

    T = te.compute((n, ), lambda i: A[i]+B[i])
    s = te.create_schedule(T.op)
    x = T.op.axis[0]
    xo, xi = s[T].split(x, factor=16)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))

def test_multi_likely():
    n = 94
    m = 62
    A = te.placeholder((n, m), name='A')
    B = te.placeholder((n, m), name='B')

    T = te.compute((n, m), lambda i, j: A[i, j]+B[i, j])
    s = te.create_schedule(T.op)
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    x, y = T.op.axis
    xo, xi = s[T].split(x, factor=16)
    yo, yi = s[T].split(y, factor=16)
    s[T].reorder(xo, yo, xi, yi)

    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_oneD_pool():
    m = te.size_var('m')
    ib = tvm.tir.ir_builder.create()
    #data = te.placeholder((16,), name = 'data')
    data = ib.pointer("float32", name="A")
    out = ib.pointer("float32", name="A")
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow > 0)):
                with ib.if_scope(ib.likely(ow < 15)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow < 1)):
                with ib.if_scope(ib.likely(kw > 0)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])
    with ib.for_range(0, 16, 'ow') as ow:
        with ib.for_range(0, 3, 'kw') as kw:
            with ib.if_scope(ib.likely(ow > 14)):
                with ib.if_scope(ib.likely(kw < 2)):
                    out[ow] = tvm.te.max(out[ow], data[ow + kw - 1])

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([m, data, out], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_cce_loop_1():
  ib = tvm.tir.ir_builder.create()
  dtype = 'float16'
  n = 514
  m = 514
  _A = te.placeholder((n*m,), name = 'A')
  Ab = tvm.tir.decl_buffer((n*m,), dtype, name="A")
  A = ib.buffer_ptr(Ab)
  _B = te.placeholder((n*m,), name = 'B')
  Bb = tvm.tir.decl_buffer((n*m,), dtype, name="B")
  B = ib.buffer_ptr(Bb)
  #for i in 0 to n-1:
  with ib.for_range(0, 11, name="i") as i:
      with ib.for_range(0, 160, name="j") as j:
          with ib.if_scope(ib.likely(((i*160) + j) < 1600)):
               A[(i+1)*m+j+1] = B[(i)*m+j+1] + B[(i+1)*m+j+1] + B[(i+2)*m+j+1]
  stmt = ib.get()

  mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([Ab, Bb], stmt))
  mod = tvm.tir.transform.LoopPartition(True)(mod)
  stmt = tvm.tir.transform.Simplify()(mod)["main"].body

  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))

def test_cce_loop_2():
  ib = tvm.tir.ir_builder.create()
  len = 112
  tile = 32
  loop = (len + tile - 1) // tile
  with ib.for_range(0, loop, 'i') as i:
    head = i * tile
    with ib.if_scope(ib.likely(head + tile > len)):
      tail = len
      ib.emit(tvm.tir.call_extern('float32', "cce_intrisic", head, tail))
    with ib.else_scope():
      tail = head + tile
      ib.emit(tvm.tir.call_extern('float32', "cce_intrisic", head, tail))

  stmt = ib.get()


  mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
  mod = tvm.tir.transform.LoopPartition(True)(mod)
  stmt = tvm.tir.transform.Simplify()(mod)["main"].body

  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_cce_loop_3():
    ib = tvm.tir.ir_builder.create()
    loop1 = 4
    loop2 = 9998
    tile = 39991
    with ib.for_range(0,loop2,'i') as i:
        with ib.for_range(0,loop1,'j') as j:
            head1 = i
            head2 = j
            with ib.if_scope(ib.likely(head1*loop1 + head2 < tile)):
                ib.emit(tvm.tir.call_extern('float16',"cce_intrisic",head1))

    stmt = ib.get()

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_conv_tiling():
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = te.placeholder((batch_size, in_channel, in_height, in_width), name='data')
    kernel = te.placeholder((kernel_height, kernel_width, in_channel,
        out_channel), name='kernel')
    ic = te.reduce_axis((0, in_channel), name='ic')
    kh = te.reduce_axis((0, kernel_height), name='kh')
    kw = te.reduce_axis((0, kernel_width), name='kw')
    conv = te.compute((batch_size, out_channel, out_height, out_width),
                       lambda n, oc, oh, ow: te.sum(data[n, ic, oh*HSTR + kh, ow*WSTR + kw] *
                                                     kernel[kh, kw, ic, oc],
                                                     axis=[ic, kh, kw]),
                       name="conv2d")
    s = te.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)

    mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt))
    mod = tvm.tir.transform.LoopPartition(True)(mod)
    stmt = tvm.tir.transform.Simplify()(mod)["main"].body

    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.tir.IfThenElse))))


def test_multilevel_splitting_with_indivisble_factors():
    import topi
    A = te.placeholder((130,), dtype="float32")
    B = topi.nn.relu(A)
    s = te.create_schedule(B.op)
    (y,) = s[B].op.axis
    (yo, yi) = s[B].split(y, factor=8)
    (yoo, yoi) = s[B].split(yo, factor=16)
    s[B].reorder(yoo, yoi, yi)
    s[B].unroll(yi)

    ## But this does the right thing.
    with tvm.target.build_config(partition_const_loop=True):
        lowered_body = tvm.lower(s, [A, B], name="x")["x"].body
        def visit_stmt(op):
            return(isinstance(op, tvm.tir.Max))
        num_max = collect_visit(lowered_body, visit_stmt)
        assert num_max.count(True) == 10


def test_double_splitting_with_indivisible_factors():
    m = 48
    dtype="float32"
    A = te.placeholder((m,), name='A', dtype=dtype)
    C = te.compute((m,), lambda i: A[i], name='C')
    D = te.compute((m,), lambda i: C[i], name='D')

    s = te.create_schedule(D.op)
    co, ci = s[C].split(C.op.axis[0], factor=10)
    do, di = s[D].split(D.op.axis[0], 32)
    s[C].compute_at(s[D], do)

    target = 'llvm'
    with tvm.target.build_config(partition_const_loop=True):
        f = tvm.lower(s, [A, C, D], name="fadd1", simple_mode=False)
        func = tvm.build(f, target=target)

    top_produce = f["fadd1"].body
    assert(not any(collect_visit(top_produce, lambda x: isinstance(x, tvm.tir.IfThenElse))))

    # check functional correctness of generated code
    ctx = tvm.context(target, 0)
    a = tvm.nd.array(numpy.ones(m,).astype(dtype), ctx)
    c = tvm.nd.array(numpy.zeros(m,).astype(dtype), ctx)
    d = tvm.nd.array(numpy.zeros(m,).astype(dtype), ctx)
    func(a, c, d)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy(), rtol=1e-5)
    tvm.testing.assert_allclose(d.asnumpy(), a.asnumpy(), rtol=1e-5)

def test_simple_rfactor():
    K = 16*4+4
    k = te.reduce_axis((0, K), 'k')

    A = te.placeholder((1, K), name='A')

    B = te.compute( (1,), lambda b:
            te.sum(A[b, k], axis=k),
            name='B'
    )

    s = te.create_schedule(B.op)
    ko, _ = s[B].split(s[B].op.reduce_axis[0], 16)
    BF = s.rfactor(B, ko, 0)

    s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt1 = tvm.te.schedule.ScheduleOps(s, bounds)

    mod1 = tvm.IRModule.from_expr(tvm.tir.PrimFunc([], stmt1))
    stmt1 = tvm.tir.transform.Simplify()(mod1)["main"].body

    mod2 = tvm.tir.transform.LoopPartition(True)(mod1)
    stmt2 = tvm.tir.transform.Simplify()(mod2)["main"].body

    # make sure loop partition actually did something
    assert not tvm.ir.structural_equal(stmt1.body, stmt2.body)


if __name__ == "__main__":
    test_basic()
    test_const_loop()
    test_multi_loop()
    test_multi_if()
    test_thread_axis()
    test_vectorize()
    test_condition()
    test_condition_EQ()
    test_thread_axis2()
    test_everything_during_deduction()
    test_single_likely()
    test_multi_likely()
    test_oneD_pool()
    test_cce_loop_1()
    test_cce_loop_2()
    test_cce_loop_3()
    test_conv_tiling()
    test_double_splitting_with_indivisible_factors()
    test_multilevel_splitting_with_indivisble_factors()
    test_simple_rfactor()
