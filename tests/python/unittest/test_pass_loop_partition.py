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
import numpy

def collect_visit(stmt, f):
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, lambda x : ret.append(f(x)))
    return ret

def find_top_produce(stmt):
    def f(x, ret):
        if isinstance(x, tvm.stmt.ProducerConsumer):
            ret.append(x)
    ret = []
    tvm.ir_pass.PostOrderVisit(stmt, lambda x : f(x, ret))
    return ret[-1]

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

def test_condition():
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, tvm.truncdiv(n+3,4), 'i') as i:
      with ib.for_range(0, 4, 'j') as j:
        ib.emit(tvm.make.Evaluate(
          tvm.make.Select(ib.likely(i*4+j<n), m, n)))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, False)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt.first, lambda x: isinstance(x, tvm.expr.Select))))

def test_condition_EQ():
    ib = tvm.ir_builder.create()
    m = tvm.var('m')
    n = tvm.var('n')
    with ib.for_range(0, 10, 'i') as i:
            ib.emit(tvm.make.Evaluate(
                tvm.make.Select(ib.likely(tvm.expr.EQ(i, 5)), m, n)))
    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
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
            with ib.if_scope(ib.likely(tvm.truncdiv(i,j) < m)):
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

def test_cce_loop_1():
  ib = tvm.ir_builder.create()
  dtype = 'float16'
  n = 514
  m = 514
  _A = tvm.placeholder((n*m,), name = 'A')
  Ab = tvm.decl_buffer((n*m,), dtype, name="A")
  A = ib.buffer_ptr(Ab)
  _B = tvm.placeholder((n*m,), name = 'B')
  Bb = tvm.decl_buffer((n*m,), dtype, name="B")
  B = ib.buffer_ptr(Bb)
  #for i in 0 to n-1:
  with ib.for_range(0, 11, name="i") as i:
      with ib.for_range(0, 160, name="j") as j:
          with ib.if_scope(ib.likely(((i*160) + j) < 1600)):
               A[(i+1)*m+j+1] = B[(i)*m+j+1] + B[(i+1)*m+j+1] + B[(i+2)*m+j+1]
  stmt = ib.get()
  stmt = tvm.ir_pass.LoopPartition(stmt, True)
  stmt = tvm.ir_pass.Simplify(stmt)
  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_cce_loop_2():
  ib = tvm.ir_builder.create()
  len = 112
  tile = 32
  loop = (len + tile - 1) // tile
  with ib.for_range(0, loop, 'i') as i:
    head = i * tile
    with ib.if_scope(ib.likely(head + tile > len)):
      tail = len
      ib.emit(tvm.call_extern('float32', "cce_intrisic", head, tail))
    with ib.else_scope():
      tail = head + tile
      ib.emit(tvm.call_extern('float32', "cce_intrisic", head, tail))

  stmt = ib.get()
  stmt = tvm.ir_pass.LoopPartition(stmt, True)
  stmt = tvm.ir_pass.Simplify(stmt)
  assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))


def test_cce_loop_3():
    ib = tvm.ir_builder.create()
    loop1 = 4
    loop2 = 9998
    tile = 39991
    with ib.for_range(0,loop2,'i') as i:
        with ib.for_range(0,loop1,'j') as j:
            head1 = i
            head2 = j
            with ib.if_scope(ib.likely(head1*loop1 + head2 < tile)):
                ib.emit(tvm.call_extern('float16',"cce_intrisic",head1))

    stmt = ib.get()
    stmt = tvm.ir_pass.LoopPartition(stmt,True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_conv_tiling():
    HSTR = WSTR = 1
    in_channel = 128
    kernel_height = kernel_width = 3
    out_channel = 64
    batch_size = 1
    in_height = in_width = 64
    out_height = out_width = in_height - kernel_height + 1
    data = tvm.placeholder((batch_size, in_channel, in_height, in_width), name='data')
    kernel = tvm.placeholder((kernel_height, kernel_width, in_channel,
        out_channel), name='kernel')
    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')
    conv = tvm.compute((batch_size, out_channel, out_height, out_width),
                       lambda n, oc, oh, ow: tvm.sum(data[n, ic, oh*HSTR + kh, ow*WSTR + kw] *
                                                     kernel[kh, kw, ic, oc],
                                                     axis=[ic, kh, kw]),
                       name="conv2d")
    s = tvm.create_schedule(conv.op)

    n, oc, oh, ow = conv.op.axis
    oho, owo, ohi, owi = s[conv].tile(oh, ow, 16, 16)
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    stmt = tvm.ir_pass.LoopPartition(stmt, True)
    stmt = tvm.ir_pass.Simplify(stmt)
    assert(not any(collect_visit(stmt, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

def test_double_splitting_with_indivisible_factors():
    m = 48
    dtype="float32"
    A = tvm.placeholder((m,), name='A', dtype=dtype)
    C = tvm.compute((m,), lambda i: A[i], name='C')
    D = tvm.compute((m,), lambda i: C[i], name='D')

    s = tvm.create_schedule(D.op)
    co, ci = s[C].split(C.op.axis[0], factor=10)
    do, di = s[D].split(D.op.axis[0], 32)
    s[C].compute_at(s[D], do)

    target = 'llvm'
    with tvm.build_config(partition_const_loop=True):
        f = tvm.lower(s, [A, C, D], name="fadd1", simple_mode=False)
        func = tvm.build(f, target=target)

    # Find the beginning of the Halide IR corresponding to kernel code
    # and make sure it doesn't have an if statements left
    top_produce = find_top_produce(f.body)
    assert(not any(collect_visit(top_produce, lambda x: isinstance(x, tvm.stmt.IfThenElse))))

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
    k = tvm.reduce_axis((0, K), 'k')

    A = tvm.placeholder((1, K), name='A')

    B = tvm.compute( (1,), lambda b:
            tvm.sum(A[b, k], axis=k),
            name='B'
    )

    s = tvm.create_schedule(B.op)
    ko, _ = s[B].split(s[B].op.reduce_axis[0], 16)
    BF = s.rfactor(B, ko, 0)

    s.normalize()
    bounds = tvm.schedule.InferBound(s)

    stmt1 = tvm.schedule.ScheduleOps(s, bounds)
    stmt1 = tvm.ir_pass.Simplify(stmt1)

    stmt2 = tvm.ir_pass.LoopPartition(stmt1, True)
    stmt2 = tvm.ir_pass.Simplify(stmt2)

    #make sure loop partition actually did something
    assert not tvm.ir_pass.Equal(stmt1.body, stmt2.body)


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
    test_simple_rfactor()
