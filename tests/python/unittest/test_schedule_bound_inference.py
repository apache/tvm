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

def test_bound1():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule([A2.op])
    xo, xi = s[A2].split(s[A2].op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)

def test_bound2():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    s = tvm.create_schedule(A2.op)
    xo, yo, xi, yi = s[A2].tile(A2.op.axis[0], A2.op.axis[1], 8, 8)
    # test normalize not affecting schedule
    _ = s.normalize()
    s[A1].compute_at(s[A2], yo)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)
    assert(bounds[A1.op.axis[1]].extent.value == 8)

def test_bound3():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    s[A1].set_scope("shared")
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, nparts=16)
    s[A2].bind(xi0, tvm.thread_axis("threadIdx.x"))
    yo, yi = s[A2].split(A2.op.axis[1], 16)
    # test normalize not affecting schedule
    _ = s.normalize()
    s[A2].reorder(xo, xi0, yo, xi1, yi)
    s[A1].compute_at(s[A2], yo)

    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value==32)
    assert(bounds[A1.op.axis[1]].extent.value==16)


def test_bound_warp():
    m = tvm.var('m')
    l = tvm.var('l')
    A = tvm.placeholder((m, l), name='A')
    A1 = tvm.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = tvm.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = tvm.create_schedule(A2.op)
    s[A1].set_scope("warp")
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, factor=16)
    tx = tvm.thread_axis("threadIdx.x")
    s[A2].bind(xi1, tx)
    s[A2].bind(xi0, tvm.thread_axis("threadIdx.y"))
    y = s[A2].op.axis[1]
    s[A1].compute_at(s[A2], y)
    xo, xi = s[A1].split(s[A1].op.axis[0], factor=16)
    s[A1].bind(xi, tx)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value==16)

def test_bound_scan():
    m = tvm.var("m")
    n = tvm.var("n")
    X = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    s_scan = tvm.scan(s_init, s_update, s_state)

    assert tuple(s_scan.shape) == (m, n)
    s = tvm.create_schedule(s_scan.op)
    XX = s.cache_read(X, "local", s_update)
    xo, xi = s[s_update].split(s_update.op.axis[1], factor=4)
    s[XX].compute_at(s[s_update], xo)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    stmt = tvm.schedule.ScheduleOps(s, bounds)
    assert bounds[XX.op.axis[1]].extent.value == 4

def test_bound_conv1d():
    n = tvm.var('n')
    A = tvm.compute((n+2), lambda i: 1,  name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = tvm.compute(n, computeB, name='B')
    s = tvm.create_schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[0])
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)

def test_bound_blur():
    n = tvm.convert(12)
    A = tvm.compute((n, n), lambda i, j: 1, name='A')
    def computeB(ii, jj):
        # set the correct center
        i = ii + 1
        j = jj + 1
        return A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j+1] + A[i][j-1]
    B = tvm.compute((n-2, n-2), computeB, name='B')
    s = tvm.create_schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[1])
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)
    assert(bounds[A.op.axis[1]].extent.value == 3)

def test_bound_rfactor():
    n = tvm.var('n')
    A = tvm.placeholder((n,), name='A')
    k = tvm.reduce_axis((0, n))
    B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k, where=(i>1)), name='B')
    # schedule
    s = tvm.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)

    assert(bounds[BF.op.axis[0]].extent.value == 4)
    assert(bounds[BF.op.axis[1]].extent.value == 1)

def test_bound_group_schedule():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.create_schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    g.compute_at(s[x2], x2.op.axis[0])
    assert s[x1].group == g
    assert s[x].group == g
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent == n

def test_bound_nest_group():
    m = tvm.var("m")
    n = tvm.var("n")
    x = tvm.compute((m, n), lambda i, j: tvm.const(1, "float32"), name="x")
    x1 = tvm.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = tvm.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = tvm.create_schedule(x2.op)
    g1 = s.create_group(outputs=x, inputs=x, include_inputs=True)
    g2 = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x].group == g1
    assert s[x1].group == g2
    g2.compute_at(s[x2], x2.op.axis[0])
    g1.compute_at(s[x1], s[x1].op.axis[1])
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent.value == 1
    assert bounds[x1.op.axis[0]].extent.value == 1
    assert bounds[x1.op.axis[1]].extent == n


def test_bound_nest_thread():
    m = tvm.var('m')
    A = tvm.placeholder((m), name='A')
    A1 = tvm.compute((m,), lambda i: A[i], name='A1')
    A2 = tvm.compute((m,), lambda i: A1[i] + 2, name='A2')
    A3 = tvm.compute((m,), lambda i: A2[i] + 3, name='A3')

    s = tvm.create_schedule(A3.op)
    s[A2].set_scope("shared")
    s[A1].set_scope("local")

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    bx, tx = s[A3].split(A3.op.axis[0], factor=32)
    s[A3].bind(bx, block_x)
    s[A3].bind(tx, thread_x)
    s[A2].compute_at(s[A3], tx)
    _, xi = s[A2].split(A2.op.axis[0], nparts=1)
    s[A2].bind(xi, thread_x)
    s[A1].compute_at(s[A3], tx)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[A1.op.axis[0]].extent.value==1)
    assert(bounds[A2.op.axis[0]].extent.value==32)
    assert(bounds[A3.op.axis[0]].extent == m)

def test_gemm_bound():
    nn = 1024
    n = tvm.convert(nn)
    A = tvm.placeholder((n, n), name='A')
    B = tvm.placeholder((n, n), name='B')
    k = tvm.reduce_axis((0, n), name='k')
    C = tvm.compute(
        (n, n),
        lambda ii, jj: tvm.sum(A[ii, k] * B[jj, k], axis=k),
        name='CC')
    # schedule
    s = tvm.create_schedule(C.op)
    xtile, ytile = 32, 32
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_y = tvm.thread_axis("threadIdx.y")

    CC = s.cache_write(C, "local")
    AA = s.cache_read(A, "shared", [CC])
    BB = s.cache_read(B, "shared", [CC])
    by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    s[C].reorder(by, bx, yi, xi)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    ty, yi = s[C].split(yi, nparts=num_thread)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].reorder(ty, tx, yi, xi)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    yo, xo = CC.op.axis
    s[CC].reorder(k, yo, xo)

    s[CC].compute_at(s[C], tx)
    s[AA].compute_at(s[CC], k)
    s[BB].compute_at(s[CC], k)

    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
    tx, xi = s[AA].split(xi, nparts=num_thread)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
    tx, xi = s[BB].split(xi, nparts=num_thread)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s = s.normalize()
    bounds = tvm.schedule.InferBound(s)
    assert(bounds[BB.op.axis[0]].extent.value==64)
    assert(bounds[AA.op.axis[0]].extent.value==64)
    assert(bounds[CC.op.axis[0]].extent.value == 8)
    assert(bounds[CC.op.axis[1]].extent.value == 8)


def test_bound_tensor_compute_op():
    def intrin_test():
      m1 = tvm.var("m1")
      n1 = tvm.var("n1")
      a = tvm.placeholder((m1, n1), name='a')
      c = tvm.compute((1, n1), lambda i, j : a[0, j] + a[1, j] + a[2, j], name='c')

      Ab = tvm.decl_buffer(a.shape, name="Abuf", offset_factor=1)
      Cb = tvm.decl_buffer(c.shape, name="Cbuf", offset_factor=1)

      def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]
        def _body():
          ib = tvm.ir_builder.create()
          ib.emit(tvm.call_extern("int32", "test", cc.access_ptr("w"), aa.access_ptr("r")))
          return ib.get()
        return _body()
      with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a : Ab, c : Cb})

    test_func = intrin_test()
    A = tvm.placeholder((20,20), name='A')
    B = tvm.compute(A.shape, lambda i,j : A[i,j], name='B')
    C = tvm.compute((10, 20), lambda i : test_func(B[i:10, 0:20]), name='C')
    s = tvm.create_schedule(C.op)
    bounds = tvm.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[B.op.axis[0]].extent.value == 10)

def test_bound_simplification_failure():
    # Check that the bounds are not expanded
    A = tvm.compute((2,), lambda j: j, "A")

    def _check(B, A=A):
        s = tvm.create_schedule(B.op)
        s = s.normalize()
        bounds = tvm.schedule.InferBound(s)
        stmt = tvm.lower(s, [B, A], simple_mode=True)
        if not bounds[A.op.axis[0]].extent.value <= 2:
            print(stmt)
            assert bounds[A.op.axis[0]].extent.value <= 2

    # These are hard to simplify, moreover we don't simplify them
    _check(tvm.compute((10,), lambda i: A[tvm.min(3*i, 4*i) + tvm.min(-3*i, -2*i)]))
    _check(tvm.compute((10,), lambda i: A[tvm.min(3*i, 4*i) + tvm.max(-3*i, -4*i)]))
    _check(tvm.compute((10,), lambda i: A[-2*(i/2) - tvm.min(i, 0-i)]))
    _check(tvm.compute((10,), lambda i: A[i + (0 - i)]))
    # This would cause out of bounds, but we nevertheless include it
    _check(tvm.compute((10,), lambda i: A[i]))

if __name__ == "__main__":
    test_bound_nest_thread()
    test_bound1()
    test_bound_nest_group()
    test_bound_group_schedule()
    test_bound_scan()
    test_bound3()
    test_bound_rfactor()
    test_bound_blur()
    test_bound_conv1d()
    test_bound2()
    test_gemm_bound()
    test_bound_warp()
    test_bound_tensor_compute_op()
    test_bound_simplification_failure()
