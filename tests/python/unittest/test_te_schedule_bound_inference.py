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

def test_bound1():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule([A2.op])
    xo, xi = s[A2].split(s[A2].op.axis[0], 8)
    s[A1].compute_at(s[A2], xo)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)

def test_bound2():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')
    s = te.create_schedule(A2.op)
    xo, yo, xi, yi = s[A2].tile(A2.op.axis[0], A2.op.axis[1], 8, 8)
    # test normalize not affecting schedule
    _ = s.normalize()
    s[A1].compute_at(s[A2], yo)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value == 8)
    assert(bounds[A1.op.axis[1]].extent.value == 8)

def test_bound3():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    s[A1].set_scope("shared")
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, nparts=16)
    s[A2].bind(xi0, te.thread_axis("threadIdx.x"))
    yo, yi = s[A2].split(A2.op.axis[1], 16)
    # test normalize not affecting schedule
    _ = s.normalize()
    s[A2].reorder(xo, xi0, yo, xi1, yi)
    s[A1].compute_at(s[A2], yo)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value==32)
    assert(bounds[A1.op.axis[1]].extent.value==16)

def test_bound_split_ext_less_than_factor():
    m = 8
    I = te.placeholder((m,), name='I')
    EF = te.compute((m,), lambda i: I[i] * 2, name = "EF")
    E = te.compute((m,), lambda i: EF[i] * 2, name = "E")
    s = te.create_schedule([E.op])
    xo, xi = s[E].split(s[E].op.axis[0], factor = 32)
    s[EF].compute_at(s[E], xo)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert bounds[xi].extent.value == m

def test_bound_split_ext_less_than_naprts():
    m = 8
    I = te.placeholder((m,), name='I')
    EF = te.compute((m,), lambda i: I[i] * 2, name = "EF")
    E = te.compute((m,), lambda i: EF[i] * 2, name = "E")
    s = te.create_schedule([E.op])
    xo, xi = s[E].split(s[E].op.axis[0], nparts = 32)
    s[EF].compute_at(s[E], xo)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert bounds[xo].extent.value == m

def test_bound_split_divisible():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((8 * m, l), name='A')
    B = te.compute((8 * m, l), lambda i, j: A[i, j], name='B')
    s = te.create_schedule(B.op)
    xo, xi = s[B].split(B.op.axis[0], 8)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert bounds[xo].extent == m
    assert bounds[xi].extent.value == 8

def test_bound_tile_divisible():
    m = te.var('m')
    l = te.var('l')
    shape = (8 * m, 32 * l)
    A = te.placeholder(shape, name='A')
    B = te.compute(shape, lambda i, j: A[i, j], name='B')
    s = te.create_schedule(B.op)
    xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], 8, 32)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert bounds[xo].extent == m
    assert bounds[xi].extent.value == 8
    assert bounds[yo].extent == l
    assert bounds[yi].extent.value == 32

def test_bound_fusesplit1():
    m = te.var('m')
    l = te.var('l')
    split1 = te.var('s')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    fused_axes = s[A2].fuse(A2.op.axis[0], A2.op.axis[1])
    xo, xi = s[A2].split(fused_axes, split1)
    s[A1].compute_at(s[A2], xo)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    idxdiv = tvm.tir.indexdiv
    tvm.testing.assert_prim_expr_equal(
        bounds[A1.op.axis[0]].min, idxdiv(xo * split1, l))

    expected_extent = (idxdiv((xo + 1) * split1 - 1, l) - idxdiv(xo * split1, l) + 1)
    for i in range(1, 6):
        for j in range(1, 6):
            for k in range(1, 6):
                vars = tvm.runtime.convert({split1: tvm.tir.const(i, "int32"), l: tvm.tir.const(j, "int32"), xo.var: tvm.tir.const(k, "int32")})
                tvm.testing.assert_prim_expr_equal(
                    tvm.tir.stmt_functor.substitute(bounds[A1.op.axis[0]].extent, vars),
                    tvm.tir.stmt_functor.substitute(expected_extent, vars)
                )

    tvm.testing.assert_prim_expr_equal(bounds[A1.op.axis[1]].extent, l)

def test_bound_fusesplit2():
    m = te.var("m")
    l = tvm.runtime.convert(6)
    split = tvm.runtime.convert(3)
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    fused_axes = s[A2].fuse(A2.op.axis[0], A2.op.axis[1])
    xo, xi = s[A2].split(fused_axes, split)
    s[A1].compute_at(s[A2], xo)

    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    vars = tvm.runtime.convert({xo.var: tvm.tir.const(5, "int32")})
    tvm.testing.assert_prim_expr_equal(tvm.tir.stmt_functor.substitute(bounds[A1.op.axis[0]].min, vars), 2)
    tvm.testing.assert_prim_expr_equal(tvm.tir.stmt_functor.substitute(bounds[A1.op.axis[1]].min, vars), 3)
    tvm.testing.assert_prim_expr_equal(tvm.tir.stmt_functor.substitute(bounds[A1.op.axis[0]].extent, vars), 1)
    tvm.testing.assert_prim_expr_equal(tvm.tir.stmt_functor.substitute(bounds[A1.op.axis[1]].extent, vars), 3)


def test_bound_warp():
    m = te.var('m')
    l = te.var('l')
    A = te.placeholder((m, l), name='A')
    A1 = te.compute((m, l), lambda i, j: A[i, j], name='A1')
    A2 = te.compute((m, l), lambda i, j: A1[i, j] + 3, name='A2')

    s = te.create_schedule(A2.op)
    s[A1].set_scope("warp")
    xo, xi = s[A2].split(A2.op.axis[0], 32)
    xi0, xi1 = s[A2].split(xi, factor=16)
    tx = te.thread_axis("threadIdx.x")
    s[A2].bind(xi1, tx)
    s[A2].bind(xi0, te.thread_axis("threadIdx.y"))
    y = s[A2].op.axis[1]
    s[A1].compute_at(s[A2], y)
    xo, xi = s[A1].split(s[A1].op.axis[0], factor=16)
    s[A1].bind(xi, tx)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[A1.op.axis[0]].extent.value==16)

def test_bound_scan():
    m = te.var("m")
    n = te.var("n")
    X = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    s_state = te.placeholder((m, n))
    s_init = te.compute((1, n), lambda _, i: X[0, i])
    s_update = te.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    s_scan = tvm.te.scan(s_init, s_update, s_state)

    assert tuple(s_scan.shape) == (m, n)
    s = te.create_schedule(s_scan.op)
    XX = s.cache_read(X, "local", s_update)
    xo, xi = s[s_update].split(s_update.op.axis[1], factor=4)
    s[XX].compute_at(s[s_update], xo)
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    stmt = tvm.te.schedule.ScheduleOps(s, bounds)
    assert bounds[XX.op.axis[1]].extent.value == 4

def test_bound_conv1d():
    n = te.var('n')
    A = te.compute((n+2), lambda i: 1,  name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = te.compute(n, computeB, name='B')
    s = te.create_schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[0])
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)

def test_bound_blur():
    n = tvm.runtime.convert(12)
    A = te.compute((n, n), lambda i, j: 1, name='A')
    def computeB(ii, jj):
        # set the correct center
        i = ii + 1
        j = jj + 1
        return A[i][j] + A[i-1][j] + A[i+1][j] + A[i][j+1] + A[i][j-1]
    B = te.compute((n-2, n-2), computeB, name='B')
    s = te.create_schedule(B.op)
    s[A].compute_at(s[B], B.op.axis[1])
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    assert(bounds[A.op.axis[0]].extent.value == 3)
    assert(bounds[A.op.axis[1]].extent.value == 3)

def test_bound_rfactor():
    n = te.var('n')
    A = te.placeholder((n,), name='A')
    k = te.reduce_axis((0, n))
    B = te.compute((1,), lambda i: te.sum(A[k], axis=k, where=(i>1)), name='B')
    # schedule
    s = te.create_schedule(B.op)
    kf, ki = s[B].split(k, nparts=4)
    BF = s.rfactor(B, kf)
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)

    assert(bounds[BF.op.axis[0]].extent.value == 4)
    assert(bounds[BF.op.axis[1]].extent.value == 1)

def test_bound_group_schedule():
    m = te.var("m")
    n = te.var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    g.compute_at(s[x2], x2.op.axis[0])
    assert s[x1].group == g
    assert s[x].group == g
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent == n

def test_bound_nest_group():
    m = te.var("m")
    n = te.var("n")
    x = te.compute((m, n), lambda i, j: tvm.tir.const(1, "float32"), name="x")
    x1 = te.compute(x.shape, lambda *i: x(*i) + 1, name="x1")
    x2 = te.compute(x.shape, lambda *i: x1(*i) + 2, name="x2")
    s = te.create_schedule(x2.op)
    g1 = s.create_group(outputs=x, inputs=x, include_inputs=True)
    g2 = s.create_group(outputs=x1, inputs=x, include_inputs=True)
    assert s[x].group == g1
    assert s[x1].group == g2
    g2.compute_at(s[x2], x2.op.axis[0])
    g1.compute_at(s[x1], s[x1].op.axis[1])
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    assert bounds[x.op.axis[0]].extent.value == 1
    assert bounds[x.op.axis[1]].extent.value == 1
    assert bounds[x1.op.axis[0]].extent.value == 1
    assert bounds[x1.op.axis[1]].extent == n


def test_bound_nest_thread():
    m = te.var('m')
    A = te.placeholder((m), name='A')
    A1 = te.compute((m,), lambda i: A[i], name='A1')
    A2 = te.compute((m,), lambda i: A1[i] + 2, name='A2')
    A3 = te.compute((m,), lambda i: A2[i] + 3, name='A3')

    s = te.create_schedule(A3.op)
    s[A2].set_scope("shared")
    s[A1].set_scope("local")

    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    bx, tx = s[A3].split(A3.op.axis[0], factor=32)
    s[A3].bind(bx, block_x)
    s[A3].bind(tx, thread_x)
    s[A2].compute_at(s[A3], tx)
    _, xi = s[A2].split(A2.op.axis[0], nparts=1)
    s[A2].bind(xi, thread_x)
    s[A1].compute_at(s[A3], tx)
    s = s.normalize()
    bounds = tvm.te.schedule.InferBound(s)
    assert(bounds[A1.op.axis[0]].extent.value==1)
    assert(bounds[A2.op.axis[0]].extent.value==32)
    assert(bounds[A3.op.axis[0]].extent == m)

def test_gemm_bound():
    nn = 1024
    n = tvm.runtime.convert(nn)
    A = te.placeholder((n, n), name='A')
    B = te.placeholder((n, n), name='B')
    k = te.reduce_axis((0, n), name='k')
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum(A[ii, k] * B[jj, k], axis=k),
        name='CC')
    # schedule
    s = te.create_schedule(C.op)
    xtile, ytile = 32, 32
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis("threadIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_y = te.thread_axis("threadIdx.y")

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
    bounds = tvm.te.schedule.InferBound(s)
    assert(bounds[BB.op.axis[0]].extent.value==64)
    assert(bounds[AA.op.axis[0]].extent.value==64)
    assert(bounds[CC.op.axis[0]].extent.value == 8)
    assert(bounds[CC.op.axis[1]].extent.value == 8)


def test_bound_tensor_compute_op():
    def intrin_test():
      m1 = te.var("m1")
      n1 = te.var("n1")
      a = te.placeholder((m1, n1), name='a')
      c = te.compute((1, n1), lambda i, j : a[0, j] + a[1, j] + a[2, j], name='c')

      Ab = tvm.tir.decl_buffer(a.shape, name="Abuf", offset_factor=1)
      Cb = tvm.tir.decl_buffer(c.shape, name="Cbuf", offset_factor=1)

      def intrin_func(ins, outs):
        aa = ins[0]
        cc = outs[0]
        def _body():
          ib = tvm.tir.ir_builder.create()
          ib.emit(tvm.tir.call_extern("int32", "test", cc.access_ptr("w"), aa.access_ptr("r")))
          return ib.get()
        return _body()
      with tvm.target.build_config(offset_factor=1):
        return te.decl_tensor_intrin(c.op, intrin_func, binds={a : Ab, c : Cb})

    test_func = intrin_test()
    A = te.placeholder((20,20), name='A')
    B = te.compute(A.shape, lambda i,j : A[i,j], name='B')
    C = te.compute((10, 20), lambda i : test_func(B[i:10, 0:20]), name='C')
    s = te.create_schedule(C.op)
    bounds = tvm.te.schedule.InferBound(s)
    assert isinstance(bounds, tvm.container.Map)
    assert(bounds[B.op.axis[0]].extent.value == 10)

def test_bound_simplification_failure():
    # Check that the bounds are not expanded
    A = te.compute((2,), lambda j: j, "A")

    def _check(B, A=A):
        s = te.create_schedule(B.op)
        s = s.normalize()
        bounds = tvm.te.schedule.InferBound(s)
        stmt = tvm.lower(s, [B, A], simple_mode=True)
        if not bounds[A.op.axis[0]].extent.value <= 2:
            print(stmt)
            assert bounds[A.op.axis[0]].extent.value <= 2
    tdiv = tvm.tir.truncdiv
    # These are hard to simplify, moreover we don't simplify them
    _check(te.compute((10,), lambda i: A[tvm.te.min(3*i, 4*i) + tvm.te.min(-3*i, -2*i)]))
    _check(te.compute((10,), lambda i: A[tvm.te.min(3*i, 4*i) + tvm.te.max(-3*i, -4*i)]))
    _check(te.compute((10,), lambda i: A[-2*tdiv(i,2) - tvm.te.min(i, 0-i)]))
    _check(te.compute((10,), lambda i: A[i + (0 - i)]))
    # This would cause out of bounds, but we nevertheless include it
    _check(te.compute((10,), lambda i: A[i]))

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
    test_bound_fusesplit1()
    test_bound_fusesplit2()
    test_bound_split_divisible()
    test_bound_tile_divisible()
