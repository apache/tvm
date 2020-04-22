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
from topi.nn.pooling import pool

def test_tensor():
    m = te.size_var('m')
    n = te.size_var('n')
    l = te.size_var('l')
    A = te.placeholder((m, l), name='A')
    B = te.placeholder((n, l), name='B')
    T = te.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == (m, n, l))
    assert(isinstance(A.op, tvm.te.PlaceholderOp))
    assert(A == A)
    assert(T.op.output(0) == T)
    assert(T.op.output(0).__hash__() == T.__hash__())
    d = {T.op.output(0) : 1}
    assert(d[T] == 1)
    assert(T[0][0][0].astype('float16').dtype == 'float16')


def test_rank_zero():
    m = te.size_var('m')
    A = te.placeholder((m,), name='A')
    scale = te.placeholder((), name='s')
    k = te.reduce_axis((0, m), name="k")
    T = te.compute((), lambda : te.sum(A[k] * scale(), axis=k))
    print(T)
    print(T.op.body)
    assert(tuple(T.shape) == ())


def test_conv1d():
    n = te.size_var('n')
    A = te.placeholder((n+2), name='A')
    def computeB(ii):
        i = ii + 1
        return A[i-1] + A[i] + A[i+1]
    B = te.compute(n, computeB)


def test_tensor_slice():
    n = te.size_var('n')
    A = te.compute((n, n), lambda i, j: 1)
    B = te.compute((n,), lambda i: A[0][i] + A[0][i])


def test_tensor_reduce_multi_axis():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    k1 = te.reduce_axis((0, n), "k")
    k2 = te.reduce_axis((0, m), "k")
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=(k1, k2)))
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=[k1, k2]))


def test_tensor_comm_reducer():
    m = te.size_var('m')
    n = te.size_var('n')
    A = te.placeholder((m, n), name='A')
    k = te.reduce_axis((0, n), "k")
    mysum = te.comm_reducer(lambda x, y: x+y, lambda t: tvm.tir.const(0, dtype=t))
    C = te.compute((m,), lambda i: mysum(A[i, k], axis=k))

def test_tensor_comm_reducer_overload():
    m = te.size_var('m')
    n = te.size_var('n')
    mysum = te.comm_reducer(lambda x, y: x+y, lambda t: tvm.tir.const(0, dtype=t))
    sum_res = mysum(m, n)

def test_tensor_reduce():
    m = te.size_var('m')
    n = te.size_var('n')
    l = te.size_var('l')
    A = te.placeholder((m, l), name='A')
    B = te.placeholder((n, l), name='B')
    T = te.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = te.reduce_axis((0, A.shape[1]), "k")
    C = te.compute((m, n), lambda i, j: te.sum(T(i, j, rv+1), axis=rv))
    # json load save
    C_json = tvm.ir.save_json(C)
    C_loaded = tvm.ir.load_json(C_json)
    assert(isinstance(C_loaded, te.tensor.Tensor))
    assert(str(C_loaded) == str(C))

def test_tensor_compute1():
    m = 1024
    factor = 16
    dtype = 'float32'

    def intrin_vadd(n):
        x = te.placeholder((n,))
        y = te.placeholder((n,))
        z = te.compute(x.shape, lambda i: x[i] + y[i])

        def intrin_func(ins, outs):
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern(outs[0].dtype, 'vadd', ins[0].access_ptr("r"), ins[1].access_ptr('r'), outs[0].access_ptr('wr')))
            return ib.get()

        with tvm.target.build_config(offset_factor=n):
            return te.decl_tensor_intrin(z.op, intrin_func)

    vadd = intrin_vadd(factor)

    A = te.placeholder((m//factor, factor), name="A", dtype=dtype)
    B = te.placeholder((m//factor, factor), name="B", dtype=dtype)
    C = te.compute((m//factor, factor),
          lambda i: vadd(A[i, 0:factor], B[i, 0:factor]))

    s = te.create_schedule(C.op)
    stmt = tvm.lower(s, [A, B, C])["main"].body
    assert isinstance(stmt.body, tvm.tir.Evaluate)

def test_tensor_compute2():
    M = 2048
    N = 1024
    L = 1024
    factor = 16
    factor1 = 32
    factor2 = 32
    dtype = 'float32'

    def intrin_gemm(m, n, l):
        k = te.reduce_axis((0, l))
        x = te.placeholder((m, l))
        y = te.placeholder((n, l))
        # in theory, no relation
        z = te.compute((m, n), lambda i, j: te.sum(x[i][k] * y[j][k], axis=k))

        def intrin_func(ins, outs):
            x_ptr = ins[0].access_ptr("r")
            y_ptr = ins[1].access_ptr("r")
            z_ptr = outs[0].access_ptr("w")
            body = tvm.tir.call_packed(
                "gemv", x_ptr, y_ptr, z_ptr, m, n, l)
            reset = tvm.tir.call_packed(
                "fill_zero", z_ptr, m, n)
            update = tvm.tir.call_packed(
                "gemv_add", x_ptr, y_ptr, z_ptr, m, n, l)
            return body, reset, update

        with tvm.target.build_config(offset_factor=n):
            return te.decl_tensor_intrin(z.op, intrin_func)

    vgemm = intrin_gemm(factor1, factor2, factor)

    A = te.placeholder((M//factor1, L//factor, factor1, factor), name="A", dtype=dtype)
    B = te.placeholder((N//factor2, L//factor, factor2, factor), name="B", dtype=dtype)
    k = te.reduce_axis((0, L//factor), name='k')
    C = te.compute((M//factor1, N//factor2, factor1, factor2),
          lambda i, j: vgemm(A[i, k, 0:factor1, 0:factor], B[j, k, 0:factor2, 0:factor], reduce_axis=k))

    s = te.create_schedule(C.op)
    stmt = tvm.lower(s, [A, B, C])["main"].body
    assert isinstance(stmt.body.body[0], tvm.tir.Evaluate)
    assert isinstance(stmt.body.body[1].body, tvm.tir.Evaluate)

def test_tensor_scan():
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.placeholder((m, n))
    s = te.placeholder((m, n))
    res = tvm.te.scan(te.compute((1, n), lambda _, i: x[0, i]),
                   te.compute((m, n), lambda t, i: s[t-1, i] + x[t, i]),
                   s)
    assert tuple(res.shape) == (m, n)

def test_scan_multi_out():
    m = te.size_var("m")
    n = te.size_var("n")
    x1 = te.placeholder((m, n))
    s1 = te.placeholder((m, n))
    x2 = te.placeholder((m, n))
    s2 = te.placeholder((m, n))
    s1_init = te.compute((1, n), lambda _, i: x1[0, i])
    s2_init = te.compute((1, n), lambda _, i: x2[0, i])
    s1_update = te.compute((m, n), lambda t, i: s1[t-1, i] + s2[t-1, i] + x1[t, i])
    s2_update = te.compute((m, n), lambda t, i: x2[t, i] + s2[t-1,i])

    r0, r1 = tvm.te.scan([s1_init, s2_init],
                      [s1_update, s2_update],
                      [s1, s2])
    assert(r0.value_index == 0)
    assert(r1.value_index == 1)
    json_str = tvm.ir.save_json(r0.op)
    zz = tvm.ir.load_json(json_str)
    assert isinstance(zz, tvm.te.ScanOp)

def test_extern():
    m = te.size_var('m')
    A = te.placeholder((m,), name='A')

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.te.schedule.Buffer))
        return tvm.tir.call_packed("myadd", ins[0].data, outs[0].data, m)
    B = te.extern((m,), [A], extern_func)
    assert(tuple(B.shape) == (m,))


def test_extern_multi_out():
    m = te.size_var('m')
    A = te.placeholder((m,), name='A')
    B = te.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert(isinstance(ins[0], tvm.te.schedule.Buffer))
        return tvm.tir.call_packed(
            "myadd", ins[0].data, outs[0].data, outs[1].data, m)
    res = te.extern([A.shape, A.shape], [A, B], extern_func)
    assert(len(res) == 2)
    assert(res[1].value_index == 1)

def test_tuple_inputs():
    m = te.size_var('m')
    n = te.size_var('n')
    A0 = te.placeholder((m, n), name='A0')
    A1 = te.placeholder((m, n), name='A1')
    T0, T1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='T')
    s = te.create_schedule(T0.op)

    for i in range(len(T0.shape)):
      assert(T0.shape[i] == T1.shape[i])
    assert(T0.op == T1.op)
    assert(T0.value_index == 0)
    assert(T1.value_index == 1)

def test_tuple_with_different_deps():
    m = te.size_var('m')
    n = te.size_var('n')
    A0 = te.placeholder((m, n), name='A1')
    A1 = te.placeholder((m, n), name='A2')
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name='B')
    C = te.compute((m, n), lambda i, j: B0[i, j] + 4, name='C')

    s = te.create_schedule(C.op)
    xo, xi = s[C].split(C.op.axis[0], factor=10)
    s[B0.op].compute_at(s[C], xo)
    sch = s.normalize()
    bounds = tvm.te.schedule.InferBound(sch)
    stmt = tvm.te.schedule.ScheduleOps(sch, bounds)

    def get_B1_realize(x):
        if isinstance(x, tvm.tir.Realize) and \
           x.func == B1.op and x.value_index == 1:
            ret.append(x)
    ret = []
    tvm.tir.stmt_functor.post_order_visit(stmt, get_B1_realize)

    assert stmt.node == C.op and len(ret) == 1


def test_tensor_inputs():
    x = te.placeholder((1,), name='x')
    y = te.compute(x.shape, lambda i: x[i] + x[i])
    assert tuple(y.op.input_tensors) == (x,)


def test_tensor_pool():
    def intrin_pool():
        A = te.placeholder((64, 16, 16), name='A')
        kh = te.reduce_axis((0, 3), name='kh')
        kw = te.reduce_axis((0, 3), name='kw')
        P = te.compute((64, 14, 14),
                        lambda c, oh, ow: tvm.te.max(A[c, oh + kh, ow + kw],
                                                  axis=[kh, kw]),
                        name='p')

        def intrin_func(ins, outs):
            dinp = ins[0]
            dout = outs[0]
            return tvm.tir.call_packed("op", dinp, dout)

        with tvm.target.build_config(offset_factor=1):
            return te.decl_tensor_intrin(P.op, intrin_func)

    A = te.placeholder((1, 64, 16, 16), name='A')
    P = pool(data=A, kernel=(3, 3), stride=(1, 1), padding=(0, 0, 0, 0),
             pool_type='max')
    s = te.create_schedule(P.op)
    _, oh, _, _ = P.op.axis
    intrin = intrin_pool()
    s[P].tensorize(oh, intrin)
    tvm.lower(s, [A, P])


if __name__ == "__main__":
    test_rank_zero()
    test_tensor_inputs()
    test_tensor_reduce_multi_axis()
    test_conv1d()
    test_tensor_slice()
    test_tensor()
    test_tensor_compute1()
    test_tensor_compute2()
    test_tensor_reduce()
    test_tensor_scan()
    test_scan_multi_out()
    test_extern()
    test_extern_multi_out()
    test_tuple_inputs()
    test_tuple_with_different_deps()
    test_tensor_pool()
