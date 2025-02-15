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
import numpy as np
import tvm
import tvm.testing
from tvm import te
from tvm.topi.nn.pooling import pool2d


def test_tensor():
    m = te.size_var("m")
    n = te.size_var("n")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")
    B = te.placeholder((n, l), name="B")
    T = te.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    print(T)
    print(T.op.body)
    assert tuple(T.shape) == (m, n, l)
    assert isinstance(A.op, tvm.te.PlaceholderOp)
    assert A == A
    assert T.op.output(0) == T
    assert T.op.output(0).__hash__() == T.__hash__()
    d = {T.op.output(0): 1}
    assert d[T] == 1
    assert T[0][0][0].astype("float16").dtype == "float16"


def test_rank_zero():
    m = te.size_var("m")
    A = te.placeholder((m,), name="A")
    scale = te.placeholder((), name="s")
    k = te.reduce_axis((0, m), name="k")
    T = te.compute((), lambda: te.sum(A[k] * scale(), axis=k))
    print(T)
    print(T.op.body)
    assert tuple(T.shape) == ()


def test_conv1d():
    n = te.size_var("n")
    A = te.placeholder((n + 2), name="A")

    def computeB(ii):
        i = ii + 1
        return A[i - 1] + A[i] + A[i + 1]

    B = te.compute(n, computeB)


def test_tensor_slice():
    n = te.size_var("n")
    A = te.compute((n, n), lambda i, j: 1)
    B = te.compute((n,), lambda i: A[0][i] + A[0][i])


def test_tensor_reduce_multi_axis():
    m = te.size_var("m")
    n = te.size_var("n")
    A = te.placeholder((m, n), name="A")
    k1 = te.reduce_axis((0, n), "k")
    k2 = te.reduce_axis((0, m), "k")
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=(k1, k2)))
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=[k1, k2]))


def test_tensor_comm_reducer():
    m = te.size_var("m")
    n = te.size_var("n")
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), "k")
    mysum = te.comm_reducer(lambda x, y: x + y, lambda t: tvm.tir.const(0, dtype=t))
    C = te.compute((m,), lambda i: mysum(A[i, k], axis=k))


def test_tensor_comm_reducer_overload():
    m = te.size_var("m")
    n = te.size_var("n")
    mysum = te.comm_reducer(lambda x, y: x + y, lambda t: tvm.tir.const(0, dtype=t))
    sum_res = mysum(m, n)


def test_tensor_reduce():
    m = te.size_var("m")
    n = te.size_var("n")
    l = te.size_var("l")
    A = te.placeholder((m, l), name="A")
    B = te.placeholder((n, l), name="B")
    T = te.compute((m, n, l), lambda i, j, k: A[i, k] * B[j, k])
    rv = te.reduce_axis((0, A.shape[1]), "k")
    C = te.compute((m, n), lambda i, j: te.sum(T(i, j, rv + 1), axis=rv))
    # json load save
    C_json = tvm.ir.save_json(C)
    C_loaded = tvm.ir.load_json(C_json)
    assert isinstance(C_loaded, te.tensor.Tensor)
    assert str(C_loaded) == str(C)


def test_tensor_reduce_multiout_with_cond():
    def fcombine(x, y):
        return x[0] + y[0], x[1] + y[1]

    def fidentity(t0, t1):
        return tvm.tir.const(0, t0), tvm.tir.const(1, t1)

    mysum = te.comm_reducer(fcombine, fidentity, name="mysum")

    m = te.var("m")
    n = te.var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="int32")
    k = te.reduce_axis((0, n), "k")
    cond = te.floormod(k, 2) == 0
    T0, T1 = te.compute((m,), lambda i: mysum((idx[i, k], val[i, k]), axis=k, where=cond), name="T")


def test_tensor_scan():
    m = te.size_var("m")
    n = te.size_var("n")
    x = te.placeholder((m, n))
    s = te.placeholder((m, n))
    res = tvm.te.scan(
        te.compute((1, n), lambda _, i: x[0, i]),
        te.compute((m, n), lambda t, i: s[t - 1, i] + x[t, i]),
        s,
    )
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
    s1_update = te.compute((m, n), lambda t, i: s1[t - 1, i] + s2[t - 1, i] + x1[t, i])
    s2_update = te.compute((m, n), lambda t, i: x2[t, i] + s2[t - 1, i])

    r0, r1 = tvm.te.scan([s1_init, s2_init], [s1_update, s2_update], [s1, s2])
    assert r0.value_index == 0
    assert r1.value_index == 1
    json_str = tvm.ir.save_json(r0.op)
    zz = tvm.ir.load_json(json_str)
    assert isinstance(zz, tvm.te.ScanOp)


def test_extern():
    m = te.size_var("m")
    A = te.placeholder((m,), name="A")

    def extern_func(ins, outs):
        assert isinstance(ins[0], tvm.tir.Buffer)
        return tvm.tir.call_packed("myadd", ins[0].data, outs[0].data, m)

    B = te.extern((m,), [A], extern_func)
    assert tuple(B.shape) == (m,)


def test_extern_multi_out():
    m = te.size_var("m")
    A = te.placeholder((m,), name="A")
    B = te.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert isinstance(ins[0], tvm.tir.Buffer)
        return tvm.tir.call_packed("myadd", ins[0].data, outs[0].data, outs[1].data, m)

    res = te.extern([A.shape, A.shape], [A, B], extern_func)
    assert len(res) == 2
    assert res[1].value_index == 1


def test_tuple_inputs():
    m = te.size_var("m")
    n = te.size_var("n")
    A0 = te.placeholder((m, n), name="A0")
    A1 = te.placeholder((m, n), name="A1")
    T0, T1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name="T")
    s = te.create_prim_func([A0, A1, T0])


def test_tuple_with_different_deps():
    m = te.size_var("m")
    n = te.size_var("n")
    A0 = te.placeholder((m, n), name="A1")
    A1 = te.placeholder((m, n), name="A2")
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name="B")
    C = te.compute((m, n), lambda i, j: B0[i, j] + 4, name="C")

    te.create_prim_func([A0, A1, C])


def test_tensor_inputs():
    x = te.placeholder((1,), name="x")
    y = te.compute(x.shape, lambda i: x[i] + x[i])
    assert tuple(y.op.input_tensors) == (x,)


if __name__ == "__main__":
    test_tensor()
    test_rank_zero()
    test_conv1d()
    test_tensor_slice()
    test_tensor_reduce_multi_axis()
    test_tensor_comm_reducer()
    test_tensor_comm_reducer_overload()
    test_tensor_reduce()
    test_tensor_reduce_multiout_with_cond()
    test_tensor_compute1()
    test_tensor_compute2()
    test_tensor_scan()
    test_scan_multi_out()
    test_extern()
    test_extern_multi_out()
    test_tuple_inputs()
    test_tuple_with_different_deps()
    test_tensor_inputs()
