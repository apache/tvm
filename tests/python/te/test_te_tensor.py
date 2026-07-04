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
# ruff: noqa: E741, F401, F821, F841
import importlib
import operator
import os
import subprocess
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import te, topi
from tvm.te import _te_tensor_overload
from tvm.topi.nn.pooling import pool2d


def test_tensor():
    m = te.var("m")
    n = te.var("n")
    l = te.var("l")
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
    assert T[0][0][0].astype("float16").ty == tvm.ir.PrimType("float16")


def test_rank_zero():
    m = te.var("m")
    A = te.placeholder((m,), name="A")
    scale = te.placeholder((), name="s")
    k = te.reduce_axis((0, m), name="k")
    T = te.compute((), lambda: te.sum(A[k] * scale(), axis=k))
    print(T)
    print(T.op.body)
    assert tuple(T.shape) == ()


def test_conv1d():
    n = te.var("n")
    A = te.placeholder((n + 2), name="A")

    def computeB(ii):
        i = ii + 1
        return A[i - 1] + A[i] + A[i + 1]

    B = te.compute(n, computeB)


def test_tensor_slice():
    n = te.var("n")
    A = te.compute((n, n), lambda i, j: 1)
    B = te.compute((n,), lambda i: A[0][i] + A[0][i])


def test_tensor_reduce_multi_axis():
    m = te.var("m")
    n = te.var("n")
    A = te.placeholder((m, n), name="A")
    k1 = te.reduce_axis((0, n), "k")
    k2 = te.reduce_axis((0, m), "k")
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=(k1, k2)))
    C = te.compute((1,), lambda _: te.sum(A[k1, k2], axis=[k1, k2]))


def test_tensor_comm_reducer():
    m = te.var("m")
    n = te.var("n")
    A = te.placeholder((m, n), name="A")
    k = te.reduce_axis((0, n), "k")
    mysum = te.comm_reducer(lambda x, y: x + y, lambda t: tvm.tirx.const(0, dtype=t))
    C = te.compute((m,), lambda i: mysum(A[i, k], axis=k))


def test_tensor_comm_reducer_overload():
    m = te.var("m")
    n = te.var("n")
    mysum = te.comm_reducer(lambda x, y: x + y, lambda t: tvm.tirx.const(0, dtype=t))
    sum_res = mysum(m, n)


def test_tensor_reduce():
    m = te.var("m")
    n = te.var("n")
    l = te.var("l")
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
        return tvm.tirx.const(0, t0), tvm.tirx.const(1, t1)

    mysum = te.comm_reducer(fcombine, fidentity, name="mysum")

    m = te.var("m")
    n = te.var("n")
    idx = te.placeholder((m, n), name="idx", dtype="int32")
    val = te.placeholder((m, n), name="val", dtype="int32")
    k = te.reduce_axis((0, n), "k")
    cond = te.floormod(k, 2) == 0
    T0, T1 = te.compute((m,), lambda i: mysum((idx[i, k], val[i, k]), axis=k, where=cond), name="T")


def test_tensor_scan():
    m = te.var("m")
    n = te.var("n")
    x = te.placeholder((m, n))
    s = te.placeholder((m, n))
    res = tvm.te.scan(
        te.compute((1, n), lambda _, i: x[0, i]),
        te.compute((m, n), lambda t, i: s[t - 1, i] + x[t, i]),
        s,
    )
    assert tuple(res.shape) == (m, n)


def test_scan_multi_out():
    m = te.var("m")
    n = te.var("n")
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
    m = te.var("m")
    A = te.placeholder((m,), name="A")

    def extern_func(ins, outs):
        assert isinstance(ins[0], tvm.tirx.Buffer)
        return tvm.tirx.call_packed("myadd", ins[0].data, outs[0].data, m)

    B = te.extern((m,), [A], extern_func)
    assert tuple(B.shape) == (m,)


def test_extern_multi_out():
    m = te.var("m")
    A = te.placeholder((m,), name="A")
    B = te.compute((m,), lambda i: A[i] * 10)

    def extern_func(ins, outs):
        assert isinstance(ins[0], tvm.tirx.Buffer)
        return tvm.tirx.call_packed("myadd", ins[0].data, outs[0].data, outs[1].data, m)

    res = te.extern([A.shape, A.shape], [A, B], extern_func)
    assert len(res) == 2
    assert res[1].value_index == 1


def test_tuple_inputs():
    m = te.var("m")
    n = te.var("n")
    A0 = te.placeholder((m, n), name="A0")
    A1 = te.placeholder((m, n), name="A1")
    T0, T1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name="T")
    s = te.create_prim_func([A0, A1, T0])


def test_tuple_with_different_deps():
    m = te.var("m")
    n = te.var("n")
    A0 = te.placeholder((m, n), name="A1")
    A1 = te.placeholder((m, n), name="A2")
    B0, B1 = te.compute((m, n), lambda i, j: (A0[i, j] * 2, A1[i, j] * 3), name="B")
    C = te.compute((m, n), lambda i, j: B0[i, j] + 4, name="C")

    te.create_prim_func([A0, A1, C])


def test_tensor_inputs():
    x = te.placeholder((1,), name="x")
    y = te.compute(x.shape, lambda i: x[i] + x[i])
    assert tuple(y.op.input_tensors) == (x,)


@pytest.mark.parametrize(
    "operation,direct,reflected",
    [
        (operator.add, "__add__", "__radd__"),
        (operator.sub, "__sub__", "__rsub__"),
        (operator.mul, "__mul__", "__rmul__"),
        (operator.truediv, "__truediv__", "__rtruediv__"),
    ],
)
def test_tensor_operator_ownership(monkeypatch, operation, direct, reflected):
    tensor = te.placeholder((4,), name="tensor", dtype="float32")
    other_tensor = te.placeholder((4,), name="other_tensor", dtype="float32")
    scalar = tvm.tirx.Var("scalar", "float32")
    tensor_slice = other_tensor[0]
    calls = []
    sentinel = object()

    def record_direct(lhs, rhs):
        calls.append(("direct", lhs, rhs))
        return sentinel

    def record_reflected(lhs, rhs):
        calls.append(("reflected", lhs, rhs))
        return sentinel

    monkeypatch.setattr(_te_tensor_overload, direct, record_direct)
    monkeypatch.setattr(_te_tensor_overload, reflected, record_reflected)

    # Python does not retry the reflected method for two objects of the same type.
    assert operation(tensor, other_tensor) is sentinel
    kind, lhs, rhs = calls.pop()
    assert kind == "direct"
    assert lhs is tensor
    assert rhs is other_tensor

    assert operation(tensor, scalar) is sentinel
    kind, lhs, rhs = calls.pop()
    assert kind == "direct"
    assert lhs is tensor
    assert rhs is scalar

    assert operation(scalar, tensor) is sentinel
    kind, lhs, rhs = calls.pop()
    assert kind == "reflected"
    assert lhs is tensor
    assert rhs is scalar

    assert operation(tensor_slice, tensor) is sentinel
    kind, lhs, rhs = calls.pop()
    assert kind == "direct"
    assert lhs is tensor_slice
    assert rhs is tensor


@pytest.mark.parametrize(
    "operation,expected_type",
    [
        (lambda lhs, rhs: lhs + rhs, tvm.tirx.Add),
        (lambda lhs, rhs: lhs - rhs, tvm.tirx.Sub),
        (lambda lhs, rhs: lhs * rhs, tvm.tirx.Mul),
        (lambda lhs, rhs: lhs / rhs, tvm.tirx.Div),
    ],
)
def test_scalar_operator_smart_constructors(operation, expected_type):
    lhs = tvm.tirx.Var("lhs", "float32")
    rhs = tvm.tirx.Var("rhs", "float32")

    assert isinstance(operation(lhs, rhs), expected_type)
    assert isinstance(lhs.astype("float16"), tvm.tirx.Cast)


@pytest.mark.parametrize(
    "operation,expected_type",
    [
        (operator.add, tvm.tirx.Add),
        (operator.sub, tvm.tirx.Sub),
        (operator.mul, tvm.tirx.Mul),
        (operator.truediv, tvm.tirx.Div),
    ],
)
def test_tensor_and_slice_operator_behavior(operation, expected_type):
    tensor = te.placeholder((4,), name="tensor", dtype="float32")
    other_tensor = te.placeholder((4,), name="other_tensor", dtype="float32")
    scalar = tvm.tirx.Var("scalar", "float32")
    tensor_slice = tensor[0]
    other_slice = other_tensor[0]

    for result in [
        operation(tensor, other_tensor),
        operation(tensor, scalar),
        operation(scalar, tensor),
        operation(tensor, tensor_slice),
        operation(tensor_slice, tensor),
        operation(tensor, 2.0),
        operation(2.0, tensor),
    ]:
        assert isinstance(result, te.Tensor)
        assert tuple(result.shape) == (4,)

    assert not isinstance(tensor_slice, tvm.tirx.expr.ExprOp)
    for result in [
        operation(tensor_slice, other_slice),
        operation(tensor_slice, scalar),
        operation(scalar, tensor_slice),
        operation(tensor_slice, 2.0),
        operation(2.0, tensor_slice),
    ]:
        assert isinstance(result, expected_type)


@pytest.mark.parametrize("operation", [operator.sub, operator.truediv])
def test_tensor_noncommutative_operand_order(operation):
    tensor = te.placeholder((4,), name="tensor", dtype="float32")
    scalar = tvm.tirx.Var("scalar", "float32")

    direct = operation(tensor, scalar).op.body[0]
    assert isinstance(direct.a, tvm.tirx.ProducerLoad)
    assert direct.b.same_as(scalar)

    reflected = operation(scalar, tensor).op.body[0]
    assert reflected.a.same_as(scalar)
    assert isinstance(reflected.b, tvm.tirx.ProducerLoad)

    slice_direct = operation(tensor[0], scalar)
    assert isinstance(slice_direct.a, tvm.tirx.ProducerLoad)
    assert slice_direct.b.same_as(scalar)

    slice_reflected = operation(scalar, tensor[0])
    assert slice_reflected.a.same_as(scalar)
    assert isinstance(slice_reflected.b, tvm.tirx.ProducerLoad)


def test_tensor_slice_scalar_surface():
    tensor = te.placeholder((4,), name="tensor", dtype="int32")
    tensor_slice = tensor[0]

    assert not isinstance(tensor_slice, tvm.tirx.expr.ExprOp)
    assert isinstance(tensor_slice // 2, tvm.tirx.FloorDiv)
    assert isinstance(tensor_slice % 2, tvm.tirx.FloorMod)
    assert isinstance(tensor_slice << 1, tvm.ir.Expr)
    assert isinstance(tensor_slice < 2, tvm.tirx.LT)
    assert isinstance(tensor_slice.astype("float32"), tvm.tirx.Cast)
    with pytest.raises(ValueError, match="Cannot use and / or / not operator"):
        bool(tensor_slice)

    partial_slice = te.placeholder((4, 4), name="matrix", dtype="float32")[0]
    with pytest.raises(ValueError, match="Need to provide 2 index"):
        partial_slice + 1.0


def test_tensor_slice_scalar_fallback_without_topi_hook(monkeypatch):
    tensor_slice = te.placeholder((4,), name="tensor", dtype="float32")[0]
    scalar = tvm.tirx.Var("scalar", "float32")

    monkeypatch.setattr(_te_tensor_overload, "__add__", lambda _lhs, _rhs: NotImplemented)
    monkeypatch.setattr(_te_tensor_overload, "__radd__", lambda _lhs, _rhs: NotImplemented)
    monkeypatch.setattr(
        _te_tensor_overload, "astype", lambda _value, _dtype, _span=None: NotImplemented
    )

    assert isinstance(tensor_slice + scalar, tvm.tirx.Add)
    assert isinstance(scalar + tensor_slice, tvm.tirx.Add)
    assert isinstance(tensor_slice.astype("float16"), tvm.tirx.Cast)


def test_primitive_call_scalar_operand():
    scalar = tvm.tirx.Var("scalar", "float32")
    call = tvm.ir.Call(tvm.ir.GlobalVar("f"), [], ret_ty="float32")

    assert tvm.ir.is_prim_expr(call)
    assert isinstance(scalar + call, tvm.tirx.Add)
    assert isinstance(call + scalar, tvm.tirx.Add)
    assert isinstance(call + call, tvm.tirx.Add)


def test_tensor_rank_zero_and_astype(monkeypatch):
    tensor = te.placeholder((), name="tensor", dtype="float32")
    other_tensor = te.placeholder((), name="other_tensor", dtype="float32")
    scalar = tvm.tirx.Var("scalar", "float32")
    cast_sentinel = object()
    cast_calls = []

    for result in [tensor + other_tensor, tensor + scalar, scalar + tensor]:
        assert isinstance(result, te.Tensor)
        assert tuple(result.shape) == ()

    tensor_slice = tensor[()]
    assert isinstance(tensor_slice + scalar, tvm.tirx.Add)
    assert isinstance(scalar + tensor_slice, tvm.tirx.Add)

    def record_cast(value, dtype, span=None):
        cast_calls.append((value, dtype, span))
        return cast_sentinel

    monkeypatch.setattr(_te_tensor_overload, "astype", record_cast)
    assert tensor.astype("float16") is cast_sentinel
    value, dtype, span = cast_calls.pop()
    assert value is tensor
    assert dtype == "float16"
    assert span is None


def test_tensor_operator_hook_import_order():
    script = """
import inspect
import tvm
from tvm import te, topi
from tvm.te import _te_tensor_overload
import tvm.te.tensor as tensor_module

assert _te_tensor_overload.__add__.__module__ == "tvm.topi._te_tensor_overload"
assert "from tvm import topi" not in inspect.getsource(tensor_module)
tensor = te.placeholder((4,), name="tensor", dtype="float32")
other = te.placeholder((4,), name="other", dtype="float32")
assert isinstance(tensor + other, te.Tensor)
assert isinstance(tensor.astype("float16"), te.Tensor)
"""
    env = os.environ.copy()
    env["TVM_DEVICE_BACKEND_AUTOLOAD"] = "0"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )


def test_tensor_integer_division_remains_ambiguous():
    tensor = te.placeholder((4,), name="tensor", dtype="int32")
    other_tensor = te.placeholder((4,), name="other_tensor", dtype="int32")
    scalar = tvm.tirx.Var("scalar", "int32")
    tensor_slice = tensor[0]
    other_slice = other_tensor[0]

    # ExprOp must decline a whole DataProducer before applying scalar-only
    # integer-division ambiguity checks.  Tensor then owns the final decision.
    assert scalar.__truediv__(tensor) is NotImplemented

    for divide in [
        lambda: tensor / other_tensor,
        lambda: tensor / 2,
        lambda: 2 / tensor,
        lambda: tensor / scalar,
        lambda: scalar / tensor,
        lambda: tensor / tensor_slice,
        lambda: tensor_slice / tensor,
        lambda: tensor_slice / other_slice,
        lambda: tensor_slice / scalar,
        lambda: scalar / tensor_slice,
    ]:
        with pytest.raises(RuntimeError, match="multiple types of integer divisions"):
            divide()


def test_removed_tirx_generic_surface():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tvm.tirx.generic")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("tvm.topi.generic_op_impl")

    assert not hasattr(tvm.tirx, "generic")
    assert not hasattr(tvm.tirx, "add")
    assert not hasattr(tvm.tirx, "subtract")
    assert not hasattr(tvm.tirx, "multiply")
    assert not hasattr(tvm.te, "add")
    assert not hasattr(tvm.te, "subtract")
    assert not hasattr(tvm.te, "multiply")
    assert "__floordiv__" not in te.Tensor.__dict__


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
