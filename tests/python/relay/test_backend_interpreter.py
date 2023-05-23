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
import pytest
import tvm
from tvm import testing
from tvm import nd
from tvm import relay
from tvm.runtime import container
from tvm.relay.backend.interpreter import RefValue, ConstructorValue
from tvm.relay.scope_builder import ScopeBuilder


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    # TODO(tqchen) add more types once the schedule register is fixed.
    for target in ["llvm"]:
        dev = tvm.device(target, 0)
        if not testing.device_enabled(target):
            return
        func = relay.create_executor(mod=mod, device=dev, target=target).evaluate(expr)
        result = func if args is None else func(*args)
        # use testing which also set atol
        testing.assert_allclose(result.numpy(), expected_result, rtol=rtol)


def test_tuple_value():
    tv = container.tuple_object([relay.const(1), relay.const(2), relay.const(3)])
    np.testing.assert_allclose(tv[0].data.numpy(), 1)
    np.testing.assert_allclose(tv[1].data.numpy(), 2)
    np.testing.assert_allclose(tv[2].data.numpy(), 3)


def test_tuple_getitem():
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], relay.TupleGetItem(relay.Tuple([relay.const(1), relay.const(2)]), 0))
    check_eval(func, [], 1)


def test_id():
    x = relay.var("x", "float32")
    ident = relay.Function([x], x)
    one = np.array(1.0, "float32")
    check_eval(ident, [one], one)


def test_add_const():
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    check_eval(func, [], 2)


def test_mul_param():
    x = relay.var("x", shape=(10, 10))
    y = relay.var("y", shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")
    check_eval(func, [x_data, y_data], x_data * y_data)


def test_equal():
    i = relay.var("i", shape=[], dtype="int32")
    j = relay.var("i", shape=[], dtype="int32")
    z = relay.equal(i, j)
    func = relay.Function([i, j], z, ret_type=relay.TensorType([], "bool"))
    i_data = relay.const(0, "int32")
    j_data = relay.const(0, "int32")
    check_eval(func, [i_data, j_data], True)


def test_subtract():
    i = relay.var("i", shape=[], dtype="int32")
    sub = relay.subtract(i, relay.const(1, dtype="int32"))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], "int32"))
    i_data = np.array(1, dtype="int32")
    check_eval(func, [i_data], 0)


def test_simple_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype="int32"))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype="int32"))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], "int32"))
    mod[sum_up] = func
    i_data = np.array(10, dtype="int32")
    check_eval(sum_up, [i_data], sum(range(1, 11)), mod=mod)


def test_loop():
    mod = tvm.IRModule({})
    sum_up = relay.GlobalVar("sum_up")
    i = relay.var("i", shape=[], dtype="int32")
    accum = relay.var("accum", shape=[], dtype="int32")
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, "int32"))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, "int32"))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    i_data = np.array(10, dtype="int32")
    accum_data = np.array(0, dtype="int32")
    check_eval(sum_up, [i_data, accum_data], sum(range(1, 11)), mod=mod)


def test_ref():
    mod = tvm.IRModule()
    three_with_ref = relay.GlobalVar("three_with_ref")
    i = relay.Var("i")
    iv = relay.Var("iv")
    u = relay.Var("u")
    uv = relay.Var("uv")
    body = relay.add(iv, uv)
    body = relay.Let(uv, relay.RefRead(i), body)
    body = relay.Let(u, relay.RefWrite(i, relay.const(2)), body)
    body = relay.Let(iv, relay.RefRead(i), body)
    body = relay.Let(i, relay.RefCreate(relay.const(1)), body)
    mod[three_with_ref] = relay.Function([], body)
    check_eval(three_with_ref, [], 3, mod=mod)


def test_binds():
    x = relay.var("x")
    y = relay.add(x, x)
    xx = np.ones((10, 20))
    res = relay.create_executor().evaluate(y, binds={x: xx}).numpy()
    testing.assert_allclose(xx + xx, res)


def test_kwargs_params():
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.var("z", shape=(1, 10))
    f = relay.Function([x, y, z], x + y + z)
    x_data = np.random.rand(1, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")
    z_data = np.random.rand(1, 10).astype("float32")
    params = {"y": y_data, "z": z_data}
    res = relay.create_executor().evaluate(f)(x_data, **params)
    testing.assert_allclose(res.numpy(), x_data + y_data + z_data)


def test_function_taking_adt_ref_tuple():
    mod = tvm.IRModule()
    prelude = relay.prelude.Prelude(mod)
    _, cons, nil = prelude.mod.get_type("List")

    nil_value = ConstructorValue(nil.tag, [], nil)
    cons_value = ConstructorValue(
        cons.tag,
        [nd.array(np.random.rand(1, 10).astype("float32")), nil_value],
        cons,
    )

    ref_value = RefValue(nd.array(np.random.rand(1, 10).astype("float32")))
    tuple_value = container.tuple_object(
        [nd.array(np.random.rand(1, 10).astype("float32")) for _ in range(10)]
    )

    id_func = relay.create_executor(mod=mod).evaluate(prelude.id)

    res_nil = id_func(nil_value)
    assert res_nil.tag == nil_value.tag
    assert len(res_nil.fields) == 0

    res_cons = id_func(cons_value)
    assert res_cons.tag == cons_value.tag
    assert len(res_cons.fields) == len(cons_value.fields)
    testing.assert_allclose(res_cons.fields[0].numpy(), cons_value.fields[0].numpy())
    assert isinstance(res_cons.fields[1], ConstructorValue)
    assert res_cons.fields[1].tag == nil.tag
    assert len(res_cons.fields[1].fields) == 0

    res_ref = id_func(ref_value)
    testing.assert_allclose(res_ref.value.numpy(), ref_value.value.numpy())

    res_tuple = id_func(tuple_value)
    for i in range(10):
        testing.assert_allclose(res_tuple[i].numpy(), tuple_value[i].numpy())


def test_tuple_passing():
    x = relay.var(
        "x",
        type_annotation=relay.ty.TupleType(
            [relay.ty.TensorType((), "int64"), relay.ty.TensorType((), "int64")]
        ),
    )

    fn = relay.Function([x], relay.expr.TupleGetItem(x, 0))
    mod = tvm.IRModule({})
    gv = relay.GlobalVar("main")
    mod[gv] = fn
    mod = relay.transform.InferType()(mod)

    dev = tvm.cpu()
    target = tvm.target.Target("llvm")
    f = relay.create_executor(mod=mod, device=dev, target=target).evaluate(gv)
    # First use a Python tuple.
    out = f((10, 8))
    testing.assert_allclose(out.numpy(), np.array(10))
    # Second use a tuple value.
    value_tuple = container.tuple_object([nd.array(np.array(11)), nd.array(np.array(12))])
    out = f(value_tuple)
    testing.assert_allclose(out.numpy(), np.array(11))


def test_dynamic():
    n = 3
    m = 2
    x = relay.Var("x", relay.TensorType([relay.Any(), m], "float32"))
    y = relay.Var("y", relay.TensorType([relay.Any(), m], "float32"))
    xx = x - relay.expr.const(3.0)
    yy = y * relay.expr.const(5.0)
    z = relay.op.concatenate([xx, yy], axis=0)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)
    x_np = np.random.uniform(size=(n, m)).astype("float32")
    y_np = np.random.uniform(size=(n, m)).astype("float32")
    expected = np.concatenate([x_np - 3.0, y_np * 5.0], axis=0)
    check_eval(None, [x_np, y_np], expected, mod)


def test_ref_global_from_expr():
    n = 3
    x = relay.Var("x", relay.TensorType([n], "float32"))
    y = relay.Var("y", relay.TensorType([n], "float32"))
    mod = tvm.IRModule()
    mod["add"] = relay.Function([x, y], relay.add(x, y))
    x_np = np.random.uniform(size=(n,)).astype("float32")
    y_np = np.random.uniform(size=(n,)).astype("float32")
    expected = np.add(x_np, y_np)
    expr = relay.Call(mod.get_global_var("add"), [relay.const(x_np), relay.const(y_np)])
    check_eval(expr, None, expected, mod)


def test_keyword_args():
    n = 3
    x = relay.Var("x", relay.TensorType([n], "float32"))
    y = relay.Var("y", relay.TensorType([n], "float32"))
    z = relay.add(x, y)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)
    x_np = np.random.uniform(size=(n,)).astype("float32")
    y_np = np.random.uniform(size=(n,)).astype("float32")
    expected = np.add(x_np, y_np)
    actual = relay.create_executor(mod=mod).evaluate()(y=y_np, x=x_np)
    testing.assert_allclose(actual.numpy(), expected)


# TODO(mbs): Support? Would help reduce wasted work when we need to prepare
# multiple functions w.r.t. the same module.
@pytest.mark.skip(reason="closures are currently not directly Python callable")
def test_functional_returns():
    n = 3
    x = relay.Var("x", relay.TensorType([n], "float32"))
    f = relay.Function([x], x)
    t = relay.Tuple([f, f])
    c = np.random.rand(n).astype("float32")
    result1, result2 = relay.create_executor().evaluate(t)
    testing.assert_allclose(result1(c).numpy(), c)
    testing.assert_allclose(result2(c).numpy(), c)


if __name__ == "__main__":
    tvm.testing.main()
