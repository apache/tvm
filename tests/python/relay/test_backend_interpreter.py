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
from tvm import te
import tvm.testing
from tvm import nd
from tvm import relay
from tvm.runtime import container
from tvm.relay.backend.interpreter import RefValue, ConstructorValue
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import testing, create_executor


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    # TODO(tqchen) add more types once the schedule register is fixed.
    for target in ["llvm"]:
        ctx = tvm.context(target, 0)
        if not tvm.testing.device_enabled(target):
            return
        intrp = create_executor(mod=mod, ctx=ctx, target=target)
        result = intrp.evaluate(expr)(*args)
        # use tvm.testing which also set atol
        tvm.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def test_tuple_value():
    tv = container.tuple_object([relay.const(1), relay.const(2), relay.const(3)])
    np.testing.assert_allclose(tv[0].data.asnumpy(), 1)
    np.testing.assert_allclose(tv[1].data.asnumpy(), 2)
    np.testing.assert_allclose(tv[2].data.asnumpy(), 3)


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
    intrp = create_executor("debug")
    xx = np.ones((10, 20))
    res = intrp.evaluate(y, binds={x: xx}).asnumpy()
    tvm.testing.assert_allclose(xx + xx, res)


def test_kwargs_params():
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.var("z", shape=(1, 10))
    f = relay.Function([x, y, z], x + y + z)
    x_data = np.random.rand(1, 10).astype("float32")
    y_data = np.random.rand(1, 10).astype("float32")
    z_data = np.random.rand(1, 10).astype("float32")
    params = {"y": y_data, "z": z_data}
    intrp = create_executor("debug")
    res = intrp.evaluate(f)(x_data, **params)
    tvm.testing.assert_allclose(res.asnumpy(), x_data + y_data + z_data)


def test_function_taking_adt_ref_tuple():
    mod = tvm.IRModule()
    prelude = relay.prelude.Prelude(mod)
    intrp = create_executor("debug", mod)

    nil_value = ConstructorValue(prelude.nil.tag, [], prelude.nil)
    cons_value = ConstructorValue(
        prelude.cons.tag,
        [nd.array(np.random.rand(1, 10).astype("float32")), nil_value],
        prelude.cons,
    )

    ref_value = RefValue(nd.array(np.random.rand(1, 10).astype("float32")))
    tuple_value = container.tuple_object(
        [nd.array(np.random.rand(1, 10).astype("float32")) for _ in range(10)]
    )

    id_func = intrp.evaluate(prelude.id)

    res_nil = id_func(nil_value)
    assert res_nil.tag == nil_value.tag
    assert len(res_nil.fields) == 0

    res_cons = id_func(cons_value)
    assert res_cons.tag == cons_value.tag
    assert len(res_cons.fields) == len(cons_value.fields)
    tvm.testing.assert_allclose(res_cons.fields[0].asnumpy(), cons_value.fields[0].asnumpy())
    assert isinstance(res_cons.fields[1], ConstructorValue)
    assert res_cons.fields[1].tag == prelude.nil.tag
    assert len(res_cons.fields[1].fields) == 0

    res_ref = id_func(ref_value)
    tvm.testing.assert_allclose(res_ref.value.asnumpy(), ref_value.value.asnumpy())

    res_tuple = id_func(tuple_value)
    for i in range(10):
        tvm.testing.assert_allclose(res_tuple[i].asnumpy(), tuple_value[i].asnumpy())


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

    ctx = tvm.cpu()
    target = tvm.target.Target("llvm")
    exec = relay.create_executor(mod=mod, ctx=ctx, target=target)
    f = exec.evaluate(gv)
    # First use a Python tuple.
    out = f((10, 8))
    tvm.testing.assert_allclose(out.asnumpy(), np.array(10))
    # Second use a tuple value.
    value_tuple = container.tuple_object([nd.array(np.array(11)), nd.array(np.array(12))])
    out = f(value_tuple)
    tvm.testing.assert_allclose(out.asnumpy(), np.array(11))


if __name__ == "__main__":
    test_id()
    test_add_const()
    test_equal()
    test_subtract()
    test_simple_loop()
    test_loop()
    test_binds()
    test_kwargs_params()
    test_ref()
    test_tuple_value()
    test_tuple_getitem()
    test_function_taking_adt_ref_tuple()
    test_tuple_passing()
