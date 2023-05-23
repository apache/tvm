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
import collections
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import GlobalVar
from tvm.relay.analysis import free_vars, free_type_vars
from tvm.relay import create_executor, transform
from tvm.relay.transform import gradient
from tvm.relay.prelude import Prelude
from tvm.relay.testing import (
    make_nat_expr,
    run_infer_type,
    check_grad,
    rand,
    count_ops,
)
import tvm.relay.op as op


def test_fo_id():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func, mode="first_order"))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), np.ones_like(x.numpy()))


def test_id():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), np.ones_like(x.numpy()))


def test_relu():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], op.nn.relu(x))
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    # gradient will implicitly check that no graph appear in result


def test_add():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x + x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), 2 * x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), 2 * np.ones_like(x.numpy()))


def test_check_grad():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    y = relay.var("y", t)
    func = relay.Function([x, y], x + y)
    check_grad(func)


def test_temp_add():
    scope = relay.ScopeBuilder()
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    y = scope.let("y", x + x)
    scope.ret(y + y)
    func = relay.Function([x], scope.get())
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), 4 * x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), 4 * np.ones_like(x.numpy()))


def test_sub():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x - x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), np.zeros_like(x.numpy()))
    tvm.testing.assert_allclose(grad.numpy(), np.zeros_like(x.numpy()))


def test_broadcast_add():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    x_nd = rand(dtype, *shape1)
    y_nd = rand(dtype, *shape2)
    x_np = x_nd.numpy()
    y_np = y_nd.numpy()
    expected_forward = x_np + y_np
    t1 = relay.TensorType(shape1, dtype)
    t2 = relay.TensorType(shape2, dtype)
    x = relay.var("x", t1)
    y = relay.var("y", t2)
    func = relay.Function([x, y], x + y)
    func = run_infer_type(func)
    full_func = run_infer_type(gradient(func))
    assert full_func.checked_type == relay.FuncType(
        [t1, t2],
        relay.TupleType(
            [relay.TensorType(expected_forward.shape, dtype), relay.TupleType([t1, t2])]
        ),
    )
    forward, (grad_x, grad_y) = create_executor().evaluate(full_func)(x_nd, y_nd)
    tvm.testing.assert_allclose(forward.numpy(), expected_forward)
    tvm.testing.assert_allclose(
        grad_x.numpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True)
    )
    tvm.testing.assert_allclose(
        grad_y.numpy(),
        np.ones_like(expected_forward).sum(axis=(0, 1), keepdims=True).squeeze(axis=0),
    )


def test_broadcast_subtract():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    x_nd = rand(dtype, *shape1)
    y_nd = rand(dtype, *shape2)
    x_np = x_nd.numpy()
    y_np = y_nd.numpy()
    expected_forward = x_np - y_np
    t1 = relay.TensorType(shape1, dtype)
    t2 = relay.TensorType(shape2, dtype)
    x = relay.var("x", t1)
    y = relay.var("y", t2)
    func = relay.Function([x, y], x - y)
    func = run_infer_type(func)
    full_func = run_infer_type(gradient(func))
    assert full_func.checked_type == relay.FuncType(
        [t1, t2],
        relay.TupleType(
            [relay.TensorType(expected_forward.shape, dtype), relay.TupleType([t1, t2])]
        ),
    )
    forward, (grad_x, grad_y) = create_executor().evaluate(full_func)(x_nd, y_nd)
    tvm.testing.assert_allclose(forward.numpy(), expected_forward)
    tvm.testing.assert_allclose(
        grad_x.numpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True)
    )
    tvm.testing.assert_allclose(
        grad_y.numpy(),
        -np.ones_like(expected_forward).sum(axis=(0, 1), keepdims=True).squeeze(axis=0),
    )


def _test_tuple(mode):
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    y = relay.var("y", t)
    z = relay.var("z", t)
    if mode == "higher_order":
        tup = relay.Var("tup")
        func = relay.Function(
            [x, y, z],
            relay.Let(
                tup,
                relay.Tuple([x, y, z]),
                relay.TupleGetItem(tup, 0)
                + relay.TupleGetItem(tup, 1)
                - relay.TupleGetItem(tup, 2),
            ),
        )
    else:
        # first order does not do let.
        tup = relay.Tuple([x, y, z])
        func = relay.Function(
            [x, y, z],
            relay.TupleGetItem(tup, 0) + relay.TupleGetItem(tup, 1) - relay.TupleGetItem(tup, 2),
        )
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func, mode=mode))
    assert back_func.checked_type == relay.FuncType(
        [t, t, t], relay.TupleType([t, relay.TupleType([t, t, t])])
    )
    x_nd = rand(dtype, *shape)
    y_nd = rand(dtype, *shape)
    z_nd = rand(dtype, *shape)
    x_np = x_nd.numpy()
    y_np = y_nd.numpy()
    z_np = z_nd.numpy()
    expected_forward = x_np + y_np - z_np
    forward, (grad_x, grad_y, grad_z) = create_executor().evaluate(back_func)(x_nd, y_nd, z_nd)
    tvm.testing.assert_allclose(forward.numpy(), expected_forward)
    tvm.testing.assert_allclose(grad_x.numpy(), np.ones_like(grad_x.numpy()))
    tvm.testing.assert_allclose(grad_y.numpy(), np.ones_like(grad_y.numpy()))
    tvm.testing.assert_allclose(grad_z.numpy(), -1 * np.ones_like(grad_z.numpy()))


def _test_tuple_argument(mode):
    shape = (2, 3)
    dtype = "float32"
    tensor_type = relay.TensorType(shape, dtype)
    fields = 3
    tuple_type = relay.TupleType([tensor_type] * fields)
    tup = relay.var("tup", type_annotation=tuple_type)
    body = relay.TupleGetItem(tup, 0)
    for i in range(1, fields):
        body = relay.add(body, relay.TupleGetItem(tup, i))
    func = relay.Function([tup], body)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func, mode=mode))
    xs = [rand(dtype, *shape) for _ in range(fields)]
    xs_np = np.array([x.numpy() for x in xs])
    expected_forward = np.sum(xs_np, axis=0)
    forward, grad = create_executor().evaluate(back_func)(tuple(xs))
    tvm.testing.assert_allclose(forward.numpy(), expected_forward)
    for field in grad[0]:
        tvm.testing.assert_allclose(field.numpy(), np.ones_like(field.numpy()))


def test_tuple():
    _test_tuple("higher_order")


def test_tuple_first_order():
    _test_tuple("first_order")


@pytest.mark.xfail(raises=tvm.error.TVMError)
def test_tuple_argument():
    # fails until we add support for top-level tuple arguments in higher-order AD
    _test_tuple_argument("higher_order")


def test_tuple_argument_first_order():
    _test_tuple_argument("first_order")


def test_pow():
    mod = tvm.IRModule()
    p = Prelude(mod)
    p.mod.import_from_std("nat.rly")
    nat_iterate = mod.get_global_var("nat_iterate")
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    double = relay.Function([x], x + x)
    i = relay.var("i", t)
    func = relay.Function([i], nat_iterate(double, make_nat_expr(p, 3))(i))
    mod["main"] = func
    mod = transform.InferType()(mod)
    mod["main"] = gradient(mod["main"], mod=mod)
    m = transform.InferType()(mod)
    back_func = m["main"]
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    i_nd = rand(dtype, *shape)
    forward, (grad_i,) = create_executor(mod=mod).evaluate(back_func)(i_nd)
    tvm.testing.assert_allclose(forward.numpy(), 8 * i_nd.numpy())
    tvm.testing.assert_allclose(grad_i.numpy(), 8 * np.ones_like(grad_i.numpy()))


def test_ref():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    r = relay.Var("r")
    u = relay.Var("u")
    body = relay.RefRead(r)
    body = relay.Let(u, relay.RefWrite(r, relay.RefRead(r) + relay.RefRead(r)), body)
    body = relay.Let(r, relay.RefCreate(x), body)
    func = relay.Function([x], body)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x_nd = rand(dtype, *shape)
    forward, (grad_x,) = create_executor().evaluate(back_func)(x_nd)
    tvm.testing.assert_allclose(forward.numpy(), 2 * x_nd.numpy())
    tvm.testing.assert_allclose(grad_x.numpy(), 2 * np.ones_like(grad_x.numpy()))


def test_square_second_order():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x * x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    y = relay.var("y", t)
    back_func_adjusted = relay.Function(
        [y], relay.TupleGetItem(relay.TupleGetItem(back_func(y), 1), 0)
    )
    back_func_adjusted = run_infer_type(back_func_adjusted)
    back_back_func = run_infer_type(gradient(back_func_adjusted))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x_nd = rand(dtype, *shape)
    forward, (grad_x,) = create_executor().evaluate(back_back_func)(x_nd)
    tvm.testing.assert_allclose(forward.numpy(), 2 * x_nd.numpy())
    tvm.testing.assert_allclose(grad_x.numpy(), 2 * np.ones_like(grad_x.numpy()))


def test_if():
    x = relay.var("x", shape=(1, 16, 64, 64))
    y = relay.var("y", shape=(1, 16, 64, 64))
    cond = relay.var("cond", shape=(), dtype="uint1")
    net = relay.If(cond, x, y)
    net = relay.log(net)
    func = relay.Function(free_vars(net), net)
    func = run_infer_type(func)
    net = gradient(func, mode="higher_order")
    net = run_infer_type(net)


def test_grad_tuple():
    scope = relay.ScopeBuilder()
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    y = scope.let("y", x + x)
    scope.ret(relay.Tuple([y + y, y]))
    func = relay.Function([x], scope.get())
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType(
        [t], relay.TupleType([relay.TupleType([t, t]), relay.TupleType([t])])
    )
    x = rand(dtype, *shape)
    (forward_four, forward_two), (grad,) = create_executor().evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward_four.numpy(), 4 * x.numpy())
    tvm.testing.assert_allclose(forward_two.numpy(), 2 * x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), 4 * np.ones_like(x.numpy()))


def test_concat():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    rt = relay.TensorType((10, 20), dtype)
    x = relay.var("x", t)
    y = op.concatenate([x, x], axis=1)
    func = relay.Function([x], y)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    tvm.ir.assert_structural_equal(
        back_func.checked_type, relay.FuncType([t], relay.TupleType([rt, relay.TupleType([t])]))
    )
    # no value validation as concatenate has dummy gradient right now.


def test_no_duplication():
    x = tvm.relay.Var("x", type_annotation=tvm.relay.TensorType([12, 12]))
    y = tvm.relay.Var("y", type_annotation=tvm.relay.TensorType([12, 12]))
    xy = tvm.relay.nn.dense(x, y)

    m = tvm.relay.sum(xy, keepdims=True)
    s = tvm.relay.sum(xy - m)
    fn = tvm.relay.Function([x, y], s)
    fn = run_infer_type(fn)
    gr = tvm.relay.transform.gradient(fn, mode="first_order")

    counts = count_ops(gr)
    assert counts["nn.dense"] == 3, "We expect 3 dense (1 forward, two backward)"


def test_no_duplication_tuples():
    x = tvm.relay.Var("x", type_annotation=tvm.relay.TensorType([12, 12]))
    y = tvm.relay.Var("y", type_annotation=tvm.relay.TensorType([12, 12]))
    xy = tvm.relay.nn.dense(x, y)

    t = relay.Tuple([xy, xy])

    m = tvm.relay.sum(xy, keepdims=True)
    s = tvm.relay.sum(relay.TupleGetItem(t, 0) - m)
    fn = tvm.relay.Function([x, y], s)
    fn = run_infer_type(fn)
    gr = tvm.relay.transform.gradient(fn, mode="first_order")

    counts = count_ops(gr)
    assert counts["nn.dense"] == 3, "We expect 3 dense (1 forward, two backward)"


def test_global_function():
    m = tvm.IRModule()
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.Var("x", t)
    d = GlobalVar("double")
    m[d] = relay.Function([x], x + x)
    y = relay.Var("y", t)
    q = GlobalVar("q")
    m[q] = relay.Function([y], d(d(y)))
    g = GlobalVar("grad")
    m = tvm.relay.transform.InferType()(m)
    m[g] = tvm.relay.transform.gradient(q, m)
    m = tvm.relay.transform.InferType()(m)
    back_func = m[g]
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    x = rand(dtype, *shape)
    forward, (grad,) = create_executor(mod=m).evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.numpy(), 4 * x.numpy())
    tvm.testing.assert_allclose(grad.numpy(), 4 * np.ones_like(x.numpy()))


if __name__ == "__main__":
    tvm.testing.main()
