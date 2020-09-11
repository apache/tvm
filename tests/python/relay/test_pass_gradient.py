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
    add_nat_definitions,
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
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), np.ones_like(x.asnumpy()))


def test_id():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), np.ones_like(x.asnumpy()))


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
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), 2 * x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), 2 * np.ones_like(x.asnumpy()))


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
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), 4 * x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), 4 * np.ones_like(x.asnumpy()))


def test_sub():
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x - x)
    func = run_infer_type(func)
    back_func = run_infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), np.zeros_like(x.asnumpy()))
    tvm.testing.assert_allclose(grad.asnumpy(), np.zeros_like(x.asnumpy()))


def test_broadcast_add():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    x_nd = rand(dtype, *shape1)
    y_nd = rand(dtype, *shape2)
    x_np = x_nd.asnumpy()
    y_np = y_nd.asnumpy()
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
    ex = create_executor()
    forward, (grad_x, grad_y) = ex.evaluate(full_func)(x_nd, y_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), expected_forward)
    tvm.testing.assert_allclose(
        grad_x.asnumpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True)
    )
    tvm.testing.assert_allclose(
        grad_y.asnumpy(),
        np.ones_like(expected_forward).sum(axis=(0, 1), keepdims=True).squeeze(axis=0),
    )


def test_broadcast_subtract():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    x_nd = rand(dtype, *shape1)
    y_nd = rand(dtype, *shape2)
    x_np = x_nd.asnumpy()
    y_np = y_nd.asnumpy()
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
    ex = create_executor()
    forward, (grad_x, grad_y) = ex.evaluate(full_func)(x_nd, y_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), expected_forward)
    tvm.testing.assert_allclose(
        grad_x.asnumpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True)
    )
    tvm.testing.assert_allclose(
        grad_y.asnumpy(),
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
    x_np = x_nd.asnumpy()
    y_np = y_nd.asnumpy()
    z_np = z_nd.asnumpy()
    expected_forward = x_np + y_np - z_np
    ex = create_executor()
    forward, (grad_x, grad_y, grad_z) = ex.evaluate(back_func)(x_nd, y_nd, z_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), expected_forward)
    tvm.testing.assert_allclose(grad_x.asnumpy(), np.ones_like(grad_x.asnumpy()))
    tvm.testing.assert_allclose(grad_y.asnumpy(), np.ones_like(grad_y.asnumpy()))
    tvm.testing.assert_allclose(grad_z.asnumpy(), -1 * np.ones_like(grad_z.asnumpy()))


def test_tuple():
    _test_tuple("higher_order")


def test_tuple_first_order():
    _test_tuple("first_order")


def test_pow():
    mod = tvm.IRModule()
    p = Prelude(mod)
    add_nat_definitions(p)
    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    double = relay.Function([x], x + x)
    i = relay.var("i", t)
    func = relay.Function([i], p.nat_iterate(double, make_nat_expr(p, 3))(i))
    mod["main"] = func
    mod["main"] = gradient(mod["main"], mod=mod)
    m = transform.InferType()(mod)
    back_func = m["main"]
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    i_nd = rand(dtype, *shape)
    ex = create_executor(mod=mod)
    forward, (grad_i,) = ex.evaluate(back_func)(i_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), 8 * i_nd.asnumpy())
    tvm.testing.assert_allclose(grad_i.asnumpy(), 8 * np.ones_like(grad_i.asnumpy()))


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
    ex = create_executor()
    forward, (grad_x,) = ex.evaluate(back_func)(x_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), 2 * x_nd.asnumpy())
    tvm.testing.assert_allclose(grad_x.asnumpy(), 2 * np.ones_like(grad_x.asnumpy()))


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
    ex = create_executor()
    forward, (grad_x,) = ex.evaluate(back_back_func)(x_nd)
    tvm.testing.assert_allclose(forward.asnumpy(), 2 * x_nd.asnumpy())
    tvm.testing.assert_allclose(grad_x.asnumpy(), 2 * np.ones_like(grad_x.asnumpy()))


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
    ex = create_executor()
    x = rand(dtype, *shape)
    (forward_four, forward_two), (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward_four.asnumpy(), 4 * x.asnumpy())
    tvm.testing.assert_allclose(forward_two.asnumpy(), 2 * x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), 4 * np.ones_like(x.asnumpy()))


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
    m[g] = tvm.relay.transform.gradient(q, m)
    back_func = m[g]
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor(mod=m)
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    tvm.testing.assert_allclose(forward.asnumpy(), 4 * x.asnumpy())
    tvm.testing.assert_allclose(grad.asnumpy(), 4 * np.ones_like(x.asnumpy()))


if __name__ == "__main__":
    pytest.main([__file__])
