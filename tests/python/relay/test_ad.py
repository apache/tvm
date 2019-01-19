import tvm
from tvm import relay
from tvm.relay.ir_pass import free_vars, free_type_vars, gradient
from tvm.relay import create_executor

import numpy as np

def rand(dtype='float32', *shape):
    return tvm.nd.array(np.random.rand(*shape).astype(dtype))

def test_id():
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x)
    back_func = relay.ir_pass.infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    np.testing.assert_allclose(forward.asnumpy(), x.asnumpy())
    np.testing.assert_allclose(grad.asnumpy(), np.ones_like(x.asnumpy()))


def test_add():
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x + x)
    back_func = relay.ir_pass.infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    np.testing.assert_allclose(forward.asnumpy(), 2 * x.asnumpy())
    np.testing.assert_allclose(grad.asnumpy(), 2 * np.ones_like(x.asnumpy()))


def test_temp_add():
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    y = x + x
    func = relay.Function([x], y + y)
    back_func = relay.ir_pass.infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    np.testing.assert_allclose(forward.asnumpy(), 4 * x.asnumpy())
    np.testing.assert_allclose(grad.asnumpy(), 4 * np.ones_like(x.asnumpy()))


def test_sub():
    shape = (10, 10)
    dtype = 'float32'
    t = relay.TensorType(shape, dtype)
    x = relay.var("x", t)
    func = relay.Function([x], x - x)
    back_func = relay.ir_pass.infer_type(gradient(func))
    assert back_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))
    ex = create_executor()
    x = rand(dtype, *shape)
    forward, (grad,) = ex.evaluate(back_func)(x)
    np.testing.assert_allclose(forward.asnumpy(), np.zeros_like(x.asnumpy()))
    np.testing.assert_allclose(grad.asnumpy(), np.zeros_like(x.asnumpy()))


def test_broadcast_add():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = 'float32'
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
    full_func = relay.ir_pass.infer_type(gradient(func))
    assert full_func.checked_type == relay.FuncType([t1, t2],
                                                    relay.TupleType([relay.TensorType(expected_forward.shape, dtype),
                                                                     relay.TupleType([t1, t2])]))
    ex = create_executor()
    forward, (grad_x, grad_y) = ex.evaluate(full_func)(x_nd, y_nd)
    np.testing.assert_allclose(forward.asnumpy(), expected_forward)
    np.testing.assert_allclose(grad_x.asnumpy(),
                               np.ones_like(expected_forward).sum(axis=2, keepdims=True))
    np.testing.assert_allclose(grad_y.asnumpy(),
                               np.ones_like(expected_forward).sum(axis=(0, 1), keepdims=True).squeeze(axis=0))


def test_broadcast_subtract():
    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = 'float32'
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
    full_func = relay.ir_pass.infer_type(gradient(func))
    assert full_func.checked_type == relay.FuncType([t1, t2],
                                                    relay.TupleType([relay.TensorType(expected_forward.shape, dtype),
                                                                     relay.TupleType([t1, t2])]))
    ex = create_executor()
    forward, (grad_x, grad_y) = ex.evaluate(full_func)(x_nd, y_nd)
    np.testing.assert_allclose(forward.asnumpy(), expected_forward)
    np.testing.assert_allclose(grad_x.asnumpy(),
                               np.ones_like(expected_forward).sum(axis=2, keepdims=True))
    np.testing.assert_allclose(grad_y.asnumpy(),
                               -np.ones_like(expected_forward).sum(axis=(0, 1), keepdims=True).squeeze(axis=0))


if __name__ == "__main__":
    test_id()
    test_add()
    test_temp_add()
    test_sub()
    test_broadcast_add()
    test_broadcast_subtract()
