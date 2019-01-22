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


if __name__ == "__main__":
    test_id()
    test_add()
    test_temp_add()
    test_sub()
