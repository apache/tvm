import numpy as np
import tvm
from tvm import relay
from tvm.relay.interpreter import Value, TupleValue, Interpreter
from tvm.relay import op
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import testing


def check_eval(expr, args, expected_result, env=None, rtol=1e-07):
    if env is None:
        env = relay.env.Environment({})
    intrp = Interpreter(env=env)
    result = intrp.evaluate(expr)(*args)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def test_from_scalar():
    np.testing.assert_allclose(Value.from_scalar(1, 'int32').asnumpy(), 1)
    np.testing.assert_allclose(Value.from_scalar(10.0, 'float32').asnumpy(), 10.0)
    np.testing.assert_allclose(Value.from_scalar(True).asnumpy(), True)


def test_tuple_value():
    tv = TupleValue(Value.from_scalar(
        1), Value.from_scalar(2), Value.from_scalar(3))
    np.testing.assert_allclose(tv[0].asnumpy(), 1)
    np.testing.assert_allclose(tv[1].asnumpy(), 2)
    np.testing.assert_allclose(tv[2].asnumpy(), 3)


def test_id():
    x = relay.var('x', 'float32')
    ident = relay.Function([x], x)
    check_eval(ident, [1.0], 1.0)


def test_add_const():
    two = op.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    check_eval(func, [], 2)


def test_mul_param():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(1, 10))
    func = relay.Function([x, y], op.multiply(x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(1, 10).astype('float32')
    check_eval(func, [x_data, y_data], x_data * y_data)


# failing due to numeric issues

# def test_dense():
#     x = relay.var('x', shape=(10, 10))
#     w = relay.var('w', shape=(10, 10))
#     y = op.nn.dense(x, w)
#     func = relay.Function([x, w], y)
#     x_data = np.random.rand(10, 10).astype('float32')
#     w_data = np.random.rand(10, 10).astype('float32')
#     check_eval(func, [x_data, w_data], x_data @ w_data, rtol=0.1)

# def test_linear():
#     x = relay.var('x', shape=(10, 10))
#     w = relay.var('w', shape=(10, 10))
#     b = relay.var('b', shape=(10,))
#     y = op.add(op.nn.dense(x, w), b)
#     func = relay.Function([x, w, b], y)
#     x_data = np.random.rand(10, 10).astype('float32')
#     w_data = np.random.rand(10, 10).astype('float32')
#     b_data = np.random.rand(10).astype('float32')
#     check_eval(func, [x_data, w_data, b_data], x_data @ w_data + b_data)

def test_equal():
    i = relay.var('i', shape=[], dtype='int32')
    j = relay.var('i', shape=[], dtype='int32')
    z = op.equal(i, j)
    func = relay.Function([i, j], z, ret_type=relay.TensorType([], 'bool'))
    i_data = relay.const(0)
    j_data = relay.const(0)
    check_eval(func, [i_data, j_data], True)

def test_subtract():
    i = relay.var('i', shape=[], dtype='int32')
    sub = op.subtract(i, relay.const(1, dtype='int32'))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], 'int32'))
    i_data = np.array(1, dtype='int32')
    check_eval(func, [i_data], 0)

def test_simple_loop():
    env = relay.env.Environment({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(op.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = op.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(op.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    env[sum_up] = func
    i_data = np.array(10, dtype='int32')
    check_eval(sum_up, [i_data], sum(range(1, 11)), env=env)

def test_loop():
    env = relay.env.Environment({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(op.equal(i, relay.const(0))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = op.subtract(i, relay.const(1))
        new_accum = op.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    env[sum_up] = func
    i_data = np.array(10, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    check_eval(sum_up, [i_data, accum_data], sum(range(1, 11)), env=env)

def test_mlp():
    pass
    # net = testing.mlp.get_workload(1)
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_id()
    test_add_const()
    # test_dense()
    # test_linear()
    test_equal()
    test_subtract()
    test_simple_loop()
    test_loop()
    test_mlp()

