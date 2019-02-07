import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend.interpreter import Value, TupleValue
from tvm.relay.scope_builder import ScopeBuilder
from tvm.relay import testing, create_executor


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    # TODO(tqchen) add more types once the schedule register is fixed.
    for target in ["llvm"]:
        ctx = tvm.context(target, 0)
        if not ctx.exist:
            return
        intrp = create_executor(mod=mod, ctx=ctx, target=target)
        result = intrp.evaluate(expr)(*args)
        # use tvm.testing which also set atol
        tvm.testing.assert_allclose(
            result.asnumpy(), expected_result, rtol=rtol)


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
    one = np.array(1.0, 'float32')
    check_eval(ident, [one], one)


def test_add_const():
    two = relay.add(relay.const(1), relay.const(1))
    func = relay.Function([], two)
    check_eval(func, [], 2)


def test_mul_param():
    x = relay.var('x', shape=(10, 10))
    y = relay.var('y', shape=(1, 10))
    func = relay.Function([x, y], relay.multiply(x, y))
    x_data = np.random.rand(10, 10).astype('float32')
    y_data = np.random.rand(1, 10).astype('float32')
    check_eval(func, [x_data, y_data], x_data * y_data)


def test_equal():
    i = relay.var('i', shape=[], dtype='int32')
    j = relay.var('i', shape=[], dtype='int32')
    z = relay.equal(i, j)
    func = relay.Function([i, j], z, ret_type=relay.TensorType([], 'bool'))
    i_data = relay.const(0, 'int32')
    j_data = relay.const(0, 'int32')
    check_eval(func, [i_data, j_data], True)


def test_subtract():
    i = relay.var('i', shape=[], dtype='int32')
    sub = relay.subtract(i, relay.const(1, dtype='int32'))
    func = relay.Function([i], sub, ret_type=relay.TensorType([], 'int32'))
    i_data = np.array(1, dtype='int32')
    check_eval(func, [i_data], 0)


def test_simple_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, dtype='int32'))):
        sb.ret(i)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, dtype='int32'))
        rec_call = relay.Call(sum_up, [one_less])
        sb.ret(relay.add(rec_call, i))
    func = relay.Function([i], sb.get(), ret_type=relay.TensorType([], 'int32'))
    mod[sum_up] = func
    i_data = np.array(10, dtype='int32')
    check_eval(sum_up, [i_data], sum(range(1, 11)), mod=mod)


def test_loop():
    mod = relay.module.Module({})
    sum_up = relay.GlobalVar('sum_up')
    i = relay.var('i', shape=[], dtype='int32')
    accum = relay.var('accum', shape=[], dtype='int32')
    sb = ScopeBuilder()
    with sb.if_scope(relay.equal(i, relay.const(0, 'int32'))):
        sb.ret(accum)
    with sb.else_scope():
        one_less = relay.subtract(i, relay.const(1, 'int32'))
        new_accum = relay.add(accum, i)
        sb.ret(relay.Call(sum_up, [one_less, new_accum]))
    func = relay.Function([i, accum], sb.get())
    mod[sum_up] = func
    i_data = np.array(10, dtype='int32')
    accum_data = np.array(0, dtype='int32')
    check_eval(sum_up, [i_data, accum_data], sum(range(1, 11)), mod=mod)


def test_ref():
    mod = relay.Module()
    three_with_ref = relay.GlobalVar('three_with_ref')
    i = relay.Var('i')
    iv = relay.Var('iv')
    u = relay.Var('u')
    uv = relay.Var('uv')
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
    x_data = np.random.rand(1, 10).astype('float32')
    y_data = np.random.rand(1, 10).astype('float32')
    z_data = np.random.rand(1, 10).astype('float32')
    params = { 'y': y_data, 'z': z_data }
    intrp = create_executor("debug")
    res = intrp.evaluate(f)(x_data, **params).data
    tvm.testing.assert_allclose(res.asnumpy(), x_data + y_data + z_data)


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
