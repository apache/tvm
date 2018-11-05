import math
import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing import ctx_list

def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))

def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy

def test_unary_op():
    def check_single_op(opfunc, ref):
        shape = (10, 4)
        dtype = 'float32'
        tp = relay.TensorType(shape, dtype)
        x = relay.var("x", tp)
        y = opfunc(x)
        # test printer
        assert ("%0 = {}(%x)".format(y.op.name)) in y.astext()
        # test type inference
        assert relay.ir_pass.infer_type(y).checked_type == tp

        if ref is not None:
            data = np.random.rand(*shape).astype(dtype)
            ref_res = ref(data)
            func = relay.Function([x], y)
            for target, ctx in ctx_list():
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(data)
                np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)


    for opfunc, ref in [(tvm.relay.log, np.log),
                   (tvm.relay.exp, np.exp),
                   (tvm.relay.sqrt, np.sqrt),
                   (tvm.relay.sigmoid, sigmoid),
                   (tvm.relay.tanh, np.tanh),
                   (relay.nn.relu, None)]: # Just add RELU here after registering.
        check_single_op(opfunc, ref)


def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref):
        # TODO(@jroesch): this piece of code improperly uses type variables.
        n = tvm.var("n")
        s1 = (5, n, 5)
        s2 = (n, 1)
        t1 = relay.TensorType(s1)
        t2 = relay.TensorType(s2)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        # test printer
        assert ("%0 = {}(%x, %y)".format(z.op.name)) in z.astext()
        assert relay.ir_pass.infer_type(z).checked_type == t1

        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            t2 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            y_data = np.random.rand(5, 10, 5).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)
            for target, ctx in ctx_list():
                # use graph by execuor default for testing, as we need
                # create function explicitly to avoid constant-folding.
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

    for opfunc, ref in [(relay.add, np.add),
                   (relay.subtract, np.subtract),
                   (relay.multiply, np.multiply),
                   (relay.divide, np.divide)]:
        check_binary_op(opfunc, ref)


def test_bias_add():
    x = relay.var("x", shape=(10, 2, 3, 4))
    bias = relay.var("bias")
    z = relay.nn.bias_add(x, bias)
    zz = relay.ir_pass.infer_type(z)
    assert "axis=" not in zz.astext()
    assert zz.args[1].checked_type == relay.TensorType((2,))


def test_expand_dims_infer_type():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", shape=(n, t, d))
    y = relay.expand_dims(x, axis=2)
    assert "axis=2" in y.astext()
    checked = relay.ir_pass.infer_type(y)
    assert checked.checked_type == relay.TensorType((n, t, 1, 100))


def test_softmax():
    n, d = tvm.var("n"), tvm.var("d")
    x = relay.var("x", shape=(n, d))
    y = relay.nn.softmax(x, axis=1)
    assert "nn.softmax" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, d))


def test_log_softmax():
    n, d = tvm.var("n"), tvm.var("d")
    x = relay.var("x", shape=(n, d))
    y = relay.nn.log_softmax(x, axis=0)
    assert "nn.log_softmax" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, d))


def test_concatenate():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", shape=(n, t, d))
    y = relay.var("y", shape=(n, t, d))
    z = relay.concatenate((x, y), axis=-1)
    assert "axis=" in z.astext()
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t, 200))

    x = relay.exp(x)
    z = relay.concatenate((x, y), axis=2)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t, 200))

    z = relay.concatenate((x, y), axis=1)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t + t, 100))

    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(10, 5))
    z = relay.concatenate((x, y), axis=1)

    # Check result.
    func = relay.Function([x, y], z)
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    ref_res = np.concatenate((x_data, y_data), axis=1)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, y_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=0.01)
        op_res2 = intrp2.evaluate(func)(x_data, y_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=0.01)

def test_dropout():
    n, t, d = tvm.var("n"), tvm.var("t"), tvm.var("d")
    input_ty = relay.TensorType((n, t, d), "float32")
    x = relay.var("x", input_ty)
    y = relay.nn.dropout(x, rate=0.75)
    assert "rate=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == input_ty


def test_batch_norm():
    # beta and gamma ignored
    data = relay.var("data", relay.TensorType((3, 2, 1)))
    beta = relay.var("beta", relay.TensorType((2,)))
    gamma = relay.var("gamma", relay.TensorType((2,)))
    moving_mean = relay.var("moving_mean", relay.TensorType((2,)))
    moving_var = relay.var("moving_var", relay.TensorType((2,)))
    y = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var,
                            center=False, scale=False)
    yy = relay.ir_pass.infer_type(y.astuple())
    assert "center=" in yy.astext()
    assert yy.checked_type == relay.ty.TupleType(tvm.convert([
        relay.TensorType((3, 2, 1), "float32"),
        relay.TensorType((2,), "float32"),
        relay.TensorType((2,), "float32")
    ]))

    beta = relay.var("beta", relay.TensorType((3,)))
    gamma = relay.var("gamma", relay.TensorType((3,)))
    moving_mean = relay.var("moving_mean", relay.TensorType((3,)))
    moving_var = relay.var("moving_var", relay.TensorType((3,)))

    y = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var,
                            axis=0, center=False, scale=False)
    yy = relay.ir_pass.infer_type(y.astuple())
    assert yy.checked_type == relay.ty.TupleType(tvm.convert([
        relay.ty.TensorType((3, 2, 1), "float32"),
        relay.ty.TensorType((3,), "float32"),
        relay.ty.TensorType((3,), "float32")
    ]))

    # axis=-1
    data = relay.var("data", relay.TensorType((1, 2, 3)))
    beta = relay.var("beta", relay.TensorType((3,)))
    gamma = relay.var("gamma", relay.TensorType((3,)))
    moving_mean = relay.var("moving_mean", relay.TensorType((3,)))
    moving_var = relay.var("moving_var", relay.TensorType((3,)))
    y = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var,
                            axis=-1, center=False, scale=False)
    yy = relay.ir_pass.infer_type(y.astuple())
    assert yy.checked_type == relay.ty.TupleType(tvm.convert([
        relay.ty.TensorType((1, 2, 3), "float32"),
        relay.ty.TensorType((3,), "float32"),
        relay.ty.TensorType((3,), "float32")
    ]))


if __name__ == "__main__":
    test_bias_add()
    test_unary_op()
    test_binary_op()
    test_expand_dims_infer_type()
    test_concatenate()
    test_softmax()
    test_log_softmax()
    test_dropout()
    test_batch_norm()
