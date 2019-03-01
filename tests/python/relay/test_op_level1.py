import math
import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing import ctx_list
import topi.testing

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
                        (relay.nn.relu, relu)]:
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


def test_expand_dims():
    # based on topi test
    def verify_expand_dims(dshape, dtype, oshape, axis, num_newaxis):
        x = relay.Var("x", relay.TensorType(dshape, dtype))
        func = relay.Function([x], relay.expand_dims(x, axis, num_newaxis))
        for target, ctx in ctx_list():
            data = np.random.uniform(size=dshape).astype(dtype)
            ref_res = data.reshape(oshape)
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

    verify_expand_dims((3, 10), 'float32', (3, 10, 1, 1), 2, 2)
    verify_expand_dims((3, 10), 'float32', (1, 3, 10), -3, 1)


def test_bias_add():
    xshape=(10, 2, 3, 4)
    bshape=(2,)
    dtype="float32"
    x = relay.var("x", shape=xshape)
    bias = relay.var("bias")
    z = relay.nn.bias_add(x, bias)
    zz = relay.ir_pass.infer_type(z)
    assert "axis=" not in zz.astext()
    assert zz.args[1].checked_type == relay.TensorType(bshape)

    func = relay.Function([x, bias], z)
    x_data = np.random.uniform(size=xshape).astype(dtype)
    y_data = np.random.uniform(size=bshape).astype(dtype)
    ref_res = x_data + y_data.reshape((2, 1, 1))
    for target, ctx in ctx_list():
        intrp = relay.create_executor("graph", ctx=ctx, target=target)
        op_res = intrp.evaluate(func)(x_data, y_data)
        np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


def test_expand_dims_infer_type():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", shape=(n, t, d))
    y = relay.expand_dims(x, axis=2)
    assert "axis=2" in y.astext()
    checked = relay.ir_pass.infer_type(y)
    assert checked.checked_type == relay.TensorType((n, t, 1, 100))


def test_softmax():
    shape = (10, 4)
    x = relay.var("x", shape=shape)
    y = relay.nn.softmax(x, axis=1)
    assert "nn.softmax" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(shape)
    func = relay.Function([x], y)
    x_data = np.random.uniform(size=shape).astype("float32")
    ref_res = topi.testing.softmax_python(x_data)
    for target, ctx in ctx_list():
        intrp = relay.create_executor("graph", ctx=ctx, target=target)
        op_res = intrp.evaluate(func)(x_data)
        np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


def test_log_softmax():
    shape = (10, 4)
    x = relay.var("x", shape=shape)
    y = relay.nn.log_softmax(x, axis=1)
    assert "nn.log_softmax" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(shape)
    func = relay.Function([x], y)
    x_data = np.random.uniform(size=shape).astype("float32")
    ref_res = topi.testing.log_softmax_python(x_data)
    for target, ctx in ctx_list():
        intrp = relay.create_executor("graph", ctx=ctx, target=target)
        op_res = intrp.evaluate(func)(x_data)
        np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


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
    t = relay.var("z", shape=())
    z = relay.concatenate((x, y), axis=1)
    z = relay.add(z, t)
    # Check result.
    func = relay.Function([x, y, t], z)
    x_data = np.random.rand(10, 5).astype('float32')
    y_data = np.random.rand(10, 5).astype('float32')
    t_data = np.random.uniform(size=()).astype('float32')
    ref_res = np.concatenate((x_data, y_data), axis=1) + t_data

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, y_data, t_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=0.01)
        op_res2 = intrp2.evaluate(func)(x_data, y_data, t_data)
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


def test_dense():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.TensorType((2, w), "float32"))
    y = relay.nn.dense(x, w, units=2)
    "units=2" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, 2), "float32")

    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), 2
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    wh, ww = tvm.var("wh"), tvm.var("ww")
    w = relay.var("w", relay.TensorType((ww, wh), "float32"))
    y = relay.nn.dense(x, w)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, ww), "float32")

    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), 2
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    w = relay.var("w", relay.IncompleteType())
    y = relay.nn.dense(x, w, units=2)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, 2), "float32")

    x = relay.var("x", shape=(10, 5))
    w = relay.var("w", shape=(2, 5))
    z = relay.nn.dense(x, w)

    # Check result.
    func = relay.Function([x, w], z)
    x_data = np.random.rand(10, 5).astype('float32')
    w_data = np.random.rand(2, 5).astype('float32')
    ref_res = np.dot(x_data, w_data.T)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, w_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data, w_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)


if __name__ == "__main__":
    test_concatenate()
    test_bias_add()
    test_unary_op()
    test_binary_op()
    test_expand_dims_infer_type()
    test_expand_dims()
    test_softmax()
    test_log_softmax()
    test_dropout()
    test_batch_norm()
    test_dense()
