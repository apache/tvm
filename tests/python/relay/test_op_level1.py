import tvm
import numpy as np
from tvm import relay


def test_unary_op():
    def check_single_op(opfunc):
        tp = relay.TensorType((10, 4), "float32")
        x = relay.var("x", tp)
        y = opfunc(x)
        # test printer
        assert ("%0 = {}(%x)".format(y.op.name)) in y.astext()
        # test type inference
        assert relay.ir_pass.infer_type(y).checked_type == tp

    for opfunc in [tvm.relay.log,
                   tvm.relay.exp,
                   tvm.relay.sqrt,
                   tvm.relay.sigmoid,
                   tvm.relay.tanh,
                   relay.nn.relu]:
        check_single_op(opfunc)


def test_binary_op():
    def check_binary_op(opfunc):
        n = tvm.var("n")
        t1 = relay.TensorType((5, n, 5))
        t2 = relay.TensorType((n, 1))
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        # test printer
        assert ("%0 = {}(%x, %y)".format(z.op.name)) in z.astext()
        assert relay.ir_pass.infer_type(z).checked_type == t1

    for opfunc in [relay.add,
                   relay.subtract,
                   relay.mod,
                   relay.multiply,
                   relay.divide]:
        check_binary_op(opfunc)


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


def test_concatenate_infer_type():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", shape=(n, t, d))
    y = relay.var("y", shape=(n, t, d))
    z = relay.concatenate((x, y), axis=-1)
    assert "axis=" in z.astext()
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t, 200))

    z = relay.concatenate((x, y), axis=2)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t, 200))

    z = relay.concatenate((x, y), axis=1)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, t + t, 100))


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
    test_concatenate_infer_type()
    test_softmax()
    test_log_softmax()
    test_dropout()
    test_batch_norm()
