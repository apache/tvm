""" Support level3 operator test cases.
"""
import tvm
import numpy as np
from tvm import relay
from tvm.relay import create_executor
from nose.tools import raises

def test_zeros_ones():
    for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
        y = op(shape=(124, 50), dtype="float64")
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType((124, 50), "float64")
        intrp = create_executor()
        intrp_res = intrp.evaluate(y).asnumpy()
        np.testing.assert_allclose(intrp_res, ref((124, 50), 'float64'))

def test_unary_identity():
    for op, ref in [(relay.zeros_like, np.zeros_like),
               (relay.ones_like, np.ones_like),
               (relay.ceil, np.ceil),
               (relay.floor, np.floor),
               (relay.trunc, np.trunc),
               (relay.round, np.round),
               (relay.abs, np.abs),
               (relay.copy, None), # np.copy
               (relay.negative, np.negative)]:
        shape = (8, 9, 4)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = op(x)
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType(shape, "float32")

        if ref is not None:
            data = np.random.rand(*shape).astype('float32')
            intrp = create_executor()
            op_res = intrp.evaluate(y, { x: relay.const(data) })
            ref_res = ref(data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

def test_cast():
    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = x.astype("int32")
    yy = relay.ir_pass.infer_type(y)
    assert "dtype=" in yy.astext()
    assert yy.checked_type == relay.TensorType((8, 9, 4), "int32")


def test_clip():
    a = relay.var("a", relay.TensorType((10, 4), "float32"))
    y = relay.clip(a, 1., 4.)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "float32")

    data = np.random.rand(10, 4).astype('float32')
    intrp = create_executor()
    op_res = intrp.evaluate(y, { a: relay.const(data) })
    ref_res = np.clip(data, 1., 4.)
    np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)




def test_transpose_infer_type():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.transpose(x, axes=(1, 0, 2))
    "axes=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (t, n, 100), "float32")


def test_squeeze_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(2,))
    assert "axis=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (1, 4), "float32")

    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x)
    assert "axis=" not in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (4,), "float32")


@raises(tvm._ffi.base.TVMError)
def test_squeeze_bad_axes_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(1,))
    yy = relay.ir_pass.infer_type(y)


def test_reshape_infer_type():
    n, t, d1, d2 = tvm.var("n"), tvm.var("t"), 100, 20
    x = relay.var("x", relay.TensorType((n, t, d1, d2), "float32"))
    y = relay.reshape(x, newshape=(n, t, 2000))
    assert "newshape=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (n, t, 2000), "float32")


def test_reshape_like():
    # concrete shape
    x = relay.var("x", relay.TensorType((1, 2, 3), "float32"))
    y = relay.var("y", relay.TensorType((1,6), "float32"))
    z = relay.reshape_like(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 6), "float32")

    # symbolic shape
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.var("y", relay.TensorType((1, 8, 8), "float32"))
    z = relay.reshape_like(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 8, 8), "float32")


def test_take_infer_type():
    def verify_take(dshape, indices_shape, oshape, axis=None):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        y = relay.take(x, indices, axis=axis)
        y.astext()
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType(oshape, "float32")

    d1, d2, d3 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3")
    d4, d5, d6 = tvm.var("d4"), tvm.var("d5"), tvm.var("d6")
    verify_take((d1,), (1,), (1,), 0)
    verify_take((4,), (d1, d2), (d1, d2))
    verify_take((3, 3, 3), (1, d2), (1, d2))
    verify_take((d1, d2), (d3, d4, d5), (d3, d4, d5, d2), 0)
    verify_take((d1, d2), (d3, d4, d5), (d1, d3, d4, d5), 1)
    verify_take((d1, d2, d3, d4), (d5, d6), (d1, d2, d5, d6, d4), -2)

def test_split_infer_type():
    def verify_split(dshape, indices_or_sections, ret_type, axis=None):
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)
        y.astext()
        yy = relay.ir_pass.infer_type(y.astuple())
        assert yy.checked_type == ret_type

    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    axis = tvm.var("axis")
    verify_split((5, 5, 2, 2), 5,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32")])),
                  axis=1)
    verify_split((d1, d2, d3, d4), 4,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((d1, d2, d3/4, d4), "float32"),
                     relay.ty.TensorType((d1, d2, d3/4, d4), "float32"),
                     relay.ty.TensorType((d1, d2, d3/4, d4), "float32"),
                     relay.ty.TensorType((d1, d2, d3/4, d4), "float32")])),
                  axis=2)
    verify_split((d1, d2, d3, d4), (2, 4, 7),
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                     relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                     relay.ty.TensorType((d1, 3, d3, d4), "float32"),
                     relay.ty.TensorType((d1, (d2-7), d3, d4), "float32")])),
                  axis=1)

def test_full():
    # default settings: match input dtype
    x = relay.var("x", relay.TensorType((), "int8"))
    y = relay.full(x, ())
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((), "int8")

    # change the shape and dtype
    x = relay.var("x", relay.TensorType((), "float32"))
    y = relay.full(x, (1, 2), "int8")
    "shape=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 2), "int8")


def test_full_like():
    # concrete shape
    base = relay.var("base", relay.TensorType((1, 2, 3), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 2, 3), "float32")

    # symbolic shape
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    base = relay.var("base", relay.TensorType((n, c, h, w), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), "float32")

def test_infer_type_leaky_relu():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.leaky_relu(x, alpha=0.1)
    "alpha=0.1" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), "float32")

def verify_infer_type_prelu(data, alpha, axis, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    if alpha:
        y = relay.var("alpha", relay.TensorType(alpha, dtype))
    else:
        y = relay.var("alpha", relay.IncompleteType())
    z = relay.nn.prelu(x, y, axis=axis)
    zz = relay.ir_pass.infer_type(z)
    if axis != 1:
        assert "axis" in z.astext()
    assert zz.checked_type == relay.ty.TensorType(output, dtype)
    if not alpha:
        axis = axis if axis else 1
        alpha_shape = (data[axis],)
        assert zz.args[1].checked_type == relay.TensorType(alpha_shape, "float32")

def test_infer_type_prelu():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    verify_infer_type_prelu((n, c, h, w), (c,), 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), (c,), 3, (n, h, w, c))
    verify_infer_type_prelu((n, c, h, w), None, 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), None, 3, (n, h, w, c))
    verify_infer_type_prelu((1, 3, 2, 2), (3,), 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), (3,), 3, (1, 2, 2, 3))
    verify_infer_type_prelu((1, 3, 2, 2), None, 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), None, 3, (1, 2, 2, 3))

if __name__ == "__main__":
    test_cast()
    test_zeros_ones()
    test_unary_identity()
    test_clip()
    test_transpose_infer_type()
    test_reshape_infer_type()
    test_reshape_like()
    test_take_infer_type()
    test_full()
    test_full_like()
    test_infer_type_leaky_relu()
    test_infer_type_prelu()
    test_squeeze_infer_type()
    test_squeeze_bad_axes_infer_type()
    test_split_infer_type()
