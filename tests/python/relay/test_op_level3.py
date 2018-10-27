""" Support level3 operator test cases.
"""
import tvm
import numpy as np
from tvm import relay
from nose.tools import raises

def test_zeros_ones():
    for op in [relay.zeros, relay.ones]:
        y = op(shape=(124, 50), dtype="float64")
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType((124, 50), "float64")

def test_unary_identity():
    for op in [relay.zeros_like,
               relay.ones_like,
               relay.ceil,
               relay.floor,
               relay.trunc,
               relay.round,
               relay.abs,
               relay.copy,
               relay.negative]:
        x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
        y = op(x)
        yy = relay.ir_pass.infer_type(y)
        assert yy.checked_type == relay.TensorType((8, 9, 4), "float32")


def test_cast():
    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = x.astype("int32")
    yy = relay.ir_pass.infer_type(y)
    assert "dtype=" in yy.astext()
    assert yy.checked_type == relay.TensorType((8, 9, 4), "int32")


def test_clip_type():
    a = relay.var("a", relay.TensorType((10, 4), "float32"))
    y = relay.clip(a, 1., 4.)
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "float32")


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
    y = relay.squeeze(x, axes=(2,))
    assert "axes=" in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (1, 4), "float32")

    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x)
    assert "axes=" not in y.astext()
    yy = relay.ir_pass.infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (4,), "float32")


@raises(tvm._ffi.base.TVMError)
def test_squeeze_bad_axes_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axes=(1,))
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


if __name__ == "__main__":
    test_cast()
    test_zeros_ones()
    test_unary_identity()
    test_clip_type()
    test_transpose_infer_type()
    test_reshape_infer_type()
    test_reshape_like()
    test_take_infer_type()
    test_full()
    test_full_like()
    test_infer_type_leaky_relu()
    test_squeeze_infer_type()
    test_squeeze_bad_axes_infer_type()
    test_split_infer_type()
