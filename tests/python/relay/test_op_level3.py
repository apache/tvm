""" Support level3 operator test cases.
"""
import tvm
import numpy as np
from tvm import relay
from tvm.relay.ir_pass import infer_type
from tvm.relay.ir_builder import IRBuilder, func_type
from tvm.relay.env import Environment
from nose.tools import raises

def test_zeros_ones():
    for op in [relay.zeros, relay.ones]:
        ib = relay.ir_builder.IRBuilder()
        with ib.function() as func:
            ib.ret(op((124, 50), "float64"))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((124, 50), "float64")


def test_unary_identity():
    for op in [relay.zeros_like, relay.ones_like]:
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((8, 9, 4), "int32"))
        with ib.function(x) as func:
            ib.ret(op(x))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((8, 9, 4), "int32")


def test_clip_type():
    ib = relay.ir_builder.IRBuilder()
    a = ib.param("a", relay.TensorType((10, 4), "float32"))
    with ib.function(a) as func:
        ib.ret(relay.clip(a, 1., 4.))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((10, 4), "float32")


def test_copy_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.copy(x))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 100), "float32")


def test_transpose_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.transpose(x, axes=(1, 0, 2)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (t, n, 100), "float32")


def test_squeeze_default_axes_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = 1, 4, 1
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.squeeze(x))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (4,), "float32")


def test_squeeze_axes_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = 1, 4, 1
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.squeeze(x, axes=(2,)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (1, 4), "float32")


@raises(tvm._ffi.base.TVMError)
def test_squeeze_bad_axes_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = 1, 4, 1
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.squeeze(x, axes=(1,)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type


def test_reshape_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d1, d2 = tvm.var("n"), tvm.var("t"), 100, 20
    x = ib.param("x", relay.ty.TensorType((n, t, d1, d2), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.reshape(x, newshape=(n, t, 2000)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 2000), "float32")


def assert_has_type(expr, typ, env=Environment({})):
    checked_expr = infer_type(env, expr)
    checked_type = checked_expr.checked_type
    if checked_type != typ:
        raise RuntimeError("Type mismatch %s vs %s" % (
            checked_type, typ))

def test_single_op():
    def check_single_op(opfunc):
        "Program: fn (x : float32) { let t1 = f(x); t1 }"
        b = IRBuilder()
        with b.function(('x', 'float32')) as func:
            x, = func.param_ids()
            t1 = b.let('t1', opfunc(x))
            b.ret(t1)
        assert_has_type(func.to_func(), func_type(['float32'], 'float32'))

    for opfunc in [tvm.relay.ceil, tvm.relay.floor, tvm.relay.trunc,
                   tvm.relay.round, tvm.relay.abs, tvm.relay.negative]:
        check_single_op(opfunc)

def test_take_infer_type():
    def verify_take(dshape, indices_shape, oshape, axis=None):
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.ty.TensorType(dshape, "float32"))
        indices = ib.param("indices", relay.ty.TensorType(indices_shape, "int32"))
        with ib.function(x, indices) as func:
            ib.ret(relay.take(x, indices, axis=axis))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.ty.TensorType(oshape, "float32")

    d1, d2, d3 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3")
    d4, d5, d6 = tvm.var("d4"), tvm.var("d5"), tvm.var("d6")
    verify_take((d1,), (1,), (1,), 0)
    verify_take((4,), (d1, d2), (d1, d2))
    verify_take((3, 3, 3), (1, d2), (1, d2))
    verify_take((d1, d2), (d3, d4, d5), (d3, d4, d5, d2), 0)
    verify_take((d1, d2), (d3, d4, d5), (d1, d3, d4, d5), 1)
    verify_take((d1, d2, d3, d4), (d5, d6), (d1, d2, d5, d6, d4), -2)


def test_full():
    # default settings: match input dtype
    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.TensorType((), "int8"))
    with ib.function(x) as func:
        ib.ret(relay.full(x, ()))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((), "int8")

    # change the shape and dtype
    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.TensorType((), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.full(x, (1, 2), "int8"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((1, 2), "int8")


def test_full_like():
    # concrete shape
    ib = relay.ir_builder.IRBuilder()
    base = ib.param("base", relay.TensorType((1, 2, 3), "float32"))
    fill = ib.param("fill", relay.TensorType((), "float32"))
    with ib.function(base, fill) as func:
        ib.ret(relay.full_like(base, fill))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((1, 2, 3), "float32")

    # symbolic shape
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    base = ib.param("base", relay.TensorType((n, c, h, w), "float32"))
    fill = ib.param("fill", relay.TensorType((), "float32"))
    with ib.function(base, fill) as func:
        ib.ret(relay.full_like(base, fill))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((n, c, h, w), "float32")

def test_infer_type_leaky_relu():
   ib = relay.ir_builder.IRBuilder()
   n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
   x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))

   with ib.function(x) as func:
       ib.ret(relay.nn.leaky_relu(x, alpha=0.1))
   ib.ret(func)
   func = relay.ir_pass.infer_type(ib.env, func.to_func())
   ftype = func.checked_type
   assert ftype.ret_type == relay.ty.TensorType((n, c, h, w), "float32")

if __name__ == "__main__":
    test_single_op()
    test_zeros_ones()
    test_unary_identity()
    test_clip_type()
    test_copy_infer_type()
    test_transpose_infer_type()
    test_reshape_infer_type()
    test_take_infer_type()
    test_full()
    test_full_like()
    test_infer_type_leaky_relu()
    test_squeeze_axes_infer_type()
    test_squeeze_default_axes_infer_type()
