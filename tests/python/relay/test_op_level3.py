""" Support level3 operator test cases.
"""
import tvm
import numpy as np
from tvm import relay
from tvm.relay.ir_pass import infer_type
from tvm.relay.ir_builder import IRBuilder, func_type
from tvm.relay.env import Environment


def test_unary_identity():
    for op in [relay.zeros_like, relay.ones_like]:
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((8, 9, 4), "int32"))
        with ib.function(x) as func:
            ib.ret(op(x.var))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((8, 9, 4), "int32")


def test_clip_type():
    ib = relay.ir_builder.IRBuilder()
    a = ib.param("a", relay.TensorType((10, 4), "float32"))
    with ib.function(a) as func:
        ib.ret(relay.clip(a.var, 1., 4.))
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


if __name__ == "__main__":
    test_single_op()
    test_unary_identity()
    test_clip_type()
    test_copy_infer_type()
    test_transpose_infer_type()
    test_reshape_infer_type()
