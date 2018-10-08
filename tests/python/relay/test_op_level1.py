import tvm
import numpy as np
from tvm import relay
from tvm.relay.ir_pass import infer_type
from tvm.relay.ir_builder import IRBuilder, func_type
from tvm.relay.ir_builder import scalar_type, convert, tensor_type
from tvm.relay.env import Environment

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

    for opfunc in [tvm.relay.log, tvm.relay.exp, tvm.relay.sqrt,
                   tvm.relay.sigmoid, tvm.relay.tanh]:
        check_single_op(opfunc)

def test_expand_dims_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    # let's mimic a batch of sequences
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.expand_dims(x, axis=2))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 1, 100), "float32")


def test_softmax():
    ib = relay.ir_builder.IRBuilder()
    n, d = tvm.var("n"), tvm.var("d")
    x = ib.param("x", relay.ty.TensorType((n, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.softmax(x, axis=1))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, d), "float32")


def test_log_softmax():
    ib = relay.ir_builder.IRBuilder()
    n, d = tvm.var("n"), tvm.var("d")
    x = ib.param("x", relay.ty.TensorType((n, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.log_softmax(x, axis=1))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, d), "float32")

def test_unary_op():
    for op in [relay.exp,
               relay.log,
               relay.sqrt,
               relay.sigmoid]:
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((10, 4), "int32"))
        with ib.function(x) as func:
            ib.ret(op(x.var))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((10, 4), "int32")

def test_binary_op():
    def check_binary_op(opfunc):
        """
        Program:
            fn (x, y) {
                return x <op> y;
            }
        """
        b = IRBuilder()

        x = b.param('x', tensor_type(5, 5, 5))
        y = b.param('y', tensor_type(5, 5, 5))
        with b.function(x, y) as func:
            b.ret(opfunc(x.var, y.var))
        b.ret(func)
        prog, env = b.get()
        ttype = tensor_type(5, 5, 5)
        expected_ty = func_type([ttype, ttype], ttype)
        assert_has_type(func.to_func(), expected_ty)

    for opfunc in [relay.add, relay.subtract, relay.mod,
                   relay.multiply, relay.divide]:
        check_binary_op(opfunc)


def test_binary_broadcast_op():
    def check_binary_broadcast_op(opfunc):
        """
        Program:
            fn (x: Tensor[(10, 4), f32], y: Tensor[(5, 10, 1), f32]) -> Tensor[(5, 10, 4), f32] {
                return x <op> y;
            }
        """
        b = IRBuilder()
        x = b.param('x', tensor_type(10, 4))
        y = b.param('y', tensor_type(5, 10, 1))
        with b.function(x, y) as func:
            b.ret(opfunc(x.var, y.var))
        b.ret(func)
        prog, env = b.get()

        expected_ty = func_type([tensor_type(10, 4), tensor_type(5, 10, 1)],
                                tensor_type(5, 10, 4))
        assert_has_type(func.to_func(), expected_ty)

    for opfunc in [relay.add, relay.subtract, relay.mod,
                   relay.multiply, relay.divide]:
        check_binary_broadcast_op(opfunc)


def test_concatenate_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    y = ib.param("y", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x, y) as func:
        ib.ret(relay.concatenate((x, y), axis=-1))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 200), "float32")

    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    y = ib.param("y", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x, y) as func:
        ib.ret(relay.concatenate((x, y), axis=2))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 200), "float32")

    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    y = ib.param("y", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x, y) as func:
        ib.ret(relay.concatenate((x, y), axis=1))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t + t, 100), "float32")


if __name__ == "__main__":
    test_unary_op()
    test_single_op()
    test_expand_dims_infer_type()
    test_concatenate_infer_type()
    test_softmax()
    test_log_softmax()
    test_binary_op()
    test_binary_broadcast_op()
