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
            b.ret(opfunc(x, y))
        b.ret(func)
        prog, env = b.get()
        ttype = tensor_type(5, 5, 5)
        expected_ty = func_type([ttype, ttype], ttype)
        assert_has_type(func.to_func(), expected_ty)

    for opfunc in [relay.pow]:
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
            b.ret(opfunc(x, y))
        b.ret(func)
        prog, env = b.get()

        expected_ty = func_type([tensor_type(10, 4), tensor_type(5, 10, 1)],
                                tensor_type(5, 10, 4))
        assert_has_type(func.to_func(), expected_ty)

    for opfunc in [relay.pow]:
        check_binary_broadcast_op(opfunc)

def test_cmp_type():
    for op in (relay.greater,
               relay.greater_equal,
               relay.less,
               relay.less_equal,
               relay.equal,
               relay.not_equal):
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((10, 4), "float32"))
        y = ib.param("y", relay.TensorType((5, 10, 1), "float32"))
        with ib.function(x, y) as func:
            ib.ret(op(x, y))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((5, 10, 4), "uint1")

def test_binary_broadcast():
    for op in [relay.right_shift,
               relay.left_shift,
               relay.maximum,
               relay.minimum]:
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((10, 4), "int32"))
        y = ib.param("y", relay.TensorType((5, 10, 1), "int32"))
        with ib.function(x, y) as func:
            ib.ret(op(x, y))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((5, 10, 4), "int32")

def test_argmax():
    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmax(x, axis=(1,)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, h, w), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmax(x, axis=(2,), keepdims=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c , 1, w), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmax(x, axis=(2,), keepdims=True, exclude=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((1, 1 , h, 1), "int32")

def test_argmin():
    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmax(x, axis=(1,)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, h, w), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmin(x, axis=(2,), keepdims=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c , 1, w), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmin(x, axis=(2,), keepdims=True, exclude=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((1, 1 , h, 1), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmin(x, axis=(2,1), keepdims=True, exclude=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((1, c , h, 1), "int32")

    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c , h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.argmin(x, axis=None, keepdims=True, exclude=True))
    ib.ret(func)

    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((1, 1 , 1, 1), "int32")

def test_where():
    ib = relay.ir_builder.IRBuilder()
    cond = ib.param("cond", relay.TensorType((3, 4), "float32"))
    x = ib.param("x", relay.TensorType((3, 4), "float32"))
    y = ib.param("y", relay.TensorType((3, 4), "float32"))
    with ib.function(cond, x, y) as func:
        ib.ret(relay.where(cond, x, y))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.TensorType((3, 4), "float32")


if __name__ == "__main__":
    test_binary_op()
    test_binary_broadcast_op()
    test_cmp_type()
    test_binary_broadcast()
    test_where()
    test_multibox_prior()
    test_argmax()
    test_argmin()
