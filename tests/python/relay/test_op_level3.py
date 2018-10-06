import tvm
from tvm import relay


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


def test_copy_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.copy(x))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
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
    ftype = func.checked_type()
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
    ftype = func.checked_type()
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 2000), "float32")


if __name__ == "__main__":
    test_unary_identity()
    test_copy_infer_type()
    test_transpose_infer_type()
    test_reshape_infer_type()
