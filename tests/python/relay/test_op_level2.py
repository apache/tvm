import tvm
from tvm import relay


def test_conv2d_infer_type():
    # symbolic in batch dimension
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))
    w = ib.param("w", relay.ty.IncompleteType())

    with ib.function(x, w) as func:
        ib.ret(relay.nn.conv2d(x.var, w.var,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               channels=2))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
    assert ftype.ret_type == relay.ty.TensorType(
        (n, 2, 224, 224), "float32")
    assert ftype.arg_types[1] == relay.ty.TensorType(
        (2, 10, 3, 3), "float32")

    # infer by shape of w, mixed precision
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "int8"))
    w = ib.param("w", relay.ty.TensorType((2, 10, 3, 3), "int8"))
    with ib.function(x, w) as func:
        ib.ret(relay.nn.conv2d(x.var, w.var, out_dtype="int32"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
    assert ftype.ret_type == relay.ty.TensorType(
        (n, 2, 222, 222), "int32")

    # Infer with a different layout
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = 4, 32, 224, 224
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "int8"))
    w = ib.param("w", relay.ty.IncompleteType())
    with ib.function(x, w) as func:
        ib.ret(relay.nn.conv2d(x.var, w.var,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               channels=16,
                               data_layout="NCHW4n4c",
                               weight_layout="OIHW4o4i",
                               out_dtype="int32"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
    assert ftype.ret_type == relay.ty.TensorType(
        (1, 4, 224, 224, 4, 4), "int32")
    assert ftype.arg_types[1] == relay.ty.TensorType(
        (4, 8, 3, 3, 4, 4), "int8")



if __name__ == "__main__":
    test_conv2d_infer_type()
