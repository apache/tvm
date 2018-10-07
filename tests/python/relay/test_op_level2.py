""" Support level2 operator test cases.
"""
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
    ftype = func.checked_type
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
    ftype = func.checked_type
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
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (1, 4, 224, 224, 4, 4), "int32")
    assert ftype.arg_types[1] == relay.ty.TensorType(
        (4, 8, 3, 3, 4, 4), "int8")

def test_upsampling_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.upsampling(x.var, scale=2, layout="NCHW", method="BILINEAR"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, h*2, w*2), "float32")

    ib = relay.ir_builder.IRBuilder()
    n, c = tvm.var("n"), tvm.var("c")
    x = ib.param("x", relay.ty.TensorType((n, c, 100, 200), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.upsampling(x.var, scale=2, layout="NCHW", method="BILINEAR"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, 200, 400), "float32")

def _test_pool2d_infer_type(opfunc):
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(opfunc(x.var, pool_size=(1, 1)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, 10, 224, 224), "float32")

    ph, pw = tvm.var("ph"), tvm.var("pw")
    sh, sw = tvm.var("sh"), tvm.var("sw")

    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 10, 224, 224
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(opfunc(x.var, pool_size=(ph, pw), strides=(sh, sw)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (n, 10, (((224 - ph)/sh) + 1), (((224 - pw)/sw) + 1)), "float32")

def _test_global_pool2d_infer_type(opfunc):
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), tvm.var("c"), 224, 224
    x = ib.param("x", relay.ty.TensorType((n, h, w, c), "float32"))
    with ib.function(x) as func:
        ib.ret(opfunc(x.var, layout="NHWC"))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, 1, 1, c), "float32")

    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))
    with ib.function(x) as func:
        ib.ret(opfunc(x.var))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, 1, 1), "float32")

def test_pool2d_infer_type():
    _test_pool2d_infer_type(relay.nn.max_pool2d)
    _test_pool2d_infer_type(relay.nn.avg_pool2d)
    _test_global_pool2d_infer_type(relay.nn.global_avg_pool2d)
    _test_global_pool2d_infer_type(relay.nn.global_avg_pool2d)

def test_flatten_infer_type():
    ib = relay.ir_builder.IRBuilder()
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    x = ib.param("x", relay.ty.TensorType((d1, d2, d3, d4), "float32"))

    with ib.function(x) as func:
        ib.ret(relay.nn.batch_flatten(x.var))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((d1, ((d2*d3)*d4)), "float32")

    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((3, 2, 4, 3), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.batch_flatten(x.var))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((3, 24), "float32")

    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((d1, 2, d3, 3), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.nn.batch_flatten(x.var))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((d1, ((2*d3)*3)), "float32")


if __name__ == "__main__":
    test_conv2d_infer_type()
    test_pool2d_infer_type()
    test_upsampling_infer_type()
    test_flatten_infer_type()
