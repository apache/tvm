""" Support level5 operator test cases.
"""
import tvm
from tvm import relay

def test_resize_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "int8"))
    th, tw = tvm.var("th"), tvm.var("tw")

    with ib.function(x) as func:
        ib.ret(relay.image.resize(x, (th, tw)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, th, tw), "int8")

    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "int8"))
    with ib.function(x) as func:
        ib.ret(relay.image.resize(x, (100, 200), "NCHW", "BILINEAR", False))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, 100, 200), "int8")



def test_multibox_prior():
    sizes = (0.3, 1.5, 0.7)
    ratios = (1.3, 2.4)
    steps = (2.0, 1.5)
    offsets = (0.2, 0.3)
    clip = True

    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 3, 56, 56
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))

    with ib.function(x) as func:
        ib.ret(relay.vision.multibox_prior(x, sizes, ratios,
                                           steps, offsets, clip))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (1, h * w * (len(sizes) + len(ratios) - 1), 4), "float32")

    ib = relay.ir_builder.IRBuilder()
    n, c, h, w = tvm.var("n"), 24, 32, 32
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "float32"))

    with ib.function(x) as func:
        ib.ret(relay.vision.multibox_prior(x))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType(
        (1, h * w, 4), "float32")


if __name__ == "__main__":
    test_resize_infer_type()
    test_multibox_prior()
