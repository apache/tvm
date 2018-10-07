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
        ib.ret(relay.image.resize(x.var, (th, tw)))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, th, tw), "int8")

    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((n, c, h, w), "int8"))
    with ib.function(x) as func:
        ib.ret(relay.image.resize(x.var, (100, 200), "NCHW", "BILINEAR", False))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((n, c, 100, 200), "int8")

if __name__ == "__main__":
    test_resize_infer_type()
