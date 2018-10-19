""" Support level10 operator test cases.
"""
import tvm
from tvm import relay

def test_collapse_sum_like():
    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((3, 4, 5, 6), "int8"))
    y = ib.param("y", relay.ty.TensorType((4, 1, 6), "int8"))
    with ib.function(x, y) as func:
        ib.ret(relay.collapse_sum_like(x, y))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((4, 1, 6), "int8")


def test_broadcast_to_like():
    ib = relay.ir_builder.IRBuilder()
    x = ib.param("x", relay.ty.TensorType((3, 4, 5, 6), "int8"))
    y = ib.param("x", relay.ty.TensorType((4, 1, 6), "int8"))
    with ib.function(x, y) as func:
        ib.ret(relay.broadcast_to_like(y, x))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type
    assert ftype.ret_type == relay.ty.TensorType((3, 4, 5, 6), "int8")

if __name__ == "__main__":
    test_collapse_sum_like()
    test_broadcast_to_like()
