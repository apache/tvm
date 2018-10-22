""" Support level10 operator test cases.
"""
import tvm
from tvm import relay

def test_collapse_sum_like():
    x = relay.Var("x", relay.ty.TensorType((3, 4, 5, 6), "int8"))
    y = relay.Var("y", relay.ty.TensorType((4, 1, 6), "int8"))
    z = relay.collapse_sum_like(x, y)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType((4, 1, 6), "int8")


def test_broadcast_to_like():
    x = relay.Var("x", relay.ty.TensorType((3, 4, 5, 6), "int8"))
    y = relay.Var("y", relay.ty.TensorType((4, 1, 6), "int8"))
    z = relay.broadcast_to_like(y, x)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType((3, 4, 5, 6), "int8")

if __name__ == "__main__":
    test_collapse_sum_like()
    test_broadcast_to_like()
