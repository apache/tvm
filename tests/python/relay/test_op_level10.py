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


def verify_slice_like(data, slice_like, axes, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    y = relay.var("slice_like", relay.TensorType(slice_like, dtype))
    z = relay.slice_like(x, y, axes)
    zz = relay.ir_pass.infer_type(z)
    if axes:
        assert "axes" in z.astext()
    assert zz.checked_type == relay.ty.TensorType(output, dtype)

def test_slice_like():
    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    verify_slice_like(data=(d1, d2, d3), slice_like=(1, 2, 3), axes=[], output=(1, 2, 3))
    verify_slice_like(data=(1, 2, 3), slice_like=(d1, d2, d3), axes=[], output=(d1, d2, d3))
    verify_slice_like(data=(d2, d3, d4), slice_like=(d1, d2, d3), axes=(1,2), output=(d2, d2, d3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=[], output=(1, 2, 3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2), axes=[], output=(1, 2, 5))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(1, 2), output=(3, 2, 3))
    verify_slice_like(data=(3, 4, 5), slice_like=(1, 2, 3), axes=(-1, -3), output=(1, 4, 3))
    verify_slice_like(data=(1, 3, 224, 224),
                      slice_like=(1, 3, 112, 112),
                      axes=(2, 3),
                      output=(1, 3, 112, 112))


if __name__ == "__main__":
    test_collapse_sum_like()
    test_broadcast_to_like()
    test_slice_like()
