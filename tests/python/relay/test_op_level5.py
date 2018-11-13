""" Support level5 operator test cases.
"""
import tvm
from tvm import relay

def test_resize_infer_type():
    n, c, h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    th, tw = tvm.var("th"), tvm.var("tw")
    z = relay.image.resize(x, (th, tw))
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, th, tw), "int8")

    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    z= relay.image.resize(x, (100, 200), "NCHW", "BILINEAR", False)
    assert "size=" in z.astext()
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 100, 200), "int8")


def test_multibox_prior():
    sizes = (0.3, 1.5, 0.7)
    ratios = (1.3, 2.4)
    steps = (2.0, 1.5)
    offsets = (0.2, 0.3)
    clip = True

    n, c, h, w = tvm.var("n"), 3, 56, 56
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))

    z = relay.vision.multibox_prior(x, sizes, ratios,
                                    steps, offsets, clip)
    assert "sizes=" in z.astext()
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType(
        (1, h * w * (len(sizes) + len(ratios) - 1), 4), "float32")

    n, c, h, w = tvm.var("n"), 24, 32, 32
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    z = relay.vision.multibox_prior(x)
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.TensorType(
        (1, h * w, 4), "float32")


def test_nms():
    num_anchors = 60

    overlap_threshold = 0.5
    force_suppress = True
    nms_topk = 10

    n = tvm.var("n")
    x0 = relay.var("x0", relay.ty.TensorType((n, num_anchors, 6), "float32"))
    x1 = relay.var("x1", relay.ty.TensorType((n,), "int"))

    z = relay.vision.nms(x0, x1, overlap_threshold, force_suppress, nms_topk)

    assert "overlap_threshold" in z.astext()
    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(
        (n, num_anchors, 6), "float32")

    n = tvm.var("n")
    x0 = relay.var("x0", relay.ty.TensorType((n, num_anchors, 6), "float32"))
    x1 = relay.var("x1", relay.ty.TensorType((n,), "int"))

    z = relay.vision.nms(x0, x1)

    zz = relay.ir_pass.infer_type(z)
    assert zz.checked_type == relay.ty.TensorType(
        (n, num_anchors, 6), "float32")


if __name__ == "__main__":
    test_resize_infer_type()
    test_multibox_prior()
    test_nms()
