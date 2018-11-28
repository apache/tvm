""" Support level5 operator test cases.
"""
import numpy as np
import tvm
from tvm import relay
from tvm.relay.testing import ctx_list
import topi.testing

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

def test_resize():
    def verify_resize(dshape, scale, method, layout):
        if layout == "NHWC":
            size = (dshape[1] * scale, dshape[2] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale)

        x_data = np.random.uniform(size=dshape).astype("float32")
        if method == "BILINEAR":
            ref_res = topi.testing.bilinear_resize_python(x_data, size, layout)
        else:
            ref_res = topi.testing.upsampling_python(x_data, scale, layout)
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize(x, size, layout, method, False)
        assert "size=" in z.astext()
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    for method in ["BILINEAR", "NEAREST_NEIGHBOR"]:
        for layout in ["NHWC", "NCHW"]:
            verify_resize((1, 4, 4, 4), 2, method, layout)

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
    test_resize()
    test_multibox_prior()
    test_nms()
