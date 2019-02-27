""" Support level5 operator test cases.
"""
import math
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
    def get_ref_result(dshape, sizes=(1.0,),
                       ratios=(1.0,), steps=(-1.0, -1.0),
                       offsets=(0.5, 0.5), clip=True):
        in_height = dshape[2]
        in_width = dshape[3]
        num_sizes = len(sizes)
        num_ratios = len(ratios)
        size_ratio_concat = sizes + ratios
        steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
        steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
        offset_h = offsets[0]
        offset_w = offsets[1]

        oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
        dtype = "float32"
        np_out = np.zeros(oshape).astype(dtype)

        for i in range(in_height):
            center_h = (i + offset_h) * steps_h
            for j in range(in_width):
                center_w = (j + offset_w) * steps_w
                for k in range(num_sizes + num_ratios - 1):
                    w = size_ratio_concat[k] * in_height / in_width / 2.0 if k < num_sizes else \
                        size_ratio_concat[0] * in_height / in_width * math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    h = size_ratio_concat[k] / 2.0 if k < num_sizes else \
                        size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    count = i * in_width * (num_sizes + num_ratios - 1) + j * (num_sizes + num_ratios - 1) + k
                    np_out[0][count][0] = center_w - w
                    np_out[0][count][1] = center_h - h
                    np_out[0][count][2] = center_w + w
                    np_out[0][count][3] = center_h + h
        if clip:
            np_out = np.clip(np_out, 0, 1)

        return np_out

    def verify_multibox_prior(x, dshape, ref_res, sizes=(1.0,),
                              ratios=(1.0,), steps=(-1.0, -1.0),
                              offsets=(0.5, 0.5), clip=True, check_size=False,
                              check_type_only=False):

        z = relay.vision.multibox_prior(x, sizes, ratios, steps, offsets, clip)
        zz = relay.ir_pass.infer_type(z)
        if check_size:
            assert "sizes=" in z.astext()
        assert zz.checked_type == relay.TensorType(
            (1, dshape[2] * dshape[3] * (len(sizes) + len(ratios) - 1), 4),
            "float32")

        if check_type_only:
            return

        data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        func = relay.Function([x], z)
        func = relay.ir_pass.infer_type(func)
        for target, ctx in ctx_list():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

    sizes = (0.3, 1.5, 0.7)
    ratios = (1.3, 2.4)
    steps = (2.0, 1.5)
    offsets = (0.2, 0.3)
    dshape = (1, 3, 56, 56)
    ref_res = get_ref_result(dshape, sizes, ratios, steps, offsets)
    x = relay.var("x", relay.TensorType(dshape, "float32"))
    verify_multibox_prior(x, dshape, ref_res, sizes, ratios, steps, offsets,
                          check_size=True)
    y = relay.var("y", relay.TensorType((tvm.var("n"), 3, 56, 56), "float32"))
    verify_multibox_prior(x, dshape, ref_res, sizes, ratios, steps, offsets,
                          check_size=True, check_type_only=True)

    dshape = (1, 24, 32, 32)
    ref_res = get_ref_result(dshape, clip=False)
    x = relay.var("x", relay.TensorType(dshape, "float32"))
    verify_multibox_prior(x, dshape, ref_res, clip=False)
    y = relay.var("y", relay.TensorType((tvm.var("n"), 24, 32, 32), "float32"))
    verify_multibox_prior(x, dshape, ref_res, clip=False, check_type_only=True)


def test_nms():
    def verify_nms(x0_data, x1_data, dshape, ref_res, valid_count,
                   overlap_threshold=0.5, force_suppress=False, topk=-1,
                   check_type_only=False):
        x0 = relay.var("x0", relay.ty.TensorType(dshape, "float32"))
        x1 = relay.var("x1", relay.ty.TensorType((dshape[0],), "int"))
        z = relay.vision.nms(x0, x1, overlap_threshold, force_suppress, topk)
        assert "overlap_threshold" in z.astext()
        zz = relay.ir_pass.infer_type(z)
        assert zz.checked_type == relay.ty.TensorType(dshape, "float32")

        if check_type_only:
            return

        func = relay.Function([x0, x1], z)
        func = relay.ir_pass.infer_type(func)
        ctx_list = [("llvm", tvm.cpu(0))]
        for target, ctx in ctx_list:
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(x0_data, x1_data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(x0_data, x1_data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [0, 0.4, 4, 21, 19, 40], [-1, 0.9, 35, 61, 52, 79],
                           [-1, -1, -1, -1, -1, -1]]])
    num_anchors = 5

    dshape = (tvm.var("n"), num_anchors, 6)
    verify_nms(np_data, np_valid_count, dshape, np_result, dshape[0],
               force_suppress=True, topk=2, check_type_only=True)
    dshape = (1, num_anchors, 6)
    verify_nms(np_data, np_valid_count, dshape, np_result, dshape[0],
               force_suppress=True, topk=2, check_type_only=False)

    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [1, 0.7, 30, 60, 50, 80], [-1, 0.9, 35, 61, 52, 79],
                           [-1, -1, -1, -1, -1, -1]]])
    dshape = (tvm.var("n"), num_anchors, 6)
    verify_nms(np_data, np_valid_count, dshape, np_result, dshape[0],
               check_type_only=True)
    dshape = (1, num_anchors, 6)
    verify_nms(np_data, np_valid_count, dshape, np_result, dshape[0],
               topk=3)


def test_multibox_transform_loc():
    def test_default_value():
        num_anchors = 3
        num_classes = 3

        np_cls_prob = np.array(
            [[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45],
              [0.7, 0.1, 0.2]]]).astype("float32")
        np_loc_preds = np.array(
            [[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4,
              -0.8]]).astype("float32")
        np_anchors = np.array(
            [[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2],
              [1.2, 1.2, 1.5, 1.5]]]).astype("float32")

        expected_np_out = np.array([[[1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                                     [0, 0.44999999, 1, 1, 1, 1],
                                     [0, 0.30000001, 0, 0, 0.22903419, 0.20435292]]])


        cls_prob = relay.var(
            "cls_prob",
            relay.ty.TensorType((1, num_anchors, num_classes), "float32"))
        loc_pred = relay.var(
            "loc_pred", relay.ty.TensorType((1, num_anchors * 4), "float32"))
        anchors = relay.var(
            "anchors", relay.ty.TensorType((1, num_anchors, 4), "float32"))

        mtl = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob, loc_pred=loc_pred, anchor=anchors)
        ret = relay.ir_pass.infer_type(mtl.astuple())
        ref_type = relay.ty.TupleType(
            tvm.convert([
                relay.ty.TensorType((1, num_anchors, 6), "float32"),
                relay.ty.TensorType((1, ), "int")
            ]))

        assert ret.checked_type == ref_type

        nms = relay.vision.nms(mtl[0], mtl[1])
        func = relay.Function([cls_prob, loc_pred, anchors], nms)
        func = relay.ir_pass.infer_type(func)
        ctx_list = [("llvm", tvm.cpu(0))]
        for target, ctx in ctx_list:
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_cls_prob, np_loc_preds,
                                            np_anchors)
            tvm.testing.assert_allclose(op_res1.asnumpy(), expected_np_out, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_cls_prob, np_loc_preds,
                                            np_anchors)
            tvm.testing.assert_allclose(op_res2.asnumpy(), expected_np_out, rtol=1e-5)

    def test_threshold():
        num_anchors = 5
        num_classes = 5
        n = tvm.var("n")
        cls_prob = relay.var(
            "cls_prob",
            relay.ty.TensorType((n, num_anchors, num_classes), "float32"))
        loc_pred = relay.var(
            "loc_pred", relay.ty.TensorType((n, num_anchors * 4), "float32"))
        anchors = relay.var(
            "anchors", relay.ty.TensorType((1, num_anchors, 4), "float32"))
        threshold = 0.02
        variances = (0.2, 0.2, 0.3, 0.3)

        ret = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob,
            loc_pred=loc_pred,
            anchor=anchors,
            threshold=threshold,
            variances=variances)
        ret = relay.ir_pass.infer_type(ret.astuple())
        ref_type = relay.ty.TupleType(
            tvm.convert([
                relay.ty.TensorType((n, num_anchors, 6), "float32"),
                relay.ty.TensorType((n, ), "int")
            ]))
        assert ret.checked_type == ref_type

    test_default_value()
    test_threshold()


def test_roi_align():
    def verify_roi_align(data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio):
        data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
        rois = relay.var("rois", relay.ty.TensorType(rois_shape, "float32"))
        z = relay.vision.roi_align(data, rois, pooled_size=(pooled_size, pooled_size),
                                   spatial_scale=spatial_scale, sample_ratio=sample_ratio,
                                   layout="NCHW")
        zz = relay.ir_pass.infer_type(z)

        batch, channel, in_size, _ = data_shape
        num_roi = rois_shape[0]
        assert zz.checked_type == relay.ty.TensorType(
                (num_roi, channel, pooled_size, pooled_size), "float32")

        func = relay.Function([data, rois], z)
        func = relay.ir_pass.infer_type(func)
        np_data = np.random.uniform(size=data_shape).astype("float32")
        np_rois = np.random.uniform(size=rois_shape).astype('float32') * in_size
        np_rois[:, 0] = np.random.randint(low = 0, high = batch, size = num_roi)
        ref_res = topi.testing.roi_align_nchw_python(np_data, np_rois, pooled_size=pooled_size,
                                                     spatial_scale=spatial_scale,
                                                     sample_ratio=sample_ratio)
        for target, ctx in ctx_list():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-4)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-4)

    verify_roi_align((1, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=1.0, sample_ratio=-1)
    verify_roi_align((4, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=0.5, sample_ratio=2)


def test_yolo_reorg_infer_shape():
    def verify_yolo_reorg(shape, stride, out_shape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.vision.yolo_reorg(x, stride=stride)
        zz = relay.ir_pass.infer_type(z)
        assert "stride=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(out_shape, "float32")

    n, c, h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    verify_yolo_reorg((n, c, 20, 20), 10, (n, c*10*10, 2, 2))
    verify_yolo_reorg((n, c, h, w), 2, (n, c*2*2, h/2, w/2))

def test_yolo_reorg():
    def verify_yolo_reorg(shape, stride):
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = topi.testing.reorg_python(x_data, stride)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.vision.yolo_reorg(x, stride=stride)
        zz = relay.ir_pass.infer_type(z)
        assert "stride=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(ref_res.shape, "float32")

        func = relay.Function([x], z)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

    verify_yolo_reorg((1, 100, 20, 20), 10)
    verify_yolo_reorg((1, 4, 6, 6), 2)

if __name__ == "__main__":
    test_resize_infer_type()
    test_resize()
    test_multibox_prior()
    test_multibox_transform_loc()
    test_nms()
    test_roi_align()
    test_yolo_reorg_infer_shape()
    test_yolo_reorg()
