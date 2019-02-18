"""Test code for vision package"""
from __future__ import print_function
import math
import numpy as np
import tvm
import topi
import topi.testing

from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi.vision import ssd, nms


def test_nms():
    dshape = (1, 5, 6)
    data = tvm.placeholder(dshape, name="data")
    valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
    nms_threshold = 0.7
    force_suppress = True
    nms_topk = 2

    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype(data.dtype)
    np_valid_count = np.array([4]).astype(valid_count.dtype)
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [0, 0.4, 4, 21, 19, 40], [-1, 0.9, 35, 61, 52, 79],
                           [-1, -1, -1, -1, -1, -1]]])

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                out = nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
            else:
                out = topi.cuda.nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
            s = topi.generic.schedule_nms(out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result, rtol=1e-4)

    for device in ['llvm']:
        check_device(device)


def verify_multibox_prior(dshape, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    data = tvm.placeholder(dshape, name="data")

    dtype = data.dtype
    input_data = np.random.uniform(size=dshape).astype(dtype)

    in_height = data.shape[2].value
    in_width = data.shape[3].value
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
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

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                out = ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)
            else:
                out = topi.cuda.ssd.multibox_prior(data, sizes, ratios, steps, offsets, clip)
            s = topi.generic.schedule_multibox_prior(out)

        tvm_input_data = tvm.nd.array(input_data, ctx)
        tvm_out = tvm.nd.array(np.zeros(oshape, dtype=dtype), ctx)
        f = tvm.build(s, [data, out], device)
        f(tvm_input_data, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_out, rtol=1e-3)

    for device in ['llvm', 'opencl', 'cuda']:
        check_device(device)


def test_multibox_prior():
    verify_multibox_prior((1, 3, 50, 50))
    verify_multibox_prior((1, 3, 224, 224), sizes=(0.5, 0.25, 0.1), ratios=(1, 2, 0.5))
    verify_multibox_prior((1, 32, 32, 32), sizes=(0.5, 0.25), ratios=(1, 2), steps=(2, 2), clip=True)


def test_multibox_detection():
    batch_size = 1
    num_anchors = 3
    num_classes = 3
    cls_prob = tvm.placeholder((batch_size, num_anchors, num_classes), name="cls_prob")
    loc_preds = tvm.placeholder((batch_size, num_anchors * 4), name="loc_preds")
    anchors = tvm.placeholder((1, num_anchors, 4), name="anchors")

    # Manually create test case
    np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]])
    np_loc_preds = np.array([[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]])
    np_anchors = np.array([[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]])

    expected_np_out = np.array([[[1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                                 [0, 0.44999999, 1, 1, 1, 1],
                                 [0, 0.30000001, 0, 0, 0.22903419, 0.20435292]]])

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            if device == 'llvm':
                out = ssd.multibox_detection(cls_prob, loc_preds, anchors)
            else:
                out = topi.cuda.ssd.multibox_detection(cls_prob, loc_preds, anchors)
            s = topi.generic.schedule_multibox_detection(out)

        tvm_cls_prob = tvm.nd.array(np_cls_prob.astype(cls_prob.dtype), ctx)
        tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), ctx)
        tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), ctx)
        tvm_out = tvm.nd.array(np.zeros((batch_size, num_anchors, 6)).astype(out.dtype), ctx)
        f = tvm.build(s, [cls_prob, loc_preds, anchors, out], device)
        f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), expected_np_out, rtol=1e-4)

    for device in ['llvm', 'opencl']:
        check_device(device)


def verify_roi_align(batch, in_channel, in_size, num_roi, pooled_size, spatial_scale, sample_ratio):
    a_shape = (batch, in_channel, in_size, in_size)
    rois_shape = (num_roi, 5)

    a = tvm.placeholder(a_shape)
    rois = tvm.placeholder(rois_shape)

    @memoize("topi.tests.test_topi_vision.verify_roi_align")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype('float32')
        rois_np = np.random.uniform(size=rois_shape).astype('float32') * in_size
        rois_np[:, 0] = np.random.randint(low = 0, high = batch, size = num_roi)
        b_np = topi.testing.roi_align_nchw_python(a_np, rois_np, pooled_size=pooled_size,
                                                  spatial_scale=spatial_scale,
                                                  sample_ratio=sample_ratio)

        return a_np, rois_np, b_np

    a_np, rois_np, b_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            b = topi.vision.rcnn.roi_align_nchw(a, rois, pooled_size=pooled_size,
                                                spatial_scale=spatial_scale,
                                                sample_ratio=sample_ratio)
            s = topi.generic.schedule_roi_align(b)

        tvm_a = tvm.nd.array(a_np, ctx)
        tvm_rois = tvm.nd.array(rois_np, ctx)
        tvm_b = tvm.nd.array(np.zeros(get_const_tuple(b.shape), dtype=b.dtype), ctx=ctx)
        f = tvm.build(s, [a, rois, b], device)
        f(tvm_a, tvm_rois, tvm_b)
        tvm.testing.assert_allclose(tvm_b.asnumpy(), b_np, rtol=1e-3)

    for device in ['llvm', 'cuda']:
        check_device(device)


def test_roi_align():
    verify_roi_align(1, 16, 32, 64, 7, 1.0, -1)
    verify_roi_align(4, 16, 32, 64, 7, 0.5, 2)


def verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs):
    cls_prob = tvm.placeholder(np_cls_prob.shape)
    bbox_pred = tvm.placeholder(np_bbox_pred.shape)
    im_info = tvm.placeholder(np_im_info.shape, dtype='int32')

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            out = topi.vision.proposal(cls_prob, bbox_pred, im_info, **attrs)
            s = topi.generic.schedule_proposal(out)
            f = tvm.build(s, [cls_prob, bbox_pred, im_info, out], device)
            tvm_cls_prob = tvm.nd.array(np_cls_prob, ctx=ctx)
            tvm_bbox_pred = tvm.nd.array(np_bbox_pred, ctx=ctx)
            tvm_im_info = tvm.nd.array(np_im_info, ctx=ctx)
            tvm_out = tvm.nd.empty(ctx=ctx, shape=out.shape, dtype=out.dtype)
            f(tvm_cls_prob, tvm_bbox_pred, tvm_im_info, tvm_out)
            tvm.testing.assert_allclose(tvm_out.asnumpy(), np_out, rtol=1e-4)

    for device in ['cuda']:
        check_device(device)


def test_proposal():
    attrs = {'scales': (0.5,),'ratios': (0.5,),
        'feature_stride': 16,
        'iou_loss': False,
        'rpn_min_size': 16,
        'threshold': 0.7,
        'rpn_pre_nms_top_n': 200,
        'rpn_post_nms_top_n': 4,
    }
    np_cls_prob = np.array([[
        [[0.3, 0.6, 0.2], [0.4, 0.7, 0.5], [0.1, 0.4, 0.3]],
        [[0.7, 0.5, 0.3], [0.6, 0.4, 0.8], [0.9, 0.2, 0.5]]
    ]], dtype='float32')
    np_bbox_pred = np.array([[
        [[0.5, 1.0, 0.6], [0.8,  1.2, 2.0], [0.9, 1.0, 0.8]],
        [[0.5, 1.0, 0.7], [0.8,  1.2, 1.6], [2.1, 1.5, 0.7]],
        [[1.0, 0.5, 0.7], [1.5,  0.9, 1.6], [1.4, 1.5, 0.8]],
        [[1.0, 0.5, 0.6], [1.5,  0.9, 2.0], [1.8, 1.0, 0.9]],
    ]], dtype='float32')
    np_im_info = np.array([[48, 48, 1]], dtype='int32')
    np_out = np.array([
        [0., 0., 2.8451548,28.38012, 18.154846],
        [0., 0., 15.354933, 41.96971, 41.245064],
        [0., 18.019852, 1.0538368, 51.98015, 25.946163],
        [0., 27.320923, -1.266357, 55., 24.666357]
    ], dtype='float32')

    verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)

    np_out = np.array([
        [ 0., -5.25, -2.5, 21.75, 19.],
        [ 0., 11.25, -2., 37.25, 18.5],
        [ 0., 26.849998, -2.3000002, 53.45, 18.6],
        [ 0., -4.95, 13.799999, 22.25, 35.5]
    ], dtype='float32')

    attrs['iou_loss'] = True
    verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)


if __name__ == "__main__":
    test_nms()
    test_multibox_prior()
    test_multibox_detection()
    test_roi_align()
    test_proposal()
