# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test code for vision package"""
from __future__ import print_function
import math
import numpy as np
import tvm
from tvm import te
import topi
import topi.testing

from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple
from topi.vision import ssd, non_max_suppression, get_valid_counts

_get_valid_counts_implement = {
    "generic": (topi.vision.get_valid_counts, topi.generic.schedule_get_valid_counts),
    "gpu": (topi.cuda.get_valid_counts, topi.cuda.schedule_get_valid_counts),
}

_nms_implement = {
    "generic": (topi.vision.non_max_suppression, topi.generic.schedule_nms),
    "gpu": (topi.cuda.non_max_suppression, topi.cuda.schedule_nms),
}

_multibox_prior_implement = {
    "generic": (topi.vision.ssd.multibox_prior, topi.generic.schedule_multibox_prior),
    "gpu": (topi.cuda.multibox_prior, topi.cuda.schedule_multibox_prior),
}

_multibox_detection_implement = {
    "generic": (topi.vision.ssd.multibox_detection, topi.generic.schedule_multibox_detection),
    "gpu": (topi.cuda.multibox_detection, topi.cuda.schedule_multibox_detection),
}

_roi_align_implement = {
    "generic": (topi.vision.roi_align_nchw, topi.generic.schedule_roi_align),
    "cpu": (topi.x86.roi_align_nchw, topi.generic.schedule_roi_align),
    "gpu": (topi.vision.roi_align_nchw, topi.cuda.schedule_roi_align),
}

_roi_pool_schedule = {
    "generic": topi.generic.schedule_roi_pool,
    "gpu": topi.cuda.schedule_roi_pool,
}

_proposal_implement = {
    "generic": (topi.vision.rcnn.proposal, topi.generic.schedule_proposal),
    "gpu": (topi.cuda.proposal, topi.cuda.schedule_proposal),
}

def verify_get_valid_counts(dshape, score_threshold, id_index, score_index):
    dtype = "float32"
    batch_size, num_anchor, elem_length = dshape
    np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
    np_out1 = np.zeros(shape=(batch_size,))
    np_out2 = np.zeros(shape=dshape).astype(dtype)
    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor):
            score = np_data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                for k in range(elem_length):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(elem_length):
                    np_out2[i, j, k] = -1.0

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            fcompute, fschedule = topi.testing.dispatch(device, _get_valid_counts_implement)
            data = te.placeholder(dshape, name="data", dtype=dtype)
            outs = fcompute(data, score_threshold, id_index, score_index)
            s = fschedule(outs)

        tvm_input_data = tvm.nd.array(np_data, ctx)
        tvm_out1 = tvm.nd.array(np.zeros(np_out1.shape, dtype="int32"), ctx)
        tvm_out2 = tvm.nd.array(np.zeros(np_out2.shape, dtype=dtype), ctx)
        f = tvm.build(s, [data, outs[0], outs[1]], device)
        f(tvm_input_data, tvm_out1, tvm_out2)
        tvm.testing.assert_allclose(tvm_out1.asnumpy(), np_out1, rtol=1e-3)
        tvm.testing.assert_allclose(tvm_out2.asnumpy(), np_out2, rtol=1e-3)

    """ Skip this test as it is intermittent
        see https://github.com/apache/incubator-tvm/pull/4901#issuecomment-595040094
    for device in ['llvm', 'cuda', 'opencl']:
        # Disable gpu test for now
        if device != "llvm":
            continue
        check_device(device)
    """


def test_get_valid_counts():
    verify_get_valid_counts((1, 2500, 6), 0, 0, 1)
    verify_get_valid_counts((1, 2500, 5), -1, -1, 0)
    verify_get_valid_counts((3, 1000, 6), 0.55, 1, 0)
    verify_get_valid_counts((16, 500, 5), 0.95, -1, 1)


def verify_non_max_suppression(np_data, np_valid_count, np_result, np_indices_result, iou_threshold,
                               force_suppress, top_k, coord_start, score_index, id_index):
    dshape = np_data.shape
    batch, num_anchors, _ = dshape
    indices_dshape = (batch, num_anchors)
    data = te.placeholder(dshape, name="data")
    valid_count = te.placeholder((batch,), dtype="int32", name="valid_count")

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            fcompute, fschedule = topi.testing.dispatch(device, _nms_implement)
            out = fcompute(data, valid_count, -1, iou_threshold, force_suppress, top_k,
                           coord_start=coord_start, score_index=score_index, id_index=id_index,
                           return_indices=False)
            indices_out = fcompute(data, valid_count, -1, iou_threshold, force_suppress, top_k,
                                   coord_start=coord_start, score_index=score_index, id_index=id_index)
            s = fschedule(out)
            indices_s = fschedule(indices_out)

        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)

        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f = tvm.build(s, [data, valid_count, out], device)
        f(tvm_data, tvm_valid_count, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), np_result, rtol=1e-4)

        tvm_indices_out = tvm.nd.array(np.zeros(indices_dshape, dtype="int32"), ctx)
        f = tvm.build(indices_s, [data, valid_count, indices_out], device)
        f(tvm_data, tvm_valid_count, tvm_indices_out)
        tvm.testing.assert_allclose(tvm_indices_out.asnumpy(), np_indices_result, rtol=1e-4)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)


def test_non_max_suppression():
    np_data = np.array([[[0, 0.8, 1, 20, 25, 45], [1, 0.7, 30, 60, 50, 80],
                         [0, 0.4, 4, 21, 19, 40], [2, 0.9, 35, 61, 52, 79],
                         [1, 0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_result = np.array([[[2, 0.9, 35, 61, 52, 79], [0, 0.8, 1, 20, 25, 45],
                           [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1, -1]]])
    np_indices_result = np.array([[3, 0, -1, -1, -1]])

    verify_non_max_suppression(np_data, np_valid_count, np_result, np_indices_result, 0.7, True, 2, 2, 1, 0)

    np_data = np.array([[[0.8, 1, 20, 25, 45], [0.7, 30, 60, 50, 80],
                         [0.4, 4, 21, 19, 40], [0.9, 35, 61, 52, 79],
                         [0.5, 100, 60, 70, 110]]]).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_result = np.array([[[0.9, 35, 61, 52, 79], [0.8, 1, 20, 25, 45],
                           [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1]]])
    np_indices_result = np.array([[3, 0, -1, -1, -1]])
    verify_non_max_suppression(np_data, np_valid_count, np_result, np_indices_result, 0.7, False, 2, 1, 0, -1)



def verify_multibox_prior(dshape, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    data = te.placeholder(dshape, name="data")

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

        fcompute, fschedule = topi.testing.dispatch(device, _multibox_prior_implement)
        with tvm.target.create(device):
            out = fcompute(data, sizes, ratios, steps, offsets, clip)
            s = fschedule(out)

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
    cls_prob = te.placeholder((batch_size, num_anchors, num_classes), name="cls_prob")
    loc_preds = te.placeholder((batch_size, num_anchors * 4), name="loc_preds")
    anchors = te.placeholder((1, num_anchors, 4), name="anchors")

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

        fcompute, fschedule = topi.testing.dispatch(device, _multibox_detection_implement)
        with tvm.target.create(device):
            out = fcompute(cls_prob, loc_preds, anchors)
            s = fschedule(out)

        tvm_cls_prob = tvm.nd.array(np_cls_prob.astype(cls_prob.dtype), ctx)
        tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), ctx)
        tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), ctx)
        tvm_out = tvm.nd.array(np.zeros((batch_size, num_anchors, 6)).astype(out.dtype), ctx)
        f = tvm.build(s, [cls_prob, loc_preds, anchors, out], device)
        f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
        tvm.testing.assert_allclose(tvm_out.asnumpy(), expected_np_out, rtol=1e-4)

    for device in ['llvm', 'opencl', 'cuda']:
        check_device(device)


def verify_roi_align(batch, in_channel, in_size, num_roi, pooled_size, spatial_scale, sample_ratio):
    a_shape = (batch, in_channel, in_size, in_size)
    rois_shape = (num_roi, 5)

    a = te.placeholder(a_shape)
    rois = te.placeholder(rois_shape)

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
            fcompute, fschedule = topi.testing.dispatch(device, _roi_align_implement)
            b = fcompute(a, rois, pooled_size=pooled_size,
                         spatial_scale=spatial_scale,
                         sample_ratio=sample_ratio)
            s = fschedule(b)

        tvm_a = tvm.nd.array(a_np, ctx)
        tvm_rois = tvm.nd.array(rois_np, ctx)
        tvm_b = tvm.nd.array(np.zeros(get_const_tuple(b.shape), dtype=b.dtype), ctx=ctx)
        f = tvm.build(s, [a, rois, b], device)
        f(tvm_a, tvm_rois, tvm_b)
        tvm.testing.assert_allclose(tvm_b.asnumpy(), b_np, rtol=1e-3)

    for device in ['llvm', 'cuda', 'opencl']:
        check_device(device)


def test_roi_align():
    verify_roi_align(1, 16, 32, 64, 7, 1.0, -1)
    verify_roi_align(4, 16, 32, 64, 7, 0.5, 2)
    verify_roi_align(1, 32, 32, 80, 8, 0.0625, 2)
    verify_roi_align(1, 32, 500, 80, 8, 0.0625, 2)


def verify_roi_pool(batch, in_channel, in_size, num_roi, pooled_size, spatial_scale):
    a_shape = (batch, in_channel, in_size, in_size)
    rois_shape = (num_roi, 5)

    a = te.placeholder(a_shape)
    rois = te.placeholder(rois_shape)

    @memoize("topi.tests.test_topi_vision.verify_roi_pool")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype('float32')
        rois_np = np.random.uniform(size=rois_shape).astype('float32') * in_size
        rois_np[:, 0] = np.random.randint(low = 0, high = batch, size = num_roi).astype('float32')

        b_np = topi.testing.roi_pool_nchw_python(a_np, rois_np, pooled_size=pooled_size,
                                                 spatial_scale=spatial_scale)
        return a_np, rois_np, b_np

    a_np, rois_np, b_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        with tvm.target.create(device):
            b = topi.vision.rcnn.roi_pool_nchw(a, rois, pooled_size=pooled_size,
                                                spatial_scale=spatial_scale)
            s_func = topi.testing.dispatch(device, _roi_pool_schedule)
            s = s_func(b)

        tvm_a = tvm.nd.array(a_np, ctx)
        tvm_rois = tvm.nd.array(rois_np, ctx)
        tvm_b = tvm.nd.array(np.zeros(get_const_tuple(b.shape), dtype=b.dtype), ctx=ctx)
        f = tvm.build(s, [a, rois, b], device)
        f(tvm_a, tvm_rois, tvm_b)
        tvm.testing.assert_allclose(tvm_b.asnumpy(), b_np, rtol=1e-4)

    for device in ['cuda', 'llvm']:
        check_device(device)


def test_roi_pool():
    verify_roi_pool(1, 4, 16, 32, 7, 1.0)
    verify_roi_pool(4, 4, 16, 32, 7, 0.5)


def verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs):
    cls_prob = te.placeholder(np_cls_prob.shape)
    bbox_pred = te.placeholder(np_bbox_pred.shape)
    im_info = te.placeholder(np_im_info.shape)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            fcompute, fschedule = topi.testing.dispatch(device, _proposal_implement)
            out = fcompute(cls_prob, bbox_pred, im_info, **attrs)
            s = fschedule(out)
            f = tvm.build(s, [cls_prob, bbox_pred, im_info, out], device)
            tvm_cls_prob = tvm.nd.array(np_cls_prob, ctx=ctx)
            tvm_bbox_pred = tvm.nd.array(np_bbox_pred, ctx=ctx)
            tvm_im_info = tvm.nd.array(np_im_info, ctx=ctx)
            tvm_out = tvm.nd.empty(ctx=ctx, shape=out.shape, dtype=out.dtype)
            f(tvm_cls_prob, tvm_bbox_pred, tvm_im_info, tvm_out)
            tvm.testing.assert_allclose(tvm_out.asnumpy(), np_out, rtol=1e-4)

    for device in ['llvm', 'cuda']:
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
    np_im_info = np.array([[48., 48., 1.]], dtype='float32')
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
    test_get_valid_counts()
    test_non_max_suppression()
    test_multibox_prior()
    test_multibox_detection()
    test_roi_align()
    test_roi_pool()
    test_proposal()
