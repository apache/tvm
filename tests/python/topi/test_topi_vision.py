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
import math
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi
from tvm.topi.vision import ssd, non_max_suppression, get_valid_counts

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

_all_class_nms_implement = {
    "generic": (topi.vision.all_class_non_max_suppression, topi.generic.schedule_nms),
    "gpu": (topi.cuda.all_class_non_max_suppression, topi.cuda.schedule_nms),
}


class TestValidCounts:
    dshape, score_threshold, id_index, score_index = tvm.testing.parameters(
        ((1, 1000, 5), 0.5, -1, 0),
        ((1, 2500, 6), 0, 0, 1),
        ((1, 2500, 5), -1, -1, 0),
        ((3, 1000, 6), 0.55, 1, 0),
        ((16, 500, 5), 0.95, -1, 1),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(self, dtype, dshape, score_threshold, id_index, score_index):
        batch_size, num_anchor, elem_length = dshape
        np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
        np_out1 = np.zeros(shape=(batch_size,))
        np_out2 = np.zeros(shape=dshape).astype(dtype)
        np_out3 = np.zeros(shape=(batch_size, num_anchor))
        for i in range(batch_size):
            np_out1[i] = 0
            inter_idx = 0
            for j in range(num_anchor):
                score = np_data[i, j, score_index]
                if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                    for k in range(elem_length):
                        np_out2[i, inter_idx, k] = np_data[i, j, k]
                    np_out1[i] += 1
                    np_out3[i, inter_idx] = j
                    inter_idx += 1
                if j >= np_out1[i]:
                    for k in range(elem_length):
                        np_out2[i, j, k] = -1.0
                    np_out3[i, j] = -1

        return np_data, np_out1, np_out2, np_out3

    def test_get_valid_counts(
        self, target, dev, ref_data, dtype, dshape, score_threshold, id_index, score_index
    ):
        np_data, np_out1, np_out2, np_out3 = ref_data

        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _get_valid_counts_implement)
            data = te.placeholder(dshape, name="data", dtype=dtype)
            outs = fcompute(data, score_threshold, id_index, score_index)
            s = fschedule(outs)

        tvm_input_data = tvm.nd.array(np_data, dev)
        tvm_out1 = tvm.nd.array(np.zeros(np_out1.shape, dtype="int32"), dev)
        tvm_out2 = tvm.nd.array(np.zeros(np_out2.shape, dtype=dtype), dev)
        tvm_out3 = tvm.nd.array(np.zeros(np_out3.shape, dtype="int32"), dev)

        f = tvm.build(s, [data, outs[0], outs[1], outs[2]], target)
        f(tvm_input_data, tvm_out1, tvm_out2, tvm_out3)
        tvm.testing.assert_allclose(tvm_out1.numpy(), np_out1, rtol=1e-3)
        tvm.testing.assert_allclose(tvm_out2.numpy(), np_out2, rtol=1e-3)
        tvm.testing.assert_allclose(tvm_out3.numpy(), np_out3, rtol=1e-3)


def verify_non_max_suppression(
    target,
    dev,
    np_data,
    np_valid_count,
    np_indices,
    np_result,
    np_indices_result,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    score_index,
    id_index,
):
    dshape = np_data.shape
    batch, num_anchors, _ = dshape
    indices_dshape = (batch, num_anchors)
    data = te.placeholder(dshape, name="data")
    valid_count = te.placeholder((batch,), dtype="int32", name="valid_count")
    indices = te.placeholder((batch, num_anchors), dtype="int32", name="indices")

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _nms_implement)
        out = fcompute(
            data,
            valid_count,
            indices,
            max_output_size,
            iou_threshold,
            force_suppress,
            top_k,
            coord_start=coord_start,
            score_index=score_index,
            id_index=id_index,
            return_indices=False,
        )
        indices_out = fcompute(
            data,
            valid_count,
            indices,
            max_output_size,
            iou_threshold,
            force_suppress,
            top_k,
            coord_start=coord_start,
            score_index=score_index,
            id_index=id_index,
            return_indices=True,
        )
        s = fschedule(out)
        indices_s = fschedule(indices_out)

    tvm_data = tvm.nd.array(np_data, dev)
    tvm_valid_count = tvm.nd.array(np_valid_count, dev)
    tvm_indices = tvm.nd.array(np_indices, dev)

    tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), dev)
    f = tvm.build(s, [data, valid_count, indices, out], target)
    f(tvm_data, tvm_valid_count, tvm_indices, tvm_out)
    tvm.testing.assert_allclose(tvm_out.numpy(), np_result, rtol=1e-4)

    tvm_indices_out = tvm.nd.array(np.zeros(indices_dshape, dtype="int32"), dev)
    f = tvm.build(indices_s, [data, valid_count, indices, indices_out[0]], target)
    f(tvm_data, tvm_valid_count, tvm_indices, tvm_indices_out)
    tvm.testing.assert_allclose(tvm_indices_out.numpy(), np_indices_result, rtol=1e-4)


def test_non_max_suppression(target, dev):
    np_data = np.array(
        [
            [
                [0, 0.8, 1, 20, 25, 45],
                [1, 0.7, 30, 60, 50, 80],
                [0, 0.4, 4, 21, 19, 40],
                [2, 0.9, 35, 61, 52, 79],
                [1, 0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 2, 3, 4]]).astype("int32")
    max_output_size = -1
    np_result = np.array(
        [
            [
                [2, 0.9, 35, 61, 52, 79],
                [0, 0.8, 1, 20, 25, 45],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ]
        ]
    )
    np_indices_result = np.array([[3, 0, -1, -1, -1]])

    verify_non_max_suppression(
        target,
        dev,
        np_data,
        np_valid_count,
        np_indices,
        np_result,
        np_indices_result,
        max_output_size,
        0.7,
        True,
        2,
        2,
        1,
        0,
    )

    np_data = np.array(
        [
            [
                [0.8, 1, 20, 25, 45],
                [0.7, 30, 60, 50, 80],
                [0.4, 4, 21, 19, 40],
                [0.9, 35, 61, 52, 79],
                [0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 2, 3, 4]]).astype("int32")
    max_output_size = 2
    np_result = np.array(
        [
            [
                [0.9, 35, 61, 52, 79],
                [0.8, 1, 20, 25, 45],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1],
            ]
        ]
    )
    np_indices_result = np.array([[3, 0, -1, -1, -1]])
    verify_non_max_suppression(
        target,
        dev,
        np_data,
        np_valid_count,
        np_indices,
        np_result,
        np_indices_result,
        max_output_size,
        0.7,
        False,
        2,
        1,
        0,
        -1,
    )


class TestMultiboxPrior:
    dshape, sizes, ratios, steps, offsets, clip = tvm.testing.parameters(
        ((1, 3, 50, 50), (1,), (1,), (-1, -1), (0.5, 0.5), False),
        ((1, 3, 224, 224), (0.5, 0.25, 0.1), (1, 2, 0.5), (-1, -1), (0.5, 0.5), False),
        ((1, 32, 32, 32), (0.5, 0.25), (1, 2), (2, 2), (0.5, 0.5), True),
    )

    dtype = tvm.testing.parameter("float32")

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(self, dtype, dshape, sizes, ratios, offsets, steps, clip):
        in_height = dshape[2]
        in_width = dshape[3]
        num_sizes = len(sizes)
        num_ratios = len(ratios)
        size_ratio_concat = sizes + ratios
        steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
        steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
        offset_h = offsets[0]
        offset_w = offsets[1]

        out_shape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)

        np_in = np.random.uniform(size=dshape).astype(dtype)
        np_out = np.zeros(out_shape).astype(dtype)

        for i in range(in_height):
            center_h = (i + offset_h) * steps_h
            for j in range(in_width):
                center_w = (j + offset_w) * steps_w
                for k in range(num_sizes + num_ratios - 1):
                    w = (
                        size_ratio_concat[k] * in_height / in_width / 2.0
                        if k < num_sizes
                        else size_ratio_concat[0]
                        * in_height
                        / in_width
                        * math.sqrt(size_ratio_concat[k + 1])
                        / 2.0
                    )
                    h = (
                        size_ratio_concat[k] / 2.0
                        if k < num_sizes
                        else size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    )
                    count = (
                        i * in_width * (num_sizes + num_ratios - 1)
                        + j * (num_sizes + num_ratios - 1)
                        + k
                    )
                    np_out[0][count][0] = center_w - w
                    np_out[0][count][1] = center_h - h
                    np_out[0][count][2] = center_w + w
                    np_out[0][count][3] = center_h + h
        if clip:
            np_out = np.clip(np_out, 0, 1)

        return np_in, np_out

    def test_multibox_prior(
        self, target, dev, dtype, dshape, ref_data, sizes, ratios, steps, offsets, clip
    ):
        np_in, np_out = ref_data

        data = te.placeholder(dshape, name="data", dtype=dtype)

        fcompute, fschedule = tvm.topi.testing.dispatch(target, _multibox_prior_implement)
        with tvm.target.Target(target):
            out = fcompute(data, sizes, ratios, steps, offsets, clip)
            s = fschedule(out)

        tvm_input_data = tvm.nd.array(np_in, dev)
        tvm_out = tvm.nd.array(np.zeros(np_out.shape, dtype=dtype), dev)
        f = tvm.build(s, [data, out], target)
        f(tvm_input_data, tvm_out)
        tvm.testing.assert_allclose(tvm_out.numpy(), np_out, rtol=1e-3)


class TestMultiboxDetection:
    (batch_size,) = tvm.testing.parameters((1,), (6,))

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(
        self,
        batch_size,
    ):
        # Manually create test case
        np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]] * batch_size)
        np_loc_preds = np.array(
            [[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]] * batch_size
        )
        np_anchors = np.array(
            [[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]] * batch_size
        )
        expected_np_out = np.array(
            [
                [
                    [1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                    [0, 0.44999999, 1, 1, 1, 1],
                    [0, 0.30000001, 0, 0, 0.22903419, 0.20435292],
                ]
            ]
            * batch_size
        )
        return np_cls_prob, np_loc_preds, np_anchors, expected_np_out

    def test_multibox_detection(self, target, dev, ref_data):

        np_cls_prob, np_loc_preds, np_anchors, expected_np_out = ref_data

        batch_size = np_cls_prob.shape[0]
        num_anchors = 3
        num_classes = 3
        cls_prob = te.placeholder((batch_size, num_anchors, num_classes), name="cls_prob")
        loc_preds = te.placeholder((batch_size, num_anchors * 4), name="loc_preds")
        anchors = te.placeholder((batch_size, num_anchors, 4), name="anchors")

        fcompute, fschedule = tvm.topi.testing.dispatch(target, _multibox_detection_implement)
        with tvm.target.Target(target):
            out = fcompute(cls_prob, loc_preds, anchors)
            s = fschedule(out)

        tvm_cls_prob = tvm.nd.array(np_cls_prob.astype(cls_prob.dtype), dev)
        tvm_loc_preds = tvm.nd.array(np_loc_preds.astype(loc_preds.dtype), dev)
        tvm_anchors = tvm.nd.array(np_anchors.astype(anchors.dtype), dev)
        tvm_out = tvm.nd.array(np.zeros((batch_size, num_anchors, 6)).astype(out.dtype), dev)
        f = tvm.build(s, [cls_prob, loc_preds, anchors, out], target)
        f(tvm_cls_prob, tvm_loc_preds, tvm_anchors, tvm_out)
        tvm.testing.assert_allclose(tvm_out.numpy(), expected_np_out, rtol=1e-4)


class TestRoiAlign:
    (
        batch,
        in_channel,
        in_size,
        num_roi,
        pooled_size,
        spatial_scale,
        sample_ratio,
        mode,
    ) = tvm.testing.parameters(
        (1, 16, 32, 64, 7, 1.0, -1, 0),
        (4, 16, 32, 64, 7, 0.5, 2, 0),
        (1, 32, 32, 80, 8, 0.0625, 2, 0),
        (1, 32, 500, 80, 8, 0.0625, 2, 0),
        (1, 16, 32, 64, 7, 1.0, -1, 1),
        (4, 16, 32, 64, 7, 0.5, 2, 1),
        (1, 32, 32, 80, 8, 0.0625, 2, 1),
        (1, 32, 500, 80, 8, 0.0625, 2, 1),
    )

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(
        self,
        batch,
        in_channel,
        in_size,
        num_roi,
        pooled_size,
        spatial_scale,
        sample_ratio,
        mode,
    ):
        a_shape = (batch, in_channel, in_size, in_size)
        rois_shape = (num_roi, 5)

        a_np = np.random.uniform(-1, 1, size=a_shape).astype("float32")
        rois_np = np.random.uniform(-1, 1, size=rois_shape).astype("float32") * in_size
        rois_np[:, 0] = np.random.randint(low=0, high=batch, size=num_roi)
        b_np = tvm.topi.testing.roi_align_nchw_python(
            a_np,
            rois_np,
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            sample_ratio=sample_ratio,
            mode=mode,
        )

        return a_np, rois_np, b_np

    def test_roi_align(
        self,
        target,
        dev,
        ref_data,
        pooled_size,
        spatial_scale,
        sample_ratio,
        mode,
    ):
        # For mode, 0 = avg, 1 = max
        a_np, rois_np, b_np = ref_data

        a = te.placeholder(a_np.shape)
        rois = te.placeholder(rois_np.shape)

        with tvm.target.Target(target):
            fcompute, fschedule = tvm.topi.testing.dispatch(target, _roi_align_implement)
            b = fcompute(
                a,
                rois,
                pooled_size=pooled_size,
                spatial_scale=spatial_scale,
                sample_ratio=sample_ratio,
                mode=mode,
            )
            s = fschedule(b)

        tvm_a = tvm.nd.array(a_np, dev)
        tvm_rois = tvm.nd.array(rois_np, dev)
        tvm_b = tvm.nd.array(np.zeros(b_np.shape, dtype=b.dtype), device=dev)
        f = tvm.build(s, [a, rois, b], target)
        f(tvm_a, tvm_rois, tvm_b)
        tvm_val = tvm_b.numpy()
        tvm.testing.assert_allclose(tvm_val, b_np, rtol=1e-3, atol=1e-4)


class TestRoiPool:
    batch, in_channel, in_size, num_roi, pooled_size, spatial_scale = tvm.testing.parameters(
        (1, 4, 16, 32, 7, 1.0),
        (4, 4, 16, 32, 7, 0.5),
    )

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(self, batch, in_channel, in_size, num_roi, pooled_size, spatial_scale):
        a_shape = (batch, in_channel, in_size, in_size)
        rois_shape = (num_roi, 5)

        a_np = np.random.uniform(size=a_shape).astype("float32")
        rois_np = np.random.uniform(size=rois_shape).astype("float32") * in_size
        rois_np[:, 0] = np.random.randint(low=0, high=batch, size=num_roi).astype("float32")

        b_np = tvm.topi.testing.roi_pool_nchw_python(
            a_np, rois_np, pooled_size=pooled_size, spatial_scale=spatial_scale
        )
        return a_np, rois_np, b_np

    def test_roi_pool(self, target, dev, ref_data, pooled_size, spatial_scale):
        a_np, rois_np, b_np = ref_data

        a = te.placeholder(a_np.shape)
        rois = te.placeholder(rois_np.shape)

        with tvm.target.Target(target):
            b = topi.vision.rcnn.roi_pool_nchw(
                a, rois, pooled_size=pooled_size, spatial_scale=spatial_scale
            )
            s_func = tvm.topi.testing.dispatch(target, _roi_pool_schedule)
            s = s_func(b)

        tvm_a = tvm.nd.array(a_np, dev)
        tvm_rois = tvm.nd.array(rois_np, dev)
        tvm_b = tvm.nd.array(np.zeros(b_np.shape, dtype=b.dtype), device=dev)
        f = tvm.build(s, [a, rois, b], target)
        f(tvm_a, tvm_rois, tvm_b)
        tvm.testing.assert_allclose(tvm_b.numpy(), b_np, rtol=1e-4)


def verify_proposal(target, dev, np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs):
    cls_prob = te.placeholder(np_cls_prob.shape)
    bbox_pred = te.placeholder(np_bbox_pred.shape)
    im_info = te.placeholder(np_im_info.shape)

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _proposal_implement)
        out = fcompute(cls_prob, bbox_pred, im_info, **attrs)
        s = fschedule(out)
        f = tvm.build(s, [cls_prob, bbox_pred, im_info, out], target)
        tvm_cls_prob = tvm.nd.array(np_cls_prob, device=dev)
        tvm_bbox_pred = tvm.nd.array(np_bbox_pred, device=dev)
        tvm_im_info = tvm.nd.array(np_im_info, device=dev)
        tvm_out = tvm.nd.empty(device=dev, shape=out.shape, dtype=out.dtype)
        f(tvm_cls_prob, tvm_bbox_pred, tvm_im_info, tvm_out)
        tvm.testing.assert_allclose(tvm_out.numpy(), np_out, rtol=1e-4)


@tvm.testing.known_failing_targets("vulkan")
def test_proposal(target, dev):
    attrs = {
        "scales": (0.5,),
        "ratios": (0.5,),
        "feature_stride": 16,
        "iou_loss": False,
        "rpn_min_size": 16,
        "threshold": 0.7,
        "rpn_pre_nms_top_n": 200,
        "rpn_post_nms_top_n": 4,
    }
    np_cls_prob = np.array(
        [
            [
                [[0.3, 0.6, 0.2], [0.4, 0.7, 0.5], [0.1, 0.4, 0.3]],
                [[0.7, 0.5, 0.3], [0.6, 0.4, 0.8], [0.9, 0.2, 0.5]],
            ]
        ],
        dtype="float32",
    )
    np_bbox_pred = np.array(
        [
            [
                [[0.5, 1.0, 0.6], [0.8, 1.2, 2.0], [0.9, 1.0, 0.8]],
                [[0.5, 1.0, 0.7], [0.8, 1.2, 1.6], [2.1, 1.5, 0.7]],
                [[1.0, 0.5, 0.7], [1.5, 0.9, 1.6], [1.4, 1.5, 0.8]],
                [[1.0, 0.5, 0.6], [1.5, 0.9, 2.0], [1.8, 1.0, 0.9]],
            ]
        ],
        dtype="float32",
    )
    np_im_info = np.array([[48.0, 48.0, 1.0]], dtype="float32")
    np_out = np.array(
        [
            [0.0, 0.0, 2.8451548, 28.38012, 18.154846],
            [0.0, 0.0, 15.354933, 41.96971, 41.245064],
            [0.0, 18.019852, 1.0538368, 51.98015, 25.946163],
            [0.0, 27.320923, -1.266357, 55.0, 24.666357],
        ],
        dtype="float32",
    )

    verify_proposal(target, dev, np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)

    np_out = np.array(
        [
            [0.0, -5.25, -2.5, 21.75, 19.0],
            [0.0, 11.25, -2.0, 37.25, 18.5],
            [0.0, 26.849998, -2.3000002, 53.45, 18.6],
            [0.0, -4.95, 13.799999, 22.25, 35.5],
        ],
        dtype="float32",
    )

    attrs["iou_loss"] = True
    verify_proposal(target, dev, np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)


def verify_all_class_non_max_suppression(
    target,
    dev,
    boxes_np,
    scores_np,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    expected_indices,
):
    dshape = boxes_np.shape
    batch, num_boxes, _ = dshape
    _, num_class, _ = scores_np.shape
    boxes = te.placeholder(dshape, name="boxes")
    scores = te.placeholder(scores_np.shape, dtype="float32", name="scores")

    with tvm.target.Target(target):
        fcompute, fschedule = tvm.topi.testing.dispatch(target, _all_class_nms_implement)
        out = fcompute(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
        s = fschedule(out)

    tvm_boxes = tvm.nd.array(boxes_np, dev)
    tvm_scores = tvm.nd.array(scores_np, dev)
    selected_indices = tvm.nd.array(np.zeros((batch * num_class * num_boxes, 3), "int64"), dev)
    num_detections = tvm.nd.array(np.zeros((1,), "int64"), dev)

    f = tvm.build(s, [boxes, scores, out[0], out[1]], target)
    f(tvm_boxes, tvm_scores, selected_indices, num_detections)

    tvm_res = selected_indices.numpy()[: num_detections.numpy()[0]]
    np.testing.assert_equal(tvm_res, expected_indices)


def test_all_class_non_max_suppression(target, dev):
    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.95, 0.95],
                [0.5, 0.5, 0.96, 0.96],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")

    max_output_boxes_per_class = 2
    iou_threshold = 0.8
    score_threshold = 0.0

    expected = np.array(
        [[0, 0, 4], [0, 0, 2], [0, 1, 4], [0, 1, 2], [1, 0, 4], [1, 0, 1], [1, 1, 4], [1, 1, 1]]
    )

    verify_all_class_non_max_suppression(
        target,
        dev,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        expected,
    )

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = 3
    iou_threshold = 0.5
    score_threshold = 0.4

    expected = np.array([[0, 0, 3], [0, 0, 0]])

    verify_all_class_non_max_suppression(
        target,
        dev,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        expected,
    )


if __name__ == "__main__":
    tvm.testing.main()
