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
# pylint: disable=invalid-name
"""Roi align operator"""
import tvm
from tvm import te
from ...utils import get_const_tuple
from ...cpp.utils import bilinear_sample_nchw, bilinear_sample_nhwc


def _sample_common(
    i,
    c,
    ph,
    pw,
    rois,
    pooled_size_h,
    pooled_size_w,
    spatial_scale,
    sample_ratio,
    dtype,
    avg_mode,
    bilinear_func,
):
    roi = rois[i]
    batch_index = roi[0].astype("int32")
    roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1], roi[2], roi[3], roi[4]
    roi_start_h *= spatial_scale
    roi_end_h *= spatial_scale
    roi_start_w *= spatial_scale
    roi_end_w *= spatial_scale

    # force malformed ROIs to be 1x1
    roi_h = tvm.te.max(roi_end_h - roi_start_h, tvm.tir.const(1.0, dtype))
    roi_w = tvm.te.max(roi_end_w - roi_start_w, tvm.tir.const(1.0, dtype))

    bin_h = roi_h / pooled_size_h
    bin_w = roi_w / pooled_size_w

    if sample_ratio > 0:
        roi_bin_grid_h = roi_bin_grid_w = tvm.tir.const(sample_ratio, "int32")
    else:
        roi_bin_grid_h = te.ceil(roi_h / pooled_size_h).astype("int32")
        roi_bin_grid_w = te.ceil(roi_w / pooled_size_w).astype("int32")

    count = roi_bin_grid_h * roi_bin_grid_w
    rh = te.reduce_axis((0, roi_bin_grid_h))
    rw = te.reduce_axis((0, roi_bin_grid_w))
    roi_start_h += ph * bin_h
    roi_start_w += pw * bin_w

    if avg_mode:
        return te.sum(
            bilinear_func(
                batch_index,
                c,
                roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h,
                roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w,
            )
            / count,
            axis=[rh, rw],
        )
    # max mode
    return te.max(
        bilinear_func(
            batch_index,
            c,
            roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h,
            roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w,
        ),
        axis=[rh, rw],
    )


def roi_align_nchw(data, rois, pooled_size, spatial_scale, mode, sample_ratio=-1):
    """ROI align operator in NCHW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, channel, height, width]

    rois : tvm.te.Tensor
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : int or list/tuple of two ints
        output size, or [out_height, out_width]

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    mode : int or str
        There are two modes, average and max. For the average mode, you can pass b'avg' or 0, and
        for the max mode, you can pass b'max' or 1.

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    avg_mode = mode in (b"avg", 0)
    max_mode = mode in (b"max", 1)
    assert avg_mode or max_mode, "Mode must be avg or max. Please pass in a valid mode."
    dtype = rois.dtype
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _bilinear(i, c, y, x):
        outside = tvm.tir.any(y < -1.0, x < -1.0, y > height, x > width)
        y = tvm.te.min(tvm.te.max(y, 0.0), height - 1)
        x = tvm.te.min(tvm.te.max(x, 0.0), width - 1)
        val = bilinear_sample_nchw(data, (i, c, y, x), height - 1, width - 1)
        return tvm.tir.if_then_else(outside, 0.0, val)

    def _sample(i, c, ph, pw):
        return _sample_common(
            i,
            c,
            ph,
            pw,
            rois,
            pooled_size_h,
            pooled_size_w,
            spatial_scale,
            sample_ratio,
            dtype,
            avg_mode,
            _bilinear,
        )

    return te.compute(
        (num_roi, channel, pooled_size_h, pooled_size_w), _sample, tag="pool,roi_align_nchw"
    )


def roi_align_nhwc(data, rois, pooled_size, spatial_scale, mode, sample_ratio=-1):
    """ROI align operator in NHWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, height, width, channel]

    rois : tvm.te.Tensor
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : int or list/tuple of two ints
        output size, or [out_height, out_width]

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    mode : int or str
        There are two modes, average and max. For the average mode, you can pass b'avg' or 0, and
        for the max mode, you can pass b'max' or 1.

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_roi, pooled_size, pooled_size, channel]
    """
    avg_mode = mode in (b"avg", 0)
    max_mode = mode in (b"max", 1)
    assert avg_mode or max_mode, "Mode must be avg or max. Please pass in a valid mode."
    dtype = rois.dtype
    _, height, width, channel = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _bilinear(i, c, y, x):
        outside = tvm.tir.any(y < -1.0, x < -1.0, y > height, x > width)
        y = tvm.te.min(tvm.te.max(y, 0.0), height - 1)
        x = tvm.te.min(tvm.te.max(x, 0.0), width - 1)
        val = bilinear_sample_nhwc(data, (i, y, x, c), height - 1, width - 1)
        return tvm.tir.if_then_else(outside, 0.0, val)

    def _sample(i, ph, pw, c):
        return _sample_common(
            i,
            c,
            ph,
            pw,
            rois,
            pooled_size_h,
            pooled_size_w,
            spatial_scale,
            sample_ratio,
            dtype,
            avg_mode,
            _bilinear,
        )

    return te.compute(
        (num_roi, pooled_size_h, pooled_size_w, channel), _sample, tag="pool,roi_align_nchw"
    )
