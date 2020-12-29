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
"""ROI pool operator"""
import tvm
from tvm import te
from ...utils import get_const_tuple


def roi_pool_nchw(data, rois, pooled_size, spatial_scale):
    """ROI pool operator in NCHW layout.

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

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    dtype = rois.dtype
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _pool(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype("int32")
        roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1], roi[2], roi[3], roi[4]

        roi_start_h = te.round(roi_start_h * spatial_scale).astype("int32")
        roi_start_w = te.round(roi_start_w * spatial_scale).astype("int32")
        roi_end_h = te.round(roi_end_h * spatial_scale).astype("int32")
        roi_end_w = te.round(roi_end_w * spatial_scale).astype("int32")

        # force malformed ROIs to be 1x1
        roi_h = tvm.te.max(roi_end_h - roi_start_h + 1, tvm.tir.const(1, "int32"))
        roi_w = tvm.te.max(roi_end_w - roi_start_w + 1, tvm.tir.const(1, "int32"))

        bin_h = roi_h.astype(dtype) / pooled_size_h
        bin_w = roi_w.astype(dtype) / pooled_size_w

        # use epsilon to prevent floating point precision loss in floor/ceil
        epsilon = tvm.tir.const(0.00001, dtype)
        hstart = te.floor(ph * bin_h + epsilon).astype("int32")
        wstart = te.floor(pw * bin_w + epsilon).astype("int32")
        hend = te.ceil((ph + 1) * bin_h - epsilon).astype("int32")
        wend = te.ceil((pw + 1) * bin_w - epsilon).astype("int32")
        hstart = tvm.te.min(tvm.te.max(hstart + roi_start_h, 0), height)
        wstart = tvm.te.min(tvm.te.max(wstart + roi_start_w, 0), width)
        hend = tvm.te.min(tvm.te.max(hend + roi_start_h, 0), height)
        wend = tvm.te.min(tvm.te.max(wend + roi_start_w, 0), width)

        non_empty = tvm.tir.all(hstart < hend, wstart < wend)
        min_value = lambda dtype: tvm.tir.if_then_else(
            non_empty, tvm.te.min_value(dtype), tvm.tir.const(0.0, dtype)
        )
        # pylint: disable=unnecessary-lambda
        _max = te.comm_reducer(lambda x, y: tvm.te.max(x, y), min_value, name="max")
        rh = te.reduce_axis((0, hend - hstart), "rh")
        rw = te.reduce_axis((0, wend - wstart), "rw")
        return _max(data[batch_index, c, hstart + rh, wstart + rw], axis=[rh, rw])

    return te.compute((num_roi, channel, pooled_size_h, pooled_size_w), _pool, tag="pool,roi_pool")
