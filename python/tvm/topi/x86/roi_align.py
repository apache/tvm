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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements
"""Non-maximum suppression operator for intel cpu"""
import math

import tvm
from tvm.te import hybrid
from ..tensor import full
from ..utils import get_const_tuple


@hybrid.script
def roi_align_nchw_ir(
    data, rois, num_rois, w_pc, pos_pc, pooled_size, spatial_scale, sample_ratio, mode
):
    """Hybrid routing fo ROI align operator in NCHW layout.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        4-D with shape [batch, channel, height, width]

    rois : tvm.te.Tensor or numpy NDArray
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    num_rois : tvm.tir.IntImm or tvm.tir.Var
        Number of roi. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    w_pc : tvm.te.Tensor or numpy NDArray
        3-D weight pre-calculation buffer

    pos_pc : tvm.te.Tensor or numpy NDArray
        3-D position pre-calculation buffer

    pooled_size : tvm ConsExpr
        [out_height, out_width]

    spatial_scale : tvm.tir.const
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    sample_ratio : tvm.tir.const
        Sampling ratio of ROI align, using adaptive size by default.

    mode : tvm.tir.const
        Mode of RoiAlign. A value of 0 corrensponds to b'avg', while a value of 1 corresponds to
        b'max'.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    channels = data.shape[1]
    height = data.shape[2]
    width = data.shape[3]
    pooled_size_h = pooled_size[0]
    pooled_size_w = pooled_size[1]
    output = output_tensor((num_rois, channels, pooled_size_h, pooled_size_w), data.dtype)

    for n in parallel(num_rois):
        roi_batch_index = int32(rois[n, 0])
        roi_start_w = rois[n, 1] * spatial_scale
        roi_start_h = rois[n, 2] * spatial_scale
        roi_end_w = rois[n, 3] * spatial_scale
        roi_end_h = rois[n, 4] * spatial_scale

        roi_h = max(roi_end_h - roi_start_h, 1.0)
        roi_w = max(roi_end_w - roi_start_w, 1.0)

        bin_h = roi_h / pooled_size_h
        bin_w = roi_w / pooled_size_w

        roi_bin_grid_h = sample_ratio
        roi_bin_grid_w = roi_bin_grid_h
        rounded_bin_h = int32(bin_h) * 1.0
        rounded_bin_w = int32(bin_w) * 1.0
        if sample_ratio <= 0:
            # Cannot use ceil function since hybrid script
            # doesn't support Call as indexing
            roi_bin_grid_h = int32(bin_h)
            roi_bin_grid_w = int32(bin_w)
            if rounded_bin_h < bin_h:
                roi_bin_grid_h += 1
            if rounded_bin_w < bin_w:
                roi_bin_grid_w += 1

        count = roi_bin_grid_h * roi_bin_grid_w

        # Pre-calculate indices and weights shared by all channels.
        # This is the key point of optimization.
        pre_calc_index = 0
        iy_upper = roi_bin_grid_h
        ix_upper = roi_bin_grid_w
        for ph in range(pooled_size_h):
            for pw in range(pooled_size_w):
                for iy in range(iy_upper):
                    yy = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / roi_bin_grid_h
                    for ix in range(ix_upper):
                        xx = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / roi_bin_grid_w
                        x = xx
                        y = yy
                        if y < -1.0 or y > height or x < -1.0 or x > width:
                            for i in range(4):
                                w_pc[n, pre_calc_index, i] = 0.0
                                pos_pc[n, pre_calc_index, i] = 0
                        else:
                            if y < 0.0:
                                y = 0.0
                            if x < 0.0:
                                x = 0.0

                            y_low = int32(y)
                            x_low = int32(x)
                            x_high = x_low + 1
                            y_high = y_low + 1

                            if y_low >= height - 1:
                                y_high = height - 1
                                y_low = y_high
                                y = float32(y_low)

                            if x_low >= width - 1:
                                x_high = width - 1
                                x_low = x_high
                                x = float32(x_low)

                            ly = y - y_low
                            lx = x - x_low
                            hy = 1.0 - ly
                            hx = 1.0 - lx
                            w1 = hy * hx
                            w2 = hy * lx
                            w3 = ly * hx
                            w4 = ly * lx

                            pos_pc[n, pre_calc_index, 0] = x_low
                            pos_pc[n, pre_calc_index, 1] = x_high
                            pos_pc[n, pre_calc_index, 2] = y_low
                            pos_pc[n, pre_calc_index, 3] = y_high
                            w_pc[n, pre_calc_index, 0] = w1
                            w_pc[n, pre_calc_index, 1] = w2
                            w_pc[n, pre_calc_index, 2] = w3
                            w_pc[n, pre_calc_index, 3] = w4

                        pre_calc_index += 1

        for c in range(channels):
            pre_calc_index = 0
            for ph in range(pooled_size_h):
                for pw in range(pooled_size_w):
                    output_val = 0.0  # Avg mode
                    if mode == 1:  # Max mode
                        output_val = ninf("float32")
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            bilinear_val = (
                                w_pc[n, pre_calc_index, 0]
                                * data[
                                    roi_batch_index,
                                    c,
                                    pos_pc[n, pre_calc_index, 2],
                                    pos_pc[n, pre_calc_index, 0],
                                ]
                                + w_pc[n, pre_calc_index, 1]
                                * data[
                                    roi_batch_index,
                                    c,
                                    pos_pc[n, pre_calc_index, 2],
                                    pos_pc[n, pre_calc_index, 1],
                                ]
                                + w_pc[n, pre_calc_index, 2]
                                * data[
                                    roi_batch_index,
                                    c,
                                    pos_pc[n, pre_calc_index, 3],
                                    pos_pc[n, pre_calc_index, 0],
                                ]
                                + w_pc[n, pre_calc_index, 3]
                                * data[
                                    roi_batch_index,
                                    c,
                                    pos_pc[n, pre_calc_index, 3],
                                    pos_pc[n, pre_calc_index, 1],
                                ]
                            )
                            pre_calc_index += 1
                            if mode == 0:  # Avg mode
                                output_val += bilinear_val / count
                            if mode == 1:  # Max mode
                                output_val = max(output_val, bilinear_val)
                        output[n, c, ph, pw] = output_val
    return output


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

    mode : str
        Mode of RoiAlign. Should be b'max' or b'avg'.

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    if not isinstance(pooled_size, (tuple, list)):
        pooled_size = (pooled_size, pooled_size)

    # Pre-allocate intermediate buffer
    if sample_ratio > 0:
        max_roi_bin_grid_w = max_roi_bin_grid_h = sample_ratio
    else:
        _, _, height, width = get_const_tuple(data.shape)
        max_roi_bin_grid_h = math.ceil(height / pooled_size[0])
        max_roi_bin_grid_w = math.ceil(width / pooled_size[1])
    num_rois = rois.shape[0]
    max_pc_shape = (
        rois.shape[0],
        max_roi_bin_grid_h * max_roi_bin_grid_w * pooled_size[0] * pooled_size[1],
        4,
    )
    w_pc_buffer = full(max_pc_shape, data.dtype, 0)
    pos_pc_buffer = full(max_pc_shape, "int32", 0)

    pooled_size = tvm.runtime.convert(pooled_size)
    spatial_scale = tvm.tir.const(spatial_scale, "float32")
    sample_ratio = tvm.tir.const(sample_ratio, "int32")
    if mode in (b"avg", 0):
        mode = tvm.tir.const(0, dtype="float32")
    elif mode in (b"max", 1):
        mode = tvm.tir.const(1, dtype="float32")
    else:
        raise ValueError(mode, "Value %s passed in for mode not supported", mode)

    return roi_align_nchw_ir(
        data,
        rois,
        num_rois,
        w_pc_buffer,
        pos_pc_buffer,
        pooled_size,
        spatial_scale,
        sample_ratio,
        mode,
    )
