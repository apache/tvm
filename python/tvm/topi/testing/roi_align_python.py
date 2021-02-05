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
# pylint: disable=invalid-name, too-many-nested-blocks
"Roi align in python"
import math
import numpy as np


def roi_align_nchw_python(a_np, rois_np, pooled_size, spatial_scale, sample_ratio):
    """Roi align in python"""
    _, channel, height, width = a_np.shape
    num_roi = rois_np.shape[0]
    b_np = np.zeros((num_roi, channel, pooled_size, pooled_size), dtype=a_np.dtype)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _bilinear(n, c, y, x):
        if y < -1 or y > height or x < -1 or x > width:
            return 0

        y = min(max(y, 0), height - 1)
        x = min(max(x, 0), width - 1)

        y_low = int(math.floor(y))
        x_low = int(math.floor(x))
        y_high = y_low + 1
        x_high = x_low + 1

        wy_h = y - y_low
        wx_h = x - x_low
        wy_l = 1 - wy_h
        wx_l = 1 - wx_h

        val = 0
        for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
            for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
                if 0 <= yp < height and 0 <= xp < width:
                    val += wx * wy * a_np[n, c, yp, xp]
        return val

    for i in range(num_roi):
        roi = rois_np[i]
        batch_index = int(roi[0])
        roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1:] * spatial_scale
        roi_h = max(roi_end_h - roi_start_h, 1.0)
        roi_w = max(roi_end_w - roi_start_w, 1.0)

        bin_h = roi_h / pooled_size_h
        bin_w = roi_w / pooled_size_w

        if sample_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = int(sample_ratio)
        else:
            roi_bin_grid_h = int(math.ceil(roi_h / pooled_size))
            roi_bin_grid_w = int(math.ceil(roi_w / pooled_size))

        count = roi_bin_grid_h * roi_bin_grid_w

        for c in range(channel):
            for ph in range(pooled_size_h):
                for pw in range(pooled_size_w):
                    total = 0.0
                    for iy in range(roi_bin_grid_h):
                        for ix in range(roi_bin_grid_w):
                            y = roi_start_h + ph * bin_h + (iy + 0.5) * bin_h / roi_bin_grid_h
                            x = roi_start_w + pw * bin_w + (ix + 0.5) * bin_w / roi_bin_grid_w
                            total += _bilinear(batch_index, c, y, x)
                    b_np[i, c, ph, pw] = total / count
    return b_np
