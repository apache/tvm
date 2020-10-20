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
"Roi pool in python"
import math
import numpy as np


def roi_pool_nchw_python(a_np, rois_np, pooled_size, spatial_scale):
    """Roi pool in python"""
    _, channel, height, width = a_np.shape
    num_roi = rois_np.shape[0]
    b_np = np.zeros((num_roi, channel, pooled_size, pooled_size), dtype=a_np.dtype)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    for i in range(num_roi):
        roi = rois_np[i]
        batch_index = int(roi[0])
        roi_start_w = int(round(roi[1] * spatial_scale))
        roi_start_h = int(round(roi[2] * spatial_scale))
        roi_end_w = int(round(roi[3] * spatial_scale))
        roi_end_h = int(round(roi[4] * spatial_scale))
        roi_h = max(roi_end_h - roi_start_h + 1, 1)
        roi_w = max(roi_end_w - roi_start_w + 1, 1)

        bin_h = float(roi_h) / pooled_size_h
        bin_w = float(roi_w) / pooled_size_w

        for ph in range(pooled_size_h):
            for pw in range(pooled_size_w):
                hstart = int(math.floor(ph * bin_h))
                wstart = int(math.floor(pw * bin_w))
                hend = int(math.ceil((ph + 1) * bin_h))
                wend = int(math.ceil((pw + 1) * bin_w))
                hstart = min(max(hstart + roi_start_h, 0), height)
                hend = min(max(hend + roi_start_h, 0), height)
                wstart = min(max(wstart + roi_start_w, 0), width)
                wend = min(max(wend + roi_start_w, 0), width)
                is_empty = (hend <= hstart) or (wend <= wstart)

                for c in range(channel):
                    if is_empty:
                        b_np[i, c, ph, pw] = 0.0
                    else:
                        b_np[i, c, ph, pw] = np.max(a_np[batch_index, c, hstart:hend, wstart:wend])
    return b_np
