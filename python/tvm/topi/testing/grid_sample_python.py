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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""affine_grid and grid_sample operators in python"""
import math
import numpy as np


def affine_grid_python(data, target_shape):
    yv, xv = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]))
    yv = yv.T * 2 / (target_shape[0] - 1) - 1
    xv = xv.T * 2 / (target_shape[1] - 1) - 1
    ones = np.ones_like(xv)
    grid = np.stack([xv, yv, ones]).reshape(3, -1)
    return data.reshape(-1, 3).dot(grid).reshape(data.shape[0], 2, *target_shape)


def _bilinear_sample_nchw_python(data, grid):
    batch, in_channel, in_height, in_width = data.shape
    _, _, out_height, out_width = grid.shape
    out = np.zeros((batch, in_channel, out_height, out_width), dtype=data.dtype)

    def _within_bound(y, x):
        return 0 <= y < in_height and 0 <= x < in_width

    for n in range(0, batch):
        for h in range(0, out_height):
            for w in range(0, out_width):
                x, y = grid[n, :, h, w]
                y = (y + 1) * (in_height - 1) / 2
                x = (x + 1) * (in_width - 1) / 2
                y0 = int(math.floor(y))
                x0 = int(math.floor(x))
                y1 = y0 + 1
                x1 = x0 + 1
                if _within_bound(y0, x0):
                    out[n, :, h, w] += data[n, :, y0, x0] * (1.0 - (y - y0)) * (1.0 - (x - x0))
                if _within_bound(y0, x1):
                    out[n, :, h, w] += data[n, :, y0, x1] * (1.0 - (y - y0)) * (x - x0)
                if _within_bound(y1, x0):
                    out[n, :, h, w] += data[n, :, y1, x0] * (y - y0) * (1.0 - (x - x0))
                if _within_bound(y1, x1):
                    out[n, :, h, w] += data[n, :, y1, x1] * (y - y0) * (x - x0)
    return out


def grid_sample_nchw_python(data, grid, method="bilinear"):
    if method == "bilinear":
        return _bilinear_sample_nchw_python(data, grid)
    raise ValueError("invalid method")
