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
"""Bilinear Scale in python"""
import math
import numpy as np
from tvm.topi.util import nchw_pack_layout


def bilinear_resize_python(image, out_size, layout, coordinate_transformation_mode="align_corners"):
    """ Bilinear scaling using python"""
    (new_h, new_w) = out_size
    (ib, ic) = (1, 1)

    if layout == "NHWC":
        (batch, h, w, channel) = image.shape
        scaled_image = np.ones((batch, new_h, new_w, channel))
    # NCHWinic
    elif nchw_pack_layout(layout):
        (batch, channel, h, w, ib, ic) = image.shape
        scaled_image = np.ones((batch, channel, new_h, new_w, ib, ic))
    else:
        (batch, channel, h, w) = image.shape
        scaled_image = np.ones((batch, channel, new_h, new_w))

    if coordinate_transformation_mode == "align_corners":
        height_scale = np.float32(h - 1) / np.float32(out_size[0] - 1)
        width_scale = np.float32(w - 1) / np.float32(out_size[1] - 1)
    else:
        height_scale = np.float32(h) / np.float32(out_size[0])
        width_scale = np.float32(w) / np.float32(out_size[1])

    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    def _img_scale(b, m, i, n):
        for j in range(new_h):
            for k in range(new_w):
                if coordinate_transformation_mode == "half_pixel":
                    in_y = (j + 0.5) * height_scale - 0.5
                else:
                    in_y = j * height_scale
                y0 = int(math.floor(in_y))
                y1 = max(min(y0 + 1, h - 1), 0)
                y0 = max(y0, 0)
                y_lerp = in_y - math.floor(in_y)

                if coordinate_transformation_mode == "half_pixel":
                    in_x = (k + 0.5) * width_scale - 0.5
                else:
                    in_x = k * width_scale
                x0 = int(math.floor(in_x))
                x1 = max(min(x0 + 1, w - 1), 0)
                x0 = max(x0, 0)
                x_lerp = in_x - math.floor(in_x)

                if layout == "NHWC":
                    A = image[b][y0][x0][i]
                    B = image[b][y0][x1][i]
                    C = image[b][y1][x0][i]
                    D = image[b][y1][x1][i]
                elif nchw_pack_layout(layout):
                    A = image[b][i][y0][x0][m][n]
                    B = image[b][i][y0][x1][m][n]
                    C = image[b][i][y1][x0][m][n]
                    D = image[b][i][y1][x1][m][n]
                else:
                    A = image[b][i][y0][x0]
                    B = image[b][i][y0][x1]
                    C = image[b][i][y1][x0]
                    D = image[b][i][y1][x1]

                top = _lerp(A, B, x_lerp)
                bottom = _lerp(C, D, x_lerp)

                pixel = np.float32(_lerp(top, bottom, y_lerp))

                if layout == "NHWC":
                    scaled_image[b][j][k][i] = pixel
                elif nchw_pack_layout(layout):
                    scaled_image[b][i][j][k][m][n] = pixel
                else:
                    scaled_image[b][i][j][k] = pixel

    for b in range(batch):
        for m in range(ib):
            for i in range(channel):
                for n in range(ic):
                    _img_scale(b, m, i, n)

    return scaled_image
