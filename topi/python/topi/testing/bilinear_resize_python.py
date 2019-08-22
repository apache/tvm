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

def bilinear_resize_python(image, out_size, layout, align_corners=True):
    """ Bilinear scaling using python"""
    (new_h, new_w) = out_size

    if layout == 'NHWC':
        (batch, h, w, channel) = image.shape
        scaled_image = np.ones((batch, new_h, new_w, channel))
    else:
        (batch, channel, h, w) = image.shape
        scaled_image = np.ones((batch, channel, new_h, new_w))

    if align_corners:
        height_scale = np.float32(h-1) / np.float32(out_size[0]-1)
        width_scale = np.float32(w-1) / np.float32(out_size[1]-1)
    else:
        height_scale = np.float32(h) / np.float32(out_size[0])
        width_scale = np.float32(w) / np.float32(out_size[1])

    for b in range(batch):
        for i in range(channel):
            for j in range(new_h):
                for k in range(new_w):
                    in_y = j * height_scale
                    y0 = math.floor(in_y)
                    y1 = min(math.ceil(in_y), h - 1)
                    y_lerp = in_y - y0

                    y0 = int(y0)
                    y1 = int(y1)

                    in_x = k * width_scale
                    x0 = math.floor(in_x)
                    x1 = min(math.ceil(in_x), w - 1)
                    x_lerp = in_x - x0

                    x0 = int(x0)
                    x1 = int(x1)

                    if layout == 'NHWC':
                        A = image[b][y0][x0][i]
                        B = image[b][y0][x1][i]
                        C = image[b][y1][x0][i]
                        D = image[b][y1][x1][i]
                    else:
                        A = image[b][i][y0][x0]
                        B = image[b][i][y0][x1]
                        C = image[b][i][y1][x0]
                        D = image[b][i][y1][x1]

                    top = A + (B - A) * x_lerp
                    bottom = C + (D - C) * x_lerp

                    pixel = np.float32(top + (bottom - top) * y_lerp)

                    if layout == 'NHWC':
                        scaled_image[b][j][k][i] = pixel
                    else:
                        scaled_image[b][i][j][k] = pixel

    return scaled_image
