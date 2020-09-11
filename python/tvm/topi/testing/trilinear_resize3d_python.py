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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals, too-many-nested-blocks
"""Trilinear 3D resize in python"""
import math
import numpy as np


def trilinear_resize3d_python(
    data_in, out_size, layout, coordinate_transformation_mode="align_corners"
):
    """ Trilinear 3d scaling using python"""
    (new_d, new_h, new_w) = out_size

    if layout == "NDHWC":
        (batch, d, h, w, channel) = data_in.shape
        data_out = np.ones((batch, new_d, new_h, new_w, channel))
    else:
        (batch, channel, d, h, w) = data_in.shape
        data_out = np.ones((batch, channel, new_d, new_h, new_w))

    if coordinate_transformation_mode == "align_corners":
        depth_scale = np.float32(d - 1) / np.float32(out_size[0] - 1)
        height_scale = np.float32(h - 1) / np.float32(out_size[1] - 1)
        width_scale = np.float32(w - 1) / np.float32(out_size[2] - 1)
    elif coordinate_transformation_mode in ["asymmetric", "half_pixel"]:
        depth_scale = np.float32(d) / np.float32(out_size[0])
        height_scale = np.float32(h) / np.float32(out_size[1])
        width_scale = np.float32(w) / np.float32(out_size[2])
    else:
        raise ValueError(
            "Unsupported coordinate_transformation_mode: {}".format(coordinate_transformation_mode)
        )

    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    def _in_coord(new_coord, scale, shape, mode):
        if mode == "half_pixel":
            in_coord = (new_coord + 0.5) * scale - 0.5
        else:
            in_coord = new_coord * scale
        coord0 = int(math.floor(in_coord))
        coord1 = max(min(coord0 + 1, shape - 1), 0)
        coord0 = max(coord0, 0)
        coord_lerp = in_coord - math.floor(in_coord)
        return coord0, coord1, coord_lerp

    for b in range(batch):
        for i in range(channel):
            for m in range(new_d):
                for j in range(new_h):
                    for k in range(new_w):
                        z0, z1, z_lerp = _in_coord(
                            m, depth_scale, d, coordinate_transformation_mode
                        )
                        y0, y1, y_lerp = _in_coord(
                            j, height_scale, h, coordinate_transformation_mode
                        )
                        x0, x1, x_lerp = _in_coord(
                            k, width_scale, w, coordinate_transformation_mode
                        )

                        if layout == "NDHWC":
                            A0 = data_in[b][z0][y0][x0][i]
                            B0 = data_in[b][z0][y0][x1][i]
                            C0 = data_in[b][z0][y1][x0][i]
                            D0 = data_in[b][z0][y1][x1][i]
                            A1 = data_in[b][z1][y0][x0][i]
                            B1 = data_in[b][z1][y0][x1][i]
                            C1 = data_in[b][z1][y1][x0][i]
                            D1 = data_in[b][z1][y1][x1][i]
                        else:
                            A0 = data_in[b][i][z0][y0][x0]
                            B0 = data_in[b][i][z0][y0][x1]
                            C0 = data_in[b][i][z0][y1][x0]
                            D0 = data_in[b][i][z0][y1][x1]
                            A1 = data_in[b][i][z1][y0][x0]
                            B1 = data_in[b][i][z1][y0][x1]
                            C1 = data_in[b][i][z1][y1][x0]
                            D1 = data_in[b][i][z1][y1][x1]

                        A = _lerp(A0, A1, z_lerp)
                        B = _lerp(B0, B1, z_lerp)
                        C = _lerp(C0, C1, z_lerp)
                        D = _lerp(D0, D1, z_lerp)
                        top = _lerp(A, B, x_lerp)
                        bottom = _lerp(C, D, x_lerp)

                        pixel = np.float32(_lerp(top, bottom, y_lerp))

                        if layout == "NDHWC":
                            data_out[b][m][j][k][i] = pixel
                        else:
                            data_out[b][i][m][j][k] = pixel

    return data_out
