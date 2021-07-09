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
"""Upsampling in python"""
import math
import numpy as np
from tvm.topi.utils import nchw_pack_layout


def get_inx(x, image_width, target_width, coordinate_transformation_mode):
    """Infer input x from output x with various coordinate transformation methods"""
    scale = image_width / target_width
    if coordinate_transformation_mode == "half_pixel":
        in_x = (x + 0.5) * scale - 0.5
    elif coordinate_transformation_mode == "align_corners":
        in_x = (image_width - 1) / (target_width - 1) * x if target_width > 1 else 0
    elif coordinate_transformation_mode == "asymmetric":
        in_x = scale * x
    else:
        raise ValueError(
            "Unsupported coordinate_transformation_mode: {}".format(coordinate_transformation_mode)
        )
    return in_x


def get_index(x, image_width, target_width, coordinate_transformation_mode):
    """get and round the nearest index for nearest_neighbor"""
    in_x = get_inx(x, image_width, target_width, coordinate_transformation_mode)
    if coordinate_transformation_mode == "align_corners":
        # round prefer ceil
        out = int(math.floor(in_x + 0.5))
    else:
        out = int(math.floor(in_x))
    out = max(min(out, image_width - 1), 0)
    return out


def resize3d_nearest(arr, scale, coordinate_transformation_mode):
    """Populate the array by scale factor"""
    d, h, w = arr.shape
    out_d, out_h, out_w = [int(round(i * s)) for i, s in zip(arr.shape, scale)]
    out = np.empty((out_d, out_h, out_w))
    for z in range(out_d):
        for y in range(out_h):
            for x in range(out_w):
                in_z = get_index(z, d, out_d, coordinate_transformation_mode)
                in_y = get_index(y, h, out_h, coordinate_transformation_mode)
                in_x = get_index(x, w, out_w, coordinate_transformation_mode)
                out[z, y, x] = arr[in_z, in_y, in_x]
    return out


def resize3d_linear(data_in, scale, coordinate_transformation_mode):
    """Trilinear 3d scaling using python"""
    d, h, w = data_in.shape
    new_d, new_h, new_w = [int(round(i * s)) for i, s in zip(data_in.shape, scale)]
    data_out = np.ones((new_d, new_h, new_w))

    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    def _in_coord(new_coord, in_shape, out_shape):
        in_coord = get_inx(new_coord, in_shape, out_shape, coordinate_transformation_mode)
        coord0 = int(math.floor(in_coord))
        coord1 = max(min(coord0 + 1, in_shape - 1), 0)
        coord0 = max(coord0, 0)
        coord_lerp = in_coord - math.floor(in_coord)
        return coord0, coord1, coord_lerp

    for m in range(new_d):
        for j in range(new_h):
            for k in range(new_w):
                z0, z1, z_lerp = _in_coord(m, d, new_d)
                y0, y1, y_lerp = _in_coord(j, h, new_h)
                x0, x1, x_lerp = _in_coord(k, w, new_w)

                A0 = data_in[z0][y0][x0]
                B0 = data_in[z0][y0][x1]
                C0 = data_in[z0][y1][x0]
                D0 = data_in[z0][y1][x1]
                A1 = data_in[z1][y0][x0]
                B1 = data_in[z1][y0][x1]
                C1 = data_in[z1][y1][x0]
                D1 = data_in[z1][y1][x1]

                A = _lerp(A0, A1, z_lerp)
                B = _lerp(B0, B1, z_lerp)
                C = _lerp(C0, C1, z_lerp)
                D = _lerp(D0, D1, z_lerp)
                top = _lerp(A, B, x_lerp)
                bottom = _lerp(C, D, x_lerp)

                data_out[m][j][k] = np.float32(_lerp(top, bottom, y_lerp))

    return data_out


def resize3d_cubic(data_in, scale, coordinate_transformation_mode):
    """Tricubic 3d scaling using python"""
    d, h, w = data_in.shape
    new_d, new_h, new_w = [int(round(i * s)) for i, s in zip(data_in.shape, scale)]
    data_out = np.ones((new_d, new_h, new_w))

    def _cubic_spline_weights(t, alpha=-0.5):
        """create cubic spline weights in 1D"""
        t2 = t * t
        t3 = t * t * t
        w1 = alpha * (t3 - 2 * t2 + t)
        w2 = (alpha + 2) * t3 - (3 + alpha) * t2 + 1
        w3 = -(alpha + 2) * t3 + (3 + 2 * alpha) * t2 - alpha * t
        w4 = -alpha * t3 + alpha * t2
        return [w1, w2, w3, w4]

    def _cubic_kernel(inputs, w):
        """perform cubic interpolation in 1D"""
        return sum([a_i * w_i for a_i, w_i in zip(inputs, w)])

    def _get_input_value(z, y, x):
        z = max(min(z, d - 1), 0)
        y = max(min(y, h - 1), 0)
        x = max(min(x, w - 1), 0)
        return data_in[z][y][x]

    def _get_patch(zint, yint, xint):
        # Get the surrounding values
        p = [[[0 for i in range(4)] for j in range(4)] for k in range(4)]
        for kk in range(4):
            for jj in range(4):
                for ii in range(4):
                    p[kk][jj][ii] = _get_input_value(
                        zint + kk - 1,
                        yint + jj - 1,
                        xint + ii - 1,
                    )
        return p

    for m in range(new_d):
        for j in range(new_h):
            for k in range(new_w):
                in_z = get_inx(m, d, new_d, coordinate_transformation_mode)
                in_y = get_inx(j, h, new_h, coordinate_transformation_mode)
                in_x = get_inx(k, w, new_w, coordinate_transformation_mode)
                zint = math.floor(in_z)
                zfract = in_z - math.floor(in_z)

                yint = math.floor(in_y)
                yfract = in_y - math.floor(in_y)

                xint = math.floor(in_x)
                xfract = in_x - math.floor(in_x)

                wz = _cubic_spline_weights(zfract)
                wy = _cubic_spline_weights(yfract)
                wx = _cubic_spline_weights(xfract)

                p = _get_patch(zint, yint, xint)

                l = [[0 for i in range(4)] for j in range(4)]
                for jj in range(4):
                    for ii in range(4):
                        l[jj][ii] = _cubic_kernel(p[jj][ii], wx)

                col0 = _cubic_kernel(l[0], wy)
                col1 = _cubic_kernel(l[1], wy)
                col2 = _cubic_kernel(l[2], wy)
                col3 = _cubic_kernel(l[3], wy)
                data_out[m][j][k] = _cubic_kernel([col0, col1, col2, col3], wz)

    return data_out


def resize3d_ncdhw(
    data, scale, method="nearest_neighbor", coordinate_transformation_mode="align_corners"
):
    """reference kernel for 3D image resizing"""
    ishape = data.shape

    oshape = (
        ishape[0],
        ishape[1],
        int(round(ishape[2] * scale[0])),
        int(round(ishape[3] * scale[1])),
        int(round(ishape[4] * scale[2])),
    )

    output_np = np.zeros(oshape, dtype=data.dtype)

    for b in range(oshape[0]):
        for c in range(oshape[1]):
            if method == "nearest_neighbor":
                output_np[b, c, :, :, :] = resize3d_nearest(
                    data[b, c, :, :, :], scale, coordinate_transformation_mode
                )
            elif method == "linear":
                output_np[b, c, :, :, :] = resize3d_linear(
                    data[b, c, :, :, :], scale, coordinate_transformation_mode
                )
            elif method == "cubic":
                output_np[b, c, :, :, :] = resize3d_cubic(
                    data[b, c, :, :, :], scale, coordinate_transformation_mode
                )
            else:
                raise ValueError("Unknown resize method", method)

    return output_np


def resize1d_python(
    data,
    scale,
    layout="NCW",
    method="nearest_neighbor",
    coordinate_transformation_mode="align_corners",
):
    """Python version of 3D scaling using nearest neighbour"""

    if layout == "NWC":
        data = data.transpose([0, 2, 1])

    data = np.expand_dims(data, axis=[2, 3])
    output_np = resize3d_ncdhw(data, (1, 1) + scale, method, coordinate_transformation_mode)
    output_np = np.squeeze(output_np, axis=2)
    output_np = np.squeeze(output_np, axis=2)

    if layout == "NWC":
        output_np = output_np.transpose([0, 2, 1])

    return output_np


def resize2d_python(
    data,
    scale,
    layout="NCHW",
    method="nearest_neighbor",
    coordinate_transformation_mode="align_corners",
):
    """Python version of scaling using nearest neighbour"""

    if layout == "NHWC":
        data = data.transpose([0, 3, 1, 2])
    elif nchw_pack_layout(layout):
        ishape = data.shape
        transposed = data.transpose([0, 4, 1, 5, 2, 3])
        tshape = transposed.shape
        data = transposed.reshape(
            tshape[0] * tshape[1], tshape[2] * tshape[3], tshape[4], tshape[5]
        )

    data = np.expand_dims(data, axis=2)
    output_np = resize3d_ncdhw(data, (1,) + scale, method, coordinate_transformation_mode)
    output_np = np.squeeze(output_np, axis=2)

    if layout == "NHWC":
        output_np = output_np.transpose([0, 2, 3, 1])
    elif nchw_pack_layout(layout):
        output_np = output_np.reshape(tshape[0:4] + output_np.shape[2:])
        output_np = output_np.transpose([0, 2, 4, 5, 1, 3])

    return output_np


def resize3d_python(
    data,
    scale,
    layout="NCDHW",
    method="nearest_neighbor",
    coordinate_transformation_mode="align_corners",
):
    """Python version of 3D scaling using nearest neighbour"""

    if layout == "NDHWC":
        data = data.transpose([0, 4, 1, 2, 3])

    output_np = resize3d_ncdhw(data, scale, method, coordinate_transformation_mode)

    if layout == "NDHWC":
        output_np = output_np.transpose([0, 2, 3, 4, 1])

    return output_np
