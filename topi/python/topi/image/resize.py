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
"""TVM operator input resize compute."""
from __future__ import absolute_import
import tvm
from .. import tag


def resize(data, size, layout="NCHW", method="bilinear", align_corners=True, out_dtype=None):
    """Perform resize operation on the data.

    Parameters
    ----------
    inputs : tvm.Tensor
        inputs is a 4-D tensor with shape
        [batch, channel, in_height, in_width]
        or  [batch, in_height, in_width, channel]

    size: Tuple
        Output resolution scale to

    layout: string, optional
        "NCHW", "NHWC", or "NCHWc".

    align_corners: Boolean, optional
        To preserve the values at the corner pixels.

    method: {"bilinear", "nearest_neighbor", "bicubic"}
        Method to be used for resizing.

    out_dtype: string, optional
        Type to return. If left None will be same as input type.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, channel, in_height*scale, in_width*scale]
        or [batch, in_height*scale, in_width*scale, channel]
        or 5-D with shape [batch, channel-major, in_height*scale, in_width*scale, channel-minor]
    """
    method = method.lower()

    if layout == 'NHWC':
        in_n, in_h, in_w, in_c = data.shape
        output_shape = [in_n, size[0], size[1], in_c]
    elif layout == 'NCHW':
        in_n, in_c, in_h, in_w = data.shape
        output_shape = [in_n, in_c, size[0], size[1]]
    # Otherwise layout must be NCHWxc
    else:
        in_n, in_c, in_h, in_w, in_cc = data.shape
        output_shape = [in_n, in_c, size[0], size[1], in_cc]

    if align_corners:
        y_ratio = (in_h - 1).astype('float') / (size[0] - 1)
        x_ratio = (in_w - 1).astype('float') / (size[1] - 1)
    else:
        y_ratio = (in_h).astype('float') / (size[0])
        x_ratio = (in_w).astype('float') / (size[1])

    def _get_pixel(n, c, y, x, cc):
        y = tvm.max(tvm.min(y, in_h - 1), 0)
        x = tvm.max(tvm.min(x, in_w - 1), 0)
        if layout == 'NHWC':
            return data(n, y, x, c).astype('float')
        if layout == 'NCHW':
            return data(n, c, y, x).astype('float')
        # else must be NCHWxc
        return data(n, c, y, x, cc).astype('float')

    def _get_indices(*indices):
        if layout == 'NHWC':
            n, y, x, c = indices
            cc = None
        elif layout == 'NCHW':
            n, c, y, x = indices
            cc = None
        else:
            n, c, y, x, cc = indices

        return n, c, y, x, cc

    def _cast_output(value):
        if out_dtype:
            dtype = out_dtype
        else:
            dtype = data.dtype
        return value.astype(dtype)

    # Nearest neighbor computation
    def _nearest_neighbor(*indices):
        n, c, y, x, cc = _get_indices(*indices)

        in_y = y_ratio * y
        in_x = x_ratio * x

        if align_corners:
            yint = tvm.round(in_y).astype('int32')
            xint = tvm.round(in_x).astype('int32')
        else:
            # Add epsilon to floor to prevent gpu rounding errors.
            epsilon = 1e-5
            yint = tvm.floor(in_y + epsilon).astype('int32')
            xint = tvm.floor(in_x + epsilon).astype('int32')

        return _cast_output(_get_pixel(n, c, yint, xint, cc))

    # Bilinear helper functions and computation.
    def _lerp(A, B, t):
        return A * (1.0 - t) + B * t

    def _bilinear(*indices):
        n, c, y, x, cc = _get_indices(*indices)

        in_y = y_ratio * y
        in_x = x_ratio * x

        xint = tvm.floor(in_x).astype('int32')
        xfract = in_x - tvm.floor(in_x)

        yint = tvm.floor(in_y).astype('int32')
        yfract = in_y - tvm.floor(in_y)

        p00 = _get_pixel(n, c, yint, xint, cc)
        p10 = _get_pixel(n, c, yint, xint + 1, cc)
        p01 = _get_pixel(n, c, yint + 1, xint, cc)
        p11 = _get_pixel(n, c, yint + 1, xint + 1, cc)

        col0 = _lerp(p00, p10, xfract)
        col1 = _lerp(p01, p11, xfract)
        value = _lerp(col0, col1, yfract)
        return _cast_output(value)

    # Bicubic helper function and computation.
    def _cubic_kernel(A, B, C, D, t):
        a = -A / 2.0 + (3.0*B) / 2.0 - (3.0*C) / 2.0 + D / 2.0
        b = A - (5.0*B) / 2.0 + 2.0*C - D / 2.0
        c = -A / 2.0 + C / 2.0
        d = B

        return a*t*t*t + b*t*t + c*t + d

    def _bicubic(*indices):
        n, c, y, x, cc = _get_indices(*indices)

        in_y = y_ratio * y
        in_x = x_ratio * x

        xint = tvm.floor(in_x).astype('int32')
        xfract = in_x - tvm.floor(in_x)

        yint = tvm.floor(in_y).astype('int32')
        yfract = in_y - tvm.floor(in_y)

        # 1st row
        p00 = _get_pixel(n, c, yint - 1, xint - 1, cc)
        p10 = _get_pixel(n, c, yint - 1, xint + 0, cc)
        p20 = _get_pixel(n, c, yint - 1, xint + 1, cc)
        p30 = _get_pixel(n, c, yint - 1, xint + 2, cc)

        # 2nd row
        p01 = _get_pixel(n, c, yint + 0, xint - 1, cc)
        p11 = _get_pixel(n, c, yint + 0, xint + 0, cc)
        p21 = _get_pixel(n, c, yint + 0, xint + 1, cc)
        p31 = _get_pixel(n, c, yint + 0, xint + 2, cc)

        # 3rd row
        p02 = _get_pixel(n, c, yint + 1, xint - 1, cc)
        p12 = _get_pixel(n, c, yint + 1, xint + 0, cc)
        p22 = _get_pixel(n, c, yint + 1, xint + 1, cc)
        p32 = _get_pixel(n, c, yint + 1, xint + 2, cc)

        # 4th row
        p03 = _get_pixel(n, c, yint + 2, xint - 1, cc)
        p13 = _get_pixel(n, c, yint + 2, xint + 0, cc)
        p23 = _get_pixel(n, c, yint + 2, xint + 1, cc)
        p33 = _get_pixel(n, c, yint + 2, xint + 2, cc)

        # Interpolate bicubically
        col0 = _cubic_kernel(p00, p10, p20, p30, xfract)
        col1 = _cubic_kernel(p01, p11, p21, p31, xfract)
        col2 = _cubic_kernel(p02, p12, p22, p32, xfract)
        col3 = _cubic_kernel(p03, p13, p23, p33, xfract)
        value = _cubic_kernel(col0, col1, col2, col3, yfract)
        return _cast_output(value)

    # Determine which interpolation method to use then run it.
    if method == "nearest_neighbor":
        compute_func = _nearest_neighbor
    elif method == "bilinear":
        compute_func = _bilinear
    elif method == "bicubic":
        compute_func = _bicubic
    else:
        raise ValueError('%s method is not supported.' % method)

    return tvm.compute(output_shape, compute_func, name='resize', tag=tag.INJECTIVE)
