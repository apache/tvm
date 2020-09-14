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
"""affine_grid and grid_sample operator"""
from tvm import te, tir


def affine_grid(data, target_shape):
    """affine_grid operator that generates 2D sampling grid.

    This operation is described in https://arxiv.org/pdf/1506.02025.pdf. It generates a uniform
    sampling grid within the target shape and normalizes it to [-1, 1]. The provided affine
    transformation is then applied on the sampling grid.

    Parameters
    ----------
    data : tvm.Tensor
        3-D with shape [batch, 2, 3]. The affine matrix.

    target_shape: list/tuple of two int
        Specifies the output shape (H, W).

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, 2, target_height, target_width]
    """
    assert target_shape is not None
    assert len(target_shape) == 2
    assert (
        target_shape[0] > 1 and target_shape[1] > 1
    ), "target height/width should be greater than 1"

    dtype = data.dtype
    y_step = tir.const((2.0 - 1e-7) / (target_shape[0] - 1), dtype=dtype)
    x_step = tir.const((2.0 - 1e-7) / (target_shape[1] - 1), dtype=dtype)
    start = tir.const(-1.0, dtype=dtype)

    def _compute(n, dim, i, j):
        y = start + i * y_step
        x = start + j * x_step
        return data[n, dim, 0] * x + data[n, dim, 1] * y + data[n, dim, 2]

    oshape = (data.shape[0], len(target_shape), *target_shape)
    return te.compute(oshape, _compute, tag="affine_grid")


def grid_sample(data, grid, method="bilinear", layout="NCHW"):
    """Applies bilinear sampling to input feature map.

    Given :math:`data` and :math:`grid`, assuming NCHW layout, then the output is computed by

    .. math::

        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})

    :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and
    :math:`G()` denotes the interpolation method.
    The out-boundary points will be padded with zeros. The shape of the output will be
    (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).

    The operator assumes that :math:`grid` has been normalized to [-1, 1].

    grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    grid : tvm.Tensor
        4-D with shape [batch, 2, out_height, out_width]

    method : str
        The interpolation method. Only 'bilinear' is supported.

    layout : str
        The layout of input data and the output.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, 2, out_height, out_width]
    """
    batch, in_channel, in_height, in_width = data.shape
    out_height, out_width = grid.shape[2:]
    assert method == "bilinear", "Only bilinear is supported"
    assert layout == "NCHW", "Only NCHW is supported"

    def _get_pixel_value(n, c, h, w):
        return te.if_then_else(
            te.all(h >= 0, w >= 0, h < in_height, w < in_width),
            data[n, c, h, w],
            tir.const(0.0, dtype=data.dtype),
        )

    def _bilinear_sample(n, c, h, w):
        x = grid[n, 0, h, w]
        y = grid[n, 1, h, w]
        y = (y + 1) * (in_height - 1) / 2
        x = (x + 1) * (in_width - 1) / 2
        x0 = te.floor(x).astype("int32")
        y0 = te.floor(y).astype("int32")
        x1 = x0 + tir.const(1, "int32")
        y1 = y0 + tir.const(1, "int32")
        return (
            _get_pixel_value(n, c, y0, x0) * (1.0 - (y - y0)) * (1.0 - (x - x0))
            + _get_pixel_value(n, c, y0, x1) * (1.0 - (y - y0)) * (x - x0)
            + _get_pixel_value(n, c, y1, x0) * (y - y0) * (1.0 - (x - x0))
            + _get_pixel_value(n, c, y1, x1) * (y - y0) * (x - x0)
        )

    return te.compute(
        (batch, in_channel, out_height, out_width), _bilinear_sample, tag="grid_sample"
    )
