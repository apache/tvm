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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin, no-else-return
"""Conv3D operators"""
import tvm
from tvm import te

from ..utils import get_const_tuple
from .winograd_util import winograd_transform_matrices
from .conv2d import conv


def conv3d_ncdhw(Input, Filter, stride, padding, dilation, groups, out_dtype=None):
    """Conv3D operator in NCDHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    Filter : tvm.te.Tensor
        5-D with shape [num_filter, in_channel, filter_depth, filter_height, filter_width]

    stride : int or a list/tuple of three ints
        Stride size, or [strid_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups.

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_channel, out_depth, out_height, out_width]
    """
    return conv(Input, Filter, stride, padding, dilation, groups, "NCDHW", out_dtype)


def conv3d_ndhwc(
    Input,
    Filter,
    stride,
    padding,
    dilation,
    groups,
    out_dtype="float32",
    auto_scheduler_rewritten_layout="",
):
    """Convolution operator in NDHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        5-D with shape [batch, in_depth, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        5-D with shape [filter_depth, filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of three ints
        Stride size, or [stride_depth, stride_height, stride_width]

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of three ints
        dilation size, or [dilation_depth, dilation_height, dilation_width]

    groups: int
        Number of groups.

    out_dtype: str = "float32",
        The type of output tensor

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    Output : tvm.te.Tensor
        5-D with shape [batch, out_depth, out_height, out_width, out_channel]
    """
    return conv(
        Input,
        Filter,
        stride,
        padding,
        dilation,
        groups,
        "NDHWC",
        out_dtype,
        auto_scheduler_rewritten_layout,
    )


def conv3d_winograd_weight_transform(kernel, tile_size):
    """Weight transformation for 3D winograd

    Parameters
    ----------
    kernel: Tensor
        The raw kernel tensor with layout "NCDHW".
    tile_size: int
        Tile size of winograd transform. e.g. 2 for F(2x2, 3x3) and 4 for F(4x4, 3x3)

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [alpha, alpha, alpha, CO, CI]
    """
    CO, CI, KD, KH, KW = get_const_tuple(kernel.shape)

    depth_transform = 2 < KD < 8 and KD == KH

    if depth_transform:
        assert KD == KH == KW, "Only support NxNxN kernel"
    else:
        assert KH == KW, "Only supports DxNxN kernel"

    r = tile_size + KH - 1

    r_kh = te.reduce_axis((0, KH), name="r_kh")
    r_kw = te.reduce_axis((0, KW), name="r_kw")
    _, _, G = winograd_transform_matrices(tile_size, KH, kernel.dtype)
    if depth_transform:
        shape = (r, r, r, CO, CI)
        r_kd = te.reduce_axis((0, KD), name="r_kd")
        return te.compute(
            shape,
            lambda omg, eps, nu, co, ci: te.sum(
                kernel[co][ci][r_kd][r_kh][r_kw] * G[omg][r_kd] * G[eps][r_kh] * G[nu][r_kw],
                axis=[r_kd, r_kh, r_kw],
            ),
            name="transform_weight",
        )
    else:
        shape = (r, r, KD, CO, CI)
        return te.compute(
            shape,
            lambda eps, nu, d, co, ci: te.sum(
                kernel[co][ci][d][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="transform_weight",
        )


@tvm.target.generic_func
def conv3d_alter_layout(attrs, inputs, tinfos, out_type):
    """Change Conv3D layout.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : tvm.relay.Expr
        Grouped input symbols
    tinfos : list
        Input shape and dtype
    out_type: type
        The output type

    Note
    ----
    Unlike other TOPI functions, this function operates on both graph level and operator level.
    """
    # not to change by default
    return None
