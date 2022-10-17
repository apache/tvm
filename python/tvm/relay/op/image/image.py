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
"""Image operations."""
from . import _make
from ..dyn.image import _make as _dyn_make
from ...expr import Expr, Constant, const


def resize1d(
    data,
    size,
    roi=None,
    layout="NCW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    cubic_alpha=-0.5,
    cubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
):
    """Image resize1d operator.

    This operator takes data as input and does 1D scaling to the given scale factor.
    In the default case, where the data_layout is `NCW`
    with data of shape (n, c, w)
    out will have a shape (n, c, size[0])

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("linear", "nearest_neighbor", "cubic")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    size: Tuple of Int or Expr
        The out size to which the image will be resized.

    roi: Tuple of Float or Expr, optional
        The region of interest for cropping the input image. Expected to be of
        size 2, and format [start_w, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, linear, cubic].

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor. Defintions can be found
        in topi/image/resize.py.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    cubic_alpha: float
        Spline Coefficient for cubic interpolation

    cubic_exclude: int
        Flag to exclude exterior of the image during cubic interpolation

    extrapolation_value: float
        Fill value to use when roi is outside of the image

    out_dtype : str, optional
        Type to return. If left None returns the same type as input.

    Returns
    -------
    result: relay.Expr
        The resized result.
    """
    if roi is None:
        roi = [0.0] * 2
    if isinstance(size, Constant):
        size = list(size.data.numpy().astype("int32"))
    if isinstance(roi, Constant):
        roi = list(roi.data.numpy().astype("int32"))
    if isinstance(size, Expr) or isinstance(roi, Expr):
        raise NotImplementedError(
            "dyn.resize1d is not yet implemented, got size", size, "and roi", roi
        )
    return _make.resize1d(
        data,
        size,
        roi,
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
        cubic_alpha,
        cubic_exclude,
        extrapolation_value,
        out_dtype,
    )


def resize2d(
    data,
    size,
    roi=None,
    layout="NCHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    cubic_alpha=-0.5,
    cubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
):
    """Image resize2d operator.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, size[0], size[1])

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("linear", "nearest_neighbor", "cubic")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    size: Tuple of Int or Expr
        The out size to which the image will be resized.

    roi: Tuple of Float or Expr, optional
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, linear, cubic].

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor. Defintions can be found
        in topi/image/resize.py.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    cubic_alpha: float
        Spline Coefficient for bicubic interpolation

    cubic_exclude: int
        Flag to exclude exterior of the image during bicubic interpolation

    extrapolation_value: float
        Fill value to use when roi is outside of the image

    out_dtype : str, optional
        Type to return. If left None returns the same type as input.

    Returns
    -------
    result: relay.Expr
        The resized result.
    """
    if roi is None:
        roi = [0.0] * 4
    if isinstance(size, Constant):
        size = list(size.data.numpy().astype("int32"))
    if isinstance(roi, Constant):
        roi = list(roi.data.numpy().astype("float32"))
    if isinstance(size, Expr) or isinstance(roi, Expr):
        if not isinstance(size, Expr):
            size = const(size, "int64")
        if not isinstance(roi, Expr):
            roi = const(roi, "float32")
        return _dyn_make.resize2d(
            data,
            size,
            roi,
            layout,
            method,
            coordinate_transformation_mode,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            extrapolation_value,
            out_dtype,
        )
    return _make.resize2d(
        data,
        size,
        roi,
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
        cubic_alpha,
        cubic_exclude,
        extrapolation_value,
        out_dtype,
    )


def resize3d(
    data,
    size,
    roi=None,
    layout="NCDHW",
    method="linear",
    coordinate_transformation_mode="half_pixel",
    rounding_method="",
    cubic_alpha=-0.5,
    cubic_exclude=0,
    extrapolation_value=0.0,
    out_dtype=None,
):
    """Image resize3d operator.

    This operator takes data as input and does 3D scaling to the given scale factor.
    In the default case, where the data_layout is `NCDHW`
    with data of shape `(n, c, d, h, w)`
    out will have a shape `(n, c, size[0], size[1], size[2])`

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("linear", "nearest_neighbor", "cubic")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    size: Tuple of Int or Expr
        The out size to which the image will be resized.

    roi: Tuple of Float or Expr, optional
        The region of interest for cropping the input image. Expected to be of
        size 6, and format [start_d, start_h, start_w, end_d, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, linear, cubic].

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor. Defintions can be found
        in topi/image/resize.py.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: string, optional
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    cubic_alpha: float
        Spline Coefficient for cubic interpolation

    cubic_exclude: int
        Flag to exclude exterior of the image during cubic interpolation

    extrapolation_value: float
        Fill value to use when roi is outside of the image

    out_dtype : str, optional
        Type to return. If left None returns the same type as input.

    Returns
    -------
    result: relay.Expr
        The resized result.
    """
    if roi is None:
        roi = [0.0] * 6
    if isinstance(size, Constant):
        size = list(size.data.numpy().astype("int32"))
    if isinstance(roi, Constant):
        roi = list(roi.data.numpy().astype("int32"))
    if isinstance(size, Expr) or isinstance(roi, Expr):
        raise NotImplementedError(
            "dyn.resize3d is not yet implemented, got size", size, "and roi", roi
        )
    return _make.resize3d(
        data,
        size,
        roi,
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
        cubic_alpha,
        cubic_exclude,
        extrapolation_value,
        out_dtype,
    )


def crop_and_resize(
    data,
    boxes,
    box_indices,
    crop_size,
    layout,
    method="bilinear",
    extrapolation_value=0,
    out_dtype=None,
):
    """Crop input images and resize them.

    method indicates the algorithm to be used while calculating the out value
    and method can be either "bilinear" or "nearest_neighbor".

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    boxes : relay.Expr
        A 2-D tensor of shape [num_boxes, 4]. Each row of the tensor specifies
        the coordinates of a box.

    box_indices : relay.Expr
        A 1-D tensor of shape [num_boxes], box_ind[i] specifies the data that
        the i-th box refers to.

    crop_size : Tuple of PrimExpr
        The target size to which each box will be resized.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method, it can be either "nearest_neighbor" or "bilinear".

    extrapolation_value : float, optional
        Value used for extrapolation, when applicable.

    out_dtype : str, optional
        Type to return. If left None returns the same type as input.

    Returns
    -------
    result: relay.Expr
        The computed result.
    """
    return _make.crop_and_resize(
        data, boxes, box_indices, crop_size, layout, method, extrapolation_value, out_dtype
    )


def dilation2d(
    data,
    weight,
    strides=(1, 1),
    padding=(0, 0),
    dilations=(1, 1),
    data_layout="NCHW",
    kernel_layout="IHW",
    out_dtype="",
):
    r"""Morphological Dilation 2D.
    This operator takes the weight as the dilation kernel and dilates it with
    data to produce an output. In the default case, where the data_layout is `NCHW`
    and kernel_layout is `OIHW`, dilation2d takes in a data Tensor with shape
    `(batch_size, in_channels, height, width)`, and a weight Tensor with shape
    `(channels, kernel_height, kernel_width)` to produce an output Tensor
    with the following rule:

    .. math::
        \mbox{out}[b, c, y, x] = \max_{dy, dx}
           \mbox{data}[b, c, \mbox{strides}[0] * y  + dy, \mbox{strides}[1] * x + dx] +
           \mbox{weight}[c, dy, dx]

    Padding and dilation are applied to data and weight respectively before the computation.
    This operator accepts data layout specification. Semantically, the operator
    will convert the layout to the canonical layout
    (`NCHW` for data and `IHW` for weight) and perform the computation.

    weight : tvm.relay.Expr
        The weight expressions.

    strides : Optional[Tuple[int]]
        The strides of convolution.

    padding : Optional[Tuple[int]]
        The padding of convolution on both sides of inputs before convolution.

    dilations : Optional[Tuple[int]]
        Specifies the dilation rate to be used for dilated convolution.

    data_layout : Optional[str]
        Layout of the input.

    kernel_layout : Optional[str]
        Layout of the weight.

    out_dtype : Optional[str]
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dilation2d(
        data, weight, strides, padding, dilations, data_layout, kernel_layout, out_dtype
    )


def affine_grid(data, target_shape=None):
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
    return _make.affine_grid(data, target_shape)


def grid_sample(
    data, grid, method="bilinear", layout="NCHW", padding_mode="zeros", align_corners=True
):
    """Applies grid sampling to input feature map.

    Given :math:`data` and :math:`grid`, then for 4-D the output is computed by

    .. math::

        x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
        y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
        output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src}])

    :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and
    :math:`G()` denotes the interpolation function.

    The out-boundary points will be padded with zeros if padding_mode is "zeros", or
    border pixel value if padding_mode is "border", or
    inner pixel value if padding_mode is "reflection".

    The left-top corner (-1, -1) and right-bottom corner (1, 1) in grid will be map to
    (0, 0) and (h - 1, w - 1) of data if align_corners is "True", or
    (-0.5, -0.5) and (h + 0.5, w + 0.5) of data if align_corners is "False".

    The shape of the output will be
    4-D (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]), or
    5-D (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3], grid.shape[4]).

    The operator assumes that :math:`grid` has been normalized to [-1, 1].

    grid_sample often cooperates with affine_grid which generates sampling grids for grid_sample.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width], or
        5-D with shape [batch, in_channel, in_depth, in_height, in_width]

    grid : tvm.Tensor
        4-D with shape [batch, 2, out_height, out_width], or
        5-D with shape [batch, 3, out_depth, out_height, out_width]

    method : str
        The interpolation method, 4-D "nearest", "bilinear", "bicubic" and
        5-D "nearest", "bilinear"("trilinear") are supported.

    layout : str
        The layout of input data and the output.

    padding_mode : str
        The padding mode for outside grid values, "zeros", "border", "reflection" are supported.

    align_corners: bool
        Geometrically, we consider the pixels of the input as squares rather than points.
        If set to "True", the extrema ("-1" and "1") are considered as referring
        to the center points of the input corner pixels. If set to "False", they
        are instead considered as referring to the corner points of the input corner
        pixels, making the sampling more resolution agnostic.

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, in_channel, out_height, out_width], or
        5-D with shape [batch, in_channel, out_depth, out_height, out_width]
    """
    return _make.grid_sample(data, grid, method, layout, padding_mode, align_corners)
