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

def resize(data,
           size,
           layout="NCHW",
           method="bilinear",
           coordinate_transformation_mode="half_pixel",
           out_dtype=None):
    """Image resize operator.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, size[0], size[1])

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("bilinear", "nearest_neighbor", "bicubic")

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    size: Tuple of Expr
        The out size to which the image will be resized.

    layout : str, optional
        Layout of the input.

    method : str, optional
        Scale method to used [nearest_neighbor, bilinear, bicubic].

    coordinate_transformation_mode : string, optional
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor.
        Refer to the ONNX Resize operator specification for details.
        [half_pixel, align_corners, asymmetric]

    out_dtype : str, optional
        Type to return. If left None returns the same type as input.

    Returns
    -------
    result: relay.Expr
        The resized result.
    """
    return _make.resize(data, size, layout, method, coordinate_transformation_mode, out_dtype)


def crop_and_resize(data,
                    boxes,
                    box_indices,
                    crop_size,
                    layout,
                    method="bilinear",
                    extrapolation_value=0,
                    out_dtype=None):
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

    crop_size : Tuple of Expr
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
    return _make.crop_and_resize(data, boxes, box_indices, crop_size,
                                 layout, method, extrapolation_value, out_dtype)


def dilation2d(data,
               weight,
               strides=(1, 1),
               padding=(0, 0),
               dilations=(1, 1),
               data_layout="NCHW",
               kernel_layout="IHW",
               out_dtype=""):
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

    return _make.dilation2d(data, weight, strides, padding, dilations, data_layout,
                            kernel_layout, out_dtype)
