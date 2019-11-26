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
from __future__ import absolute_import as _abs
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
