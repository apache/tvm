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
"""Image operators."""
from typing import Optional, Tuple, Union

from tvm import DataType
from tvm.ir.expr import PrimExpr

from . import _ffi_api
from ...expr import Expr, ShapeExpr


PrimExprLike = Union[int, PrimExpr]


def resize2d(
    data: Expr,
    size: Union[Expr, PrimExprLike, Tuple[PrimExprLike]],
    roi: Optional[Union[float, Tuple[float]]] = None,
    layout: str = "NCHW",
    method: str = "linear",
    coordinate_transformation_mode: str = "half_pixel",
    rounding_method: str = "round",
    cubic_alpha: float = -0.5,
    cubic_exclude: int = 0,
    extrapolation_value: float = 0.0,
    out_dtype: Optional[Union[str, DataType]] = None,
) -> Expr:
    """Image resize2d operator.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, size[0], size[1])

    method indicates the algorithm to be used while calculating the out value
    and method can be one of ("linear", "nearest_neighbor", "cubic")

    Parameters
    ----------
    data : relax.Expr
        The input data to the operator.

    size: Union[Expr, PrimExprLike, Tuple[PrimExprLike]]
        The out size to which the image will be resized.
        If specified as a list, it is required to have length either 1 or 2.
        If specified as an Expr, it is required to have ndim 2.

    roi: Optional[Union[float, Tuple[float]]]
        The region of interest for cropping the input image. Expected to be of
        size 4, and format [start_h, start_w, end_h, end_w].
        Only used if coordinate_transformation_mode is tf_crop_and_resize.

    layout : str
        Layout of the input.

    method : str
        Scale method to used [nearest_neighbor, linear, cubic].

    coordinate_transformation_mode : str
        Describes how to transform the coordinate in the resized tensor
        to the coordinate in the original tensor. Definitions can be found
        in topi/image/resize.py.
        [half_pixel, align_corners, asymmetric, pytorch_half_pixel,
        tf_half_pixel_for_nn, and tf_crop_and_resize].

    rounding_method: str
        indicates how to find the "nearest" pixel in nearest_neighbor method
        [round, floor, ceil]

    cubic_alpha: float
        Spline Coefficient for bicubic interpolation

    cubic_exclude: int
        Flag to exclude exterior of the image during bicubic interpolation

    extrapolation_value: float
        Fill value to use when roi is outside of the image

    out_dtype : Optional[Union[str, DataType]]
        The dtype of the output tensor.
        It it is not specified, the output will have the same dtype as input if not specified.

    Returns
    -------
    result: relax.Expr
        The resized result.
    """
    if roi is None:
        roi = (0.0, 0.0, 0.0, 0.0)  # type: ignore
    elif isinstance(roi, float):
        roi = (roi, roi, roi, roi)  # type: ignore

    if isinstance(size, (int, PrimExpr)):
        size = (size, size)
    if isinstance(size, (tuple, list)):
        if len(size) == 1:
            size = ShapeExpr([size[0], size[0]])
        else:
            size = ShapeExpr(size)

    return _ffi_api.resize2d(  # type: ignore
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
