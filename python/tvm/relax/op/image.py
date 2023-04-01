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
# pylint: disable=redefined-builtin
"""Image operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name,too-many-arguments


## (TVM-TOOL) py_op begin image/*
def resize2d(
    x: ty.Tensor,
    size: ty.Shape,
    roi: ty.Array[ty.Float] = (0.0, 0.0, 0.0, 0.0),
    layout: ty.Str = "NCHW",
    method: ty.Str = "linear",
    coordinate_transformation_mode: ty.Str = "half_pixel",
    rounding_method: ty.Str = "round",
    bicubic_alpha: ty.Float = -0.5,
    bicubic_exclude: ty.Float = 0.0,
    extrapolation_value: ty.Float = 0.0,
    out_dtype: ty.DType = None,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    size : ty.Shape
        TODO(tvm-unity-team): add doc
    roi : ty.Array[ty.Float]
        TODO(tvm-unity-team): add doc
    layout : ty.Str
        TODO(tvm-unity-team): add doc
    method : ty.Str
        TODO(tvm-unity-team): add doc
    coordinate_transformation_mode : ty.Str
        TODO(tvm-unity-team): add doc
    rounding_method : ty.Str
        TODO(tvm-unity-team): add doc
    bicubic_alpha : ty.Float
        TODO(tvm-unity-team): add doc
    bicubic_exclude : ty.Float
        TODO(tvm-unity-team): add doc
    extrapolation_value : ty.Float
        TODO(tvm-unity-team): add doc
    out_dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    size = tg.check(1, "size", tg.Shape(), size)
    roi = tg.check(2, "roi", tg.Array(tg.Float(), [4]), roi)
    layout = tg.check(3, "layout", tg.Str(), layout)
    method = tg.check(4, "method", tg.Str(), method)
    coordinate_transformation_mode = tg.check(
        5, "coordinate_transformation_mode", tg.Str(), coordinate_transformation_mode
    )
    rounding_method = tg.check(6, "rounding_method", tg.Str(), rounding_method)
    bicubic_alpha = tg.check(7, "bicubic_alpha", tg.Float(), bicubic_alpha)
    bicubic_exclude = tg.check(8, "bicubic_exclude", tg.Float(), bicubic_exclude)
    extrapolation_value = tg.check(
        9, "extrapolation_value", tg.Float(), extrapolation_value
    )
    out_dtype = tg.check(10, "out_dtype", tg.DType(), out_dtype)
    _ffi_func = _ffi.get_global_func("relax.op.image.resize2d")
    return _ffi_func(
        x,
        size,
        roi,
        layout,
        method,
        coordinate_transformation_mode,
        rounding_method,
        bicubic_alpha,
        bicubic_exclude,
        extrapolation_value,
        out_dtype,
    )


## (TVM-TOOL) py_op end image/*
