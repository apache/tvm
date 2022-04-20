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
# pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

import tvm.topi
from tvm.runtime import convert
from tvm.te.hybrid import script
from tvm.topi.utils import nchw_pack_layout, nchw_xc_layout
from ... import op as reg


# resize
@reg.register_compute("dyn.image.resize2d")
def compute_resize2d(attrs, inputs, out_type):
    """
    Compute function calls into topi
    """
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    rounding_method = attrs.rounding_method
    cubic_alpha = attrs.cubic_alpha
    cubic_exclude = attrs.cubic_exclude
    extrapolation_value = attrs.extrapolation_value
    out_dtype = attrs.out_dtype
    return [
        tvm.topi.image.resize2d(
            inputs[0],
            inputs[2],
            inputs[1],
            layout,
            method,
            coord_trans,
            rounding_method,
            cubic_alpha,
            cubic_exclude,
            extrapolation_value,
            out_dtype,
            out_type.shape,
        )
    ]


reg.register_injective_schedule("dyn.image.resize2d")


@script
def _resize2d_shape_func(dshape, size, ndim, height_axis, width_axis):
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = int64(dshape[i])
    out[height_axis] = int64(size[0])
    out[width_axis] = int64(size[1])
    return out


@reg.register_shape_func("dyn.image.resize2d", True)
def resize2d_shape_func(attrs, inputs, _):
    """
    Shape function for dyn.image.resize op.
    """
    layout = attrs.layout
    if nchw_pack_layout(layout) or nchw_xc_layout(layout):
        out = [
            _resize2d_shape_func(
                inputs[0].shape, inputs[1], convert(len(inputs[0].shape)), convert(2), convert(3)
            )
        ]
    else:
        height_axis = width_axis = 1
        for i, letter in enumerate(layout):
            if letter == "H":
                height_axis = i
            if letter == "W":
                width_axis = i
        out = [
            _resize2d_shape_func(
                inputs[0].shape,
                inputs[1],
                convert(len(inputs[0].shape)),
                convert(height_axis),
                convert(width_axis),
            )
        ]
    return out
