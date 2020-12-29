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
# pylint: disable=no-else-return, invalid-name, unused-argument, too-many-arguments, consider-using-in
"""Backend compiler related feature registration for dynamic relay ops in nn namespace"""

from __future__ import absolute_import

from tvm import topi

from tvm.runtime import convert
from tvm.te.hybrid import script
from ...op import register_shape_func, register_compute
from ...op import register_injective_schedule, register_broadcast_schedule

# upsampling
@register_compute("dyn.nn.upsampling")
def compute_upsampling(attrs, inputs, out_dtype):
    data = inputs[0]
    scale_h = inputs[1]
    scale_w = inputs[2]
    layout = attrs.layout
    method = attrs.method
    align_corners = attrs.align_corners
    return [
        topi.nn.upsampling(data, scale_h, scale_w, layout, method, align_corners, out_dtype.shape)
    ]


# upsampling3d
@register_compute("dyn.nn.upsampling3d")
def compute_upsampling3d(attrs, inputs, out_dtype):
    data = inputs[0]
    scale_d = inputs[1]
    scale_h = inputs[2]
    scale_w = inputs[3]
    layout = attrs.layout
    method = attrs.method
    coordinate_transformation_mode = attrs.coordinate_transformation_mode
    return [
        topi.nn.upsampling3d(
            data,
            scale_d,
            scale_h,
            scale_w,
            layout,
            method,
            coordinate_transformation_mode,
            out_dtype.shape,
        )
    ]


register_injective_schedule("dyn.nn.upsampling")
register_injective_schedule("dyn.nn.upsampling3d")
register_broadcast_schedule("dyn.nn.pad")

#####################
#  Shape functions  #
#####################

# upsampling
@script
def _upsampling_shape_func(dshape, scale_h, scale_w, height_axis, width_axis):
    out = output_tensor((4,), "int64")
    for i in const_range(4):
        out[i] = int64(dshape[i])
    out[height_axis] = int64(round(dshape[height_axis] * scale_h[0]))
    out[width_axis] = int64(round(dshape[width_axis] * scale_w[0]))
    return out


@register_shape_func("dyn.nn.upsampling", True)
def upsampling_shape_func(attrs, inputs, _):
    """Shape function for upsampling. Supports NCHW and NHWC layouts."""
    layout = attrs.layout
    height_axis = width_axis = 1
    for i, letter in enumerate(layout):
        if letter == "H":
            height_axis = i
        if letter == "W":
            width_axis = i
    return [
        _upsampling_shape_func(
            inputs[0].shape, inputs[1], inputs[2], convert(height_axis), convert(width_axis)
        )
    ]


# upsampling3d
@script
def _upsampling3d_shape_func(
    dshape, scale_d, scale_h, scale_w, depth_axis, height_axis, width_axis
):
    out = output_tensor((5,), "int64")
    for i in const_range(5):
        out[i] = int64(dshape[i])
    out[depth_axis] = int64(round(dshape[depth_axis] * scale_d[0]))
    out[height_axis] = int64(round(dshape[height_axis] * scale_h[0]))
    out[width_axis] = int64(round(dshape[width_axis] * scale_w[0]))
    return out


@register_shape_func("dyn.nn.upsampling3d", True)
def upsampling3d_shape_func(attrs, inputs, _):
    """Shape function for upsampling. Supports NCHW and NHWC layouts."""
    layout = attrs.layout
    depth_axis = height_axis = width_axis = 1
    for i, letter in enumerate(layout):
        if letter == "D":
            depth_axis = i
        if letter == "H":
            height_axis = i
        if letter == "W":
            width_axis = i
    return [
        _upsampling3d_shape_func(
            inputs[0].shape,
            inputs[1],
            inputs[2],
            inputs[3],
            convert(depth_axis),
            convert(height_axis),
            convert(width_axis),
        )
    ]


# pad
@script
def _dyn_pad_shape_func(data, pad_width):
    ndim = len(data.shape)
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = int64(pad_width[i, 0] + pad_width[i, 1] + data.shape[i])
    return out


@register_shape_func("dyn.nn.pad", True)
def pad_shape_func(attrs, inputs, data):
    """
    Shape function for dynamic pad op.
    """
    return [_dyn_pad_shape_func(inputs[0], inputs[1])]
