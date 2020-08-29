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
    return [topi.nn.upsampling(data, scale_h, scale_w, layout,
                               method, align_corners, out_dtype.shape)]

register_injective_schedule("dyn.nn.upsampling")
register_broadcast_schedule("dyn.nn.pad")

#####################
#  Shape functions  #
#####################

# upsampling
@script
def _upsampling_shape_func(dshape, scale_h, scale_w, height_axis, width_axis, channel_axis):
    out = output_tensor((4,), "int64")
    out[0] = int64(dshape[0])
    out[height_axis] = int64(round(dshape[height_axis] * scale_h[0]))
    out[width_axis] = int64(round(dshape[width_axis] * scale_w[0]))
    out[channel_axis] = int64(dshape[channel_axis])
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
        if letter == "C":
            channel_axis = i
    return [_upsampling_shape_func(inputs[0].shape, inputs[1], inputs[2],
                                   convert(height_axis), convert(width_axis),
                                   convert(channel_axis))]
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
