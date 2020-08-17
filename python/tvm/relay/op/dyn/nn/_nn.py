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
"""Backend compiler related feature registration"""

from __future__ import absolute_import

from tvm import topi

from tvm.runtime import convert
from tvm.te.hybrid import script
from ...op import register_shape_func, register_compute
from ...op import register_injective_schedule

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

#####################
#  Shape functions  #
#####################

# upsampling
@script
def _upsampling_nhwc_shape_func(dshape, scale_h, scale_w, ndim):
    out = output_tensor((ndim,), "int64")
    batch_size = dshape[0]
    in_height = dshape[1]
    in_width = dshape[2]
    channels = dshape[3]
    out[0] = int64(batch_size)
    out[1] = int64(round(in_height * scale_h[0]))
    out[2] = int64(round(in_width * scale_w[0]))
    out[3] = int64(channels)
    return out

@script
def _upsampling_nchw_shape_func(dshape, scale_h, scale_w, ndim):
    out = output_tensor((ndim,), "int64")
    batch_size = dshape[0]
    channels = dshape[1]
    in_height = dshape[2]
    in_width = dshape[3]
    out[0] = int64(batch_size)
    out[1] = int64(channels)
    out[2] = int64(round(in_height * scale_h[0]))
    out[3] = int64(round(in_width * scale_w[0]))
    return out

@register_shape_func("dyn.nn.upsampling", True)
def upsampling_shape_func(attrs, inputs, _):
    """Shape function for upsampling. Supports NCHW and NHWC layouts."""
    if attrs.layout == "NHWC":
        shape_func = _upsampling_nhwc_shape_func(inputs[0].shape, inputs[1], inputs[2],
                                                 convert(len(inputs[0].shape)))
    elif attrs.layout == "NCHW":
        shape_func = _upsampling_nchw_shape_func(inputs[0].shape, inputs[1], inputs[2],
                                                 convert(len(inputs[0].shape)))
    else:
        assert false, "Layout passed to the upsampling shape func must be NCHW or NHWC"
    return [shape_func]
