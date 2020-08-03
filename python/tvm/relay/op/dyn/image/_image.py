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
#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

import tvm.topi
from tvm.runtime import convert
from tvm.te.hybrid import script
from tvm.topi.util import nchw_pack_layout, nchw_xc_layout
from ... import op as reg


# resize
@reg.register_compute("dyn.image.resize")
def compute_resize(attrs, inputs, out_type):
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    out_dtype = attrs.out_dtype
    return [
        tvm.topi.image.resize(inputs[0], inputs[1], layout, method, coord_trans, out_dtype,
                              out_type.shape)
    ]


reg.register_injective_schedule("dyn.image.resize")


@script
def _NCHW_resize_shape_func(dshape, size, ndim):
    out = output_tensor((ndim, ), "int64")
    for i in const_range(ndim):
        out[i] = int64(dshape[i])
    out[2] = int64(size[0])
    out[3] = int64(size[1])
    return out


@script
def _NHWC_resize_shape_func(dshape, size, ndim):
    out = output_tensor((ndim, ), "int64")
    for i in const_range(ndim):
        out[i] = int64(dshape[i])
    out[1] = int64(size[0])
    out[2] = int64(size[1])
    return out


@reg.register_shape_func("dyn.image.resize", True)
def resize_shape_func(attrs, inputs, _):
    """
    Shape function for dyn.image.resize op.
    """
    layout = attrs.layout
    if layout == 'NHWC':
        out = [_NHWC_resize_shape_func(inputs[0].shape, inputs[1], convert(len(inputs[0].shape)))]
    elif (layout == 'NCHW') or nchw_pack_layout(layout) or nchw_xc_layout(layout):
        out = [_NCHW_resize_shape_func(inputs[0].shape, inputs[1], convert(len(inputs[0].shape)))]
    else:
        raise ValueError("Resize Unsupported Layout", layout)
    return out
