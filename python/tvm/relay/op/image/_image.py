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

import topi
from topi.util import get_const_tuple
from .. import op as reg
from .. import strategy
from ..op import OpPattern


# resize
@reg.register_compute("image.resize")
def compute_resize(attrs, inputs, out_type):
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    out_dtype = attrs.out_dtype
    return [topi.image.resize(inputs[0], size, layout, method, coord_trans, out_dtype)]

reg.register_injective_schedule("image.resize")


@reg.register_compute("image.resize3d")
def compute_resize3d(attrs, inputs, out_type):
    size = attrs.size
    layout = attrs.layout
    method = attrs.method
    coord_trans = attrs.coordinate_transformation_mode
    out_dtype = attrs.out_dtype
    return [topi.image.resize3d(inputs[0], size, layout, method, coord_trans, out_dtype)]

reg.register_injective_schedule("image.resize3d")


# crop and resize
@reg.register_compute("image.crop_and_resize")
def compute_crop_and_resize(attrs, inputs, out_type):
    crop_size = attrs.crop_size
    layout = attrs.layout
    method = attrs.method
    extrapolation_value = attrs.extrapolation_value
    out_dtype = attrs.out_dtype
    return [topi.image.crop_and_resize(inputs[0], inputs[1], inputs[2],
                                       crop_size, layout, method,
                                       extrapolation_value, out_dtype)]

reg.register_injective_schedule("image.crop_and_resize")


# dilation2d
reg.register_strategy("image.dilation2d", strategy.dilation2d_strategy)
reg.register_pattern("image.dilation2d", OpPattern.OUT_ELEMWISE_FUSABLE)


# affine_grid
@reg.register_compute("image.affine_grid")
def compute_affine_grid(attrs, inputs, out_dtype):
    target_shape = get_const_tuple(attrs.target_shape)
    return [topi.image.affine_grid(inputs[0], target_shape)]

reg.register_injective_schedule("image.affine_grid")


# grid_sample
@reg.register_compute("image.grid_sample")
def compute_grid_sample(attrs, inputs, out_dtype):
    method = attrs.method
    layout = attrs.layout
    return [topi.image.grid_sample(inputs[0], inputs[1], method, layout)]

reg.register_injective_schedule("image.grid_sample")
