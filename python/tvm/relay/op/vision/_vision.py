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
"""Definition of vision ops"""
from __future__ import absolute_import

from tvm import topi
from tvm.te.hybrid import script
from tvm.runtime import convert

from .. import op as reg
from .. import strategy
from ..op import OpPattern

# multibox_prior
reg.register_strategy("vision.multibox_prior", strategy.multibox_prior_strategy)
reg.register_pattern("vision.multibox_prior", OpPattern.OPAQUE)


# multibox_transform_loc
reg.register_strategy("vision.multibox_transform_loc", strategy.multibox_transform_loc_strategy)
reg.register_pattern("vision.multibox_transform_loc", OpPattern.OPAQUE)


# Get counts of valid boxes
reg.register_strategy("vision.get_valid_counts", strategy.get_valid_counts_strategy)
reg.register_pattern("vision.get_valid_counts", OpPattern.OPAQUE)


# non-maximum suppression
reg.register_strategy("vision.non_max_suppression", strategy.nms_strategy)
reg.register_pattern("vision.non_max_suppression", OpPattern.OPAQUE)


@script
def _get_valid_counts_shape_func(data_shape):
    valid_counts_shape = output_tensor((1,), "int64")
    out_tensor_shape = output_tensor((data_shape.shape[0],), "int64")
    out_indices_shape = output_tensor((2,), "int64")

    valid_counts_shape[0] = data_shape[0]
    for i in const_range(data_shape.shape[0]):
        out_tensor_shape[i] = data_shape[i]
    out_indices_shape[0] = data_shape[0]
    out_indices_shape[1] = data_shape[1]

    return valid_counts_shape, out_tensor_shape, out_indices_shape


@reg.register_shape_func("vision.get_valid_counts", False)
def get_valid_counts_shape_func(attrs, inputs, _):
    return _get_valid_counts_shape_func(inputs[0])


@script
def _nms_shape_func(data_shape):
    out_shape = output_tensor((2,), "int64")
    count_shape = output_tensor((2,), "int64")

    out_shape[0] = data_shape[0]
    out_shape[1] = data_shape[1]
    count_shape[0] = data_shape[0]
    count_shape[1] = int64(1)
    return out_shape, count_shape


@reg.register_shape_func("vision.non_max_suppression", False)
def nms_shape_func(attrs, inputs, _):
    if attrs.return_indices:
        return _nms_shape_func(inputs[0])
    return [topi.math.identity(inputs[0])]


@script
def _roi_align_shape_func(data_shape, rois_shape, pooled_size):
    out = output_tensor((4,), "int64")
    out[0] = rois_shape[0]
    out[1] = data_shape[1]
    out[2] = int64(pooled_size[0])
    out[3] = int64(pooled_size[1])
    return out


@reg.register_shape_func("vision.roi_align", False)
def roi_align_shape_func(attrs, inputs, _):
    return [_roi_align_shape_func(inputs[0], inputs[1], convert(attrs.pooled_size))]
