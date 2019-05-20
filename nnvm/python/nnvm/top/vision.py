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
"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

@reg.register_compute("yolo_reorg")
def compute_reorg(attrs, inputs, _):
    """Compute definition of reorg"""
    return topi.vision.reorg(inputs[0], attrs.get_int("stride"))

@reg.register_schedule("yolo_reorg")
def schedule_reorg(attrs, outs, target):
    """Schedule definition of reorg"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

reg.register_pattern("yolo_reorg", OpPattern.INJECTIVE)

# multibox_prior
@reg.register_schedule("multibox_prior")
def schedule_multibox_prior(_, outs, target):
    """Schedule definition of multibox_prior"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_prior(outs)

@reg.register_compute("multibox_prior")
def compute_multibox_prior(attrs, inputs, _):
    """Compute definition of multibox_prior"""
    sizes = attrs.get_float_tuple('sizes')
    ratios = attrs.get_float_tuple('ratios')
    steps = attrs.get_float_tuple('steps')
    offsets = attrs.get_float_tuple('offsets')
    clip = attrs.get_bool('clip')

    return topi.vision.ssd.multibox_prior(inputs[0], sizes, ratios,
                                          steps, offsets, clip)

reg.register_pattern("multibox_prior", OpPattern.OPAQUE)

# multibox_transform_loc
@reg.register_schedule("multibox_transform_loc")
def schedule_multibox_transform_loc(_, outs, target):
    """Schedule definition of multibox_detection"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_transform_loc(outs)

@reg.register_compute("multibox_transform_loc")
def compute_multibox_transform_loc(attrs, inputs, _):
    """Compute definition of multibox_detection"""
    clip = attrs.get_bool('clip')
    threshold = attrs.get_float('threshold')
    variance = attrs.get_float_tuple('variances')

    return topi.vision.ssd.multibox_transform_loc(inputs[0], inputs[1], inputs[2],
                                                  clip, threshold, variance)

reg.register_pattern("multibox_detection", OpPattern.OPAQUE)

# non-maximum suppression
@reg.register_schedule("non_max_suppression")
def schedule_nms(_, outs, target):
    """Schedule definition of non_max_suppression"""
    with tvm.target.create(target):
        return topi.generic.schedule_nms(outs)

@reg.register_compute("non_max_suppression")
def compute_nms(attrs, inputs, _):
    """Compute definition of non_max_suppression"""
    return_indices = attrs.get_bool('return_indices')
    max_output_size = attrs.get_int('max_output_size')
    iou_threshold = attrs.get_float('iou_threshold')
    force_suppress = attrs.get_bool('force_suppress')
    top_k = attrs.get_int('top_k')
    id_index = attrs.get_int('id_index')
    invalid_to_bottom = attrs.get_bool('invalid_to_bottom')

    return topi.vision.non_max_suppression(inputs[0], inputs[1],
                                           max_output_size=max_output_size,
                                           iou_threshold=iou_threshold,
                                           force_suppress=force_suppress,
                                           top_k=top_k, id_index=id_index,
                                           return_indices=return_indices,
                                           invalid_to_bottom=invalid_to_bottom)

reg.register_pattern("non_max_suppression", OpPattern.OPAQUE)
