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

import topi
from topi.util import get_const_int, get_const_float, get_float_tuple
from .. import op as reg
from .. import strategy
from ..op import OpPattern

# multibox_prior
@reg.register_compute("vision.multibox_prior")
def compute_multibox_prior(attrs, inputs, _):
    """Compute definition of multibox_prior"""
    sizes = get_float_tuple(attrs.sizes)
    ratios = get_float_tuple(attrs.ratios)
    steps = get_float_tuple(attrs.steps)
    offsets = get_float_tuple(attrs.offsets)
    clip = bool(get_const_int(attrs.clip))
    return [topi.vision.ssd.multibox_prior(inputs[0], sizes, ratios, steps,
                                           offsets, clip)]

reg.register_schedule("vision.multibox_prior", strategy.schedule_multibox_prior)
reg.register_pattern("vision.multibox_prior", OpPattern.OPAQUE)


# multibox_transform_loc
@reg.register_compute("vision.multibox_transform_loc")
def compute_multibox_transform_loc(attrs, inputs, _):
    """Compute definition of multibox_detection"""
    clip = bool(get_const_int(attrs.clip))
    threshold = get_const_float(attrs.threshold)
    variances = get_float_tuple(attrs.variances)
    return topi.vision.ssd.multibox_transform_loc(
        inputs[0], inputs[1], inputs[2], clip, threshold, variances)

reg.register_schedule("vision.multibox_transform_loc", strategy.schedule_multibox_transform_loc)
reg.register_pattern("vision.multibox_transform_loc", OpPattern.OPAQUE)


# Get counts of valid boxes
reg.register_strategy("vision.get_valid_counts", strategy.get_valid_counts_strategy)
reg.register_pattern("vision.get_valid_counts", OpPattern.OPAQUE)


# non-maximum suppression
reg.register_strategy("vision.non_max_suppression", strategy.nms_strategy)
reg.register_pattern("vision.non_max_suppression", OpPattern.OPAQUE)
