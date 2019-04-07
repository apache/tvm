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
"""Faster R-CNN and Mask R-CNN operations."""
import topi
from topi.util import get_const_tuple, get_float_tuple, get_const_int
from .. import op as reg
from ..op import OpPattern


@reg.register_compute("vision.roi_align")
def compute_roi_align(attrs, inputs, _, target):
    """Compute definition of roi_align"""
    assert attrs.layout == "NCHW"
    return [topi.vision.rcnn.roi_align_nchw(
        inputs[0], inputs[1], pooled_size=get_const_tuple(attrs.pooled_size),
        spatial_scale=attrs.spatial_scale, sample_ratio=attrs.sample_ratio)]

@reg.register_schedule("vision.roi_align")
def schedule_roi_align(_, outs, target):
    """Schedule definition of roi_align"""
    with target:
        return topi.generic.vision.schedule_roi_align(outs)

reg.register_pattern("vision.roi_align", OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("vision.roi_pool")
def compute_roi_pool(attrs, inputs, _, target):
    """Compute definition of roi_pool"""
    assert attrs.layout == "NCHW"
    return [topi.vision.rcnn.roi_pool_nchw(
        inputs[0], inputs[1], pooled_size=get_const_tuple(attrs.pooled_size),
        spatial_scale=attrs.spatial_scale)]

@reg.register_schedule("vision.roi_pool")
def schedule_roi_pool(_, outs, target):
    """Schedule definition of roi_pool"""
    with target:
        return topi.generic.vision.schedule_roi_pool(outs)

reg.register_pattern("vision.roi_pool", OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("vision.proposal")
def compute_proposal(attrs, inputs, _, target):
    """Compute definition of proposal"""
    scales = get_float_tuple(attrs.scales)
    ratios = get_float_tuple(attrs.ratios)
    feature_stride = attrs.feature_stride
    threshold = attrs.threshold
    rpn_pre_nms_top_n = attrs.rpn_pre_nms_top_n
    rpn_post_nms_top_n = attrs.rpn_post_nms_top_n
    rpn_min_size = attrs.rpn_min_size
    iou_loss = bool(get_const_int(attrs.iou_loss))
    with target:
        return [
            topi.vision.rcnn.proposal(inputs[0], inputs[1], inputs[2], scales, ratios,
                                      feature_stride, threshold, rpn_pre_nms_top_n,
                                      rpn_post_nms_top_n, rpn_min_size, iou_loss)
        ]

@reg.register_schedule("vision.proposal")
def schedule_proposal(_, outs, target):
    """Schedule definition of proposal"""
    with target:
        return topi.generic.schedule_proposal(outs)

reg.register_pattern("vision.proposal", OpPattern.OPAQUE)
