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
from tvm import topi
from tvm.topi.util import get_const_tuple
from .. import op as reg
from .. import strategy
from ..op import OpPattern

# roi_align
reg.register_strategy("vision.roi_align", strategy.roi_align_strategy)
reg.register_pattern("vision.roi_align", OpPattern.OUT_ELEMWISE_FUSABLE)

# roi_pool
@reg.register_compute("vision.roi_pool")
def compute_roi_pool(attrs, inputs, _):
    """Compute definition of roi_pool"""
    assert attrs.layout == "NCHW"
    return [
        topi.vision.rcnn.roi_pool_nchw(
            inputs[0],
            inputs[1],
            pooled_size=get_const_tuple(attrs.pooled_size),
            spatial_scale=attrs.spatial_scale,
        )
    ]


reg.register_schedule("vision.roi_pool", strategy.schedule_roi_pool)
reg.register_pattern("vision.roi_pool", OpPattern.OUT_ELEMWISE_FUSABLE)

# proposal
reg.register_strategy("vision.proposal", strategy.proposal_strategy)
reg.register_pattern("vision.proposal", OpPattern.OPAQUE)
